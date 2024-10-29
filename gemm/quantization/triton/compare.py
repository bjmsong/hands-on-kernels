import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.ops.matmul import matmul as triton_matmul
from triton.ops.matmul import _kernel
from triton import Config
from torch._inductor import config
from torch import _dynamo
torch._inductor.config.coordinate_descent_tuning = True
import torchao
# quantize_affine?
# from torchao.quantization.quant_primitives import groupwise_affine_quantize_tensor
aten = torch.ops.aten

def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M': 8},
                               num_stages=num_stages, num_warps=num_warps))
    return configs

config_list = [
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    # good for int8
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
]+get_configs_io_bound()

@triton.autotune(
    configs = [
        Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=1),
        Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=1),
    ]+config_list,
    key=['M', 'N', 'K'],
)
@triton.jit
def int8_weight_only_linear_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, b_ptr, s_ptr, y_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `x_ptr`
    # by to get the element one row down (A has M rows).
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_b,
    stride_ym, stride_yn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of Y it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of X and W.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `x_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `w_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_xm = tl.max_contiguous((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M,BLOCK_SIZE_M)
    offs_wn = tl.max_contiguous((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N,BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)
    b_ptrs = b_ptr + (offs_wn * stride_b)
    step_w = BLOCK_SIZE_K * stride_wk
    step_x = BLOCK_SIZE_K * stride_xk


    # -----------------------------------------------------------
    # Iterate to compute a block of the Y matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(x, w.to(tl.bfloat16))
        # Advance the ptrs to the next K block.
        x_ptrs += step_x
        w_ptrs += step_w

    s = tl.load(s_ptr)
    b = tl.load(b_ptrs)
    y = (accumulator.to(tl.bfloat16) * s + b)
    # y = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix Y with masks.
    offs_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)

def int8_weight_only_linear(x, w, b, s):
    # Check constraints.
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    # assert x.is_contiguous(), "Matrix x must be contiguous"
    # assert w.is_contiguous(), "Matrix w must be contiguous"
    M, K = x.shape
    K, N = w.shape
    assert b.shape[0] == N
    # Allocates output.
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    int8_weight_only_linear_kernel[grid](
        x, w, b, s, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        b.stride(0),
        y.stride(0), y.stride(1),
    )
    return y

@triton.autotune(
    configs = [
        Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ]+config_list,
    key=['M', 'N', 'K'],
)
@triton.jit
def uint4x2_weight_only_linear_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, b_ptr, s_ptr, y_ptr,
    # Matrix dimensions
    M, N, K, # x is Mx(K*2) and w is KxN
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `x_ptr`
    # by to get the element one row down (A has M rows).
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_b,
    stride_ym, stride_yn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of Y it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of X and W.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `x_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `w_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_xm = tl.max_contiguous((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M,BLOCK_SIZE_M)
    offs_wn = tl.max_contiguous((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N,BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None]//2 * stride_wk + offs_wn[None, :] * stride_wn)
    w_shifts = (offs_k % 2) * 4
    b_ptrs = b_ptr + (offs_wn * stride_b)
    step_w = BLOCK_SIZE_K//2 * stride_wk
    step_x = BLOCK_SIZE_K * stride_xk
    # -----------------------------------------------------------
    # Iterate to compute a block of the Y matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        w = ((w >> w_shifts[:, None]) & 0xF) - 8
        # We accumulate along the K dimension.
        accumulator += tl.dot(x, w.to(tl.bfloat16))
        # Advance the ptrs to the next K block.
        x_ptrs += step_x
        w_ptrs += step_w
    s = tl.load(s_ptr)
    b = tl.load(b_ptrs)
    y = (accumulator.to(tl.bfloat16) * s)+b

    # -----------------------------------------------------------
    # Write back the block of the output matrix Y with masks.
    offs_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)

def uint4x2_weight_only_linear(x, w, b, s):
    # Check constraints.
    assert x.shape[1] == w.shape[0]*2, "Incompatible dimensions"
    # assert x.is_contiguous(), "Matrix x must be contiguous"
    # assert w.is_contiguous(), "Matrix w must be contiguous"
    M, K = x.shape
    _, N = w.shape
    assert b.shape[0] == N
    # Allocates output.
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    uint4x2_weight_only_linear_kernel[grid](
        x, w, b, s, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        b.stride(0),
        y.stride(0), y.stride(1),
    )
    return y


quantiles = [0.5, 0.2, 0.8]

result = {}

for D in [2**6, 2**7, 2**8, 2**9, 2**10, 2**11]: #, 2**10, 2**12, 2**14]:
    result[D]={}
    result[D]["cublas linear"]={}
    result[D]["triton matmul"]={}
    result[D]["int8 linear"]={}
    result[D]["alt int8 linear"]={}
    result[D]["uint4x2 linear"]={}
    result[D]["int4 tinygemm"]={}
    N = D
    for t_x in [0,1]:
        for t_w in [1,0]:
            print("D tx tw",D, t_x, t_w)
            # x * w + bias
            x = torch.randn(1,D).to('cuda').to(torch.bfloat16)
            w_bf16 = torch.randn(D, N, dtype=torch.bfloat16).cuda()
            bias = torch.randn(N, dtype=torch.bfloat16).cuda()
            # make tensor contiguous
            if t_x:
                x = x.t().contiguous().t()
            if t_w:
                w_bf16 = w_bf16.t().contiguous().t()

            print("cublas linear")
            try:
                torch.nn.functional.linear(x, w_bf16, bias)
                torch.cuda.synchronize()
                result[D]["cublas linear"][(t_x, t_w)] = triton.testing.do_bench(lambda: torch.nn.functional.linear(x, w_bf16, bias), quantiles=quantiles)[0]
            except:
                print("err")
                pass
            torch.cuda.synchronize()

            # print("int4 tinygemm")
            # try:
            #     I=min(D//32,8)
            #     G=min(128, D//I)
            #     w_int4, scales_and_zeros = groupwise_affine_quantize_tensor(w_bf16, 4, 32)
            #     w_int4pack = aten._convert_weight_to_int4pack(w_int4.contiguous(), I)
            #     del w_int4
            #     result[D]["int4 tinygemm"][(t_x, t_w)] = triton.testing.do_bench(lambda: aten._weight_int4pack_mm(x.contiguous(), w_int4pack, 32, scales_and_zeros), quantiles=quantiles)[0]
            #     del w_bf16, scales_and_zeros, w_int4pack
            # except:
            #     print("err")
            #     pass

            w_int8 = torch.randint(-128, 127, (D, N), dtype=torch.int8).cuda()
            if t_w:
                w_int8 = w_int8.t().contiguous().t()

            scale = torch.randn(N, dtype=torch.bfloat16).cuda()   # scale by column
            print("triton matmul")
            try:
                triton_matmul(x, w_int8)
                torch.cuda.synchronize()
                result[D]["triton matmul"][(t_x, t_w)] = triton.testing.do_bench(lambda: triton_matmul(x, w_int8), quantiles=quantiles)[0]
            except:
                print("err")
                pass
            torch.cuda.synchronize()
            torch._dynamo.reset()

            print("alt wo mm")
            def alt_wo_mm(x, w_int8, bias, scale):
                # return (x.unsqueeze(-1) * w_int8.unsqueeze(0)).sum(dim=1)*scale+bias
                return torch.mm(x, w_int8)*scale+bias

            # python < 3.12
            comp_fn=torch.compile(alt_wo_mm, mode='max-autotune')
            comp_fn(x, w_int8, bias, scale)
            comp_fn(x, w_int8, bias, scale)
            comp_fn(x, w_int8, bias, scale)
            result[D]["alt int8 linear"][(t_x, t_w)] = triton.testing.do_bench(lambda: comp_fn(x, w_int8, bias, scale), quantiles=quantiles)[0]

            print("int8 linear")
            try:
                int8_weight_only_linear(x, w_int8, bias, scale)
                torch.cuda.synchronize()
                result[D]["int8 linear"][(t_x, t_w)] = triton.testing.do_bench(lambda: int8_weight_only_linear(x, w_int8, bias, scale), quantiles=quantiles)[0]
            except:
                print("err")
                pass
            torch.cuda.synchronize()

            del w_int8
            w_uint4x2 = torch.randint(0, 255, (D//2, N), dtype=torch.uint8).cuda()
            if t_w:
                w_uint4x2 = w_uint4x2.t().contiguous().t()

            print("uint4x2 linear")
            try:
                assert t_w==1
                uint4x2_weight_only_linear(x, w_uint4x2, bias, scale)
                torch.cuda.synchronize()
                result[D]["uint4x2 linear"][(t_x, t_w)] = triton.testing.do_bench(lambda: uint4x2_weight_only_linear(x, w_uint4x2, bias, scale), quantiles=quantiles)[0]
            except:
                print("err")
                pass
            torch.cuda.synchronize()
            del w_uint4x2, scale, bias



caches = {"triton matmul": _kernel.cache, "int8 linear": int8_weight_only_linear_kernel.cache, "uint4x2 linear": uint4x2_weight_only_linear_kernel.cache}
print("|  X . W  | X . Wt | Xt . W  | Xt . Wt |      model      | config")
for D in result.keys():
    print(f"{1}, {D}, {D}")
    for name in result[D].keys():
        r = result[D][name]
        used_config = None
        if name in caches:
            cache = caches[name]
            for key,config in cache.items():
                if key[0]==1 and key[1]==D and key[2]==D:
                    used_config=config
                    break
        print(f"| {(r[(0,0)] if (0,0) in r else 0):2.4f} | {(r[(0,1)] if (0,1) in r else 0):2.4f} | {(r[(1,0)] if (1,0) in r else 0):2.4f} | {(r[(1,1)] if (1,1) in r else 0):2.4f} | {name:<15} | {used_config}")

# install: pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
# instlal: pip install torchao
# using torch compiled from 2.1.0a0+git9c2122d or nightly
# using cuda 12.0 on A100 GPU
"""
|  X . W  | X . Wt | Xt . W  | Xt . Wt |      model      | config
1, 64, 64
| 0.0103 | 0.0089 | 0.0095 | 0.0083 | cublas linear   | None
| 0.1582 | 0.1514 | 0.1542 | 0.1551 | triton matmul   | BLOCK_M: 32, BLOCK_N: 32, BLOCK_K: 32, SPLIT_K: 1, num_warps: 2, num_ctas: 1, num_stages: 6, enable_warp_specialization: False, enable_persistent: False
| 0.0927 | 0.1114 | 0.0900 | 0.1010 | int8 linear     | BLOCK_SIZE_M: 32, BLOCK_SIZE_N: 256, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 4, num_ctas: 1, num_stages: 4, enable_warp_specialization: False, enable_persistent: False
| 0.0250 | 0.0248 | 0.0252 | 0.0248 | alt int8 linear | None
| 0.0000 | 0.1145 | 0.0000 | 0.0969 | uint4x2 linear  | BLOCK_SIZE_M: 32, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 4, num_ctas: 1, num_stages: 6, enable_warp_specialization: False, enable_persistent: False
| 0.0072 | 0.0077 | 0.0072 | 0.0077 | int4 tinygemm   | None
1, 128, 128
| 0.0091 | 0.0086 | 0.0092 | 0.0086 | cublas linear   | None
| 0.1548 | 0.1761 | 0.1569 | 0.1776 | triton matmul   | BLOCK_M: 16, BLOCK_N: 32, BLOCK_K: 64, SPLIT_K: 1, num_warps: 2, num_ctas: 1, num_stages: 5, enable_warp_specialization: False, enable_persistent: False
| 0.1129 | 0.1096 | 0.1029 | 0.0940 | int8 linear     | BLOCK_SIZE_M: 32, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 2, num_ctas: 1, num_stages: 3, enable_warp_specialization: False, enable_persistent: False
| 0.0257 | 0.0253 | 0.0256 | 0.0251 | alt int8 linear | None
| 0.0000 | 0.1070 | 0.0000 | 0.0928 | uint4x2 linear  | BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 2, num_ctas: 1, num_stages: 6, enable_warp_specialization: False, enable_persistent: False
| 0.0080 | 0.0081 | 0.0080 | 0.0080 | int4 tinygemm   | None
1, 256, 256
| 0.0089 | 0.0097 | 0.0089 | 0.0097 | cublas linear   | None
| 0.1685 | 0.1805 | 0.1648 | 0.1735 | triton matmul   | BLOCK_M: 16, BLOCK_N: 64, BLOCK_K: 32, SPLIT_K: 1, num_warps: 2, num_ctas: 1, num_stages: 5, enable_warp_specialization: False, enable_persistent: False
| 0.0938 | 0.0991 | 0.1099 | 0.1094 | int8 linear     | BLOCK_SIZE_M: 32, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 2, num_ctas: 1, num_stages: 4, enable_warp_specialization: False, enable_persistent: False
| 0.0284 | 0.0260 | 0.0286 | 0.0250 | alt int8 linear | None
| 0.0000 | 0.1109 | 0.0000 | 0.1018 | uint4x2 linear  | BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 4, num_ctas: 1, num_stages: 6, enable_warp_specialization: False, enable_persistent: False
| 0.0083 | 0.0083 | 0.0077 | 0.0083 | int4 tinygemm   | None
1, 512, 512
| 0.0111 | 0.0110 | 0.0118 | 0.0102 | cublas linear   | None
| 0.1663 | 0.1720 | 0.1597 | 0.1666 | triton matmul   | BLOCK_M: 16, BLOCK_N: 32, BLOCK_K: 32, SPLIT_K: 1, num_warps: 2, num_ctas: 1, num_stages: 5, enable_warp_specialization: False, enable_persistent: False
| 0.1089 | 0.1111 | 0.1189 | 0.0973 | int8 linear     | BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 32, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 2, num_ctas: 1, num_stages: 6, enable_warp_specialization: False, enable_persistent: False
| 0.0300 | 0.0264 | 0.0300 | 0.0265 | alt int8 linear | None
| 0.0000 | 0.0961 | 0.0000 | 0.0975 | uint4x2 linear  | BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 32, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 2, num_ctas: 1, num_stages: 4, enable_warp_specialization: False, enable_persistent: False
| 0.0086 | 0.0087 | 0.0086 | 0.0083 | int4 tinygemm   | None
1, 1024, 1024
| 0.0114 | 0.0172 | 0.0119 | 0.0167 | cublas linear   | None
| 0.1698 | 0.1659 | 0.1594 | 0.1691 | triton matmul   | BLOCK_M: 16, BLOCK_N: 32, BLOCK_K: 32, SPLIT_K: 1, num_warps: 2, num_ctas: 1, num_stages: 5, enable_warp_specialization: False, enable_persistent: False
| 0.1048 | 0.1034 | 0.0933 | 0.1035 | int8 linear     | BLOCK_SIZE_M: 32, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 2, num_ctas: 1, num_stages: 4, enable_warp_specialization: False, enable_persistent: False
| 0.0359 | 0.0312 | 0.0357 | 0.0313 | alt int8 linear | None
| 0.0000 | 0.1000 | 0.0000 | 0.1064 | uint4x2 linear  | BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 16, BLOCK_SIZE_K: 256, GROUP_SIZE_M: 8, num_warps: 2, num_ctas: 1, num_stages: 5, enable_warp_specialization: False, enable_persistent: False
| 0.0100 | 0.0100 | 0.0094 | 0.0103 | int4 tinygemm   | None
1, 2048, 2048
| 0.0174 | 0.0235 | 0.0174 | 0.0229 | cublas linear   | None
| 0.1655 | 0.1730 | 0.1585 | 0.1605 | triton matmul   | BLOCK_M: 16, BLOCK_N: 32, BLOCK_K: 64, SPLIT_K: 1, num_warps: 2, num_ctas: 1, num_stages: 6, enable_warp_specialization: False, enable_persistent: False
| 0.1054 | 0.1046 | 0.0956 | 0.0980 | int8 linear     | BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 32, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 2, num_ctas: 1, num_stages: 6, enable_warp_specialization: False, enable_persistent: False
| 0.0458 | 0.0380 | 0.0457 | 0.0379 | alt int8 linear | None
| 0.0000 | 0.1135 | 0.0000 | 0.1096 | uint4x2 linear  | BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 16, BLOCK_SIZE_K: 128, GROUP_SIZE_M: 8, num_warps: 2, num_ctas: 1, num_stages: 5, enable_warp_specialization: False, enable_persistent: False
| 0.0125 | 0.0124 | 0.0125 | 0.0125 | int4 tinygemm   | None
"""