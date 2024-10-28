import itertools
import torch
import triton
import triton.language as tl
from triton import Config
import sys
sys.path.append(".")
from utils.quantize_rowwise import quantize_rowwise
from utils.dequantize_rowwise import dequantize_rowwise
from utils.utils import _test_memory

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
] + get_configs_io_bound()

@triton.autotune(
    configs = [
        Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=1),
        Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=1),
    ] + config_list,
    key=['M', 'N', 'K'],
)
@triton.jit
def int8_weight_only_linear_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, s_ptr, y_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `x_ptr`
    # by to get the element one row down (A has M rows).
    stride_xm, stride_xk,
    stride_wk, stride_wn,
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

    s = tl.load(s_ptr + offs_wn)[None, :]
    y = (s * (accumulator * (1.0 / 127.0))).to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix Y with masks.
    offs_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)

def int8_weight_only_linear(x, w, s):
    # Check constraints.
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    # assert x.is_contiguous(), "Matrix x must be contiguous"
    # assert w.is_contiguous(), "Matrix w must be contiguous"
    M, K = x.shape
    K, N = w.shape

    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    if w.stride(0) > 1 and w.stride(1) > 1:
        w = w.contiguous()

    # Allocates output.
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    int8_weight_only_linear_kernel[grid](
        x, w, s, output,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        output.stride(0), output.stride(1),
    )
    return output

def test_correctness():
    torch.manual_seed(0)
    M, K, N = 32, 256, 1024
    x = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    w = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)

    w_int8, scale = quantize_rowwise(w)   # preprocess

    triton_output = int8_weight_only_linear(x, w_int8.t(), scale)
    torch_output = torch.matmul(x, w.t())
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

# test_correctness()

M_range = [2 ** i for i in range(0, 15, 2)]
N_K_range = [2 ** i for i in range(10, 15, 2)]
# M_range = [4096,8192,16384]
# N_K_range = [16384]
matrix_range = list(itertools.product(M_range, N_K_range, N_K_range))
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[list(_) for _ in matrix_range],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['torch', 'triton'],
        # Label name for the lines
        line_names=["torch", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    W = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
    W_int8, scale = quantize_rowwise(W)
    W_int8_t = W_int8.t()
    W_t = W.t()
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        # 对每个kernel进行25次的warm_up和100次iteration
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, W_t), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: int8_weight_only_linear(a, W_int8_t, scale), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-9 / ms
    return perf(ms), perf(max_ms), perf(min_ms)

# benchmark.run(show_plots=True, print_data=True, save_path="plot/")

## calculate diff
def calculate_diff():
    eps = 1e-3
    for M, N, K in matrix_range:
        a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
        W = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
        W_int8, state_W = quantize_rowwise(W)
        output_torch = torch.matmul(a, W.t())
        output_triton = int8_weight_only_linear(a, W_int8.t(), state_W)
        denominator = eps + torch.abs(output_torch) if torch.abs(output_torch).min() == 0 else torch.abs(output_torch)
        percentage_error = (torch.abs(output_torch - output_triton) / denominator) * 100
        print(f"diff(%) of {M,N,K} is {percentage_error.median()}")

# calculate_diff()

def peak_memory(backend):
    for M, N, K in matrix_range:
        def torch_call():
            a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
            W = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
            torch.matmul(a,W.t())

        def triton_call():
            W_int8 = torch.empty((N, K), device="cuda", dtype=torch.int8)
            state_W = torch.empty(N, device="cuda", dtype=torch.bfloat16)
            a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
            int8_weight_only_linear(a, W_int8.t(), state_W)

        QUANTILES = [0.5, 0.2, 0.8]
        if backend == "triton":
            mem_50, mem_20, mem_80 = _test_memory(triton_call, quantiles=QUANTILES)
            print(f"Triton Peak Memory of {M,N,K} is {mem_50, mem_20, mem_80}")

        if backend == "torch":
            mem_50, mem_20, mem_80 = _test_memory(torch_call, quantiles=QUANTILES)
            print(f"Torch Peak Memory of {M,N,K} is {mem_50, mem_20, mem_80}")

# peak_memory(backend = "triton")