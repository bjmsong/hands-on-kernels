import itertools
import torch
import triton
from triton import language as tl
# from quantize import quantize

M_range = [2 ** i for i in range(0, 15, 2)]
N_K_range = [2 ** i for i in range(10, 15, 2)]
matrix_range = list(itertools.product(M_range, N_K_range, N_K_range))

@triton.jit()
def swizzle_tile(pid,
                m, n,
                block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n

@triton.jit()
def matmul_split_k_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            stride_scales_g, stride_scales_n,
            stride_zeros_g, stride_zeros_n,
            groupsize,
            m, n, k,
            block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
            group_m: tl.constexpr, split_k: tl.constexpr):
    
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    total_blocks_k = tl.cdiv(k, block_k*split_k)

    pid_m, pid_n = swizzle_tile(pid,
                                m, n,
                                block_m, block_n, group_m)
    
    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    offs_k = pid_k*block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4    # [0,1,2,3,4,5,6,7,8,9...] -> [0,4,8,12,16,20,24,28,0,4...]
    zeros_shifter = (offs_bn % 8) * 4
    
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):
        
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        g_id = (k * split_k + pid_k) // (groupsize // block_k)

        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)
        
        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr) 

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF   # extract int4
        b = b * scales[None, :] - zeros[None, :]  # int4 -> fp16

        acc += tl.dot(a, b)
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += (block_k // 8) * split_k * stride_bk

    acc.to(tl.float16)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc, sem='release')

def matmul_split_k(a, b, scales, zeros):

    m, k = a.shape
    _, n = b.shape
    
    quant_groupsize = 128
    block_m = 16
    block_n = 32
    block_k = 128
    group_m = 8
    num_stages = 3
    num_warps = 4
    split_k = 4

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k
    
    grid = (total_programs_mn, total_programs_k)

    print(f"problem m size: {m}, tile size m: {block_m}, total blocks m: {total_blocks_m}")
    print(f"problem n size: {n}, tile size n: {block_n}, total blocks n: {total_blocks_n}")
    print(f"problem k size: {k}, tile size k: {block_k}, total thread blocks k: {split_k}")

    print(f"total thread blocks k: {k}, total thread blocks m and total thread blocks n = {total_blocks_m=} x {total_blocks_n} = {total_programs_mn}")
    print(f"{total_programs_mn=}, {total_programs_k=}")
    
    c = torch.zeros((m, n), device=a.device, dtype=torch.float16)
    # triton.compiler.compiler.CompiledKernel
    k = matmul_split_k_kernel[grid](a, b, c, scales, zeros,
                              a.stride(0), a.stride(1),
                              b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1),
                              scales.stride(0), scales.stride(1),
                              zeros.stride(0), zeros.stride(1),
                              quant_groupsize,
                              m, n, k,
                              block_m, block_n, block_k,
                              group_m, split_k, num_stages=num_stages, num_warps=num_warps)
    
    # print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n")
    print(f"{k.n_regs} registers used, {k.n_spills} spills \n")

    with open('matmul_split_k.txt', 'w') as f:

        print(f"{k.n_regs} registers used, {k.n_spills} spills \n")
        # print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)
        print("IR", k.asm['ttir'], file=f)
        print("TTGIR", k.asm['ttgir'], file=f)
        print("PTX", k.asm['ptx'], file=f)
        # print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)
        print(f"{k.n_regs} registers used, {k.n_spills} spills \n")

    return c

def make_tensor(M, N, dtype):
    if dtype == torch.int32:
        # Fill with random integers for int32 type
        res = torch.randint(low=-2147483648, high=2147483647, size=(M, N), dtype=dtype, device="cuda")
    else:
        # Fill with normally distributed random values for other types
        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    return res

def calculate_diff():
    eps = 1e-3
    for M, N, K in matrix_range:
        x = make_tensor(M, K, dtype=torch.float16)
        w = make_tensor(K//8, N, dtype=torch.int32)
        groupsize = 1
        g = K // groupsize
        zeros = make_tensor(g, N//8, torch.int32)
        scales = make_tensor(g, N, torch.float16)
        output_torch = torch.matmul(x, w.t())
        output_triton = matmul_split_k(x, w, scales, zeros)
        percentage_error = (torch.abs(output_torch - output_triton).to(torch.float64) / (eps + torch.abs(output_torch))) * 100
        print(f"diff(%) of {M,N,K} is {percentage_error.mean()}")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[list(_) for _ in matrix_range],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['triton'],
        # Label name for the lines
        line_names=["Triton"],
        # Line styles
        styles=[('green', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    x = make_tensor(M, K, dtype=torch.float16)
    w = make_tensor(K//8, N, dtype=torch.int32)
    groupsize = 1
    g = K // groupsize
    zeros = make_tensor(g, N//8, torch.int32)
    scales = make_tensor(g, N, torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_split_k(x, w, scales, zeros), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-9 / ms
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == '__main__':

    m = 16
    k = 4096
    n = 4096

    x = make_tensor(m, k, dtype=torch.float16)   # activation 
    # w = make_tensor(n, k, dtype=torch.float16)
    # zeros, scales, w_int = quantize(w, zeros, scales)
    w = make_tensor(k//8, n, dtype=torch.int32)  # weight, 8*int4 = int32

    # w_float = scale * w_int - zero
    groupsize = 128  # group quantization
    g = k // groupsize
    zeros = make_tensor(g, n//8, torch.int32)
    scales = make_tensor(g, n, torch.float16)

    # base = no_autotune(groupsize, a, b, scales, zeros)
    # print(f"{base.shape=}, {base[0][0:4]}")

    # c = custom_qlinear(a, b, scales, zeros)
    # print(f"{c.shape=}, {c[0][0:4]}")

    split_k_output = matmul_split_k(x, w, scales, zeros)
    print(f"{split_k_output.shape=}, {split_k_output[0][0:4]}")

    # torch_output = torch.matmul(x, w.t())
    # print(f"triton_output={split_k_output}")
    # print(f"torch_output={torch_output}")
    # if torch.allclose(split_k_output, torch_output, atol=1e-2, rtol=1e-2):
    #     print("✅ Triton and Torch match")
    # else:
    #     print("❌ Triton and Torch differ")

    benchmark.run(show_plots=True, print_data=True, save_path="plot/")

    calculate_diff()
