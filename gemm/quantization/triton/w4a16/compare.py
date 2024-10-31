import itertools
import torch
import triton
from gptq_v1 import quant_matmul_248 as quant_matmul_248_v1
from gptq_v2 import quant_matmul_248 as quant_matmul_248_v2
from splitk import matmul_split_k
torch.manual_seed(42)

def make_tensor(M, N, dtype):
    if dtype == torch.int32:
        # Fill with random integers for int32 type
        res = torch.randint(low=-2147483648, high=2147483647, size=(M, N), dtype=dtype, device="cuda")
    else:
        # Fill with normally distributed random values for other types
        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    return res

matrix_range = [(256, 4096, 4096*4)]
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[list(_) for _ in matrix_range],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['v1', 'v2', 'v3'],
        # Label name for the lines
        line_names=["v1", "v2", "v3"],
        # Line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    x = make_tensor(M, K, dtype=torch.float16)   # activation 
    w = make_tensor(K//8, N, dtype=torch.int32)  # weight, 8*int4 = int32
    groupsize = 128
    g = K // groupsize
    zeros = make_tensor(g, N//8, torch.int32)
    scales = make_tensor(g, N, torch.float16)
    bits = 4
    maxq = 2**bits - 1
    g_idx = torch.tensor([i // groupsize for i in range(K)], dtype=torch.int32, device="cuda")
    quantiles = [0.5, 0.2, 0.8]
    # g = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(g):
    #     matmul_split_k(x, w, scales, zeros)

    if provider == 'v1':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: quant_matmul_248_v1(x, w, scales, zeros, g_idx, bits, maxq), quantiles=quantiles)
    if provider == 'v2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: quant_matmul_248_v2(x, w, scales, zeros, g_idx, bits, maxq), quantiles=quantiles)
    if provider == 'v3':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_split_k(x, w, scales, zeros), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-9 / ms
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == '__main__':

    m, k, n = 2048, 4096, 4096
    x = make_tensor(m, k, dtype=torch.float16)   # activation 
    w = make_tensor(k//8, n, dtype=torch.int32)  # weight, 8*int4 = int32

    groupsize = 128
    g = k // groupsize
    zeros = make_tensor(g, n//8, torch.int32)
    scales = make_tensor(g, n, torch.float16)
    bits = 4
    maxq = 2**bits - 1
    g_idx = torch.tensor([i // groupsize for i in range(k)], dtype=torch.int32, device="cuda")

    v1_output = quant_matmul_248_v1(x, w, scales, zeros, g_idx, bits, maxq)
    v2_output = quant_matmul_248_v2(x, w, scales, zeros, g_idx, bits, maxq)
    v3_output = matmul_split_k(x, w, scales, zeros)
    print(f"v1_output={v1_output}")
    print(f"v2_output={v2_output}")
    print(f"v3_output={v3_output}")
    if torch.allclose(v1_output, v3_output, atol=1e-2, rtol=1e-2):
        print("✅ V1 and V3 match")
    else:
        print("❌ V1 and V3 differ")

    benchmark.run(show_plots=True, print_data=True, save_path="plot/")