# https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes

import itertools
import torch
import triton
import sys
sys.path.append(".")
from utils.quantize_rowwise import quantize_rowwise
from utils.utils import _test_memory
from int8_matmul_rowwise_dequantize import int8_matmul_rowwise_dequantize
import time

def matmul(X_int8, state_X, W_int8, state_W):

    return int8_matmul_rowwise_dequantize(X_int8, W_int8, state_X, state_W, bias = None)

torch.manual_seed(0)
M, K, N = 16384, 4096, 8192
a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
W = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
# a = torch.tensor([[0.1,0.2],[0.3,0.4]], device='cuda', dtype=torch.float16)
# b = torch.tensor([[0.1,0.3],[0.2,0.4]], device='cuda', dtype=torch.float16)
W_int8, state_W = quantize_rowwise(W)  # quantization before inference
W_int8_t = W_int8.t()
X_int8, state_X = quantize_rowwise(a)  # fused in former kernel
start_time = time.time()
triton_output = matmul(X_int8, state_X, W_int8_t, state_W)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"All cost: {elapsed_time} 秒")
torch_output = torch.matmul(a, W.t())
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

M_range = [2 ** i for i in range(10, 15, 2)]
N_K_range = [2 ** i for i in range(10, 15, 2)]
# M_range = [4096]
# N_K_range = [1024, 16384]
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
    W_int8, state_W = quantize_rowwise(W)
    X_int8, state_X = quantize_rowwise(a)
    W_int8_t = W_int8.t()
    W_t = W.t()
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        # 对每个kernel进行25次的warm_up和100次iteration
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, W_t), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(X_int8, state_X,  W_int8_t, state_W), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-9 / ms
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True, save_path="plot/")

## calculate diff
eps = 1e-3
for M, N, K in itertools.product(M_range, N_K_range, N_K_range):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    W = torch.randn((N, K), device='cuda', dtype=torch.float16)
    W_int8, state_W = quantize_rowwise(W)
    X_int8, state_X = quantize_rowwise(a)
    output_torch = torch.matmul(a, W.t())
    output_triton = matmul(X_int8, state_X,  W_int8.t(), state_W)
    percentage_error = (torch.abs(output_torch - output_triton).to(torch.float64) / (eps + torch.abs(output_torch))) * 100
    print(f"diff(%) of {M,N,K} is {percentage_error.mean()}")

## peak memory
# for M, N, K in itertools.product(M_range, N_K_range, N_K_range):
#     print(M,N,K)
#     a = torch.randn((M, K), device='cuda', dtype=torch.float16)
#     W = torch.randn((N, K), device='cuda', dtype=torch.float16)
#     W_int8, state_W = quantize_rowwise(W)
#     X_int8, state_X = quantize_rowwise(a)
#     W_int8_t = W_int8.t()
#     W_t = W.t()
#     QUANTILES = [0.5, 0.2, 0.8]
#     def torch_call():
#         torch.matmul(a,W_t)
#     def triton_call():
#         matmul(X_int8, state_X,  W_int8_t, state_W)
#     mem_50, mem_20, mem_80 = _test_memory(triton_call, quantiles=QUANTILES)
#     print(mem_50, mem_20, mem_80)