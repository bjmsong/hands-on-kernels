import torch
from typing import Any, Callable, Dict, List, Optional, Union

def _test_memory(
    func: Callable,
    _iter: int = 10,
    quantiles: Optional[List[float]] = None,
    return_mode="mean",
) -> float:
    assert return_mode in ["min", "max", "mean", "median"]
    total_mem = []

    for _ in range(_iter):
        torch.cuda.memory.reset_peak_memory_stats()
        func()
        # Convert to MB
        mem = torch.cuda.max_memory_allocated() / 2**20
        total_mem.append(mem)

    total_mem = torch.tensor(total_mem, dtype=torch.float)
    if quantiles is not None:
        quantiles_data = torch.quantile(
            total_mem, torch.tensor(quantiles, dtype=torch.float)
        ).tolist()
        if len(quantiles_data) == 1:
            quantiles_data = quantiles_data[0]
        return quantiles_data
    return getattr(torch, return_mode)(total_mem).item()

if __name__ == "__main__":
    M, K, N = 16384, 16384, 16384
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    QUANTILES = [0.5, 0.2, 0.8]
    def call():
        torch.matmul(a,b)
    mem_50, mem_20, mem_80 = _test_memory(call, quantiles=QUANTILES)
    print(mem_50, mem_20, mem_80)