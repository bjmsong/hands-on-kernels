import itertools

import torch
import triton
import triton.language as tl

# import os
# os.environ["TRITON_INTERPRET"]= "1"

def make_dequant_configs(block_sizes, num_warps):
    configs = []
    for bs, ws in itertools.product(block_sizes, num_warps):
        configs.append(triton.Config({"X_BLOCK": bs}, num_warps=ws))
    return configs


DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([128, 256, 512, 1024], [4, 8])


@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=["numels"])
@triton.jit
def dequant_kernel_248(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    out_ptr,
    numels,
    maxq: tl.constexpr,
    bits: tl.constexpr,
    outfeatures: tl.constexpr,
    num_groups: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    # Block indexing
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    row_idx = x_index // outfeatures
    col_idx = x_index % outfeatures

    elements_per_feature: tl.constexpr = 32 // bits

    # Load parameters
    g_idx = tl.load(g_idx_ptr + (row_idx), None, eviction_policy="evict_last")
    qweights = tl.load(
        qweight_ptr + (col_idx + (outfeatures * (row_idx // elements_per_feature))),
        None,
    )

    wf_weights = (row_idx % elements_per_feature) * bits

    wf_zeros = (col_idx % elements_per_feature) * bits

    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    tl.device_assert(g_idx >= 0, "index out of bounds: 0 <= tmp0 < 0")
    groups = tl.where(tmp2, tmp1, g_idx)  # tmp3 are g_idx

    scales = tl.load(scales_ptr + (col_idx + (outfeatures * groups)), None).to(
        tl.float32
    )

    # Unpack weights
    weights = qweights >> wf_weights  # bit shift qweight

    weights = weights & maxq

    # Unpack zeros
    qzero_ncols: tl.constexpr = outfeatures // elements_per_feature
    qzeros = tl.load(
        qzeros_ptr + ((qzero_ncols * groups) + (col_idx // elements_per_feature)),
        None,
        eviction_policy="evict_last",
    )
    zeros = qzeros >> wf_zeros
    zeros = zeros & maxq

    # Dequantize
    zeros = zeros + 1
    weights = weights - zeros
    weights = weights.to(tl.float32)
    weights = scales * weights

    tl.store(out_ptr + (x_index), weights, mask=xmask)


def dequant248(qweight, scales, qzeros, g_idx, bits, maxq=None):
    """
    Launcher for triton dequant kernel.  Only valid for bits = 2, 4, 8
    """

    num_groups = scales.shape[0]
    outfeatures = scales.shape[1]
    infeatures = g_idx.shape[0]

    out = torch.empty((infeatures, outfeatures), device="cuda", dtype=torch.float16)
    numels = out.numel()
    maxq = 2**bits - 1 if maxq is None else maxq
    grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)  # noqa: E731

    dequant_kernel_248[grid](
        g_idx,
        scales,
        qweight,
        qzeros,
        out,
        numels,
        maxq=maxq,
        bits=bits,
        outfeatures=outfeatures,
        num_groups=num_groups,
    )
    return out


def quant_matmul_248(
    input, qweight, scales, qzeros, g_idx, bits, maxq=None, transpose=False
):
    W = dequant248(qweight, scales, qzeros, g_idx, bits, maxq=maxq)
    if transpose:
        return input @ W.t()
    return input @ W

def make_tensor(M, N, dtype):
    if dtype == torch.int32:
        # Fill with random integers for int32 type
        res = torch.randint(low=-2147483648, high=2147483647, size=(M, N), dtype=dtype, device="cuda")
    else:
        # Fill with normally distributed random values for other types
        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    return res

if __name__ == '__main__':

    m, k, n = 2048, 4096, 4096
    x = make_tensor(m, k, dtype=torch.float16)   # activation 
    w = make_tensor(k//8, n, dtype=torch.int32)  # weight, 8*int4 = int32

    # w_float = scale * w_int - zero
    groupsize = 128  # group quantization
    g = k // groupsize
    zeros = make_tensor(g, n//8, torch.int32)
    scales = make_tensor(g, n, torch.float16)
    bits = 4
    maxq = 2**bits - 1
    g_idx = torch.tensor([i // groupsize for i in range(n)], dtype=torch.int32, device="cuda")

    quant_matmul_248(x, w, scales, zeros, g_idx, bits, maxq)

