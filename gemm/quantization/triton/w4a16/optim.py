import torch
import triton
from triton import language as tl
# from actual_base_gptq_4 import triton_matmul4

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
def matmul_split_k_kernel(x_ptr, w_ptr, out_ptr, scales_ptr, zeros_ptr,
            stride_xm, stride_xk,
            stride_wk, stride_wn,
            stride_outm, stride_outn,
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

    offs_xm = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_wn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + ((offs_k[:, None] // 8) * stride_wk + offs_wn[None, :] * stride_wn)

    scales_ptrs = scales_ptr + offs_wn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_wn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_wn % 8) * 4
    
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):
        
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        
        g_id = (k * split_k + pid_k) // (groupsize // block_k)

        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)
        
        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr) 

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        w = (w >> shifter[:, None]) & 0xF
        w = w * scales[None, :] - zeros[None, :]

        acc += tl.dot(x, w)
        x_ptrs += block_k * split_k * stride_xk
        w_ptrs += (block_k // 8) * split_k * stride_wk

    acc.to(tl.float16)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)

    out_ptrs = out_ptr + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    tl.atomic_add(out_ptrs, acc, sem='release')

def matmul_split_k(x, w, scales, zeros):

    m, k = x.shape
    _, n = w.shape
    
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
    
    out = torch.zeros((m, n), device=x.device, dtype=torch.float16)
    k = matmul_split_k_kernel[grid](x, w, out, scales, zeros,
                              x.stride(0), x.stride(1),
                              w.stride(0), w.stride(1),
                              out.stride(0), out.stride(1),
                              scales.stride(0), scales.stride(1),
                              zeros.stride(0), zeros.stride(1),
                              quant_groupsize,
                              m, n, k,
                              block_m, block_n, block_k,
                              group_m, split_k, num_stages=num_stages, num_warps=num_warps)
    
    # print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n")

    # with open('matmul_split_k.txt', 'w') as f:

    #     print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)
    #     print("IR", k.asm['ttir'], file=f)
    #     print("TTGIR", k.asm['ttgir'], file=f)
    #     print("PTX", k.asm['ptx'], file=f)
    #     print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)

    return out

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

    m, k, n = 16, 4096, 4096
    groupsize = 128
    g = k // groupsize

    x = make_tensor(m, k, dtype=torch.float16)
    w = make_tensor(k//8, n, dtype=torch.int32)
    zeros = make_tensor(g, n//8, torch.int32)
    scales = make_tensor(g, n, torch.float16)
    
    # base = no_autotune(groupsize, a, b, scales, zeros)
    # print(f"{base.shape=}, {base[0][0:4]}")

    # c = custom_qlinear(a, b, scales, zeros)
    # print(f"{c.shape=}, {c[0][0:4]}")

    split_k_output = matmul_split_k(x, w, scales, zeros)
    print(f"{split_k_output.shape=}, {split_k_output[0][0:4]}")