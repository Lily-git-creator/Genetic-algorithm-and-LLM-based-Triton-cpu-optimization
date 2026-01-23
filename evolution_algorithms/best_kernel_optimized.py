import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    a_ptrs = a_ptr + offs_am[:, None] * stride_am
    b_ptrs = b_ptr + offs_bn[None, :] * stride_bn
    
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(
            a_ptrs + offs_k[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
        )
        
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        b = tl.load(
            b_ptrs + offs_k[:, None] * stride_bk,
            mask=b_mask,
            other=0.0,
        )
        
        accumulator += tl.dot(a, b)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a, b):
    assert a.dim() == 2 and b.dim() == 2, "输入必须是二维张量"
    assert a.shape[1] == b.shape[0], f"矩阵维度不匹配: {a.shape} @ {b.shape}"
    
    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check, "内部维度 K 必须相等"
    
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c