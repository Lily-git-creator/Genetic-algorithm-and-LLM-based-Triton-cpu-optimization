# Triton 矩阵乘法内核 (优化平铺版本)
@triton.jit
def matmul_kernel(
    # 指向输入输出矩阵数据块的指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 步长（相邻行/列之间的内存跨度）
    stride_am, stride_ak,  # A 的步长 (M, K)
    stride_bk, stride_bn,  # B 的步长 (K, N)
    stride_cm, stride_cn,  # C 的步长 (M, N)
    # 平铺大小 (Tile Size) - 编译器常量，用于优化
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    计算 C = A @ B 的 Triton 内核。
    使用二维网格布局和双重循环平铺以优化 CPU 缓存利用率。
    """
    # 当前程序负责计算输出矩阵 C 中的哪个分块 (pid_m, pid_n)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 计算当前分块在 A 和 B 中的起始内存偏移
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 初始化累加器（存放当前分块的部分和）
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 双重循环平铺: 沿 K 维度分块加载和计算，提高缓存命中率
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 从 A 加载当前块 (BLOCK_SIZE_M x BLOCK_SIZE_K)
        a = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        # 从 B 加载当前块 (BLOCK_SIZE_K x BLOCK_SIZE_N)
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        # 矩阵乘累加: [BLOCK_SIZE_M, BLOCK_SIZE_N] += [BLOCK_SIZE_M, BLOCK_SIZE_K] @ [BLOCK_SIZE_K, BLOCK_SIZE_N]
        accumulator += tl.dot(a, b)
        # 移动到 K 维度的下一个分块
        offs_k += BLOCK_SIZE_K

    # 将最终结果写回 C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a, b):
    """
    使用 Triton 内核执行矩阵乘法 a @ b。
    参数:
        a, b: 2D PyTorch 张量 (CPU)
    返回:
        c: 2D PyTorch 张量 (CPU), 结果为 a @ b
    """
    # 检查输入
    assert a.dim() == 2 and b.dim() == 2, "输入必须是二维张量"
    assert a.shape[1] == b.shape[0], f"矩阵维度不匹配: {a.shape} @ {b.shape}"
    
    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check, "内部维度 K 必须相等"
    
    # 预分配输出张量
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    
    # 定义网格和分块大小 (可根据硬件微调)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32  # 平铺参数
    
    # 计算二维网格大小 (每个输出分块一个线程)
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # 启动内核
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c