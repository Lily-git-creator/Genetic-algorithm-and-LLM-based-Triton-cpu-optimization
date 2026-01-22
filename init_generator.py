# === 文件名: init_generator.py ===
import os
import random
import shutil

# 目标文件夹
CODE_DIR = "code"
os.makedirs(CODE_DIR, exist_ok=True)

# 基础模板 (基于你的 Baseline，但参数参数化)
TEMPLATE = """
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
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0
        )
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_SIZE_K

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a, b):
    # --- Auto Generated Config ---
    BLOCK_SIZE_M = {BM}
    BLOCK_SIZE_N = {BN}
    BLOCK_SIZE_K = {BK}
    GROUP_SIZE_M = {GM}
    num_warps = {NW}
    num_stages = {NS}
    # -----------------------------
    
    M, K = a.shape
    _, N = b.shape
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    
    # 使用 1D Launch 以支持 Group Swizzle
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return c
"""

def generate_seeds(num_seeds=8):
    # 参数搜索空间
    block_choices = [16, 32, 64, 128] # M, N
    k_choices = [16, 32, 64]          # K 通常小一点
    group_choices = [4, 8]            # L2 Cache 优化参数
    warp_choices = [2, 4, 8]          # 线程束数量
    stage_choices = [1, 2, 3]         # 流水线级数 (CPU通常没用，但GPU有用，保留以备后用)

    print(f"Generating {num_seeds} random seeds into '{CODE_DIR}/'...")

    for i in range(num_seeds):
        params = {
            "BM": random.choice(block_choices),
            "BN": random.choice(block_choices),
            "BK": random.choice(k_choices),
            "GM": random.choice(group_choices),
            "NW": random.choice(warp_choices),
            "NS": random.choice(stage_choices)
        }
        
        # 填充模板
        code_content = TEMPLATE.format(**params)
        
        filename = os.path.join(CODE_DIR, f"seed_{i}_M{params['BM']}_N{params['BN']}_K{params['BK']}.py")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code_content)
        print(f"  - Created {filename}")

if __name__ == "__main__":
    # 清空旧代码 (保留 baseline.py 如果有的话)
    # 这里简单粗暴直接覆盖生成
    generate_seeds(8)
