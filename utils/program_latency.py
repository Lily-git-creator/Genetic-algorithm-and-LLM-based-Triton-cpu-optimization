import torch
import triton
import triton.language as tl
import time
import statistics
import json
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 待测 Baseline 代码
# ==========================================
# 脚本独立运行，这里直接内嵌 Baseline Kernel
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
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_SIZE_K
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a, b):
    # 简化版 wrapper，仅用于测试
    M, K = a.shape
    _, N = b.shape
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return c

def run_analysis(num_samples=1000, warmup=50):
    print(f"🔬 Starting Analysis: {warmup} warmups + {num_samples} samples")
    
    # 1. 准备数据
    torch.manual_seed(42)
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device='cpu', dtype=torch.float32)
    b = torch.randn((K, N), device='cpu', dtype=torch.float32)

    # 2. 预热 (Warmup)
    # 观察预热阶段是否有明显的加速趋势（JIT 编译或缓存填充）
    print("🔥 Warming up...")
    warmup_times = []
    for _ in range(warmup):
        t0 = time.perf_counter()
        triton_matmul(a, b)
        t1 = time.perf_counter()
        warmup_times.append((t1 - t0) * 1000) # ms

    # 3. 正式采样 (Sampling)
    print("⚡ Sampling...")
    times = []
    for i in range(num_samples):
        t0 = time.perf_counter()
        triton_matmul(a, b)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000) # ms

    # 4. 统计分析
    avg = statistics.mean(times)
    med = statistics.median(times)
    minimum = min(times)
    maximum = max(times)
    stdev = statistics.stdev(times)
    cv = stdev / avg # 变异系数 (Coefficient of Variation)

    print("\n" + "="*40)
    print("📊 Statistical Report (ms)")
    print("="*40)
    print(f"Count    : {num_samples}")
    print(f"Mean     : {avg:.4f} ms")
    print(f"Median   : {med:.4f} ms")
    print(f"Min      : {minimum:.4f} ms")
    print(f"Max      : {maximum:.4f} ms")
    print(f"Stdev    : {stdev:.4f}")
    print(f"CV       : {cv:.2%} (Noise Level)")
    print("-" * 40)
    
    # 简单的建议逻辑
    if cv < 0.05:
        print("✅ Environment is STABLE. (Mean is okay)")
    elif cv < 0.15:
        print("⚠️ Environment is NOISY. (Use Median)")
    else:
        print("❌ Environment is VERY UNSTABLE. (Use Median + High Runs, or check Server Load)")

    if avg - med > stdev:
        print("⚠️ Detected Right-Skewed distribution (Outliers present). Median is much safer than Mean.")

    # 5. 绘图 (保存为文件)
    plt.figure(figsize=(12, 6))
    
    # 子图 1: 时间序列 (看趋势)
    plt.subplot(1, 2, 1)
    plt.plot(times, alpha=0.7, color='blue', label='Latency')
    plt.axhline(avg, color='red', linestyle='--', label=f'Mean ({avg:.2f})')
    plt.axhline(med, color='green', linestyle='-', label=f'Median ({med:.2f})')
    plt.title("Latency Timeline")
    plt.xlabel("Sample Index")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图 2: 直方图 (看分布)
    plt.subplot(1, 2, 2)
    plt.hist(times, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(avg, color='red', linestyle='--', label='Mean')
    plt.axvline(med, color='green', linestyle='-', label='Median')
    plt.title("Latency Distribution")
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("latency_distribution.png")
    print(f"\n📈 Chart saved to 'latency_distribution.png'")

    # 6. 保存原始数据
    with open("raw_latencies.json", "w") as f:
        json.dump(times, f)

if __name__ == "__main__":
    run_analysis(num_samples=500, warmup=20)
