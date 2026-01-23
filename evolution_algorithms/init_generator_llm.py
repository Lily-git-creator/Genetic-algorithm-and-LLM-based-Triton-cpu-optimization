import os
import glob
from llm_handler import query_init_generation

# 定义策略库：(策略名, 具体指导描述)
STRATEGIES = [
    (
        "Small Tile / Latency Optimized",
        "使用较小的分块大小 (例如 BLOCK_SIZE_M=32, N=32, K=32)。减少每个线程块的寄存器压力，增加并行度(Occupancy)。适用于处理小矩阵或高并发场景。减少 num_warps 到 2 或 4。"
    ),
    (
        "Large Tile / Throughput Optimized",
        "使用较大的分块大小 (例如 BLOCK_SIZE_M=128, N=128, K=32)。通过计算更多的数据来掩盖内存读取延迟。增加 num_warps 到 8 以支持大分块。注意内存合并访问。"
    ),
    (
        "L2 Cache Swizzle / Grouped Launch",
        "实现 Grouped Launch (也称为 Swizzle) 技术。通过重新映射 pid (Program ID) 来改变计算块的执行顺序，使得访存模式对 L2 Cache 更友好，增加缓存命中率。请手动计算 pid_m 和 pid_n。"
    ),
    (
        "Vectorized Load / Memory Coalescing",
        "专注于内存访问优化。确保所有 load 和 store 操作都是向量化的。检查 stride 计算，确保连续内存访问。调整 BLOCK_SIZE_K 为 64 或更大以减少循环开销。"
    ),
    (
        "Double Buffering / Pipelining",
        "尝试开启软件流水线 (Software Pipelining)。设置 num_stages=2 或 3 (即使是在 CPU 上，也可以尝试让编译器进行指令调度优化)。调整循环结构以支持预取。"
    ),
    (
        "Rectangular Tiles",
        "放弃正方形分块。尝试长方形分块，例如 BLOCK_SIZE_M=128, BLOCK_SIZE_N=32。这种形状在某些不对称的矩阵乘法或特定的 CPU 缓存架构上可能表现更好。"
    )
]

def generate_llm_seeds(baseline_path="code/baseline.py"):
    if not os.path.exists(baseline_path):
        print(f"❌ Baseline file not found: {baseline_path}")
        return

    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline_code = f.read()

    print(f"🧠 LLM Initializing Population from '{baseline_path}'...")
    print(f"   Strategies to apply: {len(STRATEGIES)}")

    # 1. 保留 Baseline 作为 seed_0
    with open("code/seed_0_baseline.py", "w", encoding='utf-8') as f:
        f.write(baseline_code)

    # 2. 生成变体
    for i, (name, desc) in enumerate(STRATEGIES):
        print(f"\n✨ Generating Seed {i+1}: [{name}]...")
        
        try:
            generated_code = query_init_generation(baseline_code, name, desc)
            
            if generated_code:
                filename = f"code/seed_{i+1}_{name.replace(' ', '_').replace('/', '')}.py"
                with open(filename, "w", encoding='utf-8') as f:
                    f.write(generated_code)
                print(f"   ✅ Saved to {filename}")
            else:
                print("   ❌ LLM failed to generate code.")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    # 确保 code 文件夹存在
    os.makedirs("code", exist_ok=True)
    
    # 如果没有 baseline，创建一个简单的
    if not os.path.exists("code/baseline.py"):
        print("⚠️ No baseline found, creating a dummy one for bootstrapping...")
        # (这里可以写入你之前提供的那个最基础的 matmul 代码)
        pass 
        
    generate_llm_seeds()
