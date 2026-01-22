import os
import requests
import json
import re

# 配置
API_KEY = os.getenv("LLM_API_KEY", "sk-") # 替换你的Key
API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-chat"

def extract_python_code(text):
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

def _send_request(system_prompt, user_prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "stream": False
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return extract_python_code(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        print(f"❌ LLM Request Failed: {e}")
        return None

# --- 1. 基础变异 / 改写 (Para / Mutation) ---
def query_mutation(code, latency, feedback=""):
    sys_prompt = "你是一位精通 Triton 和 CUDA 的高性能计算专家。你的目标是极致优化矩阵乘法算子。"
    
    # 构建优化建议列表
    hints = """
    优化方向建议：
    1. **分块调优 (Tiling)**: 调整 BLOCK_SIZE_M/N/K。尝试非 2 的幂次或极端长宽比（如 128x32 vs 64x64）。
    2. **L2 Cache 局部性**: 检查是否使用了 Grouped Launch (Swizzle) 技术来提高 L2 缓存命中率。
    3. **流水线 (Pipelining)**: 如果是 GPU 环境，尝试调整 num_stages。
    4. **向量化访存**: 确保内存加载是连续的，利用 tl.load 的 mask 和 other 参数优化边界检查。
    5. **指令并行**: 调整 num_warps 以平衡寄存器压力和掩盖延迟。
    """
    
    user_prompt = f"""
    请优化以下 Triton 矩阵乘法代码以降低计算延迟。
    
    【当前状态】
    - 平均延迟: {latency*1000:.4f} ms
    - 外部反馈: {feedback}
    
    {hints}

    【任务要求】
    1. 保持代码逻辑正确（矩阵乘法）。
    2. 如果代码中包含配置参数（如 BLOCK_SIZE），请进行针对性的修改。
    
    【待优化代码】
    ```python
    {code}
    ```
    """
    return _send_request(sys_prompt, user_prompt)

# --- 2. 杂交 / 交叉 (Crossover for GA) ---
def query_crossover(code_mom, code_dad):
    sys_prompt = "你是一位资深代码架构师。你需要将两份 Triton 内核代码的优点融合，生成一份更强的代码。"
    
    user_prompt = f"""
    我有两份不同的 Triton 矩阵乘法实现。请将它们“杂交”，结合双方的优点。
    
    【父代 A (Mom)】
    ```python
    {code_mom}
    ```
    
    【父代 B (Dad)】
    ```python
    {code_dad}
    ```
    
    【任务要求】
    1. 分析两份代码的配置（BLOCK_SIZE, num_warps, 循环结构, 内存访问模式）。
    2. 创造一份新的“子代”代码，它应该继承父母双方看起来最高效的策略。
    3. 例如：如果 A 的分块大小很大但 B 的 L2 Cache 优化写得好，请将 B 的逻辑应用到 A 的参数上。
    4. 仅输出融合后的 Python 代码。
    """
    return _send_request(sys_prompt, user_prompt)

# --- 3. 差分进化引导 (DE - Semantic) ---
def query_de_mutation(target_code, best_code, random_code):
    sys_prompt = "你是一个进化算法优化器。你的工作是分析‘好代码’相对于‘普通代码’的结构优势，并将这些优势应用到‘目标代码’上。"
    
    user_prompt = f"""
    我们需要对目标代码 (Target) 进行变异操作。请参考最佳样本 (Best) 和随机样本 (Random) 的差异。
    
    【输入数据】
    1. **目标代码 (Target)**: 待优化的代码。
    ```python
    {target_code}
    ```
    
    2. **最佳参考 (Best)**: 当前种群中性能最好的代码。
    ```python
    {best_code}
    ```
    
    3. **随机参考 (Random)**: 性能一般的代码。
    ```python
    {random_code}
    ```
    
    【思考逻辑】
    - 对比 Best 和 Random：是什么让 Best 跑得更快？是更大的 Block Size？还是特殊的 PID 映射算法？
    - 将发现的这些“优势特征”应用到 Target 代码上。
    - 类似于向量运算：New_Code = Target + (Best - Random)。
    
    【输出要求】
    仅输出变异后的 Target Python 代码。确保语法正确可运行。
    """
    return _send_request(sys_prompt, user_prompt)


def query_init_generation(baseline_code, strategy_name, strategy_desc):
    sys_prompt = "你是一位精通 Triton 编译器和高性能计算的顶级架构师。你的任务是根据特定的优化策略重写代码。"
    
    user_prompt = f"""
    我有一个基础的矩阵乘法 (MatMul) Triton 内核。请你应用【{strategy_name}】策略对其进行彻底重构或参数调整。
    
    【基础代码】
    ```python
    {baseline_code}
    ```
    
    【优化策略要求: {strategy_name}】
    {strategy_desc}
    
    【输出要求】
    1. 代码必须是完整的、可运行的 Python 代码（包含 kernel, triton_matmul 函数）。
    2. 严格遵守上述策略，不要做无关的改动。
    3. 如果涉及 Block Size，请根据策略大胆调整（例如 16x16 或 128x128）。
    4. 仅输出 Python 代码块。
    """
    return _send_request(sys_prompt, user_prompt)
