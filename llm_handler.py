import os
import requests
import json
import re
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置
API_KEY = os.getenv("LLM_API_KEY", "sk-a02aefce65eb48b6a6b65c9b5fed07c3") # 替换你的Key
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
        "temperature": 0.3, # 建议保持较低温度
        "stream": False
    }

    # --- 🔥 新增：配置重试策略 ---
    retry_strategy = Retry(
        total=3,                # 最大重试次数
        backoff_factor=1,       # 重试间隔 (1s, 2s, 4s...)
        status_forcelist=[429, 500, 502, 503, 504], # 针对这些状态码重试
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    # ---------------------------

    try:
        # 使用 http.post 而不是 requests.post
        response = http.post(API_URL, headers=headers, json=payload, timeout=60) # timeout设为60秒足够了
        response.raise_for_status()
        return extract_python_code(response.json()['choices'][0]['message']['content'])
        
    except requests.exceptions.RetryError:
        print(f"❌ LLM Max Retries Exceeded.")
        return None
    except requests.exceptions.Timeout:
        print(f"❌ LLM Request Timed Out.")
        return None
    except Exception as e:
        print(f"❌ LLM Request Failed: {e}")
        return None
    finally:
        http.close()

# --- 1. 基础变异 / 改写 (Para / Mutation) ---
def query_mutation(code, latency, feedback=""):
    sys_prompt = "你是一位精通 CPU 向量化优化和编译原理的专家。你的目标是优化 Triton 代码以在 CPU 上高效运行。"
    
    # 构建优化建议列表
    hints = """
    【Triton CPU 优化指南】:
    1. **Block Size (关键)**: CPU 的 L1/L2 缓存比 GPU 小得多。
       - 尝试较小的块大小，例如 16x16, 32x32, 16x64。
       - 避免过大的块（如 128x128），这会导致 Cache Thrashing。
    2. **简化逻辑**: CPU 讨厌复杂的掩码计算和非连续访存。
       - 尽量保持内存访问连续 (Contiguous access)。
       - 移除复杂的 swizzle 逻辑，除非你确定它能利用 CPU 的 L2 Cache。
    3. **移除 GPU 特性**: 
       - 在 CPU 上，`num_warps` 和 `num_stages` 通常应保持默认或较小值，不要盲目增加。
    4. **向量化 (AVX/AMX)**: 确保维度是 8 或 16 的倍数，以便编译器生成高效的 SIMD 指令。
    """
    
    user_prompt = f"""
    请对以下 Triton 代码进行【微调】以降低在 CPU 上的延迟。
    
    【当前性能】
    - 延迟: {latency*1000:.4f} ms
    - 反馈: {feedback}
    
    {hints}

    【任务要求】
    1. **稳健优化**: 不要重写整个逻辑，优先调整 BLOCK_SIZE_M/N/K 参数。
    2. **必须在 CPU 运行**: 不要使用任何 CUDA 特定 API。
    3. 仅输出 Python 代码（包含 triton kernel, 和 triton_matmul）。
    
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
    5. 不要使用CUDA特定的API，代码必须在CPU上运行。
    6. 只输出matmul_kernel部分和triton_matmul部分，不要输出测试代码

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
    尽可能压缩延迟，得到的输出延迟应该比我给你的高
    不要使用CUDA特定的API，代码必须在CPU上运行。
    只输出matmul_kernel部分和triton_matmul部分，不要输出测试代码
    """
    return _send_request(sys_prompt, user_prompt)



