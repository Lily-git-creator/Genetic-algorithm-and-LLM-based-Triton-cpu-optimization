# === file: llm_handler.py ===

import os
import requests
import json
from profiler import profile 

# 配置 API
API_KEY = os.getenv("LLM_API_KEY", "Change it to Your Key") 
API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-chat" 

# ==========================================
# 角色与 Prompt 定义
# ==========================================
COMMON_CONSTRAINTS = """
### ⚠️ 严格约束（必须遵守）：
1. **禁止修改函数名字**：`triton_matmul` 函数的输入参数、名称和返回值类型必须保持完全一致。
2. **禁止修改代码逻辑**：输入输出的数学逻辑必须保持不变，确保计算结果正确。
3. **仅输出代码**：不要解释，不要markdown废话，直接输出包含 `import`、`kernel` 和 `wrapper` 的完整 Python 代码。
"""

ROLES = {
    "tiling_expert": f"""
你是一位专攻 CPU 存储层次结构（Memory Hierarchy）的 Triton 优化专家。
你的目标是：**通过调整分块（Tiling）策略，使数据尽可能驻留在 L1/L2 Cache 中。**
### 优化策略：
1. **分块调整**：CPU L2 Cache 小。请大胆调整 `BLOCK_SIZE_M/N/K`。尝试 "Tall & Skinny" (如 128x32) 或 "Short & Fat" (如 32x128)。CPU 上 `BLOCK_SIZE_K` 通常较小 (16, 32)。
2. **循环顺序**：考虑重排循环或使用 `swizzle`。
{COMMON_CONSTRAINTS}
""",

    "vector_expert": f"""
你是一位精通 CPU SIMD 指令集（AVX-512/NEON）的专家。
你的目标是：**消除内存访问瓶颈，生成高效向量指令。**
### 优化策略：
1. **连续内存访问**：确保 `tl.load/store` 地址连续，Stride=1。
2. **减少 Mask**：CPU 上 mask 性能差。尝试使用 padding (填充) 技术将张量补齐到 16 或 32 的倍数，从而移除 `mask` 参数。
{COMMON_CONSTRAINTS}
""",

    "reviewer": f"""
你是一位资深的 Triton 系统架构师。你面前摆着两份代码（Code A 和 Code B）。
你的目标是：**提取两份代码各自最强的“基因”，融合出一个性能更强的新后代。**
### 决策逻辑：
1. 结合 A 的分块参数和 B 的内存访问模式。
2. 确保融合后的代码变量定义一致，无语法错误。
{COMMON_CONSTRAINTS}
"""
}

def extract_python_code(text):
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()

@profile("LLM_API_Call")
def _call_deepseek(system_prompt, user_prompt):
    """发送请求到 LLM，不依赖任何 context 变量"""
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
        "temperature": 0.8,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        return extract_python_code(content)
        
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        return None

def query_mutation(code, latency, role_key="tiling_expert"):
    sys_prompt = ROLES.get(role_key, ROLES["tiling_expert"])
    user_prompt = f"""
    Current Latency: {latency*1000:.3f} ms.
    Code:
    ```python
    {code}
    ```
    Please optimize this code.
    """
    return _call_deepseek(sys_prompt, user_prompt)

def query_crossover(code_a, lat_a, code_b, lat_b):
    sys_prompt = ROLES["reviewer"]
    user_prompt = f"""
    Code A (Lat: {lat_a*1000:.3f} ms):
    ```python
    {code_a}
    ```
    Code B (Lat: {lat_b*1000:.3f} ms):
    ```python
    {code_b}
    ```
    Combine them into a faster version.
    """
    return _call_deepseek(sys_prompt, user_prompt)
