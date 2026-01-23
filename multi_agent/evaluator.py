import subprocess
import os
import time
import sys
import uuid
from profiler import profile 

# ==========================================
# 测试脚手架模板 (Test Scaffold Template)
# 这个字符串充当一个"容器"，我们将 LLM 生成的内核代码填入其中，
# 然后作为一个完整的 Python 脚本运行。
# ==========================================
TEST_SCAFFOLD = """
import torch
import triton
import triton.language as tl
import time
import sys
import traceback

# --- [LLM 生成的代码将被插入在这里] ---
{generated_code}
# ------------------------------------

def test_entry_point():
    try:
        # 1. 准备数据
        torch.manual_seed(42)
        M, N, K = 512, 512, 512
        a = torch.randn((M, K), device='cpu', dtype=torch.float32)
        b = torch.randn((K, N), device='cpu', dtype=torch.float32)
        
        # 2. 正确性验证
        try:
            c_triton = triton_matmul(a, b)
        except Exception as e:
            print(f"RESULT:FAIL|Runtime Error: {str(e)}")
            return

        c_ref = torch.matmul(a, b)
        if not torch.allclose(c_triton, c_ref, atol=1e-3, rtol=1e-3):
            max_diff = (c_triton - c_ref).abs().max().item()
            print(f"RESULT:FAIL|Mismatch (Max Diff: {max_diff})")
            return

        # 3. 性能测试 (高噪声环境专用版)
        # -------------------------------------------------
        # 策略：Min-Latency (最小值)
        # 假设：只要有一次跑得快，说明代码逻辑是优的。
        # 慢下来的情况全部归咎于 OS 调度干扰。
        # -------------------------------------------------
        
        # A. 预热
        for _ in range(20): # 增加预热，确保 CPU 频率拉起来
            triton_matmul(a, b)
            
        # B. 暴力采样
        latencies = []
        runs = 50  # 增加采样次数，提高捕捉到"Clean Run"的概率
        
        for _ in range(runs):
            t0 = time.perf_counter()
            triton_matmul(a, b)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
        
        # C. 取最小值作为最终成绩
        # 在高波动环境下，Min 是最能反映代码上限的指标
        best_latency = min(latencies)
        
        print(f"RESULT:PASS|{best_latency}")
        
    except Exception as e:
        print(f"RESULT:FAIL|Unknown Error: {str(e)}")

if __name__ == "__main__":
    test_entry_point()
"""

class Evaluator:
    def __init__(self):
        self.timeout = 300 # 秒

    @profile("Evaluator_Run")
    def evaluate(self, code_str):
        """
        评估给定的 Triton 代码字符串。
        
        参数:
            code_str (str): LLM 生成的 Python 代码，包含 kernel 和 wrapper 函数。
            
        返回:
            tuple: (is_success: bool, latency: float, message: str)
        """
        
        unique_id = str(uuid.uuid4())
        temp_filename = f"temp_kernel_{unique_id}.py"
        
        full_script = TEST_SCAFFOLD.replace("{generated_code}", code_str)
        
        with open(temp_filename, "w", encoding="utf-8") as f:
            f.write(full_script)
            
        try:
            result = subprocess.run(
                [sys.executable, temp_filename],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            output = result.stdout.strip()
            stderr = result.stderr.strip()
            
            # 解析输出结果
            if "RESULT:PASS" in output:
                # 成功格式: RESULT:PASS|0.00234
                try:
                    latency = float(output.split("|")[1])
                    return True, latency, "Success"
                except IndexError:
                    return False, float('inf'), "Output parsing error"
                    
            elif "RESULT:FAIL" in output:
                # 失败格式: RESULT:FAIL|Error Message
                try:
                    msg = output.split("|")[1]
                    return False, float('inf'), msg
                except IndexError:
                    return False, float('inf'), "Fail message parsing error"
            
            else:
                error_msg = stderr if stderr else "No output returned (Crash?)"
                return False, float('inf'), f"Execution Error: {error_msg[-300:]}"
                
        except subprocess.TimeoutExpired:
            return False, float('inf'), "Execution Timed Out (Possible infinite loop)"
            
        except Exception as e:
            return False, float('inf'), f"System Error: {str(e)}"
            
        finally:
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass

