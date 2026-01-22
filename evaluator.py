import subprocess
import os
import time
import sys
import tempfile

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
import os

# --- [LLM 生成的代码将被插入在这里] ---
{generated_code}
# ------------------------------------

def test_entry_point():
    try:
        torch.manual_seed(42)
        # torch.set_num_threads(1) # 可选
        
        M, N, K = 512, 512, 512
        a = torch.randn((M, K), device='cpu', dtype=torch.float32)
        b = torch.randn((K, N), device='cpu', dtype=torch.float32)
        
        try:
            c_triton = triton_matmul(a, b)
        except Exception as e:
            print(f"RESULT:FAIL|Runtime Error: {{str(e)}}")
            return

        c_ref = torch.matmul(a, b)
        
        if not torch.allclose(c_triton, c_ref, atol=1e-2, rtol=1e-2):
            max_diff = (c_triton - c_ref).abs().max().item()
            print(f"RESULT:FAIL|Result Mismatch (Max Diff: {{max_diff:.6f}})")
            return

        warmup_steps = 10
        measure_steps = 50
        
        for _ in range(warmup_steps):
            triton_matmul(a, b)
            
        start_time = time.perf_counter()
        for _ in range(measure_steps):
            triton_matmul(a, b)
        end_time = time.perf_counter()
        
        total_time_sec = end_time - start_time
        avg_latency_sec = total_time_sec / measure_steps
        
        print(f"RESULT:PASS|{{avg_latency_sec}}")
        
    except Exception as e:
        print(f"RESULT:FAIL|Unknown Error: {{str(e)}}")

if __name__ == "__main__":
    test_entry_point()
"""

class Evaluator:
    def __init__(self):
        # 创建日志目录
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def _save_debug_log(self, code, stdout, stderr, error_hint):
        """将失败的现场保存到 logs 文件夹"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_{timestamp}_{abs(hash(code)) % 10000}.log"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== ERROR HINT ===\n")
            f.write(f"{error_hint}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout + "\n\n")
            f.write("=== STDERR ===\n")
            f.write(stderr + "\n\n")
            f.write("=== FULL GENERATED CODE ===\n")
            f.write(code + "\n")
        
        print(f"   ⚠️ Debug log saved to: {filepath}")

    def evaluate(self, code_snippet):
        """
        运行代码片段并返回 (success, latency/error_msg, debug_info)
        """
        full_script = TEST_SCAFFOLD.format(generated_code=code_snippet)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
            tmp.write(full_script)
            
        try:
            result = subprocess.run(
                ['python', tmp_path],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            latency = None
            error_msg = ""
            is_success = False

            # 解析逻辑
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith("RESULT:PASS|"):
                    try:
                        latency = float(line.split("|")[1])
                        is_success = True
                        break 
                    except ValueError as ve:
                        error_msg = f"Float convert failed: {ve}"
                        # 这里也要记录日志，因为格式虽然对了但数字解析挂了
                        self._save_debug_log(full_script, stdout, stderr, error_msg)
                        
                elif line.startswith("RESULT:FAIL|"):
                    error_msg = line.split("|")[1]
                    break
            
            # 如果没找到任何 RESULT 标记
            if not is_success and not error_msg:
                if stderr:
                    error_msg = f"Crash/Stderr: {stderr.strip()[:100]}..."
                elif stdout:
                    error_msg = f"Invalid Stdout: {stdout.strip()[:100]}..."
                else:
                    error_msg = "No output produced (Empty)."
                
                # 关键：保存现场！
                self._save_debug_log(full_script, stdout, stderr, "No RESULT tag found")

            return is_success, latency, error_msg

        except subprocess.TimeoutExpired:
            return False, 0.0, "Timeout (120s)"
        except Exception as e:
            # 系统级错误也要记录
            self._save_debug_log(full_script, "", str(e), "System/Python Error")
            return False, 0.0, f"Evaluator System Error: {str(e)}"
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

# --- 单元测试 (Unit Test) ---
# 只有直接运行此文件时才会执行，被 import 时不会执行
if __name__ == "__main__":
    print("Testing Evaluator with dummy code...")
    
    # 模拟一段有语法错误的代码
    dummy_code_fail = "def triton_matmul(a, b): return a + b syntax error"
    
    evaluator = Evaluator()
    success, lat, msg = evaluator.evaluate(dummy_code_fail)
    print(f"Test Fail Case: Success={success}, Msg={msg}")
    
    # 注意：要测试成功案例，你需要一段完整的 triton_matmul 代码
    print("Evaluator module is ready.")
