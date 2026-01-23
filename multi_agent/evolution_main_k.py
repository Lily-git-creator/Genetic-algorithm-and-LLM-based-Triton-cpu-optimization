# === file: evolution_main.py ===
import os
import argparse
import random
import concurrent.futures
import time
import triton
import triton.language as tl
import torch
import json 
import uuid   
from evaluator import Evaluator
from llm_handler import query_mutation, query_crossover
from profiler import print_stats


def load_baseline_code(baseline_file: str | None) -> str:
    """
    从指定文件读取 baseline code。
    - baseline_file 为 None：返回默认 baseline
    - 文件不存在/读取失败：抛异常（也可改成返回默认 baseline）
    """
    if not baseline_file:
        return DEFAULT_BASELINE_CODE

    if not os.path.isfile(baseline_file):
        raise FileNotFoundError(f"Baseline file not found: {baseline_file}")

    with open(baseline_file, "r", encoding="utf-8") as f:
        code = f.read()

    if not code.strip():
        raise ValueError(f"Baseline file is empty: {baseline_file}")

    return code

def calculate_speedup(t_baseline, t_current):
    if t_current <= 0: return 0.0
    # 公式：max(T_base / T_curr - 1, 0)
    ratio = (t_baseline / t_current) - 1.0
    return max(ratio, 0.0)

class PopulationManager:
    def __init__(self, pop_size=4):
        self.pop_size = pop_size
        self.population = [] 
        self.evaluator = Evaluator()
        # 新增：进化历史记录，用于画图
        self.genealogy_log = [] 

    def log_individual(self, ind_id, parent_ids, gen, latency, method):
        """记录个体的血缘关系"""
        self.genealogy_log.append({
            "id": ind_id,
            "parents": parent_ids, # List of parent IDs
            "generation": gen,
            "latency": latency,
            "method": method
        })

    def save_log(self):
        """保存历史记录到 JSON"""
        with open("evolution_k_history.json", "w") as f:
            json.dump(self.genealogy_log, f, indent=2)

    def add_individual(self, code, source_info, generation, parent_ids=None):
        """
        评估并尝试添加个体到种群
        注意：这里增加了 generation 和 parent_ids 参数用于画图
        """
        # 简单去重：如果代码完全一样，跳过
        for ind in self.population:
            if ind['code'].strip() == code.strip():
                return None

        success, latency, msg = self.evaluator.evaluate(code)
        
        # 生成唯一 ID (截取前8位)
        ind_id = str(uuid.uuid4())[:8]

        if success:
            # 记录到日志
            self.log_individual(ind_id, parent_ids if parent_ids else [], generation, latency, source_info)
            
            return {
                'id': ind_id,
                'code': code, 
                'latency': latency, 
                'source': source_info
            }
        else:
            # 失败的也可以记录一下（可选）， latency = -1
            self.log_individual(ind_id, parent_ids, generation, -1.0, source_info + "_FAIL")
            return None


class TritonEvoluter:
    def __init__(self, args):
        self.budget = args.budget
        self.pop_size = args.pop_size
        self.manager = PopulationManager(args.pop_size)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.baseline_code = load_baseline_code(args.baseline_file)
        self.baseline_latency = 0.0 # 存储 baseline 耗时

        # 早停参数
        self.patience = 3       
        self.min_delta = 0.01   

    def calibrate_baseline(self, retries=5):
        print(f"⚖️ Calibrating Baseline ({retries} runs)...")
        latencies = []
        
        for i in range(retries):
            # 这里的 evaluate 内部已经跑了 50 次取 min 了
            success, lat, msg = self.manager.evaluator.evaluate(self.baseline_code)
            if success:
                latencies.append(lat)
                print(f"   Run {i+1}: {lat*1000:.3f} ms")
            else:
                print(f"   Run {i+1}: Failed ({msg})")
        
        if not latencies:
            return False, 0.0
        
        # 这里我们也取 min，坚持"最快原则"
        # 只要这 5 次大循环里（总共 5 * 50 = 250 次微循环），有一次极速，我们就认。
        best_of_best = min(latencies) 
        print(f"🎯 Baseline Calibrated: {best_of_best*1000:.3f} ms (Best of {retries}x50 runs)")
        return True, best_of_best

    def init_population(self):
        print("🚀 Initializing Population...")

        is_ok, base_latency = self.calibrate_baseline()
        if not is_ok:
            print("❌ Critical: Baseline failed to run completely.")
            return
            
        self.baseline_latency = base_latency
        base_ind = {
            'id': str(uuid.uuid4())[:8],
            'code': self.baseline_code,
            'latency': base_latency,
            'source': 'baseline'
        }
        
        self.manager.log_individual(base_ind['id'], [], 0, base_latency, "baseline")
        self.manager.population = [base_ind]
        print(f"   -> Baseline added to population.")
        
        # 2. 补全种群
        print(f"🌱 Bootstrapping population to size {self.pop_size}...")
        futures = []
        for i in range(self.pop_size - 1):
            futures.append(
                (self.executor.submit(query_mutation, base_ind['code'], base_ind['latency'], "tiling_expert"), base_ind['id'])
            )
            
        for future, parent_id in futures:
            try:
                code = future.result()
                if code:
                    # 传入 generation=0, parent_ids=[base_id]
                    ind = self.manager.add_individual(code, "init_mutation", 0, [parent_id])
                    if ind:
                        self.manager.population.append(ind)
                        print(f"   -> Added init individual: {ind['latency']*1000:.3f} ms")
            except Exception as e:
                print(f"   -> Init error: {e}")
        
        self.manager.save_log()
        print(f"✅ Population initialized. Count: {len(self.manager.population)}")

    def run(self):
        global_start_time = time.time()
        self.init_population()
        
        best_global_latency = min(p['latency'] for p in self.manager.population)
        no_improve_counter = 0
        
        # 设定 K 值，即每一代选取的精英数量
        K = max(2, self.pop_size // 2) 

        for gen in range(1, self.budget + 1):
            if time.time() - global_start_time > 1200:
                print("\n Time Limit Reached (20 min). Stopping ...")
                break

            # 1. 对种群按耗时排序，选出前 K 个精英
            sorted_pop = sorted(self.manager.population, key=lambda x: x['latency'])
            elites = sorted_pop[:K] 
            best_curr = elites[0]
            
            print(f"\n🔄 === Gen {gen}/{self.budget} | Best: {best_curr['latency']*1000:.4f} ms | Source: {best_curr['source']} ===")
            
            # 早停检查 (保持原逻辑)
            improvement = (best_global_latency - best_curr['latency']) / best_global_latency
            if improvement > self.min_delta:
                best_global_latency = best_curr['latency']
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                
            if no_improve_counter >= self.patience:
                print(f"\n🛑 Early stopping triggered!")
                break

            future_to_meta = {}
            
            # 2. 策略修改：确保前 K 个精英都参与杂交和变异
            # 这里的逻辑是：对每个精英，都至少进行一次变异和一次与其他精英的杂交
            for i, p_elite in enumerate(elites):
                # --- 强制变异：确保该精英的基因被扰动 ---
                role = random.choice(["tiling_expert", "vector_expert"])
                f_mut = self.executor.submit(query_mutation, p_elite['code'], p_elite['latency'], role)
                future_to_meta[f_mut] = {
                    "type": f"mut_{role}", 
                    "parents": [p_elite['id']],
                    "parent_latency": p_elite['latency']
                }

                # --- 强制杂交：与另一个随机精英结合 ---
                # 选取除了自己以外的一个精英
                other_elites = [p for j, p in enumerate(elites) if i != j]
                if other_elites:
                    p2 = random.choice(other_elites)
                    f_cross = self.executor.submit(query_crossover, p_elite['code'], p_elite['latency'], p2['code'], p2['latency'])
                    future_to_meta[f_cross] = {
                        "type": "crossover", 
                        "parents": [p_elite['id'], p2['id']],
                        "parent_latency": min(p_elite['latency'], p2['latency'])
                    }

            # 3. 处理结果 (保持原逻辑)
            valid_offsprings = []
            for future in concurrent.futures.as_completed(future_to_meta):
                meta = future_to_meta[future]
                try:
                    generated_code = future.result()
                    if not generated_code: continue
                    ind = self.manager.add_individual(generated_code, f"gen{gen}_{meta['type']}", gen, meta['parents'])
                    if ind:
                        valid_offsprings.append(ind)
                except Exception as e:
                    print(f"      ⚠️ Error: {e}")

            # 4. 更新种群并去重 (保持原逻辑)
            combined = self.manager.population + valid_offsprings
            combined = sorted(combined, key=lambda x: x['latency'])
            unique_pop = []
            seen_code = set()
            for p in combined:
                if p['code'] not in seen_code:
                    unique_pop.append(p)
                    seen_code.add(p['code'])
            self.manager.population = unique_pop[:self.pop_size]
            self.manager.save_log()
            print_stats()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--pop_size", type=int, default=4)
    parser.add_argument("--baseline_file", type=str, default="/home/PB23111695/multi_agent/triton-cpu/matmul.py")
    args = parser.parse_args()
    
    evolver = TritonEvoluter(args)
    evolver.run()
