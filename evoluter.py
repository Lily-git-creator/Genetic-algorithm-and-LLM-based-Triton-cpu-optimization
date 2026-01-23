

import os
import glob
import heapq
import time
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluator import Evaluator
from llm_handler import query_mutation, query_crossover, query_de_mutation

class BaseEvoluter:
    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluator()
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.population_heap = [] 
        self.history_best = [] 
        self.counter = 0 
        # 设置最大并发数 (建议设为 2 或 4 以避免 API 超时)
        self.max_workers = getattr(args, 'max_workers', 2)
        
        # 🔥 新增：时间限制 (默认 20 分钟 = 1200 秒)
        self.time_limit = getattr(args, 'time_limit', 1200) 
        self.metrics = {
            "mode": args.mode,
            "baseline_latency": None,
            "generations": []
        }
        self.metrics_file = os.path.join(self.output_dir, f"{args.mode}_metrics.json")

    def _evaluate_single_worker(self, code, source):
        """Worker 函数"""
        success, latency, msg = self.evaluator.evaluate(code)
        return success, latency, msg, code, source

    def load_initial_population(self):
        files = glob.glob(os.path.join("code", "*.py"))
        print(f"📂 Loading {len(files)} initial codes from 'code/'...")
        
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for fpath in files:
                with open(fpath, 'r', encoding='utf-8') as f:
                    code = f.read()
                futures.append(executor.submit(self._evaluate_single_worker, code, os.path.basename(fpath)))
            
            for future in as_completed(futures):
                success, latency, msg, code, source = future.result()
                self._handle_eval_result(success, latency, msg, code, source)
                if success and "baseline" in source:
                    self.metrics["baseline_latency"] = latency
                    print(f"   🎯 Baseline Latency identified: {latency*1000:.4f} ms")

            # 如果没找到名为 baseline 的文件，暂时用初始种群最慢的作为基准（保守估计）
        if self.metrics["baseline_latency"] is None and self.population_heap:
            self.metrics["baseline_latency"] = max(p[0] for p in self.population_heap)

        current_pop = [item[2] for item in self.population_heap]
        while len(self.population_heap) < self.args.pop_size and len(current_pop) > 0:
            print("⚠️ Initial population too small, supplementing with mutations (Parallel)...")
            needed = self.args.pop_size - len(self.population_heap)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for _ in range(needed):
                    parent = random.choice(current_pop)
                    futures.append(executor.submit(self._generate_and_eval_init, parent))
                
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        self._handle_eval_result(*res)

    def _generate_and_eval_init(self, parent_code):
        new_code = query_mutation(parent_code, 0.1, "Random Init")
        if new_code:
            return self._evaluate_single_worker(new_code, "init_supplement")
        return None

    def _handle_eval_result(self, success, latency, msg, code, source):
        if success:
            heapq.heappush(self.population_heap, (latency, self.counter, code, source))
            self.counter += 1
            print(f"   ✅ [PASS] {latency*1000:.2f}ms | Src: {source}")
        else:
            print(f"   ❌ [FAIL] {msg[:50]}... | Src: {source}")

    def get_top_k(self, k):
        return heapq.nsmallest(k, self.population_heap)

    def save_metrics(self):
        """🔥 将当前指标保存到 JSON"""
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def record_generation(self, gen_idx, best_latency, gen_time, total_time):
        """🔥 记录每一代的数据"""
        data = {
            "generation": gen_idx,
            "best_latency": best_latency,
            "gen_duration": gen_time,
            "total_elapsed": total_time,
            "speedup": max(self.metrics["baseline_latency"] / best_latency - 1, 0) if self.metrics["baseline_latency"] else 0
        }
        self.metrics["generations"].append(data)
        self.save_metrics()

    def visualize(self):
        if not self.history_best: return
        plt.figure(figsize=(10, 6))
        plt.plot(self.history_best, marker='o', linestyle='-', color='b')
        plt.title(f"Evolution Progress ({self.args.mode})")
        plt.xlabel("Generation")
        plt.ylabel("Latency (s)")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "evolution_curve.png"))
        print(f"📊 Visualization saved to {self.output_dir}/evolution_curve.png")

    def save_best(self, gen):
        if not self.population_heap: return
        best = self.population_heap[0]
        with open(os.path.join(self.output_dir, f"best_gen_{gen}.py"), "w") as f:
            f.write(best[2])

    def run(self):
        raise NotImplementedError

# --- 策略 1: ParaEvoluter (并行版) ---
class ParaEvoluter(BaseEvoluter):
    def _process_elite(self, item, gen):
        latency, _, code, src = item
        new_code = query_mutation(code, latency, "Optimize tiling and vectorization for CPU")
        if new_code:
            return self._evaluate_single_worker(new_code, f"gen{gen}_para")
        return None

    def run(self):
        self.load_initial_population()
        
        # 🔥 记录总开始时间
        total_start_time = time.time()
        
        for gen in range(self.args.budget):
            # 🔥 早停检查
            elapsed_total = time.time() - total_start_time
            if elapsed_total > self.time_limit:
                print(f"\n🛑 [EARLY STOP] Total time {elapsed_total:.2f}s exceeded limit {self.time_limit}s.")
                break

            # 🔥 记录本代开始时间
            gen_start_time = time.time()
            print(f"\n🔄 === Generation {gen+1} (Para) [Elapsed: {elapsed_total/60:.1f}m] ===")
            
            elites = self.get_top_k(self.args.pop_size)
            if not elites: break
            best_latency = elites[0][0]
            self.history_best.append(best_latency)
            print(f"🏆 Gen Best: {best_latency*1000:.4f} ms")

            print(f"🧬 Processing {len(elites)} elites in parallel...")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._process_elite, item, gen) for item in elites]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        self._handle_eval_result(*result)
            
            best_individuals = heapq.nsmallest(self.args.pop_size, self.population_heap)
            self.population_heap = []
            for item in best_individuals:
                heapq.heappush(self.population_heap, item)

            self.save_best(gen)
            
            # 🔥 记录本代耗时
            gen_duration = time.time() - gen_start_time
            self.record_generation(gen+1, self.population_heap[0][0], gen_duration, elapsed_total + gen_duration)
            
        self.visualize()

# --- 策略 2: GAEvoluter (并行版) ---
class GAEvoluter(BaseEvoluter):
    def _process_offspring(self, current_pop, gen):
        pool_mom = random.sample(current_pop, min(3, len(current_pop)))
        pool_dad = random.sample(current_pop, min(3, len(current_pop)))
        mom = min(pool_mom, key=lambda x: x[0])
        dad = min(pool_dad, key=lambda x: x[0])
        
        child_code = query_crossover(mom[2], dad[2])
        if random.random() < 0.2 and child_code:
            child_code = query_mutation(child_code, 0, "Small tweak")
        
        if child_code:
            return self._evaluate_single_worker(child_code, f"gen{gen}_GA")
        return None

    def run(self):
        self.load_initial_population()
        
        # 🔥 记录总开始时间
        total_start_time = time.time()
        
        for gen in range(self.args.budget):
            # 🔥 早停检查
            elapsed_total = time.time() - total_start_time
            if elapsed_total > self.time_limit:
                print(f"\n🛑 [EARLY STOP] Total time {elapsed_total:.2f}s exceeded limit {self.time_limit}s.")
                break

            # 🔥 记录本代开始时间
            gen_start_time = time.time()
            print(f"\n🧬 === Generation {gen+1} (GA) [Elapsed: {elapsed_total/60:.1f}m] ===")
            
            current_pop = self.get_top_k(len(self.population_heap))
            if not current_pop: break
            best_latency = current_pop[0][0]
            self.history_best.append(best_latency)
            print(f"🏆 Gen Best: {best_latency*1000:.4f} ms")
            
            print(f"   💕 Generating {self.args.pop_size} offsprings...")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._process_offspring, current_pop, gen) 
                           for _ in range(self.args.pop_size)]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        self._handle_eval_result(*result)
            
            best_individuals = heapq.nsmallest(self.args.pop_size, self.population_heap)
            self.population_heap = []
            for item in best_individuals:
                heapq.heappush(self.population_heap, item)
                
            self.save_best(gen)
            
            # 🔥 记录本代耗时
            gen_duration = time.time() - gen_start_time
            self.record_generation(gen+1, self.population_heap[0][0], gen_duration, elapsed_total + gen_duration)
            
        self.visualize()

# --- 策略 3: DEEvoluter (并行版) ---
class DEEvoluter(BaseEvoluter):
    def _process_de_individual(self, idx, target, best_global, current_pop, gen):
        if target == best_global:
            return idx, None, True

        remaining_pool = [p for p in current_pop if p != target]
        if not remaining_pool: remaining_pool = [target]
        random_sample = random.choice(remaining_pool)
        
        trial_code = query_de_mutation(target[2], best_global[2], random_sample[2])
        
        if trial_code:
            success, latency, msg = self.evaluator.evaluate(trial_code)
            return idx, (success, latency, msg, trial_code, f"gen{gen}_DE"), False
        
        return idx, None, False

    def run(self):
        self.load_initial_population()
        
        # 🔥 记录总开始时间
        total_start_time = time.time()
        
        for gen in range(self.args.budget):
            # 🔥 早停检查
            elapsed_total = time.time() - total_start_time
            if elapsed_total > self.time_limit:
                print(f"\n🛑 [EARLY STOP] Total time {elapsed_total:.2f}s exceeded limit {self.time_limit}s.")
                break
                
            # 🔥 记录本代开始时间
            gen_start_time = time.time()
            print(f"\n🚀 === Generation {gen+1} (DE) [Elapsed: {elapsed_total/60:.1f}m] ===")
            
            current_pop = self.get_top_k(len(self.population_heap))
            if not current_pop: break
            best_global = current_pop[0]
            self.history_best.append(best_global[0])
            print(f"🏆 Gen Best: {best_global[0]*1000:.4f} ms")
            
            next_generation_candidates = [None] * len(current_pop)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for i in range(len(current_pop)):
                    future = executor.submit(
                        self._process_de_individual, 
                        i, current_pop[i], best_global, current_pop, gen
                    )
                    futures[future] = i
                
                print(f"   ⚡ Running DE ops for {len(current_pop)} individuals...")
                
                for future in as_completed(futures):
                    idx, eval_res, is_elite = future.result()
                    target = current_pop[idx]
                    
                    if is_elite:
                        next_generation_candidates[idx] = target
                        continue
                    
                    improved = False
                    if eval_res:
                        success, latency, msg, code, src = eval_res
                        if success and latency < target[0]:
                            next_generation_candidates[idx] = (latency, self.counter, code, src)
                            self.counter += 1
                            print(f"      ✅ Idx {idx} Improved! {latency*1000:.2f}ms < {target[0]*1000:.2f}ms")
                            improved = True
                    
                    if not improved:
                        next_generation_candidates[idx] = target
            
            self.population_heap = []
            for item in next_generation_candidates:
                if item:
                    heapq.heappush(self.population_heap, item)
            
            if len(self.population_heap) > self.args.pop_size:
                best_k = heapq.nsmallest(self.args.pop_size, self.population_heap)
                self.population_heap = []
                for item in best_k:
                    heapq.heappush(self.population_heap, item)
            
            self.save_best(gen)
            
            # 🔥 记录本代耗时
            gen_duration = time.time() - gen_start_time
            self.record_generation(gen+1, self.population_heap[0][0], gen_duration, elapsed_total + gen_duration)
        self.visualize()
