import os
import glob
import heapq
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from evaluator import Evaluator
from llm_handler import query_mutation, query_crossover, query_de_mutation

class BaseEvoluter:
    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluator()
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 种群堆: 存储 (latency, unique_id, code_str, source_tag)
        # 使用 Min-Heap，因为 latency 越小越好
        self.population_heap = [] 
        self.history_best = [] # 记录每代最佳 Latency 用于绘图
        self.counter = 0 # 用于生成唯一ID，防止heapq在latency相同时比较code字符串报错

    def load_initial_population(self):
        """从 code/ 文件夹加载所有 .py 文件"""
        files = glob.glob(os.path.join("code", "*.py"))
        print(f"📂 Loading {len(files)} initial codes from 'code/'...")
        
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as f:
                code = f.read()
            self._evaluate_and_push(code, source=os.path.basename(fpath))
            
        # 如果文件不够 pop_size，通过变异补齐
        current_pop = [item[2] for item in self.population_heap]
        while len(self.population_heap) < self.args.pop_size:
            print("⚠️ Initial population too small, supplementing with mutations...")
            parent = random.choice(current_pop)
            new_code = query_mutation(parent, 0.1, "Random Init")
            if new_code:
                self._evaluate_and_push(new_code, source="init_supplement")

    def _evaluate_and_push(self, code, source="unknown"):
        """评估代码并推入堆中"""
        success, latency, msg = self.evaluator.evaluate(code)
        if success:
            # heapq 放入元组 (latency, counter, code, source)
            # counter 确保即使 latency 相同也能区分，避免比较 code 字符串
            heapq.heappush(self.population_heap, (latency, self.counter, code, source))
            self.counter += 1
            print(f"   ✅ [PASS] {latency*1000:.2f}ms | Src: {source}")
        else:
            print(f"   ❌ [FAIL] {msg[:50]}... | Src: {source}")

    def get_top_k(self, k):
        """获取当前堆中最好的 k 个个体"""
        return heapq.nsmallest(k, self.population_heap)

    def visualize(self):
        """绘制进化曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history_best, marker='o', linestyle='-', color='b')
        plt.title(f"Evolution Progress ({self.args.mode})")
        plt.xlabel("Generation")
        plt.ylabel("Latency (s)")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "evolution_curve.png"))
        print(f"📊 Visualization saved to {self.output_dir}/evolution_curve.png")

    def save_best(self, gen):
        """保存当前最佳代码"""
        if not self.population_heap: return
        best = self.population_heap[0] # Heap 根节点就是最小值
        with open(os.path.join(self.output_dir, f"best_gen_{gen}.py"), "w") as f:
            f.write(best[2])

    def run(self):
        raise NotImplementedError

# --- 策略 1: ParaEvoluter (改写/爬山) ---
class ParaEvoluter(BaseEvoluter):
    def run(self):
        self.load_initial_population()
        
        for gen in range(self.args.budget):
            print(f"\n🔄 === Generation {gen+1} (Para/Hill-Climbing) ===")
            
            # 1. 精英选择: 选出 Top K
            elites = self.get_top_k(self.args.pop_size) # 保持种群大小
            
            # 2. 对每个精英进行变异 (Paraphrasing/Mutation)
            # 为了防止种群退化，我们保留精英，生成的孩子加入竞争
            # 这里简单处理：清空堆，重新评估精英+孩子 (或者只保留最好的 N 个)
            # 为简化逻辑：我们每次生成新的一批，然后全部 push 进 heap，最后截断
            
            new_candidates = []
            best_latency = elites[0][0]
            self.history_best.append(best_latency)
            print(f"🏆 Gen Best: {best_latency*1000:.4f} ms")

            for item in elites:
                latency, _, code, src = item
                # 生成新代码
                new_code = query_mutation(code, latency, "Optimize tiling and vectorization")
                if new_code:
                    new_candidates.append(new_code)
            
            # 3. 评估新候选者
            print(f"🧬 Evaluating {len(new_candidates)} offspring...")
            for code in new_candidates:
                self._evaluate_and_push(code, source=f"gen{gen}_para")
            
            # 4. 优胜劣汰 (截断堆，只保留最好的 pop_size 个)
            # heapq.nsmallest 返回列表，我们需要重新构建堆
            best_individuals = heapq.nsmallest(self.args.pop_size, self.population_heap)
            self.population_heap = [] # 清空
            for item in best_individuals:
                heapq.heappush(self.population_heap, item) # 重新入堆

            self.save_best(gen)
        self.visualize()

# --- 策略 2: GAEvoluter (遗传算法 - 杂交) ---
class GAEvoluter(BaseEvoluter):
    def run(self):
        self.load_initial_population()
        
        for gen in range(self.args.budget):
            print(f"\n🧬 === Generation {gen+1} (Genetic Algorithm) ===")
            
            current_pop = self.get_top_k(len(self.population_heap))
            best_latency = current_pop[0][0]
            self.history_best.append(best_latency)
            print(f"🏆 Gen Best: {best_latency*1000:.4f} ms")
            
            new_offsprings = []
            
            # 生成 pop_size 个孩子
            for _ in range(self.args.pop_size):
                # 1. 锦标赛选择 (Tournament Selection)
                # 随机选 3 个，取最好的作为父代
                pool_mom = random.sample(current_pop, min(3, len(current_pop)))
                pool_dad = random.sample(current_pop, min(3, len(current_pop)))
                mom = min(pool_mom, key=lambda x: x[0])
                dad = min(pool_dad, key=lambda x: x[0])
                
                # 2. 杂交 (Crossover)
                print(f"   💕 Crossover: {mom[3]} + {dad[3]}")
                child_code = query_crossover(mom[2], dad[2])
                
                # 3. 变异 (Mutation - 小概率)
                if random.random() < 0.2 and child_code:
                    print("   🧪 Mutation triggered...")
                    child_code = query_mutation(child_code, 0, "Small tweak")
                
                if child_code:
                    new_offsprings.append(child_code)

            # 评估孩子
            for code in new_offsprings:
                self._evaluate_and_push(code, source=f"gen{gen}_GA")
            
            # 种群更新：保留最好的 pop_size
            best_individuals = heapq.nsmallest(self.args.pop_size, self.population_heap)
            self.population_heap = []
            for item in best_individuals:
                heapq.heappush(self.population_heap, item)
                
            self.save_best(gen)
        self.visualize()

# --- 策略 3: DEEvoluter (差分进化) ---
class DEEvoluter(BaseEvoluter):
    def run(self):
        self.load_initial_population()
        
        for gen in range(self.args.budget):
            print(f"\n🚀 === Generation {gen+1} (Differential Evolution) ===")
            
            # 获取当前所有个体
            current_pop = self.get_top_k(len(self.population_heap))
            best_global = current_pop[0] # 堆顶即最小值（最优）
            self.history_best.append(best_global[0])
            print(f"🏆 Gen Best: {best_global[0]*1000:.4f} ms")
            
            next_generation_candidates = []
            
            # --- 核心循环 ---
            for i in range(len(current_pop)):
                target = current_pop[i]
                
                # 策略: 强制保留精英 (Elitism)
                # 如果当前个体是最好的，直接进入下一代，确保最优解不丢失
                if target == best_global:
                    next_generation_candidates.append(target)
                    continue

                # 选择 Random 个体 (不能是 target 自身)
                remaining_pool = [p for p in current_pop if p != target]
                if not remaining_pool:
                    remaining_pool = [target] # 防止极端情况
                random_sample = random.choice(remaining_pool)
                
                print(f"   ⚡ DE Op: Target({target[3]}) <- Best({best_global[3]}) - Random({random_sample[3]})")
                
                # LLM 模拟语义差分: V = Target + F(Best - Random)
                trial_code = query_de_mutation(target[2], best_global[2], random_sample[2])
                
                latency = None # 初始化
                success = False

                if trial_code:
                    success, latency, _ = self.evaluator.evaluate(trial_code)
                
                # --- 贪婪选择 (Greedy Selection) ---
                # DE 的核心：只有当孩子比父亲好，才替换父亲
                if success and latency < target[0]:
                    print(f"      ✅ Improved! {latency*1000:.2f}ms < {target[0]*1000:.2f}ms")
                    next_generation_candidates.append((latency, self.counter, trial_code, f"gen{gen}_DE"))
                    self.counter += 1
                else:
                    # 否则，保留原有的 Target
                    print(f"      ❌ No gain (Keep Target).")
                    next_generation_candidates.append(target)
            
            # --- 种群更新 (修复版) ---
            # 1. 清空旧堆
            self.population_heap = []
            
            # 2. 将下一代推入堆
            for item in next_generation_candidates:
                heapq.heappush(self.population_heap, item)
            
            # 3. 确保堆大小不超过 pop_size (使用 nsmallest 逻辑)
            # 虽然标准的 DE 种群大小不变，但为了防止意外膨胀，我们可以做一次截断
            if len(self.population_heap) > self.args.pop_size:
                # nsmallest 返回最小的 k 个元素（即 Latency 最低的）
                best_k = heapq.nsmallest(self.args.pop_size, self.population_heap)
                self.population_heap = []
                for item in best_k:
                    heapq.heappush(self.population_heap, item)
            
            self.save_best(gen)
        
        self.visualize()