# Genetic-algorithm-and-LLM-based-Triton-cpu-optimization
## Background
深度学习算子的性能表现已成为制约系统整体效率的核心瓶颈之一。Triton 作为高性能算子开发框架，为开发者提供了简洁的抽象接口，大幅降低了高性能算子的实现门槛，但在 CPU 后端的优化仍面临诸多挑战，亟需高效的自动优化方案。

当前主流的优化思路中，仅依赖大语言模型（LLM）存在明显局限：尽管 LLM 具备强大的代码生成与优化能力，但优化过程效率低下、资源消耗过大，难以在庞大的算子优化搜索空间中快速定位最优解。而遗传算法等进化算法虽能引导搜索过程高效收敛，但缺乏语义理解能力，易产生语义错误。

为解决上述问题，本项目融合进化算法的高效搜索能力与 LLM 的语义优化优势，同时引入多Agent协作机制，构建了一套 Triton 算子自动优化库。库中集成了多种进化算法与协作策略，实现了“搜索引导 + 语义优化 + 协作增强”的全流程自动优化，旨在为 CPU 后端的 Triton 算子提供高效、可靠的性能优化方案。

## Overview
### 1. 整体架构
```
triton-operator-optimization/
├── README.md                  # 项目说明文档
├── requirements.txt           # 依赖包清单
├── main.py                    # 优化入口（支持算法选择、参数配置、执行优化）
├── config/                    # 配置文件目录
│   ├── evolution_config.py    # 进化算法参数配置（种群规模、迭代次数、概率等）
│   ├── agent_config.py        # 多Agent参数配置（角色定义、提示词模板等）
│   └── evaluator_config.py    # 评估器配置（延迟测试次数、硬件监控指标等）
├── evolution_algorithms/      # 进化算法核心模块（多种算法实现）
│   ├── __init__.py
│   ├── para_evoluter.py       # 爬山算法（ParaEvoluter）
│   ├── genetic_algorithm.py   # 遗传算法（GA）
│   └── differential_evolution.py  # 差分进化算法（DE）
├── multi_agent/               # 多Agent协作模块
│   ├── __init__.py
│   ├── role_agents.py         # 角色Agent定义（Tiling优化、Vector优化等）
│   ├── maco_strategy.py       # 多Agent协作策略（MACO：变异、杂交、约束控制）
│   └── meta_agent.py          # Meta-Agent（硬件监控反馈、动态优化建议）
├── llm_handler/               # LLM交互模块
│   ├── __init__.py
│   ├── prompt_templates.py    # 提示词模板（通用约束+角色特定提示）
│   └── llm_client.py          # LLM调用客户端（支持API并发、异步请求）
├── evaluator/                 # 评估器模块
│   ├── __init__.py
│   ├── latency_evaluator.py   # 延迟评估（多次测试取稳定值）
│   ├── hardware_monitor.py    # 硬件监控（Linux Perf/FlameGraph采集Cache Miss等）
│   └── validity_checker.py    # 代码有效性校验（接口/逻辑不修改、语法正确）
├── utils/                     # 工具函数目录
│   ├── __init__.py
│   ├── data_process.py        # 结果保存、可视化（延迟/时间对比图生成）
│   ├── logger.py              # 日志记录（优化过程、参数、结果）
│   └── pipeline.py            # 异构异步流水线（解耦IO与计算密集型任务）
├── examples/                  # 示例目录
│   ├── matmul_optimize.py     # 矩阵乘法算子优化示例
│   └── rmsnorm_optimize.py    # RMSNorm算子优化示例
└── tests/                     # 单元测试目录
    ├── __init__.py
    ├── test_evolution_algos.py  # 进化算法测试
    └── test_multi_agent.py    # 多Agent协作测试
```

### 2. 核心工作流程
1. **初始化**：输入基线Triton算子代码（如matmul.py、rmsnorm.py），配置优化目标（延迟降低）、算法类型及参数；
2. **种群生成**：基于基线代码生成初始种群，通过评估器计算初始延迟；
3. **进化/协作优化**：
   - 选择进化算法（Para/GA/DE）或多Agent协作策略（MACO）；
   - 进化算法：执行选择、交叉/变异（LLM驱动语义优化）、种群更新；
   - 多Agent协作：通过角色Agent完成Tiling/Vector优化（变异）、代码融合（杂交），Meta-Agent提供动态反馈；
4. **评估筛选**：评估新生成代码的延迟、有效性，筛选最优个体进入下一代；
5. **终止迭代**：达到最大迭代次数或延迟收敛，输出最优算子代码及优化报告。

## 进化算法选择与评估
### 1. 支持的进化算法及核心实现
库中集成了3种主流进化算法，均基于LLM驱动实现语义级优化，核心逻辑如下：

| 算法名称          | 核心原理                                                                 | 关键流程（参考文档算法1-3）                                                                 |
|-------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| 爬山算法（ParaEvoluter） | 贪心搜索，基于当前最优解迭代优化                                         | 1. 初始种群初始化并评估；2. 每代筛选Top N精英；3. LLM对精英个体单独变异（Tiling/Vector优化）；4. 合并种群并保留Top N |
| 遗传算法（GA）    | 模拟生物进化，通过选择-交叉-变异实现种群优化                             | 1. 初始种群初始化并评估；2. 锦标赛选择父代（Mom/Dad）；3. LLM语义交叉（融合“最强基因”）；4. 随机变异（小调整）；5. 评估有效个体并更新种群 |
| 差分进化算法（DE） | 基于个体差异生成新解，公式：New = Target + (Best - Random)                | 1. 初始种群初始化并评估；2. 随机选择Target/Best/Random个体；3. LLM基于差异生成新代码；4. 评估有效个体并更新种群 |

### 2. 算法参数配置（`config/evolution_config.py`）
```python
# 进化算法通用参数
EVOLUTION_COMMON = {
    "population_size": 50,        # 种群规模
    "max_generations": 10,        # 最大迭代次数
    "time_limit": 800,            # 总时间限制（秒）
    "elite_ratio": 0.1,           # 精英保留比例
}

# 各算法专属参数
EVOLUTION_ALGOS = {
    "para_evoluter": {
        "parallel_workers": 8     # 并行线程数
    },
    "genetic_algorithm": {
        "mutation_prob": 0.1,     # 变异概率
        "tournament_size": 5      # 锦标赛选择规模
    },
    "differential_evolution": {
        "diff_scale": 0.5         # 差异缩放系数
    }
}
```

### 3. 算法评估结果
基于延迟、迭代时间、综合效率的评估结论（参考文档实验数据）：

| 评估维度         | 算法对比结论                                                                 |
|-------------------|------------------------------------------------------------------------------|
| 收敛效果（延迟）  | 差分进化算法（DE）> 遗传算法（GA）> 爬山算法（ParaEvoluter）                 |
| 迭代速度          | 差分进化算法（DE）< 爬山算法（ParaEvoluter）< 遗传算法（GA）                 |
| 综合效率（延迟-时间） | 差分进化算法（DE）最优（快速收敛+低延迟）；GA存在迭代异常值；Para收敛较慢     |
| 关键特性          | - DE：优化目标明确，快速推向优解区域，迭代轮数少；<br>- GA：需杂交+变异，方向随机性高，易出现异常值；<br>- Para：贪心搜索，搜索空间大，收敛慢 |

## Multi-Agents Code Optimization (MACO)
### 1. 多Agent协作核心原理

<img width="3994" height="2398" alt="overview" src="https://github.com/user-attachments/assets/817a5ea5-b077-408c-a0d8-6aeac6690dfc" />

#### （1）Role-Playing Agents（角色Agent）
- 核心思想：通过提示词为LLM赋予特定优化角色，每个Agent专注一类优化任务；
- 角色定义（可扩展）：
  - Tiling Expert：负责算子分块、循环顺序调整优化；
  - Vector Expert：负责内存访问优化（向量指令适配）；
  - Crossover Expert：负责提取两份代码的“最强基因”，融合生成新代码；
- 协作约束：禁止修改算子接口、数学逻辑，仅输出优化后代码，确保有效性。

#### （2）多Agent协作策略（MACO）
- 变异操作：由Tiling/Vector Expert Agent基于当前代码延迟，针对性优化（如分块大小调整、内存访问模式优化）；
- 杂交操作：由Crossover Expert Agent融合两个父代代码的优势特征（如A的分块策略+ B的向量优化）；
- 提示词模板：通用约束（接口/逻辑不修改）+ 角色特定提示，通过模板拼接生成LLM输入。

#### （3）Meta-Agent（动态反馈机制）
- 底层监控：通过Linux Perf和FlameGraph采集CPU运行状态（如Cache Miss、算力利用率）；
- 动态建议：根据硬件指标分析瓶颈（算力/访存受限），为下轮迭代生成针对性优化提示，注入LLM调用流程；
- 解决问题：避免LLM盲目优化，提升跨算子、跨CPU型号的泛化能力。

### 2. 协作结果评估与优化
#### （1）核心评估指标
- 性能加速比：最优优化代码延迟 / 基线代码延迟；
- 稳定性：多次优化的延迟波动范围；
- 效率：单算子优化总耗时（含LLM调用、评估时间）。

#### （2）关键问题与优化方案
| 存在问题                  | 优化方案                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| 基线代码延迟波动          | 评估器多次测试取均值/中位数，剔除异常值；                                 |
| 初代代码性能影响优化效果  | 强制Top K父代杂交+变异，提升种群多样性；极端情况触发重启动/动态调整种群参数； |
| 单算子调优时间过长        | 1. 异构异步流水线：解耦LLM API调用（IO密集）与代码验证（计算密集）；<br>2. 多分支并行搜索：多API账号轮询、种群分片； |
| 人工提示词泛化性差        | Meta-Agent基于硬件监控动态生成优化建议，替代固定Prompt；                  |

#### （3）典型优化结果
基于Deepseek V3.2模型的多Agent协作优化（MACO）结果：
- 加速比：最优优化代码相对基线平均加速比可达2-3倍；
- 稳定性：经过多次迭代后，延迟波动控制在合理范围（避免语义错误导致的性能退化）；
- 泛化性：通过Meta-Agent反馈，适配不同算子（matmul/rmsnorm）和CPU型号。

## 代码使用说明
### 1. 环境准备
1. 克隆项目到本地：
   ```bash
   git clone https://github.com/Lily-git-creator/Genetic-algorithm-and-LLM-based-Triton-cpu-optimization.git
   cd Genetic-algorithm-and-LLM-based-Triton-cpu-optimization
   ```
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
   核心依赖：`triton`（算子运行）、`numpy`（数据处理）、`matplotlib`（可视化）、`perf`（硬件监控）、`requests`（LLM API调用）。
3. 配置关键参数：
   - 修改`config/evolution_config.py`选择优化算法（`algo_choice: "DE"/"GA"/"Para"`）及参数；
   - 修改`config/agent_config.py`启用多Agent协作（`use_maco: True/False`）；
   - 在`llm_handler/llm_client.py`配置LLM API密钥、模型地址；

### 2. 快速开始（以matmul算子优化为例）
#### （1）运行示例代码
```bash
python examples/matmul_optimize.py
```
#### （2）自定义算子优化
1. 在项目根目录创建自定义算子文件（如`my_operator.py`），确保算子接口符合Triton规范；
2. 编写优化脚本：
```python
from main import TritonOptimizer

# 初始化优化器（指定算法、是否启用多Agent）
optimizer = TritonOptimizer(
    algo_choice="DE",  # 选择差分进化算法（可选："DE"/"GA"/"Para"）
    use_maco=True,     # 启用多Agent协作
    baseline_code_path="my_operator.py",  # 基线算子代码路径
    max_generations=10,  # 迭代次数
    population_size=50   # 种群规模
)

# 执行优化
best_code, optimization_report = optimizer.run()

# 保存最优代码和报告
optimizer.save_result(best_code, optimization_report, save_path="results/")
```
3. 运行脚本：
```bash
python my_optimize_script.py
```

### 3. 结果查看
- 最优代码：保存于`results/best_operator.py`；
- 优化报告：`results/optimization_report.json`（含每代最优延迟、总耗时、加速比）；
- 可视化图表：`results/`目录下生成延迟-迭代次数、总耗时-延迟对比图；
- 日志文件：`logs/optimization.log`记录优化过程、参数、异常信息。

### 4. 单元测试运行
```bash
# 测试所有进化算法
python -m pytest tests/test_evolution_algos.py -v

# 测试多Agent协作功能
python -m pytest tests/test_multi_agent.py -v
```

## 总结与展望
### 1. 项目总结
本项目构建了一个集成多种进化算法与多Agent协作的Triton算子自动优化库，核心优势如下：
- 算法丰富：提供爬山算法、遗传算法、差分进化算法，支持不同场景选择（DE最优综合效率）；
- 协作增强：多Agent角色分工+Meta-Agent动态反馈，解决LLM盲目优化、泛化性差问题；
- 高效可靠：异构异步流水线、多分支并行搜索提升优化效率，严格的有效性校验确保代码可用；
- 易用性强：模块化设计，支持自定义算子、参数配置，输出可视化报告和最优代码。

### 2. 未来展望
1. 算法改进：融合多种进化算法的优势，设计混合优化策略，进一步提升收敛速度和稳定性；
2. Agent系统升级：扩展Agent角色（如硬件适配专家、性能调试专家），增强协作智能；
3. 动态优化增强：Meta-Agent支持任务相关的针对性优化，适配更复杂的算子（如Transformer层算子）；
4. 性能瓶颈突破：优化LLM推理效率（如模型量化、本地部署），进一步降低优化耗时；
5. 功能扩展：支持GPU后端Triton算子优化，适配更多硬件平台。

## 常见问题
1. **如何切换不同的进化算法？**  
   修改`config/evolution_config.py`中的`algo_choice`字段，可选值为`"para_evoluter"`、`"genetic_algorithm"`、`"differential_evolution"`，同时可调整对应算法的专属参数。
2. **多Agent协作如何添加新的优化角色？**  
   在`multi_agent/role_agents.py`中定义新角色类，在`llm_handler/prompt_templates.py`添加对应角色的提示词模板，修改`agent_config.py`启用新角色。
3. **基线代码延迟波动过大怎么办？**  
   调整`config/evaluator_config.py`中的`test_times`（增加测试次数），启用`outlier_remove`（剔除异常值），或参考`evaluator/latency_evaluator.py`的逻辑优化延迟计算方式。
4. **LLM调用耗时过长如何优化？**  
   启用`utils/pipeline.py`的异构异步流水线，在`llm_handler/llm_client.py`中配置多API并发，或使用本地部署的轻量级LLM替代API调用。
