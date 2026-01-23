## 本目录下的文件说明

## 主函数

`evolution_main.py` 主程序，调用api+选择不同的遗传策略, 一代代的优化代码
`evolution_main_k.py` top k ，即每一代选取的精英数量
        K = max(2, self.pop_size // 2) 

`evaluator.py` 评估代码的运行时间，因为我们的评价指标主要算这个

`llm_handler.py` 调用deepseek V3, 设置了三种不同的角色，专用提示词+通用提示词

### 辅助函数

`program_error.py` 编写此脚本来观察baseline代码的运行时间波动，决定使用运行50次取最小值的方法确定代码运行时间。
    - 在本服务器上的运行结果如图 `latency_distribution.png`

`profiler.py` 观察API调用和评估过程的时间占比，看看谁拖后腿了。
    - 模型推理。Prompt和Agents协作模式需要进一步优化