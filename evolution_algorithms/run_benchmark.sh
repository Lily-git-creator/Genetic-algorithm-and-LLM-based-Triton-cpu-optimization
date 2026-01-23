#!/bin/bash

# 设置相关变量
BUDGET=5          # 每一轮的迭代次数
POP_SIZE=2        # 种群大小
TIME_LIMIT=1200   # 20分钟超时 (秒)

echo "==========================================="
echo "🚀 Starting Comprehensive Benchmark Suite"
echo "==========================================="

# 1. Run Para (Paraphrasing / Hill Climbing)
echo ""
echo "▶️  Running Mode: PARA..."
python main.py --mode para --budget $BUDGET --pop_size $POP_SIZE --time_limit $TIME_LIMIT
if [ $? -eq 0 ]; then
    echo "✅ Para finished successfully."
else
    echo "❌ Para failed."
fi

# 2. Run GA (Genetic Algorithm)
echo ""
echo "▶️  Running Mode: GA..."
python main.py --mode ga --budget $BUDGET --pop_size $POP_SIZE --time_limit $TIME_LIMIT
if [ $? -eq 0 ]; then
    echo "✅ GA finished successfully."
else
    echo "❌ GA failed."
fi

# 3. Run DE (Differential Evolution)
echo ""
echo "▶️  Running Mode: DE..."
python main.py --mode de --budget $BUDGET --pop_size $POP_SIZE --time_limit $TIME_LIMIT
if [ $? -eq 0 ]; then
    echo "✅ DE finished successfully."
else
    echo "❌ DE failed."
fi

echo ""
echo "==========================================="
echo "📊 Generating Analysis Plots..."
echo "==========================================="

# 调用可视化脚本
python visualize_results.py

echo ""
echo "🎉 All Done! Check 'plots/' directory for results."
