import json
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import pandas as pd

# 设置样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

OUTPUT_DIR = "outputs"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data():
    files = glob.glob(os.path.join(OUTPUT_DIR, "*_metrics.json"))
    all_data = []
    
    for f in files:
        with open(f, 'r') as json_file:
            data = json.load(json_file)
            mode = data['mode'].upper()
            baseline = data['baseline_latency']
            
            for gen in data['generations']:
                all_data.append({
                    "Mode": mode,
                    "Generation": gen['generation'],
                    "Best Latency (ms)": gen['best_latency'] * 1000,
                    "Speedup Ratio": gen['speedup'],
                    "Cumulative Time (s)": gen['total_elapsed'],
                    "Time per Gen (s)": gen['gen_duration']
                })
    return pd.DataFrame(all_data)

def plot_all(df):
    if df.empty:
        print("No data found to plot!")
        return

    # 1. 收敛速度 (Latency vs Generation)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Generation", y="Best Latency (ms)", hue="Mode", marker="o", linewidth=2.5)
    plt.title("Convergence Speed: Latency vs. Generation", fontsize=14)
    plt.ylabel("Latency (ms) [Lower is Better]")
    plt.xlabel("Generation")
    plt.savefig(os.path.join(PLOT_DIR, "1_convergence_latency.png"))
    plt.close()

    # 2. 加速比 (Speedup vs Generation)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Generation", y="Speedup Ratio", hue="Mode", marker="s", linewidth=2.5)
    plt.title("Optimization Effectiveness: Speedup Ratio", fontsize=14)
    plt.ylabel("Speedup Ratio (max(T_base/T_curr - 1, 0))")
    plt.xlabel("Generation")
    plt.savefig(os.path.join(PLOT_DIR, "2_speedup_ratio.png"))
    plt.close()

    # 3. 时间效率 (Best Latency vs Wall Clock Time)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Cumulative Time (s)", y="Best Latency (ms)", hue="Mode", marker="D", linewidth=2.5)
    plt.title("Time Efficiency: Latency vs. Wall-clock Time", fontsize=14)
    plt.ylabel("Latency (ms)")
    plt.xlabel("Total Time Elapsed (s)")
    plt.savefig(os.path.join(PLOT_DIR, "3_time_efficiency.png"))
    plt.close()

    print(f"✅ Plots saved to {PLOT_DIR}/")

if __name__ == "__main__":
    df = load_data()
    plot_all(df)
