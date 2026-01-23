import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def visualize_evolution_ultimate(history_path="evolution_history.json"):
    with open(history_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()
    node_info = {}
    
    # 1. 映射逻辑编号 (按 JSON 出现顺序 0, 1, 2...)
    id_to_num = {ind['id']: i for i, ind in enumerate(data)}
    num_to_id = {i: ind['id'] for i, ind in enumerate(data)}

    # 2. 预处理数据与识别最优
    valid_latencies = [ind['latency'] for ind in data if ind['latency'] > 0]
    best_latency = min(valid_latencies) if valid_latencies else 0
    best_node_num = -1

    for ind in data:
        uid = ind['id']
        nid = id_to_num[uid]
        method = ind['method'].lower()
        lat = ind['latency']
        
        # 布局层级：Baseline 设为 -1 独占顶层
        logic_gen = -1 if "baseline" in method else ind['generation']
        is_best = (lat == best_latency and lat > 0)
        if is_best: best_node_num = nid
        
        node_info[nid] = {
            'lat': lat, 'method': method, 'gen': logic_gen, 'is_best': is_best
        }
        G.add_node(nid)
        for p_id in ind['parents']:
            if p_id in id_to_num:
                G.add_edge(id_to_num[p_id], nid)

    # 3. 追踪最优路径
    best_path_edges = []
    if best_node_num != -1:
        curr = best_node_num
        while True:
            parents = list(G.predecessors(curr))
            if not parents: break
            # 追溯第一个父节点（通常是主要的进化来源）
            p = parents[0]
            best_path_edges.append((p, curr))
            curr = p

    # 4. 手动计算严格分层布局
    pos = {}
    layers = {}
    for nid, info in node_info.items():
        gen = info['gen']
        layers.setdefault(gen, []).append(nid)

    for gen in sorted(layers.keys()):
        nodes = sorted(layers[gen])
        width = len(nodes)
        for i, nid in enumerate(nodes):
            pos[nid] = (i - (width - 1) / 2.0, -gen)

    # 5. 绘图开始
    fig, ax = plt.subplots(figsize=(16, 12))
    min_l, max_l = min(valid_latencies), max(valid_latencies)

    # 6. 绘制连线
    # A. 先画普通边：更黑、箭头更大
    regular_edges = [e for e in G.edges() if e not in best_path_edges]
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, ax=ax, 
                           edge_color="#444444", alpha=0.5, 
                           width=1.5, arrows=True, arrowsize=25, 
                           connectionstyle="arc3,rad=0.1")
    
    # B. 再画最优路径：极粗黑色
    nx.draw_networkx_edges(G, pos, edgelist=best_path_edges, ax=ax,
                           edge_color="black", alpha=1.0,
                           width=4.5, arrows=True, arrowsize=35,
                           connectionstyle="arc3,rad=0.1")

    # 7. 绘制节点
    for nid in G.nodes():
        info = node_info[nid]
        lat = info['lat']
        
        if info['is_best']:
            shape, size, ec, lw = '*', 1500, "#E67E22", 3.0 # 橙色边框大星
        elif "baseline" in info['method']:
            shape, size, ec, lw = '^', 900, "black", 2.0
        elif lat <= 0:
            shape, size, ec, lw = 'x', 400, "red", 1.5
        elif "crossover" in info['method']:
            shape, size, ec, lw = 'P', 700, "black", 1.0
        else:
            shape, size, ec, lw = 'o', 600, "black", 1.0

        color = plt.cm.viridis_r((lat - min_l) / (max_l - min_l + 1e-9)) if lat > 0 else "#FDEDEC"

        nx.draw_networkx_nodes(G, pos, nodelist=[nid], 
                               node_shape=shape, node_size=size,
                               node_color=[color], edgecolors=ec,
                               linewidths=lw, ax=ax)

    # 8. 绘制标号
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    # 9. 颜色条与图例
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=plt.Normalize(vmin=min_l*1000, vmax=max_l*1000))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Latency (ms)', fontsize=12, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Best (Lowest Latency)', markerfacecolor='yellow', markeredgecolor='#E67E22', markersize=18),
        Line2D([0], [0], color='black', lw=4, label='Winning Path'),
        Line2D([0], [0], marker='^', color='w', label='Initial (0)', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='P', color='w', label='Crossover', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='x', color='w', label='Incorrect', markeredgecolor='red', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=11)

    plt.title("Triton Evolution: Optimal Path Analysis", fontsize=20, fontweight='bold', pad=30)
    plt.axis('off')
    
    out_file = "evolution_path_optimized.png"
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f"✅根据evolution_history，进化图已生成：{out_file}")

if __name__ == "__main__":
    visualize_evolution_ultimate()