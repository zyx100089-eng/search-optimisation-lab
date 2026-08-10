"""Regenerate the report figures from the current results.csv.

Run:  python experiments/make_report_figures.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "results.csv")
OUT = os.path.join(HERE, "..", "report")

ALGO_ORDER = ["A*", "BFS", "Dijkstra", "Greedy", "DFS"]
COLORS = {
    "A*": "#f39c12",
    "BFS": "#3498db",
    "Dijkstra": "#e74c3c",
    "Greedy": "#2ecc71",
    "DFS": "#9b59b6",
}

df = pd.read_csv(CSV)
found = df[df["found_path"] == True].copy()

# fig1: overall comparison (4 subplots)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, metric, title, ylabel in zip(
    axes,
    ["path_cost", "nodes_explored", "runtime_ms", "peak_memory_kb"],
    ["Average Path Cost", "Average Nodes Explored", "Average Runtime (ms)", "Average Peak Memory (KB)"],
    ["Cost", "Nodes", "ms", "KB"],
):
    means = found.groupby("algorithm")[metric].mean().reindex(ALGO_ORDER)
    bars = ax.bar(means.index, means.values, color=[COLORS[a] for a in means.index])
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_overall_comparison.png"), dpi=120)
plt.close(fig)
print("fig1 saved")

# fig2: nodes explored by grid size
by_size = found.groupby(["grid_size", "algorithm"])["nodes_explored"].mean().unstack().reindex(columns=ALGO_ORDER)
fig, ax = plt.subplots(figsize=(8, 5))
for algo in ALGO_ORDER:
    ax.plot(by_size.index, by_size[algo], "o-", label=algo, color=COLORS[algo], linewidth=2)
ax.set_xlabel("Grid size (n x n)")
ax.set_ylabel("Average nodes explored")
ax.set_title("Nodes Explored by Grid Size", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_nodes_by_size.png"), dpi=120)
plt.close(fig)
print("fig2 saved")

# fig3: found rate by density
rate = df.groupby(["obstacle_density", "algorithm"])["found_path"].mean().unstack().reindex(columns=ALGO_ORDER) * 100
fig, ax = plt.subplots(figsize=(8, 5))
for algo in ALGO_ORDER:
    ax.plot(rate.index, rate[algo], "s-", label=algo, color=COLORS[algo], linewidth=2)
ax.set_xlabel("Obstacle density")
ax.set_ylabel("Path found (%)")
ax.set_title("Path Discovery Rate by Density", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_found_rate_density.png"), dpi=120)
plt.close(fig)
print("fig3 saved")

# fig4: runtime scaling
rt = found.groupby(["grid_size", "algorithm"])["runtime_ms"].mean().unstack().reindex(columns=ALGO_ORDER)
fig, ax = plt.subplots(figsize=(8, 5))
for algo in ALGO_ORDER:
    ax.plot(rt.index, rt[algo], "o-", label=algo, color=COLORS[algo], linewidth=2)
ax.set_xlabel("Grid size (n x n)")
ax.set_ylabel("Runtime (ms)")
ax.set_title("Runtime Scaling", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_runtime_scaling.png"), dpi=120)
plt.close(fig)
print("fig4 saved")

# fig5: learned heuristic trade-off (generated fresh, not from results.csv)
from src.graph import Grid
from src.algorithms.dijkstra import dijkstra
from src.algorithms.a_star import a_star
from src.algorithms.learned_heuristic import (
    generate_training_data, train_learned_heuristic, make_learned_heuristic,
)

samples = generate_training_data(n_grids=40, rows=20, cols=20, density=0.2, seed=0, samples_per_grid=25)
tr = train_learned_heuristic(samples, epochs=200, lr=0.01, seed=0)

recs = []
for s in range(200, 230):
    g = Grid.generate_random(20, 20, 0.2, False, s)
    g.set_passable(0, 0); g.set_passable(19, 19)
    rd = dijkstra(g, (0, 0), (19, 19))
    rm = a_star(g, (0, 0), (19, 19), heuristic="manhattan")
    h = make_learned_heuristic(tr.model, g)
    rl = a_star(g, (0, 0), (19, 19), heuristic=h)
    if rd.path and rm.path and rl.path:
        recs.append({
            "node_reduction": (rm.explored_count - rl.explored_count) / rm.explored_count * 100,
            "subopt": rl.path_cost / rd.path_cost,
        })

if recs:
    nr = [r["node_reduction"] for r in recs]
    so = [r["subopt"] for r in recs]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(nr, so, s=50, alpha=0.7, c="tab:blue", edgecolors="navy")
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1, label="Optimal (1.00x)")
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Node reduction vs Manhattan (%)")
    ax.set_ylabel("Suboptimality ratio (cost / optimal)")
    ax.set_title("Learned Heuristic: Speed vs Optimality", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_learned_heuristic_tradeoff.png"), dpi=120)
    plt.close(fig)
    print("fig5 saved")
print("done")