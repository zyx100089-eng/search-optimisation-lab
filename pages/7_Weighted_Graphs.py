"""Visualiser for general weighted graphs (not just grids).

Demonstrates algorithms on hand-crafted and random graph topologies
with varying edge weights.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import math
import heapq
import time
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from src.graph import Graph
from src.algorithms.memory_util import track_memory

st.set_page_config(page_title="Weighted Graphs", layout="wide")
st.title("Weighted Graph Visualiser")

st.markdown("""
This page runs search algorithms on **general weighted graphs** — not grids.
Choose from preset graph topologies or generate random ones, then compare
how Dijkstra, A\\*, and Greedy behave with real edge weights.
""")


def _dijkstra_graph(graph, start, goal, positions):
    t0 = time.perf_counter()
    _mem_ctx = track_memory()
    mem = _mem_ctx.__enter__()
    dist = {start: 0.0}
    parent = {start: None}
    pq = [(0.0, start)]
    explored = []
    closed = set()
    while pq:
        cost, node = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)
        explored.append(node)
        if node == goal:
            break
        for nb, w in graph.neighbours(node):
            nc = cost + w
            if nb not in dist or nc < dist[nb]:
                dist[nb] = nc
                parent[nb] = node
                heapq.heappush(pq, (nc, nb))
    path = _recon(parent, goal)
    _mem_ctx.__exit__(None, None, None)
    return path, dist.get(goal, float("inf")), explored, time.perf_counter() - t0, mem["peak_bytes"]


def _astar_graph(graph, start, goal, positions):
    def h(n):
        if n not in positions or goal not in positions:
            return 0.0
        x1, y1 = positions[n]
        x2, y2 = positions[goal]
        return math.hypot(x1 - x2, y1 - y2)

    t0 = time.perf_counter()
    _mem_ctx = track_memory()
    mem = _mem_ctx.__enter__()
    g = {start: 0.0}
    parent = {start: None}
    counter = 0
    pq = [(h(start), counter, start)]
    explored = []
    closed = set()
    while pq:
        _f, _c, node = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)
        explored.append(node)
        if node == goal:
            break
        for nb, w in graph.neighbours(node):
            ng = g[node] + w
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                parent[nb] = node
                counter += 1
                heapq.heappush(pq, (ng + h(nb), counter, nb))
    path = _recon(parent, goal)
    _mem_ctx.__exit__(None, None, None)
    return path, g.get(goal, float("inf")), explored, time.perf_counter() - t0, mem["peak_bytes"]


def _greedy_graph(graph, start, goal, positions):
    def h(n):
        if n not in positions or goal not in positions:
            return 0.0
        x1, y1 = positions[n]
        x2, y2 = positions[goal]
        return math.hypot(x1 - x2, y1 - y2)

    t0 = time.perf_counter()
    _mem_ctx = track_memory()
    mem = _mem_ctx.__enter__()
    parent = {start: None}
    cost_to = {start: 0.0}
    counter = 0
    pq = [(h(start), counter, start)]
    explored = []
    closed = set()
    while pq:
        _h, _c, node = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)
        explored.append(node)
        if node == goal:
            break
        for nb, w in graph.neighbours(node):
            if nb not in closed:
                nc = cost_to[node] + w
                if nb not in cost_to or nc < cost_to[nb]:
                    cost_to[nb] = nc
                    parent[nb] = node
                counter += 1
                heapq.heappush(pq, (h(nb), counter, nb))
    path = _recon(parent, goal)
    _mem_ctx.__exit__(None, None, None)
    return path, cost_to.get(goal, float("inf")), explored, time.perf_counter() - t0, mem["peak_bytes"]


def _recon(parent, goal):
    if goal not in parent:
        return []
    path = []
    n = goal
    while n is not None:
        path.append(n)
        n = parent[n]
    return path[::-1]


def _draw_graph(graph, positions, path=None, explored=None, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    nodes = list(positions.keys())

    for node in nodes:
        for nb, w in graph.neighbours(node):
            if node < nb:
                x1, y1 = positions[node]
                x2, y2 = positions[nb]
                ax.plot([x1, x2], [y1, y2], "gray", linewidth=0.8, alpha=0.5)
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mx, my, f"{w:.0f}", fontsize=7, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="gray", alpha=0.8))

    if explored:
        for n in explored:
            if n in positions:
                x, y = positions[n]
                ax.plot(x, y, "o", color="#aed6f1", markersize=18, zorder=2)

    if path and len(path) > 1:
        for i in range(len(path) - 1):
            x1, y1 = positions[path[i]]
            x2, y2 = positions[path[i + 1]]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="#2980b9", lw=2.5), zorder=3)

    for node in nodes:
        x, y = positions[node]
        ax.plot(x, y, "o", color="#ecf0f1", markersize=22, markeredgecolor="#2c3e50", markeredgewidth=1.5, zorder=4)
        ax.text(x, y, str(node), ha="center", va="center", fontsize=9, fontweight="bold", zorder=5)

    if path:
        sx, sy = positions[path[0]]
        gx, gy = positions[path[-1]]
        ax.plot(sx, sy, "o", color="#27ae60", markersize=22, zorder=4)
        ax.text(sx, sy, str(path[0]), ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=5)
        ax.plot(gx, gy, "o", color="#e74c3c", markersize=22, zorder=4)
        ax.text(gx, gy, str(path[-1]), ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


PRESETS = {
    "City Map (12 nodes)": {
        "edges": [
            ("A", "B", 4), ("A", "C", 2), ("B", "C", 1), ("B", "D", 5),
            ("C", "D", 8), ("C", "E", 10), ("D", "E", 2), ("D", "F", 6),
            ("E", "F", 3), ("E", "G", 7), ("F", "H", 1), ("G", "H", 4),
            ("G", "I", 3), ("H", "I", 2), ("I", "J", 5), ("H", "J", 8),
            ("J", "K", 3), ("K", "L", 2), ("I", "L", 6),
        ],
        "positions": {
            "A": (0, 3), "B": (2, 5), "C": (2, 1), "D": (4, 4),
            "E": (4, 0), "F": (6, 3), "G": (6, -1), "H": (8, 2),
            "I": (8, -1), "J": (10, 3), "K": (12, 4), "L": (12, 0),
        },
        "start": "A", "goal": "L",
    },
    "Tricky Shortcut (8 nodes)": {
        "edges": [
            ("S", "A", 1), ("S", "B", 10), ("A", "C", 1), ("C", "D", 1),
            ("D", "E", 1), ("E", "G", 1), ("B", "G", 1),
            ("A", "F", 15), ("F", "G", 1),
        ],
        "positions": {
            "S": (0, 2), "A": (2, 4), "B": (2, 0), "C": (4, 5),
            "D": (6, 4), "E": (8, 3), "F": (5, 1), "G": (10, 2),
        },
        "start": "S", "goal": "G",
    },
}

with st.sidebar:
    st.header("Graph Settings")
    mode = st.selectbox("Graph type", ["Preset: City Map (12 nodes)", "Preset: Tricky Shortcut (8 nodes)", "Random graph"])

    if mode == "Random graph":
        n_nodes = st.slider("Nodes", 5, 25, 12, key="wg_nodes")
        edge_prob = st.slider("Edge probability", 0.1, 0.6, 0.3, 0.05, key="wg_eprob")
        max_weight = st.slider("Max edge weight", 1, 20, 10, key="wg_maxw")
        wg_seed = st.number_input("Seed", value=42, step=1, key="wg_seed")

run_graph = st.button("Run on Graph", type="primary")

if run_graph:
    if mode.startswith("Preset"):
        preset_key = mode.replace("Preset: ", "")
        cfg = PRESETS[preset_key]
        graph = Graph()
        for u, v, w in cfg["edges"]:
            graph.add_edge(u, v, w)
        positions = cfg["positions"]
        start, goal = cfg["start"], cfg["goal"]
    else:
        rng = random.Random(int(wg_seed))
        graph = Graph()
        nodes = [str(i) for i in range(n_nodes)]
        angle_step = 2 * math.pi / n_nodes
        positions = {}
        for i, n in enumerate(nodes):
            positions[n] = (5 * math.cos(i * angle_step), 5 * math.sin(i * angle_step))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < edge_prob:
                    w = rng.randint(1, max_weight)
                    graph.add_edge(nodes[i], nodes[j], w)
        for i in range(n_nodes - 1):
            if nodes[i + 1] not in dict(graph.neighbours(nodes[i])):
                graph.add_edge(nodes[i], nodes[i + 1], rng.randint(1, max_weight))
        start, goal = nodes[0], nodes[-1]

    algos = {
        "Dijkstra": _dijkstra_graph,
        "A*": _astar_graph,
        "Greedy": _greedy_graph,
    }

    cols_layout = st.columns(3)
    results = {}
    for col, (name, fn) in zip(cols_layout, algos.items()):
        path, cost, explored, runtime, mem_bytes = fn(graph, start, goal, positions)
        results[name] = (path, cost, explored, runtime, mem_bytes)
        with col:
            fig = _draw_graph(graph, positions, path, explored, title=f"{name}\nCost: {cost:.1f}")
            st.pyplot(fig)

    st.subheader("Comparison")
    import pandas as pd
    data = []
    for name, (path, cost, explored, runtime, mem_bytes) in results.items():
        data.append({
            "Algorithm": name,
            "Path Cost": round(cost, 1),
            "Path": " → ".join(str(n) for n in path) if path else "No path",
            "Nodes Explored": len(explored),
            "Runtime (ms)": round(runtime * 1000, 3),
            "Memory (KB)": round(mem_bytes / 1024, 1),
            "Optimal": "Yes" if abs(cost - results["Dijkstra"][1]) < 0.01 else "No",
        })
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
