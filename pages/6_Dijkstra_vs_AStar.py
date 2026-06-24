"""Dedicated experimental comparison of Dijkstra and A*.

Shows exactly why A* outperforms Dijkstra, how much, and when the gap grows.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from src.graph import Grid
from src.algorithms.dijkstra import dijkstra
from src.algorithms.a_star import a_star
from src.visualisation.grid_view import draw_grid

st.set_page_config(page_title="Dijkstra vs A*", layout="wide")
st.title("Dijkstra vs A* — Experimental Comparison")

tab1, tab2 = st.tabs(["Side-by-Side Visualisation", "Scaling Experiment"])

with tab1:
    st.markdown("""
    Run both algorithms on the same grid and compare the **exploration pattern**.
    A\\*'s heuristic focuses exploration toward the goal, while Dijkstra explores
    uniformly in all directions.
    """)

    with st.sidebar:
        st.header("Grid Settings")
        rows = st.slider("Rows", 10, 60, 30, key="da_rows")
        cols = st.slider("Cols", 10, 60, 30, key="da_cols")
        density = st.slider("Density", 0.0, 0.4, 0.2, 0.05, key="da_density")
        weighted = st.checkbox("Weighted grid", key="da_weighted")
        seed = st.number_input("Seed", value=42, step=1, key="da_seed")

    run_compare = st.button("Run Comparison", type="primary", key="da_run")

    if run_compare:
        grid = Grid.generate_random(rows, cols, density, False, int(seed), weighted)
        start, goal = (0, 0), (rows - 1, cols - 1)
        grid.set_passable(*start)
        grid.set_passable(*goal)

        r_dij = dijkstra(grid, start, goal)
        r_astar = a_star(grid, start, goal)

        c1, c2 = st.columns(2)
        with c1:
            fig = draw_grid(grid, r_dij.path, r_dij.explored, start, goal, title="Dijkstra")
            st.pyplot(fig)
        with c2:
            fig = draw_grid(grid, r_astar.path, r_astar.explored, start, goal, title="A*")
            st.pyplot(fig)

        st.subheader("Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cost (both)", f"{r_dij.path_cost:.1f}",
                  help="Both produce optimal paths — costs should be equal")
        saved = r_dij.explored_count - r_astar.explored_count
        pct = (saved / r_dij.explored_count * 100) if r_dij.explored_count > 0 else 0
        m2.metric("Nodes saved by A*", f"{saved} ({pct:.0f}%)")
        m3.metric("Dijkstra runtime", f"{r_dij.runtime*1000:.2f} ms")
        m4.metric("A* runtime", f"{r_astar.runtime*1000:.2f} ms")

        st.subheader("Memory")
        mem1, mem2 = st.columns(2)
        mem1.metric("Dijkstra peak memory", f"{r_dij.peak_memory_bytes/1024:.1f} KB")
        mem2.metric("A* peak memory", f"{r_astar.peak_memory_bytes/1024:.1f} KB")

        if abs(r_dij.path_cost - r_astar.path_cost) < 0.01 and r_dij.path_length > 0:
            st.success(
                f"Both algorithms found optimal paths with cost {r_dij.path_cost:.1f}.  "
                f"A\\* explored **{pct:.0f}% fewer nodes** by using the Manhattan heuristic "
                f"to focus toward the goal."
            )
        elif r_dij.path_length == 0:
            st.warning("No path exists on this grid.")


with tab2:
    st.markdown("""
    ### How does A\\*'s advantage scale with grid size?

    This experiment runs both algorithms on grids from 10x10 to 100x100 and
    measures nodes explored, runtime, and memory.
    """)

    run_scale = st.button("Run Scaling Experiment", type="primary", key="da_scale")

    if run_scale:
        sizes = [10, 15, 20, 30, 40, 50, 60, 80, 100]
        dij_nodes, astar_nodes = [], []
        dij_time, astar_time = [], []
        dij_mem, astar_mem = [], []
        actual_sizes = []

        progress = st.progress(0)
        for i, s in enumerate(sizes):
            grid = Grid.generate_random(s, s, 0.2, False, 42)
            start, goal = (0, 0), (s - 1, s - 1)
            grid.set_passable(*start)
            grid.set_passable(*goal)

            rd = dijkstra(grid, start, goal)
            ra = a_star(grid, start, goal)

            if rd.path_length > 0:
                actual_sizes.append(s)
                dij_nodes.append(rd.explored_count)
                astar_nodes.append(ra.explored_count)
                dij_time.append(rd.runtime * 1000)
                astar_time.append(ra.runtime * 1000)
                dij_mem.append(rd.peak_memory_bytes / 1024)
                astar_mem.append(ra.peak_memory_bytes / 1024)

            progress.progress((i + 1) / len(sizes))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax = axes[0]
        ax.plot(actual_sizes, dij_nodes, "o-", label="Dijkstra", color="#e74c3c", linewidth=2)
        ax.plot(actual_sizes, astar_nodes, "s-", label="A*", color="#f39c12", linewidth=2)
        ax.fill_between(actual_sizes, astar_nodes, dij_nodes, alpha=0.15, color="green", label="Nodes saved")
        ax.set_xlabel("Grid size (n x n)")
        ax.set_ylabel("Nodes explored")
        ax.set_title("Nodes Explored", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(actual_sizes, dij_time, "o-", label="Dijkstra", color="#e74c3c", linewidth=2)
        ax.plot(actual_sizes, astar_time, "s-", label="A*", color="#f39c12", linewidth=2)
        ax.set_xlabel("Grid size (n x n)")
        ax.set_ylabel("Runtime (ms)")
        ax.set_title("Runtime", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.plot(actual_sizes, dij_mem, "o-", label="Dijkstra", color="#e74c3c", linewidth=2)
        ax.plot(actual_sizes, astar_mem, "s-", label="A*", color="#f39c12", linewidth=2)
        ax.set_xlabel("Grid size (n x n)")
        ax.set_ylabel("Peak memory (KB)")
        ax.set_title("Memory Usage", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)

        savings = [(s, (dn - an) / dn * 100)
                   for s, dn, an in zip(actual_sizes, dij_nodes, astar_nodes)]
        avg_saving = np.mean([s for _, s in savings])
        st.metric("Average node reduction by A*", f"{avg_saving:.1f}%")
        st.markdown(
            f"A\\*'s advantage is consistent across sizes.  The heuristic eliminates "
            f"~{avg_saving:.0f}% of the work on average, and the gap tends to grow "
            f"with grid size because Dijkstra's uniform expansion wastes more effort "
            f"in larger search spaces."
        )
