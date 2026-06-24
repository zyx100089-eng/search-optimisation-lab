import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from src.graph import Grid
from src.algorithms.dynamic_programming import floyd_warshall, grid_tsp
from src.algorithms.a_star import a_star

st.set_page_config(page_title="Dynamic Programming", layout="wide")
st.title("Dynamic Programming")

tab1, tab2 = st.tabs(["Floyd-Warshall (All-Pairs Shortest Paths)", "Held-Karp (Exact TSP)"])

with tab1:
    st.markdown(
        "Floyd-Warshall computes shortest paths between **all** pairs of nodes "
        "in O(V^3).  On small grids this lets you query any source-target pair instantly."
    )

    with st.sidebar:
        st.header("Floyd-Warshall Settings")
        fw_size = st.slider("Grid size (n x n)", 3, 15, 8, key="fw_size")
        fw_density = st.slider("Obstacle density", 0.0, 0.4, 0.15, 0.05, key="fw_density")
        fw_seed = st.number_input("Seed", value=42, step=1, key="fw_seed")
        fw_run = st.button("Run Floyd-Warshall", type="primary", use_container_width=True)

    if fw_run:
        grid = Grid.generate_random(fw_size, fw_size, fw_density, False, int(fw_seed))
        grid.set_passable(0, 0)
        grid.set_passable(fw_size - 1, fw_size - 1)

        fw_result = floyd_warshall(grid)
        astar_result = a_star(grid, (0, 0), (fw_size - 1, fw_size - 1))
        fw_path = fw_result.path((0, 0), (fw_size - 1, fw_size - 1))
        fw_cost = fw_result.cost((0, 0), (fw_size - 1, fw_size - 1))

        c1, c2 = st.columns(2)

        for col, title, path, cost in [
            (c1, "Floyd-Warshall", fw_path, fw_cost),
            (c2, "A* (comparison)", astar_result.path, astar_result.path_cost),
        ]:
            with col:
                img = np.ones((fw_size, fw_size, 3))
                for r in range(fw_size):
                    for c in range(fw_size):
                        if grid.grid[r][c] == 1:
                            img[r, c] = [0.2, 0.2, 0.2]
                for r, cc in path:
                    img[r, cc] = [0.2, 0.6, 1.0]
                img[0, 0] = [0.0, 0.8, 0.0]
                img[fw_size - 1, fw_size - 1] = [1.0, 0.2, 0.2]

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img, interpolation="nearest")
                ax.set_xticks(np.arange(-0.5, fw_size, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, fw_size, 1), minor=True)
                ax.grid(which="minor", color="gray", linewidth=0.3)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{title}\nCost: {cost:.1f}  |  Steps: {len(path)}", fontweight="bold")
                st.pyplot(fig)

        st.metric("Floyd-Warshall runtime", f"{fw_result.runtime * 1000:.2f} ms")
        st.metric("Passable nodes", len(fw_result.node_list))

        if fw_cost == float("inf"):
            st.warning("No path exists between (0,0) and the bottom-right corner.")
        elif abs(fw_cost - astar_result.path_cost) < 0.01:
            st.success("Both algorithms agree on the optimal cost.")
        else:
            st.error("Cost mismatch — this indicates a bug.")


with tab2:
    st.markdown(
        "Held-Karp solves the Travelling Salesman Problem **exactly** in O(2^n n^2).  "
        "Place waypoints on a grid and find the shortest tour visiting all of them."
    )

    with st.sidebar:
        st.header("TSP Settings")
        tsp_size = st.slider("Grid size", 5, 15, 10, key="tsp_size")
        tsp_density = st.slider("Obstacle density", 0.0, 0.3, 0.1, 0.05, key="tsp_density")
        tsp_n = st.slider("Number of waypoints", 3, 12, 6, key="tsp_n")
        tsp_seed = st.number_input("Seed", value=7, step=1, key="tsp_seed")
        tsp_run = st.button("Run Held-Karp TSP", type="primary", use_container_width=True)

    if tsp_run:
        grid = Grid.generate_random(tsp_size, tsp_size, tsp_density, False, int(tsp_seed))
        rng = random.Random(int(tsp_seed))
        passable = [(r, c) for r in range(tsp_size) for c in range(tsp_size) if grid.passable(r, c)]
        if len(passable) < tsp_n:
            st.error("Not enough passable cells for the requested waypoints.")
        else:
            waypoints = sorted(rng.sample(passable, tsp_n))

            with st.spinner("Computing pairwise distances and solving TSP..."):
                result = grid_tsp(grid, waypoints)

            if result.tour_cost == float("inf"):
                st.error("Some waypoints are unreachable from each other — try lower obstacle density.")
            else:
                fig, ax = plt.subplots(figsize=(7, 7))
                img = np.ones((tsp_size, tsp_size, 3))
                for r in range(tsp_size):
                    for c in range(tsp_size):
                        if grid.grid[r][c] == 1:
                            img[r, c] = [0.2, 0.2, 0.2]
                ax.imshow(img, interpolation="nearest")

                tour = result.tour
                cmap = plt.cm.tab10
                for i in range(len(tour) - 1):
                    r1, c1 = tour[i]
                    r2, c2 = tour[i + 1]
                    color = cmap(i % 10)
                    ax.annotate(
                        "",
                        xy=(c2, r2),
                        xytext=(c1, r1),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2),
                    )

                for i, (r, c) in enumerate(waypoints):
                    ax.plot(c, r, "o", color="red", markersize=12, zorder=5)
                    ax.text(c, r, str(i), ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=6)

                ax.set_xticks(np.arange(-0.5, tsp_size, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, tsp_size, 1), minor=True)
                ax.grid(which="minor", color="gray", linewidth=0.3)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Optimal TSP Tour  |  Cost: {result.tour_cost:.1f}", fontsize=14, fontweight="bold")
                st.pyplot(fig)

                c1, c2, c3 = st.columns(3)
                c1.metric("Tour cost", f"{result.tour_cost:.1f}")
                c2.metric("Cities", result.n_cities)
                c3.metric("Runtime", f"{result.runtime * 1000:.2f} ms")

                st.subheader("Tour order")
                tour_str = " -> ".join(str(w) for w in result.tour)
                st.code(tour_str)
