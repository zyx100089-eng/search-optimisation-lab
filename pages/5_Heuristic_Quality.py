"""Experiment: How does heuristic quality affect A* performance?

Demonstrates:
- Admissible vs inadmissible heuristics
- Effect of heuristic weight on nodes explored, path cost, and runtime
- Formal proof of A* optimality with admissible heuristics
- Failure cases with bad heuristics
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from src.graph import Grid
from src.algorithms.a_star import a_star, weighted_manhattan, manhattan, HEURISTICS
from src.visualisation.grid_view import draw_grid

st.set_page_config(page_title="Heuristic Quality", layout="wide")
st.title("Effect of Heuristic Quality on A*")

tab1, tab2, tab3 = st.tabs([
    "Heuristic Sweep",
    "Bad Heuristics — Failure Cases",
    "Proof of A* Optimality",
])

with tab1:
    st.markdown("""
    This experiment scales the Manhattan heuristic by a weight factor **w** and
    measures how A\\*'s behaviour changes.

    - **w = 0**: h(n) = 0 everywhere → A\\* degenerates into Dijkstra
    - **w = 1**: admissible Manhattan → optimal A\\*
    - **w > 1**: inadmissible → faster but potentially suboptimal
    """)

    with st.sidebar:
        st.header("Sweep Settings")
        rows = st.slider("Grid rows", 10, 60, 30, key="sweep_rows")
        cols = st.slider("Grid cols", 10, 60, 30, key="sweep_cols")
        density = st.slider("Obstacle density", 0.0, 0.4, 0.2, 0.05, key="sweep_density")
        seed = st.number_input("Seed", value=42, step=1, key="sweep_seed")
        w_max = st.slider("Max weight", 2.0, 30.0, 15.0, 1.0, key="sweep_wmax")

    run_sweep = st.button("Run Heuristic Sweep", type="primary")

    if run_sweep:
        grid = Grid.generate_random(rows, cols, density, False, int(seed))
        start, goal = (0, 0), (rows - 1, cols - 1)
        grid.set_passable(*start)
        grid.set_passable(*goal)

        weights = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]
        weights = [w for w in weights if w <= w_max]
        if w_max not in weights:
            weights.append(w_max)

        results = []
        optimal_cost = None
        for w in weights:
            h = weighted_manhattan(w)
            r = a_star(grid, start, goal, heuristic=h)
            results.append({
                "weight": w,
                "path_cost": r.path_cost,
                "nodes_explored": r.explored_count,
                "runtime_ms": r.runtime * 1000,
                "memory_kb": r.peak_memory_bytes / 1024,
                "admissible": w <= 1.0,
            })
            if w == 1.0:
                optimal_cost = r.path_cost

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ws = [r["weight"] for r in results]

        ax = axes[0, 0]
        costs = [r["path_cost"] if r["path_cost"] != float("inf") else 0 for r in results]
        colors = ["green" if r["admissible"] else "red" for r in results]
        ax.bar(range(len(ws)), costs, color=colors)
        ax.set_xticks(range(len(ws)))
        ax.set_xticklabels([f"{w:.1f}" for w in ws], rotation=45)
        ax.set_xlabel("Heuristic weight (w)")
        ax.set_ylabel("Path cost")
        ax.set_title("Path Cost vs Heuristic Weight", fontweight="bold")
        if optimal_cost:
            ax.axhline(optimal_cost, color="blue", linestyle="--", label=f"Optimal = {optimal_cost:.1f}")
            ax.legend()
        ax.grid(axis="y", alpha=0.3)

        ax = axes[0, 1]
        nodes = [r["nodes_explored"] for r in results]
        ax.bar(range(len(ws)), nodes, color=colors)
        ax.set_xticks(range(len(ws)))
        ax.set_xticklabels([f"{w:.1f}" for w in ws], rotation=45)
        ax.set_xlabel("Heuristic weight (w)")
        ax.set_ylabel("Nodes explored")
        ax.set_title("Nodes Explored vs Heuristic Weight", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1, 0]
        runtimes = [r["runtime_ms"] for r in results]
        ax.plot(ws, runtimes, "o-", color="tab:blue", linewidth=2)
        ax.set_xlabel("Heuristic weight (w)")
        ax.set_ylabel("Runtime (ms)")
        ax.set_title("Runtime vs Heuristic Weight", fontweight="bold")
        ax.axvline(1.0, color="gray", linestyle=":", label="Admissible boundary")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        memory = [r["memory_kb"] for r in results]
        ax.plot(ws, memory, "s-", color="tab:purple", linewidth=2)
        ax.set_xlabel("Heuristic weight (w)")
        ax.set_ylabel("Peak memory (KB)")
        ax.set_title("Memory Usage vs Heuristic Weight", fontweight="bold")
        ax.axvline(1.0, color="gray", linestyle=":", label="Admissible boundary")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("**Legend:** 🟢 Admissible (w ≤ 1) — guaranteed optimal. 🔴 Inadmissible (w > 1) — may be suboptimal.")

        if optimal_cost:
            suboptimal = [(r["weight"], r["path_cost"]) for r in results
                          if r["path_cost"] != float("inf") and r["path_cost"] > optimal_cost + 0.01]
            if suboptimal:
                st.warning(f"Suboptimal paths found at weights: {', '.join(f'w={w:.1f} (cost={c:.1f}, +{((c/optimal_cost)-1)*100:.1f}%)' for w, c in suboptimal)}")
            else:
                st.info("All weights produced optimal paths on this grid. Try a larger or denser grid to see suboptimality.")


with tab2:
    st.markdown("""
    ### Bad Heuristics — When A* Fails

    A\\* guarantees optimality **only** with an admissible heuristic (one that never
    overestimates the true cost).  When the heuristic overestimates, A\\* can return
    suboptimal paths because it prunes nodes that it believes are too expensive —
    even though they lie on the true shortest path.

    Below we run A\\* with three heuristics on the same grid:
    - **Manhattan** (admissible, weight = 1) — optimal
    - **5x Manhattan** (inadmissible) — overestimates by 5x
    - **20x Manhattan** (inadmissible) — overestimates by 20x
    """)

    run_bad = st.button("Run Bad Heuristics Demo", type="primary", key="run_bad")

    if run_bad:
        grid = Grid.generate_random(25, 25, 0.2, False, 99)
        start, goal = (0, 0), (24, 24)
        grid.set_passable(*start)
        grid.set_passable(*goal)

        r_good = a_star(grid, start, goal, heuristic="manhattan")
        r_5x = a_star(grid, start, goal, heuristic="inadmissible_5x")
        r_20x = a_star(grid, start, goal, heuristic="inadmissible_20x")

        results = {
            "Manhattan (admissible)": r_good,
            "5x Manhattan (inadmissible)": r_5x,
            "20x Manhattan (inadmissible)": r_20x,
        }

        cols_layout = st.columns(3)
        for col, (name, r) in zip(cols_layout, results.items()):
            with col:
                fig = draw_grid(grid, r.path, r.explored, start, goal, title=name)
                st.pyplot(fig)

        st.subheader("Comparison")
        data = []
        for name, r in results.items():
            if not r.path:
                data.append({
                    "Heuristic": name,
                    "Path Cost": "inf",
                    "Cost Ratio": "N/A",
                    "Nodes Explored": r.explored_count,
                    "Runtime (ms)": round(r.runtime * 1000, 3),
                    "Memory (KB)": round(r.peak_memory_bytes / 1024, 1),
                    "Optimal?": "No path",
                })
            else:
                data.append({
                    "Heuristic": name,
                    "Path Cost": round(r.path_cost, 1),
                    "Cost Ratio": f"{r.path_cost / r_good.path_cost:.2f}x" if r_good.path_cost > 0 else "N/A",
                    "Nodes Explored": r.explored_count,
                    "Runtime (ms)": round(r.runtime * 1000, 3),
                    "Memory (KB)": round(r.peak_memory_bytes / 1024, 1),
                    "Optimal?": "Yes" if abs(r.path_cost - r_good.path_cost) < 0.01 else "No",
                })
        import pandas as pd
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        found_subopt = (r_5x.path and r_5x.path_cost > r_good.path_cost + 0.01) or \
                       (r_20x.path and r_20x.path_cost > r_good.path_cost + 0.01)
        if found_subopt:
            st.error(
                "The inadmissible heuristics returned **suboptimal** paths.  "
                "Notice how the 20x heuristic explores far fewer nodes but at the "
                "cost of path quality — it aggressively prunes promising nodes."
            )
        else:
            st.info(
                "On this particular grid, inadmissible heuristics happened to find "
                "the optimal path.  This doesn't always happen — try different seeds "
                "or denser grids to see failure cases."
            )


with tab3:
    st.markdown(r"""
    ### Proof: A* is Optimal with an Admissible Heuristic

    **Theorem.** If the heuristic $h(n)$ is *admissible* — meaning
    $h(n) \le h^*(n)$ for all nodes $n$, where $h^*(n)$ is the true
    cost from $n$ to the goal — then A\* returns an optimal path.

    **Proof (by contradiction).**

    Suppose A\* terminates and returns a path to goal $G$ with cost
    $C = g(G)$, and suppose this path is **not** optimal — i.e. there
    exists a shorter path with cost $C^* < C$.

    1. **At termination**, there must be some node $n$ on the optimal
       path that is still in the open set (frontier).  This is because
       A\* only terminates when it pops the goal from the priority
       queue, and the optimal path hasn't been fully explored.

    2. **For this node** $n$, since A\* hasn't expanded it yet:
       $$f(n) = g(n) + h(n)$$
       Since $n$ is on the optimal path, $g(n)$ is the true cost from
       start to $n$, and $h^*(n)$ is the remaining cost to the goal.
       Therefore $g(n) + h^*(n) = C^*$.

    3. **By admissibility**: $h(n) \le h^*(n)$, so:
       $$f(n) = g(n) + h(n) \le g(n) + h^*(n) = C^*$$

    4. **But A\* selected the goal** $G$ before $n$, meaning:
       $$f(G) \le f(n)$$
       Since $G$ is the goal, $h(G) = 0$, so $f(G) = g(G) = C$.

    5. **Combining**: $C = f(G) \le f(n) \le C^* < C$, giving $C < C$.

    **Contradiction.** Therefore our assumption was wrong: A\* must
    return the optimal path. $\square$

    ---

    **Experimental verification** is shown in the Heuristic Sweep tab:
    all weights $w \le 1$ (admissible) produce paths with cost equal
    to the optimal, while weights $w > 1$ (inadmissible) may produce
    suboptimal paths.

    **Key insight**: Admissibility is a *sufficient* condition for
    optimality.  An inadmissible heuristic *might* still find the
    optimal path on some instances, but there is no guarantee — and
    the Failure Cases tab demonstrates grids where it fails.
    """)
