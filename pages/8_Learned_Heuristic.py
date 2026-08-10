"""Learned Heuristic for A* — the machine-learning page.

Demonstrates a supervised-learning approach to heuristic design:
  1. Label many (node, goal) pairs with the true remaining cost h*(n) via Dijkstra.
  2. Extract geometric features for each node.
  3. Train a linear model h(n) = theta . x(n) by gradient descent.
  4. Use the learned heuristic inside A* and measure the speed-up vs the
     optimality trade-off, comparing against Manhattan and Dijkstra.

This mirrors a real research theme in AI planning: learned heuristics can
reduce search effort but trade guaranteed optimality for empirical speed —
the same trade-off that weighted A* formalises with its epsilon bound.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.graph import Grid
from src.algorithms.dijkstra import dijkstra
from src.algorithms.a_star import a_star
from src.algorithms.learned_heuristic import (
    generate_training_data,
    train_learned_heuristic,
    make_learned_heuristic,
    FEATURE_NAMES,
)
from src.visualisation.grid_view import draw_grid

st.set_page_config(page_title="Learned Heuristic", layout="wide")
st.title("Learned Heuristic for A*")
st.markdown("""
Instead of a hand-crafted Manhattan heuristic, we **learn** one from data.
Dijkstra provides the ground-truth remaining cost h\*(n) for sampled nodes;
a linear model is fit to geometric features and used as A\*'s heuristic.
The learned heuristic is **not** guaranteed admissible, so the search becomes
a *bounded-suboptimal* planner — faster, but occasionally returning a slightly
suboptimal path.  This is the same trade-off that weighted A\* formalises.
""")

tab1, tab2, tab3 = st.tabs(["Train", "Single-Grid Comparison", "Scaling Evaluation"])

with tab1:
    with st.sidebar:
        st.header("Training Data")
        t_rows = st.slider("Train grid rows", 10, 30, 20, key="lh_trows")
        t_cols = st.slider("Train grid cols", 10, 30, 20, key="lh_tcols")
        t_density = st.slider("Train density", 0.0, 0.4, 0.2, 0.05, key="lh_tden")
        t_n_grids = st.slider("Number of training grids", 10, 80, 40, key="lh_ngrids")
        t_spg = st.slider("Samples per grid", 10, 50, 25, key="lh_spg")

        st.header("Model")
        m_epochs = st.slider("Epochs", 50, 500, 200, key="lh_ep")
        m_lr = st.slider("Learning rate", 0.001, 0.1, 0.01, 0.005, key="lh_lr")

    train_btn = st.button("Generate Data & Train", type="primary", key="lh_train")

    if "lh_train_result" not in st.session_state:
        st.session_state["lh_train_result"] = None
    if "lh_samples" not in st.session_state:
        st.session_state["lh_samples"] = None

    if train_btn:
        with st.status("Generating training data...") as status:
            samples = generate_training_data(
                n_grids=t_n_grids, rows=t_rows, cols=t_cols,
                density=t_density, seed=0, samples_per_grid=t_spg,
            )
            st.session_state["lh_samples"] = samples
            status.update(label=f"Generated {len(samples)} labelled samples", state="complete")

        with st.status("Training linear model...") as status:
            tr = train_learned_heuristic(samples, epochs=m_epochs, lr=m_lr, seed=0)
            st.session_state["lh_train_result"] = tr
            status.update(label=f"Trained — final MSE {tr.train_loss_history[-1]:.4f}", state="complete")

    tr = st.session_state["lh_train_result"]
    samples = st.session_state["lh_samples"]

    if tr is None:
        st.info("Adjust the settings in the sidebar and click **Generate Data & Train**.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(tr.train_loss_history, color="tab:blue", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE (standardised)")
            ax.set_title("Training Loss", fontweight="bold")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

        with c2:
            weights = tr.model.theta
            colors = ["tab:green" if w >= 0 else "tab:red" for w in weights]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(range(len(weights)), weights, color=colors)
            ax.set_yticks(range(len(weights)))
            ax.set_yticklabels(FEATURE_NAMES)
            ax.set_xlabel("Learned weight (standardised scale)")
            ax.set_title("Model Weights", fontweight="bold")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

        st.subheader("Interpretation")
        top = np.argsort(np.abs(weights))[::-1][:3]
        parts = [f"**{FEATURE_NAMES[i]}** ({weights[i]:+.3f})" for i in top]
        st.markdown(
            "The most influential features are: "
            + ", ".join(parts)
            + ".  A positive weight on Manhattan/Euclidean distance is expected "
            "(h\\* grows with distance to goal); obstacle-density features let "
            "the model anticipate detours that pure distance misses."
        )

with tab2:
    if tr is None:
        st.warning("Train a model in the **Train** tab first.")
    else:
        with st.sidebar:
            st.header("Test Grid")
            e_rows = st.slider("Rows", 10, 40, 20, key="lh_erows")
            e_cols = st.slider("Cols", 10, 40, 20, key="lh_ecols")
            e_density = st.slider("Density", 0.0, 0.4, 0.2, 0.05, key="lh_eden")
            e_seed = st.number_input("Seed", value=99, step=1, key="lh_eseed")

        run_eval = st.button("Run Comparison", type="primary", key="lh_eval")

        if run_eval:
            grid = Grid.generate_random(e_rows, e_cols, e_density, False, int(e_seed))
            start, goal = (0, 0), (e_rows - 1, e_cols - 1)
            grid.set_passable(*start)
            grid.set_passable(*goal)

            r_dij = dijkstra(grid, start, goal)
            r_man = a_star(grid, start, goal, heuristic="manhattan")
            h_learned = make_learned_heuristic(tr.model, grid)
            r_learn = a_star(grid, start, goal, heuristic=h_learned)

            results = {
                "Dijkstra (optimal)": r_dij,
                "A* + Manhattan": r_man,
                "A* + Learned": r_learn,
            }

            cols_layout = st.columns(3)
            for col, (name, r) in zip(cols_layout, results.items()):
                with col:
                    fig = draw_grid(grid, r.path, r.explored, start, goal, title=name)
                    st.pyplot(fig)

            st.subheader("Metrics")
            data = []
            opt_cost = r_dij.path_cost if r_dij.path else float("inf")
            for name, r in results.items():
                if not r.path:
                    data.append({
                        "Algorithm": name,
                        "Path Cost": "inf",
                        "Suboptimality": "N/A",
                        "Nodes Explored": r.explored_count,
                        "Runtime (ms)": round(r.runtime * 1000, 3),
                        "Optimal?": "No path",
                    })
                else:
                    data.append({
                        "Algorithm": name,
                        "Path Cost": round(r.path_cost, 2),
                        "Suboptimality": f"{r.path_cost / opt_cost:.2f}x",
                        "Nodes Explored": r.explored_count,
                        "Runtime (ms)": round(r.runtime * 1000, 3),
                        "Optimal?": "Yes" if abs(r.path_cost - opt_cost) < 0.01 else "No",
                    })
            import pandas as pd
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

            saved_vs_man = r_man.explored_count - r_learn.explored_count
            if not r_learn.path:
                st.error("A* + Learned found no path on this grid.")
            elif saved_vs_man > 0:
                st.success(
                    f"A* + Learned explored **{saved_vs_man} fewer nodes** than A* + Manhattan "
                    f"({saved_vs_man / max(r_man.explored_count, 1) * 100:.0f}% reduction)."
                )
            elif saved_vs_man < 0:
                st.warning(
                    f"A* + Learned explored {-saved_vs_man} more nodes than Manhattan — "
                    f"the model under-fit this grid (try more training data or a denser train set)."
                )
            if r_learn.path and abs(r_learn.path_cost - opt_cost) > 0.01:
                st.error(
                    f"The learned heuristic is inadmissible here: path cost is "
                    f"{r_learn.path_cost / opt_cost:.2f}x optimal.  This is the bounded-suboptimal "
                    f"trade-off — speed gained, optimality lost."
                )

with tab3:
    if tr is None:
        st.warning("Train a model in the **Train** tab first.")
    else:
        st.markdown("""
        Evaluate the learned heuristic across many unseen grids and plot the
        trade-off: **node reduction** vs **suboptimality ratio**.  Each point
        is one test grid.  A perfect learned heuristic would sit at the
        bottom-left (0% extra nodes, 1.00x optimal).
        """)
        with st.sidebar:
            st.header("Evaluation")
            n_test = st.slider("Test grids", 5, 30, 15, key="lh_ntest")
            s_rows = st.slider("Test rows", 10, 40, 20, key="lh_srows")
            s_cols = st.slider("Test cols", 10, 40, 20, key="lh_scols")
            s_density = st.slider("Test density", 0.0, 0.4, 0.2, 0.05, key="lh_sden")
            seed_offset = st.number_input("Seed offset", value=100, step=1, key="lh_soff")

        run_scale = st.button("Run Scaling Evaluation", type="primary", key="lh_scale")

        if run_scale:
            progress = st.progress(0)
            records = []
            for i in range(n_test):
                grid = Grid.generate_random(s_rows, s_cols, s_density, False, int(seed_offset) + i)
                start, goal = (0, 0), (s_rows - 1, s_cols - 1)
                grid.set_passable(*start)
                grid.set_passable(*goal)

                r_dij = dijkstra(grid, start, goal)
                r_man = a_star(grid, start, goal, heuristic="manhattan")
                h = make_learned_heuristic(tr.model, grid)
                r_learn = a_star(grid, start, goal, heuristic=h)

                if r_dij.path and r_man.path and r_learn.path:
                    records.append({
                        "seed": int(seed_offset) + i,
                        "man_nodes": r_man.explored_count,
                        "learn_nodes": r_learn.explored_count,
                        "man_cost": r_man.path_cost,
                        "learn_cost": r_learn.path_cost,
                        "opt_cost": r_dij.path_cost,
                        "node_reduction_pct": (r_man.explored_count - r_learn.explored_count) / r_man.explored_count * 100,
                        "subopt": r_learn.path_cost / r_dij.path_cost,
                    })
                progress.progress((i + 1) / n_test)

            if not records:
                st.warning("No solvable grids in this range — try a lower density.")
            else:
                import pandas as pd
                df = pd.DataFrame(records)

                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(df["node_reduction_pct"], df["subopt"], s=60, alpha=0.7, c="tab:blue", edgecolors="navy")
                    ax.axhline(1.0, color="green", linestyle="--", linewidth=1, label="Optimal (1.00x)")
                    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
                    ax.set_xlabel("Node reduction vs Manhattan (%)")
                    ax.set_ylabel("Suboptimality ratio (path cost / optimal)")
                    ax.set_title("Learned Heuristic Trade-off", fontweight="bold")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)

                with c2:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
                    ax1.bar(["Manhattan", "Learned"],
                            [df["man_nodes"].mean(), df["learn_nodes"].mean()],
                            color=["#3498db", "#e67e22"])
                    ax1.set_ylabel("Avg nodes explored")
                    ax1.set_title("Search Effort", fontweight="bold")
                    ax1.grid(axis="y", alpha=0.3)

                    ax2.bar(["Manhattan", "Learned"],
                            [df["man_cost"].mean() / df["opt_cost"].mean(),
                             df["learn_cost"].mean() / df["opt_cost"].mean()],
                            color=["#3498db", "#e67e22"])
                    ax2.set_ylabel("Avg cost / optimal")
                    ax2.set_title("Path Quality", fontweight="bold")
                    ax2.axhline(1.0, color="green", linestyle="--", linewidth=1)
                    ax2.grid(axis="y", alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)

                st.subheader("Summary")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Avg node reduction", f"{df['node_reduction_pct'].mean():.1f}%")
                mc2.metric("Avg suboptimality", f"{df['subopt'].mean():.3f}x")
                mc3.metric("Optimal rate", f"{(df['subopt'] <= 1.001).mean() * 100:.0f}%")