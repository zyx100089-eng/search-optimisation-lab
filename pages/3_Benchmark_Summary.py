import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Benchmark Summary", layout="wide")
st.title("Benchmark Summary")

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "experiments", "results.csv")

if not os.path.exists(CSV_PATH):
    st.info(
        "No benchmark results found.  Run the benchmarking script first:\n\n"
        "```bash\npython experiments/run_benchmarks.py\n```"
    )
    st.stop()

df = pd.read_csv(CSV_PATH)
st.subheader("Raw Results")
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Filters")
sizes = sorted(df["grid_size"].unique())
selected_sizes = st.multiselect("Grid sizes", sizes, default=sizes)
densities = sorted(df["obstacle_density"].unique())
selected_densities = st.multiselect("Obstacle densities", densities, default=densities)

filtered = df[df["grid_size"].isin(selected_sizes) & df["obstacle_density"].isin(selected_densities)]

agg_cols = {
    "avg_path_cost": ("path_cost", "mean"),
    "avg_nodes_explored": ("nodes_explored", "mean"),
    "avg_runtime_ms": ("runtime_ms", "mean"),
    "path_found_pct": ("found_path", "mean"),
}
if "peak_memory_kb" in filtered.columns:
    agg_cols["avg_memory_kb"] = ("peak_memory_kb", "mean")

agg = filtered.groupby("algorithm").agg(**agg_cols).reset_index()
agg["path_found_pct"] = (agg["path_found_pct"] * 100).round(1)
agg = agg.round(2)

st.subheader("Aggregated Metrics")
st.dataframe(agg, use_container_width=True, hide_index=True)

metrics = ["avg_path_cost", "avg_nodes_explored", "avg_runtime_ms"]
titles = ["Average Path Cost", "Average Nodes Explored", "Average Runtime (ms)"]
if "avg_memory_kb" in agg.columns:
    metrics.append("avg_memory_kb")
    titles.append("Average Peak Memory (KB)")

n_charts = len(metrics)
fig, axes = plt.subplots(1, n_charts, figsize=(5 * n_charts, 5))
if n_charts == 1:
    axes = [axes]
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
for ax, metric, title in zip(axes, metrics, titles):
    bars = ax.bar(agg["algorithm"], agg[metric], color=colors[:len(agg)])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric.replace("avg_", "").replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
