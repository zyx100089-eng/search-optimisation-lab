import streamlit as st
from src.graph import Grid
from src.algorithms.bfs import bfs
from src.algorithms.dfs import dfs
from src.algorithms.dijkstra import dijkstra
from src.algorithms.a_star import a_star, HEURISTICS
from src.algorithms.greedy_best_first import greedy_best_first
from src.visualisation.grid_view import draw_grid
from src.visualisation.metrics import compare_results

st.set_page_config(page_title="Pathfinding Explorer", layout="wide")
st.title("Pathfinding Explorer")

ALGORITHMS = {
    "BFS": lambda g, s, e, **kw: bfs(g, s, e),
    "DFS": lambda g, s, e, **kw: dfs(g, s, e),
    "Dijkstra": lambda g, s, e, **kw: dijkstra(g, s, e),
    "A*": lambda g, s, e, **kw: a_star(g, s, e, heuristic=kw.get("heuristic", "manhattan")),
    "Greedy Best-First": lambda g, s, e, **kw: greedy_best_first(g, s, e, heuristic=kw.get("heuristic", "manhattan")),
}

with st.sidebar:
    st.header("Grid Settings")
    rows = st.slider("Rows", 5, 100, 20)
    cols = st.slider("Columns", 5, 100, 20)
    density = st.slider("Obstacle Density", 0.0, 0.5, 0.2, 0.05)
    weighted = st.checkbox("Weighted grid", value=False)
    diagonal = st.checkbox("Allow diagonal movement", value=False)
    seed = st.number_input("Random seed", value=42, step=1)

    st.header("Start / Goal")
    start_r = st.number_input("Start row", 0, rows - 1, 0)
    start_c = st.number_input("Start col", 0, cols - 1, 0)
    goal_r = st.number_input("Goal row", 0, rows - 1, rows - 1)
    goal_c = st.number_input("Goal col", 0, cols - 1, cols - 1)

    st.header("Algorithm")
    selected = st.multiselect("Choose algorithms", list(ALGORITHMS.keys()), default=["BFS", "A*"])
    heuristic_name = st.selectbox("Heuristic (for A*/Greedy)", list(HEURISTICS.keys()))

    run = st.button("Run", type="primary", use_container_width=True)

if run and selected:
    grid = Grid.generate_random(rows, cols, density, diagonal, int(seed), weighted)
    start = (int(start_r), int(start_c))
    goal = (int(goal_r), int(goal_c))
    grid.set_passable(*start)
    grid.set_passable(*goal)

    results = {}
    cols_layout = st.columns(min(len(selected), 3))

    for i, name in enumerate(selected):
        algo_fn = ALGORITHMS[name]
        result = algo_fn(grid, start, goal, heuristic=heuristic_name)
        results[name] = result

        col = cols_layout[i % len(cols_layout)]
        with col:
            fig = draw_grid(grid, result.path, result.explored, start, goal, title=name)
            st.pyplot(fig)

    st.subheader("Comparison")
    df = compare_results(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    best_algo = df.loc[df["Path Cost"].idxmin(), "Algorithm"] if df["Found Path"].any() else "N/A"
    fastest = df.loc[df["Runtime (ms)"].idxmin(), "Algorithm"]
    fewest_nodes = df.loc[df["Nodes Explored"].idxmin(), "Algorithm"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Lowest Cost", best_algo)
    c2.metric("Fastest", fastest)
    c3.metric("Fewest Nodes", fewest_nodes)
elif run:
    st.warning("Select at least one algorithm.")
