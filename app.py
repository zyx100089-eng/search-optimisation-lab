import streamlit as st

st.set_page_config(
    page_title="Search & Optimisation Lab",
    page_icon="🔍",
    layout="wide",
)

st.title("Search & Optimisation Algorithm Lab")

st.markdown("""
Welcome to the **Search & Optimisation Algorithm Lab** — an interactive tool
for exploring and comparing classical graph-search algorithms and metaheuristics.

### Available Pages

Use the sidebar to navigate between pages:

- **Pathfinding Explorer** — Run BFS, DFS, Dijkstra, A\\*, and Greedy Best-First
  on randomly generated grid mazes.  Watch the algorithm explore the grid and
  compare metrics side by side.

- **Metaheuristics** — Experiment with Genetic Algorithms and Simulated Annealing
  on the same grids.  Tune parameters and observe convergence behaviour.

- **Benchmark Summary** — View precomputed benchmark results across grid sizes
  and obstacle densities.  Compare algorithms on path length, nodes explored,
  and runtime.

- **Dynamic Programming** — Explore Floyd-Warshall (all-pairs shortest paths)
  and Held-Karp (exact TSP) on small grids.  Place waypoints and find the
  provably optimal tour.

### Quick Start

1. Open **Pathfinding Explorer** in the sidebar.
2. Configure the grid size and obstacle density.
3. Select one or more algorithms and click **Run**.
4. Inspect the visualisation and metrics table.
""")
