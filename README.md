# Search & Optimisation Algorithm Lab

An interactive lab for implementing, visualising, and benchmarking classical
graph-search algorithms and metaheuristic optimisers.

## Algorithms

| Category | Algorithms |
|---|---|
| **Uninformed Search** | BFS, DFS |
| **Informed Search** | Dijkstra, A\*, Greedy Best-First |
| **Metaheuristics** | Genetic Algorithm, Simulated Annealing |
| **Dynamic Programming** | Floyd-Warshall (all-pairs shortest paths), Held-Karp (exact TSP) |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the Visualiser

```bash
streamlit run app.py
```

Navigate between pages using the sidebar:
- **Pathfinding Explorer** — compare BFS / DFS / Dijkstra / A\* / Greedy on random grids
- **Metaheuristics** — GA and SA with tunable parameters and convergence plots
- **Benchmark Summary** — aggregated metrics from systematic experiments
- **Dynamic Programming** — Floyd-Warshall all-pairs shortest paths and Held-Karp exact TSP solver

## Run Benchmarks

```bash
python experiments/run_benchmarks.py
```

This generates `experiments/results.csv`, which the Benchmark Summary page reads.

## Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
search_optimisation_lab/
├── app.py                    # Streamlit entry point
├── src/
│   ├── graph.py              # Grid and Graph representations
│   └── algorithms/           # BFS, DFS, Dijkstra, A*, Greedy, GA, SA, Floyd-Warshall, Held-Karp
│   └── visualisation/        # Grid drawing and metrics helpers
├── pages/                    # Streamlit multi-page modules
├── experiments/              # Benchmarking scripts
├── tests/                    # pytest test suite
└── report/                   # Report documents and figures
```
