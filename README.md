# Search & Optimisation Algorithm Lab

[![Tests](https://github.com/zyx100089-eng/search-optimisation-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/zyx100089-eng/search-optimisation-lab/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

> **Live demo:** [linear-algebra-optimisation-visualiser-cufqkcdc6ycv62hixumjva.streamlit.app](https://linear-algebra-optimisation-visualiser-cufqkcdc6ycv62hixumjva.streamlit.app)

An interactive lab for implementing, visualising, and benchmarking classical
graph-search algorithms, metaheuristic optimisers, exact dynamic-programming
solvers, and a **learned (machine-learning) heuristic for A\***.

![Three algorithms on the same 25x25 grid](docs/showcase.png)

*Dijkstra (optimal, 185 nodes) | A\* with Manhattan (optimal, 132 nodes) | Greedy Best-First (suboptimal, 54 nodes) — same grid, same start/goal, three very different exploration patterns.*

## Algorithms

| Category | Algorithms |
|---|---|
| **Uninformed Search** | BFS, DFS |
| **Informed Search** | Dijkstra, A\*, Greedy Best-First |
| **Metaheuristics** | Genetic Algorithm, Simulated Annealing |
| **Dynamic Programming** | Floyd-Warshall (all-pairs shortest paths), Held-Karp (exact TSP) |
| **Machine Learning** | Learned heuristic via linear regression on h\*(n) |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[test]"
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
- **Heuristic Quality** — heuristic-weight sweep, inadmissible failure cases, and a formal proof of A\* optimality
- **Dijkstra vs A\*** — side-by-side exploration patterns and a scaling experiment
- **Weighted Graphs** — algorithms on general weighted graph topologies (not grids)
- **Learned Heuristic** — train a linear model on Dijkstra-labelled data and use it as an A\* heuristic; analyse the speed-up vs suboptimality trade-off

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
├── pyproject.toml            # Package metadata & test config
├── src/
│   ├── graph.py              # Grid and Graph representations
│   ├── algorithms/           # BFS, DFS, Dijkstra, A*, Greedy, GA, SA,
│   │                         # Floyd-Warshall, Held-Karp, learned heuristic
│   └── visualisation/        # Grid drawing and metrics helpers
├── pages/                    # Streamlit multi-page modules
├── experiments/              # Benchmarking scripts + results.csv
├── tests/                    # pytest test suite (39 tests)
├── report/                   # Report document and figures
└── .github/workflows/        # CI: runs pytest on push
```

## Overview of the Learned Heuristic

The headline experiment trains a linear model to predict the *true remaining
cost* `h*(n)` from lightweight geometric features (Manhattan/Euclidean distance,
local obstacle density, degree, distance to nearest wall, goal alignment).
Labels come from Dijkstra.  The model is trained from scratch with mini-batch
gradient descent (no scikit-learn) so the mathematics is explicit.

Because the learned heuristic is **not** guaranteed admissible, the resulting
search is a *bounded-suboptimal* planner — the same regime as weighted A\*.  The
**Learned Heuristic** page measures the resulting speed-up versus the
suboptimality ratio across many unseen grids, illustrating the core AI-planning
trade-off between search effort and solution quality.

![Learned heuristic: speed vs optimality](report/fig5_learned_heuristic_tradeoff.png)

*Each point is one unseen 20×20 grid: how much search effort the
learned heuristic saves over Manhattan (x-axis) versus how much
solution quality it costs (y-axis).*

## Why this lab, and what I learned

I wanted to see classical search algorithms as one family — uninformed,
informed, metaheuristic, exact — rather than ten isolated textbook chapters.
The visualiser was the tool that made the differences visible: the same grid,
three exploration patterns that you can stare at (Dijkstra's frontier, A\*'s
directed cone, Greedy's narrow sprint). The learned-heuristic experiment came
out of a question the visualiser raised: A\* with Manhattan is great on empty
grids and blind in mazes — so can the *structure of the grid itself* predict
the remaining cost better than a hand-picked heuristic? The answer in my
experiments was: sometimes, at the price of admissibility — which is exactly
the trade-off the lab's final page measures.

The honest limit of this project is breadth: ten algorithms implemented well
rather than one studied deeply. The learned-heuristic experiment is the part
I'd defend, and the one I'd take further.