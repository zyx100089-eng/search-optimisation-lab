# Search and Optimisation Algorithm Lab — Report

## 1. Introduction

This project implements, visualises, and benchmarks five classical graph-search algorithms and two metaheuristic optimisers.  The goal is to understand the practical trade-offs between uninformed search, informed search, and approximate optimisation by running controlled experiments on randomly generated grid mazes.

The algorithms studied are:

| Category | Algorithms |
|---|---|
| Uninformed search | Breadth-First Search (BFS), Depth-First Search (DFS) |
| Informed search | Dijkstra's algorithm, A\*, Greedy Best-First Search |
| Metaheuristics | Genetic Algorithm (GA), Simulated Annealing (SA) |

All implementations are from scratch in Python, with an interactive Streamlit visualiser for exploration and a benchmarking harness for systematic comparison.

## 2. Algorithm Background

### 2.1 Breadth-First Search (BFS)

BFS explores all nodes at the current depth before moving deeper.  It uses a FIFO queue and guarantees the shortest path on unweighted graphs.

- **Time complexity:** O(V + E)
- **Space complexity:** O(V)
- **Optimality:** Yes (unweighted graphs)

### 2.2 Depth-First Search (DFS)

DFS explores as deep as possible along each branch before backtracking.  It uses a LIFO stack and does not guarantee shortest paths.

- **Time complexity:** O(V + E)
- **Space complexity:** O(V) worst case
- **Optimality:** No

### 2.3 Dijkstra's Algorithm

Dijkstra's algorithm extends BFS to weighted graphs by using a priority queue keyed on accumulated cost g(n).  It always expands the cheapest-known node and guarantees optimal paths with non-negative weights.

- **Time complexity:** O((V + E) log V) with a binary heap
- **Space complexity:** O(V)
- **Optimality:** Yes (non-negative weights)

### 2.4 A\* Search

A\* combines Dijkstra's accumulated cost g(n) with a heuristic estimate h(n) of the remaining cost, using f(n) = g(n) + h(n) as the priority.  With an admissible heuristic (one that never overestimates), A\* is optimal and typically explores far fewer nodes than Dijkstra.

- **Time complexity:** O((V + E) log V), but with pruning from the heuristic
- **Space complexity:** O(V)
- **Optimality:** Yes (with admissible heuristic)

Three heuristics are implemented:
- **Manhattan distance:** |x1 - x2| + |y1 - y2| — admissible for 4-directional grids
- **Euclidean distance:** sqrt((x1 - x2)^2 + (y1 - y2)^2) — admissible for all grids
- **Chebyshev distance:** max(|x1 - x2|, |y1 - y2|) — admissible for 8-directional grids

### 2.5 Greedy Best-First Search

Greedy Best-First uses only h(n) as the priority, ignoring accumulated cost.  It tends to reach the goal quickly but may produce suboptimal or even long, winding paths.

- **Time complexity:** O((V + E) log V)
- **Space complexity:** O(V)
- **Optimality:** No

### 2.6 Genetic Algorithm (GA)

A population-based metaheuristic that encodes candidate paths as sequences of movement directions.  Each generation applies tournament selection, single-point crossover, and per-gene mutation.  Fitness is based on proximity to the goal and path length.

- **Time complexity:** O(generations x population_size x path_length)
- **Optimality:** No guarantee; approximate

### 2.7 Simulated Annealing (SA)

A neighbourhood-search method that starts from a greedy initial path and iteratively perturbs it.  Worse solutions are accepted with probability exp(-delta/T), where T decreases over time following a geometric cooling schedule.

- **Time complexity:** O(iterations x path_length)
- **Optimality:** No guarantee; approximate

### 2.8 Floyd-Warshall Algorithm

An all-pairs shortest paths algorithm that computes the optimal distance between every pair of nodes simultaneously.  It uses a three-nested-loop relaxation over intermediate nodes.

- **Time complexity:** O(V^3)
- **Space complexity:** O(V^2)
- **Optimality:** Yes (exact)

Floyd-Warshall is practical for small grids (up to ~15x15, i.e. ~225 nodes) but becomes prohibitively expensive for larger instances.  Its advantage is that once computed, any source-target query is answered in O(V) time (path reconstruction) rather than requiring a new search.

### 2.9 Held-Karp Algorithm (Exact TSP)

The Held-Karp algorithm solves the Travelling Salesman Problem exactly using dynamic programming with bitmask subsets.  It considers all possible subsets of cities and finds the minimum-cost Hamiltonian cycle.

- **Time complexity:** O(2^n * n^2)
- **Space complexity:** O(2^n * n)
- **Optimality:** Yes (exact)

This is practical for up to ~20 cities.  For the grid TSP variant, Floyd-Warshall first computes pairwise shortest distances between waypoints, then Held-Karp finds the optimal tour.

## 3. Implementation Design

### 3.1 Grid Representation

The `Grid` class models a 2D maze where each cell is passable (0) or blocked (1).  It supports:
- Configurable rows, columns, and obstacle density
- Optional per-cell weights for weighted pathfinding
- 4-directional or 8-directional (diagonal) movement
- Seeded random generation for reproducibility

Neighbours are computed on the fly with bounds and passability checks.  Diagonal moves incur a sqrt(2) multiplier on the base weight.

### 3.2 Algorithm Modules

Each algorithm lives in its own module under `src/algorithms/` and returns a dataclass containing:
- The path (list of (row, col) tuples)
- The list of explored nodes (in exploration order)
- Metrics: path length, path cost, nodes explored, runtime

This uniform interface allows the visualiser and benchmarking harness to treat all algorithms identically.

### 3.3 Visualiser

The Streamlit app provides three pages:

1. **Pathfinding Explorer** — configurable grid with multi-algorithm comparison, side-by-side grid plots, and a metrics table with summary indicators (lowest cost, fastest, fewest nodes).
2. **Metaheuristics** — GA and SA with tunable parameters alongside an A\* baseline, plus convergence plots (fitness over generations for GA, cost over iterations for SA).
3. **Benchmark Summary** — loads precomputed CSV results, provides interactive filters by grid size and density, and displays aggregated bar charts.
4. **Dynamic Programming** — two tabs: Floyd-Warshall computes all-pairs shortest paths on small grids and compares against A\*; Held-Karp solves exact TSP on user-placed waypoints, visualised with numbered cities and directional arrows showing the optimal tour.

Grid visualisations use Matplotlib with colour coding: dark grey for obstacles, light blue for explored cells, blue for the path, green for start, red for goal.

## 4. Experimental Methodology

### 4.1 Scenarios

Benchmarks were run across a factorial design:
- **Grid sizes:** 20x20, 50x50, 100x100
- **Obstacle densities:** 10%, 20%, 30%
- **Random seeds:** 0 through 4 (5 trials per configuration)

Start is always (0, 0) and goal is (n-1, n-1).  All grids use unit weights and 4-directional movement.

### 4.2 Algorithms Tested

BFS, DFS, Dijkstra, A\* (Manhattan heuristic), and Greedy Best-First (Manhattan heuristic).

### 4.3 Metrics

- **Path length:** number of cells in the returned path
- **Path cost:** sum of edge weights along the path
- **Nodes explored:** number of nodes removed from the frontier
- **Runtime:** wall-clock time via `time.perf_counter()`, reported in milliseconds
- **Path found:** whether the algorithm reached the goal

### 4.4 Total Runs

3 sizes x 3 densities x 5 seeds x 5 algorithms = **225 runs**.

## 5. Results

### 5.1 Overall Performance

| Algorithm | Avg Path Cost | Avg Nodes Explored | Avg Runtime (ms) | Path Found (%) |
|---|---|---|---|---|
| A\* | 88.58 | 2,020 | 10.72 | 80.0 |
| BFS | 89.38 | 2,878 | 13.45 | 80.0 |
| Dijkstra | 88.58 | 2,879 | 19.87 | 80.0 |
| Greedy | 98.76 | 272 | 1.18 | 80.0 |
| DFS | 800.98 | 1,542 | 29.85 | 80.0 |

![Overall comparison](fig1_overall_comparison.png)

**Key observations:**
- A\* and Dijkstra produce identical optimal costs (88.58), confirming correctness.
- A\* explores 30% fewer nodes than Dijkstra (2,020 vs 2,879), demonstrating the value of the Manhattan heuristic.
- BFS matches Dijkstra's node count (expected on unit-weight grids) but is faster due to simpler queue operations.
- Greedy explores dramatically fewer nodes (272) but at a 12% cost penalty (98.76 vs 88.58).
- DFS produces paths ~9x longer than optimal, confirming it is unsuitable for shortest-path problems.

### 5.2 Scaling with Grid Size

![Nodes explored by grid size](fig2_nodes_by_size.png)

| Algorithm | 20x20 Nodes | 50x50 Nodes | 100x100 Nodes |
|---|---|---|---|
| A\* | 198 | 1,282 | 4,581 |
| BFS | 253 | 1,879 | 6,502 |
| Dijkstra | 253 | 1,880 | 6,502 |
| Greedy | 54 | 123 | 637 |
| DFS | 154 | 1,035 | 3,437 |

Greedy's node count grows much more slowly than the others because it follows the heuristic directly toward the goal.  A\*'s advantage over Dijkstra/BFS becomes more pronounced at larger sizes: at 100x100, A\* explores 30% fewer nodes.

### 5.3 Effect of Obstacle Density

![Path discovery rate by density](fig3_found_rate_density.png)

All algorithms have identical path-found rates (since they all explore the same reachable set):
- 10% density: 93.3% found
- 20% density: 86.7% found
- 30% density: 60.0% found

Higher density increases the chance that no path exists between opposite corners.  When paths do exist at 30% density, they tend to be shorter (avg cost 61 vs 109 at 10%) because the grid has fewer passable cells overall.

### 5.4 Runtime Scaling

![Runtime scaling](fig4_runtime_scaling.png)

| Algorithm | 20x20 (ms) | 50x50 (ms) | 100x100 (ms) |
|---|---|---|---|
| A\* | 0.58 | 5.02 | 26.56 |
| BFS | 0.58 | 6.48 | 33.30 |
| Dijkstra | 0.73 | 8.01 | 50.87 |
| Greedy | 0.16 | 0.51 | 2.87 |
| DFS | 0.35 | 5.45 | 83.75 |

Dijkstra's overhead from the priority queue becomes visible at scale: it is 50% slower than BFS at 100x100 despite exploring the same nodes, because heap operations are more expensive than deque operations.  DFS is the slowest at 100x100 (83.75 ms) because it explores long, wasteful paths.

## 6. Discussion

### 6.1 Optimality vs Speed

The results confirm the classical trade-off between optimality and computational cost:

- **A\*** is the best all-round choice: optimal paths, moderate exploration, and reasonable runtime.  The Manhattan heuristic provides meaningful pruning that compounds with grid size.
- **Greedy** is 9x faster than A\* on average and explores 7x fewer nodes, making it attractive when approximate solutions suffice.  Its 12% cost penalty is modest on simple grids but can be worse on mazes with complex obstacle layouts.
- **BFS** is simple and optimal on unweighted grids, and faster than Dijkstra due to its O(1) queue operations.  For unit-cost grids, there is little reason to use Dijkstra over BFS.
- **Dijkstra** is necessary when edges have varying weights.  On unit-weight grids it behaves identically to BFS but incurs heap overhead.
- **DFS** is the worst performer for pathfinding: paths are nearly an order of magnitude longer than optimal.  Its only advantage is low space consumption (stack depth vs full frontier).

### 6.2 Heuristic Quality

A\*'s efficiency depends entirely on heuristic quality.  Manhattan distance is a tight lower bound for 4-directional grids, which explains why A\* explores 30% fewer nodes than Dijkstra.  With a less informative heuristic (e.g. Euclidean on a 4-directional grid), this gap narrows.

### 6.3 Metaheuristics

GA and SA were implemented as optional extensions.  On small grids (15x15 with 20% obstacles), both can find near-optimal paths, but they are orders of magnitude slower than A\* due to the large number of evaluations required.  They are more interesting for combinatorial optimisation problems where exact methods are intractable.

### 6.4 Dynamic Programming

Floyd-Warshall and Held-Karp demonstrate the power and limitations of exact DP methods:

- **Floyd-Warshall** produces identical costs to A\* on all tested grids (verified in unit tests), confirming both implementations are correct.  Its O(V^3) cost makes it impractical beyond ~15x15 grids (~225 nodes), but once computed, any pair query is instant.  This is useful when many source-target queries are needed on the same graph.
- **Held-Karp** solves TSP exactly, which no search algorithm or metaheuristic can guarantee.  On a 10x10 grid with 6 waypoints, it finds the optimal tour in under 1 ms.  However, the O(2^n * n^2) scaling limits it to ~20 cities — at 20 cities the bitmask has over 1 million states.  For larger instances, the metaheuristic approaches (GA, SA) become the only viable option, illustrating the complementary relationship between exact and approximate methods.

## 7. Conclusion

This lab demonstrates the practical differences between search algorithms through hands-on implementation and empirical evaluation.  The main findings are:

1. **A\* with an admissible heuristic is the gold standard** for grid pathfinding: optimal, efficient, and the heuristic's benefit grows with problem size.
2. **Greedy Best-First is a viable fast approximation** when optimality is not required, trading a modest cost increase for dramatically fewer explored nodes.
3. **BFS outperforms Dijkstra on unweighted grids** due to simpler data structures, despite theoretical equivalence.
4. **DFS is unsuitable for shortest-path problems** but illustrates the importance of exploration strategy.
5. **Obstacle density affects solvability more than algorithm choice**: all algorithms find or fail to find paths at the same rates.
6. **Exact DP methods (Floyd-Warshall, Held-Karp) provide provably optimal solutions** but are constrained by polynomial and exponential scaling respectively, motivating the use of metaheuristics for larger problem instances.

### Future Work

- Test on weighted grids to see Dijkstra's advantage over BFS
- Implement bidirectional search and iterative deepening DFS
- Apply GA and SA to combinatorial problems (TSP, graph colouring) where exact methods scale poorly
- Experiment with inadmissible heuristics and weighted A\* (f = g + w*h) to explore the speed-optimality frontier
