"""Run systematic benchmarks across grid sizes and obstacle densities."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
from src.graph import Grid
from src.algorithms.bfs import bfs
from src.algorithms.dfs import dfs
from src.algorithms.dijkstra import dijkstra
from src.algorithms.a_star import a_star
from src.algorithms.greedy_best_first import greedy_best_first

GRID_SIZES = [20, 50, 100]
DENSITIES = [0.1, 0.2, 0.3]
SEEDS = list(range(5))

ALGORITHMS = {
    "BFS": lambda g, s, e: bfs(g, s, e),
    "DFS": lambda g, s, e: dfs(g, s, e),
    "Dijkstra": lambda g, s, e: dijkstra(g, s, e),
    "A*": lambda g, s, e: a_star(g, s, e),
    "Greedy": lambda g, s, e: greedy_best_first(g, s, e),
}

OUTPUT = os.path.join(os.path.dirname(__file__), "results.csv")


def main():
    rows_out = []
    total = len(GRID_SIZES) * len(DENSITIES) * len(SEEDS) * len(ALGORITHMS)
    count = 0

    for size in GRID_SIZES:
        for density in DENSITIES:
            for seed in SEEDS:
                grid = Grid.generate_random(size, size, density, False, seed)
                start = (0, 0)
                goal = (size - 1, size - 1)
                grid.set_passable(*start)
                grid.set_passable(*goal)

                for algo_name, algo_fn in ALGORITHMS.items():
                    result = algo_fn(grid, start, goal)
                    rows_out.append({
                        "grid_size": size,
                        "obstacle_density": density,
                        "seed": seed,
                        "algorithm": algo_name,
                        "path_length": result.path_length,
                        "path_cost": round(result.path_cost, 4),
                        "nodes_explored": result.explored_count,
                        "runtime_ms": round(result.runtime * 1000, 4),
                        "found_path": result.path_length > 0,
                    })
                    count += 1
                    if count % 25 == 0:
                        print(f"  [{count}/{total}]")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Done — {len(rows_out)} runs saved to {OUTPUT}")


if __name__ == "__main__":
    main()
