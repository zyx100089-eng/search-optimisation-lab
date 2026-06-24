from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass
class GAResult:
    path: list[tuple[int, int]]
    explored: list[tuple[int, int]]
    explored_count: int
    path_length: int
    path_cost: float
    runtime: float
    best_fitness_history: list[float]


def genetic_algorithm(
    grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    population_size: int = 200,
    generations: int = 500,
    mutation_rate: float = 0.15,
    seed: int | None = None,
) -> GAResult:
    rng = random.Random(seed)
    t0 = time.perf_counter()

    dirs = list(grid.DIRECTIONS_8 if grid.allow_diagonal else grid.DIRECTIONS_4)
    max_steps = grid.rows * grid.cols

    def decode(chromosome: list[int]) -> list[tuple[int, int]]:
        path = [start]
        r, c = start
        visited = {start}
        for gene in chromosome:
            dr, dc = dirs[gene % len(dirs)]
            nr, nc = r + dr, c + dc
            if grid.passable(nr, nc) and (nr, nc) not in visited:
                r, c = nr, nc
                path.append((r, c))
                visited.add((r, c))
                if (r, c) == goal:
                    break
        return path

    def fitness(chromosome: list[int]) -> float:
        path = decode(chromosome)
        last = path[-1]
        dist_to_goal = abs(last[0] - goal[0]) + abs(last[1] - goal[1])
        if last == goal:
            return 1000.0 + (max_steps - len(path))
        return max(0.0, max_steps - dist_to_goal * 10 - len(path))

    def create_individual() -> list[int]:
        return [rng.randint(0, len(dirs) - 1) for _ in range(max_steps)]

    def crossover(p1: list[int], p2: list[int]) -> list[int]:
        cx = rng.randint(1, len(p1) - 1)
        return p1[:cx] + p2[cx:]

    def mutate(ind: list[int]) -> list[int]:
        result = ind[:]
        for i in range(len(result)):
            if rng.random() < mutation_rate:
                result[i] = rng.randint(0, len(dirs) - 1)
        return result

    population = [create_individual() for _ in range(population_size)]
    best_fitness_history = []
    all_explored: set[tuple[int, int]] = set()
    best_overall = None
    best_overall_fit = -1.0

    for _gen in range(generations):
        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored[0][0] > best_overall_fit:
            best_overall_fit = scored[0][0]
            best_overall = scored[0][1]
        best_fitness_history.append(best_overall_fit)

        path = decode(scored[0][1])
        all_explored.update(path)
        if path[-1] == goal:
            break

        elite_count = max(2, population_size // 10)
        elites = [ind for _, ind in scored[:elite_count]]

        new_pop = list(elites)
        while len(new_pop) < population_size:
            p1 = scored[rng.randint(0, elite_count - 1)][1]
            p2 = scored[rng.randint(0, elite_count - 1)][1]
            child = mutate(crossover(p1, p2))
            new_pop.append(child)
        population = new_pop

    best_path = decode(best_overall) if best_overall else [start]
    cost = sum(grid.cost(p) for p in best_path)

    return GAResult(
        path=best_path,
        explored=list(all_explored),
        explored_count=len(all_explored),
        path_length=len(best_path),
        path_cost=cost,
        runtime=time.perf_counter() - t0,
        best_fitness_history=best_fitness_history,
    )
