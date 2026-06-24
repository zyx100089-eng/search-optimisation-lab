from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass


@dataclass
class SAResult:
    path: list[tuple[int, int]]
    explored: list[tuple[int, int]]
    explored_count: int
    path_length: int
    path_cost: float
    runtime: float
    temperature_history: list[float]
    cost_history: list[float]


def _greedy_initial_path(grid, start, goal, rng) -> list[tuple[int, int]]:
    path = [start]
    visited = {start}
    current = start
    for _ in range(grid.rows * grid.cols):
        if current == goal:
            break
        neighbours = grid.neighbours(*current)
        unvisited = [(nb, w) for nb, w in neighbours if nb not in visited]
        if not unvisited:
            break
        unvisited.sort(key=lambda x: abs(x[0][0] - goal[0]) + abs(x[0][1] - goal[1]))
        chosen = unvisited[0][0]
        path.append(chosen)
        visited.add(chosen)
        current = chosen
    return path


def _path_cost(grid, path: list[tuple[int, int]]) -> float:
    if not path:
        return float("inf")
    cost = 0.0
    for i in range(1, len(path)):
        cost += grid.cost(path[i])
    if path[-1] != path[0]:
        last = path[-1]
        goal = path[-1]
    return cost


def _evaluate(grid, path, goal) -> float:
    if not path:
        return float("inf")
    cost = sum(grid.cost(p) for p in path)
    last = path[-1]
    penalty = (abs(last[0] - goal[0]) + abs(last[1] - goal[1])) * 100
    return cost + penalty


def simulated_annealing(
    grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.01,
    max_iterations: int = 10000,
    seed: int | None = None,
) -> SAResult:
    rng = random.Random(seed)
    t0 = time.perf_counter()

    current_path = _greedy_initial_path(grid, start, goal, rng)
    current_cost = _evaluate(grid, current_path, goal)
    best_path = list(current_path)
    best_cost = current_cost

    temp = initial_temp
    all_explored: set[tuple[int, int]] = set(current_path)
    temperature_history = []
    cost_history = []

    for _ in range(max_iterations):
        if temp < min_temp:
            break

        temperature_history.append(temp)
        cost_history.append(best_cost)

        new_path = _perturb(grid, current_path, goal, rng)
        new_cost = _evaluate(grid, new_path, goal)
        all_explored.update(new_path)

        delta = new_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_path = new_path
            current_cost = new_cost

        if current_cost < best_cost:
            best_path = list(current_path)
            best_cost = current_cost

        temp *= cooling_rate

        if best_path[-1] == goal and best_cost == len(best_path):
            break

    final_cost = sum(grid.cost(p) for p in best_path)
    return SAResult(
        path=best_path,
        explored=list(all_explored),
        explored_count=len(all_explored),
        path_length=len(best_path),
        path_cost=final_cost,
        runtime=time.perf_counter() - t0,
        temperature_history=temperature_history,
        cost_history=cost_history,
    )


def _perturb(grid, path, goal, rng) -> list[tuple[int, int]]:
    if len(path) < 3:
        return list(path)

    idx = rng.randint(1, len(path) - 1)
    current = path[idx - 1]
    nbrs = grid.neighbours(*current)
    if not nbrs:
        return list(path)

    visited = set(path[:idx])
    candidates = [nb for nb, w in nbrs if nb not in visited]
    if not candidates:
        return list(path)

    new_node = rng.choice(candidates)
    new_path = path[:idx] + [new_node]
    visited.add(new_node)

    node = new_node
    for _ in range(grid.rows * grid.cols):
        if node == goal:
            break
        nbrs2 = grid.neighbours(*node)
        unvisited = [nb for nb, w in nbrs2 if nb not in visited]
        if not unvisited:
            break
        unvisited.sort(key=lambda x: abs(x[0] - goal[0]) + abs(x[1] - goal[1]))
        node = unvisited[0]
        new_path.append(node)
        visited.add(node)

    return new_path
