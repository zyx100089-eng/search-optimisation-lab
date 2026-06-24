from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AStarResult:
    path: list[tuple[int, int]]
    explored: list[tuple[int, int]]
    explored_count: int
    path_length: int
    path_cost: float
    runtime: float
    exploration_order: list[list[tuple[int, int]]] = field(default_factory=list)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> float:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


HEURISTICS: dict[str, Callable] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
}


def a_star(
    grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic: Callable | str = "manhattan",
) -> AStarResult:
    if isinstance(heuristic, str):
        heuristic = HEURISTICS[heuristic]

    t0 = time.perf_counter()
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    counter = 0
    pq = [(heuristic(start, goal), counter, start)]
    closed: set[tuple[int, int]] = set()
    explored_order: list[tuple[int, int]] = []

    while pq:
        _f, _c, node = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)
        explored_order.append(node)

        if node == goal:
            break

        for (nr, nc), w in grid.neighbours(*node):
            nb = (nr, nc)
            tentative_g = g_score[node] + w
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                f = tentative_g + heuristic(nb, goal)
                parent[nb] = node
                counter += 1
                heapq.heappush(pq, (f, counter, nb))

    path = _reconstruct(parent, goal)
    return AStarResult(
        path=path,
        explored=explored_order,
        explored_count=len(explored_order),
        path_length=len(path),
        path_cost=g_score.get(goal, 0.0),
        runtime=time.perf_counter() - t0,
    )


def _reconstruct(parent, goal):
    if goal not in parent:
        return []
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]
