from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Callable

from .base import SearchResult, reconstruct_path
from .memory_util import track_memory


@dataclass
class AStarResult(SearchResult):
    """A* result (same fields as SearchResult, kept for import compatibility)."""
    pass


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> float:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def inadmissible_greedy(a: tuple[int, int], b: tuple[int, int]) -> float:
    return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * 5.0


def inadmissible_extreme(a: tuple[int, int], b: tuple[int, int]) -> float:
    return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * 20.0


def zero_heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return 0.0


def weighted_manhattan(weight: float):
    def h(a: tuple[int, int], b: tuple[int, int]) -> float:
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * weight
    h.__name__ = f"manhattan_x{weight}"
    return h


HEURISTICS: dict[str, Callable] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "zero": zero_heuristic,
    "inadmissible_5x": inadmissible_greedy,
    "inadmissible_20x": inadmissible_extreme,
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
    with track_memory() as mem:
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

        path = reconstruct_path(parent, goal)
        if not path:
            path_cost = float("inf")
        elif len(path) == 1:
            path_cost = 0.0
        else:
            path_cost = g_score.get(goal, float("inf"))

    return AStarResult(
        path=path,
        explored=explored_order,
        explored_count=len(explored_order),
        path_length=len(path),
        path_cost=path_cost,
        runtime=time.perf_counter() - t0,
        peak_memory_bytes=mem.peak_bytes,
    )