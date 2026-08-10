from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Callable

from .a_star import HEURISTICS
from .base import SearchResult, reconstruct_path
from .memory_util import track_memory


@dataclass
class GreedyResult(SearchResult):
    """Greedy result (same fields as SearchResult, kept for import compatibility)."""
    pass


def greedy_best_first(
    grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic: Callable | str = "manhattan",
) -> GreedyResult:
    if isinstance(heuristic, str):
        heuristic = HEURISTICS[heuristic]

    t0 = time.perf_counter()
    with track_memory() as mem:
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        counter = 0
        pq = [(heuristic(start, goal), counter, start)]
        closed: set[tuple[int, int]] = set()
        explored_order: list[tuple[int, int]] = []
        cost_to: dict[tuple[int, int], float] = {start: 0.0}

        while pq:
            _h, _c, node = heapq.heappop(pq)
            if node in closed:
                continue
            closed.add(node)
            explored_order.append(node)

            if node == goal:
                break

            for (nr, nc), w in grid.neighbours(*node):
                nb = (nr, nc)
                if nb not in closed:
                    new_cost = cost_to[node] + w
                    if nb not in cost_to or new_cost < cost_to[nb]:
                        cost_to[nb] = new_cost
                        parent[nb] = node
                    counter += 1
                    heapq.heappush(pq, (heuristic(nb, goal), counter, nb))

        path = reconstruct_path(parent, goal)
        if not path:
            total_cost = float("inf")
        elif len(path) == 1:
            total_cost = 0.0
        else:
            total_cost = cost_to.get(goal, float("inf"))

    return GreedyResult(
        path=path,
        explored=explored_order,
        explored_count=len(explored_order),
        path_length=len(path),
        path_cost=total_cost,
        runtime=time.perf_counter() - t0,
        peak_memory_bytes=mem.peak_bytes,
    )