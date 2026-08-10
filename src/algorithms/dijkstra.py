from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

from .base import SearchResult, reconstruct_path
from .memory_util import track_memory


@dataclass
class DijkstraResult(SearchResult):
    """Dijkstra result (same fields as SearchResult, kept for import compatibility)."""
    pass


def dijkstra(grid, start: tuple[int, int], goal: tuple[int, int]) -> DijkstraResult:
    t0 = time.perf_counter()
    with track_memory() as mem:
        dist: dict[tuple[int, int], float] = {start: 0.0}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        pq = [(0.0, start)]
        explored_order: list[tuple[int, int]] = []
        closed: set[tuple[int, int]] = set()

        while pq:
            cost_so_far, node = heapq.heappop(pq)
            if node in closed:
                continue
            closed.add(node)
            explored_order.append(node)

            if node == goal:
                break

            for (nr, nc), w in grid.neighbours(*node):
                nb = (nr, nc)
                new_cost = cost_so_far + w
                if nb not in dist or new_cost < dist[nb]:
                    dist[nb] = new_cost
                    parent[nb] = node
                    heapq.heappush(pq, (new_cost, nb))

        path = reconstruct_path(parent, goal)
        if not path:
            path_cost = float("inf")
        elif len(path) == 1:
            path_cost = 0.0
        else:
            path_cost = dist.get(goal, float("inf"))

    return DijkstraResult(
        path=path,
        explored=explored_order,
        explored_count=len(explored_order),
        path_length=len(path),
        path_cost=path_cost,
        runtime=time.perf_counter() - t0,
        peak_memory_bytes=mem.peak_bytes,
    )