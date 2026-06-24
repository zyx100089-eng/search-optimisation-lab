from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

from .memory_util import track_memory


@dataclass
class DijkstraResult:
    path: list[tuple[int, int]]
    explored: list[tuple[int, int]]
    explored_count: int
    path_length: int
    path_cost: float
    runtime: float
    peak_memory_bytes: int = 0
    exploration_order: list[list[tuple[int, int]]] = field(default_factory=list)


def dijkstra(grid, start: tuple[int, int], goal: tuple[int, int]) -> DijkstraResult:
    t0 = time.perf_counter()
    _mem_ctx = track_memory()
    mem = _mem_ctx.__enter__()
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

    path = _reconstruct(parent, goal)
    _mem_ctx.__exit__(None, None, None)
    return DijkstraResult(
        path=path,
        explored=explored_order,
        explored_count=len(explored_order),
        path_length=len(path),
        path_cost=dist.get(goal, 0.0),
        runtime=time.perf_counter() - t0,
        peak_memory_bytes=mem["peak_bytes"],
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
