from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .memory_util import track_memory


@dataclass
class BFSResult:
    path: list[tuple[int, int]]
    explored: list[tuple[int, int]]
    explored_count: int
    path_length: int
    path_cost: float
    runtime: float
    peak_memory_bytes: int = 0
    exploration_order: list[list[tuple[int, int]]] = field(default_factory=list)


def bfs(grid, start: tuple[int, int], goal: tuple[int, int]) -> BFSResult:
    t0 = time.perf_counter()
    mem = {"peak_bytes": 0}
    _mem_ctx = track_memory()
    mem = _mem_ctx.__enter__()
    queue = deque([start])
    visited = {start}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    explored_order: list[tuple[int, int]] = []
    frontier_snapshots: list[list[tuple[int, int]]] = []

    while queue:
        frontier_snapshots.append(list(queue))
        node = queue.popleft()
        explored_order.append(node)

        if node == goal:
            break

        for (nr, nc), _ in grid.neighbours(*node):
            nb = (nr, nc)
            if nb not in visited:
                visited.add(nb)
                parent[nb] = node
                queue.append(nb)

    path = _reconstruct(parent, goal)
    cost = sum(grid.cost(p) for p in path) if path else 0.0
    _mem_ctx.__exit__(None, None, None)
    return BFSResult(
        path=path,
        explored=explored_order,
        explored_count=len(explored_order),
        path_length=len(path),
        path_cost=cost,
        runtime=time.perf_counter() - t0,
        peak_memory_bytes=mem["peak_bytes"],
        exploration_order=frontier_snapshots,
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
