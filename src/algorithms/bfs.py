from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .base import SearchResult, reconstruct_path, compute_path_cost
from .memory_util import track_memory


@dataclass
class BFSResult(SearchResult):
    """BFS result (same fields as SearchResult, kept for import compatibility)."""
    pass


def bfs(grid, start: tuple[int, int], goal: tuple[int, int]) -> BFSResult:
    t0 = time.perf_counter()
    with track_memory() as mem:
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

        path = reconstruct_path(parent, goal)
        cost = compute_path_cost(grid, path)

    return BFSResult(
        path=path,
        explored=explored_order,
        explored_count=len(explored_order),
        path_length=len(path),
        path_cost=cost,
        runtime=time.perf_counter() - t0,
        peak_memory_bytes=mem.peak_bytes,
        exploration_order=frontier_snapshots,
    )