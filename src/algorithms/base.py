"""Shared infrastructure for grid-search algorithms.

Provides a uniform result dataclass and path-reconstruction helper so that
BFS, DFS, Dijkstra, A*, and Greedy Best-First can share code and present a
consistent interface to the visualiser and benchmark harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Uniform result returned by every grid-search algorithm.

    Fields are intentionally compatible with the per-algorithm Result
    dataclasses (BFSResult, DFSResult, ...) so existing code keeps working.
    """

    path: list[tuple[int, int]]
    explored: list[tuple[int, int]]
    explored_count: int
    path_length: int
    path_cost: float
    runtime: float
    peak_memory_bytes: int = 0
    exploration_order: list[list[tuple[int, int]]] = field(default_factory=list)


def reconstruct_path(parent: dict, goal) -> list:
    """Rebuild the path from the ``parent`` map produced by a search.

    Returns an empty list when ``goal`` was never reached.
    """
    if goal not in parent:
        return []
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def edge_cost(grid, a, b) -> float:
    """Weight of the edge a -> b in ``grid`` (``inf`` if not adjacent)."""
    for nb, w in grid.neighbours(*a):
        if nb == b:
            return w
    return float("inf")


def compute_path_cost(grid, path: list) -> float:
    """Consistent edge-sum cost for a path.

    Returns ``inf`` for an empty path, ``0.0`` for a single-node path, and the
    sum of edge weights otherwise.
    """
    if not path:
        return float("inf")
    if len(path) == 1:
        return 0.0
    return sum(edge_cost(grid, path[i], path[i + 1]) for i in range(len(path) - 1))