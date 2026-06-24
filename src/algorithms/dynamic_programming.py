from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass


@dataclass
class FloydWarshallResult:
    dist: list[list[float]]
    next_node: list[list[int | None]]
    node_list: list
    runtime: float

    def path(self, source, target) -> list:
        si = self.node_list.index(source)
        ti = self.node_list.index(target)
        if self.next_node[si][ti] is None:
            return []
        path = [source]
        while si != ti:
            si = self.next_node[si][ti]
            path.append(self.node_list[si])
        return path

    def cost(self, source, target) -> float:
        si = self.node_list.index(source)
        ti = self.node_list.index(target)
        return self.dist[si][ti]


def floyd_warshall(grid) -> FloydWarshallResult:
    """All-pairs shortest paths on a grid using Floyd-Warshall.

    Only practical for small grids (up to ~15x15) due to O(V^3) complexity.
    """
    t0 = time.perf_counter()

    nodes = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.passable(r, c):
                nodes.append((r, c))

    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    INF = float("inf")
    dist = [[INF] * n for _ in range(n)]
    nxt = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0.0
        nxt[i][i] = i

    for node in nodes:
        i = idx[node]
        for nb, w in grid.neighbours(*node):
            j = idx[nb]
            dist[i][j] = w
            nxt[i][j] = j

    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                candidate = dist[i][k] + dist[k][j]
                if candidate < dist[i][j]:
                    dist[i][j] = candidate
                    nxt[i][j] = nxt[i][k]

    return FloydWarshallResult(
        dist=dist,
        next_node=nxt,
        node_list=nodes,
        runtime=time.perf_counter() - t0,
    )


@dataclass
class HeldKarpResult:
    tour: list
    tour_cost: float
    runtime: float
    n_cities: int


def held_karp(dist_matrix: list[list[float]], city_labels: list | None = None) -> HeldKarpResult:
    """Exact TSP solution via Held-Karp dynamic programming.

    Solves the Travelling Salesman Problem in O(2^n * n^2) time and space.
    Practical for up to ~20 cities.

    Args:
        dist_matrix: n x n symmetric distance matrix.
        city_labels: optional labels for cities (defaults to 0..n-1).
    """
    t0 = time.perf_counter()
    n = len(dist_matrix)
    if city_labels is None:
        city_labels = list(range(n))

    if n <= 1:
        return HeldKarpResult(
            tour=list(city_labels),
            tour_cost=0.0,
            runtime=time.perf_counter() - t0,
            n_cities=n,
        )

    INF = float("inf")
    # dp[S][i] = min cost to visit all cities in set S, ending at city i
    # S is a bitmask; city 0 is always the start
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    dp[1][0] = 0.0  # start at city 0, visited = {0}

    for S in range(1 << n):
        for u in range(n):
            if dp[S][u] == INF:
                continue
            if not (S & (1 << u)):
                continue
            for v in range(n):
                if S & (1 << v):
                    continue
                new_S = S | (1 << v)
                new_cost = dp[S][u] + dist_matrix[u][v]
                if new_cost < dp[new_S][v]:
                    dp[new_S][v] = new_cost
                    parent[new_S][v] = u

    full_mask = (1 << n) - 1
    best_cost = INF
    last_city = -1
    for u in range(n):
        total = dp[full_mask][u] + dist_matrix[u][0]
        if total < best_cost:
            best_cost = total
            last_city = u

    tour_indices = []
    S = full_mask
    u = last_city
    while u != -1:
        tour_indices.append(u)
        prev = parent[S][u]
        S = S ^ (1 << u)
        u = prev
    tour_indices.reverse()
    tour_indices.append(tour_indices[0])  # return to start

    tour = [city_labels[i] for i in tour_indices]

    return HeldKarpResult(
        tour=tour,
        tour_cost=best_cost,
        runtime=time.perf_counter() - t0,
        n_cities=n,
    )


def grid_tsp(grid, waypoints: list[tuple[int, int]]) -> HeldKarpResult:
    """Solve TSP on a grid: find the shortest tour visiting all waypoints.

    Uses Floyd-Warshall to compute pairwise shortest distances, then
    Held-Karp for the exact optimal tour.
    """
    fw = floyd_warshall(grid)

    n = len(waypoints)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = fw.cost(waypoints[i], waypoints[j])

    result = held_karp(dist_matrix, city_labels=waypoints)
    return result
