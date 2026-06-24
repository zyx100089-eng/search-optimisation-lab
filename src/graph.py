from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional


class Graph:
    """General-purpose graph with optional edge weights."""

    def __init__(self):
        self._adj: dict[str, dict[str, float]] = defaultdict(dict)

    def add_edge(self, u: str, v: str, weight: float = 1.0, directed: bool = False):
        self._adj[u][v] = weight
        if not directed:
            self._adj[v][u] = weight

    def neighbours(self, node: str) -> list[tuple[str, float]]:
        return list(self._adj[node].items())

    def cost(self, u: str, v: str) -> float:
        return self._adj[u].get(v, float("inf"))

    @property
    def nodes(self) -> set[str]:
        nodes = set(self._adj.keys())
        for nbrs in self._adj.values():
            nodes.update(nbrs.keys())
        return nodes


class Grid:
    """2-D grid maze.  0 = passable, 1 = obstacle."""

    DIRECTIONS_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    DIRECTIONS_8 = DIRECTIONS_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(
        self,
        rows: int,
        cols: int,
        obstacle_density: float = 0.2,
        allow_diagonal: bool = False,
        seed: Optional[int] = None,
        weights: Optional[list[list[float]]] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.allow_diagonal = allow_diagonal

        rng = random.Random(seed)
        self.grid = [
            [1 if rng.random() < obstacle_density else 0 for _ in range(cols)]
            for _ in range(rows)
        ]
        self.weights = weights  # optional per-cell cost matrix

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.grid[r][c] == 0

    def set_passable(self, r: int, c: int):
        if self.in_bounds(r, c):
            self.grid[r][c] = 0

    def set_obstacle(self, r: int, c: int):
        if self.in_bounds(r, c):
            self.grid[r][c] = 1

    def neighbours(self, r: int, c: int) -> list[tuple[tuple[int, int], float]]:
        dirs = self.DIRECTIONS_8 if self.allow_diagonal else self.DIRECTIONS_4
        result = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if self.passable(nr, nc):
                w = 1.0
                if self.weights is not None:
                    w = self.weights[nr][nc]
                if abs(dr) + abs(dc) == 2:
                    w *= 1.4142135623730951  # sqrt(2)
                result.append(((nr, nc), w))
        return result

    def cost(self, pos: tuple[int, int]) -> float:
        if self.weights is not None:
            return self.weights[pos[0]][pos[1]]
        return 1.0

    @staticmethod
    def generate_random(
        rows: int,
        cols: int,
        obstacle_density: float = 0.2,
        allow_diagonal: bool = False,
        seed: Optional[int] = None,
        weighted: bool = False,
    ) -> Grid:
        rng = random.Random(seed)
        weights = None
        if weighted:
            weights = [
                [rng.randint(1, 9) for _ in range(cols)] for _ in range(rows)
            ]
        g = Grid(rows, cols, obstacle_density, allow_diagonal, seed, weights)
        g.grid[0][0] = 0
        g.grid[rows - 1][cols - 1] = 0
        return g
