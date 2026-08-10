"""Unit tests for search algorithms."""

import pytest
from src.graph import Grid
from src.algorithms.bfs import bfs
from src.algorithms.dfs import dfs
from src.algorithms.dijkstra import dijkstra
from src.algorithms.a_star import a_star
from src.algorithms.greedy_best_first import greedy_best_first


def _empty_grid(rows=5, cols=5):
    g = Grid(rows, cols, obstacle_density=0.0, seed=0)
    return g


def _simple_grid():
    g = Grid(5, 5, obstacle_density=0.0, seed=0)
    g.set_obstacle(1, 1)
    g.set_obstacle(1, 2)
    g.set_obstacle(1, 3)
    return g


class TestBFS:
    def test_finds_path_empty_grid(self):
        g = _empty_grid()
        r = bfs(g, (0, 0), (4, 4))
        assert r.path_length > 0
        assert r.path[0] == (0, 0)
        assert r.path[-1] == (4, 4)

    def test_path_length_empty_grid(self):
        g = _empty_grid()
        r = bfs(g, (0, 0), (4, 4))
        assert r.path_length == 9  # Manhattan: 4+4+1=9 nodes

    def test_no_path(self):
        g = _empty_grid(3, 3)
        g.set_obstacle(1, 0)
        g.set_obstacle(1, 1)
        g.set_obstacle(1, 2)
        r = bfs(g, (0, 0), (2, 2))
        assert r.path_length == 0


class TestDFS:
    def test_finds_path_empty_grid(self):
        g = _empty_grid()
        r = dfs(g, (0, 0), (4, 4))
        assert r.path_length > 0
        assert r.path[0] == (0, 0)
        assert r.path[-1] == (4, 4)

    def test_no_path(self):
        g = _empty_grid(3, 3)
        g.set_obstacle(1, 0)
        g.set_obstacle(1, 1)
        g.set_obstacle(1, 2)
        r = dfs(g, (0, 0), (2, 2))
        assert r.path_length == 0


class TestDijkstra:
    def test_finds_optimal_path(self):
        g = _empty_grid()
        r = dijkstra(g, (0, 0), (4, 4))
        assert r.path_length == 9
        assert r.path[0] == (0, 0)
        assert r.path[-1] == (4, 4)

    def test_weighted_grid(self):
        g = _empty_grid()
        g.weights = [[1] * 5 for _ in range(5)]
        g.weights[0][1] = 10
        r = dijkstra(g, (0, 0), (0, 2))
        assert r.path_cost <= 12  # should prefer going around

    def test_same_cost_as_bfs_unweighted(self):
        g = _empty_grid(10, 10)
        r_bfs = bfs(g, (0, 0), (9, 9))
        r_dij = dijkstra(g, (0, 0), (9, 9))
        assert r_dij.path_length == r_bfs.path_length


class TestAStar:
    def test_finds_optimal_path(self):
        g = _empty_grid()
        r = a_star(g, (0, 0), (4, 4))
        assert r.path_length == 9

    def test_explores_fewer_than_dijkstra(self):
        g = Grid.generate_random(20, 20, 0.15, seed=42)
        s, e = (0, 0), (19, 19)
        g.set_passable(*s)
        g.set_passable(*e)
        r_dij = dijkstra(g, s, e)
        r_astar = a_star(g, s, e)
        if r_dij.path_length > 0 and r_astar.path_length > 0:
            assert r_astar.explored_count <= r_dij.explored_count
            assert r_astar.path_cost == pytest.approx(r_dij.path_cost, abs=0.01)

    def test_different_heuristics(self):
        g = _empty_grid(10, 10)
        for h in ["manhattan", "euclidean", "chebyshev"]:
            r = a_star(g, (0, 0), (9, 9), heuristic=h)
            assert r.path_length == 19


class TestGreedyBestFirst:
    def test_finds_path(self):
        g = _empty_grid()
        r = greedy_best_first(g, (0, 0), (4, 4))
        assert r.path_length > 0
        assert r.path[-1] == (4, 4)

    def test_not_necessarily_optimal(self):
        g = _simple_grid()
        r_greedy = greedy_best_first(g, (0, 0), (4, 4))
        r_astar = a_star(g, (0, 0), (4, 4))
        if r_greedy.path_length > 0 and r_astar.path_length > 0:
            assert r_greedy.path_cost >= r_astar.path_cost


class TestEdgeCases:
    def test_start_equals_goal(self):
        g = _empty_grid()
        for algo in [bfs, dfs, dijkstra]:
            r = algo(g, (2, 2), (2, 2))
            assert r.path_length == 1

    def test_adjacent_goal(self):
        g = _empty_grid()
        r = bfs(g, (0, 0), (0, 1))
        assert r.path_length == 2
