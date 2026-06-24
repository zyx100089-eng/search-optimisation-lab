"""Unit tests for dynamic programming algorithms."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
from src.graph import Grid
from src.algorithms.dynamic_programming import floyd_warshall, held_karp, grid_tsp
from src.algorithms.a_star import a_star


def _empty_grid(rows=5, cols=5):
    return Grid(rows, cols, obstacle_density=0.0, seed=0)


class TestFloydWarshall:
    def test_agrees_with_astar_empty_grid(self):
        g = _empty_grid()
        fw = floyd_warshall(g)
        for goal in [(0, 4), (4, 0), (4, 4), (2, 3)]:
            fw_cost = fw.cost((0, 0), goal)
            astar_cost = a_star(g, (0, 0), goal).path_cost
            assert fw_cost == pytest.approx(astar_cost, abs=0.01)

    def test_path_valid(self):
        g = _empty_grid()
        fw = floyd_warshall(g)
        path = fw.path((0, 0), (4, 4))
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        assert len(path) == 9  # Manhattan: 4+4+1

    def test_unreachable(self):
        g = _empty_grid(3, 3)
        g.set_obstacle(1, 0)
        g.set_obstacle(1, 1)
        g.set_obstacle(1, 2)
        fw = floyd_warshall(g)
        assert fw.cost((0, 0), (2, 2)) == float("inf")
        assert fw.path((0, 0), (2, 2)) == []

    def test_same_node(self):
        g = _empty_grid()
        fw = floyd_warshall(g)
        assert fw.cost((2, 2), (2, 2)) == 0.0
        assert fw.path((2, 2), (2, 2)) == [(2, 2)]

    def test_with_obstacles(self):
        g = Grid.generate_random(8, 8, 0.15, seed=42)
        g.set_passable(0, 0)
        g.set_passable(7, 7)
        fw = floyd_warshall(g)
        astar_result = a_star(g, (0, 0), (7, 7))
        if astar_result.path_length > 0:
            assert fw.cost((0, 0), (7, 7)) == pytest.approx(astar_result.path_cost, abs=0.01)


class TestHeldKarp:
    def test_trivial_2_cities(self):
        dist = [[0, 5], [5, 0]]
        result = held_karp(dist)
        assert result.tour_cost == pytest.approx(10.0)
        assert len(result.tour) == 3  # 0 -> 1 -> 0

    def test_triangle(self):
        dist = [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ]
        result = held_karp(dist)
        assert result.tour_cost == pytest.approx(4.0)  # 0->1->2->0: 1+1+2=4
        assert len(result.tour) == 4
        assert result.tour[0] == result.tour[-1]

    def test_4_cities_square(self):
        # 4 cities at corners of a unit square
        dist = [
            [0, 1, math.sqrt(2), 1],
            [1, 0, 1, math.sqrt(2)],
            [math.sqrt(2), 1, 0, 1],
            [1, math.sqrt(2), 1, 0],
        ]
        result = held_karp(dist)
        assert result.tour_cost == pytest.approx(4.0, abs=0.01)  # perimeter

    def test_single_city(self):
        result = held_karp([[0]])
        assert result.tour_cost == 0.0
        assert result.n_cities == 1

    def test_custom_labels(self):
        dist = [[0, 3], [3, 0]]
        result = held_karp(dist, city_labels=["A", "B"])
        assert set(result.tour) <= {"A", "B"}
        assert result.tour[0] == result.tour[-1]


class TestGridTSP:
    def test_small_grid_tour(self):
        g = _empty_grid(5, 5)
        waypoints = [(0, 0), (0, 4), (4, 0), (4, 4)]
        result = grid_tsp(g, waypoints)
        assert result.tour_cost < float("inf")
        assert result.tour[0] == result.tour[-1]
        assert set(result.tour[:-1]) == set(waypoints)

    def test_adjacent_waypoints(self):
        g = _empty_grid(3, 3)
        waypoints = [(0, 0), (0, 1), (1, 0)]
        result = grid_tsp(g, waypoints)
        assert result.tour_cost == pytest.approx(4.0)  # triangle with sides 1,1,2
