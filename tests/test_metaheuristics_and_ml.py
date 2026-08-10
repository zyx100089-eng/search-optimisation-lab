"""Unit tests for the metaheuristic and learned-heuristic modules."""

import pytest
from src.graph import Grid
from src.algorithms.genetic import genetic_algorithm
from src.algorithms.simulated_annealing import simulated_annealing
from src.algorithms.dijkstra import dijkstra
from src.algorithms.a_star import a_star
from src.algorithms.learned_heuristic import (
    generate_training_data,
    train_learned_heuristic,
    make_learned_heuristic,
    extract_features,
    FEATURE_NAMES,
)


def _empty_grid(rows=5, cols=5):
    return Grid(rows, cols, obstacle_density=0.0, seed=0)


class TestGeneticAlgorithm:
    def test_finds_path_empty_grid(self):
        g = _empty_grid(10, 10)
        r = genetic_algorithm(g, (0, 0), (9, 9), population_size=50, generations=100, seed=0)
        assert r.path_length > 0
        assert r.path[0] == (0, 0)
        assert r.path[-1] == (9, 9)

    def test_seeded_deterministic(self):
        g = _empty_grid(10, 10)
        r1 = genetic_algorithm(g, (0, 0), (9, 9), population_size=50, generations=50, seed=7)
        r2 = genetic_algorithm(g, (0, 0), (9, 9), population_size=50, generations=50, seed=7)
        assert r1.path == r2.path
        assert r1.path_cost == r2.path_cost

    def test_returns_fitness_history(self):
        g = _empty_grid(8, 8)
        r = genetic_algorithm(g, (0, 0), (7, 7), generations=20, seed=0)
        assert len(r.best_fitness_history) > 0
        assert r.best_fitness_history[-1] >= r.best_fitness_history[0]

    def test_no_path_blocked(self):
        g = _empty_grid(3, 3)
        g.set_obstacle(1, 0)
        g.set_obstacle(1, 1)
        g.set_obstacle(1, 2)
        r = genetic_algorithm(g, (0, 0), (2, 2), population_size=20, generations=20, seed=0)
        assert r.path[-1] != (2, 2)


class TestSimulatedAnnealing:
    def test_finds_path_empty_grid(self):
        g = _empty_grid(10, 10)
        r = simulated_annealing(g, (0, 0), (9, 9), max_iterations=2000, seed=0)
        assert r.path_length > 0
        assert r.path[0] == (0, 0)
        assert r.path[-1] == (9, 9)

    def test_seeded_deterministic(self):
        g = _empty_grid(10, 10)
        r1 = simulated_annealing(g, (0, 0), (9, 9), max_iterations=500, seed=3)
        r2 = simulated_annealing(g, (0, 0), (9, 9), max_iterations=500, seed=3)
        assert r1.path == r2.path

    def test_returns_histories(self):
        g = Grid.generate_random(10, 10, 0.25, False, 5)
        g.set_passable(0, 0)
        g.set_passable(9, 9)
        r = simulated_annealing(g, (0, 0), (9, 9), max_iterations=100, seed=0)
        assert len(r.temperature_history) > 0
        assert len(r.cost_history) > 0
        assert r.temperature_history[0] >= r.temperature_history[-1]

    def test_cost_decreases_or_equal(self):
        g = _empty_grid(8, 8)
        r = simulated_annealing(g, (0, 0), (7, 7), max_iterations=200, seed=0)
        assert r.cost_history[-1] <= r.cost_history[0] + 1e-9


class TestLearnedHeuristic:
    def test_feature_shape(self):
        g = _empty_grid(8, 8)
        x = extract_features(g, (0, 0), (7, 7))
        assert x.shape == (len(FEATURE_NAMES),)

    def test_train_and_predict(self):
        samples = generate_training_data(n_grids=5, rows=10, cols=10, seed=0, samples_per_grid=10)
        assert len(samples) > 0
        tr = train_learned_heuristic(samples, epochs=20, lr=0.05, seed=0)
        assert len(tr.train_loss_history) == 20
        assert tr.train_loss_history[-1] <= tr.train_loss_history[0]

    def test_learned_astar_reaches_goal(self):
        samples = generate_training_data(n_grids=10, rows=10, cols=10, seed=1)
        tr = train_learned_heuristic(samples, epochs=50, seed=0)
        grid = Grid.generate_random(10, 10, 0.15, False, 7)
        grid.set_passable(0, 0)
        grid.set_passable(9, 9)
        h = make_learned_heuristic(tr.model, grid)
        r = a_star(grid, (0, 0), (9, 9), heuristic=h)
        assert r.path_length > 0
        assert r.path[-1] == (9, 9)

    def test_learned_never_negative(self):
        samples = generate_training_data(n_grids=5, rows=8, cols=8, seed=0)
        tr = train_learned_heuristic(samples, epochs=20, seed=0)
        grid = _empty_grid(8, 8)
        h = make_learned_heuristic(tr.model, grid)
        for r in range(8):
            for c in range(8):
                assert h((r, c), (7, 7)) >= 0.0