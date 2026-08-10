"""Learned heuristic for A* via supervised regression on h*(n).

This module is the machine-learning component of the lab.  Instead of relying
on a hand-crafted heuristic such as Manhattan distance, we *learn* a heuristic
from data:

1. Sample many (grid, node, goal) triples.
2. Compute the *true* remaining cost h*(n) with Dijkstra (a label we can trust).
3. Extract lightweight features for each node (Manhattan distance to goal,
   local obstacle density, degree, distance to nearest wall, goal alignment).
4. Train a linear model h_theta(n) = theta . x(n) by gradient descent on MSE.
5. Use the learned value *directly* as the A* heuristic — it is **not**
   guaranteed admissible, so the resulting search is a bounded-suboptimal
   planner (like weighted A*).  The experiments page measures the resulting
   speed-up *and* the suboptimality ratio, comparing against the theoretical
   ε-bound of weighted A*.

This mirrors a real research theme in AI planning: learned heuristics can
dramatically reduce search effort, but they trade guaranteed optimality for
empirical speed — exactly the trade-off that weighted A* formalises.

The model is implemented from scratch in numpy (no scikit-learn) so the maths
is explicit and inspectable — useful for an admissions interview where you
may be asked to derive the update rule.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from src.graph import Grid
from src.algorithms.dijkstra import dijkstra


@dataclass
class TrainingSample:
    features: np.ndarray
    h_star: float


@dataclass
class LearnedHeuristicModel:
    """Linear model h(n) = theta . x(n), trained by gradient descent."""

    theta: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    y_mean: float
    y_std: float
    feature_names: list[str]
    train_loss_history: list[float] = field(default_factory=list)

    def predict_raw(self, features: np.ndarray) -> float:
        """Predict h*(n) in the original (unstandardised) units."""
        x = (features - self.feature_mean) / self.feature_std
        return float((self.theta @ x) * self.y_std + self.y_mean)

    def __call__(self, grid, node, goal):
        x = extract_features(grid, node, goal)
        return max(0.0, self.predict_raw(x))


def extract_features(grid, node: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    """Feature vector describing the local geometry around `node`.

    Features (7):
      0  Manhattan distance to goal
      1  Euclidean distance to goal
      2  Open neighbours count (degree in the free graph)
      3  Local obstacle density (3x3 window)
      4  Distance to nearest wall (chebyshev, capped at 5)
      5  Goal direction alignment: (dr,dc) . (goal-node) / ||goal-node||
      6  Bias term (constant 1.0)
    """
    r, c = node
    gr, gc = goal

    manhattan = abs(r - gr) + abs(c - gc)
    euclidean = float(np.hypot(r - gr, c - gc))

    nbrs = grid.neighbours(r, c)
    degree = len(nbrs)

    window = 0
    total = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if grid.in_bounds(rr, cc):
                total += 1
                if not grid.passable(rr, cc):
                    window += 1
    density = window / max(1, total)

    wall_dist = 5
    for radius in range(1, 6):
        hit = False
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                rr, cc = r + dr, c + dc
                if not grid.in_bounds(rr, cc) or not grid.passable(rr, cc):
                    hit = True
                    break
            if hit:
                break
        if hit:
            wall_dist = radius
            break

    if manhattan > 0:
        move = np.array([gr - r, gc - c], dtype=float)
        move /= np.linalg.norm(move) + 1e-9
        alignment = float(move[0] + move[1]) / 2
    else:
        alignment = 0.0

    return np.array([manhattan, euclidean, degree, density, wall_dist, alignment, 1.0])


FEATURE_NAMES = [
    "manhattan",
    "euclidean",
    "degree",
    "obstacle_density",
    "wall_distance",
    "goal_alignment",
    "bias",
]


def generate_training_data(
    n_grids: int = 40,
    rows: int = 20,
    cols: int = 20,
    density: float = 0.2,
    seed: int = 0,
    samples_per_grid: int = 25,
) -> list[TrainingSample]:
    """Generate (features, h*) pairs by labelling with Dijkstra."""
    rng = random.Random(seed)
    samples: list[TrainingSample] = []

    for i in range(n_grids):
        grid = Grid.generate_random(rows, cols, density, False, rng.randint(0, 10**9))
        start = (0, 0)
        goal = (rows - 1, cols - 1)
        grid.set_passable(*start)
        grid.set_passable(*goal)

        result = dijkstra(grid, start, goal)
        if not result.path:
            continue

        passable = [(r, c) for r in range(rows) for c in range(cols) if grid.passable(r, c)]
        chosen = rng.sample(passable, min(samples_per_grid, len(passable)))

        for node in chosen:
            res = dijkstra(grid, node, goal)
            h_star = res.path_cost if res.path else float("inf")
            if h_star == float("inf"):
                continue
            feats = extract_features(grid, node, goal)
            samples.append(TrainingSample(features=feats, h_star=h_star))

    return samples


@dataclass
class TrainResult:
    model: LearnedHeuristicModel
    train_loss_history: list[float]
    n_samples: int
    n_features: int


def train_learned_heuristic(
    samples: list[TrainingSample],
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 0,
) -> TrainResult:
    """Train a linear model with mini-batch gradient descent on MSE loss.

    Both features and target are standardised so gradient descent is well
    conditioned.  Predictions are de-standardised at inference time.
    """
    rng = np.random.default_rng(seed)
    X = np.array([s.features for s in samples], dtype=float)
    y = np.array([s.h_star for s in samples], dtype=float)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std

    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-6:
        y_std = 1.0
    yn = (y - y_mean) / y_std

    theta = np.zeros(Xn.shape[1])
    loss_history: list[float] = []
    batch_size = min(64, len(Xn))

    for _epoch in range(epochs):
        idx = rng.permutation(len(Xn))
        for start in range(0, len(Xn), batch_size):
            b = idx[start:start + batch_size]
            Xb = Xn[b]
            yb = yn[b]
            pred = Xb @ theta
            err = pred - yb
            grad = (Xb.T @ err) / len(b)
            theta -= lr * grad

        full_pred = Xn @ theta
        loss = float(np.mean((full_pred - yn) ** 2))
        loss_history.append(loss)

    model = LearnedHeuristicModel(
        theta=theta,
        feature_mean=mean,
        feature_std=std,
        y_mean=y_mean,
        y_std=y_std,
        feature_names=FEATURE_NAMES,
        train_loss_history=loss_history,
    )
    return TrainResult(model=model, train_loss_history=loss_history, n_samples=len(samples), n_features=Xn.shape[1])


def make_learned_heuristic(model: LearnedHeuristicModel, grid):
    """Return a 2-arg heuristic callable h(node, goal) compatible with a_star().

    The grid is captured in the closure so features can be extracted at query
    time.  The learned value is used *directly* — it is not guaranteed
    admissible, so the resulting search is bounded-suboptimal (see the
    experiments page for the optimality/speed trade-off analysis).
    """
    def h(node, goal):
        return model(grid, node, goal)
    h.__name__ = "learned"
    return h