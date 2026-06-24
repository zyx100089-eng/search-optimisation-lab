from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_grid(
    grid,
    path: list[tuple[int, int]] | None = None,
    explored: list[tuple[int, int]] | None = None,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
    title: str = "",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    rows, cols = grid.rows, grid.cols
    if figsize is None:
        scale = max(4, min(12, max(rows, cols) * 0.4))
        figsize = (scale, scale)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    img = np.ones((rows, cols, 3))

    for r in range(rows):
        for c in range(cols):
            if grid.grid[r][c] == 1:
                img[r, c] = [0.2, 0.2, 0.2]

    if explored:
        for r, c in explored:
            if grid.grid[r][c] == 0:
                img[r, c] = [0.72, 0.85, 0.95]

    if path:
        for r, c in path:
            img[r, c] = [0.2, 0.6, 1.0]

    if start:
        img[start[0], start[1]] = [0.0, 0.8, 0.0]
    if goal:
        img[goal[0], goal[1]] = [1.0, 0.2, 0.2]

    ax.imshow(img, interpolation="nearest")

    if rows <= 30 and cols <= 30:
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=[0.2, 0.2, 0.2], label="Obstacle"),
        mpatches.Patch(color=[0.72, 0.85, 0.95], label="Explored"),
        mpatches.Patch(color=[0.2, 0.6, 1.0], label="Path"),
        mpatches.Patch(color=[0.0, 0.8, 0.0], label="Start"),
        mpatches.Patch(color=[1.0, 0.2, 0.2], label="Goal"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig
