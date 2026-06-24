import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import matplotlib.pyplot as plt
from src.graph import Grid
from src.algorithms.genetic import genetic_algorithm
from src.algorithms.simulated_annealing import simulated_annealing
from src.algorithms.a_star import a_star
from src.visualisation.grid_view import draw_grid
from src.visualisation.metrics import compare_results

st.set_page_config(page_title="Metaheuristics", layout="wide")
st.title("Metaheuristic Algorithms")

with st.sidebar:
    st.header("Grid Settings")
    rows = st.slider("Rows", 5, 50, 15, key="meta_rows")
    cols = st.slider("Columns", 5, 50, 15, key="meta_cols")
    density = st.slider("Obstacle Density", 0.0, 0.5, 0.2, 0.05, key="meta_density")
    seed = st.number_input("Random seed", value=42, step=1, key="meta_seed")

    st.header("Genetic Algorithm")
    ga_pop = st.slider("Population size", 50, 500, 200, key="ga_pop")
    ga_gen = st.slider("Generations", 50, 2000, 500, key="ga_gen")
    ga_mut = st.slider("Mutation rate", 0.01, 0.5, 0.15, 0.01, key="ga_mut")

    st.header("Simulated Annealing")
    sa_temp = st.slider("Initial temperature", 10.0, 500.0, 100.0, key="sa_temp")
    sa_cool = st.slider("Cooling rate", 0.9, 0.999, 0.995, 0.001, key="sa_cool")
    sa_iter = st.slider("Max iterations", 1000, 50000, 10000, key="sa_iter")

    run = st.button("Run", type="primary", use_container_width=True)

if run:
    grid = Grid.generate_random(rows, cols, density, False, int(seed))
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    grid.set_passable(*start)
    grid.set_passable(*goal)

    astar_result = a_star(grid, start, goal)
    ga_result = genetic_algorithm(grid, start, goal, ga_pop, ga_gen, ga_mut, int(seed))
    sa_result = simulated_annealing(grid, start, goal, sa_temp, sa_cool, 0.01, sa_iter, int(seed))

    results = {"A* (baseline)": astar_result, "Genetic Algorithm": ga_result, "Simulated Annealing": sa_result}

    c1, c2, c3 = st.columns(3)
    for col, (name, res) in zip([c1, c2, c3], results.items()):
        with col:
            fig = draw_grid(grid, res.path, res.explored, start, goal, title=name)
            st.pyplot(fig)

    st.subheader("Comparison")
    df = compare_results(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Convergence")
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ga_result.best_fitness_history, color="tab:blue")
    ax1.set_title("GA: Best Fitness per Generation")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.grid(True, alpha=0.3)

    ax2.plot(sa_result.cost_history, color="tab:orange")
    ax2.set_title("SA: Best Cost over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cost")
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    st.pyplot(fig2)
