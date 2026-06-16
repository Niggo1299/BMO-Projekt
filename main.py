"""Ameisenalgorithmus (Ant-Cycle) für das 0/1-Rucksackproblem."""

import os
import csv
import json
import argparse
import numpy as np
from ant import Ant
from item import Item
import visualization


def construct_solutions_vectorized(group_size, item_weights, item_values, item_scores, max_load):
    """Vektorisierte Lösungskonstruktion für alle Ameisen parallel."""
    num_items = len(item_weights)
    backpacks = np.zeros((group_size, num_items), dtype=np.int8)
    current_loads = np.zeros(group_size, dtype=np.float64)
    current_values = np.zeros(group_size, dtype=np.float64)
    active = np.ones(group_size, dtype=bool)

    while np.any(active):
        active_idx = np.where(active)[0]

        feasible_active = ((backpacks[active_idx] == 0) &
                           (item_weights[None, :] <= (max_load - current_loads[active_idx])[:, None]))

        has_feasible = np.any(feasible_active, axis=1)
        if not np.all(has_feasible):
            active[active_idx[~has_feasible]] = False
            active_idx = np.where(active)[0]
            feasible_active = feasible_active[has_feasible]
            if len(active_idx) == 0:
                break

        ant_weights = item_scores[None, :] * feasible_active
        sum_weights = ant_weights.sum(axis=1)

        zero_sum_mask = (sum_weights == 0)
        if np.any(zero_sum_mask):
            ant_weights[zero_sum_mask] = feasible_active[zero_sum_mask].astype(np.float64)
            sum_weights = ant_weights.sum(axis=1)

        cumsum_weights = np.cumsum(ant_weights, axis=1)
        r = np.random.rand(len(active_idx), 1) * sum_weights[:, None]
        chosen = (cumsum_weights >= r).argmax(axis=1)

        backpacks[active_idx, chosen] = 1
        current_loads[active_idx] += item_weights[chosen]
        current_values[active_idx] += item_values[chosen]

    return backpacks, current_loads, current_values


def main():
    # Defaults
    DEF = dict(group_size=175, evaporation=0.35, iterations=100,
               alpha=1.33, beta=25.0, stagnation_limit=40)

    parser = argparse.ArgumentParser(description="ACO für Knapsack (Ant-Cycle)")
    parser.add_argument("--group-size", type=int, default=DEF['group_size'])
    parser.add_argument("--evaporation", type=float, default=DEF['evaporation'])
    parser.add_argument("--iterations", type=int, default=DEF['iterations'])
    parser.add_argument("--alpha", type=float, default=DEF['alpha'])
    parser.add_argument("--beta", type=float, default=DEF['beta'])
    parser.add_argument("--stagnation-limit", type=int, default=DEF['stagnation_limit'])
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log-file", type=str, default="data/results.csv")
    args = parser.parse_args()

    group_size = args.group_size
    evaporation_rate = args.evaporation
    iterations = args.iterations
    alpha = args.alpha
    beta = args.beta
    stagnation_limit = args.stagnation_limit
    enable_vis = not args.no_vis
    enable_log = args.log
    log_file = args.log_file

    # Problem laden
    with open("data/problem.json", "r") as f:
        problem_data = json.load(f)

    number_items = problem_data["number_items"]
    max_load = problem_data["max_load"]
    optimal_value = problem_data["optimal_solution"]["value"]

    items = [Item(d["id"], d["weight"], d["value"]) for d in problem_data["items"]]

    # Heuristik normieren + Beta-Potenz vorberechnen
    max_eta = max(item.attractiveness for item in items)
    for item in items:
        item.attractiveness /= max_eta
        item.attractiveness_beta = item.attractiveness ** beta

    # NumPy-Arrays
    item_weights = np.array([it.weight for it in items], dtype=np.float64)
    item_values = np.array([it.value for it in items], dtype=np.float64)
    attractiveness_betas = np.array([it.attractiveness_beta for it in items], dtype=np.float64)
    pheromones = np.ones(number_items, dtype=np.float64)

    if enable_vis:
        fig, ax1, ax2 = visualization.setup_live_plot()

    print(f"=== AC: {group_size} Ameisen, {iterations} Iter, alpha={alpha}, beta={beta}, rho={evaporation_rate} ===\n")

    # Tracking
    best_fitness_per_round = []
    avg_fitness_per_round = []
    global_best_value = -1
    global_best_backpack = []
    global_best_weight = 0
    global_best_iteration = -1
    stagnation_counter = 0

    # Hauptschleife
    for iteration in range(iterations):
        scores = (pheromones ** alpha) * attractiveness_betas

        backpacks, current_loads, current_values = construct_solutions_vectorized(
            group_size, item_weights, item_values, scores, max_load)

        round_best_idx = np.argmax(current_values)
        round_best_value = current_values[round_best_idx]

        if round_best_value > global_best_value:
            global_best_value = round_best_value
            global_best_backpack = backpacks[round_best_idx].tolist()
            global_best_weight = current_loads[round_best_idx]
            global_best_iteration = iteration + 1
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        best_fitness_per_round.append(round_best_value)
        avg_fitness_per_round.append(current_values.mean())

        if stagnation_limit > 0 and stagnation_counter >= stagnation_limit:
            print(f"Stagnation nach {iteration + 1} Iterationen.")
            break

        # Pheromonupdate
        rewards = backpacks * (current_values[:, None] / optimal_value)
        pheromones = pheromones * (1.0 - evaporation_rate) + rewards.sum(axis=0)

        if enable_vis:
            for j, item in enumerate(items):
                item.pheromone = float(pheromones[j])
            visualization.update_live_plot(
                fig, ax1, ax2, items, iteration + 1,
                number_items, best_fitness_per_round, avg_fitness_per_round, "AC")

    # Ergebnis
    print(f"\nBester Wert: {global_best_value} (Iter {global_best_iteration})")
    print(f"Gewicht: {global_best_weight} / {max_load}")

    if enable_vis:
        visualization.show_final()

    if enable_log:
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f, delimiter=';')
            if not file_exists:
                writer.writerow(["alpha", "beta", "evaporation", "group_size",
                                 "best_value", "best_iteration"])
            writer.writerow([alpha, beta, evaporation_rate, group_size,
                             global_best_value, global_best_iteration])

    print(f"RESULT:AC;{alpha};{beta};{evaporation_rate};{group_size};{global_best_value};{global_best_iteration}")


if __name__ == "__main__":
    main()