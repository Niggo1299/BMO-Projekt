"""Simuliert 3 ACO-Konfigurationen und speichert Ergebnisse als JSON."""

import json
import random
import numpy as np
from item import Item
from ant import Ant

CONFIGS = {
    'Beste': dict(alpha=1.40, beta=0.65, evaporation=0.37, group_size=125),
    'Mittlere': dict(alpha=1.20, beta=2.00, evaporation=0.30, group_size=80),
    'Schlechteste': dict(alpha=1.00, beta=4.00, evaporation=0.20, group_size=40),
}

NUM_RUNS = 10
ITERATIONS = 100
OUTPUT_FILE = "data/compare_results.json"


def run_simulation(config, problem_data, iterations=100, seed=42):
    """Einzellauf des Ant-Cycle-Algorithmus."""
    random.seed(seed)
    np.random.seed(seed)

    alpha = config['alpha']
    beta = config['beta']
    evaporation_rate = config['evaporation']
    group_size = config['group_size']

    number_items = problem_data["number_items"]
    max_load = problem_data["max_load"]
    optimal_value = problem_data["optimal_solution"]["value"]

    items = [Item(d["id"], d["weight"], d["value"]) for d in problem_data["items"]]

    max_eta = max(item.attractiveness for item in items)
    for item in items:
        item.attractiveness /= max_eta
        item.attractiveness_beta = item.attractiveness ** beta

    ants = [Ant(max_load, number_items) for _ in range(group_size)]

    best_fitness_curve = []
    avg_fitness_curve = []
    best_weight_curve = []
    best_val = 0
    best_backpack = None

    for _ in range(iterations):
        for item in items:
            item.score = (item.pheromone ** alpha) * item.attractiveness_beta

        for a in ants:
            a.construct_solution(items, alpha, beta)

        round_best_ant = max(ants, key=lambda x: x.current_value)
        avg_value = sum(a.current_value for a in ants) / group_size

        best_fitness_curve.append(round_best_ant.current_value)
        avg_fitness_curve.append(avg_value)
        best_weight_curve.append(round_best_ant.current_load)

        if round_best_ant.current_value > best_val:
            best_val = round_best_ant.current_value
            best_backpack = round_best_ant.backpack.copy()

        for item in items:
            item.evaporate(evaporation_rate)
        for a in ants:
            for item in items:
                if a.backpack[item.id] == 1:
                    item.add_reward(a.current_value, optimal_value)

    pheromone = [item.pheromone for item in items]
    return best_fitness_curve, avg_fitness_curve, best_weight_curve, best_backpack, pheromone


def main():
    print("=" * 60)
    print(" Simulationen starten")
    print("=" * 60)

    with open("data/problem.json", "r") as f:
        problem_data = json.load(f)

    results = {}

    for name, config in CONFIGS.items():
        print(f"\n{name} (α={config['alpha']}, β={config['beta']}, ρ={config['evaporation']}, G={config['group_size']})")

        all_best, all_avg, all_weights, all_pheromone = [], [], [], []
        best_overall_val = 0
        best_overall_backpack = None

        for run in range(NUM_RUNS):
            seed = 100 + run * 37
            best_c, avg_c, weight_c, bp, phe = run_simulation(config, problem_data, ITERATIONS, seed)

            all_best.append(best_c)
            all_avg.append(avg_c)
            all_weights.append(weight_c)
            all_pheromone.append(phe)

            run_max = max(best_c)
            if run_max > best_overall_val:
                best_overall_val = run_max
                best_overall_backpack = bp

            print(f"  Lauf {run+1}/{NUM_RUNS}: {best_c[-1]:,.0f}".replace(",", "."))

        results[name] = {
            'config': config,
            'best_curve': np.mean(all_best, axis=0).tolist(),
            'avg_curve': np.mean(all_avg, axis=0).tolist(),
            'weight_curve': np.mean(all_weights, axis=0).tolist(),
            'best_backpack': best_overall_backpack,
            'pheromone': np.mean(all_pheromone, axis=0).tolist(),
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)

    print(f"\nErgebnisse gespeichert: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()