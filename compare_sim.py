"""Simuliert 3 ACO-Konfigurationen (vektorisiert) und speichert Ergebnisse als JSON."""

import json
import random
import numpy as np
from item import Item
from main import construct_solutions_vectorized

CONFIGS = {
    'Konfiguration A': dict(alpha=1.33, beta=25.0, evaporation=0.35, group_size=175),
    'Konfiguration B': dict(alpha=1.10, beta=8.00, evaporation=0.4, group_size=90),
    'Konfiguration C': dict(alpha=1.0, beta=4.00, evaporation=0.2, group_size=40),
}

NUM_RUNS = 10
ITERATIONS = 100
OUTPUT_FILE = "data/compare_results.json"


def run_simulation(config, problem_data, iterations=100, seed=42):
    """Einzellauf mit vektorisierter Konstruktion."""
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

    # NumPy-Arrays für vektorisierte Konstruktion
    item_weights = np.array([it.weight for it in items], dtype=np.float64)
    item_values = np.array([it.value for it in items], dtype=np.float64)
    attractiveness_betas = np.array([it.attractiveness_beta for it in items], dtype=np.float64)
    pheromones = np.ones(number_items, dtype=np.float64)

    best_fitness_curve = []
    avg_fitness_curve = []
    best_weight_curve = []
    best_val = 0
    best_backpack = None

    for _ in range(iterations):
        scores = (pheromones ** alpha) * attractiveness_betas

        # Vektorisierte Konstruktion
        backpacks, current_loads, current_values = construct_solutions_vectorized(
            group_size, item_weights, item_values, scores, max_load)

        round_best_idx = np.argmax(current_values)
        round_best_value = current_values[round_best_idx]

        best_fitness_curve.append(float(round_best_value))
        avg_fitness_curve.append(float(current_values.mean()))
        best_weight_curve.append(float(current_loads[round_best_idx]))

        if round_best_value > best_val:
            best_val = round_best_value
            best_backpack = backpacks[round_best_idx].tolist()

        # Pheromonupdate (vektorisiert)
        rewards = backpacks * (current_values[:, None] / optimal_value)
        pheromones = pheromones * (1.0 - evaporation_rate) + rewards.sum(axis=0)

    return best_fitness_curve, avg_fitness_curve, best_weight_curve, best_backpack, pheromones.tolist()


def main():
    import sys
    import os
    
    diffs = ["easy", "medium", "hard"]
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
        if choice not in diffs:
            print(f"Ungueltig: '{choice}'. Erlaubt: {', '.join(diffs)}")
            sys.exit(1)
        diffs = [choice]
        
    for diff in diffs:
        print("=" * 60)
        print(f" Simulationen fuer {diff.upper()} starten")
        print("=" * 60)
        prob_file = f"data/{diff}/problem.json"
        out_file = f"data/{diff}/compare_results.json"
        
        if not os.path.exists(prob_file):
            print(f"Fehler: Datei '{prob_file}' existiert nicht. Bitte zuerst problem.py ausfuehren.")
            continue
            
        with open(prob_file, "r") as f:
            problem_data = json.load(f)
            
        results = {}
        for name, config in CONFIGS.items():
            print(f"\n{name} (alpha={config['alpha']}, beta={config['beta']}, evap={config['evaporation']}, G={config['group_size']})")
            
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
            
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(results, f)
        print(f"\nErgebnisse gespeichert: {out_file}\n")


if __name__ == '__main__':
    main()