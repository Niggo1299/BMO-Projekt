import json
import random
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from item import Item
from ant import Ant

# Load problem
with open("data/problem.json", "r") as f:
    problem_data = json.load(f)

number_items = problem_data["number_items"]
max_load = problem_data["max_load"]
optimal_items = set(problem_data["optimal_solution"]["items"])

items = []
for data in problem_data["items"]:
    items.append(Item(data["id"], data["weight"], data["value"]))

# Normalize heuristics
max_eta_yes = max(item.attractiveness_yes for item in items)
max_eta_no = max(item.attractiveness_no for item in items)
for item in items:
    item.attractiveness_yes /= max_eta_yes
    item.attractiveness_no /= max_eta_no

# Define the three configs
configs = {
    'Beste (V14)': {'alpha': 1.40, 'beta': 0.65, 'evaporation': 0.37, 'group_size': 125},
    'Mittlere (V2)': {'alpha': 2.00, 'beta': 1.50, 'evaporation': 0.25, 'group_size': 80},
    'Schlechteste (V1)': {'alpha': 4.00, 'beta': 1.00, 'evaporation': 0.30, 'group_size': 20}
}

# Run 10 runs of 100 iterations for each config and record the best solution's items
results = {}
for name, config in configs.items():
    random.seed(42)
    np.random.seed(42)
    
    # Reset items pheromones
    for item in items:
        item.pheromone_yes = 1.0
        item.pheromone_no = 1.0
        
    ants = [Ant(max_load, number_items) for _ in range(config['group_size'])]
    
    best_overall_val = 0
    best_overall_backpack = None
    
    for iteration in range(100):
        for a in ants:
            starting_position = random.randint(0, number_items - 1)
            a.reset()
            for pos in range(number_items):
                current_item = items[(starting_position + pos) % number_items]
                a.decision(current_item, config['alpha'], config['beta'])
                
        round_best_ant = max(ants, key=lambda x: x.current_value)
        if round_best_ant.current_value > best_overall_val:
            best_overall_val = round_best_ant.current_value
            best_overall_backpack = round_best_ant.backpack.copy()
            
        for item in items:
            item.evaporate(config['evaporation'])
        for a in ants:
            for item in items:
                decision = a.backpack[item.id]
                item.add_reward(decision, a.current_value, problem_data["optimal_solution"]["value"])
                
    results[name] = {
        'value': best_overall_val,
        'backpack': best_overall_backpack
    }

print("Selection analysis for items in range 250-479:")
print(f"Optimal solution: {len([idx for idx in range(250, 480) if idx in optimal_items])} items")

for name, res in results.items():
    bp = res['backpack']
    selected_in_range = [idx for idx in range(250, 480) if bp[idx] == 1]
    total_selected = sum(bp)
    total_weight = sum(items[i].weight for i in range(number_items) if bp[i] == 1)
    print(f"\n{name}:")
    print(f"  Best Value achieved: {res['value']:,}".replace(",", "."))
    print(f"  Total items selected: {total_selected}")
    print(f"  Total weight: {total_weight:,}".replace(",", "."))
    print(f"  Selected items in range 250-479: {len(selected_in_range)} ({len(selected_in_range)/230*100:.1f}%)")
    
    # Overlap with optimal solution in this range
    overlap = len([idx for idx in selected_in_range if idx in optimal_items])
    print(f"  Overlap with optimal items in range: {overlap} ({overlap/91*100:.1f}%)")
