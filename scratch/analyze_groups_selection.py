import json
import random
import os
import sys
sys.path.append(os.getcwd())
from item import Item
from ant import Ant

with open("data/problem.json", "r") as f:
    problem_data = json.load(f)

number_items = problem_data["number_items"]
max_load = problem_data["max_load"]

items = []
for data in problem_data["items"]:
    items.append(Item(data["id"], data["weight"], data["value"]))

# Normalize heuristics
max_eta_yes = max(item.attractiveness_yes for item in items)
max_eta_no = max(item.attractiveness_no for item in items)
for item in items:
    item.attractiveness_yes /= max_eta_yes
    item.attractiveness_no /= max_eta_no

configs = {
    'Beste (V14)': {'alpha': 1.40, 'beta': 0.65, 'evaporation': 0.37, 'group_size': 125},
    'Mittlere (V2)': {'alpha': 2.00, 'beta': 1.50, 'evaporation': 0.25, 'group_size': 80},
    'Schlechteste (V1)': {'alpha': 4.00, 'beta': 1.00, 'evaporation': 0.30, 'group_size': 20}
}

groups = {
    "Group 1 (0-52)": (0, 52),
    "Group 2 (53-105)": (53, 105),
    "Group 3 (106-158)": (106, 158),
    "Group 4 (159-211)": (159, 211),
    "Group 5 (212-264)": (212, 264),
    "Group 6 (265-317)": (265, 317),
    "Group 7 (318-370)": (318, 370),
    "Group 8 (371-423)": (371, 423),
    "Group 9 (424-476)": (424, 476),
    "Group 10 (477-599)": (477, 599),
}

results = {}
for name, config in configs.items():
    random.seed(42)
    # Reset pheromones
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
                
    results[name] = best_overall_backpack

print("Items selected by group:")
for name, bp in results.items():
    print(f"\n{name}:")
    for g_name, (start, end) in groups.items():
        sel = sum(bp[start:end+1])
        print(f"  {g_name}: {sel} items")
