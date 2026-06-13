import json
import numpy as np

with open("data/problem.json", "r") as f:
    problem = json.load(f)

items = problem["items"]
opt_items = set(problem["optimal_solution"]["items"])

# Group items
all_indices = list(range(len(items)))
target_range = list(range(250, 480))

print(f"Total items: {len(items)}")
print(f"Items in optimal solution: {len(opt_items)}")

# Attractiveness eta = value / weight
etas = [item["value"] / item["weight"] for item in items]
weights = [item["weight"] for item in items]
values = [item["value"] for item in items]

opt_in_target = [idx for idx in target_range if idx in opt_items]
opt_outside = [idx for idx in all_indices if idx in opt_items and idx not in target_range]

print(f"\nTarget Range (250-479):")
print(f"  Total items in range: {len(target_range)}")
print(f"  Items in optimal solution: {len(opt_in_target)} ({len(opt_in_target)/len(target_range)*100:.1f}%)")
print(f"  Average eta in range: {np.mean([etas[i] for i in target_range]):.4f}")
print(f"  Average eta of opt items in range: {np.mean([etas[i] for i in opt_in_target]):.4f}")
print(f"  Average weight in range: {np.mean([weights[i] for i in target_range]):.1f}")

outside_range = [idx for idx in all_indices if idx not in target_range]
print(f"\nOutside Target Range:")
print(f"  Total items outside: {len(outside_range)}")
print(f"  Items in optimal solution: {len(opt_outside)} ({len(opt_outside)/len(outside_range)*100:.1f}%)")
print(f"  Average eta outside: {np.mean([etas[i] for i in outside_range]):.4f}")
print(f"  Average weight outside: {np.mean([weights[i] for i in outside_range]):.1f}")
