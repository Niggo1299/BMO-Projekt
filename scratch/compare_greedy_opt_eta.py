import json
import numpy as np

with open("data/problem.json", "r") as f:
    problem = json.load(f)

items = problem["items"]
opt_items = set(problem["optimal_solution"]["items"])

# Calculate eta
for item in items:
    item["eta"] = item["value"] / item["weight"]

# Greedy solution
sorted_items = sorted(items, key=lambda x: x["eta"], reverse=True)
greedy_capacity = 0
greedy_selected = []
for item in sorted_items:
    if greedy_capacity + item["weight"] <= problem["max_load"]:
        greedy_selected.append(item)
        greedy_capacity += item["weight"]

opt_selected = [item for item in items if item["id"] in opt_items]

avg_greedy_eta = np.mean([item["eta"] for item in greedy_selected])
avg_opt_eta = np.mean([item["eta"] for item in opt_selected])

print(f"Average eta of Greedy selection: {avg_greedy_eta:.5f}")
print(f"Average eta of Optimal selection: {avg_opt_eta:.5f}")
