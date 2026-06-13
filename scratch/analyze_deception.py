import json
import numpy as np

with open("data/problem.json", "r") as f:
    problem = json.load(f)

items = problem["items"]
opt_items = set(problem["optimal_solution"]["items"])

# Calculate eta for all items
for item in items:
    item["eta"] = item["value"] / item["weight"]

# Sort items by eta descending (this is the greedy sorting order)
sorted_items = sorted(items, key=lambda x: x["eta"], reverse=True)

# Find the cutoff point for greedy
# The greedy solver packs items until the capacity is reached. Let's see how many items it packs.
greedy_capacity = 0
greedy_selected = set()
for item in sorted_items:
    if greedy_capacity + item["weight"] <= problem["max_load"]:
        greedy_selected.add(item["id"])
        greedy_capacity += item["weight"]

# Now let's analyze "False Friends" (high eta, but NOT in optimal solution)
# and "Hidden Gems" (low eta, but IN optimal solution)
print("Deceptive analysis:")
print(f"Total items in optimal solution: {len(opt_items)}")
print(f"Total items in greedy solution: {len(greedy_selected)}")

# Let's count mismatches between greedy and optimal
false_friends = [item for item in items if item["id"] not in opt_items and item["id"] in greedy_selected]
hidden_gems = [item for item in items if item["id"] in opt_items and item["id"] not in greedy_selected]

print(f"\nFalse Friends (Greedy chose them, but they are NOT in the optimum): {len(false_friends)}")
print(f"Hidden Gems (Greedy ignored them, but they ARE in the optimum): {len(hidden_gems)}")

# Let's see which groups these belong to
groups_ff = {}
groups_hg = {}
for item in false_friends:
    idx = item["id"]
    if idx <= 52: g = 1
    elif idx <= 105: g = 2
    elif idx <= 158: g = 3
    elif idx <= 211: g = 4
    elif idx <= 264: g = 5
    elif idx <= 317: g = 6
    elif idx <= 370: g = 7
    elif idx <= 423: g = 8
    elif idx <= 476: g = 9
    else: g = 10
    groups_ff[g] = groups_ff.get(g, 0) + 1

for item in hidden_gems:
    idx = item["id"]
    if idx <= 52: g = 1
    elif idx <= 105: g = 2
    elif idx <= 158: g = 3
    elif idx <= 211: g = 4
    elif idx <= 264: g = 5
    elif idx <= 317: g = 6
    elif idx <= 370: g = 7
    elif idx <= 423: g = 8
    elif idx <= 476: g = 9
    else: g = 10
    groups_hg[g] = groups_hg.get(g, 0) + 1

print("\nDistribution of False Friends by Group:")
for g in sorted(groups_ff.keys()):
    print(f"  Group {g}: {groups_ff[g]} items")

print("\nDistribution of Hidden Gems by Group:")
for g in sorted(groups_hg.keys()):
    print(f"  Group {g}: {groups_hg[g]} items")

# Print average etas of optimal vs non-optimal items within groups
print("\nAverage eta within Groups for Optimal vs Non-Optimal items:")
for g_num in range(1, 11):
    if g_num == 1: start, end = 0, 52
    elif g_num == 2: start, end = 53, 105
    elif g_num == 3: start, end = 106, 158
    elif g_num == 4: start, end = 159, 211
    elif g_num == 5: start, end = 212, 264
    elif g_num == 6: start, end = 265, 317
    elif g_num == 7: start, end = 318, 370
    elif g_num == 8: start, end = 371, 423
    elif g_num == 9: start, end = 424, 476
    else: start, end = 477, 599
    
    g_items = items[start:end+1]
    g_opt = [item for item in g_items if item["id"] in opt_items]
    g_non_opt = [item for item in g_items if item["id"] not in opt_items]
    
    avg_opt_eta = np.mean([item["eta"] for item in g_opt]) if len(g_opt) > 0 else 0
    avg_non_opt_eta = np.mean([item["eta"] for item in g_non_opt]) if len(g_non_opt) > 0 else 0
    print(f"  Group {g_num}: Opt_avg_eta = {avg_opt_eta:.5f}, Non_opt_avg_eta = {avg_non_opt_eta:.5f}")
