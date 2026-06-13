import json

with open("data/problem.json", "r") as f:
    problem = json.load(f)

items = problem["items"]
opt_items = set(problem["optimal_solution"]["items"])

sorted_items = sorted(items, key=lambda x: x["value"] / x["weight"], reverse=True)

print("Top 20 items by eta (attractiveness):")
for i in range(20):
    item = sorted_items[i]
    ratio = item["value"] / item["weight"]
    in_opt = "YES" if item["id"] in opt_items else "NO"
    # Find which group this item belongs to
    # Group ranges: 
    # Group 1: 0-52, Group 2: 53-105, Group 3: 106-158, Group 4: 159-211, Group 5: 212-264,
    # Group 6: 265-317, Group 7: 318-370, Group 8: 371-423, Group 9: 424-476, Group 10: 477-599
    group = "Unknown"
    idx = item["id"]
    if idx <= 52: group = "Group 1"
    elif idx <= 105: group = "Group 2"
    elif idx <= 158: group = "Group 3"
    elif idx <= 211: group = "Group 4"
    elif idx <= 264: group = "Group 5"
    elif idx <= 317: group = "Group 6"
    elif idx <= 370: group = "Group 7"
    elif idx <= 423: group = "Group 8"
    elif idx <= 476: group = "Group 9"
    else: group = "Group 10 (small)"
    
    print(f"Rank {i+1}: ID {item['id']} ({group}) - Weight: {item['weight']} - Value: {item['value']} - Eta: {ratio:.6f} - In Optimal Solution: {in_opt}")
