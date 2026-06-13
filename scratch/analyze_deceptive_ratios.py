import json

with open("data/problem.json", "r") as f:
    problem = json.load(f)

items = problem["items"]
opt_items = set(problem["optimal_solution"]["items"])

# Calculate eta
for item in items:
    item["eta"] = item["value"] / item["weight"]

# Get optimal items and non-optimal items
opt_list = [item for item in items if item["id"] in opt_items]
non_opt_list = [item for item in items if item["id"] not in opt_items]

# Sort both lists by eta
opt_sorted = sorted(opt_list, key=lambda x: x["eta"])
non_opt_sorted = sorted(non_opt_list, key=lambda x: x["eta"], reverse=True)

print("Optimal items with the LOWEST eta:")
for i in range(15):
    item = opt_sorted[i]
    print(f"  ID {item['id']} (Group {item['id']//53 + 1 if item['id'] < 477 else 10}) - Weight: {item['weight']} - Value: {item['value']} - Eta: {item['eta']:.5f}")

print("\nNon-optimal items with the HIGHEST eta:")
for i in range(15):
    item = non_opt_sorted[i]
    print(f"  ID {item['id']} (Group {item['id']//53 + 1 if item['id'] < 477 else 10}) - Weight: {item['weight']} - Value: {item['value']} - Eta: {item['eta']:.5f}")
