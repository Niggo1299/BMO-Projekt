import json

with open("data/problem.json", "r") as f:
    problem = json.load(f)

items = problem["items"]
opt_items = set(problem["optimal_solution"]["items"])

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

print("Group representation in the optimal solution:")
for name, (start, end) in groups.items():
    group_size = end - start + 1
    in_opt = [idx for idx in range(start, end + 1) if idx in opt_items]
    avg_weight = sum(items[idx]["weight"] for idx in range(start, end + 1)) / group_size
    avg_val = sum(items[idx]["value"] for idx in range(start, end + 1)) / group_size
    avg_eta = sum(items[idx]["value"]/items[idx]["weight"] for idx in range(start, end + 1)) / group_size
    print(f"{name}:")
    print(f"  Total items: {group_size}")
    print(f"  In optimal solution: {len(in_opt)} ({len(in_opt)/group_size*100:.1f}%)")
    print(f"  Average weight: {avg_weight:.1f}")
    print(f"  Average value: {avg_val:.1f}")
    print(f"  Average eta: {avg_eta:.5f}")
