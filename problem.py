"""NMGE-Rucksackproblem-Generator mit exakter DP-Lösung."""

import random
import json
import math
import sys
import time


DIFFICULTY_CONFIGS = {
    "easy": dict(n=400, c=1_000_000, g=2, f=0.2, epsilon=0.1, s=100, b=2, seed=1,
                 filename="data/problem.json"),
    "medium": dict(n=600, c=1_000_000, g=10, f=0.2, epsilon=0.001, s=200, b=2, seed=2,
                   filename="data/problem.json"),
    "hard": dict(n=800, c=1_000_000, g=14, f=0.1, epsilon=0.00001, s=300, b=2, seed=3,
                 filename="data/problem.json"),
}


def solve_knapsack_dp(items, capacity):
    n = len(items)
    dp = [0] * (capacity + 1)
    chosen = [[False] * (capacity + 1) for _ in range(n)]

    for i in range(n):
        wi, vi = items[i]["weight"], items[i]["value"]
        for w in range(capacity, wi - 1, -1):
            if dp[w - wi] + vi > dp[w]:
                dp[w] = dp[w - wi] + vi
                chosen[i][w] = True

    optimal_value = dp[capacity]
    optimal_items = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if chosen[i][w]:
            optimal_items.append(items[i]["id"])
            w -= items[i]["weight"]

    optimal_items.reverse()
    total_weight = sum(items[i]["weight"] for i in range(n) if items[i]["id"] in optimal_items)
    return optimal_value, optimal_items, total_weight


def solve_knapsack_greedy(items, capacity):
    sorted_items = sorted(items, key=lambda x: x["value"] / x["weight"], reverse=True)
    total_value, total_weight, selected = 0, 0, []

    for item in sorted_items:
        if total_weight + item["weight"] <= capacity:
            selected.append(item["id"])
            total_value += item["value"]
            total_weight += item["weight"]

    return total_value, selected, total_weight


def generate_problem(n, c, g, f, epsilon, s, b, filename, seed):
    random.seed(seed)

    last_group_size = math.floor(n * f)
    m = math.floor((n - last_group_size) / (g - 1))
    last_group_size = n - (g - 1) * m

    items_data = []
    item_id = 0

    for group_i in range(1, g):
        base_value = math.floor((1.0 / (b ** group_i) + epsilon) * c)
        for _ in range(m):
            r1, r2 = random.randint(1, s), random.randint(1, s)
            items_data.append({"id": item_id, "weight": base_value + r2, "value": base_value + r1})
            item_id += 1

    for _ in range(last_group_size):
        items_data.append({"id": item_id, "weight": random.randint(1, s), "value": random.randint(1, s)})
        item_id += 1

    print(f"  DP-Lösung berechnen...", end=" ", flush=True)
    start = time.time()
    optimal_value, optimal_items, optimal_weight = solve_knapsack_dp(items_data, c)
    print(f"fertig ({time.time() - start:.2f}s)")

    greedy_value, greedy_items, greedy_weight = solve_knapsack_greedy(items_data, c)
    greedy_gap = (1 - greedy_value / optimal_value) * 100 if optimal_value > 0 else 0

    problem = {
        "number_items": n, "max_load": c, "items": items_data,
        "optimal_solution": {"value": optimal_value, "weight": optimal_weight,
                             "items": optimal_items, "num_items_selected": len(optimal_items)},
        "greedy_solution": {"value": greedy_value, "weight": greedy_weight,
                            "gap_percent": round(greedy_gap, 2)},
    }

    with open(filename, "w") as fh:
        json.dump(problem, fh, indent=4)

    print(f"  Gespeichert: '{filename}' | n={n}, c={c:,}, g={g}")
    print(f"  OPTIMUM: {optimal_value:,} | Greedy-Gap: {greedy_gap:.2f}%")
    return optimal_value


def main():
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
        if choice not in DIFFICULTY_CONFIGS:
            print(f"Ungültig: '{choice}'. Verfügbar: {', '.join(DIFFICULTY_CONFIGS.keys())}")
            sys.exit(1)
        print(f"\n>>> {choice.upper()}-Instanz...")
        generate_problem(**DIFFICULTY_CONFIGS[choice])
    else:
        for difficulty, config in DIFFICULTY_CONFIGS.items():
            print(f"\n>>> {difficulty.upper()}-Instanz...")
            generate_problem(**config)


if __name__ == "__main__":
    main()