"""
Generiert 0/1-Rucksackprobleme nach der NMGE-Methode
(Jooken, Leyman & De Causmaecker, 2022)
und berechnet die optimale Lösung per Dynamic Programming.

Aufruf:
    python problem.py              → Alle drei Schwierigkeitsstufen
    python problem.py easy         → Nur leicht
    python problem.py medium       → Nur mittel
    python problem.py hard         → Nur schwer
"""

import random
import json
import math
import sys
import time


DIFFICULTY_CONFIGS = {
    # Leicht: Theorem 2 greift häufig, viele dominierte Items
    # Paper: g=2 oder g=6, grosses epsilon, n niedrig
    "easy": {
        "n": 400,
        "c": 1_000_000,     # c=10^6 (Paper-Minimum, DP-machbar)
        "g": 2,             # Paper: g=2 → Theorem 2 greift oft
        "f": 0.2,           # Kaum Einfluss laut Paper
        "epsilon": 0.1,     # Grosses eps → bricht harte Bedingungen
        "s": 100,           # Paper: kaum Einfluss
        "b": 2,
        "seed": 1,
        "filename": "data/problem.json",
    },
    # Mittel: Theorem-2-Reduktion sinkt, Suchraum wächst
    # Paper: g=10, epsilon=10^-3, n=600-800
    # Hinweis: c=10^8 wäre Paper-konform, aber DP braucht dann
    # ~800MB RAM. Wir nutzen c=10^6 und erhöhen g+n stattdessen.
    "medium": {
        "n": 600,
        "c": 1_000_000,     # c=10^6 (DP-Limit in Python)
        "g": 10,            # Paper: g=10 → kaum Reduktion möglich
        "f": 0.2,
        "epsilon": 0.001,   # eps=10^-3
        "s": 200,           # Paper-Empfehlung für medium
        "b": 2,
        "seed": 2,
        "filename": "data/problem.json",
    },
    # Schwer: Maximale kombinatorische Härte
    # Paper: g=14, epsilon=10^-5 (härter als eps=0!), n=1000+
    # Hinweis: c=10^10 wäre Paper-konform, ist aber unmöglich
    # für Python-DP (~80GB RAM). n=800 statt 1000+ wegen Laufzeit.
    "hard": {
        "n": 800,
        "c": 1_000_000,     # c=10^6 (DP-Limit)
        "g": 14,            # Paper: g=14 → keine Reduktion mehr
        "f": 0.1,           # Paper: f in {0.1, 0.2, 0.3}
        "epsilon": 0.00001, # eps=10^-5 (härtester Wert laut Paper)
        "s": 300,           # Paper-Empfehlung für hard
        "b": 2,
        "seed": 3,
        "filename": "data/problem.json",
    },
}


def solve_knapsack_dp(items, capacity):
    """
    Löst das 0/1-Rucksackproblem exakt per Dynamic Programming.

    Args:
        items:    Liste von Dicts mit 'weight' und 'value'.
        capacity: Maximale Kapazität.

    Returns:
        optimal_value:  Bester erreichbarer Nutzwert.
        optimal_items:  Liste der IDs der gewählten Items.
        total_weight:   Gesamtgewicht der optimalen Lösung.
    """
    n = len(items)

    dp = [0] * (capacity + 1)
    chosen = [[False] * (capacity + 1) for _ in range(n)]

    for i in range(n):
        wi = items[i]["weight"]
        vi = items[i]["value"]
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
    total_weight = sum(items[i]["weight"] for i in range(n)
                       if items[i]["id"] in optimal_items)

    return optimal_value, optimal_items, total_weight


def solve_knapsack_greedy(items, capacity):
    """
    Greedy-Lösung (nach Profit/Gewicht-Verhältnis) als untere Schranke.

    Returns:
        greedy_value, greedy_items, total_weight
    """
    sorted_items = sorted(items, key=lambda x: x["value"] / x["weight"],
                          reverse=True)

    total_value = 0
    total_weight = 0
    selected = []

    for item in sorted_items:
        if total_weight + item["weight"] <= capacity:
            selected.append(item["id"])
            total_value += item["value"]
            total_weight += item["weight"]

    return total_value, selected, total_weight


def generate_problem(n, c, g, f, epsilon, s, b, filename, seed):
    """
    Erzeugt eine NMGE-Instanz und berechnet die optimale Lösung.
    """
    random.seed(seed)

    last_group_size = math.floor(n * f)
    m = math.floor((n - last_group_size) / (g - 1))
    last_group_size = n - (g - 1) * m

    items_data = []
    item_id = 0

    # Gruppen 1 bis g-1: große Items mit Ratio ≈ 1
    for group_i in range(1, g):
        base_value = math.floor((1.0 / (b ** group_i) + epsilon) * c)
        for _ in range(m):
            r1 = random.randint(1, s)
            r2 = random.randint(1, s)
            items_data.append({
                "id": item_id,
                "weight": base_value + r2,
                "value": base_value + r1,
            })
            item_id += 1

    # Gruppe g: kleine unkorrelierte Items
    for _ in range(last_group_size):
        items_data.append({
            "id": item_id,
            "weight": random.randint(1, s),
            "value": random.randint(1, s),
        })
        item_id += 1

    # Optimale Lösung berechnen
    print(f"  Berechne optimale Lösung (DP)...", end=" ", flush=True)
    start = time.time()
    optimal_value, optimal_items, optimal_weight = solve_knapsack_dp(items_data, c)
    dp_time = time.time() - start
    print(f"fertig ({dp_time:.2f}s)")

    # Greedy zum Vergleich
    greedy_value, greedy_items, greedy_weight = solve_knapsack_greedy(items_data, c)
    greedy_gap = ((1 - greedy_value / optimal_value) * 100
                  if optimal_value > 0 else 0)

    # Problem mit Lösung speichern
    problem = {
        "number_items": n,
        "max_load": c,
        "items": items_data,
        "optimal_solution": {
            "value": optimal_value,
            "weight": optimal_weight,
            "items": optimal_items,
            "num_items_selected": len(optimal_items),
        },
        "greedy_solution": {
            "value": greedy_value,
            "weight": greedy_weight,
            "gap_percent": round(greedy_gap, 2),
        },
    }

    with open(filename, "w") as fh:
        json.dump(problem, fh, indent=4)

    # Zusammenfassung ausgeben
    print(f"\n{'='*55}")
    print(f"  Problem gespeichert: '{filename}'")
    print(f"{'='*55}")
    print(f"  Items:            {n}")
    print(f"  Kapazität:        {c:,}")
    print(f"  Gruppen:          {g}")
    print(f"  Epsilon:          {epsilon}")
    print(f"{'─'*55}")
    for group_i in range(1, g):
        base = math.floor((1.0 / (b ** group_i) + epsilon) * c)
        print(f"  Gruppe {group_i}: {m:>3} Items, Gewicht ≈ {base:,} ± {s}")
    print(f"  Gruppe {g}: {last_group_size:>3} Items, Gewicht 1–{s}")
    print(f"{'─'*55}")
    print(f"  ★ OPTIMUM:        {optimal_value:,} (Gewicht: {optimal_weight:,})")
    print(f"    Items gewählt:  {len(optimal_items)} von {n}")
    print(f"  ◇ Greedy:         {greedy_value:,} (Gap: {greedy_gap:.2f}%)")
    print(f"{'='*55}")

    return optimal_value


def main():
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
        if choice not in DIFFICULTY_CONFIGS:
            print(f"Ungültige Auswahl: '{choice}'")
            print(f"Verfügbar: {', '.join(DIFFICULTY_CONFIGS.keys())}")
            sys.exit(1)
        print(f"\n>>> Erzeuge {choice.upper()}-Instanz...")
        generate_problem(**DIFFICULTY_CONFIGS[choice])
    else:
        results = {}
        for difficulty, config in DIFFICULTY_CONFIGS.items():
            print(f"\n>>> Erzeuge {difficulty.upper()}-Instanz...")
            opt = generate_problem(**config)
            results[difficulty] = opt

        print(f"\n\n{'='*55}")
        print("  ZUSAMMENFASSUNG")
        print(f"{'='*55}")
        for diff, opt in results.items():
            print(f"  {diff:>6}: Optimum = {opt:,}")
        print(f"{'='*55}")
        print("  Dateien: problem.json")


if __name__ == "__main__":
    main()