"""
Hauptprogramm: Ant-Cycle-System (AS) für das 0/1-Rucksackproblem.

Ablauf gemäß Ant-Cycle-Variante (Folie 7):
1. Alle Ameisen konstruieren vollständige Lösungen
2. Pheromonupdate: Verdampfung + Ablage ALLER Ameisen
   (kein Elite-Bonus wie beim EAS)

Formel: τij,t+1 = (1-ρ)·τij,t + Σ Δk_ij,t
"""

import random
import json
from ant import Ant
from item import Item
import visualization


def main():
    # ===================== PARAMETER =====================
    group_size = 20             # Anzahl Ameisen pro Iteration (m)
    evaporation_rate = 0.3      # Verdunstungsfaktor ρ (0 ≤ ρ ≤ 1)
    iterations = 100            # Anzahl der Iterationsschritte
    alpha = 1.0                 # Gewichtung der Pheromonspuren (τ^α)
    beta = 2.0                  # Gewichtung der heuristischen Information (η^β)

    # ===================== TRACKING =====================
    best_fitness_per_round = []     # Bester Fitness-Wert pro Iteration
    avg_fitness_per_round = []      # Durchschnittlicher Fitness-Wert pro Iteration

    # All-Time-Best (best-so-far) Variablen
    global_best_value = -1
    global_best_backpack = []
    global_best_weight = 0

    # ===================== PROBLEM LADEN =====================
    with open("problem.json", "r") as f:
        problem_data = json.load(f)

    number_items = problem_data["number_items"]
    max_load = problem_data["max_load"]

    # Item-Objekte erzeugen
    items = []
    for data in problem_data["items"]:
        items.append(Item(data["id"], data["weight"], data["value"]))

    # Maximaler theoretischer Wert (zur Normierung der Pheromonablage)
    max_possible_value = sum(item.value for item in items)

    # ===================== AMEISEN ERZEUGEN =====================
    ants = [Ant(max_load, number_items) for _ in range(group_size)]

    # ===================== VISUALISIERUNG SETUP =====================
    fig, ax1, ax2 = visualization.setup_live_plot()

    # ===================== HAUPTSCHLEIFE (Ant-Cycle) =====================
    for iteration in range(iterations):

        # --- Phase 1: Lösungskonstruktion durch alle Ameisen ---
        for a in ants:
            # Zufällige Startposition (jede Ameise beginnt bei einem anderen Item)
            starting_position = random.randint(0, number_items - 1)
            a.reset()

            # Jede Ameise besucht alle Items genau einmal (zyklische Reihenfolge)
            for position in range(number_items):
                current_item = items[(starting_position + position) % number_items]
                a.decision(current_item, alpha, beta)

        # --- Phase 2: Auswertung ---
        # Beste Ameise dieser Iteration (iteration-best)
        round_best_ant = max(ants, key=lambda x: x.current_value)

        # All-Time-Best (best-so-far) aktualisieren
        if round_best_ant.current_value > global_best_value:
            global_best_value = round_best_ant.current_value
            global_best_backpack = list(round_best_ant.backpack)
            global_best_weight = round_best_ant.current_load

        # Lernkurven-Daten speichern
        avg_value = sum(a.current_value for a in ants) / group_size
        best_fitness_per_round.append(round_best_ant.current_value)
        avg_fitness_per_round.append(avg_value)

        # --- Phase 3: Pheromonupdate gemäß Ant-Cycle (Folie 7) ---
        # τij,t+1 = (1-ρ)·τij,t + Σ Δk_ij,t

        # Schritt 3a: Verdampfung → (1-ρ)·τij,t
        for current_item in items:
            current_item.evaporate(evaporation_rate)

        # Schritt 3b: Pheromonablage ALLER Ameisen → Σ Δk_ij,t
        for a in ants:
            for current_item in items:
                decision = a.backpack[current_item.id]
                current_item.add_reward(decision, a.current_value, max_possible_value)

        # --- Visualisierung aktualisieren ---
        visualization.update_live_plot(
            fig, ax1, ax2, items, iteration + 1,
            number_items, best_fitness_per_round, avg_fitness_per_round
        )

    # ===================== ERGEBNIS =====================
    print("=== OPTIMIERUNG ABGESCHLOSSEN ===")
    print(f"Bester Rucksack-Wert: {global_best_value}")
    print(f"Benutztes Gewicht:    {global_best_weight} / {max_load}")
    print(f"Rucksack-Belegung:    {global_best_backpack}")

    visualization.show_final()


if __name__ == "__main__":
    main()