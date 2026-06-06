"""
Hauptprogramm: Ameisenalgorithmus für das 0/1-Rucksackproblem.

Unterstützte Varianten:
- Ant-Cycle (AC):    Folie 7 – alle Ameisen legen Pheromone ab
- Elitäres AS (EAS): Folie 9 – zusätzlicher Elite-Bonus (best-so-far)

Aufruf:
    python main.py              → Standardeinstellungen (siehe unten)
    python main.py AC           → Ant-Cycle
    python main.py EAS          → Elitäres Ameisen-System

Parameter-Überschreibung via Kommandozeile (z.B. für Optimizer):
    python main.py EAS --alpha 2.0 --beta 1.5 --no-vis --log-file results.csv
"""

import os
import csv
import random
import json
import argparse
from ant import Ant
from item import Item
import visualization


def main():
    # ===================== STANDARDEINSTELLUNGEN =====================
    DEF_MODE = "AC"
    DEF_GROUP_SIZE = 30
    DEF_EVAPORATION = 0.3
    DEF_ITERATIONS = 100
    DEF_ALPHA = 1.0
    DEF_BETA = 2.0
    DEF_ELITE_WEIGHT = 1.0
    DEF_STAGNATION_LIMIT = 60
    DEF_VISUALIZATION = False
    DEF_LOGGING = True
    DEF_LOG_FILE = "data/results.csv"

    # ===================== KOMMANDOZEILEN-PARSER =====================
    parser = argparse.ArgumentParser(description="Ameisenalgorithmus (AC/EAS)")
    parser.add_argument("mode", nargs="?", default=DEF_MODE,
                        help="Variante: AC oder EAS (Standard: AC)")
    parser.add_argument("--group-size", type=int, default=DEF_GROUP_SIZE,
                        help="Anzahl Ameisen pro Iteration")
    parser.add_argument("--evaporation", type=float, default=DEF_EVAPORATION,
                        help="Verdunstungsfaktor ρ")
    parser.add_argument("--iterations", type=int, default=DEF_ITERATIONS,
                        help="Anzahl Iterationen")
    parser.add_argument("--alpha", type=float, default=DEF_ALPHA,
                        help="Pheromon-Gewichtung α")
    parser.add_argument("--beta", type=float, default=DEF_BETA,
                        help="Heuristik-Gewichtung β")
    parser.add_argument("--elite-weight", type=float, default=DEF_ELITE_WEIGHT,
                        help="Elite-Faktor e (nur EAS, Folie 9: 0 ≤ e ≤ 1)")
    parser.add_argument("--stagnation-limit", type=int, default=DEF_STAGNATION_LIMIT,
                        help="Abbruch nach N Iterationen ohne Verbesserung (0=deaktiviert)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Live-Plot deaktivieren")
    parser.add_argument("--no-log", action="store_true",
                        help="CSV-Logging deaktivieren")
    parser.add_argument("--log-file", type=str, default=DEF_LOG_FILE,
                        help="Pfad zur Log-Datei")

    args = parser.parse_args()

    # ===================== PARAMETER ZUWEISEN =====================
    mode = args.mode.upper()
    group_size = args.group_size
    evaporation_rate = args.evaporation
    iterations = args.iterations
    alpha = args.alpha
    beta = args.beta
    elite_weight = args.elite_weight
    stagnation_limit = args.stagnation_limit

    enable_visualization = False if args.no_vis else DEF_VISUALIZATION
    enable_logging = False if args.no_log else DEF_LOGGING
    log_file = args.log_file

    # ===================== TRACKING =====================
    best_fitness_per_round = []
    avg_fitness_per_round = []

    global_best_value = -1
    global_best_backpack = []
    global_best_weight = 0
    global_best_iteration = -1
    stagnation_counter = 0

    # ===================== PROBLEM LADEN =====================
    with open("data/problem.json", "r") as f:
        problem_data = json.load(f)

    number_items = problem_data["number_items"]
    max_load = problem_data["max_load"]
    optimal_value = problem_data["optimal_solution"]["value"]

    items = []
    for data in problem_data["items"]:
        items.append(Item(data["id"], data["weight"], data["value"]))

    # Heuristik-Werte (eta) normieren: Teilen durch das Maximum im aktuellen Problem
    max_eta_yes = max(item.attractiveness_yes for item in items)
    max_eta_no = max(item.attractiveness_no for item in items)

    for item in items:
        item.attractiveness_yes /= max_eta_yes
        item.attractiveness_no /= max_eta_no
        
        # Heuristik-Potenzen vorab berechnen (beschleunigt die Ameisen-Entscheidungen um Faktor 3)
        item.attractiveness_yes_beta = item.attractiveness_yes ** beta
        item.attractiveness_no_beta = item.attractiveness_no ** beta

    # ===================== AMEISEN ERZEUGEN =====================
    ants = [Ant(max_load, number_items) for _ in range(group_size)]

    # ===================== VISUALISIERUNG SETUP =====================
    if enable_visualization:
        fig, ax1, ax2 = visualization.setup_live_plot()

    # ===================== INFO-AUSGABE =====================
    print(f"=== Modus: {mode} ===")
    if mode == "EAS":
        print(f"    Elite-Gewicht e = {elite_weight}")
    print(f"    {group_size} Ameisen, {iterations} Iterationen")
    print(f"    alpha={alpha}, beta={beta}, rho={evaporation_rate}")
    if stagnation_limit > 0:
        print(f"    Stagnationslimit: {stagnation_limit} Iterationen")
    print()

    # ===================== HAUPTSCHLEIFE =====================
    for iteration in range(iterations):

        # --- Phase 1: Lösungskonstruktion durch alle Ameisen ---
        for a in ants:
            starting_position = random.randint(0, number_items - 1)
            a.reset()

            for position in range(number_items):
                current_item = items[(starting_position + position) % number_items]
                a.decision(current_item, alpha, beta)

        # --- Phase 2: Auswertung ---
        round_best_ant = max(ants, key=lambda x: x.current_value)

        # All-Time-Best aktualisieren
        if round_best_ant.current_value > global_best_value:
            global_best_value = round_best_ant.current_value
            global_best_backpack = list(round_best_ant.backpack)
            global_best_weight = round_best_ant.current_load
            global_best_iteration = iteration + 1
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Lernkurven-Daten speichern
        avg_value = sum(a.current_value for a in ants) / group_size
        best_fitness_per_round.append(round_best_ant.current_value)
        avg_fitness_per_round.append(avg_value)

        # Frühes Abbrechen bei Stagnation
        if stagnation_limit > 0 and stagnation_counter >= stagnation_limit:
            print(f"Abbruch: Keine Verbesserung seit {stagnation_limit} Iterationen "
                  f"(Iteration {iteration + 1})")
            break

        # --- Phase 3: Pheromonupdate ---
        # Schritt 3a: Verdampfung → (1-ρ)·τ
        for current_item in items:
            current_item.evaporate(evaporation_rate)

        # Schritt 3b: Pheromonablage ALLER Ameisen → Σ Δk
        for a in ants:
            for current_item in items:
                decision = a.backpack[current_item.id]
                current_item.add_reward(decision, a.current_value, optimal_value)

        # Schritt 3c: Elitärer Bonus – NUR bei EAS (Folie 9)
        if mode == "EAS" and global_best_value > 0:
            for current_item in items:
                decision = global_best_backpack[current_item.id]
                current_item.add_reward(decision, global_best_value, optimal_value,
                                        elite_factor=elite_weight)

        # --- Visualisierung aktualisieren ---
        if enable_visualization:
            visualization.update_live_plot(
                fig, ax1, ax2, items, iteration + 1,
                number_items, best_fitness_per_round, avg_fitness_per_round, mode
            )

    # ===================== ERGEBNIS =====================
    print(f"\n=== OPTIMIERUNG ABGESCHLOSSEN ({mode}) ===")
    print(f"Bester Rucksack-Wert: {global_best_value} (Iteration {global_best_iteration})")
    print(f"Benutztes Gewicht:    {global_best_weight} / {max_load}")
    print(f"Rucksack-Belegung:    {global_best_backpack}")

    if enable_visualization:
        visualization.show_final()

    # ===================== CSV-PROTOKOLLIERUNG =====================
    if enable_logging:
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f, delimiter=';')
            if not file_exists:
                writer.writerow([
                    "mode", "alpha", "beta", "evaporation", "group_size",
                    "elite_weight", "best_value", "best_iteration"
                ])
            writer.writerow([
                mode, alpha, beta, evaporation_rate, group_size,
                elite_weight, global_best_value, global_best_iteration
            ])

    # Ergebnis-Ausgabe in standardisiertem Format für externe Skripte (z. B. Optimizer)
    print(f"RESULT:{mode};{alpha};{beta};{evaporation_rate};{group_size};{elite_weight};{global_best_value};{global_best_iteration}")


if __name__ == "__main__":
    main()