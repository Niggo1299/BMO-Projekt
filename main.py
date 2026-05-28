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

import sys
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
    # Diese Werte werden genutzt, wenn main.py ohne Argumente gestartet wird.
    # Der Optimizer überschreibt sie per Kommandozeile.
    # WICHTIG: Alle DEF_-Variablen MÜSSEN vor dem Parser definiert sein,
    #          da sie als default-Werte in add_argument() referenziert werden.
    DEF_MODE = "AC"                 # Variante: "AC" oder "EAS"
    DEF_GROUP_SIZE = 30             # Anzahl Ameisen pro Iteration (m)
    DEF_EVAPORATION = 0.5           # Verdunstungsfaktor ρ (0 ≤ ρ ≤ 1)
    DEF_ITERATIONS = 100            # Anzahl der Iterationsschritte
    DEF_ALPHA = 2.0                 # Gewichtung der Pheromonspuren (τ^α)
    DEF_BETA = 1.0                  # Gewichtung der heuristischen Information (η^β)
    DEF_ELITE_WEIGHT = 1.0          # Gewichtungsfaktor e für EAS (Folie 9: 0 ≤ e ≤ 1)
    DEF_VISUALIZATION = True       # Live-Plot an/aus
    DEF_LOGGING = False              # CSV-Protokollierung an/aus
    DEF_LOG_FILE = "results.csv"    # Name der Log-Datei

    # ===================== KOMMANDOZEILEN-PARSER =====================
    # Ermöglicht das Überschreiben aller Parameter durch externe Aufrufe
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

    # Flags: Kommandozeile (--no-vis / --no-log) überschreibt Standardeinstellung
    enable_visualization = False if args.no_vis else DEF_VISUALIZATION
    enable_logging = False if args.no_log else DEF_LOGGING
    log_file = args.log_file

    # ===================== TRACKING =====================
    best_fitness_per_round = []     # Bester Fitness-Wert pro Iteration
    avg_fitness_per_round = []      # Durchschnittlicher Fitness-Wert pro Iteration

    # All-Time-Best (best-so-far) Variablen
    global_best_value = -1
    global_best_backpack = []
    global_best_weight = 0
    global_best_iteration = -1

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
    if enable_visualization:
        fig, ax1, ax2 = visualization.setup_live_plot()

    # ===================== INFO-AUSGABE =====================
    print(f"=== Modus: {mode} ===")
    if mode == "EAS":
        print(f"    Elite-Gewicht e = {elite_weight}")
    print(f"    {group_size} Ameisen, {iterations} Iterationen")
    print(f"    alpha={alpha}, beta={beta}, rho={evaporation_rate}")
    print()

    # ===================== HAUPTSCHLEIFE =====================
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
            global_best_iteration = iteration + 1

        # Lernkurven-Daten speichern
        avg_value = sum(a.current_value for a in ants) / group_size
        best_fitness_per_round.append(round_best_ant.current_value)
        avg_fitness_per_round.append(avg_value)

        # --- Phase 3: Pheromonupdate ---
        # AC:  τ = (1-ρ)·τ + Σ Δk
        # EAS: τ = (1-ρ)·τ + Σ Δk + e·Δbs

        # Schritt 3a: Verdampfung → (1-ρ)·τ
        for current_item in items:
            current_item.evaporate(evaporation_rate)

        # Schritt 3b: Pheromonablage ALLER Ameisen → Σ Δk
        for a in ants:
            for current_item in items:
                decision = a.backpack[current_item.id]
                current_item.add_reward(decision, a.current_value, max_possible_value)

        # Schritt 3c: Elitärer Bonus – NUR bei EAS (Folie 9)
        if mode == "EAS" and global_best_value > 0:
            for current_item in items:
                decision = global_best_backpack[current_item.id]
                current_item.add_reward(
                    decision, global_best_value, max_possible_value,
                    elite_factor=elite_weight
                )

        # --- Visualisierung aktualisieren ---
        if enable_visualization:
            visualization.update_live_plot(
                fig, ax1, ax2, items, iteration + 1,
                number_items, best_fitness_per_round, avg_fitness_per_round, mode
            )

    # ===================== ERGEBNIS =====================
    print(f"=== OPTIMIERUNG ABGESCHLOSSEN ({mode}) ===")
    print(f"Bester Rucksack-Wert: {global_best_value} (Iteration {global_best_iteration})")
    print(f"Benutztes Gewicht:    {global_best_weight} / {max_load}")
    print(f"Rucksack-Belegung:    {global_best_backpack}")

    # Plot offen halten (falls aktiv)
    if enable_visualization:
        visualization.show_final()

    # ===================== CSV-PROTOKOLLIERUNG =====================
    if enable_logging:
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f, delimiter=';')
            # Header nur beim ersten Schreiben
            if not file_exists:
                writer.writerow([
                    "mode", "alpha", "beta", "evaporation", "group_size",
                    "elite_weight", "best_value", "best_iteration"
                ])
            writer.writerow([
                mode, alpha, beta, evaporation_rate, group_size,
                elite_weight, global_best_value, global_best_iteration
            ])


if __name__ == "__main__":
    main()