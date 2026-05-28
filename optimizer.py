"""
Parameter-Optimierer für den Ameisenalgorithmus.

Führt einen faktoriellen Versuchsplan durch, indem main.py
wiederholt mit verschiedenen Parameterkombinationen aufgerufen wird.

Pro Konfiguration werden mehrere unabhängige Läufe durchgeführt,
da das Verhalten der Ameisen stochastisch (probabilistisch) ist.

Ergebnis: CSV-Datei mit allen Konfigurationen und deren Fitness-Werten.
          Auswertung z.B. mit Excel, Pandas oder einem eigenen Skript.

Aufruf:
    python optimizer.py
"""

import itertools
import subprocess
import sys


def run_experiments():
    """
    Führt den vollständigen faktoriellen Versuchsplan durch.

    Schritte:
    1. Alle Parameterkombinationen für AC generieren (ohne elite_weight)
    2. Alle Parameterkombinationen für EAS generieren (mit elite_weight)
    3. Jede Kombination num_runs_per_config Mal ausführen
    4. Ergebnisse werden von main.py in die CSV geschrieben
    """

    # ===================== PARAMETER-GRID =====================
    # Definierte Stufen für jeden Faktor (statistische Versuchsplanung)
    alphas = [1.0, 1.5, 2.0]
    betas = [1.0, 1.5, 2.0]
    evaporations = [0.1, 0.3, 0.5]
    group_sizes = [10, 20, 30]
    elite_weights = [0.2, 0.5, 1.0]    # Nur relevant für EAS

    # ===================== AUSFÜHRUNGSPARAMETER =====================
    num_runs_per_config = 5             # Wiederholungen pro Parameterkombination
    iterations = 100                     # Iterationen pro Einzellauf
    log_file = "optimization_results.csv"

    print(f"=== Parameter-Optimierung ===")
    print(f"Ergebnisse werden in '{log_file}' gespeichert.\n")

    # ===================== ANT-CYCLE (AC) =====================
    # Bei AC hat elite_weight keine Auswirkung → wird nicht variiert
    ac_combinations = list(itertools.product(alphas, betas, evaporations, group_sizes))
    total_ac_runs = len(ac_combinations) * num_runs_per_config
    print(f"--- AC: {len(ac_combinations)} Kombinationen × {num_runs_per_config} Läufe = {total_ac_runs} Runs ---")

    run_counter = 0
    for a, b, evap, gs in ac_combinations:
        for _ in range(num_runs_per_config):
            run_counter += 1

            cmd = [
                sys.executable, "main.py", "AC",
                "--alpha", str(a),
                "--beta", str(b),
                "--evaporation", str(evap),
                "--group-size", str(gs),
                "--iterations", str(iterations),
                "--no-vis",
                "--log-file", log_file
            ]

            # main.py im Hintergrund ausführen (ohne Konsolenausgabe)
            subprocess.run(cmd, stdout=subprocess.DEVNULL)

            # Fortschrittsanzeige alle 10 Läufe
            if run_counter % 10 == 0:
                print(f"    AC Fortschritt: {run_counter}/{total_ac_runs}")

    print(f"    AC abgeschlossen: {total_ac_runs}/{total_ac_runs}\n")

    # ===================== ELITÄRES AMEISEN-SYSTEM (EAS) =====================
    # Bei EAS wird zusätzlich elite_weight variiert
    eas_combinations = list(itertools.product(
        alphas, betas, evaporations, group_sizes, elite_weights
    ))
    total_eas_runs = len(eas_combinations) * num_runs_per_config
    print(f"--- EAS: {len(eas_combinations)} Kombinationen × {num_runs_per_config} Läufe = {total_eas_runs} Runs ---")

    run_counter = 0
    for a, b, evap, gs, ew in eas_combinations:
        for _ in range(num_runs_per_config):
            run_counter += 1

            cmd = [
                sys.executable, "main.py", "EAS",
                "--alpha", str(a),
                "--beta", str(b),
                "--evaporation", str(evap),
                "--group-size", str(gs),
                "--elite-weight", str(ew),
                "--iterations", str(iterations),
                "--no-vis",
                "--log-file", log_file
            ]

            subprocess.run(cmd, stdout=subprocess.DEVNULL)

            if run_counter % 10 == 0:
                print(f"    EAS Fortschritt: {run_counter}/{total_eas_runs}")

    print(f"    EAS abgeschlossen: {total_eas_runs}/{total_eas_runs}\n")

    # ===================== ZUSAMMENFASSUNG =====================
    total_runs = total_ac_runs + total_eas_runs
    print(f"=== Alle {total_runs} Experimente abgeschlossen ===")
    print(f"Ergebnisse gespeichert in: {log_file}")
    print("Auswertung z.B. mit: pandas.read_csv('{}', delimiter=';')".format(log_file))


if __name__ == "__main__":
    run_experiments()