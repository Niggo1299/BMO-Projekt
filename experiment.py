"""
Parameter-Optimierer für den Ameisenalgorithmus.

Führt einen faktoriellen Versuchsplan durch, indem main.py
wiederholt mit verschiedenen Parameterkombinationen aufgerufen wird.

Pro Konfiguration werden mehrere unabhängige Läufe durchgeführt,
da das Verhalten der Ameisen stochastisch (probabilistisch) ist.

Ergebnis: CSV-Datei mit allen Konfigurationen und deren Fitness-Werten.

Aufruf:
    python optimizer.py
"""

import itertools
import subprocess
import sys


def run_experiments():
    """
    Führt den vollständigen faktoriellen Versuchsplan durch.
    """

    # ===================== AUSWAHL =====================
    AC = True
    EAS = False

    # ===================== PARAMETER-GRID =====================
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [1.0, 3.0, 5.0]
    evaporations = [0.1, 0.4, 0.8]
    group_sizes = [10, 40, 80]
    elite_weights = [0.2, 0.5, 1.0]

    # ===================== AUSFÜHRUNGSPARAMETER =====================
    num_runs_per_config = 3
    iterations = 200
    log_file = "data/experiment_results.csv"

    print(f"=== Parameter-Optimierung ===")
    print(f"Ergebnisse werden in '{log_file}' gespeichert.\n")

    total_ac_runs = 0
    total_eas_runs = 0

    # ===================== ANT-CYCLE (AC) =====================
    if AC:
        ac_combinations = list(itertools.product(
            alphas, betas, evaporations, group_sizes))
        total_ac_runs = len(ac_combinations) * num_runs_per_config
        print(f"--- AC: {len(ac_combinations)} Kombinationen "
              f"× {num_runs_per_config} Läufe = {total_ac_runs} Runs ---")

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
                    "--stagnation-limit", "0",
                    "--no-vis",
                    "--log-file", log_file
                ]

                subprocess.run(cmd, stdout=subprocess.DEVNULL)

                if run_counter % 10 == 0:
                    print(f"    AC Fortschritt: {run_counter}/{total_ac_runs}")

        print(f"    AC abgeschlossen: {total_ac_runs}/{total_ac_runs}\n")

    # ===================== ELITÄRES AMEISEN-SYSTEM (EAS) =====================
    if EAS:
        eas_combinations = list(itertools.product(
            alphas, betas, evaporations, group_sizes, elite_weights))
        total_eas_runs = len(eas_combinations) * num_runs_per_config
        print(f"--- EAS: {len(eas_combinations)} Kombinationen "
              f"× {num_runs_per_config} Läufe = {total_eas_runs} Runs ---")

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
                    "--stagnation-limit", "0",
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
    print("Auswertung mit: python evaluate.py")


if __name__ == "__main__":
    run_experiments()