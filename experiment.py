"""
Parameter-Optimierer für den Ameisenalgorithmus (Parallelisierte Version).

Führt einen faktoriellen Versuchsplan durch, indem main.py
wiederholt mit verschiedenen Parameterkombinationen aufgerufen wird.
Die Ausführung erfolgt parallelisiert über mehrere CPU-Kerne, um die Laufzeit zu minimieren.

Ergebnis: CSV-Datei mit allen Konfigurationen und deren Fitness-Werten.

Aufruf:
    python experiment.py
"""

import itertools
import subprocess
import sys
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_single_experiment(cmd):
    """
    Führt einen einzelnen Lauf von main.py aus und fängt das Ergebnis über stdout ab.
    """
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in res.stdout.splitlines():
        if line.startswith("RESULT:"):
            # Extrahiere die Ergebniszeile
            parts = line.replace("RESULT:", "").strip().split(";")
            return parts
    return None


def run_experiments():
    """
    Führt den vollständigen faktoriellen Versuchsplan parallelisiert durch.
    """

    # ===================== AUSWAHL =====================
    AC = True
    EAS = True

    # ===================== OPTIMIERTES PARAMETER-GRID (je 3 Varianten) =====================
    alphas = [1.0, 1.5, 2.0]
    betas = [2.0, 3.0, 5.0]
    evaporations = [0.3, 0.5, 0.8]
    group_sizes = [20, 45, 80]
    elite_weights = [0.2, 0.5, 1.0]

    # ===================== AUSFÜHRUNGSPARAMETER =====================
    num_runs_per_config = 5
    iterations = 100
    log_file = "data/experiment_results.csv"

    # Ordner erstellen, falls nicht vorhanden
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Header in CSV schreiben, falls Datei noch nicht existiert
    file_exists = os.path.isfile(log_file)
    if not file_exists:
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                "mode", "alpha", "beta", "evaporation", "group_size",
                "elite_weight", "best_value", "best_iteration"
            ])

    # ===================== TASKS VORBEREITEN =====================
    tasks = []

    # AC Kombinationen sammeln
    if AC:
        ac_combinations = list(itertools.product(alphas, betas, evaporations, group_sizes))
        for a, b, evap, gs in ac_combinations:
            for _ in range(num_runs_per_config):
                cmd = [
                    sys.executable, "main.py", "AC",
                    "--alpha", str(a),
                    "--beta", str(b),
                    "--evaporation", str(evap),
                    "--group-size", str(gs),
                    "--iterations", str(iterations),
                    "--stagnation-limit", "30",
                    "--no-vis",
                    "--no-log"  # Wir loggen direkt im Hauptthread, um Race Conditions zu vermeiden
                ]
                tasks.append(cmd)

    # EAS Kombinationen sammeln
    if EAS:
        eas_combinations = list(itertools.product(alphas, betas, evaporations, group_sizes, elite_weights))
        for a, b, evap, gs, ew in eas_combinations:
            for _ in range(num_runs_per_config):
                cmd = [
                    sys.executable, "main.py", "EAS",
                    "--alpha", str(a),
                    "--beta", str(b),
                    "--evaporation", str(evap),
                    "--group-size", str(gs),
                    "--elite-weight", str(ew),
                    "--iterations", str(iterations),
                    "--stagnation-limit", "30",
                    "--no-vis",
                    "--no-log"
                ]
                tasks.append(cmd)

    total_runs = len(tasks)
    num_cores = os.cpu_count() or 4
    print(f"=== Parameter-Optimierung ===")
    print(f"Parameter-Auswahl (je 3 Varianten):")
    print(f"  alpha:        {alphas}")
    print(f"  beta:         {betas}")
    print(f"  evaporation:  {evaporations}")
    print(f"  group_size:   {group_sizes}")
    if EAS:
        print(f"  elite_weight: {elite_weights}")
    print(f"Wiederholungen pro Konfiguration: {num_runs_per_config}")
    print(f"Ergebnisse werden in '{log_file}' gespeichert.")
    print(f"Starte {total_runs} Läufe parallel auf {num_cores} CPU-Kernen...\n")

    # ===================== PARALLELE AUSFÜHRUNG =====================
    completed_runs = 0
    
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        
        # Nutzen eines ThreadPools zur parallelen Steuerung der Subprozesse (GIL wird beim Warten auf Subprozesse freigegeben)
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            future_to_cmd = {executor.submit(run_single_experiment, cmd): cmd for cmd in tasks}
            
            for future in as_completed(future_to_cmd):
                result = future.result()
                if result:
                    # Thread-sicheres Schreiben direkt in die CSV
                    writer.writerow(result)
                    f.flush()
                
                completed_runs += 1
                if completed_runs % 50 == 0 or completed_runs == total_runs:
                    percent = (completed_runs / total_runs) * 100
                    print(f"  → Fortschritt: {completed_runs}/{total_runs} Läufe abgeschlossen ({percent:.1f}%)")

    print(f"\n=== Alle {total_runs} Experimente erfolgreich abgeschlossen ===")
    print(f"Ergebnisse gespeichert in: {log_file}")
    print("Auswertung mit: python evaluate.py")


if __name__ == "__main__":
    run_experiments()