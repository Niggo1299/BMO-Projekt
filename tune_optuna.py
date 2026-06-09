"""
Parameter-Tuning mit Optuna: Intelligente Suche nach den optimalen ACO-Parametern.

Nutzt Bayessche Optimierung (Tree-structured Parzen Estimator), um die 
bestmöglichen Einstellungen für alpha, beta, evaporation, group_size und elite_weight zu finden.

Voraussetzung:
    pip install optuna

Aufruf:
    python tune_optuna.py --mode EAS --trials 50
"""

import argparse
import subprocess
import sys
import numpy as np
import optuna

# Optuna-Log-Level anpassen (weniger Rauschen in der Konsole)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_aco_instance(mode, alpha, beta, evap, group_size, elite_weight):
    """
    Führt einen einzelnen Lauf von main.py aus und fängt das Ergebnis ab.
    """
    cmd = [
        sys.executable, "main.py", mode,
        "--alpha", f"{alpha:.4f}",
        "--beta", f"{beta:.4f}",
        "--evaporation", f"{evap:.4f}",
        "--group-size", str(int(group_size)),
        "--iterations", "200",            # Ausreichend lang für Konvergenz
        "--stagnation-limit", "60",        # Schneller Abbruch bei Stagnation
        "--no-vis",
        "--no-log"
    ]
    if mode == "EAS":
        cmd += ["--elite-weight", f"{elite_weight:.4f}"]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    
    for line in res.stdout.splitlines():
        if line.startswith("RESULT:"):
            parts = line.replace("RESULT:", "").strip().split(";")
            # parts[6] ist der global_best_value
            return float(parts[6])
            
    return 0.0


def make_objective(mode, repeats):
    """
    Erzeugt die Objective-Funktion für Optuna.
    Führt repeats Wiederholungen pro Kombination aus und nutzt den Median,
    um stochastisches Rauschen zu minimieren.
    """
    def objective(trial):
        # Definition des Suchraums
        alpha = trial.suggest_float("alpha", 1.1, 1.7)
        beta = trial.suggest_float("beta", 0.6, 1.4)
        evap = trial.suggest_float("evaporation", 0.3, 0.7)
        group_size = trial.suggest_int("group_size", 80, 140)
        
        elite_weight = 1.0
        if mode == "EAS":
            elite_weight = trial.suggest_float("elite_weight", 0.1, 1.5)

        # Mehrere Durchläufe für robuste Bewertung
        results = []
        for _ in range(repeats):
            val = run_aco_instance(mode, alpha, beta, evap, group_size, elite_weight)
            results.append(val)
            
        # Median zurückgeben (resistent gegen glückliche Ausreißer)
        return np.median(results)
        
    return objective


def main():
    parser = argparse.ArgumentParser(description="Parameter-Tuning mit Optuna")
    parser.add_argument("--mode", type=str, default="AC", choices=["AC", "EAS"],
                        help="Modus: AC oder EAS (Standard: EAS)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Anzahl der Tuning-Versuche (Standard: 50)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Wiederholungen pro Versuch für stabilere Mediane (Standard: 3)")

    args = parser.parse_args()
    mode = args.mode.upper()

    print(f"{'='*60}")
    print(f" Parameter-Tuning mit Optuna ({mode})")
    print(f"{'='*60}")
    print(f"  Anzahl Versuche:   {args.trials}")
    print(f"  Runs pro Versuch:  {args.repeats} (Median)")
    print(f"  Gesamte Testläufe: {args.trials * args.repeats}")
    print("  Starte Optimierung... (Bitte warten)\n")

    # Erstelle die Studie
    study = optuna.create_study(direction="maximize")
    
    # Callback für Fortschrittsanzeige
    def print_progress(study, trial):
        p = trial.params
        params_str = f"alpha={p['alpha']:.3f}, beta={p['beta']:.3f}, evap={p['evaporation']:.3f}, size={int(p['group_size'])}"
        if "elite_weight" in p:
            params_str += f", elite={p['elite_weight']:.3f}"
            
        print(f"  [Versuch {trial.number+1}/{args.trials}] "
              f"Wert: {trial.value:.1f} ({params_str}) | "
              f"Bester bisher: {study.best_value:.1f}")

    study.optimize(make_objective(mode, args.repeats), n_trials=args.trials, callbacks=[print_progress])

    print(f"\n{'='*60}")
    print(f" OPTIMIERUNG ABGESCHLOSSEN ({mode})")
    print(f"{'='*60}")
    print(f"  Bester gefundener Nutzwert: {study.best_value:.1f}")
    print("  Optimale Parameterkonfiguration:")
    for param_name, param_val in study.best_params.items():
        print(f"    - {param_name:<15}: {param_val:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
