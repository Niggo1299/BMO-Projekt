"""Parameter-Tuning mit Optuna (Bayessche Optimierung) für ACO."""

import argparse
import subprocess
import sys
import numpy as np
import optuna
import optuna.visualization.matplotlib as vis
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_aco_instance(alpha, beta, evap, group_size):
    cmd = [
        sys.executable, "main.py",
        "--alpha", f"{alpha:.4f}", "--beta", f"{beta:.4f}",
        "--evaporation", f"{evap:.4f}", "--group-size", str(int(group_size)),
        "--iterations", "100", "--stagnation-limit", "40",
        "--no-vis"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in res.stdout.splitlines():
        if line.startswith("RESULT:"):
            return float(line.replace("RESULT:", "").strip().split(";")[5])
    return 0.0


def make_objective(repeats):
    def objective(trial):
        # Definition des optimierten Suchraums (dritter Durchlauf)
        alpha = trial.suggest_float("alpha", 0.9, 1.6)
        beta = trial.suggest_float("beta", 12.0, 25.0)
        evap = trial.suggest_float("evaporation", 0.25, 0.55)
        group_size = trial.suggest_int("group_size", 130, 200)

        results = [run_aco_instance(alpha, beta, evap, group_size) for _ in range(repeats)]
        return np.median(results)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna-Tuning für ACO")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=8, help="Anzahl paralleler CPU-Kerne")
    args = parser.parse_args()

    print(f"Optuna-Tuning: {args.trials} Trials × {args.repeats} Repeats auf {args.jobs} Kernen\n")

    study = optuna.create_study(direction="maximize")

    def callback(study, trial):
        p = trial.params
        print(f"  [{trial.number+1}/{args.trials}] {trial.value:.1f} | "
              f"alpha={p['alpha']:.3f} beta={p['beta']:.3f} evap={p['evaporation']:.3f} G={int(p['group_size'])} | "
              f"Best: {study.best_value:.1f}")

    study.optimize(make_objective(args.repeats), n_trials=args.trials, n_jobs=args.jobs, callbacks=[callback])

    print(f"\nBestes Ergebnis: {study.best_value:.1f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")

    print("\nErzeuge Visualisierungen...")
    try:
        # 1. Haupteffekte (Wichtigkeit) plotten
        vis.plot_param_importances(study)
        plt.savefig("data/optuna_importances.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Wechselwirkungen (Contour) plotten
        vis.plot_contour(study)
        fig = plt.gcf()
        fig.set_size_inches(12, 10)
        plt.savefig("data/optuna_contour.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Grafiken erfolgreich gespeichert unter 'data/optuna_importances.png' und 'data/optuna_contour.png'")
    except Exception as e:
        print(f"Fehler beim Erzeugen der Grafiken: {e}")


if __name__ == "__main__":
    main()