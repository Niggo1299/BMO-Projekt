"""Parameter-Tuning mit Optuna (Bayessche Optimierung) für ACO."""

import argparse
import subprocess
import sys
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_aco_instance(alpha, beta, evap, group_size):
    cmd = [
        sys.executable, "main.py",
        "--alpha", f"{alpha:.4f}", "--beta", f"{beta:.4f}",
        "--evaporation", f"{evap:.4f}", "--group-size", str(int(group_size)),
        "--iterations", "150", "--stagnation-limit", "40",
        "--no-vis", "--no-log"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in res.stdout.splitlines():
        if line.startswith("RESULT:"):
            return float(line.replace("RESULT:", "").strip().split(";")[5])
    return 0.0


def make_objective(repeats):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.5, 2.0)
        beta = trial.suggest_float("beta", 1.0, 6.0)
        evap = trial.suggest_float("evaporation", 0.4, 0.95)
        group_size = trial.suggest_int("group_size", 30, 100)

        results = [run_aco_instance(alpha, beta, evap, group_size) for _ in range(repeats)]
        return np.median(results)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna-Tuning für ACO")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print(f"Optuna-Tuning: {args.trials} Trials × {args.repeats} Repeats\n")

    study = optuna.create_study(direction="maximize")

    def callback(study, trial):
        p = trial.params
        print(f"  [{trial.number+1}/{args.trials}] {trial.value:.1f} | "
              f"α={p['alpha']:.3f} β={p['beta']:.3f} ρ={p['evaporation']:.3f} G={int(p['group_size'])} | "
              f"Best: {study.best_value:.1f}")

    study.optimize(make_objective(args.repeats), n_trials=args.trials, callbacks=[callback])

    print(f"\nBestes Ergebnis: {study.best_value:.1f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()