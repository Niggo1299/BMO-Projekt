"""
Response Surface Methode (RSM) zur Bestimmung optimaler Parameter.

Liest die median-gefilterten Scores aus evaluate.py und fittet eine
polynomiale Regressionsfläche. Anschließend wird das Maximum dieser
Fläche gesucht → optimale Parametrisierung.

Schritte:
1. evaluation_results.csv laden (bereits median-gefiltert + Score)
2. Polynomiales Regressionsmodell fitten (Response Surface)
3. Maximum der Fläche suchen (scipy.optimize.differential_evolution)
4. Sensitivitätsanalyse: Welcher Parameter hat den größten Einfluss?
5. Visualisierung der Response Surface (1D-Schnitte)

Datenfluss:
    evaluate.py → evaluation_results.csv → response_surface.py → Optimale Parameter

Aufruf:
    python response_surface.py
    python response_surface.py --mode AC
    python response_surface.py --file evaluation_results.csv --mode EAS --degree 2
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


def load_scored_data(filepath, mode):
    """
    Lädt die bewerteten Ergebnisse und filtert nach Modus.

    Die Daten enthalten bereits den Median über mehrere Runs
    (berechnet in evaluate.py), sind also robust gegen Ausreißer.

    Args:
        filepath: Pfad zur CSV-Datei (aus evaluate.py).
        mode:     "AC" oder "EAS".

    Returns:
        X:             Feature-Matrix (Parameter als Spalten).
        y:             Zielvariable (Score, basierend auf Median).
        feature_names: Liste der Parameternamen.
        df:            Gefiltertes DataFrame.
    """
    df = pd.read_csv(filepath, delimiter=';')
    df = df[df["mode"] == mode].copy()

    if df.empty:
        raise ValueError(f"Keine Daten für Modus '{mode}' gefunden.")

    # Features je nach Modus (EAS hat zusätzlich elite_weight)
    if mode == "EAS":
        feature_names = ["alpha", "beta", "evaporation", "group_size", "elite_weight"]
    else:
        feature_names = ["alpha", "beta", "evaporation", "group_size"]

    X = df[feature_names].values
    y = df["score"].values

    print(f"Daten geladen: {len(df)} Konfigurationen für {mode}")
    print(f"  Features:     {feature_names}")
    print(f"  Score-Bereich: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  (Scores basieren bereits auf Median über mehrere Runs)\n")

    return X, y, feature_names, df


def fit_response_surface(X, y, degree=2):
    """
    Fittet ein polynomiales Regressionsmodell (Response Surface).

    Pipeline:
        1. StandardScaler: Normiert Features auf Mittelwert=0, Std=1
        2. PolynomialFeatures: Erzeugt quadratische Terme + Interaktionen
        3. Ridge Regression: Lineares Modell mit L2-Regularisierung

    Polynomgrad 2 erfasst:
        - Lineare Effekte (z.B. "mehr β = besser")
        - Quadratische Effekte (z.B. "β hat ein Optimum bei 3.2")
        - Interaktionen (z.B. "hoher α braucht niedrigen ρ")

    Args:
        X:      Feature-Matrix.
        y:      Zielvariable (Score).
        degree: Grad des Polynoms (Standard: 2 = quadratisch).

    Returns:
        model: Gefittete sklearn Pipeline.
        r2:    Bestimmtheitsmaß R² (Güte des Fits).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=1.0))
    ])

    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    print(f"Response Surface gefittet (Polynomgrad {degree})")
    print(f"  → R² = {r2:.4f} (1.0 = perfekt, >0.8 = gut)")

    if r2 < 0.5:
        print("  ⚠️  Warnung: R² niedrig – Modell passt schlecht auf die Daten.")
        print("     Mögliche Ursachen: zu wenig Daten, hohe Stochastik, falscher Grad.")
    elif r2 < 0.8:
        print("  ℹ️  Hinweis: R² mäßig – Ergebnisse mit Vorsicht interpretieren.")
    print()

    return model, r2


def find_optimum(model, feature_names, mode):
    """
    Sucht das Maximum der Response Surface mittels Differential Evolution.

    Differential Evolution ist ein globaler Optimierer, der nicht in
    lokalen Maxima stecken bleibt. Er sucht den Parametersatz, der den
    höchsten vorhergesagten Score liefert.

    Parametergrenzen gehen leicht über das getestete Grid hinaus,
    um mögliche Optima zwischen/außerhalb der Gitterpunkte zu finden.

    Args:
        model:         Gefittete Pipeline.
        feature_names: Liste der Parameternamen.
        mode:          "AC" oder "EAS".

    Returns:
        optimal_params: Dictionary mit optimalen Parameterwerten.
        optimal_score:  Vorhergesagter Score am Optimum.
    """
    # Suchgrenzen (etwas breiter als das getestete Grid)
    bounds_dict = {
        "alpha":        (0.5, 3.0),
        "beta":         (0.5, 5.0),
        "evaporation":  (0.05, 0.7),
        "group_size":   (5, 50),
        "elite_weight": (0.1, 1.0)
    }

    bounds = [bounds_dict[name] for name in feature_names]

    # scipy minimiert → wir minimieren den negativen Score
    def objective(params):
        X_input = np.array(params).reshape(1, -1)
        return -model.predict(X_input)[0]

    # Globale Optimierung (robust gegen lokale Maxima)
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=1000,
        tol=1e-8
    )

    optimal_params = dict(zip(feature_names, result.x))
    optimal_score = -result.fun

    print(f"=== OPTIMALE PARAMETER ({mode}) ===")
    print(f"    Vorhergesagter Score: {optimal_score:.4f}")
    print()
    for name, value in optimal_params.items():
        if name == "group_size":
            print(f"    {name:<15} = {int(round(value))}")
        else:
            print(f"    {name:<15} = {value:.3f}")
    print()

    return optimal_params, optimal_score


def sensitivity_analysis(model, X, feature_names):
    """
    Bestimmt den relativen Einfluss jedes Parameters auf den Score.

    Methode: Varianzbasierte Sensitivität.
    Für jeden Parameter wird sein Wert auf den Mittelwert fixiert
    und die resultierende Varianzreduktion der Vorhersage gemessen.

    Hohe Varianzreduktion = Parameter hat großen Einfluss auf das Ergebnis.

    Args:
        model:         Gefittete Pipeline.
        X:             Originale Feature-Matrix.
        feature_names: Liste der Parameternamen.

    Returns:
        importance: Dictionary {parameter: relativer Einfluss [0, 1]}.
    """
    base_pred = model.predict(X)
    base_variance = np.var(base_pred)

    importance = {}

    for i, name in enumerate(feature_names):
        # Parameter i auf seinen Mittelwert fixieren (eliminiert seinen Einfluss)
        X_modified = X.copy()
        X_modified[:, i] = np.mean(X[:, i])

        modified_pred = model.predict(X_modified)
        reduced_variance = np.var(modified_pred)

        # Varianzreduktion = Anteil dieses Parameters an der Gesamtvarianz
        importance[name] = (base_variance - reduced_variance) / base_variance

    # Normieren auf Summe = 1 (relative Anteile)
    total = sum(abs(v) for v in importance.values())
    if total > 0:
        importance = {k: abs(v) / total for k, v in importance.items()}

    # Sortieren nach Einfluss (absteigend)
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print("=== SENSITIVITÄTSANALYSE ===")
    print("    Relativer Einfluss jedes Parameters auf den Score:\n")
    for name, value in importance.items():
        bar = "█" * int(value * 40)
        print(f"    {name:<15} {value:>6.1%}  {bar}")
    print()

    return importance


def plot_response_surface(model, X, feature_names, optimal_params, mode):
    """
    Visualisiert 1D-Schnitte der Response Surface.

    Für jeden Parameter wird ein Plot erstellt:
    - X-Achse: Parameterwert (variiert über seinen Bereich)
    - Y-Achse: Vorhergesagter Score
    - Alle anderen Parameter sind auf ihrem Optimum fixiert.
    - Rote Linie markiert das gefundene Optimum.

    Args:
        model:          Gefittete Pipeline.
        X:              Feature-Matrix (für Wertebereiche).
        feature_names:  Liste der Parameternamen.
        optimal_params: Optimale Parameterwerte (aus find_optimum).
        mode:           "AC" oder "EAS" (für Titel/Dateinamen).
    """
    n_features = len(feature_names)
    n_plots = min(n_features, 5)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for i in range(n_plots):
        ax = axes[i]
        name = feature_names[i]

        # Wertebereich für diesen Parameter
        param_min = X[:, i].min()
        param_max = X[:, i].max()
        # Etwas über das Grid hinaus erweitern
        margin = (param_max - param_min) * 0.1
        param_range = np.linspace(param_min - margin, param_max + margin, 200)

        # Alle anderen Parameter auf Optimum fixieren
        X_plot = np.tile(
            [optimal_params[n] for n in feature_names],
            (200, 1)
        )
        X_plot[:, i] = param_range

        # Vorhersage der Response Surface
        y_pred = model.predict(X_plot)

        # Plot
        ax.plot(param_range, y_pred, 'b-', linewidth=2)
        ax.axvline(optimal_params[name], color='red', linestyle='--', linewidth=1.5,
                   label=f'Optimum: {optimal_params[name]:.2f}')
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(f'Einfluss von {name}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Response Surface – 1D-Schnitte ({mode})', fontsize=13)
    plt.tight_layout()

    # Plot speichern und anzeigen
    output_file = f"response_surface_{mode}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot gespeichert: {output_file}")
    plt.show()


def main():
    # ===================== PARAMETER =====================
    parser = argparse.ArgumentParser(description="Response Surface Optimierung")
    parser.add_argument("--file", type=str, default="evaluation_results.csv",
                        help="Pfad zur bewerteten Ergebnis-CSV (aus evaluate.py)")
    parser.add_argument("--mode", type=str, default="EAS",
                        help="Modus: AC oder EAS")
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomgrad der Response Surface (Standard: 2)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Visualisierung deaktivieren")

    args = parser.parse_args()
    mode = args.mode.upper()

    print(f"{'='*60}")
    print(f" Response Surface Methode – {mode}")
    print(f"{'='*60}\n")

    # ===================== SCHRITT 1: DATEN LADEN =====================
    # Daten sind bereits median-gefiltert (aus evaluate.py)
    X, y, feature_names, df = load_scored_data(args.file, mode)

    # ===================== SCHRITT 2: MODELL FITTEN =====================
    model, r2 = fit_response_surface(X, y, degree=args.degree)

    # ===================== SCHRITT 3: OPTIMUM SUCHEN =====================
    optimal_params, optimal_score = find_optimum(model, feature_names, mode)

    # ===================== SCHRITT 4: SENSITIVITÄT =====================
    importance = sensitivity_analysis(model, X, feature_names)

    # ===================== SCHRITT 5: VISUALISIERUNG =====================
    if not args.no_plot:
        plot_response_surface(model, X, feature_names, optimal_params, mode)

    # ===================== ZUSAMMENFASSUNG =====================
    print(f"{'='*60}")
    print(f" ZUSAMMENFASSUNG ({mode})")
    print(f"{'='*60}")
    print(f"  Modellgüte (R²):         {r2:.4f}")
    print(f"  Optimaler Score:          {optimal_score:.4f}")
    print(f"  Einflussreichster Param:  {list(importance.keys())[0]}")
    print()
    print(f"  Empfohlene Parameter für main.py:")
    print(f"    DEF_MODE = \"{mode}\"")
    print(f"    DEF_ALPHA = {optimal_params['alpha']:.2f}")
    print(f"    DEF_BETA = {optimal_params['beta']:.2f}")
    print(f"    DEF_EVAPORATION = {optimal_params['evaporation']:.2f}")
    print(f"    DEF_GROUP_SIZE = {int(round(optimal_params['group_size']))}")
    if mode == "EAS":
        print(f"    DEF_ELITE_WEIGHT = {optimal_params['elite_weight']:.2f}")
    print()


if __name__ == "__main__":
    main()