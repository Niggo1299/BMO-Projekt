"""
Response Surface Methode (RSM) zur Bestimmung optimaler Parameter.

Schritte:
1. evaluation_results.csv laden (bereits median-gefiltert)
2. Polynomiales Regressionsmodell fitten (Response Surface)
3. Cross-Validation zur Modellbewertung
4. Maximum der Fläche suchen (scipy.optimize.differential_evolution)
5. Sensitivitätsanalyse: Welcher Parameter hat den größten Einfluss?
6. Visualisierung:
   - 1D-Schnitte mit Datenpunkten und Konfidenzband
   - 2D-Contour-Plots (Heatmaps) für Parameterpaare
   - Residuenplot zur Modelldiagnostik

Aufruf:
    python response_surface.py
    python response_surface.py --mode AC
    python response_surface.py --file evaluation_results.csv --mode EAS --degree 2
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from itertools import combinations


def load_optimal_value(problem_path="data/problem.json"):
    """
    Lädt den optimalen Lösungswert aus der Problemdefinition.

    Args:
        problem_path: Pfad zur problem.json Datei.

    Returns:
        Optimaler Nutzwert (int), oder None falls Datei nicht existiert.
    """
    try:
        with open(problem_path, "r") as f:
            problem = json.load(f)
        return problem["optimal_solution"]["value"]
    except (FileNotFoundError, KeyError):
        print("  ⚠️  problem.json nicht gefunden – kein Optimum-Bezug möglich.")
        return None


def load_scored_data(filepath, mode, optimal_value=None):
    """
    Lädt die bewerteten Ergebnisse und filtert nach Modus.

    Zielvariable: median_value (direkte Lösungsqualität).
    Falls Optimum bekannt: Zusätzlich prozentuale Performance berechnen.

    Args:
        filepath:      Pfad zur CSV-Datei (aus evaluate.py).
        mode:          "AC" oder "EAS".
        optimal_value: Bekanntes Optimum aus problem.json.

    Returns:
        X:             Feature-Matrix (Parameter als Spalten).
        y:             Zielvariable (median_value).
        feature_names: Liste der Parameternamen.
        df:            Gefiltertes DataFrame.
    """
    df = pd.read_csv(filepath, delimiter=';')
    df = df[df["mode"] == mode].copy()

    if df.empty:
        raise ValueError(f"Keine Daten für Modus '{mode}' gefunden.")

    if mode == "EAS":
        feature_names = ["alpha", "beta", "evaporation", "group_size",
                         "elite_weight"]
    else:
        feature_names = ["alpha", "beta", "evaporation", "group_size"]

    X = df[feature_names].values
    y = df["median_value"].values

    print(f"Daten geladen: {len(df)} Konfigurationen für {mode}")
    print(f"  Features:        {feature_names}")
    print(f"  Median-Bereich:  [{y.min():.0f}, {y.max():.0f}]")

    if optimal_value is not None:
        pct_min = y.min() / optimal_value * 100
        pct_max = y.max() / optimal_value * 100
        print(f"  % vom Optimum:   [{pct_min:.1f}%, {pct_max:.1f}%]"
              f"  (Optimum = {optimal_value})")
        df["pct_optimal"] = df["median_value"] / optimal_value * 100
    print()

    return X, y, feature_names, df


def fit_response_surface(X, y, degree=2):
    """
    Fittet ein polynomiales Regressionsmodell (Response Surface).

    Pipeline:
        1. StandardScaler: Normiert Features auf Mittelwert=0, Std=1
        2. PolynomialFeatures: Erzeugt quadratische Terme + Interaktionen
        3. RidgeCV: Ridge mit automatischer Alpha-Optimierung

    Modellgüte wird durch 5-Fold Cross-Validation validiert.

    Args:
        X:      Feature-Matrix.
        y:      Zielvariable (median_value).
        degree: Grad des Polynoms (Standard: 2).

    Returns:
        model:  Gefittete sklearn Pipeline.
        r2:     Bestimmtheitsmaß R² (Trainingsdaten).
        r2_cv:  Mittleres R² aus 5-Fold Cross-Validation.
    """
    alphas = np.logspace(-3, 3, 50)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", RidgeCV(alphas=alphas, cv=5))
    ])

    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Cross-Validation für ehrliche Modellbewertung
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    r2_cv = cv_scores.mean()
    r2_cv_std = cv_scores.std()

    best_alpha = model.named_steps["ridge"].alpha_

    print(f"Response Surface gefittet (Polynomgrad {degree})")
    print(f"  Ridge-Alpha:     {best_alpha:.4f} (automatisch optimiert)")
    print(f"  -> R^2 (Training): {r2:.4f}")
    print(f"  -> R^2 (5-Fold CV): {r2_cv:.4f} +/- {r2_cv_std:.4f}")

    if r2_cv < 0.3:
        print("  WARNUNG: CV-R^2 sehr niedrig -- Modell hat kaum"
              " Vorhersagekraft.")
        print("     Mögliche Ursachen: zu wenig Daten, zu viel Rauschen,"
              " oder Parameter haben wenig Einfluss.")
    elif r2_cv < 0.6:
        print("  HINWEIS: CV-R^2 maessig -- mit Vorsicht interpretieren.")
    else:
        print("  OK: CV-R^2 gut -- Modell ist aussagekraeftig.")

    if abs(r2 - r2_cv) > 0.15:
        print("  WARNUNG: Grosse Differenz R^2_train vs. R^2_CV"
              " -> moegliches Overfitting!")
    print()

    return model, r2, r2_cv


def find_optimum(model, feature_names, mode):
    """
    Sucht das Maximum der Response Surface mittels Differential Evolution.

    Args:
        model:         Gefittete Pipeline.
        feature_names: Liste der Parameternamen.
        mode:          "AC" oder "EAS".

    Returns:
        optimal_params: Dictionary mit optimalen Parameterwerten.
        optimal_score:  Vorhergesagter Score am Optimum.
    """
    bounds_dict = {
        "alpha":        (0.5, 3.0),
        "beta":         (0.5, 5.0),
        "evaporation":  (0.05, 0.8),
        "group_size":   (5, 80),
        "elite_weight": (0.1, 1.0)
    }

    bounds = [bounds_dict[name] for name in feature_names]

    def objective(params):
        X_input = np.array(params).reshape(1, -1)
        return -model.predict(X_input)[0]

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=1000,
        tol=1e-8
    )

    optimal_params = dict(zip(feature_names, result.x))
    optimal_value = -result.fun

    print(f"=== OPTIMALE PARAMETER ({mode}) ===")
    print(f"    Vorhergesagter median_value: {optimal_value:.1f}")
    print()
    for name, value in optimal_params.items():
        if name == "group_size":
            print(f"    {name:<15} = {int(round(value))}")
        else:
            print(f"    {name:<15} = {value:.3f}")
    print()

    return optimal_params, optimal_value


def sensitivity_analysis(model, X, feature_names):
    """
    Bestimmt den relativen Einfluss jedes Parameters auf den median_value.

    Methode: Varianzbasierte Sensitivität.

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
        X_modified = X.copy()
        X_modified[:, i] = np.mean(X[:, i])

        modified_pred = model.predict(X_modified)
        reduced_variance = np.var(modified_pred)

        importance[name] = (base_variance - reduced_variance) / base_variance

    total = sum(abs(v) for v in importance.values())
    if total > 0:
        importance = {k: abs(v) / total for k, v in importance.items()}

    importance = dict(sorted(importance.items(), key=lambda x: x[1],
                             reverse=True))

    print("=== SENSITIVITÄTSANALYSE ===")
    print("    Relativer Einfluss jedes Parameters auf median_value:\n")
    for name, value in importance.items():
        bar = "█" * int(value * 40)
        print(f"    {name:<15} {value:>6.1%}  {bar}")
    print()

    return importance


def plot_1d_slices(model, X, y, feature_names, optimal_params, mode):
    """
    Visualisiert 1D-Schnitte der Response Surface mit Datenpunkten.

    Args:
        model:          Gefittete Pipeline.
        X:              Feature-Matrix (für Wertebereiche).
        y:              Zielvariable (median_value).
        feature_names:  Liste der Parameternamen.
        optimal_params: Optimale Parameterwerte.
        mode:           "AC" oder "EAS".
    """
    n_features = len(feature_names)
    n_plots = min(n_features, 5)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    for i in range(n_plots):
        ax = axes[i]
        name = feature_names[i]

        param_min = X[:, i].min()
        param_max = X[:, i].max()
        margin = (param_max - param_min) * 0.1
        param_range = np.linspace(param_min - margin, param_max + margin, 200)

        # Response Surface Kurve
        X_plot = np.tile(
            [optimal_params[n] for n in feature_names],
            (200, 1)
        )
        X_plot[:, i] = param_range
        y_pred = model.predict(X_plot)

        ax.plot(param_range, y_pred, 'b-', linewidth=2,
                label='Response Surface')

        # Datenpunkte einblenden (tatsächliche Messwerte)
        ax.scatter(X[:, i], y, alpha=0.3, s=15, color='gray', zorder=5,
                   label='Datenpunkte')

        # Optimum markieren
        ax.axvline(optimal_params[name], color='red', linestyle='--',
                   linewidth=1.5,
                   label=f'Optimum: {optimal_params[name]:.2f}')

        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('median_value', fontsize=11)
        ax.set_title(f'Einfluss von {name}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Response Surface – 1D-Schnitte ({mode})', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()

    output_file = f"data/response_surface_{mode}_1d.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"1D-Plot gespeichert: {output_file}")
    plt.show()


def plot_2d_contours(model, X, feature_names, optimal_params, importance,
                     mode, n_best_pairs=4):
    """
    Visualisiert 2D-Contour-Plots (Heatmaps) für die einflussreichsten
    Parameterpaare.

    Zeigt Interaktionseffekte, die in 1D-Schnitten unsichtbar sind.

    Args:
        model:          Gefittete Pipeline.
        X:              Feature-Matrix.
        feature_names:  Liste der Parameternamen.
        optimal_params: Optimale Parameterwerte.
        importance:     Sensitivitäts-Dictionary.
        mode:           "AC" oder "EAS".
        n_best_pairs:   Anzahl der zu plottenden Paare.
    """
    # Alle Parameterpaare, sortiert nach kombinierter Wichtigkeit
    pairs = list(combinations(range(len(feature_names)), 2))
    pair_importance = []
    for i, j in pairs:
        combined = (importance.get(feature_names[i], 0) +
                    importance.get(feature_names[j], 0))
        pair_importance.append((i, j, combined))

    pair_importance.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pair_importance[:n_best_pairs]

    n_cols = min(len(top_pairs), 2)
    n_rows = (len(top_pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5.5 * n_rows))
    if len(top_pairs) == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (i, j, _) in enumerate(top_pairs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        name_i = feature_names[i]
        name_j = feature_names[j]

        # Gitter erstellen
        xi_min, xi_max = X[:, i].min(), X[:, i].max()
        xj_min, xj_max = X[:, j].min(), X[:, j].max()
        margin_i = (xi_max - xi_min) * 0.05
        margin_j = (xj_max - xj_min) * 0.05

        xi_range = np.linspace(xi_min - margin_i, xi_max + margin_i, 80)
        xj_range = np.linspace(xj_min - margin_j, xj_max + margin_j, 80)
        XI, XJ = np.meshgrid(xi_range, xj_range)

        # Vorhersagen auf dem Gitter
        X_grid = np.tile(
            [optimal_params[n] for n in feature_names],
            (XI.size, 1)
        )
        X_grid[:, i] = XI.ravel()
        X_grid[:, j] = XJ.ravel()

        Z = model.predict(X_grid).reshape(XI.shape)

        # Contour-Plot
        contour = ax.contourf(XI, XJ, Z, levels=20, cmap='RdYlGn')
        fig.colorbar(contour, ax=ax, label='median_value', shrink=0.85)

        # Datenpunkte einblenden
        ax.scatter(X[:, i], X[:, j], c='black', s=10, alpha=0.4,
                   zorder=5, label='Datenpunkte')

        # Optimum markieren
        ax.plot(optimal_params[name_i], optimal_params[name_j],
                'r*', markersize=15, zorder=10, label='Optimum')

        ax.set_xlabel(name_i, fontsize=11)
        ax.set_ylabel(name_j, fontsize=11)
        ax.set_title(f'{name_i} × {name_j}', fontsize=12)
        ax.legend(fontsize=8, loc='lower right')

    # Leere Subplots ausblenden
    for idx in range(len(top_pairs), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f'Response Surface – 2D-Contour-Plots ({mode})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = f"data/response_surface_{mode}_2d.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"2D-Plot gespeichert: {output_file}")
    plt.show()


def plot_residuals(model, X, y, mode):
    """
    Residuenplot zur Modelldiagnostik.

    Zeigt, ob das Modell systematische Fehler macht.

    Args:
        model: Gefittete Pipeline.
        X:     Feature-Matrix.
        y:     Tatsächliche Werte.
        mode:  "AC" oder "EAS".
    """
    y_pred = model.predict(X)
    residuals = y - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Residuen vs. Vorhersage
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Vorhergesagter median_value', fontsize=11)
    axes[0].set_ylabel('Residuen (Ist − Modell)', fontsize=11)
    axes[0].set_title('Residuen vs. Vorhersage', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Ist vs. Vorhersage
    val_min = min(y.min(), y_pred.min())
    val_max = max(y.max(), y_pred.max())
    axes[1].scatter(y, y_pred, alpha=0.5, s=20, color='steelblue')
    axes[1].plot([val_min, val_max], [val_min, val_max], 'r--', linewidth=1,
                 label='Perfektes Modell')
    axes[1].set_xlabel('Tatsächlicher median_value', fontsize=11)
    axes[1].set_ylabel('Vorhergesagter median_value', fontsize=11)
    axes[1].set_title('Ist vs. Vorhersage', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Modelldiagnostik – Residuenanalyse ({mode})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = f"data/response_surface_{mode}_residuals.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Residuenplot gespeichert: {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Response Surface Optimierung")
    parser.add_argument("--file", type=str, default="data/evaluation_results.csv",
                        help="Pfad zur bewerteten Ergebnis-CSV")
    parser.add_argument("--mode", type=str, default="EAS",
                        help="Modus: AC oder EAS")
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomgrad der Response Surface")
    parser.add_argument("--no-plot", action="store_true",
                        help="Visualisierung deaktivieren")

    args = parser.parse_args()
    mode = args.mode.upper()

    print(f"{'='*60}")
    print(f" Response Surface Methode – {mode}")
    print(f"{'='*60}\n")

    # Optimum laden
    optimal_value = load_optimal_value()

    # Daten laden (Zielvariable: median_value)
    X, y, feature_names, df = load_scored_data(args.file, mode, optimal_value)

    # Modell fitten mit Cross-Validation
    model, r2, r2_cv = fit_response_surface(X, y, degree=args.degree)

    # Optimum suchen
    optimal_params, predicted_value = find_optimum(model, feature_names, mode)

    # Sensitivitätsanalyse
    importance = sensitivity_analysis(model, X, feature_names)

    # Visualisierungen
    if not args.no_plot:
        plot_1d_slices(model, X, y, feature_names, optimal_params, mode)
        plot_2d_contours(model, X, feature_names, optimal_params, importance,
                         mode)
        plot_residuals(model, X, y, mode)

    # Zusammenfassung
    print(f"{'='*60}")
    print(f" ZUSAMMENFASSUNG ({mode})")
    print(f"{'='*60}")
    print(f"  Modellgüte (R² Training):  {r2:.4f}")
    print(f"  Modellgüte (R² CV):        {r2_cv:.4f}")
    print(f"  Vorhergesagter Bestwert:   {predicted_value:.1f}")

    if optimal_value is not None:
        pct = predicted_value / optimal_value * 100
        print(f"  % vom Optimum:             {pct:.1f}%"
              f"  (Optimum = {optimal_value})")

    print(f"  Einflussreichster Param:   {list(importance.keys())[0]}")
    print()
    print(f"  Empfohlene Parameter für main.py:")
    print(f"    DEF_MODE = \"{mode}\"")
    print(f"    DEF_ALPHA = {optimal_params['alpha']:.2f}")
    print(f"    DEF_BETA = {optimal_params['beta']:.2f}")
    print(f"    DEF_EVAPORATION = {optimal_params['evaporation']:.2f}")
    print(f"    DEF_GROUP_SIZE = {int(round(optimal_params['group_size']))}")
    if mode == "EAS":
        print(f"    DEF_ELITE_WEIGHT = "
              f"{optimal_params['elite_weight']:.2f}")
    print()


if __name__ == "__main__":
    main()