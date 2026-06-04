"""
Parameterstudie: Umfassende Analyse der Algorithmus-Parameter.

Erzeugt präsentationsfertige Plots und statistische Auswertungen:
1. Boxplots: Verteilung der Fitness pro Parameter-Stufe
2. Haupteffekt-Plots: Mittlere Fitness pro Parameterwert
3. 2D-Interaktions-Heatmaps: Paarweise Parameterinteraktionen
4. Performance-Histogramm: Verteilung relativ zum Optimum
5. Statistische Signifikanz: Kruskal-Wallis-Test pro Parameter
6. Konvergenz-Vergleich: Beste vs. schlechteste Konfigurationen

Aufruf:
    python parameterstudie.py
    python parameterstudie.py --mode AC
    python parameterstudie.py --file data/evaluation_results.csv --mode EAS
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import kruskal


# ===================== HILFSFUNKTIONEN =====================

def load_optimal_value(problem_path="data/problem.json"):
    """Lädt das bekannte Optimum aus problem.json."""
    try:
        with open(problem_path, "r") as f:
            problem = json.load(f)
        return problem["optimal_solution"]["value"]
    except (FileNotFoundError, KeyError):
        return None


def load_data(filepath, mode):
    """
    Lädt und filtert die Evaluierungsdaten.

    Args:
        filepath: Pfad zur evaluation_results.csv.
        mode:     "AC" oder "EAS".

    Returns:
        df:            Gefiltertes DataFrame.
        feature_names: Liste der Parameternamen.
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

    return df, feature_names


# ===================== PLOT-FUNKTIONEN =====================

def plot_boxplots(df, feature_names, mode, optimal_value=None):
    """
    Boxplots der median_value-Verteilung pro Parameter-Stufe.

    Zeigt sofort, welche Parameterwerte gute/schlechte Ergebnisse liefern.
    """
    n = len(feature_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, name in enumerate(feature_names):
        ax = axes[i]
        unique_vals = sorted(df[name].unique())

        data_groups = [df[df[name] == val]["median_value"].values
                       for val in unique_vals]

        bp = ax.boxplot(data_groups, labels=[str(v) for v in unique_vals],
                        patch_artist=True, widths=0.6)

        # Farbverlauf: schlechte Mediane rot, gute grün
        medians = [np.median(g) for g in data_groups]
        med_min, med_max = min(medians), max(medians)
        for j, patch in enumerate(bp['boxes']):
            if med_max > med_min:
                norm_val = (medians[j] - med_min) / (med_max - med_min)
            else:
                norm_val = 0.5
            color = plt.cm.RdYlGn(norm_val)
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        if optimal_value is not None:
            ax.axhline(optimal_value, color='gold', linestyle='-',
                       linewidth=1.5, alpha=0.7, label=f'Optimum ({optimal_value})')
            ax.legend(fontsize=8)

        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('median_value', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Boxplots – Parametereinfluss ({mode})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output = f"data/parameterstudie_{mode}_boxplots.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"  Boxplots gespeichert: {output}")
    plt.show()


def plot_main_effects(df, feature_names, mode, optimal_value=None):
    """
    Haupteffekt-Plots: Mittlerer median_value pro Parameterwert.

    Entspricht einer ANOVA-Main-Effects-Darstellung – zeigt den
    durchschnittlichen Effekt jedes Parameterwerts über alle anderen
    Kombinationen hinweg.
    """
    n = len(feature_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, name in enumerate(feature_names):
        ax = axes[i]

        grouped = df.groupby(name)["median_value"].agg(["mean", "std", "count"])
        grouped = grouped.sort_index()

        x = grouped.index.values
        y_mean = grouped["mean"].values
        y_std = grouped["std"].values
        y_se = y_std / np.sqrt(grouped["count"].values)

        ax.plot(x, y_mean, 'o-', color='steelblue', linewidth=2,
                markersize=8, label='Mittelwert')
        ax.fill_between(x, y_mean - y_se, y_mean + y_se,
                        alpha=0.2, color='steelblue',
                        label='± Standardfehler')

        if optimal_value is not None:
            ax.axhline(optimal_value, color='gold', linestyle='-',
                       linewidth=1.5, alpha=0.7)

        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Mittlerer median_value', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Haupteffekt-Plots ({mode})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output = f"data/parameterstudie_{mode}_haupteffekte.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"  Haupteffekt-Plots gespeichert: {output}")
    plt.show()


def plot_interaction_heatmaps(df, feature_names, mode):
    """
    2D-Interaktions-Heatmaps: Mittlerer median_value für jede
    Kombination zweier Parameter.

    Zeigt, ob Parameter zusammenwirken (Interaktionseffekte).
    """
    pairs = list(combinations(range(len(feature_names)), 2))
    n_pairs = len(pairs)

    n_cols = min(n_pairs, 3)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5 * n_rows))
    if n_pairs == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, (i, j) in enumerate(pairs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        name_i = feature_names[i]
        name_j = feature_names[j]

        pivot = df.pivot_table(
            values="median_value",
            index=name_j,
            columns=name_i,
            aggfunc="mean"
        )

        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                       origin='lower')
        fig.colorbar(im, ax=ax, label='median_value', shrink=0.85)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{v:.1f}' if isinstance(v, float) else str(v)
                            for v in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{v:.1f}' if isinstance(v, float) else str(v)
                            for v in pivot.index], fontsize=9)

        # Werte in die Zellen schreiben
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f'{val:.0f}', ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='white' if val < pivot.values[
                                ~np.isnan(pivot.values)].mean() else 'black')

        ax.set_xlabel(name_i, fontsize=11)
        ax.set_ylabel(name_j, fontsize=11)
        ax.set_title(f'{name_i} × {name_j}', fontsize=12)

    # Leere Subplots ausblenden
    for idx in range(n_pairs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f'Interaktions-Heatmaps ({mode})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output = f"data/parameterstudie_{mode}_interaktionen.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"  Interaktions-Heatmaps gespeichert: {output}")
    plt.show()


def plot_performance_histogram(df, mode, optimal_value):
    """
    Histogramm: Verteilung der Performance relativ zum Optimum.
    """
    if optimal_value is None:
        print("  ⚠️  Kein Optimum bekannt – Histogramm übersprungen.")
        return

    pct = df["median_value"] / optimal_value * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(pct, bins=20, color='steelblue', edgecolor='white',
            alpha=0.8, zorder=3)
    ax.axvline(100, color='gold', linestyle='-', linewidth=2,
               label=f'Optimum (100%)', zorder=5)
    ax.axvline(pct.max(), color='red', linestyle='--', linewidth=1.5,
               label=f'Bestes Ergebnis ({pct.max():.1f}%)', zorder=5)
    ax.axvline(pct.median(), color='green', linestyle='--', linewidth=1.5,
               label=f'Median ({pct.median():.1f}%)', zorder=5)

    ax.set_xlabel('Performance (% vom Optimum)', fontsize=12)
    ax.set_ylabel('Anzahl Konfigurationen', fontsize=12)
    ax.set_title(f'Performance-Verteilung ({mode}) – '
                 f'{len(df)} Konfigurationen', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output = f"data/parameterstudie_{mode}_performance.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"  Performance-Histogramm gespeichert: {output}")
    plt.show()


def run_statistical_tests(df, feature_names, mode):
    """
    Kruskal-Wallis-Test pro Parameter.

    Testet, ob die Verteilungen der median_value sich zwischen den
    Stufen eines Parameters signifikant unterscheiden.

    Kruskal-Wallis ist ein nicht-parametrischer Test (braucht keine
    Normalverteilung) – passend für stochastische Optimierung.
    """
    print(f"\n=== STATISTISCHE SIGNIFIKANZ ({mode}) ===")
    print(f"    Kruskal-Wallis-Test: Unterscheiden sich die Stufen?\n")
    print(f"    {'Parameter':<15} {'H-Statistik':>12} {'p-Wert':>10}"
          f" {'Signifikant':>12}")
    print(f"    {'-'*50}")

    results = {}

    for name in feature_names:
        unique_vals = sorted(df[name].unique())

        if len(unique_vals) < 2:
            print(f"    {name:<15} {'—':>12} {'—':>10} {'nur 1 Stufe':>12}")
            continue

        groups = [df[df[name] == val]["median_value"].values
                  for val in unique_vals]

        stat, p_value = kruskal(*groups)
        sig = "JA ✓" if p_value < 0.05 else "nein"
        marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 \
            else "*" if p_value < 0.05 else ""

        print(f"    {name:<15} {stat:>12.2f} {p_value:>10.4f}"
              f" {sig + marker:>12}")

        results[name] = {"H": stat, "p": p_value, "significant": p_value < 0.05}

    print()
    print("    Legende: * p<0.05  ** p<0.01  *** p<0.001")
    print()

    return results


def print_summary_table(df, feature_names, mode, optimal_value=None,
                        stat_results=None):
    """
    Zusammenfassungstabelle für die Präsentation.
    """
    print(f"\n{'='*70}")
    print(f" PARAMETERSTUDIE – ZUSAMMENFASSUNG ({mode})")
    print(f"{'='*70}")
    print(f"  Anzahl Konfigurationen:  {len(df)}")
    print(f"  median_value Bereich:    [{df['median_value'].min():.0f},"
          f" {df['median_value'].max():.0f}]")

    if optimal_value is not None:
        best_pct = df['median_value'].max() / optimal_value * 100
        worst_pct = df['median_value'].min() / optimal_value * 100
        print(f"  Performance-Bereich:     [{worst_pct:.1f}%,"
              f" {best_pct:.1f}%] vom Optimum")

    print()
    print(f"  {'Parameter':<15} {'Bester Wert':>12} {'Schlechtester':>14}"
          f" {'Einfluss':>10}")
    print(f"  {'-'*55}")

    for name in feature_names:
        grouped = df.groupby(name)["median_value"].mean()
        best_level = grouped.idxmax()
        worst_level = grouped.idxmin()

        sig = ""
        if stat_results and name in stat_results:
            if stat_results[name]["significant"]:
                sig = "  ✓ signifikant"

        if isinstance(best_level, float):
            best_str = f"{best_level:.2f}"
            worst_str = f"{worst_level:.2f}"
        else:
            best_str = str(best_level)
            worst_str = str(worst_level)

        print(f"  {name:<15} {best_str:>12} {worst_str:>14}{sig}")

    # Top-5 Konfigurationen
    print(f"\n  Top 5 Konfigurationen:")
    top5 = df.nlargest(5, "median_value")
    for idx, (_, row) in enumerate(top5.iterrows()):
        params = ", ".join(
            f"{name}={row[name]:.1f}" if isinstance(row[name], float)
            else f"{name}={int(row[name])}"
            for name in feature_names
        )
        pct_str = ""
        if optimal_value:
            pct_str = f" ({row['median_value']/optimal_value*100:.1f}%)"
        print(f"    {idx+1}. median_value={row['median_value']:.0f}"
              f"{pct_str}  [{params}]")

    print(f"{'='*70}\n")


# ===================== HAUPTPROGRAMM =====================

def main():
    parser = argparse.ArgumentParser(
        description="Parameterstudie – Umfassende Analyse")
    parser.add_argument("--file", type=str,
                        default="data/evaluation_results.csv",
                        help="Pfad zur evaluation_results.csv")
    parser.add_argument("--mode", type=str, default="EAS",
                        help="Modus: AC oder EAS")
    parser.add_argument("--no-plot", action="store_true",
                        help="Visualisierungen deaktivieren")

    args = parser.parse_args()
    mode = args.mode.upper()

    print(f"{'='*60}")
    print(f" Parameterstudie – {mode}")
    print(f"{'='*60}\n")

    # Daten laden
    optimal_value = load_optimal_value()
    df, feature_names = load_data(args.file, mode)

    print(f"  {len(df)} Konfigurationen geladen für {mode}")
    if optimal_value:
        print(f"  Bekanntes Optimum: {optimal_value}")
    print()

    # Statistische Tests
    stat_results = run_statistical_tests(df, feature_names, mode)

    # Visualisierungen
    if not args.no_plot:
        plot_boxplots(df, feature_names, mode, optimal_value)
        plot_main_effects(df, feature_names, mode, optimal_value)
        plot_interaction_heatmaps(df, feature_names, mode)
        plot_performance_histogram(df, mode, optimal_value)

    # Zusammenfassung
    print_summary_table(df, feature_names, mode, optimal_value, stat_results)


if __name__ == "__main__":
    main()
