"""
Parameterstudie: Faktorieller Versuchsplan zur Berechnung der Haupteffekte und Wechselwirkungen (Fokus: AC).

Berechnet die Haupteffekte und Wechselwirkungen für den AC-Algorithmus,
speichert die Tabelle als CSV und erzeugt Regressions- und Interaktionsplots.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations


# ===================== OPTIMUM LADEN =====================

def load_optimal_value(problem_path="data/problem.json"):
    """Lädt das bekannte Optimum aus problem.json."""
    try:
        with open(problem_path, "r", encoding="utf-8") as f:
            problem = json.load(f)
        return problem["optimal_solution"]["value"]
    except (FileNotFoundError, KeyError):
        return None


# ===================== DATENLADEN UND GRUPPIERUNG =====================

def load_and_prepare_data(filepath, output_filepath, weight_quality=0.7, weight_speed=0.3):
    """
    Lädt die Ergebnisdaten. Falls es Rohdaten sind (mehrere Läufe pro Konfiguration),
    werden sie gruppiert, statistisch aggregiert und mit einem Score bewertet.
    Exportiert die gruppierten Daten anschließend nach output_filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Eingabedatei '{filepath}' nicht gefunden. Bitte führe zuerst experiment.py aus.")

    df = pd.read_csv(filepath, delimiter=';')

    # Prüfen, ob Daten bereits aggregiert sind
    if "median_value" in df.columns:
        return df
    
    # Sicherstellen, dass elite_weight für AC definiert ist (auf 1.0 setzen)
    if "elite_weight" not in df.columns:
        df["elite_weight"] = 1.0
    else:
        df["elite_weight"] = df["elite_weight"].fillna(1.0)

    group_cols = ["mode", "alpha", "beta", "evaporation", "group_size", "elite_weight"]

    grouped = df.groupby(group_cols).agg(
        median_value=("best_value", "median"),
        min_value=("best_value", "min"),
        max_value=("best_value", "max"),
        std_value=("best_value", "std"),
        median_iteration=("best_iteration", "median"),
        num_runs=("best_value", "count")
    ).reset_index()

    # Score berechnen (w_quality * norm(median_value) + w_speed * norm(speed))
    val_min = grouped["median_value"].min()
    val_max = grouped["median_value"].max()
    if val_max > val_min:
        grouped["norm_quality"] = (grouped["median_value"] - val_min) / (val_max - val_min)
    else:
        grouped["norm_quality"] = 1.0

    iter_min = grouped["median_iteration"].min()
    iter_max = grouped["median_iteration"].max()
    if iter_max > iter_min:
        grouped["norm_speed"] = (iter_max - grouped["median_iteration"]) / (iter_max - iter_min)
    else:
        grouped["norm_speed"] = 1.0

    grouped["score"] = (weight_quality * grouped["norm_quality"] +
                        weight_speed * grouped["norm_speed"])

    grouped = grouped.sort_values("score", ascending=False).reset_index(drop=True)

    optimal_value = load_optimal_value()
    if optimal_value:
        grouped["pct_optimal"] = grouped["median_value"] / optimal_value * 100

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    grouped.to_csv(output_filepath, sep=';', index=False, float_format='%.3f', encoding="utf-8")

    return grouped


# ===================== FAKTORIELLE TABELLE CSV-EXPORT =====================

def save_factorial_table_csv(df_raw, factors, levels, output_csv_path):
    """
    Berechnet die faktoriellen Spalten (+/-) für Hauptfaktoren und 2-Weg-Interaktionen
    und speichert die Tabelle samt MW-, MW+ und Effekt in einer CSV-Datei.
    """
    df_mode = df_raw[df_raw["mode"] == "AC"].copy()
    
    # 2-Level-Kombinationen filtern
    df_filtered = df_mode.copy()
    for f in factors:
        df_filtered = df_filtered[df_filtered[f].isin(levels[f])]
        
    # Daten sind bereits durch load_and_prepare_data gruppiert
    grouped = df_filtered.reset_index(drop=True)
    
    factor_letters = ["A", "B", "C", "D"]
    interaction_pairs = list(combinations(factor_letters, 2))
    
    rows = []
    for idx, row in grouped.iterrows():
        row_dict = {}
        # Vorzeichen für Hauptfaktoren setzen
        for letter, factor in zip(factor_letters, factors):
            val = row[factor]
            sign = "+" if val == levels[factor][1] else "-"
            row_dict[letter] = sign

        # Vorzeichen für Wechselwirkungen (Interaktionen) bestimmen
        for p1, p2 in interaction_pairs:
            sign_p1 = row_dict[p1]
            sign_p2 = row_dict[p2]
            row_dict[p1+p2] = "+" if sign_p1 == sign_p2 else "-"

        row_dict["y"] = row["median_value"]
        rows.append(row_dict)
        
    columns = factor_letters + [p1+p2 for p1, p2 in interaction_pairs]
    
    # MW-, MW+ und Effekte berechnen
    mw_minus = {}
    mw_plus = {}
    effekt = {}

    for col in columns:
        y_minus = [r["y"] for r in rows if r[col] == "-"]
        y_plus = [r["y"] for r in rows if r[col] == "+"]

        mw_minus[col] = np.mean(y_minus) if y_minus else 0
        mw_plus[col] = np.mean(y_plus) if y_plus else 0
        effekt[col] = mw_plus[col] - mw_minus[col]
        
    # DataFrame für den Export erstellen
    export_rows = []
    
    # 1. Datenzeilen hinzufügen
    for idx, r in enumerate(rows):
        export_row = {"Nr": idx + 1}
        for col in columns:
            export_row[col] = r[col]
        export_row["y"] = r["y"]
        export_rows.append(export_row)
        
    # 2. MW- Zeile
    mw_minus_row = {"Nr": "MW-"}
    for col in columns:
        mw_minus_row[col] = mw_minus[col]
    mw_minus_row["y"] = ""
    export_rows.append(mw_minus_row)
    
    # 3. MW+ Zeile
    mw_plus_row = {"Nr": "MW+"}
    for col in columns:
        mw_plus_row[col] = mw_plus[col]
    mw_plus_row["y"] = ""
    export_rows.append(mw_plus_row)
    
    # 4. Effekt Zeile
    effekt_row = {"Nr": "Effekt"}
    for col in columns:
        effekt_row[col] = effekt[col]
    effekt_row["y"] = ""
    export_rows.append(effekt_row)
    
    # In DataFrame konvertieren und speichern
    df_export = pd.DataFrame(export_rows)
    df_export.to_csv(output_csv_path, sep=';', index=False, encoding="utf-8")
    print(f" -> Faktorielle Tabelle erfolgreich als CSV gespeichert unter: {output_csv_path}")


# ===================== PLOT-FUNKTIONEN =====================

def calculate_main_effects_data(df_mode, factors, levels):
    """
    Berechnet für jeden Parameter P die Mittelwerte bei Min, Mid und Max,
    wobei alle anderen Parameter auf ihre Extremwerte (Min/Max) eingeschränkt sind.
    """
    effects_data = {}
    
    for p in factors:
        other_factors = [f for f in factors if f != p]
        
        # Filtere Daten: Andere Faktoren dürfen nur Min oder Max sein
        df_filtered = df_mode.copy()
        for f in other_factors:
            df_filtered = df_filtered[df_filtered[f].isin(levels[f])]
            
        # Drei Stufen von p bestimmen (Min, Mid, Max)
        p_min, p_max = levels[p]
        all_vals = sorted(df_mode[p].unique())
        p_mid = [v for v in all_vals if v != p_min and v != p_max][0]
        
        mean_min = df_filtered[df_filtered[p] == p_min]["median_value"].mean()
        mean_mid = df_filtered[df_filtered[p] == p_mid]["median_value"].mean()
        mean_max = df_filtered[df_filtered[p] == p_max]["median_value"].mean()
        
        effects_data[p] = {
            "x": [p_min, p_mid, p_max],
            "y": [mean_min, mean_mid, mean_max]
        }
        
    return effects_data


def plot_main_effects_regression(effects_data, output_path):
    """
    Erzeugt Plots für die Haupteffekte mit Regressionsgerade (Min/Max) und Mittelpunkt.
    """
    factors = list(effects_data.keys())
    fig, axes = plt.subplots(1, len(factors), figsize=(4.2 * len(factors), 3.8))
    
    for idx, p in enumerate(factors):
        ax = axes[idx]
        data = effects_data[p]
        x_vals = data["x"]
        y_vals = data["y"]
        
        p_min, p_mid, p_max = x_vals
        y_min, y_mid, y_max = y_vals
        
        # 1. Regressionsgerade zeichnen (nur aus Min und Max berechnet)
        slope = (y_max - y_min) / (p_max - p_min)
        intercept = y_min - slope * p_min
        
        line_x = np.array([p_min, p_max])
        line_y = slope * line_x + intercept
        
        ax.plot(line_x, line_y, '-', color='steelblue', linewidth=2, label='Regression (Min-Max)')
        
        # 2. Datenpunkte einzeichnen
        # Min/Max (blau)
        ax.scatter([p_min, p_max], [y_min, y_max], color='steelblue', s=80, zorder=5, label='Extremwerte (- / +)')
        # Mittelwert (rot)
        ax.scatter([p_mid], [y_mid], color='crimson', s=100, marker='o', edgecolors='black', zorder=6, label='Mittelwert (Mitte)')
        
        # Gestrichelte rote Linie zur Visualisierung des Abstands zur Regression
        y_line_at_mid = slope * p_mid + intercept
        ax.plot([p_mid, p_mid], [y_mid, y_line_at_mid], '--', color='crimson', alpha=0.7)
        
        ax.set_xlabel(p, fontsize=10)
        ax.set_ylabel('Mittlerer Nutzwert (Median)', fontsize=10)
        ax.set_title(f'Haupteffekt: {p}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
            
    plt.suptitle("Haupteffekte: Regressionsgerade (Min/Max) mit Krümmungs-Check (Mitte)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" -> Haupteffekt-Plot gespeichert unter: {output_path}")


def plot_interaction_lines(df_mode, factors, levels, output_path):
    """
    Erzeugt 2D-Interaktionsplots (Wechselwirkungen) für alle paarweisen Kombinationen.
    Jeder Plot enthält zwei Linien (für Min- und Max-Stufen des zweiten Parameters).
    """
    pairs = list(combinations(factors, 2))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes = axes.flatten()
    
    # Filtere die Rohdaten für alle Faktoren auf ihre Min- und Max-Stufen
    df_2lvl = df_mode.copy()
    for f in factors:
        df_2lvl = df_2lvl[df_2lvl[f].isin(levels[f])]
        
    for idx, (p1, p2) in enumerate(pairs):
        ax = axes[idx]
        p1_min, p1_max = levels[p1]
        p2_min, p2_max = levels[p2]
        
        # Mittelwerte berechnen (nur 2-Level-Design berücksichtigen)
        y_min_min = df_2lvl[(df_2lvl[p1] == p1_min) & (df_2lvl[p2] == p2_min)]["median_value"].mean()
        y_max_min = df_2lvl[(df_2lvl[p1] == p1_max) & (df_2lvl[p2] == p2_min)]["median_value"].mean()
        
        y_min_max = df_2lvl[(df_2lvl[p1] == p1_min) & (df_2lvl[p2] == p2_max)]["median_value"].mean()
        y_max_max = df_2lvl[(df_2lvl[p1] == p1_max) & (df_2lvl[p2] == p2_max)]["median_value"].mean()
        
        # Plotten
        ax.plot([p1_min, p1_max], [y_min_min, y_max_min], 'o-', color='steelblue', linewidth=2, label=f'{p2} Min ({p2_min})')
        ax.plot([p1_min, p1_max], [y_min_max, y_max_max], 'o-', color='forestgreen', linewidth=2, label=f'{p2} Max ({p2_max})')
        
        ax.set_xlabel(p1, fontsize=10)
        ax.set_ylabel('Mittlerer Nutzwert (Median)', fontsize=10)
        ax.set_title(f'Interaktion: {p1} x {p2}', fontsize=11, fontweight='bold')
        ax.set_xticks([p1_min, p1_max])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
    plt.suptitle("Zwei-Faktoren-Interaktionsdiagramme (Wechselwirkungen)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" -> Wechselwirkungs-Plot gespeichert unter: {output_path}")


# ===================== HAUPTPROGRAMM =====================

def main():
    parser = argparse.ArgumentParser(description="Faktorieller Versuchsplan und Regressionsplots für AC")
    parser.add_argument("--file", type=str, default="data/experiment_results.csv",
                        help="Pfad zur experiment_results.csv")
    parser.add_argument("--output-file", type=str, default="data/evaluation_results.csv",
                        help="Pfad zur evaluation_results.csv")

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f" Parameterstudie & Evaluierung (Fokus: AC)")
    print(f"{'='*60}\n")

    # Daten laden und vorbereiten
    try:
        df_raw = load_and_prepare_data(args.file, args.output_file)
    except FileNotFoundError as e:
        print(e)
        return

    # Filtern auf AC
    df_mode = df_raw[df_raw["mode"] == "AC"].copy()
    if df_mode.empty:
        print("Keine AC-Daten in den Ergebnissen vorhanden.")
        return

    factors = ["alpha", "beta", "evaporation", "group_size"]
    
    # Min/Max-Bereiche bestimmen
    levels = {}
    for f in factors:
        vals = sorted(df_mode[f].unique())
        levels[f] = (vals[0], vals[-1])

    # 1. Faktorielle Tabelle berechnen und als CSV speichern
    output_csv = "data/factorial_design_AC.csv"
    save_factorial_table_csv(df_raw, factors, levels, output_csv)
    
    # 2. Haupteffekte berechnen und plotten (mit Mittelpunkt-Check)
    effects_data = calculate_main_effects_data(df_mode, factors, levels)
    plot_main_effects_regression(effects_data, "data/parameterstudie_AC_main_effects_regression.png")
    
    # 3. Wechselwirkungen plotten (2-Linien-Diagramme)
    plot_interaction_lines(df_mode, factors, levels, "data/parameterstudie_AC_interactions_lines.png")
    
    # 4. Beste getestete AC-Konfiguration auf der Konsole ausgeben
    best_config = df_mode.sort_values("median_value", ascending=False).iloc[0]
    pct_str = f" ({best_config['pct_optimal']:.2f}% vom Optimum)" if "pct_optimal" in best_config else ""
    
    print("\n" + "="*60)
    print(" BESTE GETESTETE AC-KOMBINATION")
    print("="*60)
    print(f"  alpha        : {best_config['alpha']:.2f}")
    print(f"  beta         : {best_config['beta']:.2f}")
    print(f"  evaporation  : {best_config['evaporation']:.2f}")
    print(f"  group_size   : {int(best_config['group_size'])}")
    print(f"  ------------------------------------------------")
    print(f"  Median-Wert  : {best_config['median_value']:.1f}{pct_str}")
    print(f"  Gefunden in  : Iteration {int(best_config['median_iteration'])} (Median)")
    print("="*60)
    
    print("\nAuswertung abgeschlossen. Alle Ergebnisse unter 'data/' gespeichert.")


if __name__ == "__main__":
    main()
