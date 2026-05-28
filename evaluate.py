"""
Auswertung der Optimierungsergebnisse.

Liest die CSV-Datei des Optimizers ein, gruppiert nach Parameterkonfiguration,
berechnet den Median und ermittelt die optimale Einstellung.

Bewertung basiert auf zwei Kriterien:
- Qualität:        Median des besten Nutzwerts (best_value)
- Geschwindigkeit: Median der Iteration, in der das Optimum gefunden wurde (best_iteration)

Kombinierte Bewertung: Score = w_quality · norm(quality) + w_speed · norm(speed)

Datenfluss:
    optimizer.py → optimization_results.csv → evaluate.py → evaluation_results.csv

Aufruf:
    python evaluate.py
    python evaluate.py --file andere_datei.csv --top 10
    python evaluate.py --weight-quality 1.0 --weight-speed 0.0
"""

import argparse
import pandas as pd


def load_data(filepath):
    """
    Lädt die Ergebnis-CSV und gibt ein DataFrame zurück.

    Args:
        filepath: Pfad zur CSV-Datei (Semikolon-getrennt).

    Returns:
        DataFrame mit allen Experimentdaten.
    """
    df = pd.read_csv(filepath, delimiter=';')
    print(f"Datei geladen: {filepath}")
    print(f"  → {len(df)} Zeilen, {df['mode'].nunique()} Modi\n")
    return df


def compute_grouped_statistics(df):
    """
    Gruppiert nach Parameterkonfiguration und berechnet Statistiken.

    Für jede Gruppe (= identische Parameter) wird berechnet:
    - Median, Min, Max des best_value
    - Standardabweichung des best_value
    - Median des best_iteration (Konvergenzgeschwindigkeit)
    - Anzahl der Runs (zur Kontrolle)

    Der Median ist robust gegen einzelne Ausreißer (Glückstreffer/Stagnation).

    Args:
        df: DataFrame mit Rohdaten (mehrere Runs pro Konfiguration).

    Returns:
        DataFrame mit einer Zeile pro Konfiguration.
    """
    # Gruppierungsspalten (alle Parameter, die eine Konfiguration definieren)
    group_cols = ["mode", "alpha", "beta", "evaporation", "group_size", "elite_weight"]

    grouped = df.groupby(group_cols).agg(
        median_value=("best_value", "median"),
        min_value=("best_value", "min"),
        max_value=("best_value", "max"),
        std_value=("best_value", "std"),
        median_iteration=("best_iteration", "median"),
        num_runs=("best_value", "count")
    ).reset_index()

    return grouped


def compute_score(df, weight_quality=0.5, weight_speed=0.5):
    """
    Berechnet einen kombinierten Score für jede Konfiguration.

    Formel:
        Score = w_quality · norm(median_value) + w_speed · norm(speed)

    Normierung: Min-Max-Skalierung auf [0, 1].
    - Qualität:        höherer median_value = besser → normiert direkt
    - Geschwindigkeit: kleinere median_iteration = besser → invertiert normiert

    Args:
        df:             DataFrame mit gruppierten Statistiken.
        weight_quality: Gewichtung der Lösungsqualität (Standard: 70%).
        weight_speed:   Gewichtung der Konvergenzgeschwindigkeit (Standard: 30%).

    Returns:
        DataFrame mit zusätzlicher Score-Spalte, absteigend sortiert.
    """
    # Normierung der Qualität: höherer Wert = besser → [0, 1]
    val_min = df["median_value"].min()
    val_max = df["median_value"].max()

    if val_max > val_min:
        df["norm_quality"] = (df["median_value"] - val_min) / (val_max - val_min)
    else:
        df["norm_quality"] = 1.0

    # Normierung der Geschwindigkeit: kleinere Iteration = besser
    # Invertierung: niedrige Iteration → hoher Speed-Score
    iter_min = df["median_iteration"].min()
    iter_max = df["median_iteration"].max()

    if iter_max > iter_min:
        df["norm_speed"] = (iter_max - df["median_iteration"]) / (iter_max - iter_min)
    else:
        df["norm_speed"] = 1.0

    # Kombinierter Score (gewichtete Summe)
    df["score"] = (weight_quality * df["norm_quality"] +
                   weight_speed * df["norm_speed"])

    # Absteigend nach Score sortieren
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df


def print_results(df, mode, top_n=5):
    """
    Gibt die Top-N Konfigurationen für einen Modus formatiert aus.

    Args:
        df:     DataFrame mit Scores.
        mode:   "AC" oder "EAS".
        top_n:  Anzahl der besten Konfigurationen.
    """
    subset = df[df["mode"] == mode].head(top_n).copy()

    if subset.empty:
        print(f"Keine Daten für Modus '{mode}' vorhanden.\n")
        return

    print(f"{'='*70}")
    print(f" Top {top_n} Konfigurationen – {mode}")
    print(f"{'='*70}")
    print(f"{'Rang':<5}{'α':<6}{'β':<6}{'ρ':<6}{'m':<5}{'e':<6}"
          f"{'Median':<8}{'Min':<6}{'Max':<6}{'Iter':<6}{'Score':<7}")
    print(f"{'-'*70}")

    for idx, (_, row) in enumerate(subset.iterrows()):
        print(f"{idx + 1:<5}"
              f"{row['alpha']:<6.1f}"
              f"{row['beta']:<6.1f}"
              f"{row['evaporation']:<6.1f}"
              f"{int(row['group_size']):<5}"
              f"{row['elite_weight']:<6.1f}"
              f"{row['median_value']:<8.1f}"
              f"{row['min_value']:<6.0f}"
              f"{row['max_value']:<6.0f}"
              f"{row['median_iteration']:<6.0f}"
              f"{row['score']:<7.3f}")

    print()

    # Beste Konfiguration hervorheben
    best = subset.iloc[0]
    print(f"  → BESTE {mode}-Konfiguration:")
    print(f"    α={best['alpha']}, β={best['beta']}, ρ={best['evaporation']}, "
          f"m={int(best['group_size'])}", end="")
    if mode == "EAS":
        print(f", e={best['elite_weight']}", end="")
    print(f"\n    Median-Nutzwert: {best['median_value']:.1f}, "
          f"gefunden in Iteration: {best['median_iteration']:.0f}")
    print()


def compare_modes(df):
    """
    Vergleicht die besten Konfigurationen von AC und EAS.

    Args:
        df: DataFrame mit Scores.
    """
    ac_data = df[df["mode"] == "AC"]
    eas_data = df[df["mode"] == "EAS"]

    if ac_data.empty or eas_data.empty:
        print("Nicht genug Daten für Modusvergleich.\n")
        return

    best_ac = ac_data.iloc[0]
    best_eas = eas_data.iloc[0]

    print(f"{'='*70}")
    print(f" Vergleich: AC vs. EAS (jeweils beste Konfiguration)")
    print(f"{'='*70}")
    print(f"{'Kriterium':<25}{'AC':<20}{'EAS':<20}{'Gewinner':<10}")
    print(f"{'-'*70}")

    # Qualität
    winner_q = "AC" if best_ac["median_value"] >= best_eas["median_value"] else "EAS"
    print(f"{'Median Nutzwert':<25}"
          f"{best_ac['median_value']:<20.1f}"
          f"{best_eas['median_value']:<20.1f}"
          f"{winner_q:<10}")

    # Geschwindigkeit
    winner_s = "AC" if best_ac["median_iteration"] <= best_eas["median_iteration"] else "EAS"
    print(f"{'Median Iteration':<25}"
          f"{best_ac['median_iteration']:<20.0f}"
          f"{best_eas['median_iteration']:<20.0f}"
          f"{winner_s:<10}")

    # Gesamtscore
    winner_t = "AC" if best_ac["score"] >= best_eas["score"] else "EAS"
    print(f"{'Gesamtscore':<25}"
          f"{best_ac['score']:<20.3f}"
          f"{best_eas['score']:<20.3f}"
          f"{winner_t:<10}")

    print()
    print(f"  → Gesamtsieger: {winner_t}")
    print()


def export_results(df, filepath="evaluation_results.csv"):
    """
    Exportiert die bewerteten Ergebnisse als CSV.

    Args:
        df:       DataFrame mit Scores.
        filepath: Ausgabedatei.
    """
    df.to_csv(filepath, sep=';', index=False, float_format='%.3f')
    print(f"Bewertete Ergebnisse exportiert nach: {filepath}\n")


def main():
    # ===================== PARAMETER =====================
    parser = argparse.ArgumentParser(description="Auswertung der Optimierungsergebnisse")
    parser.add_argument("--file", type=str, default="optimization_results.csv",
                        help="Pfad zur Ergebnis-CSV")
    parser.add_argument("--top", type=int, default=5,
                        help="Anzahl Top-Konfigurationen pro Modus")
    parser.add_argument("--weight-quality", type=float, default=0.7,
                        help="Gewichtung Qualität im Score (Standard: 0.7)")
    parser.add_argument("--weight-speed", type=float, default=0.3,
                        help="Gewichtung Geschwindigkeit im Score (Standard: 0.3)")
    parser.add_argument("--output", type=str, default="evaluation_results.csv",
                        help="Pfad zur Ausgabe-CSV")

    args = parser.parse_args()

    # ===================== DATEN LADEN =====================
    df = load_data(args.file)

    # ===================== GRUPPIERUNG & MEDIAN =====================
    grouped = compute_grouped_statistics(df)
    print(f"Konfigurationen gefunden: {len(grouped)}")
    print(f"  → AC:  {len(grouped[grouped['mode'] == 'AC'])}")
    print(f"  → EAS: {len(grouped[grouped['mode'] == 'EAS'])}")
    print()

    # ===================== SCORING =====================
    scored = compute_score(grouped, args.weight_quality, args.weight_speed)

    # ===================== AUSGABE =====================
    print_results(scored, "AC", top_n=args.top)
    print_results(scored, "EAS", top_n=args.top)
    compare_modes(scored)

    # ===================== EXPORT =====================
    export_results(scored, args.output)


if __name__ == "__main__":
    main()