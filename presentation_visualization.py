"""
Präsentations-Visualisierung: ACO-Parameter-Analyse auf NMGE-Knapsack
Autor: Antigravity Coding Assistant
Beschreibung: Dieses Skript liest die Ergebnisse der 14 Versuchsreihen (V1 bis V14) ein,
              kompiliert sie in eine zentrale CSV-Datei und generiert 6 wissenschaftliche
              Grafiken für Präsentationen und Abschlussarbeiten.
              Alle Grafiken werden im Ordner 'presentation' als PNG und PDF gespeichert.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== STYLE & CONFIGURATION =====================

# Matplotlib-Stile für wissenschaftliche Arbeiten konfigurieren
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except (OSError, ValueError):
    try:
        plt.style.use('seaborn-whitegrid')
    except (OSError, ValueError):
        plt.style.use('default')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

OPTIMAL_VALUE = 1011638  # Optimum des NMGE-Knapsack
BEST_ACO_VALUE = 1005936  # Bestes ACO-Ergebnis (aus V14)

# ===================== DATEN-KOMPILIERUNG =====================

def compile_and_load_data(force=False):
    """
    Sammelt die evaluation_results.csv aus data/V1..V14 und führt sie zusammen.
    """
    csv_filename = "ergebnisse_alle_versuchsreihen.csv"
    
    if os.path.exists(csv_filename) and not force:
        print(f"Lade existierende Daten aus '{csv_filename}'...")
        return pd.read_csv(csv_filename, sep=';')
        
    print("Kompiliere Daten aus data/V1 bis data/V14...")
    dfs = []
    
    for i in range(1, 15):
        file_path = f"data/V{i}/evaluation_results.csv"
        if os.path.exists(file_path):
            try:
                df_temp = pd.read_csv(file_path, sep=';')
                df_temp = df_temp[df_temp['mode'] == 'AC'].copy()
                df_temp['versuchsreihe'] = i
                dfs.append(df_temp)
            except Exception as e:
                print(f"Fehler beim Lesen von {file_path}: {e}")
        else:
            print(f"Warnung: {file_path} nicht gefunden.")
            
    if not dfs:
        raise FileNotFoundError("Keine Daten in data/V1..V14 gefunden.")
        
    df_all = pd.concat(dfs, ignore_index=True)
    cols = [
        'versuchsreihe', 'alpha', 'beta', 'evaporation', 'group_size',
        'median_value', 'median_iteration', 'std_value', 'min_value', 'max_value'
    ]
    df_all = df_all[cols].drop_duplicates().reset_index(drop=True)
    df_all.to_csv(csv_filename, sep=';', index=False, float_format='%.3f', encoding='utf-8')
    print(f"Erfolgreich {len(df_all)} Zeilen in '{csv_filename}' gespeichert.")
    return df_all


# ===================== GRAFIK 1: VERLAUF DER VERSUCHSREIHEN =====================

def plot_learning_curve(df):
    """Generiert Grafik 1: Verlauf der Versuchsreihen mit Phasenbalken unten."""
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[7.5, 2.5], hspace=0.1)
    
    ax_main = fig.add_subplot(gs[0])
    ax_phase = fig.add_subplot(gs[1])
    
    # Statistiken berechnen
    stats = df.groupby('versuchsreihe')['median_value'].agg(['min', 'max', 'mean']).reset_index()
    
    # 1. Bandbreite schattieren
    ax_main.fill_between(
        stats['versuchsreihe'], stats['min'], stats['max'],
        color='#1f77b4', alpha=0.15, label='Bandbreite aller Konfigurationen'
    )
    
    # 2. Durchschnitt
    ax_main.plot(
        stats['versuchsreihe'], stats['mean'],
        color='grey', linestyle='--', linewidth=1.2, alpha=0.8, label='Ø aller Konfigurationen'
    )
    
    # 3. Bestes Ergebnis pro Reihe
    ax_main.plot(
        stats['versuchsreihe'], stats['max'],
        'o-', color='#0b3c5d', linewidth=2.5, markersize=8, label='Bestes Ergebnis der Versuchsreihe'
    )
    
    # Referenzlinien einzeichnen
    ax_main.axhline(OPTIMAL_VALUE, color='#d62728', linestyle=':', linewidth=1.5)
    ax_main.text(14.2, OPTIMAL_VALUE, f'Optimum ({OPTIMAL_VALUE:,.0f})'.replace(",", "."), color='#d62728', va='center', fontsize=9, fontweight='bold')
    
    ax_main.axhline(BEST_ACO_VALUE, color='forestgreen', linestyle='-.', linewidth=1.2)
    ax_main.text(14.2, BEST_ACO_VALUE, f'Bestes ACO ({BEST_ACO_VALUE:,.0f})'.replace(",", "."), color='forestgreen', va='center', fontsize=9, fontweight='bold')
    
    # Text-Labels an Stützpunkten
    for idx, row in stats[stats['versuchsreihe'].isin([1, 3, 8, 14])].iterrows():
        ax_main.text(
            row['versuchsreihe'], row['max'] + 100, f"{row['max']:,.0f}".replace(",", "."),
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#0b3c5d'
        )
        
    # Annotations
    ax_main.annotate(
        "β = 0.8 entdeckt\n→ Erster großer Sprung",
        xy=(3, stats.loc[stats['versuchsreihe'] == 3, 'max'].values[0]),
        xytext=(3.5, stats.loc[stats['versuchsreihe'] == 3, 'max'].values[0] + 500),
        arrowprops=dict(arrowstyle="->", color='black', lw=1),
        fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="grey", alpha=0.9)
    )
    
    ax_main.annotate(
        "β = 0.6, Gruppe = 120\n→ Dämpfung & Diversität",
        xy=(10, stats.loc[stats['versuchsreihe'] == 10, 'max'].values[0]),
        xytext=(8.0, stats.loc[stats['versuchsreihe'] == 10, 'max'].values[0] - 600),
        arrowprops=dict(arrowstyle="->", color='black', lw=1),
        fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="grey", alpha=0.9)
    )
    
    ax_main.annotate(
        "Optimum: β = 0.65, G = 125\n→ Feintuning-Erfolg",
        xy=(14, stats.loc[stats['versuchsreihe'] == 14, 'max'].values[0]),
        xytext=(11.5, stats.loc[stats['versuchsreihe'] == 14, 'max'].values[0] + 250),
        arrowprops=dict(arrowstyle="->", color='forestgreen', lw=1),
        fontsize=9, fontweight='bold', color='forestgreen', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="forestgreen", alpha=0.9)
    )
    
    # Haupt-Achsen-Styling
    ax_main.set_ylabel('Median-Wert (Fitness)', fontweight='bold')
    ax_main.set_xticks(range(1, 15))
    ax_main.set_xlim(0.5, 15.5)
    ax_main.set_ylim(1003400, 1006500)
    ax_main.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax_main.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    
    # Rechte Y-Achse für Gap (%)
    ax_gap = ax_main.twinx()
    ax_gap.set_ylim(ax_main.get_ylim())
    gap_ticks = [1003500, 1004000, 1004500, 1005000, 1005500, 1006000, 1006500]
    ax_gap.set_yticks(gap_ticks)
    ax_gap.set_yticklabels([f"{(OPTIMAL_VALUE - val)/OPTIMAL_VALUE*100:.2f}%" for val in gap_ticks])
    ax_gap.set_ylabel("Gap zum Optimum (%)", fontweight='bold')
    ax_gap.grid(False)
    
    # ---------------- ELEMENT B: Phasenbalken ----------------
    ax_phase.set_xlim(0.5, 15.5)
    ax_phase.set_ylim(-0.5, 0.5)
    ax_phase.axis('off')
    
    # Phasenblöcke zeichnen
    phases = [
        (0.5, 2, "EXPLORATION\n(V1 - V2)", "#b3cde3"),
        (2.5, 4, "BETA-FOKUS\n(V3 - V6)", "#fddaec"),
        (6.5, 3, "GRUPPE+\n(V7 - V9)", "#decbe4"),
        (9.5, 5, "FEINTUNING\n(V10 - V14)", "#ccebc5")
    ]
    
    for left, width, text, color in phases:
        ax_phase.barh(0, width, left=left, height=0.6, color=color, edgecolor='grey', alpha=0.7)
        ax_phase.text(left + width/2., 0, text, ha='center', va='center', fontweight='bold', fontsize=9.5, color='#333333')
        
    plt.suptitle("Entwicklung der Lösungsqualität über die Versuchsreihen", fontsize=14, fontweight='bold', y=0.98)
    
    # Speichern
    plt.savefig("presentation/01_verlauf_versuchsreihen.png", dpi=300, bbox_inches='tight')
    plt.savefig("presentation/01_verlauf_versuchsreihen.pdf", bbox_inches='tight')
    plt.close()
    print("Grafik 1 (Verlauf Versuchsreihen) erfolgreich in 'presentation/' gespeichert.")


# ===================== GRAFIK 2: MAIN EFFECTS PLOT =====================

def plot_main_effects(df):
    """Generiert Grafik 2: Haupteffekte (Beeswarm Jitter + Trend)."""
    fig = plt.figure(figsize=(16, 5.5))
    
    # Gridspec: Subplot 2 (Beta) bekommt 30% mehr Breite
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.3, 1.0, 1.0])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    
    params = ['alpha', 'beta', 'evaporation', 'group_size']
    labels = ['Alpha (α)', 'Beta (β)', 'Evaporation (ρ)', 'Gruppengröße']
    opt_spans = [(1.3, 1.8), (0.55, 0.85), (0.35, 0.55), (90, 140)]
    
    overall_mean = df['median_value'].mean()
    y_lim = (1003000, 1006500)
    
    for idx, (param, label, opt_span) in enumerate(zip(params, labels, opt_spans)):
        ax = axes[idx]
        
        # Jitter hinzufügen für die Streuung
        x_range = df[param].max() - df[param].min()
        jitter = np.random.normal(0, 0.015 * x_range if x_range > 0 else 1.0, size=len(df))
        x_jittered = df[param] + jitter
        
        # Punkte nach Qualität einfärben
        bad_mask = df['median_value'] < 1004200
        mid_mask = (df['median_value'] >= 1004200) & (df['median_value'] <= 1005200)
        good_mask = df['median_value'] > 1005200
        
        ax.scatter(x_jittered[bad_mask], df.loc[bad_mask, 'median_value'], color='#d62728', alpha=0.35, s=12, label='Schlecht (<1.0042k)' if idx == 0 else "")
        ax.scatter(x_jittered[mid_mask], df.loc[mid_mask, 'median_value'], color='#fec44f', alpha=0.35, s=12, label='Mittel' if idx == 0 else "")
        ax.scatter(x_jittered[good_mask], df.loc[good_mask, 'median_value'], color='#2ca02c', alpha=0.45, s=12, label='Gut (>1.0052k)' if idx == 0 else "")
        
        # Trendlinie (Polynom 3. Grades als robuster Standard)
        p_coefs = np.polyfit(df[param], df['median_value'], 3)
        x_line = np.linspace(df[param].min(), df[param].max(), 150)
        y_line = np.polyval(p_coefs, x_line)
        ax.plot(x_line, y_line, color='black', linewidth=2.5, label='Trend (Polynom)' if idx == 0 else "")
        
        # Gesamtmittelwert
        ax.axhline(overall_mean, color='grey', linestyle='--', linewidth=1.2, alpha=0.7, label='Gesamtmittelwert' if idx == 0 else "")
        
        # Optimalen Bereich einfärben
        ax.axvspan(opt_span[0], opt_span[1], color='lightgreen', alpha=0.15, label='Optimaler Bereich' if idx == 0 else "")
        
        # Spezifische Details
        if param == 'beta':
            ax.axvline(2.5, color='grey', linestyle=':', alpha=0.7)
            ax.text(2.6, y_lim[0] + 300, "Standard-KP\n(β = 2-5)", color='grey', fontsize=8.5, fontweight='bold')
            ax.text((opt_span[0]+opt_span[1])/2, y_lim[0] + 150, "Optimum\n(NMGE)", color='forestgreen', ha='center', fontsize=8.5, fontweight='bold')
            
        ax.set_xlabel(label, fontweight='bold')
        ax.set_ylim(y_lim)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        ax.grid(False, axis='x')
        
        if idx == 0:
            ax.set_ylabel('Median-Wert (Fitness)', fontweight='bold')
            ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
        else:
            ax.set_yticklabels([])
            
    plt.suptitle("Haupteffekte der ACO-Parameter auf die Lösungsqualität (Streuung & Trend)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Speichern
    plt.savefig("presentation/02_main_effects.png", dpi=300, bbox_inches='tight')
    plt.savefig("presentation/02_main_effects.pdf", bbox_inches='tight')
    plt.close()
    print("Grafik 2 (Main Effects) erfolgreich in 'presentation/' gespeichert.")


# ===================== GRAFIK 3: INTERAKTIONSDIAGRAMME (KLASSISCH) =====================

def plot_interaction_plots(df):
    """Generiert Grafik 3: Klassische Interaktions-Liniendiagramme (2x3 Subplots)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    axes = axes.flatten()
    
    # Parametereinteilung in Bins (3 grobe Bins pro Parameter)
    df_binned = df.copy()
    df_binned['alpha_bin'] = pd.cut(df_binned['alpha'], bins=[-np.inf, 1.0, 1.6, np.inf], labels=['niedrig', 'mittel', 'hoch'])
    df_binned['beta_bin'] = pd.cut(df_binned['beta'], bins=[-np.inf, 0.7, 1.1, np.inf], labels=['niedrig', 'mittel', 'hoch'])
    df_binned['evaporation_bin'] = pd.cut(df_binned['evaporation'], bins=[-np.inf, 0.25, 0.50, np.inf], labels=['niedrig', 'mittel', 'hoch'])
    df_binned['group_size_bin'] = pd.cut(df_binned['group_size'], bins=[-np.inf, 60, 100, np.inf], labels=['klein', 'mittel', 'groß'])
    
    # Paarungen festlegen
    pairs = [
        ('alpha_bin', 'beta_bin', 'Alpha (α)', 'Beta (β)'),
        ('alpha_bin', 'evaporation_bin', 'Alpha (α)', 'Evaporation (ρ)'),
        ('alpha_bin', 'group_size_bin', 'Alpha (α)', 'Gruppengröße'),
        ('beta_bin', 'evaporation_bin', 'Beta (β)', 'Evaporation (ρ)'),
        ('beta_bin', 'group_size_bin', 'Beta (β)', 'Gruppengröße'),
        ('evaporation_bin', 'group_size_bin', 'Evaporation (ρ)', 'Gruppengröße')
    ]
    
    # Einheitlicher Y-Achsenbereich für alle Subplots berechnen
    y_lim = (1003400, 1006400)
    
    linestyles = ['-', '--', ':']
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (param_a, param_b, label_a, label_b) in enumerate(pairs):
        ax = axes[idx]
        
        # Berechne Mittelwerte und Standardfehler (SEM) für jede Kombination
        grouped = df_binned.groupby([param_a, param_b], observed=False)['median_value'].agg(['mean', 'sem']).reset_index()
        
        categories_b = ['niedrig', 'mittel', 'hoch'] if 'group_size' not in param_b else ['klein', 'mittel', 'groß']
        categories_a = ['niedrig', 'mittel', 'hoch'] if 'group_size' not in param_a else ['klein', 'mittel', 'groß']
        
        lines_y = []
        
        for b_idx, cat_b in enumerate(categories_b):
            subset = grouped[grouped[param_b] == cat_b]
            # Sicherstellen, dass die X-Kategorien in der richtigen Reihenfolge geplottet werden
            subset = subset.set_index(param_a).reindex(categories_a).reset_index()
            
            if subset['mean'].isna().all():
                continue
                
            x_vals = range(len(categories_a))
            mean_y = subset['mean'].values
            sem_y = subset['sem'].values
            
            # Speicher Werte für Kreuzungstest (ohne NaNs)
            lines_y.append((cat_b, mean_y))
            
            # Zeichne Linie mit Fehlerbalken (SEM)
            ax.errorbar(
                x_vals, mean_y, yerr=sem_y, 
                label=f"{label_b}: {cat_b}", 
                color=colors[b_idx % len(colors)],
                linestyle=linestyles[b_idx % len(linestyles)],
                marker=markers[b_idx % len(markers)],
                linewidth=2.2, markersize=7, capsize=4, elinewidth=1.5
            )
            
        # Kreuzungstest (Wechselwirkung erkennen):
        # Wenn sich die Linien kreuzen, heben wir den Subplot rot hervor.
        crossing_detected = False
        if len(lines_y) >= 2:
            # Nur vollständige Linien (3 Punkte) vergleichen
            complete_lines = [ly for ly in lines_y if not np.isnan(ly[1]).any() and len(ly[1]) == 3]
            for i in range(len(complete_lines)):
                for j in range(i + 1, len(complete_lines)):
                    y_i = complete_lines[i][1]
                    y_j = complete_lines[j][1]
                    
                    diff_start = y_i[0] - y_j[0]
                    diff_end = y_i[-1] - y_j[-1]
                    
                    # Wenn die Differenz an den Enden das Vorzeichen wechselt -> Schnittpunkt
                    if np.sign(diff_start) != np.sign(diff_end) and diff_start != 0 and diff_end != 0:
                        crossing_detected = True
                        break
        
        # Falls eine Kreuzung vorliegt -> roter Rahmen + Textbox
        if crossing_detected:
            for spine in ax.spines.values():
                spine.set_color('#d62728')
                spine.set_linewidth(2.0)
            ax.text(
                0.95, 0.95, "Starke Interaktion!", color='#d62728', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.2', alpha=0.9)
            )
            
        ax.set_title(f"{label_a} × {label_b}", fontweight='bold')
        ax.set_xticks(range(len(categories_a)))
        ax.set_xticklabels(categories_a)
        ax.set_xlabel(label_a)
        ax.set_ylim(y_lim)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        ax.legend(loc='lower left', frameon=True, framealpha=0.9, facecolor='white', fontsize=8.5)
        
        if idx % 3 == 0:
            ax.set_ylabel('Mittlerer Median-Wert', fontweight='bold')
            
    # Erklärungstextbox unten einfügen
    fig.text(
        0.5, 0.015, "Parallele Linien = kein Interaktionseffekt; Kreuzende Linien = Interaktionseffekt (rot hervorgehoben)",
        ha="center", fontsize=11, fontweight="bold", color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="grey", alpha=0.8)
    )
    
    plt.suptitle("Interaktionseffekte zwischen ACO-Parametern (Klassische Liniendiagramme)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Speichern
    plt.savefig("presentation/03_interaction_plots.png", dpi=300, bbox_inches='tight')
    plt.savefig("presentation/03_interaction_plots.pdf", bbox_inches='tight')
    plt.close()
    print("Grafik 3 (Interaktions-Liniendiagramme) erfolgreich in 'presentation/' gespeichert.")


# ===================== GRAFIK 4: NMGE-FALLEN-EFFEKT =====================

def plot_nmge_trap(df):
    """Generiert Grafik 4: NMGE-Fallen-Effekt (Scatter oben + Konzeptboxen unten)."""
    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[6.2, 3.8], hspace=0.35)
    
    ax_main = fig.add_subplot(gs[0])
    ax_boxes = fig.add_subplot(gs[1])
    
    # ---------------- OBERER TEIL: Scatter mit Trend ----------------
    # Scatter plot
    sc = ax_main.scatter(
        df['beta'], df['median_value'],
        c=df['group_size'], s=40,
        cmap='viridis', alpha=0.65, edgecolors='black', linewidths=0.5
    )
    
    cbar = fig.colorbar(sc, ax=ax_main, label='Gruppengröße (Ameisen)')
    cbar.ax.set_ylabel('Gruppengröße (Ameisen)', fontweight='bold')
    
    # LOESS oder 3. Grad Polynom Trend
    try:
        import statsmodels.api as sm
        z = sm.nonparametric.lowess(df['median_value'], df['beta'], frac=0.45)
        ax_main.plot(z[:, 0], z[:, 1], color='#d62728', linewidth=3.0, label='Entwicklungstrend (LOESS)')
        # Einfaches Band zeichnen
        ax_main.fill_between(z[:, 0], z[:, 1] - 300, z[:, 1] + 300, color='#d62728', alpha=0.10, label='Konfidenzband')
    except ImportError:
        p_coefs = np.polyfit(df['beta'], df['median_value'], 3)
        x_line = np.linspace(df['beta'].min(), df['beta'].max(), 200)
        y_line = np.polyval(p_coefs, x_line)
        ax_main.plot(x_line, y_line, color='#d62728', linewidth=3.0, label='Entwicklungstrend (Polynom 3. Grades)')
        ax_main.fill_between(x_line, y_line - 300, y_line + 300, color='#d62728', alpha=0.10, label='Konfidenzband')
        
    # Vertikale Zonen
    ax_main.axvspan(0.50, 0.85, color='green', alpha=0.08)
    ax_main.axvspan(0.85, 1.5, color='yellow', alpha=0.06)
    ax_main.axvspan(1.5, 4.5, color='red', alpha=0.08)
    
    # Zonen beschriften
    ax_main.text(0.67, 1003600, "Pheromon-geführt\n(OPTIMAL)", color='forestgreen', fontweight='bold', ha='center', fontsize=9.5)
    ax_main.text(1.17, 1003600, "Übergang", color='#b8860b', fontweight='bold', ha='center', fontsize=9.5)
    ax_main.text(3.0, 1003600, "Heuristik-dominiert (FALLE)\n(Greedy-Falle greift)", color='darkred', fontweight='bold', ha='center', fontsize=9.5)
    
    # Leitlinien
    ax_main.axvline(2.5, color='grey', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_main.annotate(
        "Typisches β für Standard-KP\n(Standard-Literatur: beta = 2 - 5)",
        xy=(2.5, 1004600), xytext=(2.9, 1005000),
        arrowprops=dict(arrowstyle="->", color='black', linewidth=1.0),
        fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="grey", alpha=0.9)
    )
    
    ax_main.axvline(0.65, color='forestgreen', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_main.annotate(
        "Optimum für NMGE-Instanz",
        xy=(0.65, 1005700), xytext=(1.2, 1005900),
        arrowprops=dict(arrowstyle="->", color='forestgreen', connectionstyle="arc3,rad=-0.15", linewidth=1.0),
        fontsize=9, fontweight='bold', color='forestgreen', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="forestgreen", alpha=0.9)
    )
    
    # Verschiebungspfeil
    ax_main.annotate(
        "Verschiebung durch\nNMGE-Täuschung",
        xy=(0.7, 1005300), xytext=(2.4, 1005300),
        arrowprops=dict(arrowstyle="<->", color='black', linewidth=1.2, linestyle=':'),
        fontsize=9.5, fontweight='bold', ha='center'
    )
    
    # Optimum- & Best-ACO-Linien
    ax_main.axhline(OPTIMAL_VALUE, color='#d62728', linestyle=':', linewidth=1.5)
    ax_main.text(4.45, OPTIMAL_VALUE - 180, 'Optimum (1.011.638)', color='#d62728', fontsize=8.5, fontweight='bold', ha='right')
    
    ax_main.axhline(BEST_ACO_VALUE, color='forestgreen', linestyle='-.', linewidth=1.2)
    ax_main.text(4.45, BEST_ACO_VALUE + 80, f'Bestes ACO ({BEST_ACO_VALUE:,.0f})'.replace(",", "."), color='forestgreen', fontsize=8.5, fontweight='bold', ha='right')
    
    ax_main.set_xlabel('Heuristikgewichtung Beta (β)', fontweight='bold')
    ax_main.set_ylabel('Median-Wert (Fitness)', fontweight='bold')
    ax_main.set_xlim(0.35, 4.5)
    ax_main.set_ylim(1003400, 1006500)
    ax_main.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax_main.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    
    # Zweite Y-Achse für Gap (%)
    ax_gap = ax_main.twinx()
    ax_gap.set_ylim(ax_main.get_ylim())
    gap_ticks = [1003500, 1004000, 1004500, 1005000, 1005500, 1006000, 1006500]
    ax_gap.set_yticks(gap_ticks)
    ax_gap.set_yticklabels([f"{(OPTIMAL_VALUE - val)/OPTIMAL_VALUE*100:.2f}%" for val in gap_ticks])
    ax_gap.set_ylabel("Gap zum Optimum (%)", fontweight='bold')
    ax_gap.grid(False)
    
    ax_main.set_title("A. Heuristik-Gewichtung vs. Lösungsqualität (Beta-Fallen-Effekt)", fontweight='bold', loc='left')
    
    # ---------------- UNTERER TEIL: Konzeptuelle Gegenüberstellung ----------------
    ax_boxes.axis('off')
    
    # Kasten 1 (Standard-Knapsack)
    std_text = (
        "STANDARD-KNAPSACK\n\n"
        "• Bestes η-Item (Nutzen/Gewicht) →\n"
        "  gehört zur optimalen Lösung [Ja]\n"
        "• Hohes Beta hilft bei der Suche [Ja]\n"
        "• Empfehlung: β = 2 - 5"
    )
    ax_boxes.text(
        0.05, 0.5, std_text, fontsize=9.5, fontweight='bold',
        ha='left', va='center', color='black',
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#2ca02c", lw=2)
    )
    
    # Pfeil in der Mitte
    pfeil_text = (
        "NMGE-Konstruktion\n"
        "kehrt die Logik um\n"
        "=============>"
    )
    ax_boxes.text(
        0.50, 0.5, pfeil_text, fontsize=9.5, fontweight='bold',
        ha='center', va='center', color='black'
    )
    
    # Kasten 2 (NMGE-Knapsack)
    nmge_text = (
        "NMGE-KNAPSACK\n\n"
        "• Bestes η-Item (Nutzen/Gewicht) →\n"
        "  GEHÖRT NICHT zur optimalen Lösung [Nein]\n"
        "• Hohes Beta führt in die Falle [Nein]\n"
        "• Empfehlung: β = 0.6 - 0.8"
    )
    ax_boxes.text(
        0.95, 0.5, nmge_text, fontsize=9.5, fontweight='bold',
        ha='right', va='center', color='black',
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#d62728", lw=2)
    )
    
    ax_boxes.set_title("B. Konzeptuelle Erklärung der NMGE-Heuristikfalle", fontweight='bold', loc='left', pad=10)
    
    plt.suptitle("NMGE-Fallen-Effekt: Heuristik-Gewichtung vs. Lösungsqualität", fontsize=14, fontweight='bold', y=0.98)
    
    # Speichern
    plt.savefig("presentation/04_nmge_fallen_effekt.png", dpi=300, bbox_inches='tight')
    plt.savefig("presentation/04_nmge_fallen_effekt.pdf", bbox_inches='tight')
    plt.close()
    print("Grafik 4 (NMGE-Fallen-Effekt) erfolgreich in 'presentation/' gespeichert.")


# ===================== GRAFIK 5: RADAR-CHART =====================

def plot_radar_chart(df):
    """Generiert Grafik 5: Radar-Chart (6 Achsen, Beta invertiert, außen=gut)."""
    df_radar = df.copy()
    df_radar['alpha_beta'] = df_radar['alpha'] / df_radar['beta']
    
    # Normalisieren (Min-Max) aller 6 Dimensionen
    params = ['alpha', 'beta', 'evaporation', 'group_size', 'alpha_beta', 'median_iteration']
    
    for p in params:
        p_min = df_radar[p].min()
        p_max = df_radar[p].max()
        if p_max != p_min:
            df_radar[f'{p}_norm'] = (df_radar[p] - p_min) / (p_max - p_min)
        else:
            df_radar[f'{p}_norm'] = 1.0
            
    # Beta-Achse invertieren (niedriges Beta = hohe Effizienz = außen)
    df_radar['beta_norm'] = 1.0 - df_radar['beta_norm']
    
    # Top 5 und Bottom 5 filtern
    top5 = df_radar.sort_values('median_value', ascending=False).head(5)
    bottom5 = df_radar.sort_values('median_value', ascending=True).head(5)
    
    top_thresh = top5['median_value'].min()
    bot_thresh = bottom5['median_value'].max()
    
    categories = ['Alpha (α)', 'Beta (inv) *', 'Evaporation (ρ)', 'Gruppengröße', 'α / β Ratio', 'Med. Iteration']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    
    green_colors = ['#2ca02c', '#98df8a', '#006400', '#32CD32', '#00FF7F']
    red_colors = ['#d62728', '#ff7f7f', '#8B0000', '#FF4500', '#FF6347']
    
    norm_cols = [f'{p}_norm' for p in params]
    
    # Top 5 Einzellinien
    for i, (_, row) in enumerate(top5.iterrows()):
        values = row[norm_cols].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=green_colors[i], linewidth=1.2, linestyle='--', alpha=0.45)
        
    # Bottom 5 Einzellinien
    for i, (_, row) in enumerate(bottom5.iterrows()):
        values = row[norm_cols].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=red_colors[i], linewidth=1.2, linestyle='--', alpha=0.45)
        
    # Top 5 Mittelwert & Füllung
    top5_mean = top5[norm_cols].mean().values.flatten().tolist()
    top5_mean += top5_mean[:1]
    ax.plot(angles, top5_mean, color='#2ca02c', linewidth=2.5, label=f'Mittelwert Top-5 (Median ≥ {top_thresh:,.0f})'.replace(",", "."))
    ax.fill(angles, top5_mean, color='#2ca02c', alpha=0.20)
    
    # Bottom 5 Mittelwert & Füllung
    bottom5_mean = bottom5[norm_cols].mean().values.flatten().tolist()
    bottom5_mean += bottom5_mean[:1]
    ax.plot(angles, bottom5_mean, color='#d62728', linewidth=2.5, label=f'Mittelwert Bottom-5 (Median ≤ {bot_thresh:,.0f})'.replace(",", "."))
    ax.fill(angles, bottom5_mean, color='#d62728', alpha=0.20)
    
    # Polar Axis Styling
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, fontsize=10.5, fontweight='bold')
    
    # Beta-Achse extra kennzeichnen
    # Der Winkel von Beta(inv) ist der 2. Winkel (index 1)
    beta_angle = angles[1]
    ax.annotate(
        "niedriges Beta = außen",
        xy=(beta_angle, 0.95), xytext=(beta_angle + 0.15, 1.15),
        arrowprops=dict(arrowstyle="->", color='black', lw=0.8),
        fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="grey", alpha=0.9)
    )
    
    ax.set_rgrids([0.25, 0.5, 0.75, 1.0], ['0.25', '0.50', '0.75', '1.00'], color='grey', size=8, alpha=0.6)
    ax.set_ylim(0, 1.08)
    ax.grid(True, color='grey', linestyle=':', alpha=0.5)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), frameon=True, shadow=True, facecolor='white', ncol=1)
    
    plt.title("Parameterprofil: Top-5 vs. Bottom-5 Konfigurationen\n(Alle Achsen: außen = besser | * Beta invertiert)", fontsize=13, fontweight='bold', pad=25)
    
    # Speichern
    plt.savefig("presentation/05_radar_profilvergleich.png", dpi=300, bbox_inches='tight')
    plt.savefig("presentation/05_radar_profilvergleich.pdf", bbox_inches='tight')
    plt.close()
    print("Grafik 5 (Radar Profilvergleich) erfolgreich in 'presentation/' gespeichert.")


# ===================== GRAFIK 6: KONVERGENZ-SCATTER (BONUS) =====================

def plot_convergence_scatter(df):
    """Generiert Grafik 6: Konvergenz-Scatter (Bonus Plot zur Analyse der Konvergenzgeschwindigkeit)."""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    
    # Scatter plot: X=Median-Iteration, Y=Median-Wert, Color=Beta, Size=GroupSize
    # Normalisiere die Punktgröße auf visuell lesbare Pixelwerte
    point_sizes = (df['group_size'] / df['group_size'].max()) * 120 + 20
    
    sc = ax.scatter(
        df['median_iteration'], df['median_value'],
        c=df['beta'], s=point_sizes,
        cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidths=0.5
    )
    
    cbar = fig.colorbar(sc, ax=ax, label='Heuristikgewichtung Beta (β)')
    cbar.ax.set_ylabel('Heuristikgewichtung Beta (β)', fontweight='bold')
    
    # Trendlinie einzeichnen (Zusammenhang: spät gefunden = besser)
    p_coefs = np.polyfit(df['median_iteration'], df['median_value'], 1)
    x_line = np.linspace(df['median_iteration'].min(), df['median_iteration'].max(), 100)
    y_line = np.polyval(p_coefs, x_line)
    ax.plot(x_line, y_line, color='darkblue', linestyle='--', linewidth=2, label='Linearer Trend')
    
    # Quadranten-Trennlinien
    y_sep = 1005000
    x_sep = 50
    ax.axhline(y_sep, color='grey', linestyle=':', alpha=0.6, linewidth=1.2)
    ax.axvline(x_sep, color='grey', linestyle=':', alpha=0.6, linewidth=1.2)
    
    # Quadranten beschriften
    ax.text(
        x_sep - 5, y_sep - 250, "FALLE\nSchlecht + Früh konvergiert\n(Beta hoch)", 
        color='darkred', fontsize=9, fontweight='bold', ha='right', va='top',
        bbox=dict(facecolor='white', edgecolor='#ffcccc', alpha=0.85, boxstyle='round,pad=0.2')
    )
    
    ax.text(
        x_sep + 5, y_sep + 150, "IDEAL\nGut + Spät gefunden\n(Beta niedrig & geduldig)", 
        color='forestgreen', fontsize=9, fontweight='bold', ha='left', va='bottom',
        bbox=dict(facecolor='white', edgecolor='#ccffcc', alpha=0.85, boxstyle='round,pad=0.2')
    )
    
    ax.text(
        x_sep - 5, y_sep + 150, "GLÜCK\nGut + Schnell\n(Sehr selten)", 
        color='darkblue', fontsize=9, fontweight='bold', ha='right', va='bottom',
        bbox=dict(facecolor='white', edgecolor='#cce6ff', alpha=0.85, boxstyle='round,pad=0.2')
    )
    
    ax.text(
        x_sep + 5, y_sep - 250, "STAGNATION\nSchlecht + Lange gesucht\n(Suboptimale Divergenz)", 
        color='grey', fontsize=9, fontweight='bold', ha='left', va='top',
        bbox=dict(facecolor='white', edgecolor='#eeeeee', alpha=0.85, boxstyle='round,pad=0.2')
    )
    
    # Optimum- & Best-ACO-Linien
    ax.axhline(OPTIMAL_VALUE, color='#d62728', linestyle=':', linewidth=1.2, alpha=0.8)
    ax.text(ax.get_xlim()[1] - 2, OPTIMAL_VALUE - 180, 'Optimum', color='#d62728', fontsize=8, fontweight='bold', ha='right')
    
    ax.axhline(BEST_ACO_VALUE, color='forestgreen', linestyle='-.', linewidth=1.0, alpha=0.8)
    ax.text(ax.get_xlim()[1] - 2, BEST_ACO_VALUE + 80, 'Bestes ACO', color='forestgreen', fontsize=8, fontweight='bold', ha='right')
    
    # Legendengröße nach Gruppengröße simulieren
    for size in [20, 80, 140]:
        ax.scatter([], [], c='grey', alpha=0.6, s=(size/df['group_size'].max())*120 + 20, label=f'G = {size}')
        
    ax.set_xlabel('Median-Iteration der besten Lösung', fontweight='bold')
    ax.set_ylabel('Median-Wert (Fitness)', fontweight='bold')
    ax.set_ylim(1003400, 1006500)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='lower right', title='Gruppengröße (G)', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
    
    plt.title("Konvergenzverhalten: Lösungsqualität vs. Suchdauer\n(Farbe repräsentiert Beta | Punktgröße repräsentiert Gruppengröße)", fontsize=13, fontweight='bold', pad=15)
    
    # Speichern
    plt.savefig("presentation/06_konvergenz_scatter.png", dpi=300, bbox_inches='tight')
    plt.savefig("presentation/06_konvergenz_scatter.pdf", bbox_inches='tight')
    plt.close()
    print("Grafik 6 (Konvergenz Scatter) erfolgreich in 'presentation/' gespeichert.")


# ===================== HAUPTPROGRAMM =====================

def main():
    print("================================================================================")
    print(" Start: Generierung der Präsentations-Visualisierungen (Revised)")
    print("================================================================================\n")
    
    # Ordner erstellen, falls nicht vorhanden
    os.makedirs("presentation", exist_ok=True)
    
    # Daten laden und falls nötig kompilieren
    try:
        df = compile_and_load_data()
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        sys.exit(1)
        
    print(f"Datensatz geladen. Zeilenanzahl: {len(df)}")
    
    # Visualisierungsfunktionen aufrufen
    plot_learning_curve(df)
    plot_main_effects(df)
    plot_interaction_plots(df)
    plot_nmge_trap(df)
    plot_radar_chart(df)
    plot_convergence_scatter(df)
    
    print("\n================================================================================")
    print(" Fertig: Alle 6 Grafiken wurden erfolgreich in 'presentation/' gespeichert!")
    print("================================================================================")


if __name__ == "__main__":
    main()
