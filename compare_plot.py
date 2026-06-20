"""Erzeugt 4 Vergleichsgrafiken aus gespeicherten Simulationsergebnissen."""

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from problem import DIFFICULTY_CONFIGS

# --- Globale Konfiguration ---
INPUT_FILE = "data/compare_results.json"
OUTPUT_DIR = "presentation"

COLORS = {'Beste': '#2ca02c', 'Mittlere': '#ff7f0e', 'Schlechteste': '#d62728'}

# Einheitlicher Plot-Stil
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0,
})
try:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 11})
except (OSError, ValueError):
    pass

# Einheitliche Layout-Parameter
FIG_WIDE = (12, 5.5)
LEGEND_KW = dict(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=9)
TIGHT_RECT = [0, 0, 0.72, 1]
REFLINE_KW = dict(color='black', linestyle=':', linewidth=1.8)
SCATTER_KW = dict(marker='|', s=80, alpha=0.9, zorder=3)
ITEM_COLOR = '#1f77b4'  # Einheitliches Blau für Itemauswahl


# --- Hilfsfunktionen ---

def make_label(name, cfg):
    return (rf"{name} ($\alpha$={cfg['alpha']}, $\beta$={cfg['beta']}, "
            rf"$\rho$={cfg['evaporation']}, G={cfg['group_size']})")


def get_group_info(n):
    g, f = 14, 0.1
    for cfg in DIFFICULTY_CONFIGS.values():
        if cfg['n'] == n:
            g = cfg['g']
            f = cfg['f']
            break
    last = int(math.floor(n * f))
    m = int(math.floor((n - last) / (g - 1)))
    bounds = []
    start = 0
    for _ in range(g - 1):
        bounds.append((start, start + m - 1))
        start += m
    bounds.append((start, n - 1))
    return g, bounds


def get_group_color(group_idx, total):
    if total <= 1:
        return '#2ca02c'
    return plt.cm.RdYlGn((group_idx - 1) / (total - 1))


def add_group_shading(ax, bounds, total):
    for i, (s, e) in enumerate(bounds):
        ax.axvspan(s, e + 1, color=get_group_color(i + 1, total), alpha=0.08, zorder=0)


def add_group_colorbar(fig, ax, total, g1_pct, gn_pct):
    cmap = plt.cm.RdYlGn
    norm = mcolors.BoundaryNorm(np.arange(1, total + 2), cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.18, aspect=45, shrink=0.85)
    cbar.set_ticks(np.arange(1, total + 1) + 0.5)
    labels = [f"G{i}" for i in range(1, total + 1)]
    labels[0] = f"G1 (Ø: {g1_pct:.4f}%)"
    labels[-1] = f"G{total} (Ø: {gn_pct:.4f}%)"
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Gegenstands-Gruppen (G1 [Groß/Rot] → Gn [Klein/Grün])",
                   fontsize=9, fontweight='bold')


def save_fig(fig, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f"{output_dir}/{name}.{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)


def dynamic_ylim(ax, values, padding_min=500):
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.05, padding_min)
    ax.set_ylim(lo - pad, hi + pad)


# --- Hauptprogramm ---

def main():
    import sys
    diffs = ["easy", "medium", "hard"]
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
        if choice not in diffs:
            print(f"Ungueltig: '{choice}'. Erlaubt: {', '.join(diffs)}")
            sys.exit(1)
        diffs = [choice]

    for diff in diffs:
        print("=" * 60)
        print(f" Erzeuge Grafiken fuer {diff.upper()}")
        print("=" * 60)
        
        prob_file = f"data/{diff}/problem.json"
        res_file = f"data/{diff}/compare_results.json"
        out_dir = f"presentation/{diff}"
        
        if not os.path.exists(res_file):
            print(f"Fehler: Datei '{res_file}' existiert nicht. Bitte zuerst compare_sim.py ausfuehren.")
            continue
            
        with open(res_file) as f:
            results = json.load(f)
        with open(prob_file) as f:
            problem = json.load(f)

        names = list(results.keys())
        n_items = len(results[names[0]]['pheromone'])
        n_iter = len(results[names[0]]['best_curve'])
        optimum = problem["optimal_solution"]["value"]
        capacity = problem["max_load"]
        opt_items = problem["optimal_solution"]["items"]
        g, bounds = get_group_info(n_items)

        def avg_weight_pct(bound):
            items = problem["items"][bound[0]:bound[1] + 1]
            return np.mean([it["weight"] for it in items]) / capacity * 100.0

        g1_pct, gn_pct = avg_weight_pct(bounds[0]), avg_weight_pct(bounds[-1])
        iter_range = range(1, n_iter + 1)

        # --- A: Pheromonspur ---
        fig, ax = plt.subplots(figsize=FIG_WIDE)
        add_group_shading(ax, bounds, g)
        for name in names:
            d = results[name]
            norm_phe = np.array(d['pheromone']) / d['config']['group_size']
            ax.plot(range(n_items), norm_phe, color=COLORS[name],
                    linestyle='-', alpha=0.75, label=make_label(name, d['config']))
        ax.axhline(0, color='black', linewidth=1.2)
        ax.set(title="Pheromonspur (τ / Gruppengröße)", xlabel="Gegenstands-ID", ylabel="Pheromon τ (normiert)")
        ax.legend(**LEGEND_KW)
        add_group_colorbar(fig, ax, g, g1_pct, gn_pct)
        plt.tight_layout(rect=TIGHT_RECT)
        save_fig(fig, "07a_vergleich_pheromonschnitt", out_dir)

        # --- B: Konvergenz Nutzwert ---
        fig, ax = plt.subplots(figsize=FIG_WIDE)
        for name in names:
            ax.plot(iter_range, results[name]['best_curve'], color=COLORS[name],
                    linestyle='-', label=make_label(name, results[name]['config']))
        ax.axhline(optimum, **REFLINE_KW, label=f'Optimum ({optimum:,.0f})'.replace(",", "."))
        ax.set(title="Konvergenzverlauf des Kofferwerts", xlabel="Iteration", ylabel="Rucksackwert")
        dynamic_ylim(ax, [v for n in names for v in results[n]['best_curve']] + [optimum])
        ax.legend(**LEGEND_KW)
        plt.tight_layout(rect=TIGHT_RECT)
        save_fig(fig, "07b_vergleich_nutzwert", out_dir)

        # --- C: Gewichtsverlauf ---
        fig, ax = plt.subplots(figsize=FIG_WIDE)
        for name in names:
            ax.plot(iter_range, results[name]['weight_curve'], color=COLORS[name],
                    linestyle='-', label=make_label(name, results[name]['config']))
        ax.axhline(capacity, **REFLINE_KW, label=f'Kapazität ({capacity:,.0f})'.replace(",", "."))
        ax.set(title="Konvergenzverlauf des Rucksackgewichts", xlabel="Iteration", ylabel="Rucksackgewicht")
        dynamic_ylim(ax, [w for n in names for w in results[n]['weight_curve']] + [capacity])
        ax.legend(**LEGEND_KW)
        plt.tight_layout(rect=TIGHT_RECT)
        save_fig(fig, "07c_vergleich_gewicht", out_dir)

        # --- D: Gegenstandsauswahl ---
        fig, ax = plt.subplots(figsize=FIG_WIDE)
        add_group_shading(ax, bounds, g)
        ax.scatter(opt_items, [3] * len(opt_items), color=ITEM_COLOR, **SCATTER_KW)
        for y, name in enumerate(reversed(names)):
            sel = [i for i, v in enumerate(results[name]['best_backpack']) if v == 1]
            ax.scatter(sel, [y] * len(sel), color=ITEM_COLOR, **SCATTER_KW)
        ax.set(title="Gegenstandsauswahl (beste Einzellösungen)", xlabel="Gegenstands-ID")
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels([make_label(n, results[n]['config']) for n in reversed(names)] + ['Optimum'], fontsize=9)
        ax.set_ylim(-0.5, 3.5)
        add_group_colorbar(fig, ax, g, g1_pct, gn_pct)
        plt.tight_layout()
        save_fig(fig, "07d_vergleich_itemauswahl", out_dir)

        print(f"4 Grafiken gespeichert unter '{out_dir}/'\n")


if __name__ == '__main__':
    main()