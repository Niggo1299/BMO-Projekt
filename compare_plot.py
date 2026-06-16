"""Erzeugt 4 Vergleichsgrafiken aus gespeicherten Simulationsergebnissen."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except (OSError, ValueError):
    plt.style.use('default')

plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white',
                     'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--'})

INPUT_FILE = "data/compare_results.json"
OUTPUT_DIR = "presentation"

COLORS = {'Beste': '#2ca02c', 'Mittlere': '#ff7f0e', 'Schlechteste': '#d62728'}


def make_label(name, config):
    return rf"{name} ($\alpha$={config['alpha']}, $\beta$={config['beta']}, $\rho$={config['evaporation']}, G={config['group_size']})"


def get_region_bounds(number_items, problem_data):
    """Bestimmt NMGE-Bereichsgrenzen."""
    if number_items == 600:
        return 264, 476, 599
    elif number_items == 800:
        return 384, 714, 799
    else:
        return int(number_items * 0.44), int(number_items * 0.79), number_items - 1


def add_region_shading(ax, heavy_end, light_end, fine_end):
    ax.axvspan(0, heavy_end, color='#d62728', alpha=0.07, label='Schwere Gegenstände')
    ax.axvspan(heavy_end + 1, light_end, color='#2ca02c', alpha=0.07, label='Leichte Gegenstände')
    ax.axvspan(light_end + 1, fine_end, color='#1f77b4', alpha=0.07, label='Kleinstgegenstände')


def save_fig(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{OUTPUT_DIR}/{name}.pdf", bbox_inches='tight')
    plt.close(fig)


def main():
    with open(INPUT_FILE, "r") as f:
        results = json.load(f)

    with open("data/problem.json", "r") as f:
        problem_data = json.load(f)

    names = list(results.keys())
    number_items = len(results[names[0]]['pheromone'])
    iterations = len(results[names[0]]['best_curve'])
    optimal_value = problem_data["optimal_solution"]["value"]
    max_load = problem_data["max_load"]
    optimal_items = problem_data["optimal_solution"]["items"]

    heavy_end, light_end, fine_end = get_region_bounds(number_items, problem_data)

    # --- Plot A: Normierte Pheromonspur ---
    fig, ax = plt.subplots(figsize=(12, 5))
    add_region_shading(ax, heavy_end, light_end, fine_end)

    for name in names:
        data = results[name]
        norm_phe = np.array(data['pheromone']) / data['config']['group_size']
        ax.plot(range(number_items), norm_phe, color=COLORS[name], linewidth=1.5, alpha=0.75,
                label=make_label(name, data['config']))

    ax.axhline(0, color='black', linewidth=1.2)
    ax.set_title("Normierte Pheromonspur (τ / Gruppengröße)", fontweight='bold')
    ax.set_xlabel("Gegenstands-ID")
    ax.set_ylabel("Pheromon τ (normiert)")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    save_fig(fig, "07a_vergleich_pheromonschnitt")

    # --- Plot B: Konvergenz Nutzwert ---
    fig, ax = plt.subplots(figsize=(12, 5))
    iter_range = range(1, iterations + 1)

    for name in names:
        ax.plot(iter_range, results[name]['best_curve'], color=COLORS[name],
                linewidth=2.5, label=make_label(name, results[name]['config']))

    ax.axhline(optimal_value, color='black', linestyle=':', linewidth=1.8,
               label=f'Optimum ({optimal_value:,})'.replace(",", "."))
    ax.set_title("Konvergenzverlauf des Kofferwerts", fontweight='bold')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Rucksackwert")
    # Achsengrenzen dynamisch berechnen mit 5% Padding
    all_vals = [val for name in names for val in results[name]['best_curve']] + [optimal_value]
    min_val, max_val = min(all_vals), max(all_vals)
    span = max_val - min_val
    padding = max(span * 0.05, 500)
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    save_fig(fig, "07b_vergleich_nutzwert")

    # --- Plot C: Gewichtsverlauf ---
    fig, ax = plt.subplots(figsize=(12, 5))

    for name in names:
        ax.plot(iter_range, results[name]['weight_curve'], color=COLORS[name],
                linewidth=2.5, label=make_label(name, results[name]['config']))

    ax.axhline(max_load, color='black', linestyle=':', linewidth=1.8,
               label=f'Kapazität ({max_load:,})'.replace(",", "."))
    ax.set_title("Konvergenzverlauf des Rucksackgewichts", fontweight='bold')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Rucksackgewicht")
    ax.set_ylim(997_000, 1_002_000)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    save_fig(fig, "07c_vergleich_gewicht")

    # --- Plot D: Gegenstandsauswahl ---
    fig, ax = plt.subplots(figsize=(12, 5))
    add_region_shading(ax, heavy_end, light_end, fine_end)

    ax.scatter(optimal_items, [3] * len(optimal_items), color='purple', marker='|', s=80, alpha=0.7, label='Optimum')

    for y_pos, name in enumerate(reversed(names)):
        bp = results[name]['best_backpack']
        selected = [i for i, v in enumerate(bp) if v == 1]
        ax.scatter(selected, [y_pos] * len(selected), color=COLORS[name], marker='|', s=80, alpha=0.7,
                   label=make_label(name, results[name]['config']))

    ax.set_title("Gegenstandsauswahl (beste Einzellösungen)", fontweight='bold')
    ax.set_xlabel("Gegenstands-ID")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Schlechteste', 'Mittlere', 'Beste', 'Optimum'])
    ax.set_ylim(-0.5, 3.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    save_fig(fig, "07d_vergleich_itemauswahl")

    print(f"4 Grafiken gespeichert unter '{OUTPUT_DIR}/'")


if __name__ == '__main__':
    main()