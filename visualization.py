"""
Visualisierungsmodul: Live-Plot für Pheromonspuren und Lernkurve.

Zeigt in Echtzeit:
- Oberer Plot: Pheromonverteilung pro Item (JA/NEIN)
- Unterer Plot: Lernkurve (bester und durchschnittlicher Fitness-Wert)
"""

import matplotlib.pyplot as plt


def setup_live_plot():
    """
    Initialisiert das Plotfenster mit zwei Subplots.

    Returns:
        fig, ax1, ax2: Matplotlib Figure und Axes-Objekte.
    """
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    return fig, ax1, ax2


def update_live_plot(fig, ax1, ax2, items, iteration, total_items,
                     best_fitness, avg_fitness, mode=""):
    """
    Aktualisiert beide Plots in jedem Iterationsschritt.

    Args:
        fig, ax1, ax2:  Matplotlib Figure und Axes.
        items:          Liste aller Item-Objekte (mit Pheromonwerten).
        iteration:      Aktuelle Iterationsnummer.
        total_items:    Gesamtanzahl der Items.
        best_fitness:   Liste der besten Fitness-Werte pro Iteration.
        avg_fitness:    Liste der durchschnittlichen Fitness-Werte.
        mode:           Aktiver Modus ("AC" oder "EAS") für Titel.
    """
    # --- Oberer Plot: Pheromonspuren pro Item ---
    ax1.clear()
    x_axis = range(total_items)

    yes_pheromones = [item.pheromone_yes for item in items]
    no_pheromones = [-item.pheromone_no for item in items]

    ax1.bar(x_axis, yes_pheromones, color='green', label='JA-Pheromon', alpha=0.7)
    ax1.bar(x_axis, no_pheromones, color='red', label='NEIN-Pheromon', alpha=0.7)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title(f'Pheromonspuren [{mode}] – Iteration {iteration}')
    ax1.set_xlabel('Gegenstands-ID')
    ax1.set_ylabel('Pheromonmenge (JA + / NEIN −)')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Unterer Plot: Lernkurve ---
    ax2.clear()
    iter_range = range(1, len(best_fitness) + 1)
    ax2.plot(iter_range, best_fitness, label='Bester Nutzwert',
             color='green', linewidth=2)
    ax2.plot(iter_range, avg_fitness, label='Ø Nutzwert',
             color='orange', linewidth=2, linestyle='--')
    ax2.set_title(f'Lernkurve [{mode}]')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Kofferwert (Nutzen)')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    plt.tight_layout()
    plt.pause(0.01)


def show_final():
    """Beendet den interaktiven Modus und hält das Fenster offen."""
    plt.ioff()
    plt.show()