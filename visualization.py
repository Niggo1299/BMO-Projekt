"""Live-Plot: Pheromonverteilung und Lernkurve."""

import matplotlib.pyplot as plt


def setup_live_plot():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    return fig, ax1, ax2


def update_live_plot(fig, ax1, ax2, items, iteration, total_items,
                     best_fitness, avg_fitness, mode=""):
    ax1.clear()
    pheromones = [item.pheromone for item in items]
    ax1.bar(range(total_items), pheromones, color='steelblue', alpha=0.8)
    ax1.set_title(f'Pheromonspuren [{mode}] – Iteration {iteration}')
    ax1.set_xlabel('Gegenstands-ID')
    ax1.set_ylabel('Pheromonmenge τ')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.clear()
    iter_range = range(1, len(best_fitness) + 1)
    ax2.plot(iter_range, best_fitness, label='Bester', color='green', linewidth=2)
    ax2.plot(iter_range, avg_fitness, label='Durchschnitt', color='orange', linewidth=2, linestyle='--')
    ax2.set_title(f'Lernkurve [{mode}]')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Kofferwert')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    plt.tight_layout()
    plt.pause(0.01)


def show_final():
    plt.ioff()
    plt.show()