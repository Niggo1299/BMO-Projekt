import matplotlib.pyplot as plt

def setup_live_plot():
    plt.ion()  # Interaktiven Modus aktivieren
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    return fig, ax1, ax2

def update_live_plot(fig, ax1, ax2, ants, iteration, total_items, best_fitness, avg_fitness):
    # Linker Plot: Wege (Pfade)
    ax1.clear()
    x_axis = range(total_items)
    for ant in ants:
        ax1.plot(x_axis, ant.backpack, color='blue', alpha=0.15, linewidth=2)
    ax1.set_title(f'Ameisenwege - Iteration {iteration}')
    ax1.set_xlabel('Gegenstands-ID')
    ax1.set_ylabel('Entscheidung (0 = NEIN, 1 = JA)')
    ax1.set_yticks([0, 1])
    ax1.set_xticks(x_axis)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Rechter Plot: Lernkurve
    ax2.clear()
    iterations = range(1, len(best_fitness) + 1)
    ax2.plot(iterations, best_fitness, label='Bester Nutzwert', color='green', linewidth=2)
    ax2.plot(iterations, avg_fitness, label='Durschnittlischer Nutzwert', color='orange', linewidth=2, linestyle='--')
    ax2.set_title('Lernkurve')
    ax2.set_xlabel('Iterationsschritte')
    ax2.set_ylabel('Kofferwert (Nutzen)')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    plt.tight_layout()
    plt.pause(0.01)  # Kurze Pause, um die GUI zu aktualisieren und neu zu zeichnen

def show_final():
    plt.ioff() # Interaktiven Modus beenden 
    plt.show() # Plot offen halten am Ende der Optimierung