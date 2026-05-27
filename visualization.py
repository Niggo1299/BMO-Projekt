import matplotlib.pyplot as plt

def setup_live_plot():
    plt.ion()  # Interaktiven Modus aktivieren
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    return fig, ax1, ax2

def update_live_plot(fig, ax1, ax2, items, iteration, total_items, best_fitness, avg_fitness):
    # Linker Plot: Pheromonspuren
    ax1.clear()
    x_axis = range(total_items)
    
    yes_pheromones = [item.pheromone_yes for item in items]
    no_pheromones = [-item.pheromone_no for item in items]  # Negativ für y- Achse
    
    ax1.bar(x_axis, yes_pheromones, color='green', label='JA Pheromon', alpha=0.7)
    ax1.bar(x_axis, no_pheromones, color='red', label='NEIN Pheromon', alpha=0.7)
    
    ax1.axhline(0, color='black', linewidth=1) # Nulllinie
    ax1.set_title(f'Pheromonspuren - Iteration {iteration}')
    ax1.set_xlabel('Gegenstands-ID')
    ax1.set_ylabel('Pheromonmenge (JA + / NEIN -)')
    ax1.set_xticks(x_axis)
    ax1.legend(loc='upper right')
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