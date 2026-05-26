import matplotlib.pyplot as plt

def plot_paths(ants, iteration, total_items):
    """
    GRAFIK 1: Der transparente Pfad-Vergleich (Ameisenwege im Zeitverlauf)
    """
    plt.figure(figsize=(10, 6))
    x_axis = range(total_items)
    
    for ant in ants:
        plt.plot(x_axis, ant.backpack, color='blue', alpha=0.15, linewidth=2)
        
    plt.title(f'Ameisenwege - Iteration {iteration}')
    plt.xlabel('Gegenstands-ID')
    plt.ylabel('Entscheidung (0 = NEIN, 1 = JA)')
    plt.yticks([0, 1])
    plt.xticks(x_axis)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Speichert den Plot als Bilddatei, damit die Fenster den Programmablauf nicht blockieren
    plt.savefig(f'pfade_iteration_{iteration}.png')
    plt.close()

def plot_learning_curve(best_fitness, avg_fitness):
    """
    GRAFIK 2: Die Lernkurve (Kofferwert über Iterationsschritte)
    """
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(best_fitness) + 1)
    
    plt.plot(iterations, best_fitness, label='Beste Fitness (Spitze)', color='green', linewidth=2)
    plt.plot(iterations, avg_fitness, label='Schnitt Fitness (Masse)', color='orange', linewidth=2, linestyle='--')
    
    plt.title('Lernkurve der ACO-Optimierung')
    plt.xlabel('Iterationsschritte')
    plt.ylabel('Kofferwert (Nutzen / Euro)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('lernkurve.png')
    plt.close()