import random
from ant import ant
from item import item
import visualization

def main():
    number_items = 10
    group_size = 5
    max_load = 15
    evaporation_rate = 0.1
    iterations = 100
    alpha = 0.5
    beta = 0.5
    
    # Listen für das Tracking (Lernkurve)
    best_fitness_per_round = []
    avg_fitness_per_round = []
    
    # All-Time-Best Variablen
    global_best_value = -1
    global_best_backpack = []
    global_best_weight = 0

    # create items
    items = []
    for i in range(number_items):
        weight = random.randint(1, 10)
        value = random.randint(1, 10)
        items.append(item(i, weight, value))
        
    # create ants
    ants = []
    for i in range(group_size):
        ants.append(ant(max_load))
        
    # main loop
    for iteration in range(iterations):
        for a in ants:
            a.reset()
            for current_item in items:
                a.decision(current_item, alpha, beta)
                
        # Finde beste Ameise DIESER Runde
        round_best_ant = max(ants, key=lambda x: x.current_value)
        
        # All-Time-Best aktualisieren
        if round_best_ant.current_value > global_best_value:
            global_best_value = round_best_ant.current_value
            global_best_backpack = list(round_best_ant.backpack)
            global_best_weight = round_best_ant.current_load
            
        # Daten für Lernkurve speichern
        avg_value = sum(a.current_value for a in ants) / group_size
        best_fitness_per_round.append(round_best_ant.current_value)
        avg_fitness_per_round.append(avg_value)
        
        # Pheromone updaten (Verdampfung + Elitär-Belohnung)
        for current_item in items:
            current_item.evaporate(evaporation_rate)
            # Im elitären System belohnt NUR die All-Time-Best Ameise ihren Weg!
            decision = global_best_backpack[current_item.id]
            current_item.add_reward(decision, global_best_value)
            
        # Visualisierungs-Trigger (Pfade) gemäß Vorgabe 3D
        if (iteration + 1) in [1, 5, 20, 50, iterations]:
            visualization.plot_paths(ants, iteration + 1, number_items)

    # Auswertung
    print("=== OPTIMIERUNG ABGESCHLOSSEN ===")
    print(f"Bester Rucksack-Wert: {global_best_value}")
    print(f"Benutztes Gewicht:    {global_best_weight} / {max_load}")
    print(f"Rucksack-Belegung:    {global_best_backpack}")
    
    # Lernkurve generieren gemäß Vorgabe 3-Auswertung
    visualization.plot_learning_curve(best_fitness_per_round, avg_fitness_per_round)

if __name__ == "__main__":
    main()
