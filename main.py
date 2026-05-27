import random
import json
from ant import ant
from item import item
import visualization

def main():
    
    group_size = 20         # Anzahl Ameisen pro Iteration
    evaporation_rate = 0.3 # Verdampfung der Pheromone pro Iteration (MMAS Empfehlung)
    iterations = 100
    alpha = 1.0             # Einfluss der Pheromone
    beta = 2.0              # Einfluss der Heuristik (Wert/Gewicht)
    
    # Listen für das Tracking (Lernkurve)
    best_fitness_per_round = []
    avg_fitness_per_round = []
    
    # All-Time-Best Variablen
    global_best_value = -1
    global_best_backpack = []
    global_best_weight = 0

    # Problem laden
    with open("problem.json", "r") as f:
        problem_data = json.load(f)
        
    number_items = problem_data["number_items"]
    max_load = problem_data["max_load"]
    
    items = []
    for data in problem_data["items"]:
        items.append(item(data["id"], data["weight"], data["value"]))
        
    # create ants
    ants = []
    for i in range(group_size):
        ants.append(ant(max_load, number_items))
        
    # Setup Live-Plot
    fig, ax1, ax2 = visualization.setup_live_plot()
        
    # main loop
    for iteration in range(iterations):
        for a in ants:
            starting_position = random.randint(0, number_items - 1)
            a.reset()
            for position in range(number_items):
                current_item = items[(starting_position + position) % number_items]
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
        
        # Pheromone updaten (Ant-Cycle: Verdampfung + ALLE Ameisen legen Pheromone ab)
        # 1. Verdampfung
        for current_item in items:
            current_item.evaporate(evaporation_rate)
            
        # 2. Alle Ameisen auswerten und Pheromone ablegen
        for a in ants:
            for current_item in items:
                decision = a.backpack[current_item.id]
                current_item.add_reward(decision, a.current_value)
            
        # Live-Visualisierung in jedem Schritt aktualisieren
        visualization.update_live_plot(fig, ax1, ax2, items, iteration + 1, number_items, best_fitness_per_round, avg_fitness_per_round)

    # Auswertung
    print("=== OPTIMIERUNG ABGESCHLOSSEN ===")
    print(f"Bester Rucksack-Wert: {global_best_value}")
    print(f"Benutztes Gewicht:    {global_best_weight} / {max_load}")
    print(f"Rucksack-Belegung:    {global_best_backpack}")
    
    # Am Ende das Fenster offen lassen
    visualization.show_final()

if __name__ == "__main__":
    main()
