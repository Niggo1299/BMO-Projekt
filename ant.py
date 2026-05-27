import random

class ant:
    def __init__(self, max_load, num_items):
        self.max_weight = max_load
        self.num_items = num_items
        self.backpack = [0] * num_items
        self.current_load = 0
        self.current_value = 0
    
    def reset(self):
        self.backpack = [0] * self.num_items
        self.current_load = 0
        self.current_value = 0
    
    def pre_check(self, item):
        if self.current_load + item.weight > self.max_weight:
            return True
        return False
    
    def decision (self, item, alpha, beta):
        if self.pre_check(item):
            self.backpack[item.id] = 0
            return False
        
        # Schritt 1: Berechnung der gewichteten Übergangswerte (Zähler)
        # A_ja = (Tau_ja)^alpha * (Eta_ja)^beta
        a_yes = (item.pheromone_yes ** alpha) * (item.attractiveness_yes ** beta)
        
        # A_nein = (Tau_nein)^alpha * (Eta_nein)^beta
        a_no = (item.pheromone_no ** alpha) * (item.attractiveness_no ** beta)
        
        total_a = a_yes + a_no
        
        if total_a == 0:
            self.backpack[item.id] = 0
            return False
        
        # Schritt 2: Berechnung der exakten Wahrscheinlichkeit für JA (p_ja)
        prob_yes = a_yes / total_a
        
        rand_value = random.random()
        
        if rand_value < prob_yes:
            self.backpack[item.id] = 1
            self.current_load += item.weight
            self.current_value += item.value
            return True
        else:
            self.backpack[item.id] = 0
            return False