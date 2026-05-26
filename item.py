class item:
    def __init__(self, id, weight, value):
        self.id = id
        self.weight = weight
        self.value = value
        self.attractiveness_yes = value / weight
        self.attractiveness_no = weight / value
        self.pheromone_yes = 0.1
        self.pheromone_no = 0.1
    
    def evaporate (self, evaporation_rate):
        self.pheromone_yes = (1 - evaporation_rate) * self.pheromone_yes
        self.pheromone_no = (1 - evaporation_rate) * self.pheromone_no
        
    def add_reward(self, decision, best_value):
        delta_tau = best_value * 0.1
        if decision == 1:
            self.pheromone_yes += delta_tau
        else:
            self.pheromone_no += delta_tau