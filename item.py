class item:
    def __init__(self, id, weight, value):
        self.id = id
        self.weight = weight
        self.value = value
        self.attractiveness_yes = value / weight
        self.attractiveness_no = weight / value
        # Startwert (keine max/min Grenzen im reinen Ant-Cycle)
        self.pheromone_yes = 1.0
        self.pheromone_no = 1.0
        
    def evaporate(self, evaporation_rate):
        self.pheromone_yes *= (1 - evaporation_rate)
        self.pheromone_no *= (1 - evaporation_rate)
        
    def add_reward(self, decision, ant_value):
        # Beim reinen Ant-Cycle entspricht die Belohnung proportional zur Qualität der Gesamtlösung.
        # Im TSP ist das 1/Strecke. Beim Rucksackproblem ist es der erzielte Nutzwert (skaliert, z.B. / 100.0)
        delta_tau = ant_value / 100.0
        
        if decision == 1:
            self.pheromone_yes += delta_tau
        else:
            self.pheromone_no += delta_tau