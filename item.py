"""Item-Klasse für das 0/1-Rucksackproblem mit Pheromon und Heuristik."""


class Item:
    def __init__(self, id, weight, value):
        self.id = id
        self.weight = weight
        self.value = value
        self.attractiveness = value / weight
        self.pheromone = 1.0

    def evaporate(self, rho):
        self.pheromone *= (1 - rho)

    def add_reward(self, solution_value, optimal_value):
        self.pheromone += solution_value / optimal_value