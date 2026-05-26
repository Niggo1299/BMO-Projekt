class ant:
    def __init__(self, max_load):
        self.max_weight = max_load
        self.backpack = []
        self.current_load = 0
        self.current_value = 0
    
    def reset(self):
        self.backpack = []
        self.current_load = 0
        self.current_value = 0
    
    def pre_check(self, item):
        if self.current_load + item.weight > self.max_weight:
            return True
        return False
    
    def decision (self, item, alpha, beta):
        if self.pre_check(item):
            self.backpack.append(0)
            return False
        
        attractiveness_yes = item.attractiveness_yes ** beta
        attractiveness_no = item.attractiveness_no ** beta
        pheromone_yes = item.pheromone_yes ** alpha
        pheromone_no = item.pheromone_no ** alpha
        
        prob_yes = attractiveness_yes * pheromone_yes
        prob_no = attractiveness_no * pheromone_no
        
        total_prob = prob_yes + prob_no
        
        if total_prob == 0:
            return False
        
        prob_yes /= total_prob
        prob_no /= total_prob
        
        import random
        rand_value = random.random()
        
        if rand_value < prob_yes:
            self.backpack.append(1)
            self.current_load += item.weight
            self.current_value += item.value
            return True
        else:
            self.backpack.append(0)
            return False