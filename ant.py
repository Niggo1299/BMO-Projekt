"""Ameise: Konstruiert Lösungen per probabilistischer Item-Auswahl (Ant-Cycle)."""

import random


class Ant:
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

    def construct_solution(self, items, alpha, beta):
        """Iterative Item-Auswahl bis Rucksack voll. Nutzt vorberechnete item.score."""
        self.reset()

        remaining_idx = list(range(len(items)))
        pos_of = {idx: idx for idx in remaining_idx}

        while remaining_idx:
            feasible_idx = [i for i in remaining_idx
                            if self.current_load + items[i].weight <= self.max_weight]
            if not feasible_idx:
                break

            weights = [items[i].score for i in feasible_idx]
            total = sum(weights)

            if total == 0:
                chosen_item_idx = feasible_idx[random.randrange(len(feasible_idx))]
            else:
                _, chosen_item_idx = self._roulette_select(feasible_idx, weights, total)

            item = items[chosen_item_idx]
            self.backpack[item.id] = 1
            self.current_load += item.weight
            self.current_value += item.value

            # O(1) Swap-and-Pop
            pos = pos_of[chosen_item_idx]
            last = remaining_idx[-1]
            remaining_idx[pos] = last
            pos_of[last] = pos
            remaining_idx.pop()
            del pos_of[chosen_item_idx]

    def _roulette_select(self, feasible_idx, weights, total):
        threshold = random.random() * total
        cumulative = 0.0
        for list_pos, (item_idx, w) in enumerate(zip(feasible_idx, weights)):
            cumulative += w
            if cumulative >= threshold:
                return list_pos, item_idx
        return len(feasible_idx) - 1, feasible_idx[-1]