"""
Klasse Ant: Repräsentiert eine einzelne Ameise im Ameisenalgorithmus.

Jede Ameise konstruiert eine Lösung, indem sie für jedes Item
probabilistisch entscheidet, ob es eingepackt wird (1) oder nicht (0).

Verwendbar für AC und EAS (kein Unterschied in der Konstruktionsphase).
"""

import random


class Ant:
    def __init__(self, max_load, num_items):
        """
        Initialisiert eine Ameise.

        Args:
            max_load:  Maximale Kapazität des Rucksacks.
            num_items: Gesamtanzahl der verfügbaren Items.
        """
        self.max_weight = max_load
        self.num_items = num_items
        self.backpack = [0] * num_items     # Entscheidungsvektor (0 = nein, 1 = ja)
        self.current_load = 0               # Aktuelles Gesamtgewicht im Rucksack
        self.current_value = 0              # Aktueller Gesamtwert im Rucksack

    def reset(self):
        """Setzt die Ameise für eine neue Iteration zurück."""
        self.backpack = [0] * self.num_items
        self.current_load = 0
        self.current_value = 0

    def _fits_in_backpack(self, item):
        """
        Prüft, ob das Item noch in den Rucksack passt.

        Returns:
            True wenn Item NICHT mehr passt (Kapazität würde überschritten).
        """
        return self.current_load + item.weight > self.max_weight

    def decision(self, item, alpha, beta):
        """
        Probabilistische Entscheidung ob ein Item eingepackt wird.

        Berechnung gemäß Übergangsregel (Folie 4):
            p_ja = [τ_ja]^α · [η_ja]^β / ([τ_ja]^α · [η_ja]^β + [τ_nein]^α · [η_nein]^β)

        Auswahl erfolgt per Rouletterad-Selektion.

        Args:
            item:  Das zu bewertende Item-Objekt.
            alpha: Gewichtungsexponent für Pheromonspuren.
            beta:  Gewichtungsexponent für heuristische Information.

        Returns:
            True wenn Item eingepackt wurde, False sonst.
        """
        # Kapazitätsprüfung: Falls Item nicht passt → zwingend NEIN
        if self._fits_in_backpack(item):
            self.backpack[item.id] = 0
            return False

        # Berechnung der gewichteten Attraktivitäten (Zähler der Formel)
        a_yes = (item.pheromone_yes ** alpha) * (item.attractiveness_yes ** beta)
        a_no = (item.pheromone_no ** alpha) * (item.attractiveness_no ** beta)

        # Gesamtsumme (Nenner) für Wahrscheinlichkeitsberechnung
        total = a_yes + a_no

        # Schutz vor Division durch Null
        if total == 0:
            self.backpack[item.id] = 0
            return False

        # Wahrscheinlichkeit für JA berechnen
        prob_yes = a_yes / total

        # Stochastische Entscheidung (Rouletterad)
        if random.random() < prob_yes:
            self.backpack[item.id] = 1
            self.current_load += item.weight
            self.current_value += item.value
            return True
        else:
            self.backpack[item.id] = 0
            return False