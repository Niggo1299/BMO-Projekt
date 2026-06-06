"""
Klasse Item: Repräsentiert einen Gegenstand im Rucksackproblem.

Jedes Item speichert seine Pheromonspuren (τ) und heuristische
Informationen (η) für die Entscheidungen JA und NEIN.

Verwendbar für AC und EAS:
- AC:  add_reward() ohne elite_factor (Standard 1.0)
- EAS: add_reward() mit elite_factor für den Elite-Bonus
"""


class Item:
    def __init__(self, id, weight, value):
        """
        Initialisiert ein Item mit Pheromon- und Heuristikwerten.

        Args:
            id:     Eindeutige ID des Items.
            weight: Gewicht des Items.
            value:  Wert/Nutzen des Items.
        """
        self.id = id
        self.weight = weight
        self.value = value

        # Heuristische Information η (problemspezifisch angepasst)
        # η_ja: Hoher Wert pro Gewicht → Item ist attraktiv zum Einpacken
        self.attractiveness_yes = value / weight
        # η_nein: Hohes Gewicht pro Wert → Item ist attraktiv zum Weglassen
        self.attractiveness_no = weight / value

        # Pheromonwerte τ (gleicher Startwert für faire Initialisierung)
        # Keine min/max-Grenzen im AC/EAS (anders als bei MMAS)
        self.pheromone_yes = 1.0
        self.pheromone_no = 1.0

    def evaporate(self, rho):
        """
        Verdampfung der Pheromonspuren gemäß Folie 7/9.
        Formel: τ_neu = (1 - ρ) · τ_alt

        Args:
            rho: Verdunstungsfaktor ρ (0 ≤ ρ ≤ 1).
        """
        self.pheromone_yes *= (1 - rho)
        self.pheromone_no *= (1 - rho)

    def add_reward(self, decision, solution_value, optimal_value, elite_factor=1.0):
        """
        Pheromonablage auf der gewählten Entscheidungskante.

        Formel (normiert):
            Δτ = (solution_value / optimal_value) * elite_factor

        Nutzung:
            - AC:  elite_factor = 1.0 (Standard, kein Bonus)
            - EAS: elite_factor = e (0 ≤ e ≤ 1) für den Elite-Bonus

        Args:
            decision:       1 (eingepackt) oder 0 (nicht eingepackt).
            solution_value: Gesamtwert der Lösung dieser Ameise.
            optimal_value:  Optimaler Nutzwert der Instanz (für Normierung).
            elite_factor:   Gewichtungsfaktor e für Elite-Bonus (Standard: 1.0).
        """
        delta_tau = (solution_value / optimal_value) * elite_factor

        if decision == 1:
            self.pheromone_yes += delta_tau
        else:
            self.pheromone_no += delta_tau