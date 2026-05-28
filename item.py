"""
Klasse Item: Repräsentiert einen Gegenstand im Rucksackproblem.

Jedes Item speichert seine Pheromonspuren (τ) und heuristische
Informationen (η) für die Entscheidungen JA und NEIN.

Im reinen Ant-Cycle gibt es keine Pheromon-Grenzen (im Gegensatz zu MMAS).
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
        # Keine min/max-Grenzen im reinen Ant-Cycle (anders als MMAS)
        self.pheromone_yes = 1.0
        self.pheromone_no = 1.0

    def evaporate(self, rho):
        """
        Verdampfung der Pheromonspuren gemäß Folie 7.
        Formel: τ_neu = (1 - ρ) · τ_alt

        Args:
            rho: Verdunstungsfaktor ρ (0 ≤ ρ ≤ 1).
        """
        self.pheromone_yes *= (1 - rho)
        self.pheromone_no *= (1 - rho)

    def add_reward(self, decision, solution_value, max_value):
        """
        Pheromonablage auf der gewählten Entscheidungskante.

        Gemäß Ant-Cycle (Folie 7):
            Δτ_k = solution_value / max_value
            (proportional zur Lösungsqualität, normiert)

        Im TSP wäre dies 1/C_k (Inverse der Tourlänge).
        Beim Rucksackproblem (Maximierung) ist die Qualität
        direkt proportional zum erzielten Wert.

        Args:
            decision:       1 (eingepackt) oder 0 (nicht eingepackt).
            solution_value: Gesamtwert der Lösung dieser Ameise.
            max_value:      Maximaler möglicher Wert (zur Normierung).
        """
        # Δτ = Lösungsqualität / Normierungswert
        delta_tau = solution_value / max_value

        if decision == 1:
            self.pheromone_yes += delta_tau
        else:
            self.pheromone_no += delta_tau