"""
Generiert ein zufälliges 0/1-Rucksackproblem und speichert es als JSON.
"""

import random
import json


def generate_problem(number_items=50, max_load=20, filename="problem.json"):
    """
    Erzeugt ein Rucksackproblem mit zufälligen Items.

    Args:
        number_items: Anzahl der verfügbaren Gegenstände.
        max_load:     Maximale Tragfähigkeit des Rucksacks.
        filename:     Dateiname für die JSON-Ausgabe.
    """
    items_data = []
    for i in range(number_items):
        weight = random.randint(1, 10)
        value = random.randint(1, 10)
        items_data.append({"id": i, "weight": weight, "value": value})

    problem = {
        "number_items": number_items,
        "max_load": max_load,
        "items": items_data
    }

    with open(filename, "w") as f:
        json.dump(problem, f, indent=4)

    print(f"Problem erfolgreich unter '{filename}' gespeichert!")
    print(f"  → {number_items} Items, max. Gewicht: {max_load}")


if __name__ == "__main__":
    generate_problem()