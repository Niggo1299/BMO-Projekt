import random
import json

number_items = 50
max_load = 20

# create items
items_data = []
for i in range(number_items):
    weight = random.randint(1, 10)
    value = random.randint(1, 10)
    items_data.append({"id": i, "weight": weight, "value": value})

# Speichere die generierten Daten in eine JSON-Datei
with open("problem.json", "w") as f:
    json.dump({
        "number_items": number_items,
        "max_load": max_load,
        "items": items_data
    }, f, indent=4)

print("Problem erfolgreich unter 'problem.json' gespeichert!")