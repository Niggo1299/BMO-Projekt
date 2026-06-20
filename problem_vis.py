import math
import random
import matplotlib.pyplot as plt
import numpy as np

# 1. NMGE-Konfiguration für das "medium" Problem (exakt aus problem.py)
n = 600
c = 1_000_000
g = 10
f = 0.2
epsilon = 0.001
s = 200
b = 2
seed = 2

random.seed(seed)

last_group_size = math.floor(n * f)
m = math.floor((n - last_group_size) / (g - 1))
last_group_size = n - (g - 1) * m

gruppen_daten = {i: {"weights": [], "attractiveness": []} for i in range(1, 11)}

# Gruppen 1 bis 9 generieren
for group_i in range(1, g):
    base_value = math.floor((1.0 / (b ** group_i) + epsilon) * c)
    for _ in range(m):
        r1, r2 = random.randint(1, s), random.randint(1, s)
        w = base_value + r2
        v = base_value + r1
        gruppen_daten[group_i]["weights"].append(w)
        gruppen_daten[group_i]["attractiveness"].append(v / w)

# Gruppe 10 (Zufallsgruppe) generieren
for _ in range(last_group_size):
    w = random.randint(1, s)
    v = random.randint(1, s)
    gruppen_daten[10]["weights"].append(w)
    gruppen_daten[10]["attractiveness"].append(v / w)

# Werte für die Säulen berechnen: Durchschnittliche Größe & MAXIMALE Attraktivität
avg_groesse = [np.mean(gruppen_daten[i]["weights"]) for i in range(1, 11)]
max_attraktivitaet = [np.max(gruppen_daten[i]["attractiveness"]) for i in range(1, 11)]

# 2. Zwei Diagramme nebeneinander erstellen
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))

width = 0.35
x_9 = np.arange(1, 10)

# ----------------------------------------------------
# DIAGRAMM 1 (LINKS): ZOOM AUF GRUPPE 1-9 (TREPPE)
# ----------------------------------------------------
ax_left_twin = ax_left.twinx()

# Subplot-Titel wieder aktiv
ax_left.set_title('Gezoomt auf G1 bis G9', fontsize=12, fontweight='bold', pad=10)

# Balken zeichnen
balken_g1 = ax_left.bar(x_9 - width/2, avg_groesse[:9], width, label='Größe (Gewicht)', color='#1f77b4', alpha=0.5, edgecolor='navy')
balken_a1 = ax_left_twin.bar(x_9 + width/2, max_attraktivitaet[:9], width, label='Max. Attraktivität', color='#2ca02c', edgecolor='darkgreen')

# WICHTIG: Der Zoom auf die Attraktivitäts-Achse (0.99 bis 1.08)
ax_left_twin.set_ylim(0.99, 1.08) 

# Achsen-Beschriftungen und Styling
ax_left.set_xlabel('Problem-Gruppen', fontsize=11)
ax_left.set_ylabel('Durchschnittliche Größe', color='#1f77b4')
ax_left_twin.set_ylabel('Maximale Attraktivität (Zoom-Skala)', color='#2ca02c')

ax_left.tick_params(axis='y', labelcolor='#1f77b4')
ax_left_twin.tick_params(axis='y', labelcolor='#2ca02c')

ax_left.set_xticks(x_9)
ax_left.set_xticklabels([f'G {i}' for i in range(1, 10)])
ax_left.grid(axis='y', linestyle='--', alpha=0.3)

# Legende für das linke Diagramm
lines1, labels1 = ax_left.get_legend_handles_labels()
lines2, labels2 = ax_left_twin.get_legend_handles_labels()
ax_left.legend(lines1 + lines2, labels1 + labels2, loc='upper right')


# ----------------------------------------------------
# DIAGRAMM 2 (RECHTS): GESAMTBILD INKL. GRUPPE 10
# ----------------------------------------------------
x_10 = np.arange(1, 11)
ax_right_twin = ax_right.twinx()

# Subplot-Titel wieder aktiv
ax_right.set_title('Nicht gezoomt', fontsize=12, fontweight='bold', pad=10)

# Balken zeichnen
balken_g2 = ax_right.bar(x_10 - width/2, avg_groesse, width, color='#1f77b4', alpha=0.5, edgecolor='navy')
balken_a2 = ax_right_twin.bar(x_10 + width/2, max_attraktivitaet, width, color='#2ca02c', edgecolor='darkgreen')

# Achsen-Beschriftungen und Styling (Volle Skala)
ax_right.set_xlabel('Problem-Gruppen', fontsize=11)
ax_right.set_ylabel('Durchschnittliche Größe', color='#1f77b4')
ax_right_twin.set_ylabel('Maximale Attraktivität (Volle Skala)', color='#2ca02c')

ax_right.tick_params(axis='y', labelcolor='#1f77b4')
ax_right_twin.tick_params(axis='y', labelcolor='#2ca02c')

ax_right.set_xticks(x_10)
ax_right.set_xticklabels([f'G {i}' for i in range(1, 11)])
ax_right.grid(axis='y', linestyle='--', alpha=0.3)

# Legende für das rechte Diagramm
ax_right.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

plt.tight_layout()
import os
os.makedirs("data", exist_ok=True)
plt.savefig("data/problem_vis.png", dpi=300, bbox_inches='tight')
print("Grafik gespeichert unter 'data/problem_vis.png'")
plt.show()