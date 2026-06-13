"""
Vergleich der ACO-Konfigurationen (Beste, Mittlere, Schlechteste)
Autor: Antigravity Coding Assistant
Beschreibung: Führt Simulationen des Ant Cycle Algorithmus für drei repräsentative
              Parameterkombinationen aus und plottet deren finale Pheromonspuren
              und Lernkurven im direkten Vergleich.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importiere Klassen aus dem Projekt
from item import Item
from ant import Ant

# ===================== STYLE & CONFIGURATION =====================

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except (OSError, ValueError):
    try:
        plt.style.use('seaborn-whitegrid')
    except (OSError, ValueError):
        plt.style.use('default')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

OPTIMAL_VALUE = 1011638  # Bekanntes Optimum des NMGE-Knapsack

# ===================== CONFIGURATIONS =====================

CONFIGS = {
    'Beste': {
        'alpha': 1.40,
        'beta': 0.65,
        'evaporation': 0.37,
        'group_size': 125,
        'label': r'Beste ($\alpha=1.4$, $\beta=0.65$, $\rho=0.37$, $G=125$)'
    },
    'Mittlere': {
        'alpha': 1.20,
        'beta': 2.00,
        'evaporation': 0.30,
        'group_size': 80,
        'label': r'Mittlere ($\alpha=1.2$, $\beta=2.0$, $\rho=0.30$, $G=80$)'
    },
    'Schlechteste': {
        'alpha': 1.00,
        'beta': 4.00,
        'evaporation': 0.20,
        'group_size': 40,
        'label': r'Schlechteste ($\alpha=1.0$, $\beta=4.0$, $\rho=0.20$, $G=40$)'
    }
}

# ===================== SIMULATION ENGINE =====================

def run_simulation(config, problem_data, iterations=200, seed=42):
    """
    Führt einen Einzellauf des Ant Cycle Algorithmus mit den gegebenen Parametern durch.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    alpha = config['alpha']
    beta = config['beta']
    evaporation_rate = config['evaporation']
    group_size = config['group_size']
    
    number_items = problem_data["number_items"]
    max_load = problem_data["max_load"]
    optimal_value = problem_data["optimal_solution"]["value"]
    
    # Items instanziieren
    items = []
    for data in problem_data["items"]:
        items.append(Item(data["id"], data["weight"], data["value"]))
        
    # Heuristik normieren (wie in main.py)
    max_eta_yes = max(item.attractiveness_yes for item in items)
    max_eta_no = max(item.attractiveness_no for item in items)
    for item in items:
        item.attractiveness_yes /= max_eta_yes
        item.attractiveness_no /= max_eta_no
        
    # Ameisen erzeugen
    ants = [Ant(max_load, number_items) for _ in range(group_size)]
    
    # Tracking
    best_fitness_curve = []
    avg_fitness_curve = []
    best_weight_curve = []
    best_val = 0
    best_backpack = None
    
    for iteration in range(iterations):
        # 1. Lösungen konstruieren
        for a in ants:
            starting_position = random.randint(0, number_items - 1)
            a.reset()
            for pos in range(number_items):
                current_item = items[(starting_position + pos) % number_items]
                a.decision(current_item, alpha, beta)
                
        # 2. Auswerten
        round_best_ant = max(ants, key=lambda x: x.current_value)
        avg_value = sum(a.current_value for a in ants) / group_size
        
        best_fitness_curve.append(round_best_ant.current_value)
        avg_fitness_curve.append(avg_value)
        best_weight_curve.append(round_best_ant.current_load)
        
        if round_best_ant.current_value > best_val:
            best_val = round_best_ant.current_value
            best_backpack = round_best_ant.backpack.copy()
        
        # 3. Pheromonupdate
        # Verdampfung
        for item in items:
            item.evaporate(evaporation_rate)
        # Ablage (Ant-Cycle: alle Ameisen)
        for a in ants:
            for item in items:
                decision = a.backpack[item.id]
                item.add_reward(decision, a.current_value, optimal_value)
                
    # Extrahiere finale Pheromonwerte
    pheromone_yes = np.array([item.pheromone_yes for item in items])
    pheromone_no = np.array([item.pheromone_no for item in items])
    
    return best_fitness_curve, avg_fitness_curve, best_weight_curve, best_backpack, pheromone_yes, pheromone_no

# ===================== MAIN EXECUTION =====================

def main():
    print("================================================================================")
    print(" Starte Simulationen für den Konvergenz- und Pheromonvergleich")
    print("================================================================================")
    
    # Problem laden
    with open("data/problem.json", "r") as f:
        problem_data = json.load(f)
        
    results = {}
    num_runs = 10  # 10 Wiederholungen pro Konfiguration zur Glättung der stochastischen Effekte
    iterations = 100
    
    for name, config in CONFIGS.items():
        print(f"\nSimuliere: {name}...")
        
        all_best = []
        all_avg = []
        all_weights = []
        all_yes = []
        all_no = []
        
        best_overall_val = 0
        best_overall_backpack = None
        
        for run in range(num_runs):
            # Nutze unterschiedliche Seeds für die Läufe
            seed = 100 + run * 37
            best_curve, avg_curve, weight_curve, run_best_backpack, p_yes, p_no = run_simulation(config, problem_data, iterations, seed)
            all_best.append(best_curve)
            all_avg.append(avg_curve)
            all_weights.append(weight_curve)
            all_yes.append(p_yes)
            all_no.append(p_no)
            print(f"  -> Lauf {run+1}/{num_runs} abgeschlossen (Endwert: {best_curve[-1]:,.0f})".replace(",", "."))
            
            # Finde die beste gefundene Einzellösung über alle Läufe und Iterationen
            run_max_val = max(best_curve)
            if run_max_val > best_overall_val:
                best_overall_val = run_max_val
                best_overall_backpack = run_best_backpack
            
        # Median/Mittelwert über die Läufe berechnen
        results[name] = {
            'best_curve': np.mean(all_best, axis=0),
            'avg_curve': np.mean(all_avg, axis=0),
            'weight_curve': np.mean(all_weights, axis=0),
            'best_backpack': best_overall_backpack,
            'p_yes': np.mean(all_yes, axis=0),
            'p_no': np.mean(all_no, axis=0)
        }
        
    print("\nSimulationen beendet. Erzeuge vergleichende Plots...")
    
    # Hilfsfunktion zur Berechnung von Durchschnittswerten für Bereiche
    def get_region_stats(start, end):
        region_items = problem_data["items"][start:end+1]
        avg_weight = np.mean([item["weight"] for item in region_items])
        avg_ratio = np.mean([item["value"] / item["weight"] for item in region_items])
        rel_weight_pct = (avg_weight / problem_data["max_load"]) * 100
        return rel_weight_pct, avg_ratio

    # ===================== PLOTTING =====================
    # Wir erstellen 4 separate Abbildungen (A, B, C, D) für die Präsentation
    
    names = list(CONFIGS.keys())
    
    # Farben für die drei Konfigurationen
    colors = {
        'Beste': '#2ca02c',       # Grün
        'Mittlere': '#ff7f0e',     # Orange
        'Schlechteste': '#d62728'  # Rot
    }
    
    # Gegenstandsanzahl ermitteln
    x_axis = range(len(results[names[0]]['p_yes']))
    number_items = len(x_axis)
    
    # Dynamische Berechnung der NMGE-Bereiche basierend auf der Gegenstandsanzahl
    if number_items == 600:
        # Medium Instanz (g=10, m=53, f=0.2)
        heavy_end = 264     # Ende Gruppe 5 (IDs 0-264)
        light_end = 476     # Ende Gruppe 9 (IDs 265-476)
        fine_end = 599      # Ende Gruppe 10 (IDs 477-599)
        
        h_pct, h_ratio = get_region_stats(0, heavy_end)
        l_pct, l_ratio = get_region_stats(heavy_end + 1, light_end)
        f_pct, f_ratio = get_region_stats(light_end + 1, fine_end)
        
        heavy_label = f'Schwere Gegenstände\nØ-Größe: {h_pct:.2f}%, Ø-Ratio: {h_ratio:.4f}'
        light_label = f'Leichte Gegenstände\nØ-Größe: {l_pct:.2f}%, Ø-Ratio: {l_ratio:.4f}'
        fine_label = f'Kleinstgegenstände\nØ-Größe: {f_pct:.2f}%, Ø-Ratio: {f_ratio:.4f}'
    elif number_items == 800:
        # Hard Instanz (g=14, m=55, f=0.1)
        heavy_end = 384     # Ende Gruppe 7 (IDs 0-384)
        light_end = 714     # Ende Gruppe 13 (IDs 385-714)
        fine_end = 799      # Ende Gruppe 14 (IDs 715-799)
        
        h_pct, h_ratio = get_region_stats(0, heavy_end)
        l_pct, l_ratio = get_region_stats(heavy_end + 1, light_end)
        f_pct, f_ratio = get_region_stats(light_end + 1, fine_end)
        
        heavy_label = f'Schwere Gegenstände\nØ-Größe: {h_pct:.2f}%, Ø-Ratio: {h_ratio:.4f}'
        light_label = f'Leichte Gegenstände\nØ-Größe: {l_pct:.2f}%, Ø-Ratio: {l_ratio:.4f}'
        fine_label = f'Kleinstgegenstände\nØ-Größe: {f_pct:.2f}%, Ø-Ratio: {f_ratio:.4f}'
    else:
        heavy_end = int(number_items * 0.44)
        light_end = int(number_items * 0.79)
        fine_end = number_items - 1
        
        h_pct, h_ratio = get_region_stats(0, heavy_end)
        l_pct, l_ratio = get_region_stats(heavy_end + 1, light_end)
        f_pct, f_ratio = get_region_stats(light_end + 1, fine_end)
        
        heavy_label = f'Schwere Gegenstände\nØ-Größe: {h_pct:.2f}%, Ø-Ratio: {h_ratio:.4f}'
        light_label = f'Leichte Gegenstände\nØ-Größe: {l_pct:.2f}%, Ø-Ratio: {l_ratio:.4f}'
        fine_label = f'Kleinstgegenstände\nØ-Größe: {f_pct:.2f}%, Ø-Ratio: {f_ratio:.4f}'

    os.makedirs("presentation", exist_ok=True)

    # --- PLOT A: Normierte Netto-Pheromonspur ---
    fig, ax_p = plt.subplots(figsize=(12, 5))
    
    # Markiere die Bereiche farblich im Hintergrund
    ax_p.axvspan(0, heavy_end, color='#d62728', alpha=0.07, label=heavy_label)
    ax_p.axvspan(heavy_end + 1, light_end, color='#2ca02c', alpha=0.07, label=light_label)
    ax_p.axvspan(light_end + 1, fine_end, color='#1f77b4', alpha=0.07, label=fine_label)

    for name in names:
        data = results[name]
        # Netto-Pheromon berechnen und pro Ameise normieren: (JA - NEIN) / G
        net_pheromone = (data['p_yes'] - data['p_no']) / CONFIGS[name]['group_size']
        
        ax_p.plot(
            x_axis, net_pheromone, 
            color=colors[name], linewidth=1.5, alpha=0.75,
            label=CONFIGS[name]['label']
        )
        
    ax_p.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.8)
    
    # Ideallösung (Optimum) laden
    optimal_items = problem_data["optimal_solution"]["items"]
    
    ax_p.set_title("Normierte Pheromonspur", fontweight='bold', fontsize=12.5)
    ax_p.set_xlabel("Gegenstands-ID", fontweight='bold')
    ax_p.set_ylabel("Netto-Pheromon (Ja+/Nein-)", fontweight='bold')
    ax_p.grid(True, linestyle=':', alpha=0.5)
    ax_p.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    plt.savefig("presentation/07a_vergleich_pheromonschnitt.png", dpi=300, bbox_inches='tight')
    plt.savefig("presentation/07a_vergleich_pheromonschnitt.pdf", bbox_inches='tight')
    plt.close()

    # --- PLOT B: Konvergenzverlauf des Kofferwerts (Nutzen) ---
    fig, ax_l = plt.subplots(figsize=(12, 5))
    iter_range = range(1, iterations + 1)
    
    for name in names:
        data = results[name]
        # Bester Wert (durchgezogene Linie)
        ax_l.plot(
            iter_range, data['best_curve'], 
            color=colors[name], linewidth=2.5, linestyle='-',
            label=CONFIGS[name]['label']
        )
        
    # Referenzlinie für das bekannte Optimum
    ax_l.axhline(OPTIMAL_VALUE, color='black', linestyle=':', linewidth=1.8, label=f'Optimum ({OPTIMAL_VALUE:,.0f})'.replace(",", "."))
    ax_l.text(iterations + 1, OPTIMAL_VALUE, 'Optimum', color='black', va='center', fontweight='bold', fontsize=9.5)
    
    ax_l.set_title("Konvergenzverlauf des Kofferwerts (Nutzen) über 100 Iterationen", fontweight='bold', fontsize=12.5)
    ax_l.set_xlabel("Iteration", fontweight='bold')
    ax_l.set_ylabel("Rucksackwert (Nutzen)", fontweight='bold')
    
    # Y-Limits auf Wunsch des Users: 1.000.000 bis 1.015.000 (um die Stagnation besser zu verdeutlichen)
    ax_l.set_ylim(1000000, 1015000)
    
    ax_l.grid(True, linestyle=':', alpha=0.5)
    ax_l.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    plt.savefig("presentation/07b_vergleich_nutzwert.png", dpi=300, bbox_inches='tight')
    plt.savefig("presentation/07b_vergleich_nutzwert.pdf", bbox_inches='tight')
    plt.close()

    # --- PLOT C: Rucksackgewicht im direkten Vergleich ---
    fig, ax_w = plt.subplots(figsize=(12, 5))
    
    for name in names:
        data = results[name]
        ax_w.plot(
            iter_range, data['weight_curve'], 
            color=colors[name], linewidth=2.5, linestyle='-',
            label=CONFIGS[name]['label']
        )
        
    # Kapazitätsgrenze einzeichnen
    ax_w.axhline(problem_data["max_load"], color='black', linestyle=':', linewidth=1.8, label=f'Kapazitätsgrenze ({problem_data["max_load"]:,.0f})'.replace(",", "."))
    ax_w.text(iterations + 1, problem_data["max_load"], 'Kapazität', color='black', va='center', fontweight='bold', fontsize=9.5)
    
    ax_w.set_title("Konvergenzverlauf des Rucksackgewichts (Füllstand) über 100 Iterationen", fontweight='bold', fontsize=12.5)
    ax_w.set_xlabel("Iteration", fontweight='bold')
    ax_w.set_ylabel("Rucksackgewicht", fontweight='bold')
    
    # Y-Limits für das Gewicht: 997.000 (0.997e6) bis 1.002.000 (um Kapazitätsannäherung zu zeigen)
    ax_w.set_ylim(997000, 1002000)
    
    ax_w.grid(True, linestyle=':', alpha=0.5)
    ax_w.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    plt.savefig("presentation/07c_vergleich_gewicht.png", dpi=300, bbox_inches='tight')
    plt.savefig("presentation/07c_vergleich_gewicht.pdf", bbox_inches='tight')
    plt.close()
    
    # --- PLOT D: Gegenstandsauswahl im Vergleich ---
    fig, ax_i = plt.subplots(figsize=(12, 5))
    
    # Ideallösung (Optimum) einzeichnen
    ax_i.scatter(optimal_items, [3] * len(optimal_items), color='purple', marker='|', s=80, alpha=0.7, label='Optimal-Auswahl')
    
    # Gegenstände der besten gefundenen Einzellösungen einzeichnen
    # Beste
    bp_beste = results['Beste']['best_backpack']
    beste_selected = [idx for idx in range(number_items) if bp_beste[idx] == 1]
    ax_i.scatter(beste_selected, [2] * len(beste_selected), color='#2ca02c', marker='|', s=80, alpha=0.7, label=CONFIGS['Beste']['label'])
    
    # Mittlere
    bp_mittlere = results['Mittlere']['best_backpack']
    mittlere_selected = [idx for idx in range(number_items) if bp_mittlere[idx] == 1]
    ax_i.scatter(mittlere_selected, [1] * len(mittlere_selected), color='#ff7f0e', marker='|', s=80, alpha=0.7, label=CONFIGS['Mittlere']['label'])
    
    # Schlechteste
    bp_schlecht = results['Schlechteste']['best_backpack']
    schlecht_selected = [idx for idx in range(number_items) if bp_schlecht[idx] == 1]
    ax_i.scatter(schlecht_selected, [0] * len(schlecht_selected), color='#d62728', marker='|', s=80, alpha=0.7, label=CONFIGS['Schlechteste']['label'])
    
    # Markiere die Bereiche farblich im Hintergrund
    ax_i.axvspan(0, heavy_end, color='#d62728', alpha=0.07, label=heavy_label)
    ax_i.axvspan(heavy_end + 1, light_end, color='#2ca02c', alpha=0.07, label=light_label)
    ax_i.axvspan(light_end + 1, fine_end, color='#1f77b4', alpha=0.07, label=fine_label)
    
    ax_i.set_title("Gegenüberstellung der Gegenstandsauswahl (Beste Einzellösungen)", fontweight='bold', fontsize=12.5)
    ax_i.set_xlabel(f"Gegenstands-ID (0 - {number_items - 1})", fontweight='bold')
    ax_i.set_yticks([0, 1, 2, 3])
    ax_i.set_yticklabels(['Schlechteste', 'Mittlere', 'Beste', 'Optimum (Ideal)'], fontweight='bold')
    ax_i.set_ylim(-0.5, 3.5)
    ax_i.grid(True, linestyle=':', alpha=0.5, axis='x')
    ax_i.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    plt.savefig("presentation/07d_vergleich_itemauswahl.png", dpi=300, bbox_inches='tight')
    plt.savefig("presentation/07d_vergleich_itemauswahl.pdf", bbox_inches='tight')
    plt.close()
    
    print("\n================================================================================")
    print(" Fertig: Grafiken 7a-d wurden erfolgreich unter 'presentation/' gespeichert!")
    print("================================================================================")

if __name__ == '__main__':
    main()
