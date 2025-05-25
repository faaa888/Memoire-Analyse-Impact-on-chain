#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages

# Styles et configuration de matplotlib
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Définition des chemins
BASE_PATH = '.'  # Chemin relatif au répertoire courant
METRICS_DIR = os.path.join(BASE_PATH, 'FinancialResults/Outputs')
EXITS_FILE = os.path.join(BASE_PATH, 'Onchain_exits_dynamic.csv')
DECORR_FILE = os.path.join(METRICS_DIR, 'AnalyseDecorelation/impact_decorrelé.csv')
OUTPUT_DIR = os.path.join(BASE_PATH, 'VisualsOutput')

# Création du répertoire de sortie s'il n'existe pas
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Définition des fenêtres temporelles en mois
FENETRES = [1, 6, 12]

print("GÉNÉRATION DES VISUALISATIONS D'IMPACT DES SORTIES DE FONDS")
print("=========================================================")

# Chargement des données décorrélées
print("\nChargement des données...")
if os.path.exists(DECORR_FILE):
    decorr_df = pd.read_csv(DECORR_FILE)
    print(f"Données d'impact décorrélé chargées: {len(decorr_df)} entrées")
else:
    raise FileNotFoundError(f"Fichier d'impact décorrélé non trouvé: {DECORR_FILE}")

# Conversion des dates en datetime
decorr_df['date_sortie'] = pd.to_datetime(decorr_df['date_sortie'])
decorr_df['date_apres'] = pd.to_datetime(decorr_df['date_apres'])

# Filtrer les données pour les fenêtres temporelles d'intérêt
decorr_df = decorr_df[decorr_df['fenetre_mois'].isin(FENETRES)]

# Création du PDF pour stocker les visualisations
pdf_path = os.path.join(OUTPUT_DIR, 'impact_visuals.pdf')
pdf = PdfPages(pdf_path)

##########################################
# 1. VISUALISATION: RÉPARTITION DES DIRECTIONS D'IMPACT
##########################################
print("\n1. Génération de la répartition des directions d'impact...")

# Calculer la répartition des directions avant et après décorrélation
direction_counts_before = decorr_df.groupby(['fenetre_mois', 'direction_brute']).size().unstack(fill_value=0)
direction_counts_after = decorr_df.groupby(['fenetre_mois', 'direction_decorrelée']).size().unstack(fill_value=0)

# Calculer les pourcentages
direction_pct_before = direction_counts_before.div(direction_counts_before.sum(axis=1), axis=0) * 100
direction_pct_after = direction_counts_after.div(direction_counts_after.sum(axis=1), axis=0) * 100

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Avant décorrélation (subplot gauche)
bar_width = 0.25
index = np.arange(len(FENETRES))

# Vérifier les colonnes disponibles
pos_col = 'Positive' if 'Positive' in direction_counts_before.columns else None
neg_col = 'Négative' if 'Négative' in direction_counts_before.columns else None
stable_col = 'Stable' if 'Stable' in direction_counts_before.columns else None

# Définition des couleurs
colors = {'Positive': '#4CAF50', 'Stable': '#9E9E9E', 'Négative': '#F44336'}

# Tracer les barres empilées pour les données avant décorrélation
bottom = np.zeros(len(FENETRES))
for i, fenetre in enumerate(FENETRES):
    if fenetre in direction_counts_before.index:
        # Positif
        if pos_col:
            axes[0].bar(i, direction_counts_before.loc[fenetre, pos_col], bar_width, 
                       bottom=bottom[i], color=colors['Positive'], 
                       label='Positif' if i == 0 else "")
            bottom[i] += direction_counts_before.loc[fenetre, pos_col]
        # Stable
        if stable_col:
            axes[0].bar(i, direction_counts_before.loc[fenetre, stable_col], bar_width, 
                       bottom=bottom[i], color=colors['Stable'], 
                       label='Stable' if i == 0 else "")
            bottom[i] += direction_counts_before.loc[fenetre, stable_col]
        # Négatif
        if neg_col:
            axes[0].bar(i, direction_counts_before.loc[fenetre, neg_col], bar_width, 
                       bottom=bottom[i], color=colors['Négative'], 
                       label='Négatif' if i == 0 else "")

axes[0].set_title('Répartition des impacts BRUTS', fontsize=16)
axes[0].set_xlabel('Mois après sortie', fontsize=14)
axes[0].set_ylabel('Nombre de métriques', fontsize=14)
axes[0].set_xticks(index)
axes[0].set_xticklabels(FENETRES)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Après décorrélation (subplot droit)
bottom = np.zeros(len(FENETRES))
for i, fenetre in enumerate(FENETRES):
    if fenetre in direction_counts_after.index:
        # Positif
        if pos_col:
            axes[1].bar(i, direction_counts_after.loc[fenetre, pos_col], bar_width, 
                       bottom=bottom[i], color=colors['Positive'], 
                       label='Positif' if i == 0 else "")
            bottom[i] += direction_counts_after.loc[fenetre, pos_col]
        # Stable
        if stable_col:
            axes[1].bar(i, direction_counts_after.loc[fenetre, stable_col], bar_width, 
                       bottom=bottom[i], color=colors['Stable'], 
                       label='Stable' if i == 0 else "")
            bottom[i] += direction_counts_after.loc[fenetre, stable_col]
        # Négatif
        if neg_col:
            axes[1].bar(i, direction_counts_after.loc[fenetre, neg_col], bar_width, 
                       bottom=bottom[i], color=colors['Négative'], 
                       label='Négatif' if i == 0 else "")

axes[1].set_title('Répartition des impacts DÉCORRÉLÉS', fontsize=16)
axes[1].set_xlabel('Mois après sortie', fontsize=14)
axes[1].set_ylabel('Nombre de métriques', fontsize=14)
axes[1].set_xticks(index)
axes[1].set_xticklabels(FENETRES)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()

# Sauvegarder en PNG
png_path = os.path.join(OUTPUT_DIR, 'tableau2_directions_impact.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"Visualisation sauvegardée en PNG: {png_path}")

# Ajouter au PDF
pdf.savefig(fig)
plt.close()

##########################################
# 2. VISUALISATION: SMALL MULTIPLES PAR MÉTRIQUE × HORIZON
##########################################
print("\n2. Génération des 'small multiples' par métrique et horizon...")

# Chargement des données brutes des sorties
exits_file = EXITS_FILE
if not os.path.exists(exits_file):
    raise FileNotFoundError(f"Fichier des sorties non trouvé: {exits_file}")

exits_df = pd.read_csv(exits_file)
print(f"Données de sorties chargées: {len(exits_df)} entrées")

# Conversion des dates et renommage des colonnes pour correspondre au code
if 'date_de_sortie' in exits_df.columns:
    exits_df.rename(columns={
        'fond': 'fond',
        'projet_sorti': 'project',
        'date_de_sortie': 'date'
    }, inplace=True)

# Conversion des dates
exits_df['date'] = pd.to_datetime(exits_df['date'])

# Préparation des horizons temporels en mois
horizons = {
    '1m': 1,  # 1 mois
    '6m': 6,  # 6 mois
    '1y': 12  # 1 an (12 mois)
}

# Récupération de toutes les métriques uniques
all_metrics = decorr_df['metrique'].unique()
print(f"Métriques disponibles: {len(all_metrics)}")

# Sélection des métriques les plus importantes
important_metrics = [
    'Earnings',  # Données financières
    'Revenue',
    'Fully diluted market cap',
    'Price',
    
    'Active users (monthly)',  # Métriques d'utilisateurs
    'Active users (daily)',
    'Active users (weekly)',
    
    'Token turnover (fully diluted)',  # Métriques de token
    'Token holders',
    'P/S ratio (fully diluted)',
    
    'Core developers',  # Métriques de développement
    
    'Fees',  # Métriques d'activité
    'Trading volume',
    'Net deposits'
]

# Filtrer pour ne garder que les métriques qui existent dans nos données
significant_metrics = [m for m in important_metrics if m in all_metrics]

# Si certaines métriques importantes ne sont pas dans nos données, ajouter d'autres métriques
if len(significant_metrics) < len(important_metrics):
    missing_count = len(important_metrics) - len(significant_metrics)
    print(f"Attention: {missing_count} métriques importantes manquantes")
    
    # Ajouter d'autres métriques disponibles
    other_metrics = [m for m in all_metrics if m not in significant_metrics]
    significant_metrics.extend(other_metrics[:missing_count])

print(f"Analyse de {len(significant_metrics)} métriques")
print(f"Métriques sélectionnées: {significant_metrics}")

# Préparation des données pour la visualisation
print("Préparation des données pour la visualisation...")

# Structure pour stocker les séries temporelles
series_data = []

# Pour chaque métrique significative
for metric in significant_metrics:
    metric_data = decorr_df[decorr_df['metrique'] == metric]
    
    # Pour chaque projet
    for project in metric_data['projet'].unique():
        project_data = metric_data[metric_data['projet'] == project]
        
        # Pour chaque date de sortie
        for exit_date in project_data['date_sortie'].unique():
            # Pour chaque fenêtre temporelle
            for window_name, months in horizons.items():
                # Filtrer les données pour cette fenêtre temporelle
                window_data = project_data[project_data['fenetre_mois'] == months]
                window_data = window_data[window_data['date_sortie'] == exit_date]
                
                if not window_data.empty:
                    # Calculer le jour depuis la sortie (0 pour le jour de sortie)
                    for _, row in window_data.iterrows():
                        days_since_exit = (row['date_apres'] - row['date_sortie']).days
                        
                        # Calculer la variation normalisée (0% le jour de la sortie)
                        if pd.notna(row['impact_decorrelé']):  # Utiliser uniquement l'impact décorrélé
                            series_data.append({
                                'project': project,
                                'metric': metric,
                                'window': window_name,
                                'day_since_exit': days_since_exit,
                                'pct_norm_decorr': row['impact_decorrelé']  # Variation décorrélée
                            })

# Ajouter un point à jour 0 avec valeur 0 pour chaque série
normalized_series_data = []

# Obtenir toutes les combinaisons uniques de project, metric, window
unique_series = set([(d['project'], d['metric'], d['window']) for d in series_data])

# Pour chaque série
for project, metric, window in unique_series:
    # Ajouter le point de départ (jour 0, variation 0%)
    normalized_series_data.append({
        'project': project,
        'metric': metric,
        'window': window,
        'day_since_exit': 0,  # Jour 0
        'pct_norm_decorr': 0  # 0% de variation
    })
    
    # Ajouter tous les autres points de cette série
    for point in [p for p in series_data if p['project'] == project and p['metric'] == metric and p['window'] == window]:
        normalized_series_data.append(point)

# Convertir en DataFrame
if normalized_series_data:
    series_df = pd.DataFrame(normalized_series_data)
    
    # Calculer les statistiques par métrique, fenêtre et jour
    stats_df = series_df.groupby(['metric', 'window', 'day_since_exit']).agg(
        mean_pct_decorr=('pct_norm_decorr', 'mean'),
        q25=('pct_norm_decorr', lambda x: x.quantile(0.25)),
        q75=('pct_norm_decorr', lambda x: x.quantile(0.75)),
        count=('pct_norm_decorr', 'count')
    ).reset_index()
    
    # S'assurer que chaque courbe a une valeur à jour=0 et que cette valeur est 0
    for metric in significant_metrics:
        for window in horizons.keys():
            # Vérifier si le jour 0 existe pour cette combinaison
            day0_exists = ((stats_df['metric'] == metric) & 
                          (stats_df['window'] == window) & 
                          (stats_df['day_since_exit'] == 0)).any()
            
            if not day0_exists:
                # Ajouter un point à jour=0 avec valeur=0
                new_row = {
                    'metric': metric,
                    'window': window,
                    'day_since_exit': 0,
                    'mean_pct_decorr': 0,
                    'q25': 0,
                    'q75': 0,
                    'count': 1  # Valeur arbitraire pour count
                }
                stats_df = pd.concat([stats_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Génération de la visualisation "small multiples"
    print("Génération de la visualisation 'small multiples'...")
    
    # Déterminer le nombre de lignes et colonnes
    n_metrics = len(significant_metrics)
    n_windows = len(horizons)
    
    # Créer la figure avec un style adéquat
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(n_metrics, n_windows, figsize=(15, 3*n_metrics))
    
    # Pour chaque métrique et chaque fenêtre temporelle
    for i, metric in enumerate(significant_metrics):
        for j, (window_name, _) in enumerate(horizons.items()):
            # Sélectionner les données pour cette métrique et cette fenêtre
            subset = stats_df[(stats_df['metric'] == metric) & (stats_df['window'] == window_name)]
            
            # Obtenir l'axe correspondant
            if n_metrics == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            
            if not subset.empty:
                # Trier par jour
                subset = subset.sort_values('day_since_exit')
                
                # Tracer la courbe moyenne (décorrélée uniquement)
                ax.plot(subset['day_since_exit'], subset['mean_pct_decorr'], 'b-', linewidth=2)
                
                # Ajouter le bandeau de dispersion
                ax.fill_between(subset['day_since_exit'], subset['q25'], subset['q75'], color='blue', alpha=0.2)
                
                # Ajouter une ligne horizontale à y=0
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # S'assurer que l'axe x commence à 0
                xlim = ax.get_xlim()
                ax.set_xlim([0, xlim[1]])
                
                # Configurer les titres et labels
                if window_name == '1m':
                    window_label = "1 mois post-exit"
                elif window_name == '6m':
                    window_label = "6 mois post-exit"
                else:
                    window_label = "1 an post-exit"
                
                ax.set_title(f"{metric} — {window_label}", fontsize=10)
                
                # Ajouter le nombre d'observations
                obs_count = subset['count'].mean()
                ax.annotate(f"n={int(obs_count)}", xy=(0.05, 0.95), xycoords='axes fraction', 
                           fontsize=8, ha='left', va='top', 
                           bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            
            # Configurer les axes
            ax.set_xlabel("Jours depuis sortie", fontsize=8)
            ax.set_ylabel("% variation", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(True, alpha=0.3)
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder en PNG
    png_path = os.path.join(OUTPUT_DIR, 'small_multiples_metrics.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Visualisation sauvegardée en PNG: {png_path}")
    
    # Ajouter au PDF (peut nécessiter plusieurs pages)
    pdf.savefig(fig)
    plt.close()
else:
    print("Pas assez de données pour générer la visualisation 'small multiples'")

# Fermer le PDF
pdf.close()
print(f"\nToutes les visualisations ont été sauvegardées en PDF: {pdf_path}")
print("\nGénération des visualisations terminée!") 