#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration des chemins
BASE_PATH = '.'
DECORR_FILE = os.path.join(BASE_PATH, 'FinancialResults/Outputs/AnalyseDecorelation/impact_decorrelé.csv')
OUTPUT_DIR = os.path.join(BASE_PATH, 'FinancialResults/Outputs/AnalyseDecorelation')
VISUALS_DIR = os.path.join(BASE_PATH, 'VisualsOutput')

# Création du répertoire de visualisations
Path(VISUALS_DIR).mkdir(parents=True, exist_ok=True)

print("GÉNÉRATION DU TABLEAU FINAL - MÉDIANES DES IMPACTS")
print("=================================================")

# Chargement des données décorrélées
if not os.path.exists(DECORR_FILE):
    raise FileNotFoundError(f"Fichier d'impact décorrélé non trouvé: {DECORR_FILE}")

decorr_df = pd.read_csv(DECORR_FILE)
print(f"Données d'impact décorrélé chargées: {len(decorr_df)} entrées")

# Sélection des 15 métriques stratégiques pour la pérennité
metriques_perennite = [
    # FONDAMENTAUX DE PÉRENNITÉ (5 métriques)
    'Revenue',
    'Earnings', 
    'Core developers',
    'Active users (monthly)',
    'Net deposits',
    
    # VIABILITÉ ÉCONOMIQUE (4 métriques)
    'Fees',
    'Trading volume',
    'Treasury',
    'P/S ratio (fully diluted)',
    
    # ADOPTION & CROISSANCE (3 métriques)
    'Active users (daily)',
    'Token holders',
    'Fully diluted market cap',
    
    # RÉSILIENCE OPÉRATIONNELLE (3 métriques)
    'Price',
    'Token turnover (fully diluted)',
    'Circulating supply'
]

print(f"\n15 Métriques sélectionnées pour l'analyse de pérennité:")
for i, metrique in enumerate(metriques_perennite, 1):
    print(f"  {i:2d}. {metrique}")

# Filtrage des données pour les métriques de pérennité
decorr_filtered = decorr_df[decorr_df['metrique'].isin(metriques_perennite)]
print(f"\nDonnées filtrées: {len(decorr_filtered)} lignes sur {len(decorr_df)} totales")

# Filtrage pour les fenêtres temporelles d'intérêt (1, 6, 12 mois)
fenetres_interesse = [1, 6, 12]
decorr_filtered = decorr_filtered[decorr_filtered['fenetre_mois'].isin(fenetres_interesse)]
print(f"Données après filtrage temporel: {len(decorr_filtered)} lignes")

print("\nCalcul des médianes par métrique et horizon temporel...")

# Calcul des médianes ET des moyennes pour comparaison
resultats_medianes = []
resultats_moyennes = []

for metrique in metriques_perennite:
    if metrique not in decorr_filtered['metrique'].values:
        print(f"⚠️ Métrique '{metrique}' non trouvée dans les données")
        continue
    
    ligne_mediane = {'metrique': metrique}
    ligne_moyenne = {'metrique': metrique}
    
    # Pour chaque fenêtre temporelle
    for fenetre in fenetres_interesse:
        subset = decorr_filtered[
            (decorr_filtered['metrique'] == metrique) & 
            (decorr_filtered['fenetre_mois'] == fenetre)
        ]
        
        if not subset.empty:
            mediane = subset['impact_decorrelé'].median()
            moyenne = subset['impact_decorrelé'].mean()
            ligne_mediane[f'{fenetre}_mois'] = mediane
            ligne_moyenne[f'{fenetre}_mois'] = moyenne
        else:
            ligne_mediane[f'{fenetre}_mois'] = np.nan
            ligne_moyenne[f'{fenetre}_mois'] = np.nan
    
    # Calcul de la médiane/moyenne globale
    subset_global = decorr_filtered[decorr_filtered['metrique'] == metrique]
    if not subset_global.empty:
        ligne_mediane['median_global'] = subset_global['impact_decorrelé'].median()
        ligne_moyenne['median_global'] = subset_global['impact_decorrelé'].mean()
    else:
        ligne_mediane['median_global'] = np.nan
        ligne_moyenne['median_global'] = np.nan
    
    resultats_medianes.append(ligne_mediane)
    resultats_moyennes.append(ligne_moyenne)

# Création des DataFrames
df_medianes = pd.DataFrame(resultats_medianes)
df_moyennes = pd.DataFrame(resultats_moyennes)

# Tri par médiane globale décroissante
df_medianes = df_medianes.sort_values('median_global', ascending=False)
df_moyennes = df_moyennes.sort_values('median_global', ascending=False)

print(f"\nTableau final généré: {len(df_medianes)} métriques × {len(df_medianes.columns)} colonnes")

# Affichage du tableau des médianes
print("\n" + "="*80)
print("TABLEAU FINAL - IMPACT MÉDIAN DES SORTIES DE FONDS SUR LA PÉRENNITÉ")
print("(Médianes des impacts décorrélés en %)")
print("="*80)

# Formatage pour l'affichage
df_display = df_medianes.copy()
df_display = df_display.set_index('metrique')

# Formatage des pourcentages
for col in df_display.columns:
    df_display[col] = df_display[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")

print(df_display.to_string())

# Sauvegarde en CSV
csv_path = os.path.join(OUTPUT_DIR, 'tableau_final_medianes_perennite.csv')
df_medianes.to_csv(csv_path, index=False)
print(f"\nTableau sauvegardé en CSV: {csv_path}")

# Génération du graphique de comparaison moyenne vs médiane
print("\n" + "="*80)
print("GÉNÉRATION DU GRAPHIQUE DE COMPARAISON MOYENNE vs MÉDIANE")
print("="*80)

# Configuration matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Préparation des données pour le graphique
fenetres_labels = ['1 mois', '6 mois', '12 mois']
fenetres_cols = ['1_mois', '6_mois', '12_mois']

# Création de la figure avec 2 sous-graphiques
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle('Comparaison Moyenne vs Médiane - Impact des Désengagements', fontsize=16, fontweight='bold')

# Couleurs pour les horizons temporels
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Bleu, Orange, Vert

# Largeur des barres
bar_width = 0.25
x_pos = np.arange(len(metriques_perennite))

# GRAPHIQUE A: Calcul par MOYENNE
ax1.set_title('(A) Calcul par MOYENNE', fontsize=14, fontweight='bold')

for i, (fenetre_col, fenetre_label, color) in enumerate(zip(fenetres_cols, fenetres_labels, colors)):
    # Réorganiser les données selon l'ordre des métriques de pérennité
    values = []
    for metrique in metriques_perennite:
        row = df_moyennes[df_moyennes['metrique'] == metrique]
        if not row.empty and pd.notna(row[fenetre_col].iloc[0]):
            values.append(row[fenetre_col].iloc[0])
        else:
            values.append(0)
    
    # Tracer les barres
    bars = ax1.bar(x_pos + i * bar_width, values, bar_width, 
                   label=fenetre_label, color=color, alpha=0.8)
    
    # Ajouter les valeurs sur les barres
    for j, (bar, value) in enumerate(zip(bars, values)):
        if abs(value) > 5:  # Afficher seulement si > 5%
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -10),
                    f'{value:.0f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, fontweight='bold')

# GRAPHIQUE B: Calcul par MÉDIANE
ax2.set_title('(B) Calcul par MÉDIANE (plus robuste aux outliers)', fontsize=14, fontweight='bold')

for i, (fenetre_col, fenetre_label, color) in enumerate(zip(fenetres_cols, fenetres_labels, colors)):
    # Réorganiser les données selon l'ordre des métriques de pérennité
    values = []
    for metrique in metriques_perennite:
        row = df_medianes[df_medianes['metrique'] == metrique]
        if not row.empty and pd.notna(row[fenetre_col].iloc[0]):
            values.append(row[fenetre_col].iloc[0])
        else:
            values.append(0)
    
    # Tracer les barres
    bars = ax2.bar(x_pos + i * bar_width, values, bar_width, 
                   label=fenetre_label, color=color, alpha=0.8)
    
    # Ajouter les valeurs sur les barres
    for j, (bar, value) in enumerate(zip(bars, values)):
        if abs(value) > 5:  # Afficher seulement si > 5%
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (2 if height >= 0 else -5),
                    f'{value:.0f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, fontweight='bold')

# Configuration des axes pour les deux graphiques
for ax in [ax1, ax2]:
    ax.set_xlabel('Métriques de Pérennité', fontsize=12)
    ax.set_ylabel('Impact Décorrélé - MÉDIANE (%)', fontsize=12)
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(metriques_perennite, rotation=45, ha='right')
    ax.legend(title='Horizon Temporel', loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Ajustement de la mise en page
plt.tight_layout()

# Sauvegarde du graphique
png_path = os.path.join(VISUALS_DIR, 'comparaison_moyenne_vs_mediane.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Graphique de comparaison sauvegardé: {png_path}")

# Statistiques résumées (comme avant)
print("\n" + "="*80)
print("STATISTIQUES RÉSUMÉES")
print("="*80)

print("\n📊 RÉPARTITION DES IMPACTS MÉDIANS:")

for fenetre, fenetre_col in zip(fenetres_labels, fenetres_cols):
    values = df_medianes[fenetre_col].dropna()
    positifs = sum(values > 5)
    negatifs = sum(values < -5)
    stables = sum(abs(values) <= 5)
    total = len(values)
    mediane = values.median()
    
    print(f"\n{fenetre}:")
    print(f"  • Impacts positifs: {positifs}/{total} ({positifs/total*100:.1f}%)")
    print(f"  • Impacts négatifs: {negatifs}/{total} ({negatifs/total*100:.1f}%)")
    print(f"  • Impacts stables: {stables}/{total} ({stables/total*100:.1f}%)")
    print(f"  • Impact médian: {mediane:.1f}%")

# Métriques les plus résilientes et impactées
resilientes = df_medianes[df_medianes['median_global'] > 0]['metrique'].tolist()
impactees = df_medianes[df_medianes['median_global'] < -10]['metrique'].tolist()

print(f"\n🎯 MÉTRIQUES LES PLUS RÉSILIENTES (médiane globale > 0):")
for metrique in resilientes:
    mediane = df_medianes[df_medianes['metrique'] == metrique]['median_global'].iloc[0]
    print(f"  • {metrique}: {mediane:+.1f}%")

print(f"\n⚠️  MÉTRIQUES LES PLUS IMPACTÉES (médiane globale < -10%):")
for metrique in impactees:
    mediane = df_medianes[df_medianes['metrique'] == metrique]['median_global'].iloc[0]
    print(f"  • {metrique}: {mediane:.1f}%")

# Comparaison médiane vs moyenne
mediane_globale = df_medianes['median_global'].median()
moyenne_globale = df_moyennes['median_global'].mean()
ecart = moyenne_globale - mediane_globale

print(f"\n📈 COMPARAISON MÉDIANE vs MOYENNE (pour validation):")
print(f"  • Impact médian global: {mediane_globale:.1f}%")
print(f"  • Impact moyen global: {moyenne_globale:.1f}%")
print(f"  • Écart médiane-moyenne: {ecart:.1f} points")

print(f"\n💡 JUSTIFICATION MÉTHODOLOGIQUE:")
if abs(ecart) > 10:
    print(f"  ✅ Utilisation de la médiane JUSTIFIÉE (écart > 10%)")
    print(f"  • La médiane ({mediane_globale:.1f}%) est plus représentative")
    print(f"  • La moyenne ({moyenne_globale:.1f}%) est biaisée par les outliers")
else:
    print(f"  ⚠️ Écart médiane-moyenne faible (< 10%)")
    print(f"  • Les deux mesures sont similaires")

print("\n" + "="*80)
print("GÉNÉRATION TERMINÉE")
print(f"📁 Fichier principal: {csv_path}")
print(f"📊 Graphique comparatif: {png_path}")
print("="*80) 