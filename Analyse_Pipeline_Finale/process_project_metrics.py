#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import glob

"""
Ce script traite les fichiers CSV des métriques de projets individuels 
et génère le fichier resume_impact.csv nécessaire pour l'analyse d'impact.
"""

# Définition des chemins
BASE_PATH = '.'  # Chemin relatif au répertoire courant
PROJECT_FILES_DIR = os.path.join(BASE_PATH, 'FinancialResults/Outputs')
EXITS_FILE = os.path.join(BASE_PATH, 'Onchain_exits_dynamic.csv')
OUTPUT_DIR = os.path.join(BASE_PATH, 'FinancialResults/Outputs/AnalyseSimple')
ORIGINAL_FILES_DIR = '/Users/fabiobrugnone/Desktop/QUANTIIII/FinancialResults/Outputs'

# Création du répertoire de sortie s'il n'existe pas
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Fonction pour afficher le temps écoulé
def log_time(start_time, message):
    elapsed = time.time() - start_time
    print(f"{message} (Temps écoulé: {elapsed:.2f} secondes)")
    return time.time()

# Enregistrer le temps de début
start_time_total = time.time()
print("TRAITEMENT DES MÉTRIQUES DE PROJETS POUR ANALYSE D'IMPACT")
print("======================================================")

# Vérification et copie des fichiers de métriques depuis QUANTIIII
print("\nSynchronisation des fichiers de métriques...")
# S'assurer que le répertoire de destination existe
Path(PROJECT_FILES_DIR).mkdir(parents=True, exist_ok=True)

# Récupérer tous les fichiers de métriques de projets depuis le répertoire original
original_files = []
if os.path.exists(ORIGINAL_FILES_DIR):
    original_files = glob.glob(os.path.join(ORIGINAL_FILES_DIR, '*.csv'))
    original_files = [f for f in original_files if os.path.basename(f) not in ['resume_impact.csv', 'correlations_btc.csv']]

    if original_files:
        print(f"Copie des fichiers de métriques depuis le répertoire original...")
        import shutil
        # Compter les fichiers copiés
        copy_count = 0
        for file in original_files:
            if os.path.basename(file) not in ['resume_impact.csv', 'correlations_btc.csv']:
                dest_file = os.path.join(PROJECT_FILES_DIR, os.path.basename(file))
                shutil.copy2(file, dest_file)
                copy_count += 1
                print(f"  Copié: {os.path.basename(file)}")
        print(f"Nombre de fichiers copiés: {copy_count}")
else:
    print(f"Répertoire original non trouvé: {ORIGINAL_FILES_DIR}")

# Recherche des fichiers de métriques de projets dans le répertoire local
print("\nRecherche des fichiers de métriques dans le répertoire local...")
project_files = glob.glob(os.path.join(PROJECT_FILES_DIR, '*.csv'))
project_files = [f for f in project_files if os.path.basename(f) not in ['resume_impact.csv', 'correlations_btc.csv']]

if not project_files:
    print(f"Aucun fichier métrique trouvé dans le répertoire local.")
    if not original_files:
        raise FileNotFoundError(f"Aucun fichier métrique trouvé, ni dans le répertoire local ni dans le répertoire original.")
else:
    print(f"Fichiers de métriques trouvés: {len(project_files)}")

# Chargement des données des sorties de fonds
print("\nChargement des données de sorties...")
if not os.path.exists(EXITS_FILE):
    raise FileNotFoundError(f"Fichier des sorties non trouvé: {EXITS_FILE}")

exits_df = pd.read_csv(EXITS_FILE)
print(f"Données de sorties chargées: {len(exits_df)} entrées")

# Conversion des dates et standardisation des noms de colonnes
if 'date_de_sortie' in exits_df.columns:
    exits_df.rename(columns={
        'fond': 'fond',
        'projet_sorti': 'projet',
        'date_de_sortie': 'date'
    }, inplace=True)
else:
    # Assurez-vous que les colonnes nécessaires sont présentes
    required_columns = ['fond', 'projet', 'date']
    for col in required_columns:
        if col not in exits_df.columns:
            raise ValueError(f"Colonne requise manquante dans le fichier des sorties: {col}")

# Conversion des dates
exits_df['date'] = pd.to_datetime(exits_df['date'])

# Définition des fenêtres temporelles en mois
fenetres_mois = [1, 6, 12, 24]

# Préparation des données pour l'analyse d'impact
print("\nTraitement des fichiers de métriques pour analyse d'impact...")
start_time = time.time()

# Structure pour stocker les résultats
impact_results = []

# Pour chaque sortie de fonds
for _, exit_row in exits_df.iterrows():
    fond = exit_row['fond']
    projet = exit_row['projet']
    date_sortie = exit_row['date']
    
    # Chercher le fichier correspondant au projet
    project_file = None
    for file in project_files:
        if os.path.basename(file).split('.')[0].upper() == projet.upper():
            project_file = file
            break
    
    if project_file is None:
        print(f"  Avertissement: Pas de fichier métrique trouvé pour le projet {projet}")
        continue
    
    print(f"  Traitement du projet {projet} (sortie du {date_sortie.strftime('%Y-%m-%d')} par {fond})...")
    
    # Chargement des données du projet
    try:
        project_df = pd.read_csv(project_file)
    except Exception as e:
        print(f"    Erreur lors du chargement du fichier {projet}: {e}")
        continue
    
    # Préparation des données
    # Les fichiers ont une structure avec Metric en première colonne et dates en colonnes
    if 'Metric' not in project_df.columns:
        print(f"    Format de fichier non reconnu pour {projet}")
        continue
    
    # Convertir les dates en datetime
    date_columns = project_df.columns[1:]  # Toutes les colonnes sauf 'Metric'
    date_map = {}
    
    for date_col in date_columns:
        try:
            # Essayer plusieurs formats de date
            for fmt in ['%b %Y', '%B %Y', '%Y-%m-%d', '%Y-%m', '%Y/%m']:
                try:
                    parsed_date = pd.to_datetime(date_col, format=fmt)
                    date_map[date_col] = parsed_date
                    break
                except:
                    continue
        except:
            # Si aucun format ne fonctionne, ignorer cette colonne
            pass
    
    # Pour chaque métrique dans le fichier
    for _, metric_row in project_df.iterrows():
        metric_name = metric_row['Metric']
        
        # Pour chaque fenêtre temporelle
        for fenetre in fenetres_mois:
            # Calculer les dates avant et après la sortie
            date_apres = date_sortie + pd.DateOffset(months=fenetre)
            date_avant = date_sortie
            
            # Trouver les colonnes correspondant à ces dates
            col_apres = None
            col_avant = None
            
            # Trouver la colonne la plus proche pour la date avant
            min_diff_avant = timedelta(days=365*10)  # Initialiser à une grande valeur
            for col, col_date in date_map.items():
                diff = abs(col_date - date_avant)
                if diff < min_diff_avant:
                    min_diff_avant = diff
                    col_avant = col
            
            # Trouver la colonne la plus proche pour la date après
            min_diff_apres = timedelta(days=365*10)  # Initialiser à une grande valeur
            for col, col_date in date_map.items():
                diff = abs(col_date - date_apres)
                if diff < min_diff_apres:
                    min_diff_apres = diff
                    col_apres = col
            
            # Si les colonnes sont trouvées
            if col_avant and col_apres:
                # Extraire les valeurs
                valeur_avant = metric_row[col_avant]
                valeur_apres = metric_row[col_apres]
                
                # Convertir en nombres si possible
                try:
                    valeur_avant = float(valeur_avant) if valeur_avant != 'N/A' else np.nan
                    valeur_apres = float(valeur_apres) if valeur_apres != 'N/A' else np.nan
                except:
                    # Ignorer cette ligne si les valeurs ne sont pas numériques
                    continue
                
                # Si les deux valeurs sont disponibles, calculer l'impact
                if not np.isnan(valeur_avant) and not np.isnan(valeur_apres) and valeur_avant != 0:
                    # Calculer la variation en pourcentage
                    variation_pct = ((valeur_apres - valeur_avant) / abs(valeur_avant)) * 100
                    
                    # Déterminer la direction de l'impact
                    if variation_pct > 5:
                        direction = "Positive"
                    elif variation_pct < -5:
                        direction = "Négative"
                    else:
                        direction = "Stable"
                    
                    # Stocker les résultats
                    impact_results.append({
                        'fond': fond,
                        'projet': projet,
                        'metrique': metric_name,
                        'fenetre_mois': fenetre,
                        'date_sortie': date_sortie,
                        'date_avant': date_map[col_avant] if col_avant in date_map else date_avant,
                        'date_apres': date_map[col_apres] if col_apres in date_map else date_apres,
                        'valeur_avant': valeur_avant,
                        'valeur_apres': valeur_apres,
                        'variation_pct': variation_pct,
                        'direction': direction
                    })

# Conversion en DataFrame
impact_df = pd.DataFrame(impact_results)
print(f"Nombre total de points d'impact calculés: {len(impact_df)}")

log_time(start_time, "Traitement des métriques terminé")

# Enregistrement des résultats
print("\nEnregistrement des résultats...")
output_file = os.path.join(OUTPUT_DIR, 'resume_impact.csv')
impact_df.to_csv(output_file, index=False)
print(f"Résultats enregistrés dans {output_file}")

# Résumé par projet
projet_summary = impact_df.groupby('projet').agg({
    'metrique': 'nunique',
    'variation_pct': ['count', 'mean']
}).reset_index()
projet_summary.columns = ['Projet', 'Métriques uniques', 'Points de données', 'Variation moyenne (%)']
projet_summary = projet_summary.sort_values('Points de données', ascending=False)

# Enregistrement du résumé par projet
resume_projet_file = os.path.join(OUTPUT_DIR, 'resume_par_projet.csv')
projet_summary.to_csv(resume_projet_file, index=False)
print(f"Résumé par projet enregistré dans {resume_projet_file}")

log_time(start_time_total, "\nTraitement terminé avec succès!") 