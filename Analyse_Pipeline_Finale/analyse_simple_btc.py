#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import time
from scipy.stats import pearsonr

"""
Script d'analyse des corrélations entre Bitcoin et les métriques des projets.
Génère le fichier correlations_btc.csv nécessaire pour l'analyse de décorrélation.
"""

# Définition des chemins (avec chemins relatifs)
BASE_PATH = '.'  # Chemin relatif au répertoire courant
METRICS_DIR = os.path.join(BASE_PATH, 'FinancialResults/Outputs')
BTC_HISTORY_FILE = os.path.join(BASE_PATH, 'Analysis/Bitcoin_04_07_2010-03_09_2010_historical_data_coinmarketcap.csv')
OUTPUT_DIR = os.path.join(METRICS_DIR, 'AnalyseBTC')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Création des répertoires de sortie
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

# Fonction pour convertir un mois/année en date
def mois_annee_en_date(date_str):
    if pd.isna(date_str) or date_str == 'N/A':
        return None
    
    try:
        return datetime.strptime(date_str, '%b %Y')
    except ValueError:
        return None

# Fonction pour extraire le nom du projet à partir du nom de fichier
def extraire_nom_projet(filepath):
    filename = os.path.basename(filepath)
    return os.path.splitext(filename)[0]

# Fonction pour obtenir le prix BTC à une date donnée
def get_btc_price(date, btc_df):
    # Convertir la date en UTC si elle n'a pas de timezone
    if date.tzinfo is None:
        # Créer une date naïve (sans fuseau)
        date_naive = pd.Timestamp(date.year, date.month, date.day)
    else:
        # Si date a déjà un timezone, la convertir en naïve
        date_naive = pd.Timestamp(date.year, date.month, date.day)
        
    # Convertir les dates BTC en naïves pour comparer uniquement les dates (pas les heures)
    btc_df_naive = btc_df.copy()
    btc_df_naive['date_naive'] = btc_df_naive['date'].dt.date.apply(lambda x: pd.Timestamp(x))
    
    # Trouver la ligne la plus proche antérieure ou égale à la date demandée
    filtered_df = btc_df_naive[btc_df_naive['date_naive'] <= date_naive]
    
    if not filtered_df.empty:
        closest_row = filtered_df.iloc[-1]
        return closest_row['close']
    return None

# Fonction pour afficher le temps écoulé
def log_time(start_time, message):
    elapsed = time.time() - start_time
    print(f"{message} (Temps écoulé: {elapsed:.2f} secondes)")
    return time.time()

def main():
    # Enregistrer le temps de début
    start_time_total = time.time()
    print("Analyse simple de la relation entre BTC et les métriques des projets")
    print("-----------------------------------------------------------------")

    # Chargement des données historiques de Bitcoin
    print("\nChargement des données historiques Bitcoin...")
    start_time = time.time()
    btc_df = pd.read_csv(BTC_HISTORY_FILE, sep=';')

    # Conversion des dates dans le format approprié
    try:
        btc_df['date'] = pd.to_datetime(btc_df['timeClose'])
    except:
        try:
            btc_df['date'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
        except:
            btc_df['date'] = pd.to_datetime(btc_df['timeOpen'])

    # S'assurer que les dates sont en ordre croissant
    btc_df = btc_df.sort_values('date')

    # Afficher la plage de dates
    min_date = btc_df['date'].min().strftime('%Y-%m-%d')
    max_date = btc_df['date'].max().strftime('%Y-%m-%d')
    start_time = log_time(start_time, f"Données Bitcoin chargées: {len(btc_df)} jours")
    print(f"Période couverte: {min_date} à {max_date}")

    # Traitement des fichiers de métriques
    print("\nTraitement des fichiers de métriques...")
    start_time = time.time()
    donnees_metriques = []

    # Liste tous les fichiers CSV du répertoire de métriques
    fichiers_csv = [f for f in os.listdir(METRICS_DIR) if f.endswith('.csv') and f != 'impact_summary.csv' and f != 'corr_summary.csv' and f != 'categorized_impact.csv']
    print(f"Nombre de fichiers de métriques trouvés: {len(fichiers_csv)}")

    for i, fichier_csv in enumerate(fichiers_csv):
        if not fichier_csv.endswith('.csv') or fichier_csv.startswith('.'):
            continue
        
        chemin_fichier = os.path.join(METRICS_DIR, fichier_csv)
        projet = extraire_nom_projet(chemin_fichier)
        
        try:
            # Lecture du fichier CSV
            df = pd.read_csv(chemin_fichier)
            
            # Vérification de la structure du fichier
            if 'Metric' not in df.columns:
                print(f"Fichier ignoré {fichier_csv} - structure inattendue")
                continue
                
            # Obtention des colonnes de dates
            colonnes_date = [col for col in df.columns if col != 'Metric']
            
            # Transformation de large à long (melt)
            df_long = pd.melt(
                df, 
                id_vars=['Metric'], 
                value_vars=colonnes_date,
                var_name='date_str', 
                value_name='valeur'
            )
            
            # Conversion des dates
            df_long['date'] = df_long['date_str'].apply(mois_annee_en_date)
            
            # Filtrage des lignes avec dates invalides
            df_long = df_long[~df_long['date'].isna()]
            
            # Ajout de la colonne projet
            df_long['projet'] = projet
            
            # Sélection des colonnes nécessaires
            df_long = df_long[['projet', 'Metric', 'date', 'valeur']]
            
            # Renommage de 'Metric' en 'metrique'
            df_long = df_long.rename(columns={'Metric': 'metrique'})
            
            # Ajout aux données
            donnees_metriques.append(df_long)
            print(f"Traité: {projet} ({i+1}/{len(fichiers_csv)})")
            
        except Exception as e:
            print(f"Erreur lors du traitement de {fichier_csv}: {str(e)}")

    # Combinaison de toutes les données de métriques
    if not donnees_metriques:
        raise ValueError("Aucune donnée de métrique valide trouvée")

    start_time_concat = time.time()
    print("Concaténation des données de métriques...")
    metriques_df = pd.concat(donnees_metriques, ignore_index=True)
    log_time(start_time_concat, "Concaténation terminée")

    # Conversion des valeurs en numérique
    print("Conversion des valeurs en numérique...")
    start_time_convert = time.time()
    metriques_df['valeur'] = pd.to_numeric(metriques_df['valeur'].replace('N/A', np.nan), errors='coerce')
    log_time(start_time_convert, "Conversion terminée")

    # Suppression des lignes avec valeurs NaN
    metriques_df = metriques_df.dropna(subset=['valeur'])

    log_time(start_time, f"Traitement des fichiers terminé")
    print(f"\nDimension des données de métriques traitées: {metriques_df.shape}")

    # Définition des fenêtres temporelles en mois
    fenetres = [1, 6, 12, 24]

    # Calcul des variations pour chaque période et métrique
    print("\nCalcul des variations...")
    start_time_variations = time.time()
    resultats = []

    # Obtention des projets de métriques
    projets_metriques = metriques_df['projet'].unique()
    print(f"Analyse de {len(projets_metriques)} projets...")

    # Initialisation des compteurs pour le suivi
    total_projets = len(projets_metriques)
    projets_traites = 0
    points_generes = 0
    total_points_possibles = 0
    points_ignores = 0

    # Pour chaque projet et métrique, analyser différentes périodes
    for projet in projets_metriques:
        projets_traites += 1
        start_time_projet = time.time()
        
        # Filtrage des métriques pour ce projet
        metriques_projet = metriques_df[metriques_df['projet'] == projet]
        
        # Obtention des métriques uniques pour ce projet
        metriques_uniques = metriques_projet['metrique'].unique()
        
        print(f"Traitement du projet {projet} ({projets_traites}/{total_projets}) - {len(metriques_uniques)} métriques")
        
        for metrique in metriques_uniques:
            # Filtrage des données pour cette métrique
            donnees_metrique = metriques_projet[metriques_projet['metrique'] == metrique]
            
            # Tri par date
            donnees_metrique = donnees_metrique.sort_values('date')
            
            nb_points = len(donnees_metrique)
            total_points_possibles += nb_points * len(fenetres)
            
            # Pour chaque point de données, calculer la variation sur les fenêtres temporelles
            for i in range(len(donnees_metrique) - 1):
                date_base = donnees_metrique.iloc[i]['date']
                valeur_base = donnees_metrique.iloc[i]['valeur']
                
                # Ignore si la valeur de base est zéro pour éviter la division par zéro
                if valeur_base == 0:
                    points_ignores += len(fenetres)
                    continue
                
                # Prix BTC à la date de base
                btc_price_base = get_btc_price(date_base, btc_df)
                if btc_price_base is None:
                    points_ignores += len(fenetres)
                    continue
                    
                # Pour chaque période de temps d'intérêt
                for fenetre in fenetres:
                    # Calcul de la date cible
                    date_cible = date_base + relativedelta(months=fenetre)
                    
                    # Chercher le point de données le plus proche de la date cible
                    filtered_df = donnees_metrique[donnees_metrique['date'] >= date_base]
                    filtered_df = filtered_df[filtered_df['date'] <= date_cible]
                    
                    if len(filtered_df) <= 1:  # Besoin d'au moins deux points pour une variation
                        points_ignores += 1
                        continue
                    
                    # Prendre le dernier point dans la fenêtre de temps
                    donnee_cible = filtered_df.iloc[-1]
                    date_reelle = donnee_cible['date']
                    valeur_cible = donnee_cible['valeur']
                    
                    # Vérifier que la date réelle est différente de la date de base
                    if date_reelle == date_base:
                        points_ignores += 1
                        continue
                    
                    # Calculer la variation en pourcentage de la métrique
                    variation_pct = ((valeur_cible - valeur_base) / valeur_base) * 100
                    
                    # Prix BTC à la date réelle
                    btc_price_target = get_btc_price(date_reelle, btc_df)
                    if btc_price_target is None or btc_price_base == 0:
                        points_ignores += 1
                        continue
                    
                    # Calculer la variation en pourcentage du BTC
                    btc_variation_pct = ((btc_price_target - btc_price_base) / btc_price_base) * 100
                    
                    # Détermination de la direction de la métrique (positif/négatif)
                    if variation_pct > 5:
                        direction_metrique = "Positive"
                    elif variation_pct < -5:
                        direction_metrique = "Négative"
                    else:
                        direction_metrique = "Stable"
                    
                    # Détermination de la direction du BTC (positif/négatif)
                    if btc_variation_pct > 5:
                        direction_btc = "Positive"
                    elif btc_variation_pct < -5:
                        direction_btc = "Négative"
                    else:
                        direction_btc = "Stable"
                    
                    # Détermination de la concordance entre BTC et la métrique
                    if direction_metrique == direction_btc:
                        concordance = "Oui"
                    else:
                        concordance = "Non"
                    
                    # Ajout aux résultats
                    resultats.append({
                        'projet': projet,
                        'metrique': metrique,
                        'fenetre_mois': fenetre,
                        'date_base': date_base,
                        'date_cible': date_reelle,
                        'valeur_base': valeur_base,
                        'valeur_cible': valeur_cible,
                        'variation_pct': variation_pct,
                        'btc_price_base': btc_price_base,
                        'btc_price_cible': btc_price_target,
                        'btc_variation_pct': btc_variation_pct,
                        'direction_metrique': direction_metrique,
                        'direction_btc': direction_btc,
                        'concordance': concordance
                    })
                    points_generes += 1
        
        # Afficher la progression après chaque projet
        if projets_traites % 5 == 0 or projets_traites == total_projets:
            elapsed = time.time() - start_time_variations
            progress = (projets_traites / total_projets) * 100
            print(f"Progression: {progress:.1f}% ({projets_traites}/{total_projets} projets, {points_generes} points générés)")
            print(f"Temps écoulé: {elapsed:.1f} secondes, Temps estimé restant: {(elapsed / projets_traites) * (total_projets - projets_traites):.1f} secondes")

        log_time(start_time_projet, f"Projet {projet} traité")

    log_time(start_time_variations, f"Calcul des variations terminé")
    print(f"Points possibles: {total_points_possibles}, Points générés: {points_generes}, Points ignorés: {points_ignores}")

    print("\nCréation du DataFrame de résultats...")
    start_time_df = time.time()
    comparaison_df = pd.DataFrame(resultats)
    log_time(start_time_df, "DataFrame créé")

    # Filtrage des valeurs aberrantes avec méthode IQR adaptée à la crypto
    print("Filtrage des valeurs aberrantes (méthode IQR, facteur=2.0)...")
    start_time_filter = time.time()

    def remove_outliers_iqr_crypto(df, column, factor=2.0):
        """
        Supprime les valeurs aberrantes en utilisant la méthode IQR
        avec facteur adapté à la volatilité crypto
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Statistiques pour transparence
        n_total = len(df)
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        n_kept = len(df_filtered)
        n_removed = n_total - n_kept
        
        print(f"    {column}: Q1={Q1:.1f}%, Q3={Q3:.1f}%, IQR={IQR:.1f}%")
        print(f"    Limites: [{lower_bound:.1f}%, {upper_bound:.1f}%]")
        print(f"    Supprimé: {n_removed} points ({n_removed/n_total*100:.1f}%)")
        
        return df_filtered

    # Appliquer le filtrage IQR avec facteur crypto
    comparaison_df_original = comparaison_df.copy()
    print("  Filtrage variation_pct:")
    comparaison_df = remove_outliers_iqr_crypto(comparaison_df, 'variation_pct', factor=2.0)
    print("  Filtrage btc_variation_pct:")
    comparaison_df = remove_outliers_iqr_crypto(comparaison_df, 'btc_variation_pct', factor=2.0)

    nb_aberrantes = len(comparaison_df_original) - len(comparaison_df)
    pct_aberrantes = (nb_aberrantes / len(comparaison_df_original)) * 100

    log_time(start_time_filter, f"Filtrage terminé: {nb_aberrantes} aberrantes supprimées ({pct_aberrantes:.1f}%)")

    # Afficher les statistiques finales
    print(f"   Données originales: {len(comparaison_df_original)}")
    print(f"   Données conservées: {len(comparaison_df)}")
    print(f"   Taux de conservation: {(len(comparaison_df)/len(comparaison_df_original)*100):.1f}%")

    print(f"Nombre de points de comparaison générés: {len(comparaison_df)}")

    # Enregistrement du résumé de comparaison dans un CSV
    print("\nEnregistrement du résumé de comparaison...")
    start_time_save = time.time()
    chemin_resume = os.path.join(OUTPUT_DIR, 'resume_comparaison_btc.csv')
    comparaison_df.to_csv(chemin_resume, index=False)
    log_time(start_time_save, f"Résumé de comparaison enregistré dans {chemin_resume}")

    print("\nCalcul des corrélations...")
    start_time_corr = time.time()
    
    correlations = {}
    nb_metriques = len(comparaison_df['metrique'].unique())
    metriques_traitees = 0

    for metrique in comparaison_df['metrique'].unique():
        metriques_traitees += 1
        if metriques_traitees % 20 == 0 or metriques_traitees == nb_metriques:
            print(f"  - Calcul des corrélations: {metriques_traitees}/{nb_metriques} métriques traitées")
        
        for fenetre in fenetres:
            subset = comparaison_df[(comparaison_df['fenetre_mois'] == fenetre) & 
                                (comparaison_df['metrique'] == metrique)]
            
            if len(subset) >= 5:  # Au moins 5 points pour une corrélation fiable
                # Calcul de la corrélation avec test de significativité
                try:
                    corr_coeff, p_value = pearsonr(subset['btc_variation_pct'], subset['variation_pct'])
                    
                    # Détermination de la significativité
                    if p_value < 0.001:
                        significativite = "Très significatif (p<0.001)"
                    elif p_value < 0.01:
                        significativite = "Significatif (p<0.01)"
                    elif p_value < 0.05:
                        significativite = "Modérément significatif (p<0.05)"
                    else:
                        significativite = "Non significatif (p≥0.05)"
                    
                    correlations[(metrique, fenetre)] = {
                        'correlation': corr_coeff,
                        'p_value': p_value,
                        'significativite': significativite,
                        'n_observations': len(subset)
                    }
                except:
                    # En cas d'erreur, utiliser l'ancienne méthode
                    corr = subset['btc_variation_pct'].corr(subset['variation_pct'])
                    correlations[(metrique, fenetre)] = {
                        'correlation': corr,
                        'p_value': np.nan,
                        'significativite': "Non calculé",
                        'n_observations': len(subset)
                    }

    if correlations:
        print(f"  - {len(correlations)} corrélations calculées")
        
        # Création d'un DataFrame pour les corrélations
        corr_data = []
        for (metrique, fenetre), corr_info in correlations.items():
            corr_data.append({
                'metrique': metrique,
                'fenetre_mois': fenetre,
                'correlation': corr_info['correlation'],
                'p_value': corr_info['p_value'],
                'significativite': corr_info['significativite'],
                'n_observations': corr_info['n_observations']
            })
        
        corr_df = pd.DataFrame(corr_data)
        
        # Enregistrement des corrélations dans un CSV
        chemin_corr = os.path.join(OUTPUT_DIR, 'correlations_btc.csv')
        corr_df.to_csv(chemin_corr, index=False)
        print(f"  - Résumé des corrélations enregistré dans {chemin_corr}")
        
        # Affichage d'un résumé des significativités
        print("\nRésumé des tests de significativité:")
        signif_counts = corr_df['significativite'].value_counts()
        for signif, count in signif_counts.items():
            print(f"  - {signif}: {count} corrélations")
    
    log_time(start_time_corr, "Calcul des corrélations terminé")
    log_time(start_time_total, "\nAnalyse terminée avec succès!")

    print(f"Les fichiers de sortie sont dans {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 