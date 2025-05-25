import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import time
from sklearn.linear_model import LinearRegression

# Définition des chemins (avec chemins relatifs)
BASE_PATH = '.'  # Chemin relatif au répertoire courant
METRICS_DIR = os.path.join(BASE_PATH, 'FinancialResults/Outputs')
EXITS_FILE = os.path.join(BASE_PATH, 'Onchain_exits_dynamic.csv')
BTC_HISTORY_FILE = os.path.join(BASE_PATH, 'Analysis/Bitcoin_04_07_2010-03_09_2010_historical_data_coinmarketcap.csv')
BTC_ANALYSIS_DIR = os.path.join(METRICS_DIR, 'AnalyseBTC')
EXITS_ANALYSIS_DIR = os.path.join(METRICS_DIR, 'AnalyseSimple')
OUTPUT_DIR = os.path.join(METRICS_DIR, 'AnalyseDecorelation')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Création des répertoires de sortie
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

# Fonction pour afficher le temps écoulé
def log_time(start_time, message):
    elapsed = time.time() - start_time
    print(f"{message} (Temps écoulé: {elapsed:.2f} secondes)")
    return time.time()

# Enregistrer le temps de début
start_time_total = time.time()
print("Analyse de l'impact des sorties avec décorrélation du BTC")
print("--------------------------------------------------------")

# Vérification des fichiers d'analyse précédents
print("\nVérification des analyses précédentes...")
btc_correlations_file = os.path.join(BTC_ANALYSIS_DIR, 'correlations_btc.csv')
btc_comparison_file = os.path.join(BTC_ANALYSIS_DIR, 'resume_comparaison_btc.csv')
exits_impact_file = os.path.join(EXITS_ANALYSIS_DIR, 'resume_impact.csv')

if not os.path.exists(btc_correlations_file):
    raise FileNotFoundError(f"Fichier de corrélations BTC non trouvé: {btc_correlations_file}")
if not os.path.exists(exits_impact_file):
    raise FileNotFoundError(f"Fichier d'impact des sorties non trouvé: {exits_impact_file}")

# Modification: rendre optionnel le fichier de comparaison BTC
if not os.path.exists(btc_comparison_file):
    print(f"Avertissement: Fichier de comparaison BTC non trouvé: {btc_comparison_file}")
    print("L'analyse continuera sans ces données.")
    use_comparison_file = False
else:
    use_comparison_file = True

print("Fichiers d'analyse nécessaires disponibles.")

# Chargement des données
print("\nChargement des données d'analyse...")
start_time = time.time()

# Chargement des corrélations BTC
correlations_btc_df = pd.read_csv(btc_correlations_file)
print(f"Corrélations BTC chargées: {len(correlations_btc_df)} entrées")

# Chargement des données de comparaison BTC (si disponible)
if use_comparison_file:
    btc_comparison_df = pd.read_csv(btc_comparison_file)
    print(f"Données de comparaison BTC chargées: {len(btc_comparison_df)} entrées")

# Chargement des données d'impact des sorties
exits_impact_df = pd.read_csv(exits_impact_file)
print(f"Données d'impact des sorties chargées: {len(exits_impact_df)} entrées")

log_time(start_time, "Chargement des données terminé")

# Préparation des données pour l'analyse de décorrélation
print("\nPréparation des données pour l'analyse de décorrélation...")
start_time = time.time()

# Ajout d'un identifiant unique pour les points d'impact des sorties
exits_impact_df['exit_id'] = exits_impact_df['projet'] + '_' + exits_impact_df['metrique'] + '_' + exits_impact_df['date_sortie'].astype(str)

# Chargement des données historiques de Bitcoin pour obtenir les retours BTC sur les mêmes périodes
print("Chargement des données historiques Bitcoin...")
if not os.path.exists(BTC_HISTORY_FILE):
    raise FileNotFoundError(f"Fichier historique Bitcoin non trouvé: {BTC_HISTORY_FILE}")

btc_df = pd.read_csv(BTC_HISTORY_FILE, sep=';')

# Conversion des dates dans le format approprié
try:
    btc_df['date'] = pd.to_datetime(btc_df['timeClose'])
except:
    try:
        btc_df['date'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
    except:
        try:
            btc_df['date'] = pd.to_datetime(btc_df['timeOpen'])
        except:
            # Tentative de détection automatique du format de date
            date_columns = [col for col in btc_df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if date_columns:
                btc_df['date'] = pd.to_datetime(btc_df[date_columns[0]], errors='coerce')
            else:
                raise ValueError("Impossible de détecter une colonne de date dans le fichier Bitcoin")

# S'assurer que les dates sont en ordre croissant
btc_df = btc_df.sort_values('date')

# Fonction pour obtenir le prix BTC à une date donnée
def get_btc_price(date, btc_df):
    # Convertir la date en UTC si elle n'a pas de timezone
    if isinstance(date, str):
        date = pd.to_datetime(date)
        
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
        # Détection automatique de la colonne de prix
        price_col = None
        for col in ['close', 'Close', 'price', 'Price']:
            if col in closest_row:
                price_col = col
                break
        
        if price_col:
            return closest_row[price_col]
    return None

# Calcul des retours BTC pour chaque point d'impact
print("Calcul des retours BTC pour chaque point d'impact...")
exits_impact_df['date_sortie'] = pd.to_datetime(exits_impact_df['date_sortie'])
exits_impact_df['date_avant'] = pd.to_datetime(exits_impact_df['date_avant'])
exits_impact_df['date_apres'] = pd.to_datetime(exits_impact_df['date_apres'])

# Initialiser les colonnes pour les prix BTC et les retours
exits_impact_df['btc_price_exit'] = None
exits_impact_df['btc_price_after'] = None
exits_impact_df['btc_return'] = None

# Calculer les prix et retours BTC
for i, row in exits_impact_df.iterrows():
    btc_price_exit = get_btc_price(row['date_sortie'], btc_df)
    btc_price_after = get_btc_price(row['date_apres'], btc_df)
    
    if btc_price_exit is not None and btc_price_after is not None and btc_price_exit > 0:
        btc_return = ((btc_price_after - btc_price_exit) / btc_price_exit) * 100
        exits_impact_df.at[i, 'btc_price_exit'] = btc_price_exit
        exits_impact_df.at[i, 'btc_price_after'] = btc_price_after
        exits_impact_df.at[i, 'btc_return'] = btc_return

# Supprimer les lignes sans données BTC
exits_impact_df = exits_impact_df.dropna(subset=['btc_return'])
print(f"Points d'impact avec données BTC: {len(exits_impact_df)}")

log_time(start_time, "Préparation des données terminée")

# Analyse de décorrélation par métrique et fenêtre temporelle
print("\nAnalyse de décorrélation par métrique et fenêtre temporelle...")
start_time = time.time()

# Création d'un DataFrame pour stocker les résultats de décorrélation
decorrelation_results = []

# Définition des fenêtres temporelles
fenetres = [1, 6, 12]  # Réduit à 1, 6, 12 mois pour correspondre à generate_impact_visuals.py

# Obtenir les métriques uniques
metriques_uniques = exits_impact_df['metrique'].unique()
print(f"Analyse de {len(metriques_uniques)} métriques sur {len(fenetres)} fenêtres temporelles...")

# Pour chaque métrique et fenêtre temporelle
for metrique in metriques_uniques:
    # Obtenir la corrélation BTC pour cette métrique
    corr_btc_data = correlations_btc_df[correlations_btc_df['metrique'] == metrique]
    
    for fenetre in fenetres:
        # Filtrer les données d'impact pour cette métrique et fenêtre
        impact_subset = exits_impact_df[(exits_impact_df['metrique'] == metrique) & 
                                       (exits_impact_df['fenetre_mois'] == fenetre)]
        
        if len(impact_subset) < 3:  # Réduit à 3 points minimum pour plus de données
            continue
            
        # Obtenir la corrélation BTC pour cette métrique et fenêtre
        corr_btc = corr_btc_data[corr_btc_data['fenetre_mois'] == fenetre]['correlation'].values
        
        if len(corr_btc) == 0:
            # Utiliser la corrélation moyenne pour cette fenêtre si spécifique non disponible
            corr_btc = correlations_btc_df[correlations_btc_df['fenetre_mois'] == fenetre]['correlation'].mean()
            if pd.isna(corr_btc):
                corr_btc = 0  # Valeur par défaut si aucune corrélation disponible
        else:
            corr_btc = corr_btc[0]
        
        # Calcul de l'impact décorrélé pour chaque point
        for _, row in impact_subset.iterrows():
            # Impact brut (variation de la métrique)
            impact_brut = row['variation_pct']
            
            # Impact attendu du BTC basé sur la corrélation
            impact_attendu_btc = row['btc_return'] * corr_btc
            
            # Impact décorrélé (impact brut - impact attendu du BTC)
            impact_decorrelé = impact_brut - impact_attendu_btc
            
            # Détermination de la direction de l'impact décorrélé
            if impact_decorrelé > 5:
                direction_decorrelée = "Positive"
            elif impact_decorrelé < -5:
                direction_decorrelée = "Négative"
            else:
                direction_decorrelée = "Stable"
                
            # Stockage des résultats
            decorrelation_results.append({
                'fond': row['fond'],
                'projet': row['projet'],
                'metrique': row['metrique'],
                'fenetre_mois': fenetre,
                'date_sortie': row['date_sortie'],
                'date_apres': row['date_apres'],
                'valeur_avant': row['valeur_avant'],
                'valeur_apres': row['valeur_apres'],
                'variation_pct': impact_brut,
                'btc_return': row['btc_return'],
                'correlation_btc': corr_btc,
                'impact_attendu_btc': impact_attendu_btc,
                'impact_decorrelé': impact_decorrelé,
                'direction_brute': row['direction'],
                'direction_decorrelée': direction_decorrelée
            })

# Création du DataFrame de résultats
decorrelation_df = pd.DataFrame(decorrelation_results)
print(f"Nombre de points d'impact décorrélés: {len(decorrelation_df)}")

# Enregistrement des résultats dans un CSV
chemin_decorr = os.path.join(OUTPUT_DIR, 'impact_decorrelé.csv')
decorrelation_df.to_csv(chemin_decorr, index=False)
print(f"Résultats de décorrélation enregistrés dans {chemin_decorr}")

log_time(start_time, "Analyse de décorrélation terminée")

# Analyse comparative des impacts bruts vs décorrélés
print("\nAnalyse comparative des impacts bruts vs décorrélés...")
start_time = time.time()

# Répartition des directions d'impact avant et après décorrélation
impact_counts_before = decorrelation_df.groupby(['fenetre_mois', 'direction_brute']).size().unstack(fill_value=0)
impact_counts_after = decorrelation_df.groupby(['fenetre_mois', 'direction_decorrelée']).size().unstack(fill_value=0)

# Calcul des changements de direction
direction_changes = []
for _, row in decorrelation_df.iterrows():
    if row['direction_brute'] != row['direction_decorrelée']:
        direction_changes.append({
            'fond': row['fond'],
            'projet': row['projet'],
            'metrique': row['metrique'],
            'fenetre_mois': row['fenetre_mois'],
            'direction_brute': row['direction_brute'],
            'direction_decorrelée': row['direction_decorrelée'],
            'variation_pct': row['variation_pct'],
            'impact_decorrelé': row['impact_decorrelé'],
            'btc_return': row['btc_return']
        })

direction_changes_df = pd.DataFrame(direction_changes)
print(f"Nombre de changements de direction après décorrélation: {len(direction_changes_df)}")

# Enregistrement des changements de direction dans un CSV
chemin_changes = os.path.join(OUTPUT_DIR, 'changements_direction.csv')
direction_changes_df.to_csv(chemin_changes, index=False)
print(f"Changements de direction enregistrés dans {chemin_changes}")

log_time(start_time, "Analyse comparative terminée")

# VISUALISATIONS
print("\nCréation des visualisations...")
start_time_viz = time.time()

# 1. Répartition des directions d'impact avant et après décorrélation
print("1. Création des graphiques de répartition des impacts...")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Configuration des couleurs
colors = {'Positive': '#4CAF50', 'Stable': '#9E9E9E', 'Négative': '#F44336'}

# Avant décorrélation (graphique de barres empilées)
bar_width = 0.25
index = np.arange(len(fenetres))

# Vérifier les colonnes disponibles dans impact_counts_before
pos_col = 'Positive' if 'Positive' in impact_counts_before.columns else None
neg_col = 'Négative' if 'Négative' in impact_counts_before.columns else None
stable_col = 'Stable' if 'Stable' in impact_counts_before.columns else None

# Tracer les barres empilées pour les données avant décorrélation
bottom = np.zeros(len(fenetres))
for i, fenetre in enumerate(fenetres):
    if fenetre in impact_counts_before.index:
        # Positif
        if pos_col and pos_col in impact_counts_before.columns:
            axes[0].bar(i, impact_counts_before.loc[fenetre, pos_col], bar_width, 
                       bottom=bottom[i], color=colors['Positive'], 
                       label='Positif' if i == 0 else "")
            bottom[i] += impact_counts_before.loc[fenetre, pos_col]
        # Stable
        if stable_col and stable_col in impact_counts_before.columns:
            axes[0].bar(i, impact_counts_before.loc[fenetre, stable_col], bar_width, 
                       bottom=bottom[i], color=colors['Stable'], 
                       label='Stable' if i == 0 else "")
            bottom[i] += impact_counts_before.loc[fenetre, stable_col]
        # Négatif
        if neg_col and neg_col in impact_counts_before.columns:
            axes[0].bar(i, impact_counts_before.loc[fenetre, neg_col], bar_width, 
                       bottom=bottom[i], color=colors['Négative'], 
                       label='Négatif' if i == 0 else "")

axes[0].set_title('Répartition des impacts BRUTS', fontsize=16)
axes[0].set_xlabel('Mois après sortie', fontsize=14)
axes[0].set_ylabel('Nombre de métriques', fontsize=14)
axes[0].set_xticks(index)
axes[0].set_xticklabels(fenetres)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Après décorrélation (mêmes barres empilées)
bottom = np.zeros(len(fenetres))
for i, fenetre in enumerate(fenetres):
    if fenetre in impact_counts_after.index:
        # Positif
        if pos_col and pos_col in impact_counts_after.columns:
            axes[1].bar(i, impact_counts_after.loc[fenetre, pos_col], bar_width, 
                       bottom=bottom[i], color=colors['Positive'], 
                       label='Positif' if i == 0 else "")
            bottom[i] += impact_counts_after.loc[fenetre, pos_col]
        # Stable
        if stable_col and stable_col in impact_counts_after.columns:
            axes[1].bar(i, impact_counts_after.loc[fenetre, stable_col], bar_width, 
                       bottom=bottom[i], color=colors['Stable'], 
                       label='Stable' if i == 0 else "")
            bottom[i] += impact_counts_after.loc[fenetre, stable_col]
        # Négatif
        if neg_col and neg_col in impact_counts_after.columns:
            axes[1].bar(i, impact_counts_after.loc[fenetre, neg_col], bar_width, 
                       bottom=bottom[i], color=colors['Négative'], 
                       label='Négatif' if i == 0 else "")

axes[1].set_title('Répartition des impacts DÉCORRÉLÉS', fontsize=16)
axes[1].set_xlabel('Mois après sortie', fontsize=14)
axes[1].set_ylabel('Nombre de métriques', fontsize=14)
axes[1].set_xticks(index)
axes[1].set_xticklabels(fenetres)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
chemin_fig = os.path.join(FIGURES_DIR, 'tableau2_directions_impact.png')
plt.savefig(chemin_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"Graphique enregistré: {chemin_fig}")

# Terminer l'analyse et afficher le temps total
log_time(start_time_total, "\nAnalyse terminée avec succès!")
print(f"Tous les fichiers de sortie sont dans {OUTPUT_DIR}")
print(f"Les visualisations sont dans {FIGURES_DIR}") 