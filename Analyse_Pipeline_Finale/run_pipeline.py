#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess

"""
Script d'exécution complète du pipeline d'analyse d'impact des sorties de fonds
avec décorrélation BTC et génération du tableau final des médianes.
"""

def main():
    start_time = time.time()
    
    print("=========================================================")
    print("    PIPELINE D'ANALYSE D'IMPACT DES SORTIES DE FONDS")
    print("=========================================================")
    
    # Étape 0: Traitement des métriques de projet pour créer resume_impact.csv
    print("\n📊 ÉTAPE 0: Traitement des métriques de projets")
    print("-------------------------------")
    try:
        subprocess.run([sys.executable, "process_project_metrics.py"], check=True)
        print("✅ Traitement des métriques terminé")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du traitement des métriques: {e}")
        return
    
    # Étape 1: Analyse simple des impacts et corrélations BTC
    print("\n📈 ÉTAPE 1: Analyse des impacts et corrélations BTC")
    print("-------------------------------")
    try:
        subprocess.run([sys.executable, "analyse_simple_btc.py"], check=True)
        print("✅ Analyse des impacts et corrélations BTC terminée")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'analyse des impacts: {e}")
        return
    
    # Étape 2: Décorrélation et analyse finale
    print("\n🔄 ÉTAPE 2: Décorrélation et analyse finale")
    print("-------------------------------")
    try:
        subprocess.run([sys.executable, "analyse_impact_decorrelation_btc.py"], check=True)
        print("✅ Décorrélation et analyse finale terminées")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de la décorrélation: {e}")
        return
    
    # Étape 3: Génération du tableau final avec médianes et graphique comparatif
    print("\n📋 ÉTAPE 3: Génération du tableau final et graphique comparatif")
    print("-------------------------------")
    try:
        subprocess.run([sys.executable, "generate_tableau_final_medianes.py"], check=True)
        print("✅ Tableau final et graphique comparatif générés")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de la génération du tableau final: {e}")
        return
    
    # Calcul du temps d'exécution
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    
    print("\n" + "="*60)
    print(f"    ✅ PIPELINE TERMINÉ EN {minutes} MIN {seconds} SEC")
    print("="*60)
    
    print("\nFichiers générés:")
    print("  - FinancialResults/Outputs/AnalyseSimple/resume_impact.csv")
    print("  - FinancialResults/Outputs/AnalyseBTC/correlations_btc.csv")
    print("  - FinancialResults/Outputs/AnalyseDecorelation/impact_decorrelé.csv")
    print("  - FinancialResults/Outputs/AnalyseDecorelation/changements_direction.csv")
    print("  - FinancialResults/Outputs/AnalyseDecorelation/tableau_final_medianes_perennite.csv")
    
    print("\nVisualisations générées:")
    print("  - VisualsOutput/comparaison_moyenne_vs_mediane.png")
    
    print("\n🎯 RÉSULTAT PRINCIPAL:")
    print("  📊 Tableau des médianes des 15 métriques de pérennité disponible dans:")
    print("      FinancialResults/Outputs/AnalyseDecorelation/tableau_final_medianes_perennite.csv")
    print("  📈 Graphique comparatif moyenne vs médiane disponible dans:")
    print("      VisualsOutput/comparaison_moyenne_vs_mediane.png")

if __name__ == "__main__":
    main() 