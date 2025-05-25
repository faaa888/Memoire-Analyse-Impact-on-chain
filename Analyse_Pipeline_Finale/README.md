# Pipeline d'Analyse d'Impact des Sorties de Fonds

Ce répertoire contient un pipeline complet pour analyser l'impact des sorties de fonds d'investissement sur les métriques des projets de cryptomonnaies, avec une décorrélation de l'effet Bitcoin.

## Structure des Fichiers

```
Analyse_Pipeline_Finale/
├── process_project_metrics.py         # Script de traitement des métriques de projets
├── analyse_simple_btc.py              # Script de calcul des corrélations BTC
├── analyse_impact_decorrelation_btc.py    # Script d'analyse de décorrélation BTC
├── generate_impact_visuals.py             # Script de génération des visualisations
├── run_pipeline.py                        # Script d'exécution automatique du pipeline
├── Onchain_exits_dynamic.csv              # Données des sorties de fonds
├── Analysis/                              # Données historiques
│   └── Bitcoin_04_07_2010-03_09_2010_historical_data_coinmarketcap.csv
├── FinancialResults/
│   └── Outputs/
│       ├── *.csv                          # Fichiers métriques par projet
│       ├── AnalyseBTC/                    # Corrélations BTC
│       ├── AnalyseSimple/                 # Résultats d'impact brut
│       └── AnalyseDecorelation/           # Résultats d'impact décorrélé
│           └── figures/                   # Visualisations intermédiaires
└── VisualsOutput/                         # Visualisations finales
```

## Étapes du Pipeline

### 1. Traitement des métriques de projets
Script: `process_project_metrics.py`

Ce script traite les fichiers de métriques individuels des projets et les combine avec les données des sorties de fonds pour calculer l'impact brut des sorties sur différentes métriques et fenêtres temporelles.

Output: `FinancialResults/Outputs/AnalyseSimple/resume_impact.csv`

### 2. Calcul des corrélations avec Bitcoin
Script: `analyse_simple_btc.py`

Ce script analyse la relation entre les variations des métriques des projets et les variations du Bitcoin. Il calcule des coefficients de corrélation pour chaque paire métrique/fenêtre temporelle.

Output: `FinancialResults/Outputs/AnalyseBTC/correlations_btc.csv`

### 3. Analyse de l'impact avec décorrélation BTC
Script: `analyse_impact_decorrelation_btc.py`

Ce script utilise les corrélations calculées précédemment pour neutraliser l'effet du Bitcoin sur les variations des métriques et obtenir l'impact intrinsèque des sorties de fonds.

Outputs:
- `FinancialResults/Outputs/AnalyseDecorelation/impact_decorrelé.csv`
- `FinancialResults/Outputs/AnalyseDecorelation/changements_direction.csv`

### 4. Génération des visualisations finales
Script: `generate_impact_visuals.py`

Ce script génère les visualisations finales qui résument l'impact des sorties de fonds sur les métriques des projets, après décorrélation de l'effet Bitcoin.

Outputs:
- `VisualsOutput/tableau2_directions_impact.png`
- `VisualsOutput/small_multiples_metrics.png`
- `VisualsOutput/impact_visuals.pdf`

## Exécution du Pipeline

Pour exécuter l'ensemble du pipeline en une seule commande :

```
python run_pipeline.py
```

## Résultats

Les résultats finaux sont disponibles dans le répertoire `VisualsOutput/`, notamment :

1. `tableau2_directions_impact.png` : Comparaison des directions d'impact avant/après décorrélation BTC
2. `small_multiples_metrics.png` : Évolution temporelle des métriques clés après décorrélation BTC
3. `impact_visuals.pdf` : Document PDF regroupant toutes les visualisations

## Interprétation

- **Répartition des Impacts** : Montre comment la distribution des impacts (positif/neutre/négatif) change après décorrélation avec BTC
- **Small Multiples** : Présente l'évolution temporelle des métriques clés après les sorties de fonds

Cette analyse permet d'isoler l'effet spécifique des sorties de fonds d'investissement sur les projets crypto, en neutralisant l'influence du marché global (via le Bitcoin). 