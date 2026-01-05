# Projet de Machine Learning pour la Biologie - Étude du Microbiote Intestinal

Ce projet a pour objectif de développer et d'évaluer plusieurs modèles de machine learning capables de prédire la présence de maladies inflammatoires de l'intestin (MII) à partir de données du microbiote (OTUs).

## 1. Contexte

Des études récentes montrent que les MII s'accompagnent de perturbations du microbiote intestinal. En analysant la composition bactérienne (via les OTUs) et certaines métadonnées des patients, nous cherchons à construire un outil de diagnostic prédictif.

Les données brutes sont fournies sous forme de fichiers `.Rdata`.

## 2. Structure du Projet

```
.
├── r_data/                      # Contient les données brutes
│   ├── data.Rdata
│   └── otu.Rdata
├── model_comparison/          # Contient les graphiques de comparaison globaux
│   ├── fold_scores_boxplot_comparison.png
│   ├── model_comparison_report.csv
│   └── roc_curves_comparison.png
├── Logistic Regression/       # Dossier de résultats pour la Régression Logistique
├── Neural Network/           # Dossier de résultats pour le Réseau de Neurones
├── Random Forest/             # Dossier de résultats pour le Random Forest
├── XGBoost/                   # Dossier de résultats pour XGBoost
├── Projet_MLB.ipynb           # Notebook principal contenant l'analyse complète
├── fonctions_utiles.py        # Fonctions pour l'entraînement et l'évaluation des modèles
├── preprocessing.py           # Fonctions pour le pré-traitement des données
├── MLB_Rapport.pdf            # Rapport complet du projet      
├── requirements.txt          # Dépendances Python du projet
└── README.md                   # Ce fichier
```

## 3. Méthodologie

L'approche suivie dans ce projet se décompose en plusieurs étapes clés, implémentées dans les scripts Python.

### Pré-traitement (`preprocessing.py`)
1.  **Nettoyage des données** : Sélection des variables pertinentes (`age`, `diagnosis`) et suppression des valeurs manquantes.
2.  **Filtrage des OTUs** : Conservation des OTUs présents dans au moins 5% des échantillons et ayant une profondeur de lecture suffisante.
3.  **Transformation CLR** : Normalisation des données d'abondance par transformation *Centered Log-Ratio* pour gérer la nature compositionnelle des données du microbiote.

### Modélisation (`fonctions_utiles.py` et `Projet_MLB.ipynb`)

Quatre types de modèles ont été entraînés et comparés :
1.  **Régression Logistique (LR)**
2.  **Random Forest (RF)**
3.  **XGBoost (XGB)**
4.  **Réseau de Neurones (NN)**

La méthodologie d'entraînement inclut :
- **Séparation des données** en ensembles d'entraînement, de validation et de test.
- **Validation croisée stratifiée** (`StratifiedKFold`) pour une évaluation robuste, notamment sur ce jeu de données déséquilibré.
- **Optimisation des hyperparamètres** avec `GridSearchCV`.
- **Calibration des probabilités** pour améliorer la fiabilité des prédictions, particulièrement pour les modèles complexes.
- **Réduction de dimensionnalité** optionnelle basée sur l'importance des features.
- **Évaluation** basée sur des métriques multiples (AUC, Rappel, F1-score) et la recherche d'un seuil de décision optimal.

## 4. Résultats Clés

- La **Régression Logistique** s'est avérée être le modèle le plus performant, offrant le meilleur compromis entre la capacité à détecter les cas de maladie (rappel élevé) et une bonne performance globale (AUC de 0.838).
- La **calibration** a été une étape cruciale pour améliorer la fiabilité des probabilités des modèles ensemblistes (RF, XGBoost) et du réseau de neurones.
- La **réduction de dimensionnalité** a montré qu'elle permettait d'obtenir des modèles plus stables et généralisables en validation croisée.

## 5. Installation et Utilisation

### Prérequis
- Python 3.8+
- Les dépendances listées dans `requirements.txt`.

### Installation

Clonez le projet et installez les dépendances nécessaires :
```bash
git clone https://github.com/latall1017/MLB_PROJECT.git
pip install -r requirements.txt
```

### Exécution

Pour reproduire l'analyse, ouvrez et exécutez les cellules du notebook `Projet_MLB.ipynb` dans un environnement Jupyter.

```bash
jupyter notebook Projet_MLB.ipynb
```

## 6. Auteurs

- **Loïc Ledouble** (SNS)
- **Abdoulaye Tall** (SNS)
- **Ilona Richard** (CSM)