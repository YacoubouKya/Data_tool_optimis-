#  Data Project Tool

**Outil interactif d'analyse de données et de modélisation Machine Learning**

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/streamlit-1.30.0-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Démo en ligne

👉 **https://data-tool-koumai.streamlit.app/**

> **Note** : Cette application est déployée sur Streamlit Cloud et accessible sans installation.

---

## Table des matières

- [Présentation](#présentation)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Dépendances](#dépendances)
- [Troubleshooting](#troubleshooting)
- [Contribution](#contribution)

---

##  Présentation

**Data Project Tool** est une application web interactive construite avec Streamlit qui permet de :
-  Charger des données (CSV, Excel)
-  Effectuer une analyse exploratoire automatique
-  Détecter et corriger les anomalies
-  Entraîner des modèles de Machine Learning
-  Évaluer les performances
-  Générer des rapports HTML consolidés

---

##  Fonctionnalités

### 1.  Chargement de données
- Support CSV (avec choix du séparateur)
- Support Excel (avec sélection de feuille)
- Gestion automatique des encodages
- Aperçu immédiat des données

### 2.  Analyse exploratoire (EDA)
- Profiling automatique avec **ydata-profiling**
- Statistiques descriptives complètes
- Histogrammes interactifs
- Matrice de corrélation
- Export du rapport HTML

### 3.  Prétraitement intelligent
- **Détection automatique** des anomalies :
  - Valeurs manquantes
  - Doublons
  - Colonnes constantes
  - Valeurs infinies
  - Cardinalité élevée
- **Corrections proposées** :
  - Imputation (moyenne, médiane, mode)
  - Suppression de lignes/colonnes
  - Remplacement des valeurs infinies
- **Log détaillé** des modifications
- Export des données corrigées

### 4.  Modélisation Machine Learning
- **Auto-détection** du type de tâche (classification/régression)
- **Modèles disponibles** :
  - Random Forest
  - Gradient Boosting
  - Régression Linéaire/Logistique
- **Pipeline complet** :
  - Preprocessing automatique
  - Imputation des valeurs manquantes
  - Standardisation des variables numériques
  - Encodage One-Hot des variables catégorielles
- **Hyperparamètres personnalisables**
- Sauvegarde automatique des modèles

### 5.  Évaluation
#### Classification
- Métriques : Accuracy, F1-score, Precision, Recall
- Matrice de confusion
- Courbe ROC (binaire)
- Courbe Précision-Rappel

#### Régression
- Métriques : MSE, RMSE, R²
- Graphique Prédit vs Réel
- Analyse des résidus
- QQ-plot

### 6.  Reporting
- Rapport HTML consolidé
- Toutes les sections du workflow
- Graphiques intégrés (base64)
- Téléchargement direct

---

##  Installation

### Prérequis
- **Python 3.9 ou supérieur**
- **pip** (gestionnaire de paquets Python)
- **PowerShell** (pour Windows)

### Méthode 1 : Script automatique (Recommandé)

1. **Ouvrir PowerShell** dans le dossier du projet
2. **Exécuter le script de lancement** :
   ```powershell
   .\launch.ps1
   ```

Le script va automatiquement :
- ✅ Vérifier Python
- ✅ Créer un environnement virtuel
- ✅ Installer les dépendances
- ✅ Lancer l'application

### Méthode 2 : Installation manuelle

```powershell
# 1. Créer un environnement virtuel
python -m venv venv

# 2. Activer l'environnement virtuel
.\venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r modules\requirements.txt

# 4. Lancer l'application
streamlit run modules\data_tool_app.py
```

### ⚠️ Problème de permissions PowerShell ?

Si vous obtenez une erreur d'exécution de script :
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📖 Utilisation

### Démarrage rapide

1. **Lancer l'application** :
   ```powershell
   .\launch.ps1
   ```

2. **Accéder à l'interface** :
   - Ouvrir votre navigateur
   - Aller sur `http://localhost:8501`

3. **Workflow recommandé** :
   ```
    Chargement →  EDA →  Prétraitement →  Modélisation →  Évaluation →  Reporting
   ```

### Exemple avec vos données

```python
# 1. Charger le fichier Excel
# Aller dans " Chargement"
# Uploader "Portefeuille AGMF_prev.xlsx"
# Sélectionner la feuille désirée

# 2. Explorer les données
# Aller dans " EDA"
# Cliquer sur "Générer le rapport de Profiling"
# Analyser les statistiques et visualisations

# 3. Nettoyer les données
# Aller dans " Prétraitement"
# Sélectionner les corrections à appliquer
# Télécharger la base corrigée

# 4. Entraîner un modèle
# Aller dans " Modélisation"
# Choisir la variable cible
# Configurer les hyperparamètres
# Lancer l'entraînement

# 5. Évaluer le modèle
# Aller dans " Évaluation"
# Consulter les métriques
# Analyser les graphiques

# 6. Générer le rapport
# Aller dans " Reporting"
# Créer le rapport HTML
# Télécharger le rapport
```

---

##  Architecture

```
Data Tool/
│
├── modules/                      # Modules principaux
│   ├── __init__.py              # Initialisation du package
│   ├── data_tool_app.py         # Application Streamlit
│   ├── data_loader.py           # Chargement de fichiers
│   ├── eda.py                   # Analyse exploratoire
│   ├── preprocessing.py         # Prétraitement
│   ├── modeling.py              # Modélisation ML
│   ├── evaluation.py            # Évaluation
│   ├── reporting.py             # Génération de rapports
│   ├── requirements.txt         # Dépendances
│   └── utils/                   # Utilitaires
│       ├── __init__.py
│       ├── helpers.py           # Fonctions helper
│       └── metrics.py           # Métriques ML
│
├── outputs/                     # Fichiers générés
│   ├── models/                  # Modèles sauvegardés (.pkl)
│   ├── data/                    # Datasets train/test
│   └── reports/                 # Rapports HTML
│
├── launch.ps1                   # Script de lancement
├── README.md                    # Ce fichier
└── DIAGNOSTIC.md                # Diagnostic technique
```

---

##  Dépendances

### Core
- `streamlit==1.30.0` - Interface web
- `pandas==2.1.1` - Manipulation de données
- `numpy==1.24.3` - Calculs numériques
- `openpyxl==3.1.2` - Lecture Excel

### Machine Learning
- `scikit-learn==1.3.2` - Modèles ML
- `xgboost==2.0.3` - Gradient Boosting
- `lightgbm==4.1.0` - Gradient Boosting

### Visualisation
- `matplotlib==3.8.0` - Graphiques
- `seaborn==0.12.3` - Graphiques statistiques

### Profiling
- `ydata-profiling==4.5.1` - Profiling automatique
- `streamlit-pandas-profiling==0.0.4` - Intégration Streamlit

### Statistiques
- `statsmodels==0.14.0` - Modèles statistiques
- `scipy==1.11.4` - Fonctions scientifiques

### Utilitaires
- `joblib==1.3.2` - Sauvegarde de modèles
- `jinja2==3.1.2` - Templates
- `xlsxwriter==3.1.9` - Export Excel

---

##  Troubleshooting

### Problème : Import Error

```
ModuleNotFoundError: No module named 'modules'
```

**Solution** : Les fichiers `__init__.py` ont été créés. Relancez l'application.

---

### Problème : Requirements introuvable

```
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'modules\requirments.txt'
```

**Solution** : Le fichier `requirements.txt` (corrigé) a été créé dans `modules/`. Utilisez :
```powershell
pip install -r modules\requirements.txt
```

---

### Problème : Versions incompatibles

```
ERROR: Cannot install scikit-learn==1.7.1 because these package versions have conflicting dependencies.
```

**Solution** : Le nouveau fichier `requirements.txt` utilise des versions compatibles testées.

---

### Problème : Application ne démarre pas

```
streamlit : Le terme 'streamlit' n'est pas reconnu...
```

**Solution** : Activez l'environnement virtuel :
```powershell
.\venv\Scripts\Activate.ps1
```

---

### Problème : Profiling très lent

**Solution** : Dans `eda.py`, ligne 12, remplacez :
```python
profile = ProfileReport(df, title="Profiling EDA", explorative=True)
```
par :
```python
profile = ProfileReport(df, title="Profiling EDA", minimal=True)
```

---

### Problème : Division par zéro

```
ZeroDivisionError: division by zero
```

**Solution** : Vérifiez que votre dataset n'est pas vide et que la variable cible contient des valeurs.

---

## Contribution

### Bugs identifiés

Consultez `DIAGNOSTIC.md` pour la liste complète des bugs et améliorations.

### Priorités de développement

1. **Critique** : Corriger division par zéro dans `modeling.py`
2. **Majeur** : Gérer les NaN dans `evaluation.py`
3. **Mineur** : Ajouter tests unitaires
4. **Nice to have** : Support Parquet, JSON

---

##  License

MIT License - Libre d'utilisation et de modification

---

## Auteur

**Data Tool Team**  
Version 1.0.0 - Novembre 2024

---

## 📞 Support

Pour toute question ou problème :
1. Consultez `DIAGNOSTIC.md` pour les problèmes connus
2. Vérifiez que toutes les dépendances sont installées
3. Assurez-vous d'utiliser Python 3.9+

---

##  Remerciements

- **Streamlit** pour le framework web
- **scikit-learn** pour les modèles ML
- **ydata-profiling** pour le profiling automatique
- **Pandas** pour la manipulation de données

---




