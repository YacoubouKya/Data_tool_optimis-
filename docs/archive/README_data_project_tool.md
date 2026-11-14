

#  Data Project Tool – Analyse, Nettoyage & Modélisation Interactive

### *Outil Streamlit complet pour l’analyse exploratoire, la détection d’anomalies, le nettoyage des données et la modélisation automatique*


##  Présentation

Ce projet est une application **Streamlit** permettant de réaliser tout le pipeline Data Science de manière interactive :

* Chargement de fichiers CSV ou Excel
* Analyse exploratoire (EDA) avec statistiques et visualisations
* Profiling automatique des données (YData Profiling)
* Détection et proposition de corrections pour anomalies (valeurs manquantes, colonnes constantes, doublons, cardinalité élevée…)
* Nettoyage et correction des données avec journalisation (log détaillé)
* Modélisation automatique pour classification ou régression
* Reporting complet avec graphiques interactifs
* Téléchargement des datasets nettoyés et du log des corrections

L’objectif : fournir un **outil tout-en-un**, prêt à l’usage pour projets académiques ou professionnels.



##  Fonctionnalités principales

### 1️⃣ Import des données

* Chargement CSV et Excel
* Sélection de la feuille et du séparateur
* Visualisation des premières lignes
* Stockage dans `st.session_state`

### 2️⃣ Analyse exploratoire (EDA)

* Dimensions et aperçu du dataset
* Statistiques descriptives (moyenne, médiane, min, max…)
* Histogrammes et visualisations de distributions
* Matrice de corrélation
* Profiling HTML dynamique avec possibilité de téléchargement

### 3️⃣ Détection et correction des anomalies

* Valeurs manquantes
* Colonnes constantes
* Valeurs infinies
* Cardinalité élevée
* Doublons purs

Pour chaque anomalie détectée :

* Description claire
* Propositions de correction (ex. imputer, supprimer ligne ou colonne, encodage)
* Application des corrections choisies par l’utilisateur
* Log détaillé des corrections
* Téléchargement des données corrigées et du log

### 4️⃣ Modélisation automatique

* Classification et régression
* Modèles intégrés : Logistic Regression, Random Forest, XGBoost, SVM, Linear Regression, ElasticNet, Gradient Boosting…
* Sélection automatique de la cible
* Création de jeux train/test
* Affichage du pipeline et des performances
* Mise à jour du `session_state` pour usage dans l’évaluation et le reporting

### 5️⃣ Reporting automatique

* Graphiques interactifs
* Matrice de confusion
* Classification report
* Courbes ROC
* Importance des variables
* Résidus pour modèles de régression
* Score final et comparaison des modèles



## 🗂️ Organisation du projet

```
project/
│
├── data_tool_app.py                 # Point d'entrée Streamlit
├── modules/
│   ├── data_loader.py     # Chargement de fichiers
│   ├── eda.py             # Analyse exploratoire
│   ├── preprocessing.py   # Détection & correction des anomalies
│   ├── modeling.py        # Modélisation automatique
│   ├── reporting.py       # Reporting et graphiques
│   └── utils/             # Fonctions utilitaires
├── requirements.txt       # Dépendances Python
└── README.md              # Documentation

```


## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/<TON_USERNAME>/<TON_REPO>.git
cd <TON_REPO>
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l’application

```bash
streamlit run data_tool_app.py
```

---

## 📦 Requirements

* Python ≥ 3.8
* Pandas
* Numpy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn
* Streamlit
* YData Profiling
* Openpyxl / XlsxWriter
* Scipy

---

## 🛠️ Compétences démontrées

* Python appliqué au Machine Learning
* Manipulation de données avec Pandas et Numpy
* Analyse exploratoire et profiling automatique
* Détection d’anomalies et corrections interactives
* Automatisation du pipeline ML (classification et régression)
* Visualisation et reporting interactif
* Déploiement d’application web avec Streamlit
* Structuration modulaire et maintenable

---

## 📝 Améliorations possibles

* Intégration d’un **dictionnaire métier** pour valider les modalités acceptées et les intervalles numériques
* AutoML complet (LazyPredict, Auto-Sklearn)
* Export PDF automatique du reporting
* Intégration MLFlow pour suivi de modèles

---

##  Auteur

**Yacoubou KOUMAI**
Étudiant en Master Ingénierie Mathématique & Actuariat
Alternant chargé d’études actuarielles – Groupe Pasteur Mutualité

---

