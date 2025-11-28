# modules/model_utils.py
"""
Utilitaires partagés pour la modélisation ML
Contient les fonctions communes utilisées par modeling.py et model_comparison.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional, Any


def validate_and_clean_target(y: pd.Series, target_name: str, silent: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Valide et nettoie la variable cible
    
    Args:
        y: Série pandas de la variable cible
        target_name: Nom de la variable cible
        silent: Si True, n'affiche pas le titre "Validation des Données"
        
    Returns:
        Tuple (y_cleaned, indices_to_keep)
    """
    if not silent:
        st.markdown("### 🔍 Validation des Données")
    
    # Vérifier les valeurs manquantes
    y_missing = y.isna().sum()
    valid_idx = pd.Series([True] * len(y), index=y.index)
    
    if y_missing > 0:
        st.warning(f"⚠️ Variable cible contient {y_missing} valeurs manquantes ({y_missing/len(y)*100:.1f}%)")
        
        action = st.radio(
            "Comment traiter les valeurs manquantes dans la cible ?",
            ["Supprimer les lignes", "Imputer (moyenne/mode)", "Annuler"],
            key=f"missing_target_action_{target_name}"
        )
        
        if action == "Annuler":
            st.info("Veuillez nettoyer vos données avant la modélisation")
            st.stop()
        elif action == "Supprimer les lignes":
            valid_idx = y.notna()
            y = y[valid_idx].reset_index(drop=True)
            st.success(f"✅ {y_missing} lignes supprimées. Nouvelles dimensions : {len(y)} lignes")
        else:  # Imputer
            if y.dtype in ['object', 'category']:
                mode_val = y.mode()[0] if not y.mode().empty else y.dropna().iloc[0]
                y = y.fillna(mode_val)
                st.success(f"✅ Valeurs manquantes imputées avec le mode : {mode_val}")
            else:
                mean_val = y.mean()
                y = y.fillna(mean_val)
                st.success(f"✅ Valeurs manquantes imputées avec la moyenne : {mean_val:.2f}")
    
    # Vérifier les valeurs infinies (pour régression)
    if y.dtype in ['int64', 'float64']:
        y_inf = (~y.isna() & ((y == float('inf')) | (y == float('-inf')))).sum()
        if y_inf > 0:
            st.warning(f"⚠️ Variable cible contient {y_inf} valeurs infinies")
            y = y.replace([float('inf'), float('-inf')], pd.NA)
            y = y.fillna(y.median())
            st.success(f"✅ Valeurs infinies remplacées par la médiane")
    
    return y, valid_idx


def detect_task_type(y: pd.Series, task: str = "auto") -> str:
    """
    Détecte automatiquement le type de tâche (classification ou régression)
    
    Args:
        y: Série pandas de la variable cible
        task: Type de tâche ("auto", "classification", "regression")
        
    Returns:
        Type de tâche détecté
    """
    if task == "auto":
        if y.dtype == "O" or (y.nunique() <= 20 and y.nunique()/len(y) < 0.1):
            return "classification"
        else:
            return "regression"
    return task


def select_task_type_with_ui(y: pd.Series, key_suffix: str = "") -> str:
    """
    Affiche la détection automatique et permet à l'utilisateur de changer le type de tâche
    
    Args:
        y: Série pandas de la variable cible
        key_suffix: Suffixe pour la clé du widget (pour éviter les doublons)
        
    Returns:
        Type de tâche choisi par l'utilisateur (ou détecté automatiquement)
    """
    # Détection automatique
    auto_task = detect_task_type(y, "auto")
    
    # Interface compacte et épurée
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Déterminer l'index par défaut (0=Classification, 1=Régression)
        default_index = 0 if auto_task == "classification" else 1
        
        task_choice = st.radio(
            "🎯 Type de modélisation",
            options=["Classification", "Régression"],
            index=default_index,
            key=f"task_type_{key_suffix}",
            horizontal=True
        )
    
    with col2:
        # Indication simple si modifié
        if task_choice.lower() != auto_task:
            st.caption(f"🔄 Modifié (auto: {auto_task})")
    
    return task_choice.lower()


def display_target_stats(y: pd.Series, task: str):
    """
    Affiche les statistiques de la variable cible
    
    Args:
        y: Série pandas de la variable cible
        task: Type de tâche
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Lignes valides", len(y))
    
    with col2:
        st.metric("Valeurs uniques", y.nunique())
    
    with col3:
        if task == "regression" and y.dtype in ['int64', 'float64']:
            st.metric("Moyenne", f"{y.mean():.2f}")
        else:
            mode_val = y.mode()[0] if not y.mode().empty else "N/A"
            st.metric("Mode", mode_val)


def build_preprocessor(X: pd.DataFrame, do_scale: bool = True) -> ColumnTransformer:
    """
    Construit le preprocesseur pour les données
    
    Args:
        X: DataFrame des features
        do_scale: Si True, standardise les colonnes numériques
        
    Returns:
        ColumnTransformer configuré
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Convertir les catégorielles en string
    if cat_cols:
        X[cat_cols] = X[cat_cols].astype(str)
    
    # Pipeline pour les colonnes numériques
    num_steps = []
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if do_scale:
            num_steps.append(("scaler", StandardScaler()))
    
    # Pipeline pour les colonnes catégorielles avec gestion de la haute cardinalité
    cat_steps = []
    if cat_cols:
        # Filtrer les colonnes à haute cardinalité (> 100 valeurs uniques)
        low_card_cols = []
        high_card_cols = []
        
        for col in cat_cols:
            n_unique = X[col].nunique()
            if n_unique <= 100:
                low_card_cols.append(col)
            else:
                high_card_cols.append(col)
                import streamlit as st
                st.warning(f"⚠️ Colonne '{col}' ignorée : {n_unique} valeurs uniques (> 100)")
        
        # Utiliser OneHotEncoder seulement pour les colonnes à faible cardinalité
        if low_card_cols:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=100))
            ]
    
    # Assembler les transformers
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols and low_card_cols:
        transformers.append(("cat", Pipeline(cat_steps), low_card_cols))
    
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )


def prepare_data_for_modeling(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    validate_target: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """
    Prépare les données pour la modélisation
    
    Args:
        df: DataFrame complet
        target: Nom de la variable cible
        test_size: Proportion du jeu de test
        random_state: Seed pour la reproductibilité
        validate_target: Si True, valide et nettoie la cible
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test, task_type)
    """
    from sklearn.model_selection import train_test_split
    
    # Séparer X et y
    X = df.drop(columns=[target])
    y = df[target]
    
    # Validation optionnelle de la cible
    if validate_target:
        y, valid_idx = validate_and_clean_target(y, target)
        if not valid_idx.all():
            X = X[valid_idx].reset_index(drop=True)
    
    # Détecter le type de tâche
    task = detect_task_type(y)
    
    # Afficher les statistiques
    display_target_stats(y, task)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, task


def store_model_in_session(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task: str,
    model_name: Optional[str] = None,
    metrics_dict: Optional[dict] = None
):
    """
    Stocke le modèle et les données dans session_state de manière cohérente
    
    Args:
        model: Modèle entraîné (Pipeline)
        X_train, X_test: DataFrames des features
        y_train, y_test: Series des cibles
        task: Type de tâche
        model_name: Nom du modèle (optionnel)
        metrics_dict: Dictionnaire des métriques (optionnel)
    """
    st.session_state.update({
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "task_type": task
    })
    
    if model_name:
        st.session_state["best_model_name"] = model_name
    
    if metrics_dict:
        st.session_state["evaluation_metrics"] = pd.DataFrame([metrics_dict])


def format_metrics(d: dict, decimals: int = 3) -> dict:
    """
    Arrondit les valeurs numériques du dictionnaire pour l'affichage
    
    Args:
        d: Dictionnaire de métriques
        decimals: Nombre de décimales
        
    Returns:
        Dictionnaire formaté
    """
    from math import isfinite
    
    out = {}
    for k, v in d.items():
        try:
            fv = float(v)
            if not isfinite(fv):
                out[k] = v
            else:
                out[k] = round(fv, decimals)
        except Exception:
            out[k] = v
    return out


def get_fast_models(task: str) -> list:
    """
    Retourne la liste des modèles rapides pour un démarrage rapide
    
    Args:
        task: Type de tâche
        
    Returns:
        Liste des noms de modèles rapides
    """
    if task == "classification":
        return ["Random Forest", "Logistic Regression", "Decision Tree"]
    else:
        return ["Random Forest", "Linear Regression", "Ridge"]


def display_model_info(model_name: str, task: str):
    """
    Affiche des informations sur un modèle spécifique
    
    Args:
        model_name: Nom du modèle
        task: Type de tâche
    """
    model_descriptions = {
        "Random Forest": {
            "description": "Ensemble de décision trees avec bagging",
            "pros": "Robuste, peu de tuning nécessaire",
            "cons": "Peut être lent sur gros datasets",
            "use_case": "Excellent choix par défaut"
        },
        "Gradient Boosting": {
            "description": "Ensemble séquentiel d'arbres",
            "pros": "Très performant, gagne souvent les compétitions",
            "cons": "Sensible au tuning, risque d'overfitting",
            "use_case": "Quand la performance maximale est requise"
        },
        "Logistic Regression": {
            "description": "Modèle linéaire pour classification",
            "pros": "Rapide, interprétable",
            "cons": "Limité aux relations linéaires",
            "use_case": "Baseline rapide, données linéaires"
        },
        "Linear Regression": {
            "description": "Modèle linéaire pour régression",
            "pros": "Très rapide, interprétable",
            "cons": "Limité aux relations linéaires",
            "use_case": "Baseline rapide, données linéaires"
        },
        "SVM": {
            "description": "Support Vector Machine",
            "pros": "Efficace en haute dimension",
            "cons": "Lent sur gros datasets",
            "use_case": "Petits datasets, haute dimension"
        },
        "K-Nearest Neighbors": {
            "description": "Classification/régression par proximité",
            "pros": "Simple, pas de phase d'entraînement",
            "cons": "Lent en prédiction, sensible à l'échelle",
            "use_case": "Petits datasets, patterns locaux"
        }
    }
    
    if model_name in model_descriptions:
        info = model_descriptions[model_name]
        with st.expander(f"ℹ️ À propos de {model_name}"):
            st.markdown(f"**Description** : {info['description']}")
            st.markdown(f"**✅ Avantages** : {info['pros']}")
            st.markdown(f"**⚠️ Limitations** : {info['cons']}")
            st.markdown(f"**🎯 Cas d'usage** : {info['use_case']}")


def save_model_to_disk(model: Any, target: str, model_name: str = "model") -> str:
    """
    Sauvegarde un modèle sur le disque
    
    Args:
        model: Modèle à sauvegarder
        target: Nom de la variable cible
        model_name: Nom du modèle
        
    Returns:
        Chemin du fichier sauvegardé
    """
    import joblib
    import helpers
    
    helpers.ensure_dir("outputs/models")
    model_path = f"outputs/models/{model_name}_{target}.pkl"
    joblib.dump(model, model_path)
    
    return model_path


def extract_feature_importance(pipeline: Any, X: pd.DataFrame) -> Optional[pd.Series]:
    """
    Extrait l'importance des features d'un pipeline si disponible
    
    Args:
        pipeline: Pipeline sklearn
        X: DataFrame des features
        
    Returns:
        Series avec l'importance des features ou None
    """
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            try:
                feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            except:
                num_cols = X.select_dtypes(include="number").columns.tolist()
                cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
                feature_names = num_cols + cat_cols
            
            fi = pd.Series(
                model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)
            
            return fi
    except Exception:
        pass
    
    return None
