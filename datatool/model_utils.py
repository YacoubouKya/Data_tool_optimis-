"""Fonctions communes à l'affinage et à la comparaison de modèles."""

import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
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
        st.markdown("### Validation des Données")

    # Vérifier les valeurs manquantes
    y_missing = y.isna().sum()
    valid_idx = pd.Series([True] * len(y), index=y.index)

    if y_missing > 0:
        st.warning(f"Variable cible contient {y_missing} valeurs manquantes ({y_missing/len(y)*100:.1f}%)")

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
            st.success(f"{y_missing} lignes supprimées. Nouvelles dimensions : {len(y)} lignes")
        else:  # Imputer
            if y.dtype in ['object', 'category']:
                mode_val = y.mode()[0] if not y.mode().empty else y.dropna().iloc[0]
                y = y.fillna(mode_val)
                st.success(f"Valeurs manquantes imputées avec le mode : {mode_val}")
            else:
                mean_val = y.mean()
                y = y.fillna(mean_val)
                st.success(f"Valeurs manquantes imputées avec la moyenne : {mean_val:.2f}")

    # Vérifier les valeurs infinies (pour régression)
    if y.dtype in ['int64', 'float64']:
        y_inf = (~y.isna() & ((y == float('inf')) | (y == float('-inf')))).sum()
        if y_inf > 0:
            st.warning(f"Variable cible contient {y_inf} valeurs infinies")
            y = y.replace([float('inf'), float('-inf')], pd.NA)
            y = y.fillna(y.median())
            st.success("Valeurs infinies remplacées par la médiane")

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
    # Déterminer l'index par défaut (0=Classification, 1=Régression)
    default_index = 0 if auto_task == "classification" else 1

    task_choice = st.radio(
        "Type de modélisation",
        options=["Classification", "Régression"],
        index=default_index,
        key=f"task_type_{key_suffix}",
        horizontal=True
    )

    # Indication simple si modifié (en dessous du radio)
    if task_choice.lower() != auto_task:
        st.caption(f"Modifié (détection auto: {auto_task})")

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


def build_preprocessor(X: pd.DataFrame, do_scale: bool = True, use_target_encoding: bool = True) -> ColumnTransformer:
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

    # Gestion des variables catégorielles
    transformers = []

    # 1. Colonnes numériques
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))

    # 2. Colonnes catégorielles
    if cat_cols:
        # Séparation basse/élevée cardinalité
        low_card_cols = [col for col in cat_cols if X[col].nunique() <= 100]
        high_card_cols = [col for col in cat_cols if X[col].nunique() > 100]

        # Pipeline pour basse cardinalité (OneHot)
        if low_card_cols:
            cat_steps_low = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=100))
            ])
            transformers.append(("cat_low", cat_steps_low, low_card_cols))

        # Pipeline pour haute cardinalité (Target Encoding)
        if high_card_cols and not use_target_encoding:
            st.warning(f"Colonnes à haute cardinalité ignorées : {', '.join(high_card_cols)}")
        elif high_card_cols:
            st.warning(f"Colonnes à haute cardinalité détectées : {', '.join(high_card_cols)}")
            st.info("Utilisation de Target Encoding pour ces variables")

            cat_steps_high = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("target_enc", TargetEncoder())
            ])
            transformers.append(("cat_high", cat_steps_high, high_card_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

def store_model_in_session(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task: str,
    model_name: Optional[str] = None,
    metrics_dict: Optional[dict] = None,
    is_comparison: bool = False
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
        is_comparison: True si vient de la comparaison de modèles
    """
    # Stocker les données communes
    st.session_state.update({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "task_type": task
    })

    if is_comparison:
        # Cas de la comparaison : stocker dans best_model
        st.session_state.update({
            "best_model": model,
            "best_model_name": model_name or "Modèle",
            "model": None,  # Vider le modèle affiné précédent
            "current_model_name": None
        })
    else:
        # Cas de l'affinage : stocker dans model
        st.session_state.update({
            "model": model,
            "current_model_name": model_name or "Modèle affiné"
        })

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
            except Exception:
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
