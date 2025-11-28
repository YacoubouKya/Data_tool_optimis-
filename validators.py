"""
Module de validation préventive pour l'application Streamlit
Fournit des fonctions pour valider les données avant traitement
"""

import streamlit as st
import pandas as pd
import numpy as np


def validate_session_state(required_keys, show_message=True):
    """
    Vérifie que les clés requises existent dans session_state
    
    Args:
        required_keys: Liste des clés requises
        show_message: Si True, affiche un message d'erreur
        
    Returns:
        bool: True si toutes les clés existent, False sinon
    """
    missing = [key for key in required_keys if key not in st.session_state]
    
    if missing:
        if show_message:
            st.error(f"❌ **Données manquantes** : {', '.join(missing)}")
            st.info("💡 Retournez aux sections précédentes pour compléter les étapes requises.")
        return False
    
    return True


def validate_dataframe(df, min_rows=1, min_cols=1, show_message=True):
    """
    Vérifie qu'un DataFrame est valide pour le traitement
    
    Args:
        df: DataFrame à valider
        min_rows: Nombre minimum de lignes requis
        min_cols: Nombre minimum de colonnes requis
        show_message: Si True, affiche un message d'erreur
        
    Returns:
        bool: True si le DataFrame est valide, False sinon
    """
    if df is None:
        if show_message:
            st.error("❌ Aucune donnée disponible")
            st.info("💡 Chargez d'abord vos données dans la section 'Chargement'")
        return False
    
    if not isinstance(df, pd.DataFrame):
        if show_message:
            st.error("❌ Format de données invalide (doit être un DataFrame)")
        return False
    
    if len(df) < min_rows:
        if show_message:
            st.error(f"❌ Pas assez de lignes (minimum: {min_rows}, actuel: {len(df)})")
            st.info("💡 Chargez un fichier avec plus de données")
        return False
    
    if len(df.columns) < min_cols:
        if show_message:
            st.error(f"❌ Pas assez de colonnes (minimum: {min_cols}, actuel: {len(df.columns)})")
        return False
    
    # Vérifier si le DataFrame est vide (toutes valeurs NaN)
    if df.isnull().all().all():
        if show_message:
            st.error("❌ Le DataFrame ne contient que des valeurs manquantes")
        return False
    
    return True


def validate_target_column(df, target_col, show_message=True):
    """
    Vérifie qu'une colonne cible existe et est valide
    
    Args:
        df: DataFrame contenant les données
        target_col: Nom de la colonne cible
        show_message: Si True, affiche un message d'erreur
        
    Returns:
        bool: True si la colonne est valide, False sinon
    """
    if target_col is None or target_col == "":
        if show_message:
            st.error("❌ Aucune variable cible sélectionnée")
        return False
    
    if target_col not in df.columns:
        if show_message:
            st.error(f"❌ La colonne '{target_col}' n'existe pas dans les données")
            st.info(f"💡 Colonnes disponibles : {', '.join(df.columns.tolist())}")
        return False
    
    # Vérifier que la colonne n'est pas entièrement vide
    if df[target_col].isnull().all():
        if show_message:
            st.error(f"❌ La colonne '{target_col}' ne contient que des valeurs manquantes")
        return False
    
    # Vérifier qu'il y a au moins 2 valeurs uniques (pour la classification)
    if df[target_col].nunique() < 2:
        if show_message:
            st.warning(f"⚠️ La colonne '{target_col}' ne contient qu'une seule valeur unique")
            st.info("💡 Une variable cible doit avoir au moins 2 valeurs différentes")
        return False
    
    return True


def validate_model_data(X_train, X_test, y_train, y_test, show_message=True):
    """
    Vérifie que les données de modélisation sont valides
    
    Args:
        X_train, X_test: Features d'entraînement et de test
        y_train, y_test: Cibles d'entraînement et de test
        show_message: Si True, affiche un message d'erreur
        
    Returns:
        bool: True si les données sont valides, False sinon
    """
    # Vérifier que les données existent
    if any(x is None for x in [X_train, X_test, y_train, y_test]):
        if show_message:
            st.error("❌ Données de modélisation manquantes")
            st.info("💡 Préparez d'abord les données dans la section appropriée")
        return False
    
    # Vérifier les dimensions
    if len(X_train) == 0 or len(X_test) == 0:
        if show_message:
            st.error("❌ Les données d'entraînement ou de test sont vides")
        return False
    
    # Vérifier la cohérence des longueurs
    if len(X_train) != len(y_train):
        if show_message:
            st.error(f"❌ Incohérence : X_train ({len(X_train)} lignes) != y_train ({len(y_train)} lignes)")
        return False
    
    if len(X_test) != len(y_test):
        if show_message:
            st.error(f"❌ Incohérence : X_test ({len(X_test)} lignes) != y_test ({len(y_test)} lignes)")
        return False
    
    # Vérifier qu'il n'y a pas de valeurs infinies
    if isinstance(X_train, pd.DataFrame):
        if np.isinf(X_train.select_dtypes(include=[np.number])).any().any():
            if show_message:
                st.error("❌ Les données contiennent des valeurs infinies")
                st.info("💡 Nettoyez les données avant la modélisation")
            return False
    
    return True


def validate_model_exists(show_message=True):
    """
    Vérifie qu'un modèle a été entraîné et existe dans la session
    
    Args:
        show_message: Si True, affiche un message d'erreur
        
    Returns:
        bool: True si un modèle existe, False sinon
    """
    has_model = "model" in st.session_state or "best_model" in st.session_state
    
    if not has_model and show_message:
        st.error("❌ Aucun modèle entraîné disponible")
        st.info("""
        💡 **Pour obtenir un modèle :**
        
        **Option 1 (Recommandée)** : Section "🔬 Comparaison de Modèles"
        - Comparez plusieurs modèles automatiquement
        - Le meilleur sera sélectionné
        
        **Option 2** : Section "🎯 Affinage de Modèle"
        - Configurez et entraînez un modèle spécifique
        """)
    
    return has_model


def validate_file_upload(uploaded_file, allowed_extensions=None, show_message=True):
    """
    Vérifie qu'un fichier uploadé est valide
    
    Args:
        uploaded_file: Fichier uploadé via st.file_uploader
        allowed_extensions: Liste des extensions autorisées (ex: ['.csv', '.xlsx'])
        show_message: Si True, affiche un message d'erreur
        
    Returns:
        bool: True si le fichier est valide, False sinon
    """
    if uploaded_file is None:
        if show_message:
            st.warning("⚠️ Aucun fichier sélectionné")
        return False
    
    if allowed_extensions:
        file_ext = '.' + uploaded_file.name.split('.')[-1].lower()
        if file_ext not in allowed_extensions:
            if show_message:
                st.error(f"❌ Extension de fichier non autorisée : {file_ext}")
                st.info(f"💡 Extensions autorisées : {', '.join(allowed_extensions)}")
            return False
    
    # Vérifier la taille du fichier (limite à 200 MB)
    max_size = 200 * 1024 * 1024  # 200 MB
    if uploaded_file.size > max_size:
        if show_message:
            st.error(f"❌ Fichier trop volumineux : {uploaded_file.size / (1024*1024):.1f} MB")
            st.info("💡 Taille maximale autorisée : 200 MB")
        return False
    
    return True


def validate_numeric_input(value, min_val=None, max_val=None, param_name="Paramètre", show_message=True):
    """
    Vérifie qu'une valeur numérique est dans une plage valide
    
    Args:
        value: Valeur à valider
        min_val: Valeur minimale autorisée
        max_val: Valeur maximale autorisée
        param_name: Nom du paramètre (pour les messages)
        show_message: Si True, affiche un message d'erreur
        
    Returns:
        bool: True si la valeur est valide, False sinon
    """
    if value is None:
        if show_message:
            st.error(f"❌ {param_name} : valeur manquante")
        return False
    
    if not isinstance(value, (int, float)):
        if show_message:
            st.error(f"❌ {param_name} : doit être un nombre")
        return False
    
    if min_val is not None and value < min_val:
        if show_message:
            st.error(f"❌ {param_name} : doit être >= {min_val} (actuel: {value})")
        return False
    
    if max_val is not None and value > max_val:
        if show_message:
            st.error(f"❌ {param_name} : doit être <= {max_val} (actuel: {value})")
        return False
    
    return True


def check_data_quality(df, show_warnings=True):
    """
    Effectue des vérifications de qualité sur un DataFrame
    Affiche des avertissements mais ne bloque pas l'exécution
    
    Args:
        df: DataFrame à vérifier
        show_warnings: Si True, affiche des avertissements
        
    Returns:
        dict: Dictionnaire avec les résultats des vérifications
    """
    results = {
        "has_missing": False,
        "has_duplicates": False,
        "has_high_cardinality": False,
        "memory_size_mb": 0
    }
    
    if df is None or not isinstance(df, pd.DataFrame):
        return results
    
    # Vérifier les valeurs manquantes
    missing_pct = (df.isnull().sum() / len(df) * 100)
    if missing_pct.any():
        results["has_missing"] = True
        if show_warnings:
            cols_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
            st.warning(f"⚠️ {len(cols_with_missing)} colonne(s) avec valeurs manquantes")
    
    # Vérifier les doublons
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        results["has_duplicates"] = True
        if show_warnings:
            st.warning(f"⚠️ {n_duplicates} ligne(s) dupliquée(s) détectée(s)")
    
    # Vérifier la cardinalité élevée (colonnes avec beaucoup de valeurs uniques)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 100:
            results["has_high_cardinality"] = True
            if show_warnings:
                st.warning(f"⚠️ Colonne '{col}' : cardinalité élevée ({df[col].nunique()} valeurs uniques)")
    
    # Vérifier la taille en mémoire
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    results["memory_size_mb"] = memory_mb
    if memory_mb > 500 and show_warnings:
        st.warning(f"⚠️ Dataset volumineux : {memory_mb:.1f} MB en mémoire")
    
    return results
