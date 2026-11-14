"""Validations préalables (état de session, DataFrame) avant chaque section."""

import pandas as pd
import streamlit as st


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
            st.error(f"**Données manquantes** : {', '.join(missing)}")
            st.info("Retournez aux sections précédentes pour compléter les étapes requises.")
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
            st.error("Aucune donnée disponible")
            st.info("Chargez d'abord vos données dans la section 'Chargement'")
        return False

    if not isinstance(df, pd.DataFrame):
        if show_message:
            st.error("Format de données invalide (doit être un DataFrame)")
        return False

    if len(df) < min_rows:
        if show_message:
            st.error(f"Pas assez de lignes (minimum: {min_rows}, actuel: {len(df)})")
            st.info("Chargez un fichier avec plus de données")
        return False

    if len(df.columns) < min_cols:
        if show_message:
            st.error(f"Pas assez de colonnes (minimum: {min_cols}, actuel: {len(df.columns)})")
        return False

    # Vérifier si le DataFrame est vide (toutes valeurs NaN)
    if df.isnull().all().all():
        if show_message:
            st.error("Le DataFrame ne contient que des valeurs manquantes")
        return False

    return True
