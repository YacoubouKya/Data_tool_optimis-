# Chargement de fichiers
# data_loader.py

# modules/data_loader.py
import pandas as pd
import io
import streamlit as st
from typing import Optional, Union, Dict, Any

def detect_separator(content: str, sample_size: int = 5) -> str:
    """
    Détecte automatiquement le séparateur utilisé dans un fichier CSV.
    
    Args:
        content: Contenu du fichier sous forme de chaîne
        sample_size: Nombre de lignes à analyser pour la détection
        
    Returns:
        Le séparateur détecté (',' par défaut)
    """
    # Séparateurs courants à tester
    possible_separators = [',', ';', '\t', '|', ' ']
    lines = content.split('\n')[:sample_size]
    lines = [line for line in lines if line.strip()]  # Enlever les lignes vides
    
    if not lines:
        return ','  # Valeur par défaut si pas de lignes
    
    # Compter les occurrences de chaque séparateur
    separator_counts = {sep: 0 for sep in possible_separators}
    
    for line in lines:
        for sep in possible_separators:
            separator_counts[sep] += line.count(sep)
    
    # Trouver le séparateur le plus fréquent
    detected_sep = max(separator_counts.items(), key=lambda x: x[1])[0]
    
    # Si aucun séparateur n'est trouvé, utiliser la virgule par défaut
    return detected_sep if separator_counts[detected_sep] > 0 else ','

def load_file(uploaded_file, sep: Optional[str] = None, sheet_name: Optional[Union[str, int]] = None) -> Optional[pd.DataFrame]:
    """
    Charge un fichier CSV ou Excel avec gestion automatique du séparateur.
    
    Args:
        uploaded_file: Fichier téléchargé via Streamlit
        sep: Séparateur à utiliser (si None, détection automatique)
        sheet_name: Nom ou index de la feuille Excel (None = première feuille)
        
    Returns:
        DataFrame chargé ou None en cas d'échec
    """
    if uploaded_file is None:
        return None
    
    filename = uploaded_file.name.lower()
    df = None
    error_msg = None

    try:
        # Lire le contenu du fichier
        content = uploaded_file.read()
        
        # Si c'est un fichier Excel
        if filename.endswith(('.xls', '.xlsx', '.xlsm', '.xlsb')):
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
        # Si c'est un fichier CSV
        elif filename.endswith(('.csv', '.txt', '.tsv')):
            # Détecter le séparateur si non spécifié
            if sep is None:
                try:
                    # Essayer avec le séparateur par défaut d'abord
                    content_str = content.decode('utf-8', errors='ignore')
                    sep = detect_separator(content_str)
                    st.info(f"🔍 Séparateur détecté automatiquement : '{sep}'")
                except Exception as e:
                    st.warning("⚠️ Impossible de détecter le séparateur, utilisation de la virgule par défaut")
                    sep = ','
            
            # Essayer de lire avec le séparateur détecté
            try:
                df = pd.read_csv(io.StringIO(content_str), sep=sep, on_bad_lines='warn')
            except Exception as e:
                st.error(f"❌ Erreur lors de la lecture du fichier avec le séparateur '{sep}'. Tentative avec détection automatique...")
                # Essayer avec différents séparateurs
                for possible_sep in [',', ';', '\t', '|', ' ']:
                    if possible_sep != sep:  # Ne pas réessayer le séparateur déjà testé
                        try:
                            df = pd.read_csv(io.StringIO(content_str), sep=possible_sep)
                            st.success(f"✅ Fichier chargé avec succès avec le séparateur: '{possible_sep}'")
                            break
                        except:
                            continue
                
                if df is None:
                    raise ValueError("Impossible de charger le fichier avec les séparateurs testés")
        
        # Nettoyage des noms de colonnes
        if df is not None:
            df.columns = df.columns.str.strip()  # Enlever les espaces
            # Supprimer les colonnes vides
            df = df.dropna(axis=1, how='all')
            # Supprimer les lignes vides
            df = df.dropna(how='all')
            
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Erreur lors du chargement du fichier : {error_msg}")
        st.stop()
    
    return df