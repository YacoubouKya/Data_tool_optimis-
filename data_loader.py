# Chargement de fichiers
# data_loader.py

# modules/data_loader.py
import pandas as pd
import io
import streamlit as st
from typing import Optional, Union

def load_file(uploaded_file, sep: Optional[str] = None, sheet_name: Optional[Union[str, int]] = None) -> Optional[pd.DataFrame]:
    """
    Lit un fichier CSV ou Excel envoyé via Streamlit file_uploader.
    Retourne un DataFrame.
    
    Args:
        uploaded_file: Fichier téléchargé
        sep: Séparateur pour les CSV (par défaut ",")
        sheet_name: Nom ou index de la feuille Excel (None = première feuille)
    """
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()
    df = None

    try:
        if filename.endswith(('.xls', '.xlsx', '.xlsm', '.xlsb')):
            return pd.read_excel(uploaded_file, sheet_name=sheet_name)
            
        elif filename.endswith(('.csv', '.txt', '.tsv')):
            try:
                # Essayer avec le séparateur fourni
                if sep is None:
                    sep = ','  # Valeur par défaut
                
                # Réinitialiser le pointeur du fichier
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep=sep)
                
            except Exception as e:
                st.error("❌ Erreur lors du chargement du fichier.")
                st.warning("ℹ️ Le séparateur actuel ne semble pas correct. Veuillez sélectionner le bon séparateur dans le menu déroulant ci-dessus.")
                return None
                
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier : {str(e)}")

        return None
