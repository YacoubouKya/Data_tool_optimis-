# Chargement de fichiers
# data_loader.py

# modules/data_loader.py
import pandas as pd
import io
import streamlit as st
from typing import Optional, Union, Dict, Any

def load_file(uploaded_file, sep: Optional[str] = None, sheet_name: Optional[Union[str, int]] = None) -> Optional[pd.DataFrame]:
    """
    Charge un fichier CSV ou Excel avec gestion des séparateurs.
    
    Args:
        uploaded_file: Fichier téléchargé via Streamlit
        sep: Séparateur à utiliser (si None, tentative de détection automatique)
        sheet_name: Nom ou index de la feuille Excel (None = première feuille)
        
    Returns:
        DataFrame chargé ou None en cas d'échec
    """
    if uploaded_file is None:
        return None
    
    filename = uploaded_file.name.lower()
    df = None
    content_str = None

    try:
        # Lire le contenu du fichier
        content = uploaded_file.read()
        
        # Si c'est un fichier Excel
        if filename.endswith(('.xls', '.xlsx', '.xlsm', '.xlsb')):
            return pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
        
        # Pour les fichiers texte (CSV, TXT, TSV)
        content_str = content.decode('utf-8', errors='ignore')
        
        # Essayer de lire avec le séparateur fourni ou détecté
        try:
            if sep is None:
                # Essayer les séparateurs courants
                for possible_sep in [',', ';', '\t', '|', ' ']:
                    try:
                        df = pd.read_csv(io.StringIO(content_str), sep=possible_sep)
                        st.success(f"✅ Fichier chargé avec succès avec le séparateur: '{possible_sep}'")
                        return df
                    except:
                        continue
                raise ValueError("Aucun séparateur standard n'a fonctionné")
            else:
                return pd.read_csv(io.StringIO(content_str), sep=sep)
                
        except Exception as e:
            # Si échec, proposer à l'utilisateur de choisir le séparateur
            st.error("❌ Impossible de charger le fichier avec les séparateurs standards.")
            st.warning("Veuillez spécifier le bon séparateur :")
            
            # Afficher un aperçu du fichier
            st.text("Aperçu des premières lignes :")
            st.code("\n".join(content_str.split('\n')[:5]))
            
            # Proposer les séparateurs courants + personnalisé
            sep_options = {
                "Virgule (,)" : ",",
                "Point-virgule (;)" : ";",
                "Tabulation" : "\t",
                "Barre verticale (|)" : "|",
                "Espace" : " ",
                "Autre (à préciser)" : "custom"
            }
            
            selected_sep = st.radio("Séparateur :", list(sep_options.keys()))
            
            if selected_sep == "Autre (à préciser)":
                custom_sep = st.text_input("Veuillez entrer le séparateur :", value=",")
                if custom_sep:
                    try:
                        df = pd.read_csv(io.StringIO(content_str), sep=custom_sep)
                        st.success(f"✅ Fichier chargé avec succès avec le séparateur personnalisé")
                        return df
                    except Exception as e:
                        st.error(f"❌ Échec avec le séparateur personnalisé : {str(e)}")
                        st.stop()
            else:
                try:
                    sep = sep_options[selected_sep]
                    df = pd.read_csv(io.StringIO(content_str), sep=sep)
                    st.success(f"✅ Fichier chargé avec succès avec le séparateur : '{sep}'")
                    return df
                except Exception as e:
                    st.error(f"❌ Échec avec le séparateur sélectionné : {str(e)}")
                    st.stop()
                    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier : {str(e)}")
        st.stop()
    
    return None