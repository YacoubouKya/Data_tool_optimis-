# modules/preprocessing.py
# preprocessing.py

import pandas as pd
import streamlit as st
from io import BytesIO
from typing import Tuple
import sys
import os

# Ajouter le répertoire parent au path pour importer data_quality
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_quality import DataQualityChecker
    HAS_DATA_QUALITY = True
except ImportError:
    HAS_DATA_QUALITY = False

# ------------------------
# Détection anomalies (utilise profile_report)
# ------------------------
def detect_and_propose_corrections(profile_report, df: pd.DataFrame):
    desc = profile_report.get_description()
    # compatibilité selon version
    vars_desc = desc.get("variables") if isinstance(desc, dict) else getattr(desc, "variables", {})
    results = []

    # Parcours du dictionnaire produit par profiling (faible coût, petite structure)
    for col, info in vars_desc.items():
        anomalies = []
        corrections = []

        n_missing = info.get("n_missing", 0) or 0
        if n_missing > 0:
            anomalies.append(f"{n_missing} valeurs manquantes")
            corrections.extend([
                "Imputer (moyenne)",
                "Imputer (médiane)",
                "Imputer (mode)",
                "Supprimer lignes",
                "Supprimer colonne"
            ])

        n_unique = info.get("n_unique", 0) or 0
        if n_unique == 1:
            anomalies.append("Colonne constante")
            corrections.append("Supprimer colonne")

        if len(df) > 0 and n_unique > 0.5 * len(df):
            anomalies.append("Cardinalité élevée")
            corrections.append("Encodage alternatif (hashing/target)")

        n_infinite = info.get("n_infinite", 0) or 0
        if n_infinite > 0:
            anomalies.append(f"{n_infinite} valeurs infinies")
            corrections.extend(["Remplacer par NaN + imputer", "Supprimer lignes"])

        if anomalies:
            results.append({
                "colonne": col,
                "anomalies": anomalies,
                "propositions": list(dict.fromkeys(corrections))  # garde l'ordre, dédoublonne
            })

    # Doublons purs (table-level) : tenter d'extraire depuis profile_report, fallback sur df
    duplicates_count = None
    if isinstance(desc, dict):
        # plusieurs chemins possibles selon version
        duplicates_count = desc.get("table", {}).get("n_duplicates") if isinstance(desc.get("table"), dict) else None
        if duplicates_count is None:
            duplicates_count = desc.get("dataset", {}).get("n_duplicates") if isinstance(desc.get("dataset"), dict) else None
        if duplicates_count is None:
            duplicates_count = desc.get("table", {}).get("n_duplicated") if isinstance(desc.get("table"), dict) else None

    if duplicates_count is None:
        duplicates_count = int(df.duplicated().sum())

    if duplicates_count and duplicates_count > 0:
        results.append({
            "colonne": "DOUBLONS",
            "anomalies": [f"{int(duplicates_count)} doublons purs détectés"],
            "propositions": ["Supprimer doublons purs", "Conserver"]
        })

    return results


# ------------------------
# Application correction avec suivi (optimisée)
# ------------------------
def apply_corrections_with_log(df: pd.DataFrame, corrections_dict: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    corrections_dict e.g. {'col1': 'Imputer (moyenne)', 'DOUBLONS': 'Supprimer doublons purs'}
    Retourne : df corrigé, log_df
    """
    # Créer une copie explicite pour éviter SettingWithCopyWarning
    df = df.copy()
    log = []

    # Regrouper colonnes par type de correction pour appliquer vectorisé quand possible
    # Ex: imputer moyenne/mediane/mode pour plusieurs colonnes numériques
    # Construire des buckets
    buckets = {}
    for col, corr in corrections_dict.items():
        buckets.setdefault(corr, []).append(col)

    # Traitement pour doublons (dataset-level)
    if "Supprimer doublons purs" in buckets:
        cols = buckets.pop("Supprimer doublons purs")
        if "DOUBLONS" in cols:
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            log.append({
                "colonne": "DOUBLONS",
                "correction_appliquee": "Supprimer doublons purs",
                "nb_valeurs_modifiees": int(before - after)
            })

    # Imputation moyenne/mediane en vectorisé pour colonnes numériques uniquement
    # Moyenne
    if "Imputer (moyenne)" in buckets:
        cols = [c for c in buckets.pop("Imputer (moyenne)") if c in df.columns]
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            means = df[num_cols].mean()
            df[num_cols] = df[num_cols].fillna(means)
            for c in num_cols:
                log.append({"colonne": c, "correction_appliquee": "Imputer (moyenne)", "nb_valeurs_modifiees": int(df[c].isna().sum() == 0)})  # best-effort

        # non-numériques fallback: per-column mode
        cat_cols = [c for c in cols if c not in num_cols and c in df.columns]
        for c in cat_cols:
            mode_val = df[c].mode()[0] if not df[c].mode().empty else None
            before_missing = df[c].isna().sum()
            df.loc[:, c] = df[c].fillna(mode_val)
            modified = int(before_missing - df[c].isna().sum())
            log.append({"colonne": c, "correction_appliquee": "Imputer (moyenne) (fallback mode)", "nb_valeurs_modifiees": modified})

    # Imputer mediane
    if "Imputer (médiane)" in buckets:
        cols = [c for c in buckets.pop("Imputer (médiane)") if c in df.columns]
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            meds = df[num_cols].median()
            before = df[num_cols].isna().sum()
            df[num_cols] = df[num_cols].fillna(meds)
            after = df[num_cols].isna().sum()
            for c in num_cols:
                log.append({"colonne": c, "correction_appliquee": "Imputer (médiane)", "nb_valeurs_modifiees": int(before[c] - after[c])})
        # categorical fallback
        cat_cols = [c for c in cols if c not in num_cols]
        for c in cat_cols:
            mode_val = df[c].mode()[0] if not df[c].mode().empty else None
            before_missing = df[c].isna().sum()
            df.loc[:, c] = df[c].fillna(mode_val)
            log.append({"colonne": c, "correction_appliquee": "Imputer (médiane) (fallback mode)", "nb_valeurs_modifiees": int(before_missing - df[c].isna().sum())})

    # Imputer mode
    if "Imputer (mode)" in buckets:
        cols = [c for c in buckets.pop("Imputer (mode)") if c in df.columns]
        for c in cols:
            before_missing = df[c].isna().sum()
            mode_val = df[c].mode()[0] if not df[c].mode().empty else None
            df.loc[:, c] = df[c].fillna(mode_val)
            log.append({"colonne": c, "correction_appliquee": "Imputer (mode)", "nb_valeurs_modifiees": int(before_missing - df[c].isna().sum())})

    # Supprimer lignes (per-column)
    if "Supprimer lignes" in buckets:
        cols = [c for c in buckets.pop("Supprimer lignes") if c in df.columns]
        for c in cols:
            before = len(df)
            df = df.dropna(subset=[c])
            after = len(df)
            log.append({"colonne": c, "correction_appliquee": "Supprimer lignes", "nb_valeurs_modifiees": int(before - after)})

    # Supprimer colonne
    if "Supprimer colonne" in buckets:
        cols = [c for c in buckets.pop("Supprimer colonne") if c in df.columns]
        for c in cols:
            df = df.drop(columns=[c])
            log.append({"colonne": c, "correction_appliquee": "Supprimer colonne", "nb_valeurs_modifiees": "colonne supprimée"})

    # Remplacer infinis + imputer
    if "Remplacer par NaN + imputer" in buckets:
        cols = [c for c in buckets.pop("Remplacer par NaN + imputer") if c in df.columns]
        for c in cols:
            before_neg = df[c].isna().sum()
            df[c] = df[c].replace([float("inf"), -float("inf")], pd.NA)
            if pd.api.types.is_numeric_dtype(df[c]):
                med = df[c].median()
                df[c] = df[c].fillna(med)
            else:
                mode_val = df[c].mode()[0] if not df[c].mode().empty else None
                df[c] = df[c].fillna(mode_val)
            log.append({"colonne": c, "correction_appliquee": "Remplacer par NaN + imputer", "nb_valeurs_modifiees": int(df[c].isna().sum() - before_neg)})

    # Encodage / autres => journaliser pour attention
    for corr, cols in list(buckets.items()):
        if corr.startswith("Encodage") or corr == "Ne pas corriger":
            for c in cols:
                log.append({"colonne": c, "correction_appliquee": corr, "nb_valeurs_modifiees": 0})
        else:
            for c in cols:
                # fallback safe: apply per-column
                if c in df.columns:
                    df_before = df[c].copy()
                    df = apply_correction(df, c, corr)
                    modified = int(df_before.ne(df[c]).sum()) if c in df.columns else "colonne supprimée"
                    log.append({"colonne": c, "correction_appliquee": corr, "nb_valeurs_modifiees": modified})
                else:
                    log.append({"colonne": c, "correction_appliquee": corr, "nb_valeurs_modifiees": "colonne absente"})

    log_df = pd.DataFrame(log)
    # formatage propre
    if "nb_valeurs_modifiees" in log_df.columns:
        # convertir en int quand possible
        def safe_int(x):
            try:
                return int(x)
            except:
                return x
        log_df["nb_valeurs_modifiees"] = log_df["nb_valeurs_modifiees"].apply(safe_int)

    return df, log_df


# ------------------------
# Fallback applicateur (colonne unique)
# ------------------------
def apply_correction(df: pd.DataFrame, col: str, correction: str) -> pd.DataFrame:
    if col not in df.columns and col != "DOUBLONS":
        return df

    if correction == "Imputer (moyenne)":
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            df[col] = df[col].fillna(mode_val)

    elif correction == "Imputer (médiane)":
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            df[col] = df[col].fillna(mode_val)

    elif correction == "Imputer (mode)":
        mode_val = df[col].mode()[0] if not df[col].mode().empty else None
        df[col] = df[col].fillna(mode_val)

    elif correction == "Supprimer lignes":
        df = df.dropna(subset=[col])

    elif correction == "Supprimer colonne":
        df = df.drop(columns=[col])

    elif correction == "Remplacer par NaN + imputer":
        df[col] = df[col].replace([float("inf"), -float("inf")], pd.NA)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            df[col] = df[col].fillna(mode_val)

    elif correction.startswith("Encodage"):
        st.warning(f"⚠️ Encodage non encore implémenté pour {col}")

    return df


# ------------------------
# Télécharger base corrigée ou log
# ------------------------
def download_df(df: pd.DataFrame, label="Télécharger", file_name="data", file_format="csv"):
    if file_format == "csv":
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label=f"{label} (CSV)", data=csv_data, file_name=f"{file_name}.csv", mime="text/csv", key=f"{file_name}_csv")
    elif file_format == "excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        processed_data = output.getvalue()
        st.download_button(label=f"{label} (Excel)", data=processed_data, file_name=f"{file_name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{file_name}_excel")
    else:
        st.warning(f"Format {file_format} non supporté")


# ------------------------
# Prétraitement basé sur Dictionnaire de Données
# ------------------------
def run_dictionary_based_preprocessing(df: pd.DataFrame):
    """
    Prétraitement avancé basé sur un dictionnaire de données
    """
    if not HAS_DATA_QUALITY:
        st.error("❌ Module data_quality non disponible. Vérifiez que data_quality.py est présent.")
        return
    
    st.subheader(" Prétraitement Basé sur Dictionnaire de Données")
    
    # Étape 1 : Charger le dictionnaire
    st.markdown("###  Charger le Dictionnaire de Données")
    
    uploaded_dict = st.file_uploader(
        "Charger votre dictionnaire (Excel ou CSV)",
        type=['xlsx', 'xls', 'csv'],
        help="Le dictionnaire doit contenir les colonnes : Colonne, Type, Obligatoire, Valeurs_Autorisées, Min, Max, Format, Action_Si_Anomalie"
    )
    
    if uploaded_dict is None:
        st.info(" Chargez un dictionnaire de données pour commencer")
        st.markdown("---")
        st.markdown("** Format du dictionnaire requis :**")
        st.markdown("""
        **Colonnes requises** :
        - `Colonne` : Nom de la colonne
        - `Type` : numerique, categorique, texte, date
        - `Obligatoire` : oui/non
        - `Valeurs_Autorisées` : Liste séparée par virgules (pour catégoriques)
        - `Min` : Valeur minimale
        - `Max` : Valeur maximale
        - `Format` : Format attendu (email, regex, date format)
        - `Action_Si_Anomalie` : imputer_moyenne, imputer_mediane, imputer_mode, supprimer_ligne, mettre_vide, ignorer
        
        Consultez `TEMPLATE_DICTIONNAIRE.md` pour plus de détails.
        """)
        st.markdown("---")
        return
    
    # Charger le dictionnaire
    try:
        if uploaded_dict.name.endswith('.csv'):
            dictionnaire = pd.read_csv(uploaded_dict)
        else:
            dictionnaire = pd.read_excel(uploaded_dict)
        
        st.success(f"✅ Dictionnaire chargé : {len(dictionnaire)} règles définies")
        
        # Étape 1.1 : Validation stricte du dictionnaire
        st.markdown("####  Validation du dictionnaire")
        
        from data_quality import validate_dictionnaire, normalize_dictionnaire
        
        errors = validate_dictionnaire(dictionnaire)
        
        if errors:
            st.error(f"❌ Le dictionnaire contient {len(errors)} erreur(s) :")
            for error in errors:
                st.error(error)
            
            st.warning("**Colonnes actuelles dans votre fichier :**")
            st.write(list(dictionnaire.columns))
            
            st.info("""
            ** Consultez la documentation pour plus de détails :**
            - Format attendu pour chaque colonne
            - Exemples de valeurs valides
            - Règles de validation
            
            Voir : `DOCUMENTATION_DICTIONNAIRE_DETAILLEE.md`
            """)
            return
        else:
            st.success("✅ Dictionnaire valide !")
        
        # Étape 1.2 : Normalisation automatique
        st.markdown("####  Normalisation automatique")
        
        dictionnaire = normalize_dictionnaire(dictionnaire)
        st.success("✅ Dictionnaire normalisé (majuscules/minuscules, virgules/points, espaces)")

        # Aperçu complet du dictionnaire (sans expander)
        st.markdown("####  Aperçu du dictionnaire normalisé")
        st.dataframe(dictionnaire, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du dictionnaire : {e}")
        import traceback
        st.code(traceback.format_exc())
        return
    
    # Étape 1.5 : Pré-validation et nettoyage des données
    st.markdown("###  Pré-validation des Données")
    
    # Analyse rapide des problèmes de qualité
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes", df.shape[0])
    with col2:
        st.metric("Colonnes", df.shape[1])
    with col3:
        st.metric("Valeurs manquantes", f"{missing_cells} ({missing_pct:.1f}%)")
    
    # Afficher les colonnes avec des NaN
    cols_with_nan = df.columns[df.isna().any()].tolist()
    if cols_with_nan:
        st.warning(f"⚠️ {len(cols_with_nan)} colonnes contiennent des valeurs manquantes")
        
        # Détail des valeurs manquantes affiché directement (sans expander)
        st.markdown("####  Détail des valeurs manquantes par colonne")
        missing_df = pd.DataFrame({
            'Colonne': cols_with_nan,
            'NaN': [df[col].isna().sum() for col in cols_with_nan],
            'Pourcentage': [f"{(df[col].isna().sum()/len(df))*100:.1f}%" for col in cols_with_nan]
        }).sort_values('NaN', ascending=False)
        st.dataframe(missing_df, use_container_width=True)
        
        # Proposer un nettoyage préalable
        st.markdown("**Options de pré-nettoyage** :")
        pre_clean = st.radio(
            "Voulez-vous nettoyer les données avant la détection par dictionnaire ?",
            ["Non, continuer avec les NaN", "Oui, nettoyer automatiquement", "Oui, choisir les actions"],
            key="pre_clean_choice",
            help="Le nettoyage préalable peut améliorer la détection par dictionnaire"
        )
        
        if pre_clean == "Oui, nettoyer automatiquement":
            if st.button(" Nettoyer Automatiquement", key="auto_clean"):
                with st.spinner("Nettoyage en cours..."):
                    df_cleaned = df.copy()
                    clean_log = []
                    
                    for col in cols_with_nan:
                        before = df_cleaned[col].isna().sum()
                        
                        if df_cleaned[col].dtype in ['int64', 'float64']:
                            # Numériques : imputer par la médiane
                            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                            clean_log.append({
                                'Colonne': col,
                                'Type': 'Numérique',
                                'Action': 'Imputation médiane',
                                'NaN nettoyés': before
                            })
                        else:
                            # Catégoriques : imputer par le mode
                            mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'UNKNOWN'
                            df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                            clean_log.append({
                                'Colonne': col,
                                'Type': 'Catégorique',
                                'Action': f'Imputation mode ({mode_val})',
                                'NaN nettoyés': before
                            })
                    
                    # Mettre à jour le dataframe
                    df = df_cleaned
                    st.session_state['preprocessed_data'] = df
                    
                    st.success(f"✅ Nettoyage terminé : {missing_cells} valeurs manquantes traitées")
                    
                    # Afficher le log directement
                    st.markdown("** Log du nettoyage :**")
                    st.dataframe(pd.DataFrame(clean_log), use_container_width=True)
        
        elif pre_clean == "Oui, choisir les actions":
            st.info(" Utilisez l'onglet 'Prétraitement Standard' pour un nettoyage personnalisé, puis revenez ici.")
    else:
        st.success("✅ Aucune valeur manquante détectée")
    
    st.markdown("---")
    
    # Étape 2 : Détecter les anomalies
    st.markdown("###  Détecter les Anomalies")
    
    if st.button(" Lancer la Détection", key="detect_anomalies"):
        with st.spinner("Détection en cours..."):
            try:
                checker = DataQualityChecker(dictionnaire)
                anomalies_df = checker.detect_anomalies(df)
                
                st.session_state['anomalies_df'] = anomalies_df
                st.session_state['quality_checker'] = checker
                
                if len(anomalies_df) == 0:
                    st.success(" Aucune anomalie détectée ! Vos données sont conformes au dictionnaire.")
                else:
                    st.warning(f"⚠️ {len(anomalies_df)} anomalies détectées")
                    
                    # Statistiques par type
                    st.markdown("** Répartition par type d'anomalie**")
                    anomalie_counts = anomalies_df['Anomalie'].value_counts()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(anomalie_counts)
                    with col2:
                        st.bar_chart(anomalie_counts)
                    
                    # Statistiques par colonne
                    st.markdown("** Répartition par colonne**")
                    colonne_counts = anomalies_df['Colonne'].value_counts()
                    st.dataframe(colonne_counts)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de la détection : {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Afficher les anomalies si détectées
    if 'anomalies_df' in st.session_state and len(st.session_state['anomalies_df']) > 0:
        anomalies_df = st.session_state['anomalies_df']
        
        st.markdown("###  Rapport d'Anomalies")
        
        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            type_filter = st.multiselect(
                "Filtrer par type d'anomalie",
                options=anomalies_df['Anomalie'].unique().tolist(),
                default=anomalies_df['Anomalie'].unique().tolist()
            )
        with col2:
            col_filter = st.multiselect(
                "Filtrer par colonne",
                options=anomalies_df['Colonne'].unique().tolist(),
                default=anomalies_df['Colonne'].unique().tolist()
            )
        
        filtered_anomalies = anomalies_df[
            (anomalies_df['Anomalie'].isin(type_filter)) &
            (anomalies_df['Colonne'].isin(col_filter))
        ]
        
        st.dataframe(filtered_anomalies, use_container_width=True)
        
        # Télécharger le rapport
        st.markdown("###  Télécharger le Rapport d'Anomalies")
        download_df(filtered_anomalies, "Télécharger rapport anomalies", "rapport_anomalies", "excel")
        
        # Étape 3 : Appliquer les corrections
        st.markdown("###  Appliquer les Corrections")
        
        st.info(" Les corrections seront appliquées selon les actions définies dans le dictionnaire")
        
        if st.button("✅ Appliquer Toutes les Corrections", key="apply_corrections"):
            with st.spinner("Application des corrections..."):
                try:
                    checker = st.session_state['quality_checker']
                    df_corrected, log_df = checker.apply_corrections(df, anomalies_df)
                    
                    # Supprimer les lignes marquées pour suppression
                    rows_to_delete = log_df[log_df['Action_Appliquée'] == 'supprimer_ligne']['Ligne'].unique()
                    if len(rows_to_delete) > 0:
                        indices_to_drop = [int(r) - 1 for r in rows_to_delete if r != 'N/A']
                        df_corrected = df_corrected.drop(indices_to_drop, errors='ignore').reset_index(drop=True)
                    
                    st.session_state['clean_data'] = df_corrected
                    st.session_state['correction_log'] = log_df
                    
                    st.success(f"✅ Corrections appliquées ! {len(log_df)} modifications effectuées")
                    
                    # Afficher le log
                    st.markdown("###  Log des Corrections")
                    st.dataframe(log_df, use_container_width=True)
                    
                    # Statistiques de qualité
                    st.markdown("###  Statistiques de Qualité")
                    stats = checker.get_quality_stats(df_corrected, anomalies_df)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Lignes avant", stats['total_lignes'])
                        st.metric("Lignes après", len(df_corrected))
                    with col2:
                        st.metric("Colonnes", stats['total_colonnes'])
                        st.metric("Anomalies corrigées", len(log_df))
                    with col3:
                        st.metric("Taux de réussite", f"{(1 - len(anomalies_df)/len(df))*100:.1f}%")
                    
                    # Taux de complétude
                    st.markdown("**Taux de complétude par colonne**")
                    completude_df = pd.DataFrame(list(stats['taux_completude'].items()), columns=['Colonne', 'Taux'])
                    st.dataframe(completude_df)
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'application des corrections : {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
        
        # Étape 4 : Télécharger les résultats
        if 'clean_data' in st.session_state and 'correction_log' in st.session_state:
            st.markdown("###  Télécharger les Résultats")
            
            col1, col2 = st.columns(2)
            with col1:
                download_df(st.session_state['clean_data'], " Base corrigée", "base_corrigee", "excel")
            with col2:

                download_df(st.session_state['correction_log'], " Log des corrections", "log_corrections", "excel")
