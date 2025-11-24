# data_quality_optimized.py
"""
Module de contrôle qualité des données OPTIMISÉ (vectorisé, sans boucles)
Version 2.0 - Performance améliorée
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Tuple

class DataQualityChecker:
    """Classe optimisée pour vérifier la qualité des données selon un dictionnaire"""
    
    def __init__(self, dictionnaire: pd.DataFrame):
        """
        Initialise le checker avec un dictionnaire de données
        
        Args:
            dictionnaire: DataFrame avec colonnes [Colonne, Type, Obligatoire, Valeurs_Autorisées, Min, Max, Format, Action_Si_Anomalie]
        """
        self.dictionnaire = dictionnaire
        self.anomalies = []
        
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Détecte toutes les anomalies dans le DataFrame (VERSION OPTIMISÉE)
        
        Returns:
            DataFrame avec les anomalies détectées
        """
        self.anomalies = []
        
        for _, rule in self.dictionnaire.iterrows():
            col = rule['Colonne']
            
            # Vérifier si la colonne existe
            if col not in df.columns:
                self.anomalies.append({
                    'Ligne': 'N/A',
                    'Colonne': col,
                    'Anomalie': 'Colonne manquante',
                    'Valeur_Actuelle': 'N/A',
                    'Valeur_Attendue': 'Colonne requise',
                    'Action_Proposée': 'Ajouter colonne'
                })
                continue
            
            # 1. Vérifier valeurs manquantes (VECTORISÉ)
            if str(rule['Obligatoire']).lower() == 'oui':
                self._check_missing_values_vectorized(df, col, rule)
            
            # 2. Vérifier type et limites numériques (VECTORISÉ)
            if str(rule['Type']).lower() == 'numerique':
                self._check_numeric_values_vectorized(df, col, rule)
            
            # 3. Vérifier valeurs catégoriques (VECTORISÉ)
            elif str(rule['Type']).lower() == 'categorique':
                self._check_categorical_values_vectorized(df, col, rule)
            
            # 4. Vérifier format texte (VECTORISÉ)
            elif str(rule['Type']).lower() == 'texte':
                self._check_text_format_vectorized(df, col, rule)
            
            # 5. Vérifier format date (VECTORISÉ)
            elif str(rule['Type']).lower() == 'date':
                self._check_date_format_vectorized(df, col, rule)
        
        return pd.DataFrame(self.anomalies)
    
    def _check_missing_values_vectorized(self, df: pd.DataFrame, col: str, rule: pd.Series):
        """Vérifie les valeurs manquantes (VECTORISÉ - pas de boucle)"""
        missing_mask = df[col].isna()
        
        if missing_mask.any():
            # Créer toutes les anomalies en une seule opération
            missing_df = pd.DataFrame({
                'Ligne': df[missing_mask].index + 1,
                'Colonne': col,
                'Anomalie': 'Valeur manquante',
                'Valeur_Actuelle': 'NaN',
                'Valeur_Attendue': 'Valeur obligatoire',
                'Action_Proposée': rule['Action_Si_Anomalie']
            })
            self.anomalies.extend(missing_df.to_dict('records'))
    
    def _check_numeric_values_vectorized(self, df: pd.DataFrame, col: str, rule: pd.Series):
        """Vérifie les valeurs numériques (VECTORISÉ - pas de boucle)"""
        # Convertir en numérique
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Vérifier les valeurs non numériques (VECTORISÉ)
        non_numeric_mask = df[col].notna() & numeric_col.isna()
        if non_numeric_mask.any():
            non_numeric_df = pd.DataFrame({
                'Ligne': df[non_numeric_mask].index + 1,
                'Colonne': col,
                'Anomalie': 'Type incorrect',
                'Valeur_Actuelle': df.loc[non_numeric_mask, col].astype(str),
                'Valeur_Attendue': 'Valeur numérique',
                'Action_Proposée': rule['Action_Si_Anomalie']
            })
            self.anomalies.extend(non_numeric_df.to_dict('records'))
        
        # 2. Vérifier les limites min (VECTORISÉ)
        if pd.notna(rule['Min']):
            min_val = float(rule['Min'])
            below_min_mask = (numeric_col < min_val) & numeric_col.notna()
            if below_min_mask.any():
                below_min_df = pd.DataFrame({
                    'Ligne': df[below_min_mask].index + 1,
                    'Colonne': col,
                    'Anomalie': 'Valeur < minimum',
                    'Valeur_Actuelle': df.loc[below_min_mask, col].astype(str),
                    'Valeur_Attendue': f'>= {min_val}',
                    'Action_Proposée': rule['Action_Si_Anomalie']
                })
                self.anomalies.extend(below_min_df.to_dict('records'))
        
        # 3. Vérifier les limites max (VECTORISÉ)
        if pd.notna(rule['Max']):
            max_val = float(rule['Max'])
            above_max_mask = (numeric_col > max_val) & numeric_col.notna()
            if above_max_mask.any():
                above_max_df = pd.DataFrame({
                    'Ligne': df[above_max_mask].index + 1,
                    'Colonne': col,
                    'Anomalie': 'Valeur > maximum',
                    'Valeur_Actuelle': df.loc[above_max_mask, col].astype(str),
                    'Valeur_Attendue': f'<= {max_val}',
                    'Action_Proposée': rule['Action_Si_Anomalie']
                })
                self.anomalies.extend(above_max_df.to_dict('records'))
    
    def _check_categorical_values_vectorized(self, df: pd.DataFrame, col: str, rule: pd.Series):
        """Vérifie les valeurs catégoriques (VECTORISÉ - pas de boucle)"""
        if pd.notna(rule['Valeurs_Autorisées']):
            allowed = [v.strip() for v in str(rule['Valeurs_Autorisées']).split(',')]
            invalid_mask = (~df[col].isin(allowed)) & df[col].notna()
            
            if invalid_mask.any():
                invalid_df = pd.DataFrame({
                    'Ligne': df[invalid_mask].index + 1,
                    'Colonne': col,
                    'Anomalie': 'Valeur non autorisée',
                    'Valeur_Actuelle': df.loc[invalid_mask, col].astype(str),
                    'Valeur_Attendue': f'Parmi: {", ".join(allowed)}',
                    'Action_Proposée': rule['Action_Si_Anomalie']
                })
                self.anomalies.extend(invalid_df.to_dict('records'))
    
    def _check_text_format_vectorized(self, df: pd.DataFrame, col: str, rule: pd.Series):
        """Vérifie le format texte (VECTORISÉ avec apply optimisé)"""
        if pd.notna(rule['Format']):
            format_rule = str(rule['Format']).lower()
            
            # Filtrer les valeurs non nulles
            valid_data = df[df[col].notna()].copy()
            
            if len(valid_data) == 0:
                return
            
            # Vérifier email (VECTORISÉ)
            if format_rule == 'email':
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                # Utiliser str.match qui est vectorisé
                is_valid = valid_data[col].astype(str).str.match(email_pattern)
            
            # Vérifier regex personnalisée (VECTORISÉ)
            elif format_rule.startswith('^'):
                is_valid = valid_data[col].astype(str).str.match(format_rule)
            
            else:
                return
            
            # Créer anomalies pour les invalides
            invalid_mask = ~is_valid
            if invalid_mask.any():
                invalid_df = pd.DataFrame({
                    'Ligne': valid_data[invalid_mask].index + 1,
                    'Colonne': col,
                    'Anomalie': 'Format invalide',
                    'Valeur_Actuelle': valid_data.loc[invalid_mask, col].astype(str),
                    'Valeur_Attendue': f'Format: {format_rule}',
                    'Action_Proposée': rule['Action_Si_Anomalie']
                })
                self.anomalies.extend(invalid_df.to_dict('records'))
    
    def _check_date_format_vectorized(self, df: pd.DataFrame, col: str, rule: pd.Series):
        """Vérifie le format date (VECTORISÉ)"""
        date_format = str(rule['Format']) if pd.notna(rule['Format']) else '%Y-%m-%d'
        
        # Filtrer les valeurs non nulles
        valid_data = df[df[col].notna()].copy()
        
        if len(valid_data) == 0:
            return
        
        # Convertir en dates (VECTORISÉ)
        try:
            dates = pd.to_datetime(valid_data[col], format=date_format, errors='coerce')
            
            # Trouver les dates invalides
            invalid_mask = dates.isna()
            if invalid_mask.any():
                invalid_df = pd.DataFrame({
                    'Ligne': valid_data[invalid_mask].index + 1,
                    'Colonne': col,
                    'Anomalie': 'Format date invalide',
                    'Valeur_Actuelle': valid_data.loc[invalid_mask, col].astype(str),
                    'Valeur_Attendue': f'Format: {date_format}',
                    'Action_Proposée': rule['Action_Si_Anomalie']
                })
                self.anomalies.extend(invalid_df.to_dict('records'))
            
            # Vérifier limites min (VECTORISÉ)
            if pd.notna(rule['Min']):
                min_date = pd.to_datetime(rule['Min'])
                below_min_mask = (dates < min_date) & dates.notna()
                if below_min_mask.any():
                    below_min_df = pd.DataFrame({
                        'Ligne': valid_data[below_min_mask].index + 1,
                        'Colonne': col,
                        'Anomalie': 'Date < minimum',
                        'Valeur_Actuelle': valid_data.loc[below_min_mask, col].astype(str),
                        'Valeur_Attendue': f'>= {rule["Min"]}',
                        'Action_Proposée': rule['Action_Si_Anomalie']
                    })
                    self.anomalies.extend(below_min_df.to_dict('records'))
            
            # Vérifier limites max (VECTORISÉ)
            if pd.notna(rule['Max']):
                max_date = pd.to_datetime(rule['Max'])
                above_max_mask = (dates > max_date) & dates.notna()
                if above_max_mask.any():
                    above_max_df = pd.DataFrame({
                        'Ligne': valid_data[above_max_mask].index + 1,
                        'Colonne': col,
                        'Anomalie': 'Date > maximum',
                        'Valeur_Actuelle': valid_data.loc[above_max_mask, col].astype(str),
                        'Valeur_Attendue': f'<= {rule["Max"]}',
                        'Action_Proposée': rule['Action_Si_Anomalie']
                    })
                    self.anomalies.extend(above_max_df.to_dict('records'))
        
        except Exception:
            # Si erreur de conversion, toutes les dates sont invalides
            invalid_df = pd.DataFrame({
                'Ligne': valid_data.index + 1,
                'Colonne': col,
                'Anomalie': 'Format date invalide',
                'Valeur_Actuelle': valid_data[col].astype(str),
                'Valeur_Attendue': f'Format: {date_format}',
                'Action_Proposée': rule['Action_Si_Anomalie']
            })
            self.anomalies.extend(invalid_df.to_dict('records'))
    
    def apply_corrections(self, df: pd.DataFrame, anomalies_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applique les corrections selon les actions proposées (VERSION OPTIMISÉE)
        
        Returns:
            Tuple (DataFrame corrigé, DataFrame log des corrections)
        """
        df_corrected = df.copy()
        corrections_log = []
        
        # Grouper par action pour traitement vectorisé
        for action in anomalies_df['Action_Proposée'].unique():
            action_anomalies = anomalies_df[anomalies_df['Action_Proposée'] == action]
            
            # Grouper par colonne pour traitement vectorisé
            for col in action_anomalies['Colonne'].unique():
                col_anomalies = action_anomalies[action_anomalies['Colonne'] == col]
                
                # Extraire les indices (lignes - 1 car Ligne est 1-indexed)
                indices = col_anomalies['Ligne'].apply(lambda x: int(x) - 1 if x != 'N/A' else None).dropna().astype(int)
                
                if len(indices) == 0:
                    continue
                
                # Appliquer l'action de manière vectorisée
                valeur_apres = self._apply_action_vectorized(df_corrected, indices, col, action)
                
                # Log (vectorisé)
                for idx, val_apres in zip(indices, valeur_apres if isinstance(valeur_apres, list) else [valeur_apres] * len(indices)):
                    corrections_log.append({
                        'Ligne': idx + 1,
                        'Colonne': col,
                        'Anomalie': col_anomalies[col_anomalies['Ligne'] == idx + 1]['Anomalie'].iloc[0] if len(col_anomalies[col_anomalies['Ligne'] == idx + 1]) > 0 else 'N/A',
                        'Valeur_Avant': str(df.loc[idx, col]) if idx < len(df) else 'N/A',
                        'Valeur_Après': val_apres,
                        'Action_Appliquée': action,
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        return df_corrected, pd.DataFrame(corrections_log)
    
    def _apply_action_vectorized(self, df: pd.DataFrame, indices, col: str, action: str):
        """Applique une action de correction de manière vectorisée"""
        action_lower = str(action).lower()
        
        # Filtrer les indices valides
        valid_indices = [idx for idx in indices if idx < len(df)]
        
        if len(valid_indices) == 0:
            return ['N/A'] * len(indices)
        
        if action_lower == 'imputer_moyenne':
            moyenne = df[col].mean()
            df.loc[valid_indices, col] = moyenne
            return [f'{moyenne:.2f}'] * len(valid_indices)
        
        elif action_lower == 'imputer_mediane':
            mediane = df[col].median()
            df.loc[valid_indices, col] = mediane
            return [f'{mediane:.2f}'] * len(valid_indices)
        
        elif action_lower == 'imputer_mode':
            mode = df[col].mode()
            if len(mode) > 0:
                df.loc[valid_indices, col] = mode[0]
                return [str(mode[0])] * len(valid_indices)
            return ['N/A'] * len(valid_indices)
        
        elif action_lower == 'mettre_vide':
            df.loc[valid_indices, col] = np.nan
            return ['NaN'] * len(valid_indices)
        
        elif action_lower == 'supprimer_ligne':
            return ['Ligne supprimée'] * len(valid_indices)
        
        elif action_lower == 'ignorer':
            return ['Ignoré'] * len(valid_indices)
        
        else:
            return ['Action inconnue'] * len(valid_indices)
    
    def get_quality_stats(self, df: pd.DataFrame, anomalies_df: pd.DataFrame) -> Dict:
        """Génère des statistiques de qualité (OPTIMISÉ)"""
        # Calcul vectorisé du taux de complétude
        completude = ((df.notna().sum() / len(df)) * 100).round(2).astype(str) + '%'
        
        stats = {
            'total_lignes': len(df),
            'total_colonnes': len(df.columns),
            'total_anomalies': len(anomalies_df),
            'anomalies_par_type': anomalies_df['Anomalie'].value_counts().to_dict(),
            'anomalies_par_colonne': anomalies_df['Colonne'].value_counts().to_dict(),
            'taux_completude': completude.to_dict()
        }
        
        return stats
