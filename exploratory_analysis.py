"""Module d'analyse exploratoire approfondie pour Data Project Tool.

Ce module permet une analyse complète des données prétraitées avant la modélisation,
avec des graphiques bivariés intelligents et des indicateurs actuariels.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from typing import Tuple, List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configuration des styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def detect_variable_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Détecte automatiquement les types de variables."""
    
    categorical_vars = []
    numerical_vars = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_vars.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            # Si peu de valeurs uniques, considérer comme catégorielle
            if df[col].nunique() < 10 and df[col].dtype == 'int64':
                categorical_vars.append(col)
            else:
                numerical_vars.append(col)
        else:
            categorical_vars.append(col)
    
    return {
        'categorical': categorical_vars,
        'numerical': numerical_vars
    }


def quali_vs_quali_analysis(df: pd.DataFrame, var1: str, var2: str) -> go.Figure:
    """Analyse bivariée qualitative vs qualitative."""
    
    # Tableau de contingence
    contingency = pd.crosstab(df[var1], df[var2])
    
    # Heatmap avec annotations
    fig = px.imshow(
        contingency.values,
        x=contingency.columns,
        y=contingency.index,
        title=f"Analyse : {var1} vs {var2}",
        labels=dict(x=var2, y=var1, color="Effectif"),
        color_continuous_scale="Blues"
    )
    
    # Ajouter les valeurs dans les cellules
    fig.update_traces(
        text=contingency.values,
        texttemplate="%{text}",
        textfont={"size": 12}
    )
    
    return fig


def quali_vs_quanti_analysis(df: pd.DataFrame, var_quali: str, var_quanti: str) -> go.Figure:
    """Analyse bivariée qualitative vs quantitative."""
    
    # Créer des boxplots par catégorie
    fig = px.box(
        df, 
        x=var_quali, 
        y=var_quanti,
        title=f"Distribution de {var_quanti} par {var_quali}",
        color=var_quali
    )
    
    # Ajouter les points individuels (jitter)
    fig.add_traces([
        go.Scatter(
            x=df[var_quali],
            y=df[var_quanti],
            mode='markers',
            marker=dict(size=4, opacity=0.3),
            name='Données individuelles',
            showlegend=False
        )
    ])
    
    return fig


def quanti_vs_quanti_analysis(df: pd.DataFrame, var1: str, var2: str) -> go.Figure:
    """Analyse bivariée quantitative vs quantitative."""
    
    # Calculer la corrélation
    corr = df[var1].corr(df[var2])
    
    # Scatter plot avec droite de régression
    fig = px.scatter(
        df, 
        x=var1, 
        y=var2,
        title=f"Analyse : {var1} vs {var2} (corrélation: {corr:.3f})",
        trendline="ols"
    )
    
    # Ajouter des informations statistiques
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[var1], df[var2])
    
    fig.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f"R² = {r_value**2:.3f}<br>p-value = {p_value:.4f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        font=dict(size=12)
    )
    
    return fig


def calculate_actuarial_indicators(df: pd.DataFrame, numerical_vars: List[str]) -> pd.DataFrame:
    """Calcule les indicateurs actuariels pour les variables numériques."""
    
    indicators = []
    
    for var in numerical_vars:
        data = df[var].dropna()
        
        # Statistiques descriptives de base
        stats_dict = {
            'Variable': var,
            'Moyenne': data.mean(),
            'Médiane': data.median(),
            'Écart-type': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data),
            'CV': data.std() / data.mean() if data.mean() != 0 else np.inf
        }
        
        # Tests de normalité
        if len(data) >= 8:  # Minimum pour Shapiro-Wilk
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limite à 5000 échantillons
                stats_dict['Shapiro_p'] = shapiro_p
                stats_dict['Normalité'] = 'Oui' if shapiro_p > 0.05 else 'Non'
            except:
                stats_dict['Normalité'] = 'Test échoué'
        
        # Détection d'outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        stats_dict['Outliers_IQR'] = len(outliers)
        stats_dict['%_Outliers'] = len(outliers) / len(data) * 100
        
        indicators.append(stats_dict)
    
    return pd.DataFrame(indicators)


def detect_outliers_methods(df: pd.DataFrame, numerical_vars: List[str]) -> Dict[str, pd.DataFrame]:
    """Détection des outliers avec différentes méthodes."""
    
    outliers_results = {}
    
    for var in numerical_vars:
        data = df[var].dropna()
        
        # Méthode 1: IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        
        # Méthode 2: Z-score
        z_scores = np.abs(stats.zscore(data))
        outliers_zscore = data[z_scores > 3]
        
        # Méthode 3: Isolation Forest (si assez de données)
        if len(data) > 100:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
            outliers_iso = data[outlier_labels == -1]
        else:
            outliers_iso = pd.Series([], dtype=data.dtype)
        
        # Synthèse
        outliers_summary = pd.DataFrame({
            'Méthode': ['IQR', 'Z-score', 'Isolation Forest'],
            'Nombre_outliers': [len(outliers_iqr), len(outliers_zscore), len(outliers_iso)],
            'Pourcentage': [
                len(outliers_iqr) / len(data) * 100,
                len(outliers_zscore) / len(data) * 100,
                len(outliers_iso) / len(data) * 100 if len(data) > 0 else 0
            ]
        })
        
        outliers_results[var] = outliers_summary
    
    return outliers_results


def correlation_analysis(df: pd.DataFrame, numerical_vars: List[str]) -> go.Figure:
    """Matrice de corrélation avec significativité."""
    
    # Matrice de corrélation
    corr_matrix = df[numerical_vars].corr()
    
    # Heatmap avec masque pour la diagonale
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    fig = px.imshow(
        corr_matrix,
        title="Matrice de corrélation",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        range_color=[-1, 1]
    )
    
    # Ajouter les valeurs de corrélation
    fig.update_traces(
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    )
    
    return fig


def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Génère un profil complet des données."""
    
    profile = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    
    # Types de variables
    var_types = detect_variable_types(df)
    profile['variable_types'] = var_types
    
    # Statistiques descriptives
    profile['descriptive_stats'] = df.describe().to_dict()
    
    return profile


def exploratory_analysis_interface(df: pd.DataFrame) -> None:
    """Interface Streamlit pour l'analyse exploratoire."""
    
    if df is None or df.empty:
        st.error("❌ Aucune donnée disponible pour l'analyse")
        return
    
    st.markdown("## 🔍 Analyse Exploratoire Approfondie")
    
    # Profil des données
    st.markdown("#### 📊 Profil des données")
    profile = generate_data_profile(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lignes", profile['shape'][0])
    col2.metric("Colonnes", profile['shape'][1])
    col3.metric("Valeurs manquantes", sum(profile['missing_values'].values()))
    col4.metric("Mémoire (MB)", f"{profile['memory_usage']:.2f}")
    
    # Types de variables
    var_types = profile['variable_types']
    st.write(f"**Variables numériques :** {len(var_types['numerical'])}")
    st.write(f"**Variables catégorielles :** {len(var_types['categorical'])}")
    
    # Types de variables détectés
    var_types = detect_variable_types(df)
    
    # Analyse bivariée
    st.markdown("### 📈 Analyse Bivariée")
    
    # Sélection des variables
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox("Première variable", df.columns)
    
    with col2:
        var2 = st.selectbox("Seconde variable", df.columns)
    
    if var1 and var2 and var1 != var2:
        # Déterminer le type d'analyse
        var1_type = 'categorical' if var1 in var_types['categorical'] else 'numerical'
        var2_type = 'categorical' if var2 in var_types['categorical'] else 'numerical'
        
        if var1_type == 'categorical' and var2_type == 'categorical':
            fig = quali_vs_quali_analysis(df, var1, var2)
        elif var1_type == 'categorical' and var2_type == 'numerical':
            fig = quali_vs_quanti_analysis(df, var1, var2)
        elif var1_type == 'numerical' and var2_type == 'categorical':
            fig = quali_vs_quanti_analysis(df, var2, var1)
        else:  # quanti vs quanti
            fig = quanti_vs_quanti_analysis(df, var1, var2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Indicateurs actuariels
    if var_types['numerical']:
        st.markdown("### 📊 Indicateurs Actuariels")
        
        st.markdown("##### 📈 Statistiques descriptives avancées")
        indicators_df = calculate_actuarial_indicators(df, var_types['numerical'])
        st.dataframe(indicators_df, use_container_width=True)
        
        st.markdown("##### 🎯 Détection d'outliers")
        outliers_results = detect_outliers_methods(df, var_types['numerical'])
        
        for var, outliers_df in outliers_results.items():
            st.write(f"**{var}**")
            st.dataframe(outliers_df, use_container_width=True)
        
        st.markdown("##### 🔗 Matrice de corrélation")
        if len(var_types['numerical']) > 1:
            corr_fig = correlation_analysis(df, var_types['numerical'])
            st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Besoin d'au moins 2 variables numériques pour la matrice de corrélation")
    
    st.success("✅ Analyse exploratoire terminée !")


if __name__ == "__main__":
    # Test du module
    pass
