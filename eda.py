# eda.py
# modules/eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from ydata_profiling import ProfileReport

@st.cache_data(show_spinner="Génération du profiling en cours...")
def generate_profile(df: pd.DataFrame):
    """
    Génère un rapport de profiling avec ydata-profiling (avec cache).
    Nécessite Python 3.11 (configuré via runtime.txt).
    Le cache évite de recalculer le profiling si les données n'ont pas changé.
    
    Optimisé pour les gros datasets :
    - Échantillonnage si >10000 lignes
    - Profiling minimal pour éviter les timeouts
    """
    # Calculer la taille du dataset
    dataset_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    n_rows = len(df)
    
    # Échantillonnage intelligent pour les gros datasets
    if n_rows > 10000 or dataset_size_mb > 5:
        st.warning(f"⚠️ Dataset volumineux détecté ({n_rows:,} lignes, {dataset_size_mb:.1f} MB)")
        st.info("💡 Échantillonnage de 10,000 lignes pour accélérer le profiling")
        df_sample = df.sample(n=min(10000, n_rows), random_state=42)
        
        # Profiling ultra-minimal pour gros datasets
        profile = ProfileReport(
            df_sample,
            title="Profiling EDA (Échantillon)",
            minimal=True,
            explorative=False,
            correlations=None,
            missing_diagrams=None,
            interactions=None,
            samples=None
        )
    else:
        # Profiling minimal pour petits datasets
        profile = ProfileReport(df, title="Profiling EDA", minimal=True)
    
    return profile

def run_eda(df: pd.DataFrame):
    st.subheader("Aperçu général")
    st.write("Dimensions :", df.shape)
    st.dataframe(df.head())

    st.markdown("**Statistiques descriptives (numériques)**")
    st.dataframe(df.describe().T.round(4))

    # --------------------------
    # Rapport de profiling
    # --------------------------
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
    if "show_report" not in st.session_state:
        st.session_state.show_report = False

    if not st.session_state.report_generated:
        if st.button("📊 Générer le rapport de Profiling"):
            prof = generate_profile(df)
            prof.to_file("profiling_report.html")
            st.session_state.report_generated = True
            st.session_state.show_report = True

    if st.session_state.report_generated:
        st.success("✅ Rapport de profiling généré.")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("👁️ Afficher le rapport"):
                st.session_state.show_report = True
        with col2:
            if st.button("🙈 Masquer le rapport"):
                st.session_state.show_report = False
        with col3:
            with open("profiling_report.html", "rb") as f:
                st.download_button(label="💾 Télécharger le rapport HTML", data=f, file_name="profiling_report.html", mime="text/html")

        if st.session_state.show_report:
            with open("profiling_report.html", "r", encoding="utf-8") as f:
                report_html = f.read()
            st.components.v1.html(report_html, height=800, scrolling=True)

    # --------------------------
    # Histogrammes : sélection interactive (évite boucle coûteuse par défaut)
    # --------------------------
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        st.subheader("Histogrammes (sélectionner une variable ou afficher un échantillon)")
        col_choice = st.selectbox("Choisir une variable à afficher", ["--Tous (limité)-->"] + num_cols)
        if col_choice == "--Tous (limité)-->":
            # On propose un échantillon des premières 6 variables pour éviter surcharge
            to_plot = num_cols[:6]
        else:
            to_plot = [col_choice]

        for col in to_plot:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Histogramme de {col}")
            st.pyplot(fig)
            plt.close(fig)

    # --------------------------
    # Corrélation
    # --------------------------
    if "corr_generated" not in st.session_state:
        st.session_state.corr_generated = False
    if "show_corr" not in st.session_state:
        st.session_state.show_corr = False

    if not st.session_state.corr_generated:
        if st.button("🔗 Générer la matrice de corrélation"):
            st.session_state.corr_generated = True
            st.session_state.show_corr = True

    if st.session_state.corr_generated:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👁️ Afficher corrélation"):
                st.session_state.show_corr = True
        with col2:
            if st.button("🙈 Masquer corrélation"):
                st.session_state.show_corr = False

        if st.session_state.show_corr:
            st.subheader("Matrice de corrélation")
            
            # Choix de la méthode de corrélation
            col1, col2 = st.columns([2, 1])
            with col1:
                corr_method = st.selectbox(
                    "Méthode de corrélation :",
                    ["pearson", "spearman", "kendall"],
                    help="• Pearson : Relations linéaires\n• Spearman : Relations monotones\n• Kendall : Robuste aux outliers"
                )
            with col2:
                st.info("📊 Choisissez selon vos données")
            
            # Calculer la matrice avec la méthode choisie
            corr = df.corr(method=corr_method, numeric_only=True)
            
            # Informations sur la méthode
            method_info = {
                "pearson": "📈 Corrélation linéaire de Pearson",
                "spearman": "📊 Corrélation de rang Spearman", 
                "kendall": "🎯 Tau de Kendall (robuste)"
            }
            
            st.caption(f"Méthode utilisée : {method_info[corr_method]}")
            
            # arrondir pour lisibilité
            corr_display = corr.round(3)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_display, annot=True, cmap="coolwarm", center=0, ax=ax)
            ax.set_title(f"Matrice de corrélation ({corr_method.capitalize()})")
            st.pyplot(fig)
            plt.close(fig)