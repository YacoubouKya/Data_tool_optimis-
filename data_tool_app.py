# app.py
import streamlit as st
import pandas as pd
import data_loader, eda, preprocessing, modeling, evaluation, reporting, model_comparison
from sklearn.model_selection import train_test_split

# ------------------------
# ⚙️ Configuration de la page
# ------------------------
st.set_page_config(
    page_title="Data Project Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# 🎨 CSS personnalisé (V1)
# ------------------------
st.markdown("""
<style>

/********* HEADER *********/
.custom-header { position: fixed; top: 0; left: 0; width: 100%; height: 60px; background-color: #1E3A5F; color: white; display: flex; align-items: center; justify-content: space-between; padding: 0 40px; z-index: 9999; box-shadow: 0px 2px 5px rgba(0,0,0,0.3);}
.custom-header .logo { font-size: 22px; font-weight: bold; color: #FFD700; }
.custom-header .menu { display: flex; gap: 20px; }
.custom-header .menu a { color: white; text-decoration: none; font-weight: 500; font-family: 'Segoe UI', sans-serif; transition: color 0.3s;}
.custom-header .menu a:hover { color: #FFD700; }

.block-container { padding-top: 80px !important; }
.stApp { background-color: #1E3A5F; }
.block-container, .st-emotion-cache-18e3th9, .st-emotion-cache-1y4p8pa { background-color: transparent !important; }

/********* TITRES *********/
h1, h2, h3, h4 { color: #FFD700; font-family: 'Segoe UI', sans-serif; }

/********* TEXTE GLOBAL (VERSION FIXÉE !) *********/
/* On exclut les éléments critiques utilisés par Streamlit */
.block-container p,
.block-container span,
.block-container label,
.block-container div:not([data-testid="stFileUploader"]):not(.stSelectbox):not([role="radiogroup"]) {
    color: #FFFFFF !important;
    font-family: 'Segoe UI', sans-serif;
}

/********* SIDEBAR *********/
[data-testid="stSidebar"] { background-color: #1569C7 !important; color: yellow !important; }
[data-testid="stSidebar"] h1, h2, h3, label { color: yellow !important; }

/********* BOUTONS *********/
.stButton>button { background-color: #FFD700; color: #1E3A5F; border-radius: 10px; padding: 10px 20px; border: none; font-weight: bold; }
.stButton>button:hover { background-color: #FFA500; color: white; }

/********* FILE UPLOADER (fix complet) *********/
[data-testid="stFileUploader"] {
    background-color: #FFD700 !important;
    border-radius: 10px;
    padding: 10px;
}

[data-testid="stFileUploader"] * {
    color: #1E3A5F !important;
    font-weight: 600;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #FFF6BF !important;
    border: 2px dashed #1E3A5F !important;
}

/********* RADIO + SELECTBOX (fix complet) *********/
div[role="radiogroup"] label {
    background: #34495E !important;
    color: yellow !important;
    padding: 8px 15px;
    border-radius: 8px;
    margin: 3px 0;
    cursor: pointer;
}

div[role="radiogroup"] label:hover {
    background: #1ABC9C !important;
}

.stSelectbox * {
    background-color: #34495E !important;
    color: yellow !important;
}

/********* MÉTRIQUES & PIPELINE (Modélisation/Évaluation) *********/
/* JSON display pour les métriques */
[data-testid="stJson"] {
    background-color: #000000 !important;
    border-radius: 8px;
    padding: 10px;
}

[data-testid="stJson"] *,
[data-testid="stJson"] div,
[data-testid="stJson"] span,
[data-testid="stJson"] p {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    font-family: 'Courier New', monospace !important;
}

/* Code blocks pour le pipeline */
code, pre {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border-radius: 5px;
    padding: 10px !important;
}

/* Forcer le fond noir pour les conteneurs de métriques */
.stMarkdown code {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}

/* Cibler spécifiquement les éléments JSON internes */
[data-testid="stJson"] > div {
    background-color: #000000 !important;
}

[data-testid="stJson"] pre {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}

/* Dataframes (métriques en tableau) */
[data-testid="stDataFrame"] {
    background-color: #000000 !important;
}

[data-testid="stDataFrame"] * {
    color: #FFFFFF !important;
}

/* Tables */
.stDataFrame table {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}

.stDataFrame th {
    background-color: #1E3A5F !important;
    color: #FFD700 !important;
    font-weight: bold;
}

.stDataFrame td {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}

/* Expander pour les détails */
[data-testid="stExpander"] {
    background-color: #1E3A5F !important;
    border: 1px solid #FFD700 !important;
}

[data-testid="stExpander"] * {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)




# Injection HTML du header
st.markdown("""
<div class="custom-header">
    <div class="logo">🐍 Data Project Tool</div>
    <div class="menu">
        <a href="#">About</a>
        <a href="#">Documentation</a>
        <a href="#">Community</a>
        <a href="#">Success Stories</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------
# 🎯 En-tête principal
# ------------------------
st.title("📊 Data Project Tool")
st.markdown("Bienvenue dans ton outil de projet data interactif 🚀")

# ------------------------
# 📌 Sidebar Navigation
# ------------------------
st.sidebar.title("📌 Navigation")

# Gérer la navigation automatique via session_state
if "target_section" in st.session_state:
    default_index = ["📥 Chargement", "🔎 EDA", "🛠️ Prétraitement", "🔬 Comparaison de Modèles", "🎯 Affinage de Modèle", "📈 Évaluation", "📝 Reporting"].index(st.session_state.target_section)
    del st.session_state.target_section
else:
    default_index = 0

section = st.sidebar.radio(
    "Aller à :",
    ["📥 Chargement", "🔎 EDA", "🛠️ Prétraitement", "🔬 Comparaison de Modèles", "🎯 Affinage de Modèle", "📈 Évaluation", "📝 Reporting"],
    index=default_index
)

# Messages d'aide dans la sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Guide Rapide")
if section == "🔬 Comparaison de Modèles":
    st.sidebar.info("⭐ **Commencez ici** pour explorer plusieurs modèles automatiquement")
elif section == "🎯 Affinage de Modèle":
    st.sidebar.info("🎯 **Optionnel** : Optimisez un modèle spécifique avec tuning fin")
elif section == "📈 Évaluation":
    st.sidebar.info("📊 Analysez votre modèle en détail après comparaison ou affinage")

# ------------------------
# Sections
# ------------------------
if section == "📥 Chargement":
    st.header("📥 Chargement des données")
    uploaded = st.file_uploader("Charger un fichier (CSV ou Excel)", type=["csv", "xlsx", "xls"])
    sep = ","; sheet = None
    if uploaded:
        if uploaded.name.lower().endswith(".csv"):
            sep = st.selectbox("Séparateur CSV", options=[",", ";", "\t"], index=0)
        elif uploaded.name.lower().endswith((".xls", ".xlsx")):
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Choisissez la feuille Excel", options=xls.sheet_names)
        df = data_loader.load_file(uploaded, sep=sep, sheet_name=sheet)
        if df is not None:
            st.session_state["data"] = df
            st.success("✅ Données chargées avec succès !")
            st.dataframe(df.head())

elif section == "🔎 EDA":
    st.header("🔎 Analyse exploratoire (EDA)")
    if "data" in st.session_state:
        eda.run_eda(st.session_state["data"])
    else:
        st.warning("⚠️ Chargez d'abord les données dans l'onglet Chargement.")

elif section == "🛠️ Prétraitement":
    st.header("🛠️ Prétraitement")
    if "data" in st.session_state:
        df = st.session_state["data"]
        
        # Choix du mode de prétraitement
        st.markdown("### 🎯 Choisir le Mode de Prétraitement")
        mode = st.radio(
            "Mode",
            ["📊 Mode Automatique (Profiling)", "📋 Mode Dictionnaire de Données"],
            help="Mode Automatique : Détection basée sur ydata-profiling | Mode Dictionnaire : Validation basée sur vos règles métier"
        )
        
        st.markdown("---")
        
        if mode == "📊 Mode Automatique (Profiling)":
            # Mode classique existant
            profile = eda.generate_profile(df)
            issues = preprocessing.detect_and_propose_corrections(profile, df)
            if issues:
                st.subheader("🚨 Anomalies détectées et corrections proposées")
                corrections_dict = {}
                for issue in issues:
                    col = issue["colonne"]
                    anomalies = ", ".join(issue["anomalies"])
                    st.markdown(f"**Colonne : `{col}`**"); st.write(f"Anomalies : {anomalies}")
                    choice = st.selectbox(f"Choisir correction pour `{col}`", ["Ne pas appliquer de correction"] + issue["propositions"], key=f"choice_{col}")
                    corrections_dict[col] = choice
                if st.button("✅ Appliquer toutes les corrections sélectionnées"):
                    valid_corrections = {col: corr for col, corr in corrections_dict.items() if corr != "Ne pas appliquer de correction"}
                    if valid_corrections:
                        df_corrige, log_df = preprocessing.apply_corrections_with_log(df, valid_corrections)
                        st.session_state["clean_data"] = df_corrige
                        st.session_state["correction_log"] = log_df
                        st.success("✅ Toutes les corrections appliquées !")
                        st.subheader("📋 Tableau récapitulatif des corrections")
                        st.dataframe(log_df)
                        preprocessing.download_df(df_corrige, label="Télécharger la base corrigée", file_name="base_corrigee", file_format="excel")
                        preprocessing.download_df(log_df, label="Télécharger le log des corrections", file_name="log_corrections", file_format="excel")
                    else:
                        st.info("Aucune correction sélectionnée à appliquer.")
            else:
                st.info("✅ Aucune anomalie détectée !")
        
        else:
            # Nouveau mode dictionnaire
            preprocessing.run_dictionary_based_preprocessing(df)
    
    else:
        st.warning("⚠️ Chargez d'abord les données.")

elif section == "🎯 Affinage de Modèle":
    st.header("🎯 Affinage de Modèle")
    
    # Message d'orientation
    st.info("""
    💡 **Quand utiliser cette section ?**
    - Vous voulez configurer finement les hyperparamètres d'un modèle spécifique
    - Vous avez déjà identifié un modèle prometteur via la Comparaison
    - Vous voulez un contrôle total sur l'entraînement
    
    ⭐ **Nouveau ?** Commencez plutôt par "🔬 Comparaison de Modèles" pour explorer rapidement !
    """)
    
    df_to_use = st.session_state.get("clean_data", st.session_state.get("data"))
    if df_to_use is not None:
        res = modeling.run_modeling(df_to_use)
        st.success("✅ Modèle entraîné et jeux train/test créés avec succès !")
        st.subheader("📌 Pipeline"); st.write(res["pipeline"])
        st.subheader("📌 Jeux de données")
        st.write(f"**X_train shape**: {res['X_train'].shape} | **X_test shape**: {res['X_test'].shape}")
        st.write(f"**y_train shape**: {res['y_train'].shape} | **y_test shape**: {res['y_test'].shape}")
        st.subheader("📌 Aperçu X_train"); st.dataframe(res["X_train"].head())
        st.subheader("📌 Aperçu y_train"); st.dataframe(res["y_train"].head())
        st.session_state.update({
            "model": res["pipeline"], "X_train": res["X_train"], "X_test": res["X_test"],
            "y_train": res["y_train"], "y_test": res["y_test"], "task_type": res["task"]
        })
        
        # Bouton pour aller à l'évaluation
        st.markdown("---")
        st.markdown("### 🎯 Prochaine Étape")
        if st.button("📈 Évaluer ce modèle", type="primary"):
            st.session_state.target_section = "📈 Évaluation"
            st.rerun()
    else:
        st.warning("⚠️ Chargez et/ou prétraitez d'abord les données.")

elif section == "🔬 Comparaison de Modèles":
    st.header("🔬 Comparaison de Modèles ML")
    
    # Message d'accueil
    st.success("""
    ⭐ **Point d'entrée recommandé pour la modélisation !**
    
    Cette section vous permet de :
    - 🚀 Comparer 9-10 modèles automatiquement
    - 📊 Identifier le meilleur modèle en quelques secondes
    - 📈 Visualiser les performances côte à côte
    - 💾 Exporter et sauvegarder les résultats
    """)
    
    df_to_use = st.session_state.get("clean_data", st.session_state.get("data"))
    if df_to_use is not None:
        model_comparison.run_model_comparison(df_to_use)
        
        # Boutons de navigation après comparaison
        if "comparison_results" in st.session_state and "best_model" in st.session_state:
            st.markdown("---")
            st.markdown("### 🎯 Prochaines Étapes Recommandées")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📈 Analyser en détail**")
                st.write("Évaluez le meilleur modèle avec des graphiques détaillés")
                if st.button("📈 Aller à l'Évaluation", key="goto_eval"):
                    st.session_state.target_section = "📈 Évaluation"
                    st.rerun()
            
            with col2:
                st.markdown("**🎯 Optimiser davantage**")
                st.write("Affinez les hyperparamètres du meilleur modèle")
                if st.button("🎯 Aller à l'Affinage", key="goto_tuning"):
                    st.session_state.target_section = "🎯 Affinage de Modèle"
                    st.rerun()
            
            with col3:
                st.markdown("**📝 Créer le rapport**")
                st.write("Générez un rapport HTML complet")
                if st.button("📝 Aller au Reporting", key="goto_report"):
                    st.session_state.target_section = "📝 Reporting"
                    st.rerun()
    else:
        st.warning("⚠️ Chargez et/ou prétraitez d'abord les données.")

elif section == "📈 Évaluation":
    st.header("📈 Évaluation du modèle")
    
    # Détecter la source du modèle
    if "best_model_name" in st.session_state:
        st.success(f"🏆 **Modèle sélectionné** : {st.session_state['best_model_name']}")
        if "comparator" in st.session_state:
            st.info(f"📊 **Score** : {st.session_state['comparator'].best_score:.4f}")
    
    if "model" in st.session_state:
        evaluation.run_evaluation(st.session_state["model"], st.session_state["X_test"], st.session_state["y_test"])
        
        # Bouton pour le reporting
        st.markdown("---")
        st.markdown("### 🎯 Prochaine Étape")
        if st.button("📝 Générer le Rapport Complet", type="primary"):
            st.session_state.target_section = "📝 Reporting"
            st.rerun()
    else:
        st.warning("⚠️ Entraînez un modèle d'abord.")
        st.info("""
        💡 **Comment obtenir un modèle à évaluer ?**
        
        **Option 1 (Recommandée)** : Allez dans "🔬 Comparaison de Modèles"
        - Comparez plusieurs modèles automatiquement
        - Le meilleur sera automatiquement sélectionné
        
        **Option 2** : Allez dans "🎯 Affinage de Modèle"
        - Configurez et entraînez un modèle spécifique
        """)

elif section == "📝 Reporting":
    st.header("📝 Reporting")
    reporting.generate_report(st.session_state)
