# modules/reporting.py
"""
Module de génération de rapports HTML professionnels
Crée des rapports consolidés avec visualisations et métriques
"""

import streamlit as st
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import numpy as np

OUT_DIR = "outputs/reports"
os.makedirs(OUT_DIR, exist_ok=True)

# Configuration matplotlib pour de meilleurs rendus
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

def _img_to_base64(fig, width=800):
    """Convertit une figure matplotlib en base64 pour HTML"""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white', edgecolor='none')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<div class="figure-container"><img src="data:image/png;base64,{img_str}" style="max-width:{width}px; width:100%; height:auto;"></div>'

def _get_modern_css():
    """Retourne le CSS moderne pour le rapport"""
    return """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            text-align: center;
        }
        
        h2 {
            color: #34495e;
            font-size: 1.8em;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-left: 15px;
            border-left: 5px solid #3498db;
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
        }
        
        h3 {
            color: #2980b9;
            font-size: 1.4em;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        
        h4 {
            color: #7f8c8d;
            font-size: 1.1em;
            margin-top: 20px;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        p {
            margin: 10px 0;
            font-size: 1em;
        }
        
        .metric-box {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            margin: 10px 10px 10px 0;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            font-weight: bold;
        }
        
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
            display: block;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.8em;
            display: block;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        th {
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        tbody tr:hover {
            background: #f8f9fa;
            transition: background 0.3s ease;
        }
        
        tbody tr:nth-child(even) {
            background: #f9f9f9;
        }
        
        .figure-container {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .figure-container img {
            border-radius: 5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .info-box {
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        .success-box {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        ul {
            list-style: none;
            padding-left: 0;
        }
        
        ul li {
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }
        
        ul li:before {
            content: "▸";
            position: absolute;
            left: 0;
            color: #3498db;
            font-weight: bold;
        }
        
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid #ecf0f1;
        }
        
        @media print {
            body {
                background: white;
            }
            .container {
                box-shadow: none;
            }
        }
    </style>
    """

def generate_report(session_state: dict):
    st.subheader("📝 Générer rapport consolidé")
    
    # Options de personnalisation
    col1, col2 = st.columns(2)
    with col1:
        title_default = f"Rapport_ML_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        title = st.text_input("Titre du rapport", value=title_default)
    
    with col2:
        report_sections = st.multiselect(
            "Sections à inclure",
            ["Données", "Prétraitement", "Modèle", "Évaluation", "Visualisations"],
            default=["Données", "Modèle", "Évaluation"]
        )
    
    include_plots = st.checkbox("Inclure les visualisations détaillées", value=True)
    
    if st.button("📄 Créer rapport HTML", type="primary"):
        with st.spinner("Génération du rapport en cours..."):
            html = []
            html.append(f"<html><head><meta charset='utf-8'><title>{title}</title>")
            html.append(_get_modern_css())
            html.append("</head><body>")
            html.append('<div class="container">')
            html.append(f"<h1>📊 {title}</h1>")
            
            # Informations générales
            html.append('<div class="info-box">')
            html.append(f'<p><strong>📅 Date de génération :</strong> {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}</p>')
            model_name = session_state.get("current_model_name", session_state.get("best_model_name", "Non spécifié"))
            html.append(f'<p><strong>🤖 Modèle :</strong> {model_name}</p>')
            task_type = session_state.get("task_type", session_state.get("task", "Non spécifié"))
            html.append(f'<p><strong>🎯 Type de tâche :</strong> {task_type.capitalize()}</p>')
            html.append('</div>')

            # 1. Données brutes
            if "Données" in report_sections and "data" in session_state:
                df = session_state["data"]
                html.append("<h2>📁 1. Données brutes</h2>")
                
                # Métriques clés
                html.append('<div class="metric-box">')
                html.append(f'<span class="metric-label">Nombre de lignes</span>')
                html.append(f'<span class="metric-value">{df.shape[0]:,}</span>')
                html.append('</div>')
                
                html.append('<div class="metric-box">')
                html.append(f'<span class="metric-label">Nombre de colonnes</span>')
                html.append(f'<span class="metric-value">{df.shape[1]}</span>')
                html.append('</div>')
                
                missing_total = df.isna().sum().sum()
                html.append('<div class="metric-box">')
                html.append(f'<span class="metric-label">Valeurs manquantes</span>')
                html.append(f'<span class="metric-value">{missing_total:,}</span>')
                html.append('</div>')
                
                html.append("<h3>📋 Aperçu des données (5 premières lignes)</h3>")
                html.append(df.head(5).to_html(index=False, classes='dataframe'))
                
                html.append("<h3>📊 Statistiques descriptives</h3>")
                html.append(df.describe().round(3).to_html(classes='dataframe'))
                
                # Valeurs manquantes
                missing = df.isna().sum()
                if missing.sum() > 0:
                    html.append("<h3>⚠️ Valeurs manquantes par colonne</h3>")
                    missing_df = pd.DataFrame({
                        'Colonne': missing.index,
                        'Manquantes': missing.values,
                        'Pourcentage': (missing.values / len(df) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['Manquantes'] > 0].sort_values('Manquantes', ascending=False)
                    html.append(missing_df.to_html(index=False, classes='dataframe'))
                
                # Visualisations
                if include_plots:
                    num_cols = df.select_dtypes(include='number').columns.tolist()
                    if len(num_cols) > 0:
                        html.append("<h3>📈 Distributions des variables numériques</h3>")
                        
                        # Créer une grille de subplots
                        n_cols_to_plot = min(6, len(num_cols))
                        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                        axes = axes.flatten()
                        
                        for idx, col in enumerate(num_cols[:n_cols_to_plot]):
                            sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], color='#667eea')
                            axes[idx].set_title(f'Distribution de {col}', fontweight='bold')
                            axes[idx].set_xlabel(col)
                            axes[idx].set_ylabel('Fréquence')
                            axes[idx].grid(True, alpha=0.3)
                        
                        # Masquer les axes inutilisés
                        for idx in range(n_cols_to_plot, 6):
                            axes[idx].set_visible(False)
                        
                        plt.tight_layout()
                        html.append(_img_to_base64(fig, width=1000))

            # 2. Prétraitement
            if "Prétraitement" in report_sections and "clean_data" in session_state:
                cdf = session_state["clean_data"]
                html.append("<h2>🔧 2. Prétraitement des données</h2>")
                
                html.append('<div class="success-box">')
                html.append(f'<p><strong>✅ Données nettoyées :</strong> {cdf.shape[0]} lignes × {cdf.shape[1]} colonnes</p>')
                if "data" in session_state:
                    original_shape = session_state["data"].shape
                    rows_removed = original_shape[0] - cdf.shape[0]
                    if rows_removed > 0:
                        html.append(f'<p><strong>🗑️ Lignes supprimées :</strong> {rows_removed} ({rows_removed/original_shape[0]*100:.1f}%)</p>')
                html.append('</div>')
                
                if "correction_log" in session_state:
                    html.append("<h3>📝 Log des corrections appliquées</h3>")
                    html.append(session_state["correction_log"].to_html(index=False, classes='dataframe'))
                
                html.append("<h3>📋 Aperçu des données nettoyées</h3>")
                html.append(cdf.head(5).to_html(index=False, classes='dataframe'))

            # 3. Modèle
            if "Modèle" in report_sections and "model" in session_state:
                model_obj = session_state["model"]
                model_display_name = session_state.get("current_model_name", session_state.get("best_model_name", "Modèle ML"))
                
                html.append("<h2>🤖 3. Modèle de Machine Learning</h2>")
                
                html.append('<div class="info-box">')
                html.append(f'<p><strong>📌 Nom du modèle :</strong> {model_display_name}</p>')
                html.append(f'<p><strong>🔧 Type de pipeline :</strong> {type(model_obj).__name__}</p>')
                html.append(f'<p><strong>🎯 Tâche :</strong> {task_type.capitalize()}</p>')
                html.append('</div>')
                
                if all(k in session_state for k in ("X_train","X_test","y_train","y_test")):
                    html.append("<h3>📊 Dimensions des ensembles de données</h3>")
                    html.append('<div class="grid-2">')
                    
                    html.append('<div class="card">')
                    html.append('<h4>🎓 Ensemble d\'entraînement</h4>')
                    html.append(f'<p><strong>Features (X_train) :</strong> {session_state["X_train"].shape[0]} × {session_state["X_train"].shape[1]}</p>')
                    html.append(f'<p><strong>Cible (y_train) :</strong> {session_state["y_train"].shape[0]} valeurs</p>')
                    html.append('</div>')
                    
                    html.append('<div class="card">')
                    html.append('<h4>🧪 Ensemble de test</h4>')
                    html.append(f'<p><strong>Features (X_test) :</strong> {session_state["X_test"].shape[0]} × {session_state["X_test"].shape[1]}</p>')
                    html.append(f'<p><strong>Cible (y_test) :</strong> {session_state["y_test"].shape[0]} valeurs</p>')
                    html.append('</div>')
                    
                    html.append('</div>')
                
                # Feature importance
                if "feature_importance" in session_state:
                    fi = session_state["feature_importance"]
                    html.append("<h3>🎯 Importance des features</h3>")
                    
                    # Limiter au Top 10 pour la compacité
                    fi_top = fi.head(10)
                    
                    # Créer un tableau compact avec les valeurs
                    fi_df = pd.DataFrame({
                        'Feature': fi_top.index,
                        'Importance': fi_top.values,
                        'Importance (%)': (fi_top.values / fi_top.sum() * 100).round(2)
                    })
                    
                    html.append("<h4>📊 Top 10 des features les plus importantes</h4>")
                    html.append(fi_df.to_html(index=False, classes='dataframe'))
                    
                    # Graphique compact uniquement si visualisations activées
                    if include_plots:
                        # Graphique horizontal compact avec taille fixe
                        fig, ax = plt.subplots(figsize=(10, 5))  # Taille fixe compacte
                        
                        # Barres horizontales avec dégradé
                        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_top)))
                        bars = ax.barh(range(len(fi_top)), fi_top.values, color=colors, edgecolor='white', linewidth=0.5)
                        
                        # Inverser l'ordre pour avoir le plus important en haut
                        ax.set_yticks(range(len(fi_top)))
                        ax.set_yticklabels(fi_top.index)
                        ax.invert_yaxis()
                        
                        # Labels et style
                        ax.set_xlabel("Importance", fontsize=11, fontweight='bold')
                        ax.set_title("Visualisation des Top 10 Features", fontsize=12, fontweight='bold', pad=15)
                        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
                        
                        # Ajouter les valeurs sur les barres
                        for i, (bar, value) in enumerate(zip(bars, fi_top.values)):
                            width = bar.get_width()
                            ax.text(width, bar.get_y() + bar.get_height()/2, 
                                   f'{value:.4f}', 
                                   ha='left', va='center', fontsize=9, 
                                   fontweight='bold', color='#2c3e50',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
                        
                        plt.tight_layout()
                        html.append(_img_to_base64(fig, width=800))

            # 4. Évaluation
            if "Évaluation" in report_sections and "evaluation_metrics" in session_state:
                html.append("<h2>📈 4. Évaluation des performances</h2>")
                
                em = session_state["evaluation_metrics"]
                try:
                    em_display = em.round(4)
                except:
                    em_display = em
                
                html.append("<h3>🎯 Métriques de performance</h3>")
                html.append(em_display.to_html(index=False, classes='dataframe'))
                
                # Visualisations d'évaluation
                if "Visualisations" in report_sections and include_plots:
                    html.append("<h3>📊 Visualisations des performances</h3>")
                    
                    if "y_pred" in session_state and "y_test" in session_state:
                        y_test = session_state["y_test"]
                        y_pred = session_state["y_pred"]
                        
                        # Déterminer si c'est classification ou régression
                        is_classification = task_type == "classification" or (hasattr(y_test, 'dtype') and y_test.dtype == 'object')
                        
                        if is_classification:
                            # Matrice de confusion
                            from sklearn.metrics import confusion_matrix
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            cm = confusion_matrix(y_test, y_pred)
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                      xticklabels=np.unique(y_test),
                                      yticklabels=np.unique(y_test),
                                      cbar_kws={'label': 'Nombre de prédictions'})
                            ax.set_xlabel('Prédictions', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Valeurs réelles', fontsize=12, fontweight='bold')
                            ax.set_title('Matrice de Confusion', fontsize=14, fontweight='bold', pad=20)
                            plt.tight_layout()
                            html.append(_img_to_base64(fig, width=800))
                            
                        else:
                            # Régression: Prédit vs Réel + Résidus
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                            
                            # Prédit vs Réel
                            ax1.scatter(y_test, y_pred, alpha=0.6, s=50, c='#667eea', edgecolors='white', linewidth=0.5)
                            min_val = min(y_test.min(), y_pred.min())
                            max_val = max(y_test.max(), y_pred.max())
                            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prédiction parfaite')
                            ax1.set_xlabel('Valeurs réelles', fontsize=12, fontweight='bold')
                            ax1.set_ylabel('Prédictions', fontsize=12, fontweight='bold')
                            ax1.set_title('Prédictions vs Valeurs Réelles', fontsize=14, fontweight='bold', pad=15)
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # Résidus
                            residuals = y_test - y_pred
                            ax2.scatter(y_pred, residuals, alpha=0.6, s=50, c='#764ba2', edgecolors='white', linewidth=0.5)
                            ax2.axhline(y=0, color='r', linestyle='--', lw=2)
                            ax2.set_xlabel('Prédictions', fontsize=12, fontweight='bold')
                            ax2.set_ylabel('Résidus (Réel - Prédit)', fontsize=12, fontweight='bold')
                            ax2.set_title('Analyse des Résidus', fontsize=14, fontweight='bold', pad=15)
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            html.append(_img_to_base64(fig, width=1200))
                            
                            # Distribution des résidus
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(residuals, kde=True, ax=ax, color='#667eea', bins=30)
                            ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Résidu = 0')
                            ax.set_xlabel('Résidus', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Fréquence', fontsize=12, fontweight='bold')
                            ax.set_title('Distribution des Résidus', fontsize=14, fontweight='bold', pad=15)
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            html.append(_img_to_base64(fig, width=900))

            # Footer
            html.append('<div class="footer">')
            html.append('<h3>📋 Résumé du rapport</h3>')
            html.append(f'<p><strong>Date de génération :</strong> {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}</p>')
            if "data" in session_state:
                html.append(f'<p><strong>Dataset initial :</strong> {session_state["data"].shape[0]:,} lignes × {session_state["data"].shape[1]} colonnes</p>')
            if "clean_data" in session_state:
                html.append(f'<p><strong>Dataset nettoyé :</strong> {session_state["clean_data"].shape[0]:,} lignes × {session_state["clean_data"].shape[1]} colonnes</p>')
            html.append(f'<p><strong>Modèle :</strong> {model_name}</p>')
            html.append(f'<p><strong>Type de tâche :</strong> {task_type.capitalize()}</p>')
            html.append('<p style="margin-top:20px; color:#95a5a6;">Rapport généré automatiquement par Data Tool v2.2</p>')
            html.append('</div>')
            
            html.append('</div>')  # Fermeture container
            html.append("</body></html>")
            
            # Sauvegarde
            safe_title = title.replace(" ", "_").replace("/", "-")
            out_path = os.path.join(OUT_DIR, f"{safe_title}.html")
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(html))
            
            st.success(f"✅ Rapport généré avec succès !")
            
            # Afficher un aperçu
            with st.expander("📄 Aperçu du rapport", expanded=False):
                st.info(f"**Chemin :** `{out_path}`")
                st.info(f"**Taille :** {os.path.getsize(out_path) / 1024:.1f} KB")
                st.info(f"**Sections incluses :** {', '.join(report_sections)}")
            
            # Bouton de téléchargement
            with open(out_path, "rb") as f:
                st.download_button(
                    label="📥 Télécharger le rapport HTML",
                    data=f,
                    file_name=f"{safe_title}.html",
                    mime="text/html",
                    type="primary"
                )