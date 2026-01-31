# modules/advanced_evaluation.py
"""
Module d'évaluation avancée des modèles ML
Ajoute des métriques sophistiquées : SHAP, déciles, learning curves, etc.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Optional imports avec gestion d'erreur
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def shap_analysis(model, X_test, feature_names=None, max_display=20):
    """Analyse SHAP pour l'interprétabilité du modèle"""
    if not SHAP_AVAILABLE:
        st.warning("⚠️ SHAP n'est pas installé. Installez avec : `pip install shap`")
        return
    
    with st.spinner("🔍 Calcul des valeurs SHAP..."):
        try:
            # Extraire le préprocesseur et le modèle
            if hasattr(model, 'named_steps'):
                preprocessor = model.named_steps.get('preprocessor')
                model_core = model.named_steps.get('model')
                X_processed = preprocessor.transform(X_test)
                
                if feature_names is None:
                    try:
                        feature_names = preprocessor.get_feature_names_out()
                    except:
                        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                
                # Créer l'explainer SHAP
                if hasattr(model_core, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model_core)
                else:
                    explainer = shap.KernelExplainer(model_core, X_processed[:100])
                
                shap_values = explainer.shap_values(X_processed)
                
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # Classe positive
                
                st.success("✅ Analyse SHAP calculée !")
                
                # Importance globale
                shap_importance = np.abs(shap_values).mean(0)
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': shap_importance
                }).sort_values('importance', ascending=False).head(max_display)
                
                fig = px.bar(feature_importance_df, x='importance', y='feature',
                            title="Importance Globale des Features (SHAP)", orientation='h')
                fig.update_layout(height=max(400, len(feature_importance_df) * 25))
                st.plotly_chart(fig, use_container_width=True)
                
                return shap_values, feature_names
                
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse SHAP : {str(e)}")
            return None, None

def decile_analysis(y_true, y_pred, y_pred_proba=None, task_type="classification"):
    """Analyse par déciles des performances du modèle"""
    
    try:
        if task_type == "classification" and y_pred_proba is not None:
            # Classification : analyser par déciles de probabilité
            df_analysis = pd.DataFrame({
                'true': y_true,
                'pred_proba': y_pred_proba,
                'pred': y_pred
            })
            
            # Découper en déciles
            df_analysis['decile'] = pd.qcut(df_analysis['pred_proba'], 10, labels=False, duplicates='drop')
            
            # Convertir les labels en numérique pour le calcul
            # Gérer le cas où y_true peut être textuel
            try:
                # Si les labels sont textuels, les convertir en binaire (positif/négatif)
                unique_labels = df_analysis['true'].unique()
                if len(unique_labels) == 2:
                    # Cas binaire : convertir en 0/1
                    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                    df_analysis['true_numeric'] = df_analysis['true'].map(label_map)
                else:
                    # Cas multi-classe : utiliser LabelEncoder
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df_analysis['true_numeric'] = le.fit_transform(df_analysis['true'])
            except:
                # En cas d'erreur, créer une colonne binaire simple
                df_analysis['true_numeric'] = (df_analysis['true'] == df_analysis['true'].iloc[0]).astype(int)
            
            # Calculer les métriques par décile
            decile_stats = df_analysis.groupby('decile').agg({
                'true': ['count'],  # Juste le count pour les labels originaux
                'true_numeric': ['sum', 'mean'],  # Calculs sur la version numérique
                'pred_proba': ['mean', 'min', 'max']
            }).round(4)
            
            decile_stats.columns = ['Nb_observations', 'Nb_positifs', 'Taux_positif_reel', 
                                   'Probabilite_moyenne', 'Prob_min', 'Prob_max']
            
            st.subheader("📊 Analyse par Déciles (Classification)")
            st.dataframe(decile_stats, use_container_width=True)
            
            # Visualisation
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Distribution par décile", "Taux de positifs par décile",
                              "Probabilités moyennes", "Effectifs par décile"),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Effectifs par décile
            fig.add_trace(
                go.Bar(x=decile_stats.index, y=decile_stats['Nb_observations'], 
                       name="Effectifs", marker_color='lightblue'),
                row=1, col=1
            )
            
            # Taux de positifs
            fig.add_trace(
                go.Scatter(x=decile_stats.index, y=decile_stats['Taux_positif_reel'], 
                          mode='lines+markers', name="Taux positifs réel", line=dict(color='red')),
                row=1, col=2
            )
            
            # Probabilités moyennes
            fig.add_trace(
                go.Bar(x=decile_stats.index, y=decile_stats['Probabilite_moyenne'], 
                       name="Prob moyenne", marker_color='lightgreen'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Régression : analyser par déciles de valeurs prédites
            df_analysis = pd.DataFrame({
                'true': y_true,
                'pred': y_pred,
                'error': np.abs(y_true - y_pred)
            })
            
            df_analysis['decile'] = pd.qcut(df_analysis['pred'], 10, labels=False, duplicates='drop')
            
            decile_stats = df_analysis.groupby('decile').agg({
                'true': ['mean', 'std'],
                'pred': ['mean', 'std'],
                'error': ['mean', 'std']
            }).round(4)
            
            decile_stats.columns = ['Vrai_moyenne', 'Vrai_std', 
                                   'Pred_moyenne', 'Pred_std',
                                   'Erreur_moyenne', 'Erreur_std']
            
            st.subheader("📊 Analyse par Déciles (Régression)")
            st.dataframe(decile_stats, use_container_width=True)
            
            # Visualisation
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Valeurs moyennes par décile", "Erreurs moyennes par décile")
            )
            
            fig.add_trace(
                go.Scatter(x=decile_stats.index, y=decile_stats['Vrai_moyenne'], 
                          mode='lines+markers', name="Vrai", line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=decile_stats.index, y=decile_stats['Pred_moyenne'], 
                          mode='lines+markers', name="Prédit", line=dict(color='red')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=decile_stats.index, y=decile_stats['Erreur_moyenne'], 
                       name="Erreur moyenne", marker_color='orange'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse par déciles : {str(e)}")
        st.info("💡 **Solution possible** : Vérifiez que vos données contiennent des valeurs numériques valides")

def learning_curve_analysis(model, X, y, cv=5):
    """Analyse des courbes d'apprentissage"""
    
    with st.spinner("📈 Calcul des learning curves..."):
        try:
            # Déterminer le scoring en fonction du type de problème
            if hasattr(y, 'nunique'):
                unique_count = y.nunique()
            elif hasattr(y, 'unique'):
                unique_count = len(np.unique(y))
            else:
                unique_count = len(np.unique(y))
            
            scoring = 'accuracy' if unique_count < 20 else 'r2'
            
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=cv, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring=scoring
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig = go.Figure()
            
            # Courbe d'entraînement
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_mean,
                mode='lines+markers',
                name='Entraînement',
                line=dict(color='blue'),
                error_y=dict(type='data', array=train_std, visible=True)
            ))
            
            # Courbe de validation
            fig.add_trace(go.Scatter(
                x=train_sizes, y=val_mean,
                mode='lines+markers',
                name='Validation',
                line=dict(color='red'),
                error_y=dict(type='data', array=val_std, visible=True)
            ))
            
            fig.update_layout(
                title="Courbes d'Apprentissage",
                xaxis_title="Taille de l'ensemble d'entraînement",
                yaxis_title="Score",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interprétation
            overfitting = train_mean[-1] - val_mean[-1]
            if overfitting > 0.1:
                st.warning("⚠️ Signes d'overfitting détectés ! L'écart entre entraînement et validation est de {:.3f}".format(overfitting))
            elif overfitting < -0.05:
                st.info("ℹ️ Le modèle pourrait être sous-ajusté. Considérez un modèle plus complexe.")
            else:
                st.success("✅ Bon équilibre biais-variance !")
                
        except Exception as e:
            st.error(f"❌ Erreur lors du calcul des learning curves : {str(e)}")

def calibration_plot(y_true, y_pred_proba):
    """Graphe de calibration pour les probabilités"""
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    
    fig = go.Figure()
    
    # Courbe de calibration
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true,
        mode='lines+markers',
        name='Calibration',
        line=dict(color='blue', width=2)
    ))
    
    # Courbe parfaite
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Parfait',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title="Graphe de Calibration",
        xaxis_title="Probabilité prédite",
        yaxis_title="Fréquence observée",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def run_advanced_evaluation(model, X_test, y_test, task_type="classification"):
    """Interface principale pour l'évaluation avancée"""
    
    try:
        st.markdown("## 🔬 Évaluation Avancée du Modèle")
        
        # Prédictions avec gestion d'erreur
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
            return
        
        y_pred_proba = None
        if hasattr(model, "predict_proba") and task_type == "classification":
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except Exception as e:
                st.warning(f"⚠️ Erreur lors du calcul des probabilités : {str(e)}")
        
        # Onglets pour les différentes analyses
        tabs = ["📊 Déciles", "📈 Learning Curves", "🎯 SHAP", "📋 Calibration"]
        if task_type != "classification":
            tabs.remove("📋 Calibration")
        
        selected_tab = st.tabs(tabs)
        
        # Analyse par déciles
        with selected_tab[0]:
            decile_analysis(y_test, y_pred, y_pred_proba, task_type)
        
        # Learning curves
        with selected_tab[1]:
            st.write("� **Analyse des Learning Curves**")
            st.info("💡 Les learning curves aident à détecter l'overfitting et l'underfitting")
            
            # Bouton simple et direct
            if st.button("� Générer les Learning Curves", key="learning_curves_btn", help="Génère les courbes d'apprentissage"):
                with st.spinner("� Calcul des learning curves en cours..."):
                    try:
                        # Récupérer les données d'entraînement depuis session_state
                        X_train = st.session_state.get("X_train", X_test)
                        y_train = st.session_state.get("y_train", y_test)
                        
                        # Validation des données
                        if X_train is None or y_train is None:
                            st.error("❌ Données d'entraînement non disponibles")
                        else:
                            learning_curve_analysis(model, X_train, y_train)
                            st.success("✅ **Learning Curves générées avec succès !**")
                            
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération des learning curves : {str(e)}")
        
        # SHAP
        with selected_tab[2]:
            st.write("🎯 **Analyse SHAP (SHapley Additive exPlanations)**")
            st.info("� SHAP explique l'impact de chaque feature sur les prédictions du modèle")
            
            # Vérifier si SHAP est disponible
            if not SHAP_AVAILABLE:
                st.warning("⚠️ SHAP n'est pas installé. Installez-le avec : `pip install shap`")
                st.code("pip install shap")
            else:
                # Bouton simple et direct
                if st.button("🔍 Analyser avec SHAP", key="shap_analysis_btn", help="Génère l'analyse SHAP"):
                    with st.spinner("� Analyse SHAP en cours..."):
                        try:
                            # Récupérer les noms de features si disponibles
                            feature_names = None
                            if hasattr(X_test, 'columns'):
                                feature_names = X_test.columns.tolist()
                            
                            shap_analysis(model, X_test, feature_names)
                            st.success("✅ **Analyse SHAP terminée avec succès !**")
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'analyse SHAP : {str(e)}")
        
        # Calibration (classification seulement)
        if task_type == "classification" and len(selected_tab) > 3:
            with selected_tab[3]:
                if y_pred_proba is not None:
                    calibration_plot(y_test, y_pred_proba)
                else:
                    st.warning("⚠️ Le modèle ne fournit pas de probabilités")
    
    except Exception as e:
        st.error(f"❌ Erreur générale dans l'évaluation avancée : {str(e)}")
        st.info("💡 **Conseil** : Vérifiez que votre modèle et vos données sont valides")
