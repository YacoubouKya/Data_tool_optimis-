# evaluation.py
# modules/evaluation.py


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder

def run_evaluation(X_test, y_test):
    """
    Interface d'évaluation du modèle avec sélection du modèle à évaluer
    """
    # Récupérer les modèles disponibles
    refined_model = st.session_state.get("model", None)
    best_model = st.session_state.get("best_model", None)
    refined_model_name = st.session_state.get("current_model_name", None)
    best_model_name = st.session_state.get("best_model_name", None)
    best_model_score = st.session_state.get("best_model_score", None)
    
    # Déterminer quel modèle évaluer
    model_to_evaluate = None
    model_display_name = None
    
    # Si les deux modèles sont disponibles, proposer un choix
    if refined_model is not None and best_model is not None and refined_model_name != best_model_name:
        st.info(" Vous avez affiné un modèle après la comparaison. Quel modèle souhaitez-vous évaluer ?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f" **Modèle affiné**\n{refined_model_name}", use_container_width=True, type="primary"):
                st.session_state["selected_eval_model"] = "refined"
        with col2:
            score_text = f" (Score: {best_model_score:.4f})" if best_model_score else ""
            if st.button(f" **Meilleur modèle**\n{best_model_name}{score_text}", use_container_width=True):
                st.session_state["selected_eval_model"] = "best"
        
        # Déterminer le modèle sélectionné (par défaut: modèle affiné)
        selected = st.session_state.get("selected_eval_model", "refined")
        if selected == "refined":
            model_to_evaluate = refined_model
            model_display_name = refined_model_name
        else:
            model_to_evaluate = best_model
            model_display_name = best_model_name
    
    # Sinon, utiliser le modèle disponible
    elif refined_model is not None:
        model_to_evaluate = refined_model
        model_display_name = refined_model_name or "Modèle affiné"
    elif best_model is not None:
        model_to_evaluate = best_model
        model_display_name = best_model_name or "Meilleur modèle"
    else:
        st.error("❌ Aucun modèle disponible pour l'évaluation.")
        return
    
    # Afficher le modèle en cours d'évaluation
    st.success(f" **Modèle évalué** : {model_display_name}")
    st.markdown("---")
    
    preds = model_to_evaluate.predict(X_test)

    # Détection automatique du type
    is_classification = y_test.dtype == "object" or y_test.nunique() < 20

    if is_classification:
        st.write("Classification — métriques :")
        metrics_result = metrics.classification_metrics(y_test, preds)
        metrics_df = pd.DataFrame([metrics_result])
        st.dataframe(metrics_df)
        st.session_state["evaluation_metrics"] = metrics_df
        st.session_state["y_pred"] = preds

        st.write(" Choisir graphiques :")
        show_cm = st.checkbox("Matrice de confusion", True)
        show_roc = st.checkbox("Courbe ROC", True)
        show_pr = st.checkbox("Courbe Précision-Rappel", False)

        if show_cm:
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=np.unique(y_test),
                        yticklabels=np.unique(y_test))
            ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
            st.pyplot(fig)

        # Cas binaire uniquement pour ROC / PR
        if len(np.unique(y_test)) == 2:
            # Encodage des labels texte en 0/1 si nécessaire
            le = LabelEncoder()
            y_true_encoded = le.fit_transform(y_test)
            
            if hasattr(model_to_evaluate, "predict_proba"):
                proba = model_to_evaluate.predict_proba(X_test)[:, 1]
            else:
                # fallback si pas de predict_proba
                proba = preds if np.issubdtype(preds.dtype, np.number) else y_true_encoded

            if show_roc:
                fpr, tpr, _ = roc_curve(y_true_encoded, proba, pos_label=1)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
                ax.plot([0,1],[0,1],"--",color="gray")
                ax.set_xlabel("Faux positifs"); ax.set_ylabel("Vrais positifs"); ax.legend()
                st.pyplot(fig)

            if show_pr:
                precision, recall, _ = precision_recall_curve(y_true_encoded, proba, pos_label=1)
                fig, ax = plt.subplots()
                ax.plot(recall, precision, label="PR Curve")
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Courbe PR")
                st.pyplot(fig)

    else:
        st.write("Régression — métriques :")
        metrics_result = metrics.regression_metrics(y_test, preds)
        metrics_df = pd.DataFrame([metrics_result])
        st.dataframe(metrics_df)
        st.session_state["evaluation_metrics"] = metrics_df
        st.session_state["y_pred"] = preds
        residuals = y_test - preds
        st.session_state["residuals"] = residuals

        st.write(" Choisir graphiques :")
        show_scatter = st.checkbox("Prédit vs Réel", True)
        show_resid = st.checkbox("Résidus vs Prédit", True)
        show_hist = st.checkbox("Histogramme des résidus", True)
        show_qq = st.checkbox("QQ-plot résidus", False)

        if show_scatter:
            fig,ax=plt.subplots(figsize=(6,4))
            ax.scatter(y_test,preds,alpha=0.6)
            ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"r--")
            ax.set_xlabel("Réel"); ax.set_ylabel("Prédit"); ax.set_title("Prédit vs Réel")
            st.pyplot(fig)
        if show_resid:
            fig,ax=plt.subplots(figsize=(6,4))
            ax.scatter(preds,residuals,alpha=0.6)
            ax.axhline(0,color="red",linestyle="--")
            ax.set_xlabel("Prédit"); ax.set_ylabel("Résidu (y-ŷ)"); ax.set_title("Résidus vs Prédit")
            st.pyplot(fig)
        if show_hist:
            fig,ax=plt.subplots(figsize=(6,4))
            sns.histplot(residuals,bins=30,kde=True,ax=ax); ax.set_title("Histogramme des résidus")
            st.pyplot(fig)
        if show_qq:
            fig,ax=plt.subplots(figsize=(6,4))
            stats.probplot(residuals,dist="norm",plot=ax); ax.set_title("QQ-plot des résidus")

            st.pyplot(fig)
