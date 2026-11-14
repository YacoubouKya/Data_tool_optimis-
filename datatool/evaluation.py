# evaluation.py
# modules/evaluation.py


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from . import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from .advanced_evaluation import run_advanced_evaluation

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

    if best_model is not None:
        # Si on a un best_model (venant de la comparaison), c'est le modèle principal
        if refined_model is not None and refined_model_name != best_model_name:
            # Cas : l'utilisateur a fait une comparaison puis un affinage
            st.info("Vous avez affiné un modèle après la comparaison. Quel modèle souhaitez-vous évaluer ?")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"**Modèle affiné**\n{refined_model_name}", use_container_width=True, type="primary"):
                    st.session_state["selected_eval_model"] = "refined"
            with col2:
                score_text = f" (Score: {best_model_score:.4f})" if best_model_score else ""
                if st.button(f"**Meilleur modèle de la comparaison**\n{best_model_name}{score_text}", use_container_width=True):
                    st.session_state["selected_eval_model"] = "best"

            # Déterminer le modèle sélectionné
            selected = st.session_state.get("selected_eval_model", "best")  # Par défaut: best_model
            if selected == "refined":
                model_to_evaluate = refined_model
                model_display_name = refined_model_name
            else:
                model_to_evaluate = best_model
                model_display_name = best_model_name
        else:
            # Cas : seulement le best_model disponible (après comparaison)
            model_to_evaluate = best_model
            model_display_name = best_model_name or "Meilleur modèle"
            st.success(f"**Modèle évalué** : {model_display_name}")

    elif refined_model is not None:
        # Cas : seulement un modèle affiné (sans comparaison préalable)
        model_to_evaluate = refined_model
        model_display_name = refined_model_name or "Modèle affiné"
        st.success(f"**Modèle évalué** : {model_display_name}")

    else:
        st.error("Aucun modèle disponible pour l'évaluation.")
        st.info("**Solution** : Entraînez d'abord un modèle via la comparaison ou l'affinage.")
        return

    # Afficher clairement le modèle en cours d'évaluation
    st.markdown("---")
    st.markdown(f"## Modèle évalué : {model_display_name}")

    # Ajouter des informations supplémentaires si disponible
    if best_model_score and model_display_name == best_model_name:
        st.info(f"**Meilleur modèle de la comparaison** avec un score de {best_model_score:.4f}")
    elif refined_model_name and model_display_name == refined_model_name:
        st.info("**Modèle affiné** avec hyperparamètres optimisés")

    st.markdown("---")

    preds = model_to_evaluate.predict(X_test)

    # Détection améliorée du type de problème
    # Vérifier d'abord si y_test est de type object (classification)
    if y_test.dtype == "object":
        is_classification = True
    # Sinon, vérifier si c'est numérique avec peu de valeurs uniques
    elif pd.api.types.is_numeric_dtype(y_test):
        unique_count = y_test.nunique()
        # Si moins de 20 valeurs uniques ET que ce sont des entiers, c probablement de la classification
        is_classification = unique_count < 20 and all(y_test.dropna() % 1 == 0)
    else:
        is_classification = False

    # Validation silencieuse du type (sans affichage de debug)
    if is_classification:
        # Validation supplémentaire pour la classification
        if not (y_test.dtype == "object" or (pd.api.types.is_numeric_dtype(y_test) and all(y_test.dropna() % 1 == 0))):
            # Forcer en régression si les données ne sont pas appropriées pour la classification
            is_classification = False
    else:
        # Validation pour la régression
        if not pd.api.types.is_numeric_dtype(y_test):
            # Forcer en classification si les données ne sont pas numériques
            is_classification = True

    if is_classification:
        st.write("Classification — métriques :")
        metrics_result = metrics.classification_metrics(y_test, preds)
        metrics_df = pd.DataFrame([metrics_result])
        st.dataframe(metrics_df)
        st.session_state["evaluation_metrics"] = metrics_df
        st.session_state["y_pred"] = preds

        st.write("Choisir graphiques :")
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

        st.write("Choisir graphiques :")
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

    # Section d'évaluation avancée
    st.markdown("---")
    st.markdown("## Évaluation Avancée")
    st.info("**Analyses sophistiquées** pour comprendre en profondeur votre modèle")

    # Bouton pour lancer l'évaluation avancée
    if st.button("Lancer l'Évaluation Avancée", type="primary"):
        task_type = "classification" if is_classification else "regression"
        run_advanced_evaluation(model_to_evaluate, X_test, y_test, task_type)
