# modules/model_comparison.py
"""
Module de comparaison de plusieurs modèles ML
Permet d'entraîner et comparer simultanément plusieurs algorithmes
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import time
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import helpers
import metrics
import model_utils  # Import du nouveau module d'utilitaires


class ModelComparator:
    """Classe pour comparer plusieurs modèles ML"""
    
    def __init__(self, task: str = "classification"):
        """
        Initialise le comparateur de modèles
        
        Args:
            task: Type de tâche ("classification" ou "regression")
        """
        self.task = task
        self.results = []
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf if task == "classification" else np.inf
        
    def get_available_models(self) -> Dict[str, Any]:
        """Retourne les modèles disponibles selon la tâche"""
        if self.task == "classification":
            return {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "Extra Trees": ExtraTreesClassifier(random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "SVM": SVC(probability=True, random_state=42),
                "Naive Bayes": GaussianNB()
            }
        else:  # regression
            return {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(random_state=42),
                "Lasso": Lasso(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "Extra Trees": ExtraTreesRegressor(random_state=42),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "SVR": SVR()
            }
    
    def build_preprocessor(self, X: pd.DataFrame, do_scale: bool = True) -> ColumnTransformer:
        """Construit le preprocesseur pour les données"""
        # Utilisation de la fonction commune de model_utils
        return model_utils.build_preprocessor(X, do_scale)
    
    def train_and_evaluate(
        self,
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        preprocessor: ColumnTransformer,
        use_cv: bool = False,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Entraîne et évalue un modèle
        
        Returns:
            Dictionnaire avec les résultats
        """
        start_time = time.time()
        
        # Créer le pipeline
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        # Entraînement
        try:
            pipe.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Prédictions
            y_pred = pipe.predict(X_test)
            
            # Métriques
            if self.task == "classification":
                metrics_result = metrics.classification_metrics(y_test, y_pred)
                primary_metric = metrics_result.get("accuracy", 0)
            else:
                metrics_result = metrics.regression_metrics(y_test, y_pred)
                primary_metric = metrics_result.get("r2", 0)
            
            # Cross-validation si demandé
            cv_score = None
            if use_cv:
                try:
                    cv_scores = cross_val_score(
                        pipe, X_train, y_train,
                        cv=cv_folds,
                        scoring='accuracy' if self.task == "classification" else 'r2'
                    )
                    cv_score = cv_scores.mean()
                except Exception as e:
                    st.warning(f"⚠️ CV échoué pour {model_name}: {str(e)}")
                    cv_score = None
            
            result = {
                "model_name": model_name,
                "pipeline": pipe,
                "metrics": metrics_result,
                "primary_metric": primary_metric,
                "cv_score": cv_score,
                "training_time": training_time,
                "predictions": y_pred,
                "status": "success"
            }
            
            # Mise à jour du meilleur modèle
            if self.task == "classification":
                if primary_metric > self.best_score:
                    self.best_score = primary_metric
                    self.best_model = model_name
            else:
                if primary_metric > self.best_score:
                    self.best_score = primary_metric
                    self.best_model = model_name
            
            return result
            
        except Exception as e:
            return {
                "model_name": model_name,
                "pipeline": None,
                "metrics": {},
                "primary_metric": 0,
                "cv_score": None,
                "training_time": time.time() - start_time,
                "predictions": None,
                "status": "failed",
                "error": str(e)
            }
    
    def compare_models(
        self,
        selected_models: List[str],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        do_scale: bool = True,
        use_cv: bool = False,
        cv_folds: int = 5
    ) -> pd.DataFrame:
        """
        Compare plusieurs modèles
        
        Returns:
            DataFrame avec les résultats de comparaison
        """
        available_models = self.get_available_models()
        preprocessor = self.build_preprocessor(X_train, do_scale)
        
        self.results = []
        self.models = {}
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, model_name in enumerate(selected_models):
            status_text.text(f"Entraînement de {model_name}... ({idx+1}/{len(selected_models)})")
            
            model = available_models[model_name]
            result = self.train_and_evaluate(
                model_name, model,
                X_train, X_test, y_train, y_test,
                preprocessor, use_cv, cv_folds
            )
            
            self.results.append(result)
            if result["status"] == "success":
                self.models[model_name] = result["pipeline"]
            
            progress_bar.progress((idx + 1) / len(selected_models))
        
        status_text.text("✅ Entraînement terminé!")
        progress_bar.empty()
        
        # Créer le DataFrame de résultats
        return self._create_results_dataframe()
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Crée un DataFrame avec les résultats de comparaison"""
        rows = []
        
        for result in self.results:
            if result["status"] == "success":
                row = {
                    "Modèle": result["model_name"],
                    "Temps (s)": round(result["training_time"], 2),
                    "Statut": "✅ Succès"
                }
                
                # Ajouter les métriques
                for metric_name, metric_value in result["metrics"].items():
                    try:
                        row[metric_name.upper()] = round(float(metric_value), 4)
                    except:
                        row[metric_name.upper()] = metric_value
                
                # Ajouter CV score si disponible
                if result["cv_score"] is not None:
                    row["CV Score"] = round(result["cv_score"], 4)
                
                rows.append(row)
            else:
                rows.append({
                    "Modèle": result["model_name"],
                    "Temps (s)": round(result["training_time"], 2),
                    "Statut": f"❌ Échec: {result.get('error', 'Unknown')}"
                })
        
        df = pd.DataFrame(rows)
        
        # Trier par métrique principale
        if self.task == "classification" and "ACCURACY" in df.columns:
            df = df.sort_values("ACCURACY", ascending=False)
        elif self.task == "regression" and "R2" in df.columns:
            df = df.sort_values("R2", ascending=False)
        
        return df.reset_index(drop=True)
    
    def plot_comparison(self, results_df: pd.DataFrame) -> plt.Figure:
        """Crée un graphique de comparaison des modèles"""
        # Filtrer les modèles réussis
        success_df = results_df[results_df["Statut"] == "✅ Succès"].copy()
        
        if len(success_df) == 0:
            st.warning("Aucun modèle n'a réussi l'entraînement")
            return None
        
        # Déterminer la métrique principale
        if self.task == "classification":
            metric_col = "ACCURACY" if "ACCURACY" in success_df.columns else success_df.columns[2]
        else:
            metric_col = "R2" if "R2" in success_df.columns else success_df.columns[2]
        
        # Créer le graphique
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Graphique 1: Métrique principale
        colors = ['#2ecc71' if model == self.best_model else '#3498db' 
                  for model in success_df["Modèle"]]
        
        ax1.barh(success_df["Modèle"], success_df[metric_col], color=colors)
        ax1.set_xlabel(metric_col)
        ax1.set_title(f"Comparaison - {metric_col}")
        ax1.grid(axis='x', alpha=0.3)
        
        # Graphique 2: Temps d'entraînement
        ax2.barh(success_df["Modèle"], success_df["Temps (s)"], color='#e74c3c')
        ax2.set_xlabel("Temps (secondes)")
        ax2.set_title("Temps d'Entraînement")
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_heatmap(self, results_df: pd.DataFrame) -> plt.Figure:
        """Crée une heatmap des métriques pour tous les modèles"""
        success_df = results_df[results_df["Statut"] == "✅ Succès"].copy()
        
        if len(success_df) == 0:
            return None
        
        # Sélectionner uniquement les colonnes numériques (métriques)
        metric_cols = success_df.select_dtypes(include=[np.number]).columns.tolist()
        metric_cols = [col for col in metric_cols if col != "Temps (s)"]
        
        if len(metric_cols) == 0:
            return None
        
        # Créer la heatmap
        fig, ax = plt.subplots(figsize=(10, len(success_df) * 0.5 + 2))
        
        heatmap_data = success_df[["Modèle"] + metric_cols].set_index("Modèle")
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0.5,
            ax=ax,
            cbar_kws={'label': 'Score'}
        )
        
        ax.set_title("Heatmap des Métriques par Modèle")
        plt.tight_layout()
        return fig
    
    def save_best_model(self, target: str, output_dir: str = "outputs/models") -> str:
        """Sauvegarde le meilleur modèle"""
        if self.best_model is None or self.best_model not in self.models:
            raise ValueError("Aucun meilleur modèle trouvé")
        
        helpers.ensure_dir(output_dir)
        model_path = f"{output_dir}/best_model_{target}.pkl"
        joblib.dump(self.models[self.best_model], model_path)
        
        return model_path
    
    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Retourne les détails d'un modèle spécifique"""
        for result in self.results:
            if result["model_name"] == model_name:
                return result
        return None


def run_model_comparison(df: pd.DataFrame) -> dict:
    """
    Interface Streamlit pour la comparaison de modèles
    
    Args:
        df: DataFrame avec les données
        
    Returns:
        Dictionnaire avec les résultats
    """
    st.subheader("🔬 Comparaison de Modèles ML")
    
    # Sélection de la variable cible
    cols = df.columns.tolist()
    target = st.selectbox("Choisir la variable cible", [""] + cols, key="comparison_target")
    
    if not target:
        st.info("Sélectionnez une variable cible pour commencer la comparaison.")
        st.stop()
    
    X = df.drop(columns=[target])
    y = df[target]
    
    # Validation de la cible avec model_utils
    y, valid_idx = model_utils.validate_and_clean_target(y, target)
    if not valid_idx.all():
        X = X[valid_idx].reset_index(drop=True)
    
    # Détection automatique de la tâche avec model_utils
    task = model_utils.detect_task_type(y)
    st.info(f"📊 Tâche détectée : **{task.upper()}**")
    
    # Afficher les statistiques avec model_utils
    model_utils.display_target_stats(y, task)
    
    # Configuration
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Taille test (%)", 5, 50, 20, key="comp_test_size") / 100.0
        do_scale = st.checkbox("Standardiser les variables numériques", value=True, key="comp_scale")
    
    with col2:
        random_state = int(st.number_input("Seed aléatoire", value=42, key="comp_seed"))
        use_cv = st.checkbox("Utiliser la validation croisée", value=False, key="comp_cv")
        if use_cv:
            cv_folds = int(st.number_input("Nombre de folds", 3, 10, 5, key="comp_cv_folds"))
        else:
            cv_folds = 5
    
    # Sélection des modèles
    st.markdown("### 🎯 Sélection des Modèles")
    
    comparator = ModelComparator(task=task)
    available_models = list(comparator.get_available_models().keys())
    
    # Initialiser selected_models dans session_state si nécessaire
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = ["Random Forest", "Gradient Boosting"]
    
    # Options de sélection rapide
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Tout sélectionner"):
            st.session_state.selected_models = available_models
            st.rerun()
    with col2:
        if st.button("🚀 Modèles rapides"):
            # Utilisation de la fonction commune
            st.session_state.selected_models = model_utils.get_fast_models(task)
            st.rerun()
    with col3:
        if st.button("❌ Tout désélectionner"):
            st.session_state.selected_models = []
            st.rerun()
    
    # Multiselect pour les modèles
    selected_models = st.multiselect(
        "Choisir les modèles à comparer",
        available_models,
        default=st.session_state.selected_models,
        key="model_multiselect"
    )
    
    # Mettre à jour session_state avec la sélection actuelle
    st.session_state.selected_models = selected_models
    
    if len(selected_models) == 0:
        st.warning("⚠️ Veuillez sélectionner au moins un modèle")
        st.stop()
    
    st.info(f"📊 {len(selected_models)} modèle(s) sélectionné(s)")
    
    # Bouton de lancement
    if st.button("🚀 Lancer la Comparaison", type="primary"):
        with st.spinner("Entraînement en cours..."):
            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Comparaison
            results_df = comparator.compare_models(
                selected_models,
                X_train, X_test, y_train, y_test,
                do_scale=do_scale,
                use_cv=use_cv,
                cv_folds=cv_folds
            )
            
            # Affichage des résultats
            st.markdown("---")
            st.markdown("## 📊 Résultats de la Comparaison")
            
            # Tableau des résultats
            st.dataframe(
                results_df.style.highlight_max(
                    subset=[col for col in results_df.columns if col not in ["Modèle", "Temps (s)", "Statut"]],
                    color='lightgreen'
                ),
                use_container_width=True
            )
            
            # Meilleur modèle
            if comparator.best_model:
                st.success(f"🏆 **Meilleur modèle** : {comparator.best_model} (Score: {comparator.best_score:.4f})")
            
            # Graphiques de comparaison
            st.markdown("### 📈 Visualisations")
            
            tab1, tab2 = st.tabs(["📊 Comparaison", "🔥 Heatmap"])
            
            with tab1:
                fig = comparator.plot_comparison(results_df)
                if fig:
                    st.pyplot(fig)
            
            with tab2:
                fig = comparator.plot_metrics_heatmap(results_df)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("Pas assez de métriques pour créer une heatmap")
            
            # Export des résultats
            st.markdown("### 💾 Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export CSV des résultats
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Télécharger les résultats (CSV)",
                    csv,
                    f"comparison_results_{target}.csv",
                    "text/csv"
                )
            
            with col2:
                # Sauvegarder le meilleur modèle
                if st.button("💾 Sauvegarder le meilleur modèle"):
                    try:
                        model_path = comparator.save_best_model(target)
                        st.success(f"✅ Meilleur modèle sauvegardé : {model_path}")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la sauvegarde : {str(e)}")
            
            # Stocker dans session_state de manière cohérente avec model_utils
            best_model_pipeline = comparator.models.get(comparator.best_model)
            
            # Stocker le modèle avec la fonction commune
            model_utils.store_model_in_session(
                model=best_model_pipeline,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                task=task,
                model_name=comparator.best_model
            )
            
            # Stocker les résultats de comparaison
            st.session_state.update({
                "comparison_results": results_df,
                "comparator": comparator,
                "best_model": best_model_pipeline
            })
            
            return {
                "results": results_df,
                "best_model": comparator.best_model,
                "comparator": comparator
            }
    
    st.stop()
