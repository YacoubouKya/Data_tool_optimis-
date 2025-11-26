# modules/modeling.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from typing import Tuple, Any
import helpers
import metrics
import model_utils
from math import isfinite

def _format_metrics(d: dict, decimals=3):
    """Arrondit les valeurs numériques du dict pour l'affichage."""
    out = {}
    for k, v in d.items():
        try:
            fv = float(v)
            if not isfinite(fv):
                out[k] = v
            else:
                out[k] = round(fv, decimals)
        except Exception:
            out[k] = v
    return out

def run_modeling(df: pd.DataFrame) -> dict:
    st.subheader("⚡ Modélisation interactive")
    
    # Détecter si on vient de la comparaison
    from_comparison = "best_model_name" in st.session_state and "comparison_results" in st.session_state
    
    if from_comparison:
        st.success(f"🏆 **Meilleur modèle détecté** : {st.session_state['best_model_name']}")
        st.info("💡 Vous pouvez affiner ce modèle ou en choisir un autre")

    cols = df.columns.tolist()
    
    # Pré-remplir la cible si elle existe déjà
    default_target = ""
    if "y_train" in st.session_state and hasattr(st.session_state["y_train"], "name"):
        default_target = st.session_state["y_train"].name
    
    target_index = 0
    if default_target and default_target in cols:
        target_index = cols.index(default_target) + 1
    
    target = st.selectbox("Choisir la variable cible", [""] + cols, index=target_index)
    if not target:
        st.info("Sélectionne une variable cible pour lancer l'entraînement.")
        st.stop()

    X = df.drop(columns=[target])
    y = df[target]
    
    # Validation et nettoyage de la variable cible
    st.markdown("### 🔍 Validation des Données")
    
    # Vérifier les valeurs manquantes dans y
    y_missing = y.isna().sum()
    if y_missing > 0:
        st.warning(f"⚠️ Variable cible contient {y_missing} valeurs manquantes ({y_missing/len(y)*100:.1f}%)")
        
        action = st.radio(
            "Comment traiter les valeurs manquantes dans la cible ?",
            ["Supprimer les lignes", "Imputer (moyenne/mode)", "Annuler"],
            key="missing_target_action"
        )
        
        if action == "Annuler":
            st.info("Veuillez nettoyer vos données avant la modélisation")
            st.stop()
        elif action == "Supprimer les lignes":
            valid_idx = y.notna()
            X = X[valid_idx].reset_index(drop=True)
            y = y[valid_idx].reset_index(drop=True)
            st.success(f"✅ {y_missing} lignes supprimées. Nouvelles dimensions : {len(y)} lignes")
        else:  # Imputer
            if y.dtype in ['object', 'category']:
                mode_val = y.mode()[0] if not y.mode().empty else y.dropna().iloc[0]
                y = y.fillna(mode_val)
                st.success(f"✅ Valeurs manquantes imputées avec le mode : {mode_val}")
            else:
                mean_val = y.mean()
                y = y.fillna(mean_val)
                st.success(f"✅ Valeurs manquantes imputées avec la moyenne : {mean_val:.2f}")
    
    # Vérifier les valeurs infinies dans y (pour régression)
    if y.dtype in ['int64', 'float64']:
        y_inf = (~y.isna() & ((y == float('inf')) | (y == float('-inf')))).sum()
        if y_inf > 0:
            st.warning(f"⚠️ Variable cible contient {y_inf} valeurs infinies")
            y = y.replace([float('inf'), float('-inf')], pd.NA)
            y = y.fillna(y.median())
            st.success(f"✅ Valeurs infinies remplacées par la médiane")
    
    # Afficher statistiques de la cible
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes valides", len(y))
    with col2:
        st.metric("Valeurs uniques", y.nunique())
    with col3:
        if y.dtype in ['int64', 'float64']:
            st.metric("Moyenne", f"{y.mean():.2f}")
        else:
            st.metric("Mode", y.mode()[0] if not y.mode().empty else "N/A")
    
    st.markdown("---")

    task = st.selectbox("Type de tâche", ["auto", "classification", "regression"], index=0)
    if task == "auto":
        if y.dtype == "O" or (y.nunique() <= 20 and y.nunique()/len(y) < 0.1):
            task = "classification"
        else:
            task = "regression"
    st.write("👉 Tâche détectée :", task)

    test_size = st.slider("Taille test (%)", 5, 50, 20) / 100.0
    random_state = int(st.number_input("Seed aléatoire", value=42))

    # Si on vient de la comparaison, proposer les modèles testés
    if from_comparison and "comparison_results" in st.session_state:
        st.markdown("### 🎯 Sélection du Modèle")
        
        comparison_models = st.session_state["comparison_results"]["Modèle"].tolist()
        best_model_name = st.session_state.get("best_model_name", comparison_models[0])
        
        # Mapper les noms de la comparaison vers les choix de modeling
        model_mapping = {
            "Random Forest": "random_forest",
            "Gradient Boosting": "gradient_boosting",
            "Logistic Regression": "linear/logistic",
            "Linear Regression": "linear/logistic"
        }
        
        # Sélection avec le meilleur modèle par défaut
        model_display_choice = st.selectbox(
            "Choisir le modèle à affiner",
            comparison_models,
            index=comparison_models.index(best_model_name) if best_model_name in comparison_models else 0,
            help="Le meilleur modèle de la comparaison est sélectionné par défaut"
        )
        
        # Convertir vers le format de modeling.py
        model_choice = model_mapping.get(model_display_choice, "auto")
        
        st.info(f"💡 Modèle sélectionné : **{model_display_choice}**")
    else:
        model_choice = st.selectbox("Choisir un modèle", ["auto", "random_forest", "gradient_boosting", "linear/logistic"])
    
    do_scale = st.checkbox("⚙️ Standardiser les numériques", value=True)

    # Hyperparamètres exposés
    if model_choice in ["random_forest", "auto"]:
        rf_n_estimators = int(st.number_input("RF - n_estimators", 10, 1000, 100))
        rf_max_depth = int(st.number_input("RF - max_depth (0=>None)", 0, 50, 0))
    else:
        rf_n_estimators = 100; rf_max_depth = 0

    if model_choice in ["gradient_boosting", "auto"]:
        gb_n_estimators = int(st.number_input("GB - n_estimators", 10, 1000, 100))
        gb_max_depth = int(st.number_input("GB - max_depth", 1, 20, 3))
        gb_lr = float(st.number_input("GB - learning_rate", 0.01, 1.0, 0.1))
    else:
        gb_n_estimators = 100; gb_max_depth = 3; gb_lr = 0.1

    if st.button("🚀 Lancer l'entraînement"):
        # Préprocessing pipeline (construit sans boucle coûteuse)
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            X[cat_cols] = X[cat_cols].astype(str)

        num_steps = []
        if num_cols:
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if do_scale:
                num_steps.append(("scaler", StandardScaler()))

        cat_steps = []
        if cat_cols:
            cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]

        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline(num_steps), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline(cat_steps), cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

        # Choix du modèle (sans boucle)
        if model_choice == "auto":
            if task == "classification":
                model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth==0 else rf_max_depth, random_state=random_state)
            else:
                model = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth==0 else rf_max_depth, random_state=random_state)
        elif model_choice == "random_forest":
            model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth==0 else rf_max_depth, random_state=random_state) if task=="classification" else RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth==0 else rf_max_depth, random_state=random_state)
        elif model_choice == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=gb_n_estimators, max_depth=gb_max_depth, learning_rate=gb_lr, random_state=random_state) if task=="classification" else GradientBoostingRegressor(n_estimators=gb_n_estimators, max_depth=gb_max_depth, learning_rate=gb_lr, random_state=random_state)
        else:
            model = LogisticRegression(max_iter=1000) if task=="classification" else LinearRegression()

        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # Split & train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        # Évaluation (metrics utilitaires)
        if task == "classification":
            metrics_result = metrics.classification_metrics(y_test, preds)
        else:
            metrics_result = metrics.regression_metrics(y_test, preds)

        metrics_display = _format_metrics(metrics_result, decimals=4)
        st.write("📊 **Metrics (test)** :")
        st.json(metrics_display)

        # Sauvegarde modèle et datasets
        helpers.ensure_dir("outputs/models")
        model_path = f"outputs/models/model_{target}.pkl"
        joblib.dump(pipe, model_path)
        st.success(f"✅ Modèle entraîné et sauvegardé : {model_path}")

        helpers.ensure_dir("outputs/data")
        X_train.assign(**{target: y_train}).to_csv(f"outputs/data/train_{target}.csv", index=False)
        X_test.assign(**{target: y_test}).to_csv(f"outputs/data/test_{target}.csv", index=False)

        # Stocker dans session_state pour reporting/evaluation
        st.session_state.update({
            "model": pipe,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "task": task,
            "y_pred": preds,
            "evaluation_metrics": pd.DataFrame([metrics_display])
        })

        # Feature importance si disponible (essayer d'extraire proprement)
        try:
            m = pipe.named_steps["model"]
            if hasattr(m, "feature_importances_") and len(num_cols) > 0:
                # get feature names from preprocessor
                try:
                    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
                except:
                    feature_names = num_cols + cat_cols
                fi = pd.Series(m.feature_importances_, index=feature_names).sort_values(ascending=False)
                st.session_state["feature_importance"] = fi
        except Exception:
            pass

        return {
            "pipeline": pipe,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "task": task
        }

    st.stop()