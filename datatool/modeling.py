# modules/modeling.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import joblib
from typing import Tuple, Any, Dict
from . import helpers
from . import metrics
from . import model_utils
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

def _validate_data_for_modeling(X: pd.DataFrame, y: pd.Series) -> bool:
    """
    Valide que les données sont prêtes pour la modélisation
    Retourne True si les données sont valides, False sinon
    """
    st.markdown("#### Validation pré-modélisation")

    validation_passed = True
    warnings = []
    errors = []

    # 1. Vérifier que X n'est pas vide
    if X.shape[0] == 0:
        errors.append("Le DataFrame X est vide (0 lignes)")
        validation_passed = False

    if X.shape[1] == 0:
        errors.append("Le DataFrame X n'a aucune colonne (features)")
        validation_passed = False

    # 2. Vérifier que y n'est pas vide
    if len(y) == 0:
        errors.append("La variable cible y est vide")
        validation_passed = False

    # 3. Vérifier que X et y ont la même longueur
    if len(X) != len(y):
        errors.append(f"Incompatibilité de taille : X a {len(X)} lignes mais y a {len(y)} valeurs")
        validation_passed = False

    # 4. Vérifier les NaN dans X
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        nan_count = len(nan_cols)
        if nan_count <= 5:
            warnings.append(f"{nan_count} colonne(s) avec valeurs manquantes : {', '.join(nan_cols)}")
        else:
            warnings.append(f"{nan_count} colonnes avec valeurs manquantes (dont {', '.join(nan_cols[:3])}...)")

    # 5. Vérifier les colonnes catégorielles avec trop de modalités
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    high_cardinality_cols = []
    for col in cat_cols:
        n_unique = X[col].nunique()
        if n_unique > 100:
            high_cardinality_cols.append(f"{col} ({n_unique} valeurs)")

    if high_cardinality_cols:
        if len(high_cardinality_cols) <= 3:
            warnings.append(f"Colonnes à haute cardinalité : {', '.join(high_cardinality_cols)}")
        else:
            warnings.append(f"{len(high_cardinality_cols)} colonnes à haute cardinalité (peut ralentir l'entraînement)")

    # 6. Vérifier les valeurs infinies dans X
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    inf_cols = []
    for col in num_cols:
        if ((X[col] == float('inf')) | (X[col] == float('-inf'))).any():
            inf_cols.append(col)

    if inf_cols:
        warnings.append(f"Colonnes avec valeurs infinies : {', '.join(inf_cols[:5])}")

    # 7. Vérifier la taille du dataset
    total_size_mb = (X.memory_usage(deep=True).sum() + y.memory_usage(deep=True)) / 1024 / 1024
    if total_size_mb > 500:
        warnings.append(f"Dataset volumineux ({total_size_mb:.1f} MB) - l'entraînement peut être lent")

    # Afficher les résultats
    if errors:
        for error in errors:
            st.error(error)

    if warnings:
        st.markdown("**Avertissements de validation**")
        for warning in warnings:
            st.warning(warning)
        st.info("Ces avertissements n'empêchent pas l'entraînement, mais peuvent affecter les performances")

    if validation_passed and not errors:
        st.success(f"Validation réussie : {X.shape[0]} lignes × {X.shape[1]} features")

    return validation_passed

def build_modeling_pipeline(model, X, do_scale=True, use_target_encoding=True):
    """
    Construit le pipeline de modélisation avec gestion des variables catégorielles

    Args:
        model: Modèle à utiliser
        X: Données d'entraînement
        do_scale: Si True, standardise les variables numériques
        use_target_encoding: Si True, utilise le Target Encoding pour les variables à haute cardinalité
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Convertir les catégorielles en string
    if cat_cols:
        X[cat_cols] = X[cat_cols].astype(str)

    # Pipeline pour les colonnes numériques
    num_steps = []
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if do_scale:
            num_steps.append(("scaler", StandardScaler()))

    # Gestion des variables catégorielles
    transformers = []

    # 1. Colonnes numériques
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))

    # 2. Colonnes catégorielles
    if cat_cols:
        # Séparation basse/élevée cardinalité
        low_card_cols = [col for col in cat_cols if X[col].nunique() <= 100]
        high_card_cols = [col for col in cat_cols if X[col].nunique() > 100]

        # Pipeline pour basse cardinalité (OneHot)
        if low_card_cols:
            cat_steps_low = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat_low", cat_steps_low, low_card_cols))

        # Pipeline pour haute cardinalité (Target Encoding)
        if high_card_cols and use_target_encoding:
            st.warning(f"Colonnes à haute cardinalité détectées : {', '.join(high_card_cols)}")
            st.info("Utilisation de Target Encoding pour ces variables")

            cat_steps_high = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("target_enc", TargetEncoder())
            ])
            transformers.append(("cat_high", cat_steps_high, high_card_cols))
        elif high_card_cols:
            st.warning(f"Colonnes à haute cardinalité ignorées : {', '.join(high_card_cols)}")

    # Création du ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    # Création du pipeline final
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline

def run_modeling(df: pd.DataFrame) -> dict:
    st.subheader("Modélisation interactive")

    # Détecter si on vient de la comparaison
    from_comparison = ("best_model_name" in st.session_state and
                      st.session_state['best_model_name'] is not None and
                      "comparison_results" in st.session_state and
                      st.session_state['comparison_results'] is not None)


    if from_comparison:
        best_model = st.session_state['best_model_name']
        comparison_results = st.session_state['comparison_results']

        # Vérifier que best_model n'est pas vide
        if not best_model:
            from_comparison = False

        # Récupérer le score du meilleur modèle
        best_row = comparison_results[comparison_results['Modèle'] == best_model]
        if not best_row.empty:
            # Trouver la colonne de score principal (vérifier différentes variantes)
            score_col = None
            score_value = None

            # Essayer différentes variantes de noms de colonnes
            possible_score_cols = ['ACCURACY', 'Accuracy', 'accuracy', 'R2', 'R²', 'r2']
            for col in possible_score_cols:
                if col in comparison_results.columns:
                    score_col = col
                    score_value = best_row[col].values[0]
                    break

            if score_col and score_value is not None:
                st.success(f"**Meilleur modèle de la comparaison** : {best_model} (Score: {score_value:.4f})")
                st.info("Les hyperparamètres du meilleur modèle sont pré-remplis. Vous pouvez les modifier pour optimiser davantage.")
            else:
                st.success(f"**Meilleur modèle de la comparaison** : {best_model}")
                st.info("Les hyperparamètres du meilleur modèle sont pré-remplis. Vous pouvez les modifier pour optimiser davantage.")
        else:
            st.success(f"**Meilleur modèle détecté** : {best_model}")
            st.info("Vous pouvez affiner ce modèle ou en choisir un autre")

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

    # Validation complète des données (silencieuse si pas d'erreurs)
    if not _validate_data_for_modeling(X, y):
        st.error("Les données ne sont pas valides pour la modélisation")
        st.info("Corrigez les erreurs ci-dessus avant de continuer")
        st.stop()

    # Vérifier les valeurs manquantes dans y
    y_missing = y.isna().sum()
    if y_missing > 0:
        st.warning(f"Variable cible contient {y_missing} valeurs manquantes ({y_missing/len(y)*100:.1f}%)")

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
            st.success(f"{y_missing} lignes supprimées. Nouvelles dimensions : {len(y)} lignes")
        else:  # Imputer
            if y.dtype in ['object', 'category']:
                mode_val = y.mode()[0] if not y.mode().empty else y.dropna().iloc[0]
                y = y.fillna(mode_val)
                st.success(f"Valeurs manquantes imputées avec le mode : {mode_val}")
            else:
                mean_val = y.mean()
                y = y.fillna(mean_val)
                st.success(f"Valeurs manquantes imputées avec la moyenne : {mean_val:.2f}")

    # Vérifier les valeurs infinies dans y (pour régression)
    if y.dtype in ['int64', 'float64']:
        y_inf = (~y.isna() & ((y == float('inf')) | (y == float('-inf')))).sum()
        if y_inf > 0:
            st.warning(f"Variable cible contient {y_inf} valeurs infinies")
            y = y.replace([float('inf'), float('-inf')], pd.NA)
            y = y.fillna(y.median())
            st.success(f"Valeurs infinies remplacées par la médiane")

    # Sélection du type de tâche avec UI (détection auto + choix utilisateur)
    task = model_utils.select_task_type_with_ui(y, key_suffix="modeling")

    # Afficher les infos de manière compacte
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes", f"{len(y):,}")
    with col2:
        st.metric("Uniques", y.nunique())
    with col3:
        if y.dtype in ['int64', 'float64']:
            st.metric("Moyenne", f"{y.mean():.2f}")
        else:
            st.metric("Mode", str(y.mode()[0])[:10] if not y.mode().empty else "N/A")

    st.markdown("---")

    # Option pour activer/désactiver le Target Encoding
    use_target_encoding = st.checkbox(
        "Utiliser Target Encoding pour les variables à haute cardinalité",
        value=True,
        help="Active le Target Encoding pour les variables catégorielles avec plus de 100 valeurs uniques"
    )

    # Configuration compacte
    st.markdown("### Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Taille test (%)", 5, 50, 20) / 100.0
    with col2:
        random_state = int(st.number_input("Seed", value=42))
    with col3:
        do_scale = st.checkbox("Standardiser", value=True)

    # Définir tous les modèles disponibles
    st.markdown("### Sélection du Modèle")

    # Mapper les noms de la comparaison vers les choix de modeling
    model_mapping = {
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "Logistic Regression": "logistic_regression",
        "Linear Regression": "linear_regression",
        "Ridge": "ridge",
        "Lasso": "lasso",
        "AdaBoost": "adaboost",
        "Extra Trees": "extra_trees",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors": "knn",
        "SVM": "svm",
        "SVR": "svr"
    }

    # Liste complète des modèles disponibles
    if task == "classification":
        available_models = [
            "Random Forest", "Gradient Boosting", "Logistic Regression",
            "AdaBoost", "Extra Trees", "Decision Tree", "K-Nearest Neighbors", "SVM"
        ]
    else:
        available_models = [
            "Random Forest", "Gradient Boosting", "Linear Regression",
            "Ridge", "Lasso", "AdaBoost", "Extra Trees", "Decision Tree",
            "K-Nearest Neighbors", "SVR"
        ]

    # Si on vient de la comparaison, proposer les modèles testés
    if from_comparison and "comparison_results" in st.session_state:
        comparison_models = st.session_state["comparison_results"]["Modèle"].tolist()
        best_model_name = st.session_state.get("best_model_name", comparison_models[0])

        # Filtrer les modèles disponibles pour ne garder que ceux de la comparaison
        models_to_show = [m for m in comparison_models if m in available_models]

        # Sélection avec le meilleur modèle par défaut
        model_display_choice = st.selectbox(
            "Choisir le modèle à affiner",
            models_to_show,
            index=models_to_show.index(best_model_name) if best_model_name in models_to_show else 0,
            help="Le meilleur modèle de la comparaison est sélectionné par défaut"
        )
    else:
        # Sélection parmi tous les modèles disponibles
        model_display_choice = st.selectbox(
            "Choisir un modèle",
            available_models,
            help="Sélectionnez le modèle à entraîner"
        )

    # Convertir vers le format interne
    model_choice = model_mapping.get(model_display_choice, "random_forest")

    st.info(f"Modèle sélectionné : **{model_display_choice}**")

    # Hyperparamètres par défaut
    default_params = {
        'rf_n_estimators': 100, 'rf_max_depth': 0,
        'gb_n_estimators': 100, 'gb_max_depth': 3, 'gb_lr': 0.1,
        'ab_n_estimators': 50, 'ab_lr': 1.0,
        'et_n_estimators': 100, 'et_max_depth': 0,
        'dt_max_depth': 0, 'dt_min_samples_split': 2,
        'knn_n_neighbors': 5,
        'svm_C': 1.0, 'svm_kernel': 'rbf',
        'ridge_alpha': 1.0,
        'lasso_alpha': 1.0
    }

    # Extraire les hyperparamètres du meilleur modèle de la comparaison si disponible
    if from_comparison and "best_model" in st.session_state and model_display_choice == st.session_state.get("best_model_name"):
        try:
            best_pipeline = st.session_state["best_model"]
            if best_pipeline and hasattr(best_pipeline, "named_steps"):
                best_model_obj = best_pipeline.named_steps.get("model")

                if best_model_obj:
                    # Extraire les paramètres selon le type de modèle
                    params = best_model_obj.get_params()

                    # Random Forest
                    if "RandomForest" in str(type(best_model_obj)):
                        default_params['rf_n_estimators'] = params.get('n_estimators', 100)
                        default_params['rf_max_depth'] = params.get('max_depth') or 0

                    # Gradient Boosting
                    elif "GradientBoosting" in str(type(best_model_obj)):
                        default_params['gb_n_estimators'] = params.get('n_estimators', 100)
                        default_params['gb_max_depth'] = params.get('max_depth', 3)
                        default_params['gb_lr'] = params.get('learning_rate', 0.1)

                    # AdaBoost
                    elif "AdaBoost" in str(type(best_model_obj)):
                        default_params['ab_n_estimators'] = params.get('n_estimators', 50)
                        default_params['ab_lr'] = params.get('learning_rate', 1.0)

                    # Extra Trees
                    elif "ExtraTrees" in str(type(best_model_obj)):
                        default_params['et_n_estimators'] = params.get('n_estimators', 100)
                        default_params['et_max_depth'] = params.get('max_depth') or 0

                    # Decision Tree
                    elif "DecisionTree" in str(type(best_model_obj)):
                        default_params['dt_max_depth'] = params.get('max_depth') or 0
                        default_params['dt_min_samples_split'] = params.get('min_samples_split', 2)

                    # KNN
                    elif "KNeighbors" in str(type(best_model_obj)):
                        default_params['knn_n_neighbors'] = params.get('n_neighbors', 5)

                    # SVM/SVR
                    elif "SVC" in str(type(best_model_obj)) or "SVR" in str(type(best_model_obj)):
                        default_params['svm_C'] = params.get('C', 1.0)
                        default_params['svm_kernel'] = params.get('kernel', 'rbf')

                    # Ridge
                    elif "Ridge" in str(type(best_model_obj)):
                        default_params['ridge_alpha'] = params.get('alpha', 1.0)

                    # Lasso
                    elif "Lasso" in str(type(best_model_obj)):
                        default_params['lasso_alpha'] = params.get('alpha', 1.0)

                    st.success("Hyperparamètres du meilleur modèle chargés automatiquement")
        except Exception as e:
            st.warning(f"Impossible d'extraire les hyperparamètres : {str(e)}")

    st.markdown("### Configuration des Hyperparamètres")

    # Afficher uniquement les hyperparamètres du modèle sélectionné
    if model_choice == "random_forest":
        st.markdown("**Random Forest**")
        rf_n_estimators = int(st.number_input("Nombre d'arbres (n_estimators)", 10, 1000, default_params['rf_n_estimators'], key="rf_n_est"))
        rf_max_depth = int(st.number_input("Profondeur max (0 = illimitée)", 0, 50, default_params['rf_max_depth'], key="rf_depth"))

    elif model_choice == "gradient_boosting":
        st.markdown("**Gradient Boosting**")
        gb_n_estimators = int(st.number_input("Nombre d'arbres (n_estimators)", 10, 1000, default_params['gb_n_estimators'], key="gb_n_est"))
        gb_max_depth = int(st.number_input("Profondeur max", 1, 20, default_params['gb_max_depth'], key="gb_depth"))
        gb_lr = float(st.number_input("Taux d'apprentissage (learning_rate)", 0.01, 1.0, default_params['gb_lr'], key="gb_lr"))

    elif model_choice == "adaboost":
        st.markdown("**AdaBoost**")
        ab_n_estimators = int(st.number_input("Nombre d'estimateurs", 10, 500, default_params['ab_n_estimators'], key="ab_n_est"))
        ab_lr = float(st.number_input("Taux d'apprentissage", 0.01, 2.0, default_params['ab_lr'], key="ab_lr"))

    elif model_choice == "extra_trees":
        st.markdown("**Extra Trees**")
        et_n_estimators = int(st.number_input("Nombre d'arbres", 10, 1000, default_params['et_n_estimators'], key="et_n_est"))
        et_max_depth = int(st.number_input("Profondeur max (0 = illimitée)", 0, 50, default_params['et_max_depth'], key="et_depth"))

    elif model_choice == "decision_tree":
        st.markdown("**Decision Tree**")
        dt_max_depth = int(st.number_input("Profondeur max (0 = illimitée)", 0, 50, default_params['dt_max_depth'], key="dt_depth"))
        dt_min_samples_split = int(st.number_input("Min samples split", 2, 20, default_params['dt_min_samples_split'], key="dt_split"))

    elif model_choice == "knn":
        st.markdown("**K-Nearest Neighbors**")
        knn_n_neighbors = int(st.number_input("Nombre de voisins (k)", 1, 50, default_params['knn_n_neighbors'], key="knn_k"))

    elif model_choice in ["svm", "svr"]:
        st.markdown("**Support Vector Machine**")
        svm_C = float(st.number_input("Paramètre C", 0.01, 100.0, default_params['svm_C'], key="svm_c"))
        svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0, key="svm_kernel")

    elif model_choice == "ridge":
        st.markdown("**Ridge Regression**")
        ridge_alpha = float(st.number_input("Alpha (régularisation)", 0.01, 100.0, default_params['ridge_alpha'], key="ridge_alpha"))

    elif model_choice == "lasso":
        st.markdown("**Lasso Regression**")
        lasso_alpha = float(st.number_input("Alpha (régularisation)", 0.01, 100.0, default_params['lasso_alpha'], key="lasso_alpha"))

    elif model_choice in ["logistic_regression", "linear_regression"]:
        st.markdown(f"**{model_display_choice}**")
        st.info("Ce modèle n'a pas d'hyperparamètres à configurer.")

    if st.button("Lancer l'entraînement"):
        with st.spinner("Préparation des données..."):
            # Préparation des données
            num_cols = X.select_dtypes(include="number").columns.tolist()
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

            if cat_cols:
                X[cat_cols] = X[cat_cols].astype(str)

            # Utiliser la fonction build_modeling_pipeline
            pipeline = build_modeling_pipeline(
                model=None,  # Le modèle sera ajouté plus tard
                X=X,
                do_scale=do_scale,
                use_target_encoding=use_target_encoding
            )


        # Choix du modèle avec tous les hyperparamètres
        if model_choice == "random_forest":
            if task == "classification":
                model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth==0 else rf_max_depth, random_state=random_state)
            else:
                model = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth==0 else rf_max_depth, random_state=random_state)

        elif model_choice == "gradient_boosting":
            if task == "classification":
                model = GradientBoostingClassifier(n_estimators=gb_n_estimators, max_depth=gb_max_depth, learning_rate=gb_lr, random_state=random_state)
            else:
                model = GradientBoostingRegressor(n_estimators=gb_n_estimators, max_depth=gb_max_depth, learning_rate=gb_lr, random_state=random_state)

        elif model_choice == "adaboost":
            if task == "classification":
                model = AdaBoostClassifier(n_estimators=ab_n_estimators, learning_rate=ab_lr, random_state=random_state)
            else:
                model = AdaBoostRegressor(n_estimators=ab_n_estimators, learning_rate=ab_lr, random_state=random_state)

        elif model_choice == "extra_trees":
            if task == "classification":
                model = ExtraTreesClassifier(n_estimators=et_n_estimators, max_depth=None if et_max_depth==0 else et_max_depth, random_state=random_state)
            else:
                model = ExtraTreesRegressor(n_estimators=et_n_estimators, max_depth=None if et_max_depth==0 else et_max_depth, random_state=random_state)

        elif model_choice == "decision_tree":
            if task == "classification":
                model = DecisionTreeClassifier(max_depth=None if dt_max_depth==0 else dt_max_depth, min_samples_split=dt_min_samples_split, random_state=random_state)
            else:
                model = DecisionTreeRegressor(max_depth=None if dt_max_depth==0 else dt_max_depth, min_samples_split=dt_min_samples_split, random_state=random_state)

        elif model_choice == "knn":
            if task == "classification":
                model = KNeighborsClassifier(n_neighbors=knn_n_neighbors)
            else:
                model = KNeighborsRegressor(n_neighbors=knn_n_neighbors)

        elif model_choice == "svm":
            model = SVC(C=svm_C, kernel=svm_kernel, random_state=random_state)

        elif model_choice == "svr":
            model = SVR(C=svm_C, kernel=svm_kernel)

        elif model_choice == "ridge":
            model = Ridge(alpha=ridge_alpha, random_state=random_state)

        elif model_choice == "lasso":
            model = Lasso(alpha=lasso_alpha, random_state=random_state)

        elif model_choice == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=random_state)

        elif model_choice == "linear_regression":
            model = LinearRegression()

        else:
            # Fallback
            if task == "classification":
                model = RandomForestClassifier(random_state=random_state)
            else:
                model = RandomForestRegressor(random_state=random_state)

        preprocessor = pipeline.named_steps['preprocessor']

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)  # Le modèle sélectionné
        ])

        # Split & train avec gestion d'erreurs robuste
        try:
            X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=random_state)

            with st.spinner("Entraînement du modèle en cours..."):
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                preds_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        except ValueError as e:
            st.error(f"**Erreur de données** : {str(e)}")
            st.info("**Suggestions** :")
            st.markdown("""
            - Vérifiez que vos données sont compatibles avec le modèle sélectionné
            - Assurez-vous qu'il n'y a pas de valeurs infinies ou NaN dans les features
            - Essayez de réduire le nombre de colonnes catégorielles avec trop de modalités
            """)
            st.stop()

        except MemoryError:
            st.error("**Mémoire insuffisante** pour entraîner ce modèle")
            st.info("**Suggestions** :")
            st.markdown("""
            - Réduisez la taille de votre dataset (échantillonnage)
            - Choisissez un modèle plus simple (ex: Logistic Regression au lieu de Random Forest)
            - Réduisez le nombre de features
            """) # Ajout de la parenthèse fermante ici
            st.stop()

        except Exception as e:
            st.error(f"**Erreur inattendue lors de l'entraînement** : {str(e)}")
            st.markdown("---")
            st.markdown("**Détails techniques :**")
            st.exception(e)
            st.markdown("---")
            st.info("Essayez de recharger vos données ou de choisir un autre modèle")
            st.stop()

        # Évaluation (metrics utilitaires)
        if task == "classification":
            metrics_result = metrics.classification_metrics(y_test, preds)
        else:
            metrics_result = metrics.regression_metrics(y_test, preds)

        metrics_display = _format_metrics(metrics_result, decimals=4)
        st.write("**Metrics (test)** :")
        st.json(metrics_display)

        # Sauvegarde modèle et datasets avec gestion d'erreurs
        # Nettoyer le nom de la cible pour éviter les caractères spéciaux
        safe_target = target.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")

        # Sauvegarde du modèle
        try:
            helpers.ensure_dir("outputs/models")
            model_path = f"outputs/models/model_{safe_target}.pkl"
            joblib.dump(pipe, model_path)
            st.success(f"Modèle sauvegardé : {model_path}")
        except PermissionError:
            st.warning("Impossible de sauvegarder le modèle : permissions insuffisantes")
            st.info("Le modèle reste disponible dans la session en cours")
        except Exception as e:
            st.warning(f"Impossible de sauvegarder le modèle : {str(e)}")
            st.info("Le modèle reste disponible dans la session en cours")

        # Sauvegarde des datasets
        try:
            helpers.ensure_dir("outputs/data")
            X_train.assign(**{target: y_train}).to_csv(f"outputs/data/train_{safe_target}.csv", index=False)
            X_test.assign(**{target: y_test}).to_csv(f"outputs/data/test_{safe_target}.csv", index=False)
            st.success(f"Datasets sauvegardés dans outputs/data/")
        except PermissionError:
            st.warning("Impossible de sauvegarder les datasets : permissions insuffisantes")
        except Exception as e:
            st.warning(f"Impossible de sauvegarder les datasets : {str(e)}")

        st.success("Modèle entraîné avec succès !")

        # Stocker dans session_state pour reporting/evaluation
        st.session_state.update({
            "model": pipe,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "task": task,
            "task_type": task,
            "y_pred": preds,
            "y_pred_proba": preds_proba,
            "evaluation_metrics": pd.DataFrame([metrics_display]),
            "current_model_name": model_display_choice,  # Stocker le nom du modèle pour l'évaluation
            "use_target_encoding": use_target_encoding
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
            "task": task,
            "y_pred_proba": preds_proba
        }


    st.stop()

