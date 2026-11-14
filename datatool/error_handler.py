"""Gestion centralisée des erreurs des sections de l'application."""

import logging
import traceback
from datetime import datetime
from functools import wraps

import streamlit as st

# Les erreurs sont journalisees sur la sortie standard (visibles dans les logs Streamlit Cloud).
logger = logging.getLogger("datatool")


def safe_execute(section_name):
    """
    Décorateur pour exécuter une fonction de manière sécurisée
    Capture les erreurs et affiche un message convivial sans faire planter l'app

    Args:
        section_name: Nom de la section pour l'affichage des erreurs

    Usage:
        @safe_execute("Comparaison de Modèles")
        def run_comparison():
            # Code qui peut générer des erreurs
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Logger l'erreur avec tous les détails
                error_msg = f"Erreur dans {section_name}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())

                # Enregistrer dans la session pour monitoring
                log_error_to_session(section_name, str(e))

                # Afficher un message convivial à l'utilisateur
                st.error(f"**Une erreur est survenue dans la section : {section_name}**")

                # Afficher le message d'erreur de manière propre (seulement si pertinent)
                error_text = str(e)

                # Ne pas afficher les erreurs techniques Streamlit à l'utilisateur
                if "key=" not in error_text.lower() and "widget" not in error_text.lower():
                    st.markdown("---")
                    st.markdown("**Détails de l'erreur :**")
                    st.code(error_text, language="text")
                    st.markdown("---")

                # Suggestions de solutions
                st.warning("""
                 **Solutions possibles :**
                - Rechargez la page (appuyez sur F5)
                - Vérifiez que vos données sont correctement chargées
                - Vérifiez que toutes les étapes précédentes sont complètes
                - Utilisez le bouton de réinitialisation ci-dessous si le problème persiste
                """)

                # Bouton pour réinitialiser la session
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("Réinitialiser", key=f"reset_{section_name}",
                                help="Efface toutes les données en mémoire et redémarre l'application"):
                        reset_application()

                return None
        return wrapper
    return decorator


def log_error_to_session(section_name, error_msg):
    """
    Enregistre les erreurs dans la session Streamlit pour monitoring

    Args:
        section_name: Nom de la section où l'erreur s'est produite
        error_msg: Message d'erreur
    """
    if "error_log" not in st.session_state:
        st.session_state["error_log"] = []

    st.session_state["error_log"].append({
        "timestamp": datetime.now(),
        "section": section_name,
        "message": error_msg
    })

    # Limiter à 50 erreurs pour éviter la surcharge mémoire
    if len(st.session_state["error_log"]) > 50:
        st.session_state["error_log"] = st.session_state["error_log"][-50:]


def reset_application():
    """
    Réinitialise complètement l'application en effaçant toutes les données de session
    """
    # Sauvegarder le log d'erreurs avant de tout effacer (optionnel)
    error_log = st.session_state.get("error_log", [])

    # Effacer toutes les clés de session
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Restaurer le log d'erreurs (optionnel)
    if error_log:
        st.session_state["error_log"] = error_log

    st.success("Application réinitialisée avec succès")
    st.rerun()


def initialize_error_handling():
    """
    Initialise le système de gestion d'erreurs
    À appeler au début de l'application
    """
    if "error_log" not in st.session_state:
        st.session_state["error_log"] = []

    if "error_handling_initialized" not in st.session_state:
        st.session_state["error_handling_initialized"] = True
        logger.info("Error handling system initialized")
