"""
Module de gestion d'erreurs pour l'application Streamlit
Fournit des décorateurs et fonctions pour capturer et gérer les erreurs de manière robuste
"""

import streamlit as st
import traceback
from functools import wraps
import logging
from datetime import datetime
import os

# Configuration du logger
log_dir = "outputs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, 'app_errors.log'),
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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
                logging.error(error_msg)
                logging.error(traceback.format_exc())

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
                    if st.button("Réinitialiser", key=f"reset_{section_name}_{datetime.now().timestamp()}",
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


def show_error_dashboard():
    """
    Affiche un tableau de bord des erreurs récentes (pour debug)
    Utile pour les développeurs ou en mode debug
    """
    if "error_log" in st.session_state and st.session_state["error_log"]:
        st.markdown("---")
        st.markdown("### Historique des erreurs (Debug)")
        st.markdown(f"**Total d'erreurs enregistrées :** {len(st.session_state['error_log'])}")
        st.markdown("**10 dernières erreurs :**")

        for i, err in enumerate(reversed(st.session_state["error_log"][-10:]), 1):
            st.text(f"{i}. [{err['timestamp'].strftime('%H:%M:%S')}] {err['section']}: {err['message'][:100]}")
        st.markdown("---")


def safe_file_operation(operation_name):
    """
    Décorateur spécifique pour les opérations sur fichiers
    Gère les erreurs courantes : permissions, fichier non trouvé, etc.

    Args:
        operation_name: Nom de l'opération (ex: "Chargement CSV")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                st.error(f"Fichier non trouvé : {str(e)}")
                st.info("Vérifiez que le fichier existe et que le chemin est correct")
                logging.error(f"FileNotFoundError in {operation_name}: {str(e)}")
                return None
            except PermissionError as e:
                st.error(f"Permission refusée : {str(e)}")
                st.info("Vérifiez que vous avez les droits d'accès au fichier")
                logging.error(f"PermissionError in {operation_name}: {str(e)}")
                return None
            except Exception as e:
                st.error(f"Erreur lors de {operation_name}: {str(e)}")
                logging.error(f"Error in {operation_name}: {str(e)}")
                logging.error(traceback.format_exc())
                return None
        return wrapper
    return decorator


def initialize_error_handling():
    """
    Initialise le système de gestion d'erreurs
    À appeler au début de l'application
    """
    if "error_log" not in st.session_state:
        st.session_state["error_log"] = []

    if "error_handling_initialized" not in st.session_state:
        st.session_state["error_handling_initialized"] = True
        logging.info("Error handling system initialized")

