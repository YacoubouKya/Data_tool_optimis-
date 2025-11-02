@echo off
REM -------------------------------------------------
REM Lancer Streamlit avec venv et modules manquants seulement
REM -------------------------------------------------

REM Se placer dans le dossier du projet
cd /d "C:\Users\MON\Desktop\Projet personnel\1-streamlit_data_project"

REM Activer le venv
call mon_env\Scripts\activate


REM Lancer Streamlit
python -m streamlit run data_tool_app.py

REM Garder la fenêtre ouverte après fermeture
pause
