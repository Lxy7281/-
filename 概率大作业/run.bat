@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python -X utf8 -m streamlit run app.py --server.headless true
pause
