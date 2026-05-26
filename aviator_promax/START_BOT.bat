@echo off
title Aviator ProMax Bot
color 0A
echo ================================================
echo      AVIATOR PRO MAX PREDICTION BOT
echo      Parimatch Live Tracker + AI Predictor
echo ================================================
echo.
echo [1/3] Installing dependencies...
pip install flask flask-socketio flask-cors python-dotenv eventlet playwright -q
echo.
echo [2/3] Installing Playwright browsers (first time only)...
playwright install chromium 2>nul
echo.
echo [3/3] Starting bot + dashboard...
echo Dashboard will open at: http://localhost:5000
echo.
python "%~dp0main.py"
pause
