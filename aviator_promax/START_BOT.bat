@echo off
title AVIATOR PRO MAX BOT v2.0
color 0A

echo ================================================
echo   AVIATOR PRO MAX PREDICTION BOT v2.0
echo ================================================
echo.

:: Move to the folder where this .bat file lives
cd /d "%~dp0"

:: Check Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: FIX: Install only missing deps (skip if already installed) for fast startup
echo [1/3] Checking dependencies...
python -c "import playwright, flask, flask_socketio, flask_cors, dotenv, eventlet" >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/3] Installing missing packages...
    pip install -q playwright flask flask-socketio flask-cors python-dotenv eventlet
    python -m playwright install chromium --with-deps
    echo [1/3] Dependencies installed!
) else (
    echo [1/3] All dependencies already installed. Skipping...
)

:: FIX: Warn user about .env credentials (never commit .env!)
if not exist ".env" (
    echo [WARNING] .env file not found. Copy .env.example and fill in your details.
    pause
    exit /b 1
)

echo.
echo [2/3] Starting bot...
echo [3/3] Dashboard will open automatically in your browser.
echo.
echo  INSTRUCTIONS:
echo  1. Run OPEN_CHROME.bat in another window first!
echo  2. Log in to your betting site and open Aviator.
echo  3. The bot auto-detects and starts tracking!
echo.
echo ================================================
echo  Dashboard: http://localhost:5000
echo  Press Ctrl+C to stop the bot.
echo ================================================
echo.

python main.py

pause
