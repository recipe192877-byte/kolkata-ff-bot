@echo off
title Launch Chrome with Remote Debugging
color 0B
echo ====================================================
echo      LAUNCHING CHROME WITH REMOTE DEBUGGING
echo ====================================================
echo.
echo Launching Chrome on port 9222...
echo Profile path: C:\chrome_aviator_manual
echo.
echo Note: If Chrome is already open, please close all
echo Chrome windows first, otherwise remote debugging
echo will not work!
echo.
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome_aviator_manual"
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome_aviator_manual"
) else (
    echo [ERROR] Chrome not found in standard paths!
    echo Please launch Chrome manually with:
    echo chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\chrome_aviator_manual"
    pause
    exit /b 1
)
echo [OK] Chrome launched!
timeout /t 3 >nul
