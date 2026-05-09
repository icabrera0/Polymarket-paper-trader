@echo off
title Polymarket Bot Launcher
cd /d "%~dp0"

echo  =====================================================
echo   POLYMARKET PAPER TRADING BOT
echo  =====================================================
echo.
echo  Starting Bot and Dashboard in separate windows...
echo.

start "Polymarket Bot" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && title Polymarket Bot && color 0A && python main.py"
timeout /t 2 /nobreak >nul
start "Polymarket Dashboard" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && title Polymarket Dashboard && streamlit run dashboard.py"

echo  Both processes started.
echo.
echo  - Bot window:       Ctrl+C to stop cleanly (sends Discord notification)
echo  - Dashboard window: Ctrl+C to stop (safe to close anytime)
echo.
echo  Closing this window is safe.
timeout /t 5 /nobreak >nul
