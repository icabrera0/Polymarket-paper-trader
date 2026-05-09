@echo off
title Polymarket Dev Launcher
cd /d "%~dp0"

echo  =====================================================
echo   POLYMARKET BOT - DEV MODE (hot reload enabled)
echo  =====================================================
echo.
echo  Save any .py file in src/ to auto-restart the bot.
echo  Dashboard auto-reloads when dashboard.py is saved.
echo.

start "Polymarket Bot [DEV]" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && title Polymarket Bot [DEV - Hot Reload] && color 0E && python dev_runner.py"
timeout /t 2 /nobreak >nul
start "Polymarket Dashboard [DEV]" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && title Polymarket Dashboard [DEV] && streamlit run dashboard.py --server.runOnSave true"

echo  Dev mode started.
echo.
echo  Hot reload: save any src/*.py file -- bot restarts automatically
echo  Dashboard:  save dashboard.py -- browser refreshes automatically
echo.
echo  Closing this window is safe.
timeout /t 5 /nobreak >nul
