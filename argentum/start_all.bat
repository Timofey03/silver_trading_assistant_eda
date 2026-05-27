@echo off
chcp 65001 >nul 2>&1
title Argentum Launcher

echo.
echo   ============================================================
echo                    ARGENTUM
echo            AI silver trading assistant
echo   ============================================================
echo.
echo   Starting backend + frontend, please wait...
echo   (2 small windows will open - don't close them while using)
echo.

REM Backend in minimized window
start "Argentum Backend" /MIN cmd /k "%~dp0start_backend.bat"

REM Wait 4 sec
timeout /t 4 /nobreak >nul

REM Frontend in minimized window
start "Argentum Frontend" /MIN cmd /k "%~dp0start_frontend.bat"

REM Wait for frontend to be ready (8 sec)
echo   Waiting for site to be ready...
timeout /t 8 /nobreak >nul

REM Open browser
echo   Opening browser at http://localhost:3000
start http://localhost:3000

REM Close THIS launcher window
exit
