@echo off
chcp 65001 >nul 2>&1
title argentum frontend - Next.js on :3000

echo ============================================================
echo   ARGENTUM FRONTEND
echo   Next.js: http://localhost:3000
echo ============================================================
echo.

REM Add Node to PATH if not there
set "PATH=C:\Program Files\nodejs;%PATH%"

cd /d "%~dp0frontend"

REM Install deps if missing
if not exist "node_modules" (
    echo First run - installing packages...
    call npm install
    echo.
)

REM Point to backend
set "NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000"
npm run dev

REM Don't close window if crashed
echo.
echo Service stopped. Press any key to close...
pause >nul
