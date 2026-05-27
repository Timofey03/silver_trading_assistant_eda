@echo off
chcp 65001 >nul 2>&1
title argentum backend - FastAPI on :8000

echo ============================================================
echo   ARGENTUM BACKEND
echo   FastAPI: http://127.0.0.1:8000
echo   API docs: http://127.0.0.1:8000/docs
echo ============================================================
echo.

cd /d "%~dp0backend"

REM Activate venv (prefer .venv, fallback to venv)
if exist "..\..\.venv\Scripts\activate.bat" (
    call "..\..\.venv\Scripts\activate.bat"
) else if exist "..\..\venv\Scripts\activate.bat" (
    call "..\..\venv\Scripts\activate.bat"
) else (
    echo ERROR: venv not found at ..\..\.venv or ..\..\venv
    pause
    exit /b 1
)

REM Launch uvicorn
python -m uvicorn main:app --port 8000 --host 127.0.0.1

REM Don't close window if crashed
echo.
echo Service stopped. Press any key to close...
pause >nul
