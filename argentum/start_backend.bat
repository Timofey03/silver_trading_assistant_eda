@echo off
chcp 65001 >nul 2>&1
title argentum backend - FastAPI on :8000

echo ============================================================
echo   ARGENTUM BACKEND
echo   FastAPI: http://127.0.0.1:8000
echo   API docs: http://127.0.0.1:8000/docs
echo ============================================================
echo.

REM === Auto-pull свежих daily_reports от cron'а ===
REM Cron на GitHub кладёт новые signal.json каждый рабочий день. Без pull
REM локальный backend читает устаревшие отчёты — сайт показывает старую дату.
REM ff-only — никогда не перезаписывает локальные правки силой; если есть
REM расходящиеся правки, просто пропускает с предупреждением.
pushd "%~dp0.."
echo [auto-pull] origin/main...
git fetch --quiet origin main && git merge --ff-only origin/main 2>nul
if errorlevel 1 (
    echo [auto-pull] skipped — local diverged from origin, доделай git pull/merge вручную
) else (
    echo [auto-pull] OK
)
popd
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
