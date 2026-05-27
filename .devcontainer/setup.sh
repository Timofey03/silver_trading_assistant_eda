#!/usr/bin/env bash
# .devcontainer/setup.sh — one-time install после создания Codespace
set -e

echo "=== Argentum Codespace Setup ==="

# === Python backend ===
echo ""
echo "[1/3] Installing Python dependencies..."
pip install --upgrade pip
pip install -r argentum/backend/requirements.txt

# Дополнительные deps which apply_optimal etc нуждаются
pip install scikit-learn lightgbm matplotlib seaborn requests pyarrow

# === Frontend ===
echo ""
echo "[2/3] Installing Node dependencies..."
cd argentum/frontend
npm install --silent
cd ../..

# === Data: лёгкая инициализация (если нет cached parquets) ===
echo ""
echo "[3/3] Initializing data cache (if needed)..."
if [ ! -f "data/multi_asset/metals/silver_daily.parquet" ]; then
  echo "  Downloading silver data..."
  python -c "import sys; sys.path.insert(0, '.'); from app.multi_asset.metal_loader import load_metals; load_metals(force_refresh=True)" || echo "  (will retry on first request)"
fi

if [ ! -f "baseline_outputs_multiasset/e3b_adaptive/trades.csv" ]; then
  echo "  No trades.csv — backend will return empty data."
  echo "  Run scripts/backfill_walkforward_ffill5.py to generate."
fi

# === Generate OOD detector if missing ===
if [ ! -f "baseline_outputs_multiasset/ood_detector.json" ]; then
  python -m app.multi_asset.ood_detector || echo "  (OOD detector skipped)"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Запуск:"
echo "  Backend:  uvicorn argentum.backend.main:app --host 0.0.0.0 --port 8000"
echo "  Frontend: cd argentum/frontend && npm run dev"
echo ""
echo "Или используйте автоматический запуск postStart."
