"""
argentum/backend/main.py — FastAPI entry point.

Главные endpoints:
  GET  /api/health             — health check
  GET  /api/signal             — current BUY/HOLD/SELL signal
  GET  /api/price              — текущая цена силера + sparkline
  GET  /api/history            — equity curve бэктеста + сделки
  GET  /api/metrics            — Sharpe, Win Rate, Max DD, etc.
  GET  /api/candles            — OHLC для свечного графика
  GET  /api/tinkoff/balance    — live баланс Tinkoff
  GET  /api/explain            — feature importance "почему BUY?"
  WS   /ws/live                — real-time price updates

Запуск:
    cd argentum/backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Добавляем корень проекта чтобы импортировать app.multi_asset
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from routers import signal, price, history, metrics, candles, tinkoff, explain, position, fx, analytics, evolution, positions
from cache import cache_load_from_disk, cache_save_to_disk
import atexit

app = FastAPI(
    title="Argentum API",
    description="AI-помощник для рынка серебра — backend API",
    version="2.0.0",
)

# Restore cache from disk at startup, save at shutdown
_restored = cache_load_from_disk()
if _restored:
    print(f"[cache] restored {_restored} entries from disk")
atexit.register(cache_save_to_disk)

# CORS для Next.js dev server (port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.app.github.dev",   # GitHub Codespaces
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    """Health check для деплоймента."""
    return {"status": "ok", "service": "argentum-api", "version": "2.0.0"}


# Routers
app.include_router(signal.router,  prefix="/api", tags=["signal"])
app.include_router(price.router,   prefix="/api", tags=["price"])
app.include_router(history.router, prefix="/api", tags=["history"])
app.include_router(metrics.router, prefix="/api", tags=["metrics"])
app.include_router(candles.router, prefix="/api", tags=["candles"])
app.include_router(tinkoff.router, prefix="/api", tags=["tinkoff"])
app.include_router(explain.router, prefix="/api", tags=["explain"])
app.include_router(position.router, prefix="/api", tags=["position"])
app.include_router(fx.router, prefix="/api", tags=["fx"])
app.include_router(analytics.router, prefix="/api", tags=["analytics"])
app.include_router(evolution.router, prefix="/api", tags=["evolution"])
app.include_router(positions.router, prefix="/api", tags=["positions"])


@app.get("/")
def root():
    return {
        "service": "Argentum API",
        "docs": "/docs",
        "endpoints": [
            "/api/health",
            "/api/signal",
            "/api/price",
            "/api/history",
            "/api/metrics",
            "/api/candles",
            "/api/tinkoff/balance",
            "/api/tinkoff/order",
            "/api/explain",
            "/api/position",
        ],
    }
