"""GET /api/metrics — Sharpe, Win Rate, Max DD, Profit Factor."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
E3B_METRICS = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "metrics.json"


class MetricsResponse(BaseModel):
    sharpe: float
    sortino: float
    annual_return: float
    total_return: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    n_trades: int
    oos_accuracy: float
    psr: float            # Probabilistic Sharpe Ratio
    period_years: float
    best_trade: float
    worst_trade: float
    model_name: str = "E3b"
    model_features: int = 30


router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Главные метрики финальной модели."""
    if not E3B_METRICS.exists():
        return MetricsResponse(
            sharpe=0, sortino=0, annual_return=0, total_return=0,
            max_drawdown=0, profit_factor=0, win_rate=0, n_trades=0,
            oos_accuracy=0, psr=0, period_years=0,
            best_trade=0, worst_trade=0,
        )
    m = json.loads(E3B_METRICS.read_text(encoding="utf-8"))
    return MetricsResponse(
        sharpe=float(m.get("sharpe", 0)),
        sortino=float(m.get("sortino", 0)),
        annual_return=float(m.get("annual_return", 0)),
        total_return=float(m.get("total_return", 0)),
        max_drawdown=float(m.get("max_dd", 0)),
        profit_factor=float(m.get("profit_factor", 0)),
        win_rate=float(m.get("win_rate", 0)),
        n_trades=int(m.get("n_trades", 0)),
        oos_accuracy=float(m.get("oos_accuracy", 0)),
        psr=float(m.get("psr", 0)),
        period_years=float(m.get("period_years", 0)),
        best_trade=float(m.get("best_trade", 0)),
        worst_trade=float(m.get("worst_trade", 0)),
    )
