"""GET /api/metrics — Sharpe, Win Rate, Max DD, Profit Factor. С опциональным period filter."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
E3B_METRICS = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "metrics.json"
E3B_TRADES = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv"


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
    psr: float
    period_years: float
    best_trade: float
    worst_trade: float
    model_name: str = "E3b"
    model_features: int = 30
    period_label: str = "all"        # 1m/3m/6m/1y/3y/all
    period_start: str = "—"
    period_end: str = "—"


router = APIRouter()


def _period_cutoff(period: str) -> Optional[pd.Timestamp]:
    """Возвращает дату начала периода или None если 'all'."""
    if period == "all":
        return None
    days_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "3y": 1095}
    days = days_map.get(period)
    if days is None:
        return None
    return pd.Timestamp(datetime.now() - timedelta(days=days))


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    period: Literal["1m", "3m", "6m", "1y", "3y", "all"] = Query("all"),
):
    """
    Метрики финальной модели.

    Если period != 'all' — пересчитываем по trades.csv, фильтруя по entry_date.
    Если period == 'all' — возвращаем cached metrics.json.
    """
    # Default fallback
    fallback = MetricsResponse(
        sharpe=0, sortino=0, annual_return=0, total_return=0,
        max_drawdown=0, profit_factor=0, win_rate=0, n_trades=0,
        oos_accuracy=0, psr=0, period_years=0,
        best_trade=0, worst_trade=0,
        period_label=period,
    )

    if period == "all":
        if not E3B_METRICS.exists():
            return fallback
        m = json.loads(E3B_METRICS.read_text(encoding="utf-8"))
        # Period start/end из trades
        period_start, period_end = "—", "—"
        if E3B_TRADES.exists():
            try:
                df = pd.read_csv(E3B_TRADES)
                df = df[df["exit_reason"] != "OPEN"]
                df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
                df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
                df = df.dropna(subset=["entry_date", "exit_date"])
                if len(df):
                    period_start = str(df["entry_date"].min().date())
                    period_end = str(df["exit_date"].max().date())
            except Exception:
                pass
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
            period_label="all",
            period_start=period_start,
            period_end=period_end,
        )

    # === Period-filtered metrics ===
    if not E3B_TRADES.exists():
        return fallback

    df = pd.read_csv(E3B_TRADES)
    df = df[df["exit_reason"] != "OPEN"]
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df = df.dropna(subset=["entry_date", "exit_date"])

    cutoff = _period_cutoff(period)
    if cutoff is not None:
        df = df[df["entry_date"] >= cutoff]

    if df.empty:
        return MetricsResponse(
            sharpe=0, sortino=0, annual_return=0, total_return=0,
            max_drawdown=0, profit_factor=0, win_rate=0, n_trades=0,
            oos_accuracy=0, psr=0, period_years=0,
            best_trade=0, worst_trade=0,
            period_label=period,
            period_start="—", period_end="—",
        )

    # Compute metrics
    nr = df["net_return"].dropna().values
    first = df["entry_date"].min()
    last = df["exit_date"].max()
    period_years = max((last - first).days / 365.25, 0.01)

    # Compound total
    total = float(np.prod(1 + nr) - 1)
    annual = float((1 + total) ** (1 / period_years) - 1) if period_years > 0 else 0
    trades_per_year = len(nr) / period_years
    sr_per_trade = nr.mean() / nr.std() if nr.std() > 0 else 0
    sr_annual = sr_per_trade * np.sqrt(trades_per_year) if trades_per_year > 0 else 0
    neg = nr[nr < 0]
    sortino_per_trade = nr.mean() / neg.std() if len(neg) > 0 and neg.std() > 0 else 0
    sortino_annual = sortino_per_trade * np.sqrt(trades_per_year) if trades_per_year > 0 else 0

    # Equity drawdown
    eq = (1 + pd.Series(nr)).cumprod()
    dd = (eq / eq.cummax() - 1).min()

    wins = nr[nr > 0]
    losses = nr[nr < 0]
    win_rate = len(wins) / len(nr) if len(nr) else 0
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else 0

    return MetricsResponse(
        sharpe=round(sr_annual, 3),
        sortino=round(sortino_annual, 3),
        annual_return=round(annual, 4),
        total_return=round(total, 4),
        max_drawdown=round(float(dd) if pd.notna(dd) else 0, 4),
        profit_factor=round(pf, 3),
        win_rate=round(win_rate, 4),
        n_trades=int(len(nr)),
        oos_accuracy=0,
        psr=0,
        period_years=round(period_years, 2),
        best_trade=round(float(nr.max()), 4),
        worst_trade=round(float(nr.min()), 4),
        period_label=period,
        period_start=str(first.date()),
        period_end=str(last.date()),
    )
