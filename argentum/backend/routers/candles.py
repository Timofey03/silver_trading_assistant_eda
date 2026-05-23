"""GET /api/candles — OHLC данные для свечного графика + BUY/SELL маркеры."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SILVER_CACHE = REPO_ROOT / "data" / "multi_asset" / "metals" / "silver_daily.parquet"
E3B_TRADES = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv"


class Candle(BaseModel):
    time: str         # ISO date
    open: float
    high: float
    low: float
    close: float


class Marker(BaseModel):
    time: str
    price: float
    type: str         # "BUY" | "SELL"
    text: Optional[str] = None      # "BUY" / "+12.3%" / "−5.4%"
    return_pct: Optional[float] = None


class CandleResponse(BaseModel):
    candles: List[Candle]
    markers: List[Marker]
    range_start: str
    range_end: str


router = APIRouter()


@router.get("/candles", response_model=CandleResponse)
def get_candles(
    period: str = "all",       # "1m" | "3m" | "6m" | "1y" | "3y" | "all"
):
    """OHLC данные + маркеры сделок для свечного графика."""
    if not SILVER_CACHE.exists():
        return CandleResponse(candles=[], markers=[], range_start="—", range_end="—")

    df = pd.read_parquet(SILVER_CACHE)

    # Period filter
    if period != "all":
        from datetime import datetime, timedelta
        days_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "3y": 1095}
        if period in days_map:
            cutoff = pd.Timestamp(datetime.now() - timedelta(days=days_map[period]))
            df = df[df.index >= cutoff]

    candles = [
        Candle(
            time=d.strftime("%Y-%m-%d"),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
        )
        for d, row in df.iterrows()
    ]

    markers = []
    if E3B_TRADES.exists():
        trades = pd.read_csv(E3B_TRADES)
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        # Фильтр по периоду
        if period != "all" and len(df):
            trades = trades[
                (trades["exit_date"] >= df.index[0])
                & (trades["entry_date"] <= df.index[-1])
            ]
        for _, t in trades.iterrows():
            ret = float(t["net_return"])
            markers.append(Marker(
                time=t["entry_date"].strftime("%Y-%m-%d"),
                price=float(t["entry_price"]),
                type="BUY",
                text="BUY",
            ))
            markers.append(Marker(
                time=t["exit_date"].strftime("%Y-%m-%d"),
                price=float(t["exit_price"]),
                type="SELL",
                text=f"{ret*100:+.1f}%",
                return_pct=ret * 100,
            ))

    return CandleResponse(
        candles=candles,
        markers=markers,
        range_start=str(df.index[0].date()) if len(df) else "—",
        range_end=str(df.index[-1].date()) if len(df) else "—",
    )
