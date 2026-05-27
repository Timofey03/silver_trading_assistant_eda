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
    type: str         # "BUY" | "SELL" | "OPEN" (наша активная позиция)
    text: Optional[str] = None      # "BUY" / "+12.3%" / "−5.4%" / "OPEN +P&L"
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
        trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
        # exit_date может быть "_OPEN" — coerce → NaT, отфильтруем потом
        trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")

        # Period filter
        if period != "all" and len(df):
            mask = (trades["entry_date"] <= df.index[-1]) & (
                (trades["exit_date"] >= df.index[0])
                | trades["exit_date"].isna()  # OPEN — всегда показываем
            )
            trades = trades[mask]

        for _, t in trades.iterrows():
            ret = float(t["net_return"])
            is_open = t.get("exit_reason") == "OPEN" or pd.isna(t["exit_date"])

            # Всегда BUY маркер на входе
            markers.append(Marker(
                time=t["entry_date"].strftime("%Y-%m-%d"),
                price=float(t["entry_price"]),
                type="BUY",
                text="OPEN" if is_open else "BUY",
            ))

            # SELL маркер только для закрытых сделок
            if not is_open:
                markers.append(Marker(
                    time=t["exit_date"].strftime("%Y-%m-%d"),
                    price=float(t["exit_price"]),
                    type="SELL",
                    text=f"{ret*100:+.1f}%",
                    return_pct=ret * 100,
                ))

    # === Our live OPEN positions from SQLite tracker ===
    live_positions = []
    current_market_rub: dict[str, float] = {}
    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT / "argentum" / "backend"))
        import db as positions_db
        from routers.positions import _current_silver_price_rub
        live_positions = positions_db.list_positions()
        for pos in live_positions:
            current_market_rub[pos["figi"]] = _current_silver_price_rub(pos["figi"])
    except Exception:
        live_positions = []

    if live_positions and len(df):
        silver_df = pd.read_parquet(SILVER_CACHE) if SILVER_CACHE.exists() else None
        current_silver_usd = float(silver_df["close"].iloc[-1]) if silver_df is not None and len(silver_df) else 0
        for pos in live_positions:
            try:
                entry_d = pd.to_datetime(str(pos["opened_at"]).replace("Z", "").split("_")[0])
            except Exception:
                continue
            # Period filter
            if period != "all" and entry_d < df.index[0]:
                continue
            # Find USD silver close на entry date (для правильного позиционирования на USD-графике)
            usd_at_entry = None
            if silver_df is not None:
                try:
                    near = silver_df.index.asof(entry_d)
                    if pd.notna(near):
                        usd_at_entry = float(silver_df.loc[near, "close"])
                except Exception:
                    pass
            usd_at_entry = usd_at_entry or current_silver_usd
            # Compute live P&L (в RUB через Tinkoff GetLastPrices)
            entry_rub = float(pos.get("entry_price", 0))
            current_rub = current_market_rub.get(pos["figi"], 0) or float(pos.get("peak_price", entry_rub))
            pnl_pct = ((current_rub - entry_rub) / entry_rub * 100) if entry_rub else 0
            markers.append(Marker(
                time=entry_d.strftime("%Y-%m-%d"),
                price=usd_at_entry,
                type="OPEN",  # наш отдельный тип
                text=f"АКТИВНА {pnl_pct:+.1f}%",
                return_pct=pnl_pct,
            ))

    return CandleResponse(
        candles=candles,
        markers=markers,
        range_start=str(df.index[0].date()) if len(df) else "—",
        range_end=str(df.index[-1].date()) if len(df) else "—",
    )
