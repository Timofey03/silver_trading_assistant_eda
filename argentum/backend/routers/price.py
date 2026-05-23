"""GET /api/price — текущая цена силера + sparkline за последние 5 дней."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf
from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SILVER_CACHE = REPO_ROOT / "data" / "multi_asset" / "metals" / "silver_daily.parquet"


class PricePoint(BaseModel):
    date: str
    close: float


class PriceResponse(BaseModel):
    current: float          # last close
    previous: float         # yesterday close
    change_pct: float       # (current - previous) / previous * 100
    currency: str = "USD"
    ticker: str = "SI=F"
    sparkline: List[PricePoint]  # последние 5 дней для микро-графика
    last_update: str


router = APIRouter()


def _load_silver_from_cache() -> pd.DataFrame:
    """Читаем silver_daily.parquet (обновляется через GitHub Actions)."""
    if SILVER_CACHE.exists():
        return pd.read_parquet(SILVER_CACHE)
    # Fallback: yfinance live
    df = yf.Ticker("SI=F").history(period="1mo")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.rename(columns={c: c.lower() for c in df.columns})


@router.get("/price", response_model=PriceResponse)
def get_price():
    """Текущая цена + sparkline 5 дней."""
    df = _load_silver_from_cache()
    if df.empty:
        return PriceResponse(
            current=0.0, previous=0.0, change_pct=0.0,
            sparkline=[], last_update="—",
        )

    last_5 = df.tail(5)
    closes = last_5["close"].tolist()
    current = float(closes[-1])
    previous = float(closes[-2]) if len(closes) > 1 else current
    change_pct = (current - previous) / previous * 100 if previous else 0.0

    sparkline = [
        PricePoint(date=str(d.date()), close=float(c))
        for d, c in zip(last_5.index, closes)
    ]

    return PriceResponse(
        current=current,
        previous=previous,
        change_pct=change_pct,
        sparkline=sparkline,
        last_update=str(df.index[-1].date()),
    )
