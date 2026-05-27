"""GET /api/fx — текущий USDRUB и оценка RUB silver price."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from fastapi import APIRouter
from pydantic import BaseModel

from cache import ttl_cache

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SILVER_PARQUET = REPO_ROOT / "data" / "multi_asset" / "metals" / "silver_daily.parquet"
USDRUB_PARQUET = REPO_ROOT / "data" / "multi_asset" / "macro" / "USDRUB_daily.parquet"


class FxRates(BaseModel):
    usd_silver:   float = 0.0       # USD per oz
    usdrub:       float = 0.0       # RUB per USD
    rub_silver:   float = 0.0       # RUB per oz (= usd * usdrub)
    usdrub_change_5d_pct: float = 0.0
    fx_volatility_flag: bool = False    # |Δ5d| > 2% → шумный FX режим
    source:       str = "yfinance"
    last_update:  str = ""


router = APIRouter()


@ttl_cache(ttl_seconds=300)
def _load_usdrub_series() -> Optional[pd.Series]:
    """USDRUB из кеша (FRED) или yfinance fallback."""
    if USDRUB_PARQUET.exists():
        try:
            df = pd.read_parquet(USDRUB_PARQUET)
            col = df.columns[0] if len(df.columns) else None
            if col:
                return df[col].dropna()
        except Exception:
            pass
    # yfinance fallback
    try:
        t = yf.Ticker("RUB=X")
        hist = t.history(period="30d")
        if not hist.empty:
            return hist["Close"]
    except Exception:
        pass
    return None


@ttl_cache(ttl_seconds=120)
def _cached_usdrub() -> Optional[float]:
    rub_series = _load_usdrub_series()
    if rub_series is None or len(rub_series) == 0:
        return None
    return float(rub_series.iloc[-1])


@router.get("/fx", response_model=FxRates)
def get_fx_rates():
    """Текущий USD silver, USDRUB курс, RUB silver, FX volatility flag."""
    # USD silver from cache
    usd_silver = 0.0
    last_update = "—"
    if SILVER_PARQUET.exists():
        try:
            df = pd.read_parquet(SILVER_PARQUET)
            if len(df):
                usd_silver = float(df["close"].iloc[-1])
                last_update = str(df.index[-1].date())
        except Exception:
            pass

    # USDRUB (с TTL cache 2 мин)
    usdrub_now_cached = _cached_usdrub()
    if usdrub_now_cached is None:
        return FxRates(
            usd_silver=usd_silver,
            usdrub=0.0,
            rub_silver=0.0,
            source="unavailable",
            last_update=last_update,
        )
    usdrub_now = usdrub_now_cached

    # Δ5d % — нужна история, отдельный (cached) запрос
    rub_series = _load_usdrub_series()
    if rub_series is not None and len(rub_series) >= 6:
        usdrub_5d_ago = float(rub_series.iloc[-6])
        delta_5d = (usdrub_now - usdrub_5d_ago) / usdrub_5d_ago * 100
    else:
        delta_5d = 0.0

    fx_volatile = abs(delta_5d) > 2.0   # |Δ5d| > 2% → FX-шумный период

    return FxRates(
        usd_silver=usd_silver,
        usdrub=usdrub_now,
        rub_silver=usd_silver * usdrub_now,
        usdrub_change_5d_pct=round(delta_5d, 2),
        fx_volatility_flag=fx_volatile,
        source="yfinance" if not USDRUB_PARQUET.exists() else "cache+yfinance",
        last_update=last_update,
    )
