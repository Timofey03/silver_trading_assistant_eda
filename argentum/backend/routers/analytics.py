"""GET /api/equity, /api/monthly — analytics endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRADES_CSV = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv"
SILVER_PARQUET = REPO_ROOT / "data" / "multi_asset" / "metals" / "silver_daily.parquet"


class EquityPoint(BaseModel):
    date: str
    model: float
    buy_hold: float


class EquityResponse(BaseModel):
    points: list[EquityPoint]
    model_final: float
    buy_hold_final: float
    outperformance_pp: float  # percentage points difference
    period_start: str
    period_end: str


class MonthlyCell(BaseModel):
    year: int
    month: int
    return_pct: float        # 0.05 = +5%
    n_trades: int


class MonthlyResponse(BaseModel):
    cells: list[MonthlyCell]
    years: list[int]
    best_month: float
    worst_month: float
    best_year: float
    worst_year: float
    avg_month: float


router = APIRouter()


@router.get("/equity", response_model=EquityResponse)
def get_equity(period: Literal["1m", "3m", "6m", "1y", "3y", "all"] = Query("all")):
    """Equity curve модели + buy-and-hold серебра."""
    if not TRADES_CSV.exists() or not SILVER_PARQUET.exists():
        return EquityResponse(
            points=[], model_final=1.0, buy_hold_final=1.0,
            outperformance_pp=0.0, period_start="—", period_end="—",
        )

    df = pd.read_csv(TRADES_CSV)
    df = df[df["exit_reason"] != "OPEN"]
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df = df.dropna(subset=["entry_date", "exit_date"]).sort_values("exit_date")

    if df.empty:
        return EquityResponse(
            points=[], model_final=1.0, buy_hold_final=1.0,
            outperformance_pp=0.0, period_start="—", period_end="—",
        )

    silver = pd.read_parquet(SILVER_PARQUET)
    if "close" not in silver.columns:
        return EquityResponse(points=[], model_final=1.0, buy_hold_final=1.0,
                              outperformance_pp=0.0, period_start="—", period_end="—")

    # Period filter
    if period != "all":
        from datetime import datetime, timedelta
        days_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "3y": 1095}
        days = days_map.get(period, 365)
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
        df = df[df["entry_date"] >= cutoff]
        silver = silver[silver.index >= cutoff]

    if df.empty:
        return EquityResponse(points=[], model_final=1.0, buy_hold_final=1.0,
                              outperformance_pp=0.0, period_start="—", period_end="—")

    # Model equity: compound по net_return на каждой exit date
    model_eq = (1.0 + df["net_return"]).cumprod()

    # Build combined timeseries aligned to silver dates
    start = df["entry_date"].min()
    end = max(df["exit_date"].max(), silver.index.max())

    silver_window = silver.loc[(silver.index >= start) & (silver.index <= end)]
    if silver_window.empty:
        return EquityResponse(points=[], model_final=1.0, buy_hold_final=1.0,
                              outperformance_pp=0.0, period_start="—", period_end="—")

    bh_start_price = float(silver_window["close"].iloc[0])
    bh_eq = silver_window["close"] / bh_start_price  # equity ratio

    # Model equity step function aligned to silver dates
    model_eq_step = pd.Series(1.0, index=silver_window.index)
    for d, eq_val in zip(df["exit_date"], model_eq):
        model_eq_step.loc[model_eq_step.index >= d] = float(eq_val)

    # Build points (downsample to ~200 points for chart performance)
    n = len(silver_window)
    step = max(1, n // 200)
    points = []
    for i in range(0, n, step):
        d = silver_window.index[i]
        points.append(EquityPoint(
            date=str(d.date()),
            model=round(float(model_eq_step.iloc[i]), 4),
            buy_hold=round(float(bh_eq.iloc[i]), 4),
        ))
    # Add last point
    if n > 0 and (n - 1) % step != 0:
        d = silver_window.index[-1]
        points.append(EquityPoint(
            date=str(d.date()),
            model=round(float(model_eq_step.iloc[-1]), 4),
            buy_hold=round(float(bh_eq.iloc[-1]), 4),
        ))

    model_final = float(model_eq_step.iloc[-1])
    bh_final = float(bh_eq.iloc[-1])
    return EquityResponse(
        points=points,
        model_final=round(model_final, 4),
        buy_hold_final=round(bh_final, 4),
        outperformance_pp=round((model_final - bh_final) * 100, 2),
        period_start=str(start.date()),
        period_end=str(end.date()),
    )


@router.get("/monthly", response_model=MonthlyResponse)
def get_monthly():
    """Helmap: net_return per year × month."""
    if not TRADES_CSV.exists():
        return MonthlyResponse(cells=[], years=[], best_month=0, worst_month=0,
                                best_year=0, worst_year=0, avg_month=0)

    df = pd.read_csv(TRADES_CSV)
    df = df[df["exit_reason"] != "OPEN"]
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df = df.dropna(subset=["exit_date"])
    if df.empty:
        return MonthlyResponse(cells=[], years=[], best_month=0, worst_month=0,
                                best_year=0, worst_year=0, avg_month=0)

    df["year"] = df["exit_date"].dt.year
    df["month"] = df["exit_date"].dt.month

    # Compound monthly returns (product of (1+r) per trade in month, minus 1)
    grouped = df.groupby(["year", "month"]).agg(
        compounded=("net_return", lambda x: float((1 + x).prod() - 1)),
        n_trades=("net_return", "count"),
    ).reset_index()

    cells = [
        MonthlyCell(
            year=int(r["year"]), month=int(r["month"]),
            return_pct=round(float(r["compounded"]), 4),
            n_trades=int(r["n_trades"]),
        )
        for _, r in grouped.iterrows()
    ]

    # Yearly aggregates
    yearly = df.groupby("year")["net_return"].apply(
        lambda x: float((1 + x).prod() - 1)
    )

    years = sorted(df["year"].unique().tolist())
    return MonthlyResponse(
        cells=cells,
        years=years,
        best_month=round(float(grouped["compounded"].max()), 4),
        worst_month=round(float(grouped["compounded"].min()), 4),
        best_year=round(float(yearly.max()) if len(yearly) else 0, 4),
        worst_year=round(float(yearly.min()) if len(yearly) else 0, 4),
        avg_month=round(float(grouped["compounded"].mean()), 4),
    )
