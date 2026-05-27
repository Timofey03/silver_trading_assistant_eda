"""GET /api/history — equity curve бэктеста + последние сделки."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
E3B_TRADES = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv"


class EquityPoint(BaseModel):
    date: str
    equity: float


class TradeItem(BaseModel):
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    net_return: float
    hold_days: int
    exit_reason: str
    pnl_label: str        # "✅ +12.3%" / "❌ -5.4%"


class HistoryResponse(BaseModel):
    equity_curve: List[EquityPoint]
    trades: List[TradeItem]
    n_trades: int
    total_return: float
    period_start: str
    period_end: str


router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
def get_history(limit: int = 10):
    """Equity curve + последние N сделок."""
    if not E3B_TRADES.exists():
        return HistoryResponse(
            equity_curve=[], trades=[], n_trades=0,
            total_return=0.0, period_start="—", period_end="—",
        )

    df = pd.read_csv(E3B_TRADES)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    # Исключаем OPEN сделки из истории (показываются в /api/position)
    df = df[df["exit_date"].notna()]
    df = df.sort_values("exit_date")

    # Equity curve через compound каждой сделки
    eq = (1.0 + df["net_return"].astype(float)).cumprod()
    equity = [
        EquityPoint(date=str(d.date()), equity=float(v))
        for d, v in zip(df["exit_date"], eq)
    ]

    # Последние N сделок
    last = df.nlargest(limit, "exit_date")
    trades = []
    for _, row in last.iterrows():
        ret = float(row["net_return"])
        emoji = "✅" if ret > 0 else "❌"
        trades.append(TradeItem(
            entry_date=str(row["entry_date"].date()),
            exit_date=str(row["exit_date"].date()),
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            net_return=ret,
            hold_days=int(row["hold_days"]),
            exit_reason=str(row.get("exit_reason", "—")),
            pnl_label=f"{emoji} {ret*100:+.2f}%",
        ))

    total = float((1 + df["net_return"]).prod() - 1)

    return HistoryResponse(
        equity_curve=equity,
        trades=trades,
        n_trades=len(df),
        total_return=total,
        period_start=str(df["entry_date"].min().date()),
        period_end=str(df["exit_date"].max().date()),
    )
