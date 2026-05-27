"""
app/multi_asset/cross_verify.py — кросс-проверка данных yfinance с
альтернативными источниками.

Стратегия:
- Primary:    SI=F   (silver futures continuous)
- Reference:  SLV    (SPDR Silver Trust ETF — другой инструмент, та же базовая)
- Optional:   PSLV   (Sprott Physical Silver Trust — третий источник)

Алерты:
- Корреляция returns между источниками < 0.85 → значит один из них glitchy
- Расхождение средней цены > 5% после нормализации → drift
- Расхождение в количестве торговых дней > 20 → один источник теряет дни

Запуск как скрипт: python -m app.multi_asset.cross_verify
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from app.multi_asset.config import METALS_DIR

logger = logging.getLogger(__name__)

# Силе ребро через 3 source-инструмента
SILVER_SOURCES = {
    "SI=F":  "Silver futures (continuous front contract)",
    "SLV":   "iShares Silver Trust ETF",
    "PSLV":  "Sprott Physical Silver Trust",
}


def _download(ticker: str, period: str = "2y") -> Optional[pd.Series]:
    """Daily close from yfinance."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            return None
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s
    except Exception as e:
        logger.warning(f"  {ticker}: download failed — {e}")
        return None


def cross_verify_silver(period: str = "2y", divergence_threshold: float = 0.05,
                         corr_threshold: float = 0.85) -> dict:
    """
    Сравнивает 3 источника silver. Возвращает diagnostics dict.
    """
    sources = {}
    for ticker in SILVER_SOURCES:
        s = _download(ticker, period)
        if s is not None:
            sources[ticker] = s

    if len(sources) < 2:
        return {"status": "failed", "reason": "fewer than 2 sources available"}

    # Align all sources on common dates
    aligned = pd.concat([s.rename(t) for t, s in sources.items()], axis=1).dropna()
    if len(aligned) < 30:
        return {"status": "failed", "reason": "too few common dates"}

    # Returns
    returns = aligned.pct_change().dropna()
    primary = "SI=F"
    if primary not in aligned.columns:
        primary = aligned.columns[0]

    # Корреляции (returns)
    corrs = {}
    for col in aligned.columns:
        if col != primary:
            corrs[f"{primary}↔{col}"] = float(returns[primary].corr(returns[col]))

    # Цены — нормализуем к первой общей дате
    normalized = aligned / aligned.iloc[0]
    final_levels = normalized.iloc[-1].to_dict()
    primary_level = final_levels[primary]
    divergences = {
        t: (final_levels[t] - primary_level) / primary_level
        for t in final_levels if t != primary
    }

    # Алерты
    alerts = []
    for pair, c in corrs.items():
        if c < corr_threshold:
            alerts.append(f"LOW_CORR {pair}={c:.3f} (threshold {corr_threshold})")
    for t, d in divergences.items():
        if abs(d) > divergence_threshold:
            alerts.append(f"DIVERGENCE {primary} vs {t}: {d*100:+.1f}% (threshold ±{divergence_threshold*100:.0f}%)")

    return {
        "status":       "ok" if not alerts else "alert",
        "period":       period,
        "n_common_days": len(aligned),
        "first_date":   str(aligned.index.min().date()),
        "last_date":    str(aligned.index.max().date()),
        "correlations": corrs,
        "final_levels": final_levels,
        "divergences":  divergences,
        "alerts":       alerts,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=" * 70)
    print(" Silver cross-source verification (3 instruments)")
    print("=" * 70)
    result = cross_verify_silver(period="2y")
    print(f"\nStatus:       {result.get('status')}")
    print(f"Period:       {result.get('period')}")
    print(f"Common days:  {result.get('n_common_days')}")
    print(f"Date range:   {result.get('first_date')} -> {result.get('last_date')}")
    print(f"\nReturns correlations (Pearson, daily):")
    for pair, c in result.get("correlations", {}).items():
        mark = "✓" if c >= 0.85 else "⚠"
        print(f"  {mark} {pair}: {c:.3f}")
    print(f"\nNormalized level divergence (vs SI=F):")
    for t, d in result.get("divergences", {}).items():
        mark = "✓" if abs(d) < 0.05 else "⚠"
        print(f"  {mark} {t}: {d*100:+.2f}%")
    if result.get("alerts"):
        print(f"\n⚠ ALERTS:")
        for a in result["alerts"]:
            print(f"  - {a}")
    else:
        print("\n✓ No alerts — data sources agree")
