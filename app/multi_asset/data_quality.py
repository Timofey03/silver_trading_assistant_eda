"""
app/multi_asset/data_quality.py — низкоуровневые data-quality helpers.

⚠ ВАЖНО: основная логика очистки теперь в metal_loader._enforce_validation
(применяется автоматически на каждой загрузке).

Этот модуль остаётся для:
1. fix_ohlc_ordering() — используется в _enforce_validation
2. detect_outliers() — используется в _enforce_validation
3. clean_metal() / clean_all_metals() — batch-операции для cleanup
   существующих parquet-файлов (legacy одноразовые миграции)

Эти функции импортируются из metal_loader, поэтому модуль не deprecated —
просто его роль сузилась.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.multi_asset.config import METALS_DIR

logger = logging.getLogger(__name__)


def fix_ohlc_ordering(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Чинит invalid OHLC: high = max(o,h,l,c), low = min(o,h,l,c).
    Возвращает (исправленный df, count_fixed).
    """
    df = df.copy()
    cols_4 = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in cols_4):
        return df, 0

    new_high = df[cols_4].max(axis=1)
    new_low = df[cols_4].min(axis=1)
    fixed_mask = (new_high != df["high"]) | (new_low != df["low"])
    n_fixed = int(fixed_mask.sum())
    df["high"] = new_high
    df["low"] = new_low
    return df, n_fixed


def detect_outliers(df: pd.DataFrame, threshold: float = 0.20) -> pd.Series:
    """
    Возвращает Series[bool] — True для дней с |daily return| > threshold.
    20% дефолт = очень аномальные дни (Silver Thursday 1980, COVID -10% это норма).
    """
    if "close" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    returns = df["close"].pct_change()
    return returns.abs() > threshold


def clean_metal(
    df: pd.DataFrame,
    metal: str,
    outlier_threshold: float = 0.20,
    drop_outliers: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Полный pipeline очистки одного актива.

    Args:
        df: raw данные из yfinance
        metal: имя для логов
        outlier_threshold: порог |daily return| для outlier (0.20 = 20%)
        drop_outliers: если True — выкидываем outliers; иначе только метим колонкой 'is_outlier'

    Returns:
        (cleaned df, report dict)
    """
    report = {"metal": metal, "n_input": len(df)}
    if df.empty:
        report["status"] = "empty"
        return df, report

    # 1. Fix OHLC ordering
    df, n_ohlc_fixed = fix_ohlc_ordering(df)
    report["ohlc_fixed"] = n_ohlc_fixed

    # 2. Detect outliers
    outliers = detect_outliers(df, threshold=outlier_threshold)
    report["outliers_detected"] = int(outliers.sum())
    report["outlier_dates"] = [str(d.date()) for d in df.index[outliers]]

    if drop_outliers:
        df = df[~outliers].copy()
        report["status"] = "cleaned_with_drops"
    else:
        df["is_outlier"] = outliers.values
        report["status"] = "cleaned_marked"

    # 3. Drop rows with non-positive prices
    if "close" in df.columns:
        bad_close = (df["close"] <= 0)
        if bad_close.any():
            report["non_positive_dropped"] = int(bad_close.sum())
            df = df[~bad_close].copy()

    # 4. Sort + dedupe
    df = df[~df.index.duplicated(keep="last")].sort_index()

    report["n_output"] = len(df)
    report["first_date"] = str(df.index.min().date()) if len(df) else "—"
    report["last_date"] = str(df.index.max().date()) if len(df) else "—"
    return df, report


def clean_all_metals(
    raw_cache_dir: Path = METALS_DIR,
    write_back: bool = True,
    outlier_threshold: float = 0.20,
) -> dict[str, dict]:
    """
    Применить очистку ко всем металлам в кеше. Перезаписывает parquet.
    """
    reports = {}
    for cache_file in sorted(raw_cache_dir.glob("*_daily.parquet")):
        metal = cache_file.stem.replace("_daily", "")
        try:
            df = pd.read_parquet(cache_file)
        except Exception as e:
            reports[metal] = {"status": "read_failed", "error": str(e)}
            continue

        cleaned, report = clean_metal(df, metal, outlier_threshold=outlier_threshold)

        if write_back:
            # Backup raw перед перезаписью
            backup = cache_file.parent / f"{metal}_daily_raw_backup.parquet"
            if not backup.exists():
                df.to_parquet(backup, compression="snappy")
            cleaned.to_parquet(cache_file, compression="snappy")
            report["written"] = str(cache_file.name)

        reports[metal] = report
    return reports


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=" * 70)
    print(" Cleaning all metal caches")
    print("=" * 70)
    reports = clean_all_metals(write_back=True, outlier_threshold=0.20)
    print(f"\n{'Metal':12s} {'In':>6s} {'OHLC fix':>10s} {'Outliers':>10s} {'Out':>6s} Outlier dates")
    print("-" * 100)
    for metal, r in reports.items():
        odates = ", ".join(r.get("outlier_dates", [])[:3])
        if len(r.get("outlier_dates", [])) > 3:
            odates += f" (+{len(r['outlier_dates']) - 3})"
        print(
            f"{metal:12s} "
            f"{r.get('n_input', 0):>6d} "
            f"{r.get('ohlc_fixed', 0):>10d} "
            f"{r.get('outliers_detected', 0):>10d} "
            f"{r.get('n_output', 0):>6d}  {odates}"
        )
