"""Загрузка макроэкономических данных.

Источники:
  - FRED (через публичный CSV endpoint) — TIPS, DXY, breakeven, VIX, oil
  - yfinance — USDRUB
  - (опционально) CFTC COT report — отдельный модуль

Использование:
    from app.multi_asset.macro_loader import load_macro
    macro = load_macro()  # dict[str, pd.DataFrame]
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from app.multi_asset.config import MACRO, MACRO_DIR, START_DATE, END_DATE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


FRED_CSV_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    "id={series_id}&cosd={start}&coed={end}"
)


def _cache_path(name: str) -> Path:
    return MACRO_DIR / f"{name}.parquet"


def _is_cache_fresh(path: Path, max_age_days: int = 1) -> bool:
    if not path.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
    return age <= max_age_days


def _load_fred(series_id: str, max_retries: int = 3) -> pd.DataFrame:
    """Скачать FRED-серию через публичный CSV endpoint."""
    url = FRED_CSV_URL.format(series_id=series_id, start=START_DATE, end=END_DATE)
    last_err = None
    for attempt in range(max_retries):
        try:
            df = pd.read_csv(url)
            # FRED returns: observation_date, <SERIES_ID>
            df.columns = ["date", series_id]
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            # FRED использует "." для missing — преобразуем в NaN
            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            return df
        except Exception as e:
            last_err = e
            logger.warning(f"  FRED {series_id} попытка {attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"FRED {series_id}: {last_err}")


def _load_yf_macro(ticker: str, series_id: str) -> pd.DataFrame:
    """Скачать макропоказатель через yfinance (например, USDRUB)."""
    t = yf.Ticker(ticker)
    df = t.history(start=START_DATE, end=END_DATE, auto_adjust=False)
    if df.empty:
        raise ValueError(f"yfinance пусто для {ticker}")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    out = df[["Close"]].rename(columns={"Close": series_id})
    out.index.name = "date"
    return out


def load_single_macro(name: str, force_refresh: bool = False) -> pd.DataFrame:
    """Загрузить одну макросерию (из кеша или источника)."""
    if name not in MACRO:
        raise ValueError(f"Unknown macro: {name}. Available: {list(MACRO.keys())}")

    cache = _cache_path(name)
    if not force_refresh and _is_cache_fresh(cache, max_age_days=1):
        df = pd.read_parquet(cache)
        logger.info(f"  {name}: cached ({len(df)} rows)")
        return df

    cfg = MACRO[name]
    source = cfg["source"]
    logger.info(f"  {name} ({source}): downloading...")

    if source == "fred":
        df = _load_fred(name)
    elif source == "yf":
        df = _load_yf_macro(cfg["ticker"], name)
    else:
        raise ValueError(f"Unknown source {source}")

    df = df.dropna()
    df.index.name = "date"
    df.to_parquet(cache, compression="snappy")
    logger.info(f"  {name}: saved {len(df)} rows")
    return df


def load_macro(
    force_refresh: bool = False,
    only: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Загрузить все макроиндикаторы."""
    targets = only or list(MACRO.keys())
    result = {}
    for name in targets:
        try:
            result[name] = load_single_macro(name, force_refresh=force_refresh)
        except Exception as e:
            logger.error(f"  {name}: FAILED — {e}")
            result[name] = pd.DataFrame()
    return result


def assemble_macro_frame(
    macro: dict[str, pd.DataFrame] | None = None,
    target_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Объединить все макросерии в один DataFrame, выровненный по target_index.

    - Дневные серии — оставляются как есть
    - Месячные (INDPRO, CPI) — forward-fill до следующего обновления
    - Добавляются "days_since_update" фичи для каждой серии

    Args:
        macro: словарь из load_macro(). Если None — загружается заново.
        target_index: pd.DatetimeIndex (обычно из metal data). Если None —
            берётся объединение всех макрорядов.

    Returns:
        DataFrame с колонками <series_name> и <series_name>_age_days
    """
    if macro is None:
        macro = load_macro()

    # Целевой индекс — все business days в диапазоне макроданных
    if target_index is None:
        min_dates = [df.index.min() for df in macro.values() if not df.empty]
        max_dates = [df.index.max() for df in macro.values() if not df.empty]
        target_index = pd.date_range(min(min_dates), max(max_dates), freq="B")

    out = pd.DataFrame(index=target_index)
    out.index.name = "date"

    for name, df in macro.items():
        if df.empty:
            continue
        # Reindex на target_index, ffill
        series = df[name].reindex(out.index)
        out[name] = series.ffill()

        # Age в днях с последнего обновления (информация о свежести)
        last_update = series.copy()
        last_update[~series.isna()] = pd.Series(
            range(len(series)), index=series.index
        )[~series.isna()].values.astype(float)
        last_update = last_update.ffill()
        age = pd.Series(range(len(out)), index=out.index, dtype=float) - last_update
        out[f"{name}_age_days"] = age.fillna(0).astype(int)

    return out


if __name__ == "__main__":
    logger.info("=== Macro loader test run ===")
    for name in MACRO:
        try:
            df = load_single_macro(name, force_refresh=True)
            print(f"  {name:12s} {len(df):5d} rows  "
                  f"{df.index.min().date()} → {df.index.max().date()}")
        except Exception as e:
            print(f"  {name:12s} FAILED: {e}")
    print()
    frame = assemble_macro_frame()
    print(f"Combined macro frame: {len(frame)} rows × {len(frame.columns)} cols")
    print(frame.tail(3))
