"""Загрузка дневных OHLCV для 5 металлов через yfinance.

Особенности:
- Continuous futures (continuous front contracts)
- Кеширование в parquet (быстрая повторная загрузка)
- Обработка gaps до 3 торговых дней (forward-fill)
- Унифицированный формат: (date, open, high, low, close, volume)

Использование:
    from app.multi_asset import load_metals
    data = load_metals()           # вернёт dict[str, pd.DataFrame]
    data = load_metals(force_refresh=True)
    silver = load_single_metal("silver")
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from app.multi_asset.config import (
    METALS,
    METALS_DIR,
    START_DATE,
    END_DATE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _cache_path(metal: str) -> Path:
    return METALS_DIR / f"{metal}_daily.parquet"


def _is_cache_fresh(path: Path, max_age_days: int = 1) -> bool:
    """True если кеш существует и моложе max_age_days."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = (datetime.now() - mtime).days
    return age <= max_age_days


def _download_yf(ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    """Скачивание через yfinance с retry-логикой."""
    last_err = None
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, auto_adjust=False)
            if df.empty:
                raise ValueError(f"yfinance вернул пустой DataFrame для {ticker}")
            # Унификация: убираем timezone, оставляем только дату
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            # Канонические имена колонок
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            cols = ["open", "high", "low", "close", "volume"]
            df = df[cols].copy()
            df.index.name = "date"
            return df
        except Exception as e:
            last_err = e
            logger.warning(f"  Попытка {attempt + 1}/{max_retries} для {ticker} провалилась: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Не удалось скачать {ticker} после {max_retries} попыток: {last_err}")


def _validate_metal_data(df: pd.DataFrame, metal: str) -> dict:
    """Базовые проверки данных. Возвращает diagnostics dict."""
    diagnostics = {"metal": metal, "n_rows": len(df), "ok": True, "warnings": []}
    if df.empty:
        diagnostics["ok"] = False
        diagnostics["warnings"].append("EMPTY")
        return diagnostics

    # Все цены положительны
    for col in ["open", "high", "low", "close"]:
        bad = (df[col] <= 0).sum()
        if bad > 0:
            diagnostics["warnings"].append(f"non-positive {col}: {bad}")

    # High >= max(open, close), Low <= min(open, close)
    bad_hl = ((df["high"] < df[["open", "close"]].max(axis=1)) |
              (df["low"] > df[["open", "close"]].min(axis=1))).sum()
    if bad_hl > 0:
        diagnostics["warnings"].append(f"invalid OHLC ordering: {bad_hl} rows")

    # Volume не должен быть весь нулевой
    if df["volume"].sum() == 0:
        diagnostics["warnings"].append("zero volume entire series")

    # Gaps: дни между торгами
    diffs = df.index.to_series().diff().dt.days
    big_gaps = (diffs > 7).sum()  # игнорируем weekend = 3
    if big_gaps > 0:
        diagnostics["warnings"].append(f"{big_gaps} gaps >7 days")

    # Покрытие периода
    if not df.empty:
        diagnostics["first_date"] = df.index.min().date().isoformat()
        diagnostics["last_date"] = df.index.max().date().isoformat()
        diagnostics["coverage_years"] = round(
            (df.index.max() - df.index.min()).days / 365.25, 2
        )

    return diagnostics


def _enforce_validation(df: pd.DataFrame, metal: str, strict: bool = False) -> pd.DataFrame:
    """
    Авто-валидация на каждой загрузке.

    - Запускает _validate_metal_data + логирует warnings
    - Применяет фиксы (clamp OHLC, выкидывает non-positive prices)
    - strict=True → raise при критических проблемах (empty, all negative)
    """
    from app.multi_asset.data_quality import fix_ohlc_ordering, detect_outliers

    diag = _validate_metal_data(df, metal)
    if not diag["ok"]:
        if strict:
            raise RuntimeError(f"{metal}: validation failed — {diag['warnings']}")
        logger.error(f"  {metal}: CRITICAL — {diag['warnings']}")
        return df

    # Auto-fix OHLC
    df_fixed, n_ohlc = fix_ohlc_ordering(df)
    if n_ohlc > 0:
        logger.warning(f"  {metal}: auto-fixed {n_ohlc} invalid OHLC rows")

    # Outlier detection (mark only, don't drop)
    outliers = detect_outliers(df_fixed, threshold=0.20)
    if outliers.sum() > 0:
        outlier_dates = [str(d.date()) for d in df_fixed.index[outliers]][:3]
        logger.warning(f"  {metal}: {outliers.sum()} outliers (|ret|>20%): "
                       f"{', '.join(outlier_dates)}{'...' if outliers.sum() > 3 else ''}")
        df_fixed["is_outlier"] = outliers.values

    # Drop non-positive
    bad = (df_fixed["close"] <= 0)
    if bad.any():
        logger.warning(f"  {metal}: dropped {bad.sum()} non-positive close rows")
        df_fixed = df_fixed[~bad].copy()

    # Log other warnings
    if diag["warnings"]:
        # Some warnings already addressed; log others
        for w in diag["warnings"]:
            if "invalid OHLC" in w:
                continue  # уже исправили
            logger.warning(f"  {metal}: {w}")

    return df_fixed


def load_single_metal(metal: str, force_refresh: bool = False,
                       skip_validation: bool = False) -> pd.DataFrame:
    """Загрузить один металл (из кеша или yfinance) + auto-validate.

    Args:
        metal: ключ из METALS ('silver', 'gold', etc.)
        force_refresh: True → принудительно перезагрузить
        skip_validation: True → пропустить auto-валидацию (для legacy совместимости)

    Returns:
        DataFrame с колонками [open, high, low, close, volume], indexed by date.
        Может содержать колонку 'is_outlier' если найдены outliers.
    """
    if metal not in METALS:
        raise ValueError(f"Unknown metal: {metal}. Available: {list(METALS.keys())}")

    cache = _cache_path(metal)
    if not force_refresh and _is_cache_fresh(cache, max_age_days=1):
        df = pd.read_parquet(cache)
        logger.info(f"  {metal}: cached ({len(df)} rows)")
        # Auto-validate cached data too (быстрая операция)
        if not skip_validation:
            df = _enforce_validation(df, metal, strict=False)
        return df

    ticker = METALS[metal]["ticker"]
    logger.info(f"  {metal} ({ticker}): downloading...")
    df = _download_yf(ticker, START_DATE, END_DATE)

    df = df[df["close"] > 0].copy()
    df.index.name = "date"

    # Auto-validate ДО сохранения в кеш (чтобы кеш был чистый)
    if not skip_validation:
        df = _enforce_validation(df, metal, strict=False)

    df.to_parquet(cache, compression="snappy")
    logger.info(f"  {metal}: saved {len(df)} rows to {cache.name}")
    return df


def load_metals(
    force_refresh: bool = False,
    only: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Загрузить все 5 металлов.

    Args:
        force_refresh: пропустить кеш
        only: ограничить список (например, ['silver', 'gold'])

    Returns:
        dict {metal_name: DataFrame}
    """
    targets = only or list(METALS.keys())
    result = {}
    for metal in targets:
        try:
            result[metal] = load_single_metal(metal, force_refresh=force_refresh)
        except Exception as e:
            logger.error(f"  {metal}: FAILED — {e}")
            result[metal] = pd.DataFrame()
    return result


def refresh_metals_cache(verbose: bool = True) -> dict:
    """Принудительная перезагрузка + diagnostics для всех металлов."""
    data = load_metals(force_refresh=True)
    diagnostics = {}
    for metal, df in data.items():
        d = _validate_metal_data(df, metal)
        diagnostics[metal] = d
        if verbose:
            status = "✓" if not d["warnings"] else "⚠"
            warnings_text = "; ".join(d["warnings"]) if d["warnings"] else "clean"
            coverage = d.get("coverage_years", 0)
            logger.info(
                f"  {status} {metal:10s} {d['n_rows']:5d} rows · "
                f"{coverage} years · {warnings_text}"
            )
    return diagnostics


if __name__ == "__main__":
    logger.info("=== Metal loader test run ===")
    diagnostics = refresh_metals_cache(verbose=True)
    n_ok = sum(1 for d in diagnostics.values() if d["ok"])
    logger.info(f"\nИтог: {n_ok}/{len(diagnostics)} активов загружены успешно")
