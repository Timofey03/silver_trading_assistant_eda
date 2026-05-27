"""Построение признаков для multi-asset ML модели.

Три уровня:
1. **Per-asset технические**: RSI, ADX, ATR, momentum, BB-width — на каждом из 5 металлов
2. **Cross-asset ratios**: Gold/Silver, Silver/Copper и их z-scores
3. **Macro context**: интеграция макрофичей с age-фичами

Итоговый DataFrame: ~80 признаков на каждый день silver (target).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.multi_asset.config import METALS
from app.multi_asset.metal_loader import load_metals
from app.multi_asset.macro_loader import assemble_macro_frame, load_macro

logger = logging.getLogger(__name__)


# =============================================================================
# Технические индикаторы (минимальный self-contained набор)
# =============================================================================

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)
    tr = _atr(high, low, close, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    logret = np.log(close / close.shift(1))
    return logret.rolling(window).std() * np.sqrt(252)  # annualized


# =============================================================================
# Уровень 1: per-asset features
# =============================================================================

def per_asset_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Технические признаки одного актива.

    Args:
        df: DataFrame с колонками [open, high, low, close, volume]
        prefix: префикс для имён колонок (например, 'silver_')

    Returns:
        DataFrame с признаками, без NaN строк.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Returns
    out[f"{prefix}_ret_1"] = close.pct_change(1)
    out[f"{prefix}_ret_5"] = close.pct_change(5)
    out[f"{prefix}_ret_20"] = close.pct_change(20)
    out[f"{prefix}_logret"] = np.log(close / close.shift(1))

    # Momentum / position
    out[f"{prefix}_ma_50"] = close.rolling(50).mean()
    out[f"{prefix}_ma_200"] = close.rolling(200).mean()
    out[f"{prefix}_dist_ma50"] = (close / out[f"{prefix}_ma_50"] - 1)
    out[f"{prefix}_dist_ma200"] = (close / out[f"{prefix}_ma_200"] - 1)

    # Volatility
    out[f"{prefix}_atr_14"] = _atr(high, low, close, 14)
    out[f"{prefix}_atr_pct"] = out[f"{prefix}_atr_14"] / close
    out[f"{prefix}_rvol_20"] = _realized_vol(close, 20)
    out[f"{prefix}_rvol_60"] = _realized_vol(close, 60)

    # Oscillators
    out[f"{prefix}_rsi_14"] = _rsi(close, 14)
    out[f"{prefix}_adx_14"] = _adx(high, low, close, 14)

    # Position in recent range
    hh_252 = high.rolling(252).max()
    ll_252 = low.rolling(252).min()
    out[f"{prefix}_pos_252"] = (close - ll_252) / (hh_252 - ll_252).replace(0, np.nan)

    # Volume features
    out[f"{prefix}_vol_z"] = (df["volume"] - df["volume"].rolling(50).mean()) / \
                            df["volume"].rolling(50).std().replace(0, np.nan)

    # Drop intermediate MA columns
    out = out.drop(columns=[f"{prefix}_ma_50", f"{prefix}_ma_200"])

    return out


# =============================================================================
# Уровень 2: cross-asset ratios
# =============================================================================

def cross_asset_features(
    metals: dict[str, pd.DataFrame],
    ffill_limit: int = 0,
) -> pd.DataFrame:
    """Cross-asset ratios и их z-scores.

    Главные ratios:
    - gold / silver        — классический леviс индикатор
    - silver / copper      — индикатор precious vs industrial demand
    - gold / platinum      — внутри precious metals group
    - silver / palladium   — индикатор autocatalyst demand

    Args:
        metals: dict из load_metals()
        ffill_limit: максимум дней для forward-fill (palladium имеет gaps).
                     0 = без ffill (academic walk-forward).
                     5 = умеренный ffill (production daily inference).

    Returns:
        DataFrame с cross-asset фичами.
    """
    close = pd.DataFrame({m: df["close"] for m, df in metals.items() if not df.empty})
    if ffill_limit > 0:
        close = close.ffill(limit=ffill_limit)
    close = close.dropna(how="all")

    out = pd.DataFrame(index=close.index)

    ratio_pairs = [
        ("gold", "silver"),
        ("silver", "copper"),
        ("gold", "platinum"),
        ("silver", "palladium"),
        ("platinum", "palladium"),
    ]
    for a, b in ratio_pairs:
        if a in close.columns and b in close.columns:
            r = close[a] / close[b]
            out[f"ratio_{a}_{b}"] = r
            # z-score относительно rolling 252 дней
            mean = r.rolling(252).mean()
            std = r.rolling(252).std().replace(0, np.nan)
            out[f"ratio_{a}_{b}_z"] = (r - mean) / std

    # Correlation features (90-day rolling)
    if "silver" in close.columns and "gold" in close.columns:
        silver_ret = np.log(close["silver"] / close["silver"].shift(1))
        gold_ret = np.log(close["gold"] / close["gold"].shift(1))
        out["corr_silver_gold_90"] = silver_ret.rolling(90).corr(gold_ret)
    if "silver" in close.columns and "copper" in close.columns:
        silver_ret = np.log(close["silver"] / close["silver"].shift(1))
        copper_ret = np.log(close["copper"] / close["copper"].shift(1))
        out["corr_silver_copper_90"] = silver_ret.rolling(90).corr(copper_ret)

    # Composite "metals index" (mean of normalized prices)
    normed = close.div(close.iloc[200] if len(close) > 200 else close.iloc[0])
    out["metals_composite"] = normed.mean(axis=1)
    out["metals_composite_ret_20"] = out["metals_composite"].pct_change(20)

    return out


# =============================================================================
# Финальная сборка
# =============================================================================

def _ffill_audit(df_before: pd.DataFrame, df_after: pd.DataFrame, label: str) -> dict:
    """Логирует сколько cells было заполнено ffill для каждой колонки."""
    import logging
    _log = logging.getLogger(__name__)
    report = {}
    total_cells_before = df_before.notna().sum().sum()
    total_cells_after = df_after.notna().sum().sum()
    cells_filled = int(total_cells_after - total_cells_before)
    n_rows = len(df_before)
    if cells_filled > 0:
        per_col = (df_after.notna().sum() - df_before.notna().sum()).sort_values(ascending=False)
        top3 = per_col[per_col > 0].head(3)
        report = {
            "label":              label,
            "cells_filled":       cells_filled,
            "rows":               n_rows,
            "fill_density":       round(cells_filled / max(1, n_rows * len(df_before.columns)) * 100, 2),
            "top_filled_cols":    {c: int(n) for c, n in top3.items()},
        }
        _log.info(
            f"  [ffill audit] {label}: filled {cells_filled} cells "
            f"({report['fill_density']}% of frame). Top: " +
            ", ".join(f"{c}={n}" for c, n in top3.items())
        )
    return report


def build_feature_frame(
    target: str = "silver",
    metals: Optional[dict[str, pd.DataFrame]] = None,
    macro_frame: Optional[pd.DataFrame] = None,
    force_refresh: bool = False,
    ffill_limit: int = 0,
    audit_ffill: bool = True,
) -> pd.DataFrame:
    """Собрать финальный feature DataFrame для целевого актива.

    Структура: индексирован по торговым дням target актива.
    Колонки: per-asset фичи (5 активов) + cross-asset + macro + age_days.

    Args:
        target: целевой актив (обычно 'silver')
        metals: dict из load_metals(). Если None — загружается заново.
        macro_frame: assembled macro frame. Если None — собирается заново.
        force_refresh: пересохранить кеш данных.
        ffill_limit: максимум дней для ffill метал-фичей (palladium gaps).
        audit_ffill: логировать сколько cells заполняется через ffill.

    Returns:
        DataFrame готовый к ML pipeline. Атрибут .attrs['ffill_audit']
        содержит отчёт о заполнениях.
    """
    import logging
    _log = logging.getLogger(__name__)

    if metals is None:
        metals = load_metals(force_refresh=force_refresh)
    if target not in metals or metals[target].empty:
        raise ValueError(f"Target {target} not loaded")

    target_index = metals[target].index
    audit_reports = {}

    # 1. Per-asset features для всех 5 металлов
    per_asset = []
    for metal, df in metals.items():
        if df.empty:
            continue
        feats = per_asset_features(df, prefix=metal)
        per_asset.append(feats)

    all_per_asset_raw = pd.concat(per_asset, axis=1, sort=False).reindex(target_index)
    # reindex без ffill — NaN там где per-asset не торгуется
    all_per_asset = all_per_asset_raw  # no ffill on per-asset (each metal has own bday calendar)

    # 2. Cross-asset features
    cross_raw = cross_asset_features(metals, ffill_limit=0)
    cross = cross_asset_features(metals, ffill_limit=ffill_limit).reindex(target_index)
    if audit_ffill and ffill_limit > 0:
        report = _ffill_audit(cross_raw.reindex(target_index), cross, "cross_asset")
        if report:
            audit_reports["cross_asset"] = report

    # 3. Macro features
    if macro_frame is None:
        macro = load_macro(force_refresh=force_refresh)
        macro_frame_raw = assemble_macro_frame(macro, target_index=target_index)
    else:
        macro_frame_raw = macro_frame.reindex(target_index)

    macro_frame_filled = macro_frame_raw.ffill()
    if audit_ffill:
        report = _ffill_audit(macro_frame_raw, macro_frame_filled, "macro")
        if report:
            audit_reports["macro"] = report

    # 4. Объединение
    result = pd.concat([all_per_asset, cross, macro_frame_filled], axis=1)
    result.index.name = "date"

    # Сохраняем close target для удобства разметки
    result["target_close"] = metals[target]["close"].reindex(target_index)
    result["target_high"] = metals[target]["high"].reindex(target_index)
    result["target_low"] = metals[target]["low"].reindex(target_index)

    # Total NaN after pipeline
    if audit_ffill:
        total_cells = result.size
        nan_cells = result.isna().sum().sum()
        _log.info(
            f"  [ffill audit] FINAL: {result.shape[0]} rows × {result.shape[1]} cols. "
            f"NaN cells: {nan_cells} ({nan_cells/total_cells*100:.1f}%). "
            f"After dropna() will keep ~{result.dropna().shape[0]} rows."
        )

    result.attrs["ffill_audit"] = audit_reports
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=== Feature engine test ===")
    frame = build_feature_frame(target="silver")
    print(f"\nFinal feature frame:")
    print(f"  Rows:    {len(frame)}")
    print(f"  Cols:    {len(frame.columns)}")
    print(f"  Period:  {frame.index.min().date()} → {frame.index.max().date()}")
    print(f"\nNaN coverage (first 50 features):")
    nan_pct = (frame.isna().sum() / len(frame) * 100).round(1)
    print(nan_pct.head(50).to_string())
    print(f"\nNon-NaN rows after dropna: {len(frame.dropna())}")
