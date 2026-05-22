"""Multi-horizon triple-barrier labelling с adaptive volatility-scaled barriers.

Логика на каждом дне `t`:
1. Вычислить realized volatility за окно (по умолчанию 20 дней)
2. Установить TP/SL барьеры как кратные vol (адаптивные)
3. Установить time barrier = horizon (5/10/20/60 дней)
4. Симулировать движение цены по дневным high/low до hit одного из 3 барьеров
5. Метка: +1 (TP), 0 (time-out), -1 (SL)

Для каждого дня создаётся **4 параллельные метки** (по одной на horizon).
Это даёт ×4 supervision signal.

Дополнительно: regime-aware asymmetric barriers — в uptrend TP > |SL|.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.multi_asset.config import HORIZONS, ADAPTIVE_BARRIERS

logger = logging.getLogger(__name__)


def _classify_regime(close: pd.Series, adx: pd.Series | None = None) -> pd.Series:
    """Простая классификация рыночного режима.

    Returns:
        Series со значениями 'uptrend' / 'downtrend' / 'sideways'.
    """
    ma_50 = close.rolling(50).mean()
    slope = ma_50.pct_change(20)  # 20-day slope of 50-day MA

    if adx is not None:
        # Trend: ADX > 25 + положительный slope → uptrend
        is_trend = adx > 25
        is_up = is_trend & (slope > 0)
        is_down = is_trend & (slope < 0)
    else:
        # Без ADX: просто по slope
        is_up = slope > 0.02
        is_down = slope < -0.02

    regime = pd.Series("sideways", index=close.index)
    regime.loc[is_up] = "uptrend"
    regime.loc[is_down] = "downtrend"
    return regime


def _barriers_for_regime(regime: str) -> tuple[float, float]:
    """Возвращает (TP_multiplier, SL_multiplier) для данного режима."""
    if regime == "uptrend":
        return ADAPTIVE_BARRIERS["asymmetric_uptrend"]
    elif regime == "downtrend":
        return ADAPTIVE_BARRIERS["asymmetric_downtrend"]
    else:
        return ADAPTIVE_BARRIERS["range_symmetric"]


def triple_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon: int,
    vol_window: int = 20,
    adaptive: bool = True,
    regime: pd.Series | None = None,
) -> pd.DataFrame:
    """Сгенерировать метки triple-barrier для одного horizon.

    Args:
        close, high, low: ценовые ряды
        horizon: длительность time barrier (дней)
        vol_window: окно для realized vol
        adaptive: использовать adaptive (vol-scaled) барьеры
        regime: серия с режимом ('uptrend'/'downtrend'/'sideways')

    Returns:
        DataFrame с колонками:
          - label: -1/0/+1
          - entry_price
          - exit_price
          - exit_date
          - hold_days
          - tp_barrier, sl_barrier (для проверки)
          - exit_reason: 'tp'/'sl'/'time'
    """
    # Adaptive vol: годовая realized vol → дневная sigma
    rvol = (np.log(close / close.shift(1))).rolling(vol_window).std()

    # Targets с регимом
    if adaptive:
        if regime is None:
            regime = _classify_regime(close)
        tp_mult = regime.map({"uptrend": ADAPTIVE_BARRIERS["asymmetric_uptrend"][0],
                              "downtrend": ADAPTIVE_BARRIERS["asymmetric_downtrend"][0],
                              "sideways": ADAPTIVE_BARRIERS["range_symmetric"][0]})
        sl_mult = regime.map({"uptrend": ADAPTIVE_BARRIERS["asymmetric_uptrend"][1],
                              "downtrend": ADAPTIVE_BARRIERS["asymmetric_downtrend"][1],
                              "sideways": ADAPTIVE_BARRIERS["range_symmetric"][1]})
    else:
        base = ADAPTIVE_BARRIERS["base_multiplier"]
        tp_mult = pd.Series(base, index=close.index)
        sl_mult = pd.Series(base, index=close.index)

    n = len(close)
    out = pd.DataFrame(
        index=close.index,
        columns=["label", "entry_price", "exit_price", "exit_date",
                 "hold_days", "tp_barrier", "sl_barrier", "exit_reason"],
    )

    close_arr = close.values
    high_arr = high.values
    low_arr = low.values
    rvol_arr = rvol.values
    tp_mult_arr = tp_mult.values
    sl_mult_arr = sl_mult.values
    dates = close.index

    for i in range(n):
        sigma = rvol_arr[i]
        if not np.isfinite(sigma) or sigma == 0:
            continue
        entry = close_arr[i]
        # Барьеры в множителях sigma
        tp_level = entry * (1 + tp_mult_arr[i] * sigma)
        sl_level = entry * (1 - sl_mult_arr[i] * sigma)

        # Симулируем вперёд до horizon или barrier hit
        end_idx = min(i + horizon, n - 1)
        exit_idx = end_idx
        exit_reason = "time"
        exit_price = close_arr[end_idx]
        label = 0

        for j in range(i + 1, end_idx + 1):
            h, l = high_arr[j], low_arr[j]
            # Сначала проверяем worst case: оба барьера могли быть hit за день
            tp_hit = h >= tp_level
            sl_hit = l <= sl_level
            if tp_hit and sl_hit:
                # Консервативно: первым считаем SL (worst case)
                exit_idx = j
                exit_reason = "sl"
                exit_price = sl_level
                label = -1
                break
            elif tp_hit:
                exit_idx = j
                exit_reason = "tp"
                exit_price = tp_level
                label = 1
                break
            elif sl_hit:
                exit_idx = j
                exit_reason = "sl"
                exit_price = sl_level
                label = -1
                break

        out.iloc[i] = [
            label,
            entry,
            exit_price,
            dates[exit_idx],
            (dates[exit_idx] - dates[i]).days,
            tp_level,
            sl_level,
            exit_reason,
        ]

    # Конвертация типов
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out["entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out["exit_price"] = pd.to_numeric(out["exit_price"], errors="coerce")
    out["hold_days"] = pd.to_numeric(out["hold_days"], errors="coerce")
    out["tp_barrier"] = pd.to_numeric(out["tp_barrier"], errors="coerce")
    out["sl_barrier"] = pd.to_numeric(out["sl_barrier"], errors="coerce")

    return out


def build_multi_horizon_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizons: list[int] | None = None,
    vol_window: int = 20,
    adaptive: bool = True,
) -> pd.DataFrame:
    """Построить метки для всех horizons.

    Args:
        close, high, low: ценовые ряды
        horizons: список длительностей. По умолчанию [5, 10, 20, 60].
        vol_window: окно vol
        adaptive: vol-scaled барьеры

    Returns:
        DataFrame с колонками `label_5`, `label_10`, `label_20`, `label_60`
        и дополнительными `exit_reason_<h>`, `hold_days_<h>` для каждого horizon.
    """
    horizons = horizons or HORIZONS
    regime = _classify_regime(close)

    out = pd.DataFrame(index=close.index)
    out["regime"] = regime

    for h in horizons:
        logger.info(f"  building labels for horizon={h}...")
        labels = triple_barrier_labels(
            close, high, low, h, vol_window=vol_window,
            adaptive=adaptive, regime=regime,
        )
        out[f"label_{h}"] = labels["label"]
        out[f"hold_days_{h}"] = labels["hold_days"]
        out[f"exit_reason_{h}"] = labels["exit_reason"]

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=== Multi-horizon labels test ===")

    from app.multi_asset.metal_loader import load_single_metal

    silver = load_single_metal("silver")
    labels = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
    )

    print(f"\nRows: {len(labels)}, cols: {len(labels.columns)}")
    print(f"\nLabel distribution by horizon:")
    for h in HORIZONS:
        col = f"label_{h}"
        vc = labels[col].value_counts(dropna=False).sort_index()
        total = vc.sum()
        print(f"\nHorizon {h} days (n={total}):")
        for lbl, cnt in vc.items():
            pct = cnt / total * 100 if total > 0 else 0
            print(f"  {lbl}: {cnt} ({pct:.1f}%)")

    print(f"\nRegime distribution:")
    print(labels["regime"].value_counts(dropna=False))

    print(f"\nExit reason for horizon=20:")
    print(labels["exit_reason_20"].value_counts(dropna=False))
