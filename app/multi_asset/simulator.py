"""Симуляция сделок по predictions с механикой как в текущей V25 модели.

Параметры (OptimalV2):
  - entry_threshold: p_up >= 0.48 → BUY
  - exit_threshold: p_up < 0.35 → SELL (если в позиции)
  - trailing stop: 12% от пикового уровня
  - max_hold: 30 торговых дней
  - cooldown: 25 торговых дней между сделками
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeConfig:
    entry_threshold: float = 0.48      # p_up для входа
    exit_threshold: float = 0.35       # p_up для выхода (если в позиции)
    trail_pct: float = 0.12            # trailing stop %
    max_hold_days: int = 30            # максимальное удержание
    cooldown_days: int = 25            # пауза между сделками
    commission_pct: float = 0.001      # 0.1% за сделку (round-trip 0.2%)
    spread_pct: float = 0.0            # bid-ask spread (0.002 = 0.2% on entry/exit)
    slippage_pct: float = 0.0          # market impact per trade (0.001 = 0.1%)
    direction_label: int = 1           # какой класс считаем "вверх"
                                       # (1 = TP в triple-barrier)
    enable_short: bool = False         # позволять short когда p_short высока
    short_entry_threshold: float = 0.48  # p_-1 порог для входа в short
    vol_target_annual: float = 0.0     # 0 = выкл; иначе target annualized vol (0.15 = 15%)
    vol_lookback: int = 20             # окно для realized vol


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    peak_price: float
    hold_days: int
    gross_return: float
    net_return: float
    exit_reason: str  # 'trail' | 'max_hold' | 'model_exit'


def simulate_trades(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    config: TradeConfig | None = None,
) -> tuple[list[Trade], pd.Series]:
    """Симулировать сделки по предсказаниям модели.

    Args:
        predictions: DataFrame с колонкой p_<class> (например, p_1 для TP)
                     или просто 'p_up'. Индекс — даты.
        prices: DataFrame с колонками [close, high, low]. Индекс совпадает.
        config: параметры стратегии

    Returns:
        trades: список Trade
        equity: Series накопленной доходности (на каждый день в позиции)
    """
    config = config or TradeConfig()

    # Извлекаем p_up как вероятность направления вверх
    proba_col = f"p_{config.direction_label}"
    if proba_col in predictions.columns:
        p_up = predictions[proba_col]
    elif "p_up" in predictions.columns:
        p_up = predictions["p_up"]
    else:
        raise ValueError(f"Cannot find probability column. Available: {list(predictions.columns)}")

    # Выровнять индексы
    common = predictions.index.intersection(prices.index)
    p_up = p_up.reindex(common)
    prices = prices.reindex(common)
    dates = common.tolist()

    trades: list[Trade] = []
    state = "FLAT"  # FLAT | LONG
    entry_idx = None
    entry_price = None
    peak_price = None
    cooldown_until_idx = -1

    closes = prices["close"].values
    highs = prices["high"].values
    lows = prices["low"].values
    p_up_arr = p_up.values

    for i, date in enumerate(dates):
        close = closes[i]
        high = highs[i]
        low = lows[i]
        p = p_up_arr[i]

        if not np.isfinite(close):
            continue

        if state == "LONG":
            # Update peak
            if high > peak_price:
                peak_price = high

            hold_days = i - entry_idx
            trail_level = peak_price * (1 - config.trail_pct)
            exit_reason = None
            exit_price = None

            # Check trail (intraday low может пробить)
            if low <= trail_level:
                exit_reason = "trail"
                exit_price = trail_level
            # Check max hold
            elif hold_days >= config.max_hold_days:
                exit_reason = "max_hold"
                exit_price = close
            # Check model exit signal
            elif np.isfinite(p) and p < config.exit_threshold:
                exit_reason = "model_exit"
                exit_price = close

            if exit_reason is not None:
                gross_return = exit_price / entry_price - 1
                # Realistic costs: commission + spread + slippage on both entry & exit
                total_cost = 2 * (config.commission_pct + config.spread_pct + config.slippage_pct)
                net_return = gross_return - total_cost
                trades.append(Trade(
                    entry_date=dates[entry_idx],
                    exit_date=date,
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    peak_price=float(peak_price),
                    hold_days=int(hold_days),
                    gross_return=float(gross_return),
                    net_return=float(net_return),
                    exit_reason=exit_reason,
                ))
                state = "FLAT"
                cooldown_until_idx = i + config.cooldown_days
                entry_idx = None
                entry_price = None
                peak_price = None

        elif state == "FLAT":
            if i < cooldown_until_idx:
                continue
            if not np.isfinite(p):
                continue
            if p >= config.entry_threshold:
                state = "LONG"
                entry_idx = i
                entry_price = close
                peak_price = high

    # Build equity series
    if not trades:
        equity = pd.Series([1.0], index=[common[-1]] if len(common) else [])
        return trades, equity

    eq_values = []
    eq_dates = []
    cum = 1.0
    for t in trades:
        cum *= (1 + t.net_return)
        eq_values.append(cum)
        eq_dates.append(t.exit_date)
    equity = pd.Series(eq_values, index=pd.DatetimeIndex(eq_dates))

    return trades, equity


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        "entry_date":   t.entry_date,
        "exit_date":    t.exit_date,
        "entry_price":  t.entry_price,
        "exit_price":   t.exit_price,
        "peak_price":   t.peak_price,
        "hold_days":    t.hold_days,
        "gross_return": t.gross_return,
        "net_return":   t.net_return,
        "exit_reason":  t.exit_reason,
    } for t in trades])
