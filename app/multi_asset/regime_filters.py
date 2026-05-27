"""
app/multi_asset/regime_filters.py — фильтры режимов рынка для simulator.

Идея: модель E3b предсказывает 20-дневное направление, но НЕ распознаёт
структурные режимы. Добавляем 3 простых rule-based фильтра + 1 HMM-based:

1. trend_filter(price, sma_period=200): не входить LONG если price < SMA200
   → исключает downtrend периоды
2. volatility_filter(price, atr_period=14, pctile=0.90): не входить если ATR
   в top 10% распределения → исключает хаотичные дни
3. consecutive_signal_filter(p_up, threshold, n=3): требовать N дней подряд
   p_up > threshold → фильтр шумовых спайков (мы это уже частично сделали
   через rolling mean, но это альтернатива)
4. hmm_filter(returns, n_states=3): тренируем HMM, разрешаем trade только
   в bull state

Эти фильтры — это GATING (не идут в модель как features). Применяются ПОСЛЕ
модели чтобы решить «стоит ли действовать на сигнал».
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def trend_filter(close: pd.Series, sma_period: int = 200) -> pd.Series:
    """
    True (можно входить) если price > SMA200.
    Industry standard «не торгуй против тренда».
    """
    sma = close.rolling(sma_period, min_periods=sma_period // 2).mean()
    return close > sma


def volatility_filter(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    pctile: float = 0.90,
) -> pd.Series:
    """
    True (можно входить) если ATR < pctile percentile.
    Не входим в очень волатильные дни (хаос/паника).
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period // 2).mean()
    # rolling percentile (expanding для устойчивости)
    threshold = atr.expanding(min_periods=252).quantile(pctile)
    return atr < threshold


def consecutive_signal_filter(
    p_up: pd.Series,
    threshold: float = 0.48,
    n_consecutive: int = 3,
) -> pd.Series:
    """
    True если p_up >= threshold для N последних дней подряд.
    """
    above = p_up >= threshold
    rolling_min = above.rolling(n_consecutive, min_periods=n_consecutive).min()
    return rolling_min == 1


def hmm_filter(
    returns: pd.Series,
    n_states: int = 3,
    train_window: int = 1000,
    refit_every: int = 30,
) -> tuple[pd.Series, pd.Series]:
    """
    Регим-детектор на returns. Использует sklearn GaussianMixture
    (HMM-замена не требующая C++ build tools).

    Подход: на каждом шаге берём rolling features [return, vol_20], кластеризуем
    в n_states. Bull state = кластер с самым высоким mean return.

    Returns:
        (in_bull: Series[bool] — True если в bull state,
         state_label: Series[int] — индекс кластера)
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        logger.warning("sklearn missing -> hmm_filter passthrough")
        return (
            pd.Series([True] * len(returns), index=returns.index),
            pd.Series([0] * len(returns), index=returns.index),
        )

    returns = returns.dropna()
    n = len(returns)

    # Feature: [return_5d_mean, rolling_vol_20d]
    feat = pd.DataFrame({
        "r5":   returns.rolling(5).mean(),
        "vol":  returns.rolling(20).std(),
    }).dropna()

    if len(feat) < train_window:
        return (
            pd.Series([True] * len(returns), index=returns.index),
            pd.Series([0] * len(returns), index=returns.index),
        )

    state_labels = pd.Series(index=feat.index, dtype=float)
    in_bull = pd.Series(index=feat.index, dtype=bool)
    last_model = None
    last_bull_state = 0

    for i in range(train_window, len(feat)):
        if last_model is None or (i - train_window) % refit_every == 0:
            try:
                X_train = feat.iloc[i - train_window:i].values
                gmm = GaussianMixture(
                    n_components=n_states,
                    covariance_type="full",
                    max_iter=100,
                    random_state=42,
                )
                gmm.fit(X_train)
                # Bull = кластер с max mean returns (первая фича = r5)
                means_r5 = gmm.means_[:, 0]
                last_bull_state = int(np.argmax(means_r5))
                last_model = gmm
            except Exception:
                last_model = None
                last_bull_state = 0

        if last_model is not None:
            try:
                x_today = feat.iloc[[i]].values
                current_state = int(last_model.predict(x_today)[0])
            except Exception:
                current_state = last_bull_state
        else:
            current_state = last_bull_state

        state_labels.iloc[i] = current_state
        in_bull.iloc[i] = current_state == last_bull_state

    # Reindex обратно на returns.index
    state_labels = state_labels.reindex(returns.index, fill_value=-1)
    in_bull = in_bull.reindex(returns.index, fill_value=True).astype(bool)
    return in_bull, state_labels


def apply_filters(
    p_up: pd.Series,
    prices: pd.DataFrame,
    use_trend: bool = True,
    use_vol: bool = True,
    use_hmm: bool = False,
    hmm_train_window: int = 1000,
) -> pd.Series:
    """
    Применяет выбранные фильтры. Возвращает мaskingovannyy p_up:
    везде где фильтр запрещает торговлю — p_up зануляется (0).
    Это позволяет использовать с обычным simulator без модификации.
    """
    common = p_up.index.intersection(prices.index)
    p = p_up.reindex(common).copy()
    px = prices.reindex(common)

    final_mask = pd.Series([True] * len(common), index=common)

    if use_trend:
        m_trend = trend_filter(px["close"]).reindex(common).fillna(False)
        final_mask &= m_trend
        logger.info(f"  trend_filter: {m_trend.sum()}/{len(m_trend)} days allowed")

    if use_vol:
        m_vol = volatility_filter(px["high"], px["low"], px["close"]).reindex(common).fillna(True)
        final_mask &= m_vol
        logger.info(f"  vol_filter:   {m_vol.sum()}/{len(m_vol)} days allowed")

    if use_hmm:
        returns = px["close"].pct_change().dropna()
        in_bull, _ = hmm_filter(returns, train_window=hmm_train_window)
        in_bull = in_bull.reindex(common).fillna(True)
        final_mask &= in_bull
        logger.info(f"  hmm_filter:   {in_bull.sum()}/{len(in_bull)} days in bull")

    logger.info(f"  total allowed: {final_mask.sum()}/{len(final_mask)}")

    # Где фильтр запретил — обнуляем p_up (не пройдёт entry_threshold)
    p_filtered = p.where(final_mask, 0.0)
    return p_filtered
