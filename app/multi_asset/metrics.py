"""Метрики для оценки эксперимента: Sharpe, Sortino, DSR, max DD, profit factor."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# Annualization factor for daily/per-trade returns
TRADING_DAYS_YEAR = 252


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_YEAR) -> float:
    """Annualized Sharpe (zero risk-free)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_YEAR) -> float:
    """Annualized Sortino (downside deviation)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> dict:
    """Максимальная просадка по equity curve."""
    if len(equity) == 0:
        return {"max_dd": 0.0, "max_dd_start": None, "max_dd_end": None}
    cummax = equity.cummax()
    dd = equity / cummax - 1
    max_dd = float(dd.min())
    end_idx = dd.idxmin() if len(dd) > 0 else None
    # Start = last time we were at peak before end_idx
    if end_idx is not None:
        before = equity.loc[:end_idx]
        start_idx = before.idxmax()
    else:
        start_idx = None
    return {"max_dd": max_dd, "max_dd_start": start_idx, "max_dd_end": end_idx}


def profit_factor(returns: np.ndarray) -> float:
    """PF = sum positive returns / |sum negative returns|."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def psr(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0,
        sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey-López de Prado).

    Вероятность того, что истинный Sharpe > sr_benchmark при наблюдаемом sr,
    учитывая skew и kurt.
    """
    if n < 2:
        return 0.0
    sr_std = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    z = (sr - sr_benchmark) / sr_std if sr_std > 0 else 0
    return float(stats.norm.cdf(z))


def dsr(sr: float, n: int, n_trials: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio.

    Корректирует на multiple testing — какова вероятность что best SR из N_trials
    действительно превышает порог, а не случайность.
    """
    if n < 2 or n_trials < 1:
        return 0.0
    # Ожидаемый максимум SR из n_trials нормальных стратегий
    emg = 0.5772  # Euler-Mascheroni constant
    expected_max = (
        (1 - emg) * stats.norm.ppf(1 - 1 / n_trials)
        + emg * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    return psr(sr, n, skew, kurt, sr_benchmark=expected_max)


def compute_all_metrics(
    trades_df: pd.DataFrame,
    n_trials: int = 1,
    period_years: float | None = None,
) -> dict:
    """Полный набор метрик для эксперимента.

    Args:
        trades_df: DataFrame со сделками (колонки: net_return, gross_return, entry_date, exit_date, hold_days)
        n_trials: количество протестированных конфигураций (для DSR)
        period_years: длительность бэктеста в годах (для annual return). Если None — считается из дат.
    """
    if trades_df.empty:
        return {"n_trades": 0}

    nr = trades_df["net_return"].dropna().values

    # Period
    if period_years is None:
        first = pd.to_datetime(trades_df["entry_date"]).min()
        last = pd.to_datetime(trades_df["exit_date"]).max()
        period_years = max((last - first).days / 365.25, 0.01)

    # Compound total return
    total = float(np.prod(1 + nr) - 1)
    annual = float((1 + total) ** (1 / period_years) - 1) if period_years > 0 else 0

    # Per-trade Sharpe (using trade returns, annualized by trades/year)
    trades_per_year = len(nr) / period_years if period_years > 0 else 0
    sr_per_trade = nr.mean() / nr.std() if nr.std() > 0 else 0
    sr_annual = sr_per_trade * np.sqrt(trades_per_year) if trades_per_year > 0 else 0
    sortino_per_trade = (
        nr.mean() / nr[nr < 0].std() if len(nr[nr < 0]) > 0 and nr[nr < 0].std() > 0 else 0
    )
    sortino_annual = sortino_per_trade * np.sqrt(trades_per_year) if trades_per_year > 0 else 0

    # Equity curve from trades
    eq = pd.Series(np.cumprod(1 + nr), index=pd.to_datetime(trades_df["exit_date"]))
    dd_info = max_drawdown(eq)

    # Skew/kurt для DSR
    skew = float(stats.skew(nr)) if len(nr) >= 3 else 0
    kurt = float(stats.kurtosis(nr, fisher=False)) if len(nr) >= 4 else 3  # excess + 3 = raw

    return {
        "n_trades":         len(nr),
        "period_years":     round(float(period_years), 2),
        "trades_per_year":  round(float(trades_per_year), 2),
        "total_return":     round(total, 4),
        "annual_return":    round(annual, 4),
        "sharpe":           round(float(sr_annual), 3),
        "sortino":          round(float(sortino_annual), 3),
        "max_dd":           round(float(dd_info["max_dd"]), 4),
        "profit_factor":    round(profit_factor(nr), 3),
        "win_rate":         round(win_rate(nr), 3),
        "mean_win":         round(float(nr[nr > 0].mean()), 4) if (nr > 0).any() else 0,
        "mean_loss":        round(float(nr[nr <= 0].mean()), 4) if (nr <= 0).any() else 0,
        "mean_return":      round(float(nr.mean()), 4),
        "median_return":    round(float(np.median(nr)), 4),
        "best_trade":       round(float(nr.max()), 4),
        "worst_trade":      round(float(nr.min()), 4),
        "skew":             round(skew, 3),
        "kurtosis":         round(kurt, 3),
        "psr":              round(psr(sr_annual, len(nr), skew, kurt), 3),
        "dsr":              round(dsr(sr_annual, len(nr), n_trials, skew, kurt), 3),
    }


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval для Sharpe."""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return {"sharpe_mean": 0, "sharpe_lower": 0, "sharpe_upper": 0}
    sharpes = []
    for _ in range(n_iterations):
        sample = rng.choice(r, size=len(r), replace=True)
        if sample.std() > 0:
            sr = sample.mean() / sample.std()
            sharpes.append(sr)
    sharpes = np.array(sharpes)
    alpha = (1 - confidence) / 2
    return {
        "sharpe_mean":  float(sharpes.mean()),
        "sharpe_lower": float(np.quantile(sharpes, alpha)),
        "sharpe_upper": float(np.quantile(sharpes, 1 - alpha)),
    }
