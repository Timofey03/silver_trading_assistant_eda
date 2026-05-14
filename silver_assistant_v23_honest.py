"""
Silver Trading Assistant v23 — Honest Math + Statistical Robustness

Чинит методологические проблемы v22, выявленные в анализе:

1) Compounded equity (cumprod) вместо суммы (cumsum)
2) Sequential single-position-at-a-time equity (никаких overlapping LONGs)
3) Apples-to-apples BnH (один и тот же режим капитала, что и стратегия)
4) Deflated Sharpe Ratio (López de Prado, 2014) — поправка на multiple testing
5) Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012)
6) Stationary block bootstrap CI (Politis-Romano) на total_return / Sharpe / MaxDD
7) CPCV (Combinatorial Purged Cross-Validation) — скелет для будущей валидации
8) Extended features: DXY, VIX, MOVE, Gold/Silver ratio
9) Realistic execution costs: spread_bps + ATR-based slippage + funding
10) Liquidity gate (volume filter)
11) Performance attribution: model / sizing / execution / regime
12) Audit log structure (JSON per decision)
13) Drift detection (KS-test) — фреймворк для онлайн-мониторинга

Запуск:
  python silver_assistant_v23_honest.py                   # пересчёт v22 trades
  python silver_assistant_v23_honest.py --bootstrap 2000  # bootstrap CI
  python silver_assistant_v23_honest.py --fetch-features  # обновить DXY/VIX/GLD
  python silver_assistant_v23_honest.py --cpcv-demo       # CPCV демонстрация
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import warnings
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


V22_DIR = Path("baseline_outputs_v22")
V23_DIR = Path("baseline_outputs_v23")
V23_DIR.mkdir(exist_ok=True)

TRADING_DAYS_PER_YEAR = 252
RNG_SEED = 20260514


# ===========================================================================
# 1. HONEST EQUITY — compounded + single-position-at-a-time
# ===========================================================================

def equity_compounded_sequential(
    trades: pd.DataFrame,
    return_col: str = "net_return",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Возвращает (equity_curve, sequential_trades).

    Логика «один счёт, одна позиция»:
      • Сортируем сделки по entry_date.
      • Берём первую сделку. Все последующие, чьё entry_date < exit_date
        текущей — ИГНОРИРУЕМ (нет капитала на параллельные позиции).
      • Equity компаундируется: E_{i+1} = E_i * (1 + r_i).

    Это превращает «279% sum-of-returns на 45 пересекающихся сделках»
    в реальное число, которое мог бы получить трейдер с одним счётом.
    """
    if trades.empty:
        return np.array([1.0]), pd.DataFrame()

    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"]  = pd.to_datetime(t["exit_date"])
    t = t.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)

    kept_idx: List[int] = []
    blocked_until = pd.Timestamp("1900-01-01")
    for i, row in t.iterrows():
        if row["entry_date"] >= blocked_until:
            kept_idx.append(i)
            blocked_until = row["exit_date"]
    seq = t.iloc[kept_idx].reset_index(drop=True)

    rets = seq[return_col].astype(float).values
    equity = np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    return equity, seq


def equity_compounded_overlapping_unit(
    trades: pd.DataFrame,
    return_col: str = "net_return",
) -> np.ndarray:
    """
    Альтернативный режим: разрешаем overlap, но каждый трейд = равный
    notional $1 (без реинвеста). Возвращает equity = 1 + cum_pnl_dollars.
    Это «честная» версия cumsum: показывает, сколько вы получите, если
    у вас бесконечный капитал и каждый трейд — $1 фиксированно.
    """
    if trades.empty:
        return np.array([1.0])
    t = trades.copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    t = t.sort_values("exit_date").reset_index(drop=True)
    return np.concatenate([[1.0], 1.0 + np.cumsum(t[return_col].astype(float).values)])


def true_buy_and_hold(
    price_series: pd.Series,
    split_dates: Tuple[pd.Timestamp, pd.Timestamp],
) -> float:
    """
    Настоящий buy-and-hold: вошли в первый день сплита, вышли в последний.
    Возвращает compound return (e.g. 0.52 = +52%).
    """
    p = price_series.dropna()
    p = p[(p.index >= split_dates[0]) & (p.index <= split_dates[1])]
    if len(p) < 2:
        return float("nan")
    return float(p.iloc[-1] / p.iloc[0] - 1.0)


# ===========================================================================
# 2. RISK METRICS на честной equity
# ===========================================================================

def risk_metrics_honest(
    equity: np.ndarray,
    n_periods_year: float = TRADING_DAYS_PER_YEAR,
    n_trades: int = 0,
    trade_days_total: int = 0,
) -> Dict[str, Optional[float]]:
    """
    Считает риск-метрики по equity (compound), а не по сумме returns.
    trade_days_total: суммарная длина в днях между entry первого
    и exit последнего трейда (для аннуализации).
    """
    if len(equity) < 2:
        return {
            "total_return": 0.0, "cagr": None, "max_drawdown": None,
            "sharpe": None, "sortino": None, "calmar": None,
            "ulcer": None, "time_underwater": None,
        }

    total_return = float(equity[-1] / equity[0] - 1.0)

    years = max(trade_days_total / 365.25, 0.1) if trade_days_total > 0 else 1.0
    if equity[-1] > 0:
        cagr = float(equity[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = -1.0

    running_max = np.maximum.accumulate(equity)
    dd_series   = equity / running_max - 1.0
    max_dd      = float(dd_series.min())
    time_uw     = float((dd_series < -0.001).mean())
    ulcer       = float(np.sqrt(np.mean(dd_series ** 2)))

    period_rets = np.diff(equity) / equity[:-1]
    if len(period_rets) >= 3 and period_rets.std(ddof=1) > 0:
        mean_r = period_rets.mean()
        std_r  = period_rets.std(ddof=1)
        per_year = (n_trades / years) if (n_trades > 0 and years > 0) else n_periods_year
        sharpe  = float(mean_r / std_r * math.sqrt(per_year))
        downside = period_rets[period_rets < 0]
        if len(downside) > 1 and downside.std(ddof=1) > 0:
            sortino = float(mean_r / downside.std(ddof=1) * math.sqrt(per_year))
        else:
            sortino = None
    else:
        sharpe  = None
        sortino = None

    calmar = (cagr / abs(max_dd)) if (cagr is not None and max_dd < -0.001) else None

    return {
        "total_return":   total_return,
        "cagr":           cagr,
        "max_drawdown":   max_dd,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "ulcer":          ulcer,
        "time_underwater": time_uw,
    }


# ===========================================================================
# 3. DEFLATED SHARPE RATIO + PROBABILISTIC SHARPE RATIO
#    (Bailey & López de Prado, 2012, 2014)
# ===========================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_ppf(p: float) -> float:
    p = max(min(p, 1 - 1e-12), 1e-12)
    a = [-39.696830, 220.946098, -275.928510, 138.357751, -30.664798, 2.506628]
    b = [-54.476098, 161.585836, -155.698979, 66.801311, -13.280681]
    c = [-0.007784894002, -0.32239645, -2.400758, -2.549732, 4.374664, 2.938163]
    d = [0.007784695709, 0.32246712, 2.445134, 3.754408]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def probabilistic_sharpe_ratio(
    sharpe_obs: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> float:
    """
    PSR = вероятность, что true Sharpe > sharpe_benchmark, при наблюдённом sharpe_obs.
    (Bailey & López de Prado, 2012, eq. 3)
    """
    if n_obs < 4:
        return float("nan")
    excess_kurt = kurt - 3.0
    se = math.sqrt(
        (1 - skew * sharpe_obs + (excess_kurt / 4.0) * sharpe_obs**2)
        / (n_obs - 1)
    )
    if se <= 0:
        return float("nan")
    z = (sharpe_obs - sharpe_benchmark) / se
    return _norm_cdf(z)


def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_obs: int,
    n_trials: int,
    sharpe_variance: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    DSR — поправка на multiple testing: вероятность что true Sharpe > 0, учитывая что
    вы выбрали лучший из n_trials стратегий со стандартным отклонением Sharpe sharpe_variance.

    (López de Prado, 2014, eq. 5)
    """
    if n_obs < 4 or n_trials < 1:
        return float("nan")
    emc = 0.5772156649  # Euler-Mascheroni
    if n_trials > 1 and sharpe_variance > 0:
        sharpe0 = math.sqrt(sharpe_variance) * (
            (1.0 - emc) * _norm_ppf(1.0 - 1.0/n_trials)
            + emc * _norm_ppf(1.0 - 1.0/(n_trials * math.e))
        )
    else:
        sharpe0 = 0.0
    return probabilistic_sharpe_ratio(
        sharpe_obs, n_obs, skew=skew, kurt=kurt, sharpe_benchmark=sharpe0,
    )


def sharpe_stats(returns: np.ndarray) -> Tuple[float, float, float]:
    """Возвращает (sharpe_per_period, skew, kurt) — без аннуализации."""
    if len(returns) < 4:
        return (float("nan"), 0.0, 3.0)
    m = np.mean(returns)
    s = np.std(returns, ddof=1)
    if s <= 0:
        return (float("nan"), 0.0, 3.0)
    z = (returns - m) / s
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    return (float(m / s), skew, kurt)


# ===========================================================================
# 4. STATIONARY BLOCK BOOTSTRAP (Politis-Romano)
# ===========================================================================

def stationary_bootstrap(
    returns: np.ndarray,
    n_boot: int = 2000,
    expected_block_len: float = 5.0,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """
    Stationary bootstrap (Politis & Romano, 1994): блоки случайной длины с
    geom(p), p = 1 / expected_block_len. Сохраняет автокорреляцию.

    Возвращает матрицу (n_boot, len(returns)).
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < 2:
        return np.tile(returns, (n_boot, 1))
    p = 1.0 / max(expected_block_len, 1.0)

    out = np.empty((n_boot, n), dtype=float)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(0, n)
        for i in range(1, n):
            if rng.random() < p:
                idx[i] = rng.integers(0, n)
            else:
                idx[i] = (idx[i - 1] + 1) % n
        out[b] = returns[idx]
    return out


def bootstrap_ci_metrics(
    seq_returns: np.ndarray,
    n_boot: int = 2000,
    block_len: float = 5.0,
    seed: int = RNG_SEED,
    n_periods_year: float = TRADING_DAYS_PER_YEAR,
) -> Dict[str, Dict[str, float]]:
    """
    Возвращает 95% CI для total_return / sharpe / max_drawdown
    по stationary block bootstrap.
    """
    if len(seq_returns) < 3:
        return {}
    boot = stationary_bootstrap(seq_returns, n_boot=n_boot,
                                expected_block_len=block_len, seed=seed)
    boot_total = np.prod(1.0 + boot, axis=1) - 1.0
    boot_eq    = np.cumprod(1.0 + boot, axis=1)
    running_max = np.maximum.accumulate(boot_eq, axis=1)
    boot_mdd = (boot_eq / running_max - 1.0).min(axis=1)
    means = boot.mean(axis=1)
    stds  = boot.std(axis=1, ddof=1)
    safe_std = np.where(stds > 0, stds, np.nan)
    boot_sharpe = means / safe_std * math.sqrt(n_periods_year / max(len(seq_returns), 1))

    def _ci(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {"lower": float("nan"), "median": float("nan"), "upper": float("nan")}
        return {
            "lower":  float(np.percentile(arr, 2.5)),
            "median": float(np.percentile(arr, 50)),
            "upper":  float(np.percentile(arr, 97.5)),
        }

    return {
        "total_return": _ci(boot_total),
        "sharpe":       _ci(boot_sharpe),
        "max_drawdown": _ci(boot_mdd),
    }


# ===========================================================================
# 5. CPCV — Combinatorial Purged Cross-Validation (López de Prado, AFML ch.7)
# ===========================================================================

@dataclass
class CPCVSplit:
    train_idx: np.ndarray
    test_idx:  np.ndarray
    test_groups: Tuple[int, ...]


def cpcv_splits(
    n_samples: int,
    label_endtimes: np.ndarray,    # exit_date.values для каждой выборки
    n_groups: int = 6,
    k_test: int = 2,
    embargo_frac: float = 0.01,
) -> List[CPCVSplit]:
    """
    CPCV: делим данные на n_groups, тестируем все комбинации из k_test групп.
    Покрываем все backtest paths без overfitting к одному train/test split.

    Purging: убираем из train те выборки, чьи labels (entry→exit) пересекают
    test-окно. Embargo: дополнительная зона после test для предотвращения утечки.

    Возвращает список (n_groups choose k_test) разбиений.
    """
    if n_samples < n_groups * 3:
        return []
    sample_idx = np.arange(n_samples)
    boundaries = np.linspace(0, n_samples, n_groups + 1, dtype=int)
    groups = [sample_idx[boundaries[g]:boundaries[g+1]] for g in range(n_groups)]
    end_t  = np.asarray(label_endtimes, dtype="datetime64[ns]")
    times  = np.arange(n_samples)
    embargo = max(1, int(embargo_frac * n_samples))

    out: List[CPCVSplit] = []
    for combo in combinations(range(n_groups), k_test):
        test_idx_list = []
        for g in combo:
            test_idx_list.extend(groups[g].tolist())
        test_idx = np.array(sorted(test_idx_list), dtype=int)

        test_windows = [(groups[g].min(), groups[g].max()) for g in combo]

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        for (t_start, t_end) in test_windows:
            train_mask[max(0, t_start - embargo): min(n_samples, t_end + embargo + 1)] = False

        end_pos = pd.to_datetime(pd.Series(end_t)).rank(method="dense").values - 1
        for (t_start, t_end) in test_windows:
            for i in np.where(train_mask)[0]:
                if i <= t_end and end_pos[i] >= t_start:
                    train_mask[i] = False

        train_idx = np.where(train_mask)[0]
        if len(train_idx) < 50 or len(test_idx) < 10:
            continue
        out.append(CPCVSplit(train_idx=train_idx, test_idx=test_idx,
                             test_groups=tuple(combo)))
    return out


# ===========================================================================
# 6. EXTENDED FEATURES: DXY, VIX, MOVE, Gold/Silver ratio
# ===========================================================================

EXTENDED_TICKERS: Dict[str, str] = {
    "DX-Y.NYB":  "dxy_close",       # Dollar index
    "^VIX":      "vix_close",       # CBOE VIX
    "^MOVE":     "move_close",      # Bond volatility (часто нет данных, fallback)
    "GC=F":      "gold_close",      # Gold futures (для silver/gold ratio)
}


def fetch_extended_features(start: str, end: str) -> pd.DataFrame:
    """Скачивает DXY/VIX/MOVE/GC через yfinance. Тихо игнорирует пустые тикеры."""
    try:
        import yfinance as yf
    except ImportError:
        print("  WARN: yfinance не установлен, расширенные фичи недоступны")
        return pd.DataFrame()

    frames: Dict[str, pd.Series] = {}
    for ticker, col in EXTENDED_TICKERS.items():
        try:
            data = yf.download(ticker, start=start, end=end,
                               progress=False, auto_adjust=False, threads=False)
            if data is None or data.empty:
                print(f"    WARN {ticker}: empty")
                continue
            s = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
            if hasattr(s, "iloc") and isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.dropna()
            if s.empty:
                continue
            s.index = pd.to_datetime(s.index)
            frames[col] = s
            print(f"    OK  {ticker:10s} → {col}: {len(s)} строк")
        except Exception as e:
            print(f"    FAIL {ticker}: {type(e).__name__}: {e}")

    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index().ffill()
    df.index.name = "Date"
    return df


def add_extended_features(daily: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет: dxy_ret_5/20d, dxy_zscore_20d, vix_level, vix_chg_5d,
               gold_silver_ratio, gsr_zscore_60d, gsr_extreme.
    """
    if ext.empty or daily.empty:
        return daily

    d = daily.copy()
    e = ext.reindex(d.index).ffill()

    if "dxy_close" in e:
        d["dxy_close"]       = e["dxy_close"]
        d["dxy_ret_5d"]      = e["dxy_close"].pct_change(5)
        d["dxy_ret_20d"]     = e["dxy_close"].pct_change(20)
        z20 = (e["dxy_close"] - e["dxy_close"].rolling(20).mean()) / e["dxy_close"].rolling(20).std()
        d["dxy_zscore_20d"]  = z20

    if "vix_close" in e:
        d["vix_level"]       = e["vix_close"]
        d["vix_chg_5d"]      = e["vix_close"].diff(5)
        d["vix_above_25"]    = (e["vix_close"] > 25).astype(int)

    if "move_close" in e:
        d["move_level"]      = e["move_close"]

    if "gold_close" in e and "silver_close" in d.columns:
        d["gold_silver_ratio"] = e["gold_close"] / d["silver_close"]
        gsr = d["gold_silver_ratio"]
        d["gsr_zscore_60d"]  = (gsr - gsr.rolling(60).mean()) / gsr.rolling(60).std()
        d["gsr_extreme_high"] = (d["gsr_zscore_60d"] > 1.5).astype(int)
        d["gsr_extreme_low"]  = (d["gsr_zscore_60d"] < -1.5).astype(int)

    return d


EXTENDED_FEATURE_NAMES: List[str] = [
    "dxy_ret_5d", "dxy_ret_20d", "dxy_zscore_20d",
    "vix_level", "vix_chg_5d", "vix_above_25",
    "move_level",
    "gold_silver_ratio", "gsr_zscore_60d", "gsr_extreme_high", "gsr_extreme_low",
]


# ===========================================================================
# 7. REALISTIC EXECUTION COST MODEL
# ===========================================================================

@dataclass
class RealisticCosts:
    """
    Реалистичная модель costs для фьючерсов SI=F (или приближённо для ETF SLV).

    Компоненты (в долях, не bps):
      spread_base:    минимальный спред (typical mid-day) — 0.0005 (5 bps)
      slippage_atr_k: множитель ATR при market-исполнении — 0.10 (10% от 14d ATR)
      commission:     комиссия roundtrip                  — 0.00025 (2.5 bps)
      funding_short_annual: годовая стоимость SHORT       — 0.005 (50 bps/год)
      illiquid_premium:    надбавка для low-volume дней   — 0.0010 (10 bps)
    """
    spread_base:         float = 0.0005
    slippage_atr_k:      float = 0.10
    commission:          float = 0.00025
    funding_short_annual: float = 0.005
    illiquid_premium:    float = 0.0010

    def round_trip_cost(
        self,
        direction: str,
        atr_pct_entry: float,
        atr_pct_exit:  float,
        hold_days:     int,
        is_illiquid:   bool = False,
    ) -> float:
        """
        Полная стоимость roundtrip: spread + slippage entry + slippage exit
        + commission + (для SHORT) funding * (hold_days / 252).
        """
        spread_total = 2.0 * self.spread_base  # in + out
        slip_in  = self.slippage_atr_k * max(atr_pct_entry, 0.0)
        slip_out = self.slippage_atr_k * max(atr_pct_exit, 0.0)
        comm     = self.commission
        funding  = 0.0
        if direction == "SHORT":
            funding = self.funding_short_annual * (max(hold_days, 0) / 252.0)
        illiq    = 2.0 * self.illiquid_premium if is_illiquid else 0.0
        return spread_total + slip_in + slip_out + comm + funding + illiq


def recompute_trades_with_realistic_costs(
    trades: pd.DataFrame,
    price_df: pd.DataFrame,
    costs:   RealisticCosts,
    atr_col: str = "silver_atr_14d",
    volume_col: str = "silver_volume",
    illiquid_threshold_pct: float = 0.5,  # 50% от median volume
) -> pd.DataFrame:
    """
    Пересчитывает net_return каждого трейда с реалистичными costs.
    Восстанавливает gross_return, потом вычитает RealisticCosts.round_trip_cost().
    Требует price_df с silver_close + silver_atr_14d + (опц.) silver_volume.
    """
    if trades.empty:
        return trades.copy()

    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"]  = pd.to_datetime(t["exit_date"])

    has_atr    = atr_col in price_df.columns
    has_volume = volume_col in price_df.columns

    if has_volume:
        vol_med = price_df[volume_col].rolling(60, min_periods=20).median()

    new_costs = []
    is_illiquid_flags = []
    for _, row in t.iterrows():
        d_in  = row["entry_date"]
        d_out = row["exit_date"]
        try:
            atr_in  = float(price_df.loc[d_in,  atr_col]) if has_atr else 0.0
            atr_out = float(price_df.loc[d_out, atr_col]) if has_atr else 0.0
            close_in  = float(price_df.loc[d_in,  "silver_close"])
            close_out = float(price_df.loc[d_out, "silver_close"])
        except KeyError:
            atr_in = atr_out = 0.0
            close_in = close_out = 1.0

        atr_pct_in  = (atr_in  / close_in)  if close_in  > 0 else 0.0
        atr_pct_out = (atr_out / close_out) if close_out > 0 else 0.0

        illiquid = False
        if has_volume:
            try:
                v_today = float(price_df.loc[d_in, volume_col])
                v_med   = float(vol_med.loc[d_in])
                if v_med > 0:
                    illiquid = (v_today < illiquid_threshold_pct * v_med)
            except KeyError:
                pass

        c = costs.round_trip_cost(
            direction=row["direction"],
            atr_pct_entry=atr_pct_in,
            atr_pct_exit=atr_pct_out,
            hold_days=int(row.get("hold_days", 0)),
            is_illiquid=illiquid,
        )
        new_costs.append(c)
        is_illiquid_flags.append(illiquid)

    t["realistic_cost"] = new_costs
    t["is_illiquid"]    = is_illiquid_flags
    t["net_return_v22"] = t["net_return"]
    t["net_return"]     = t["gross_return"] - t["realistic_cost"]
    return t


def liquidity_gate(
    decisions: pd.DataFrame,
    volume_col: str = "silver_volume",
    min_volume_pct_of_median: float = 0.5,
    median_window: int = 60,
) -> pd.DataFrame:
    """
    Блокирует signal_long/signal_short в дни с volume < threshold.
    Возвращает копию decisions с обнулёнными сигналами на low-volume.
    """
    if decisions.empty or volume_col not in decisions.columns:
        return decisions

    d = decisions.copy()
    vol_med = d[volume_col].rolling(median_window, min_periods=20).median()
    illiquid = d[volume_col] < min_volume_pct_of_median * vol_med
    if "signal_long" in d.columns:
        d.loc[illiquid & (d["signal_long"] == "BUY"), "signal_long"] = "HOLD"
    if "signal_short" in d.columns:
        d.loc[illiquid & (d["signal_short"] == "SHORT"), "signal_short"] = "HOLD"
    d["illiquid_flag"] = illiquid.astype(int)
    return d


# ===========================================================================
# 8. PERFORMANCE ATTRIBUTION
# ===========================================================================

def performance_attribution(
    trades_base:      pd.DataFrame,    # «голый» сигнал: fixed cost, equal sizing
    trades_with_atr:  pd.DataFrame,    # + ATR trail
    trades_with_kelly: pd.DataFrame,   # + Kelly sizing
    trades_with_realistic_costs: pd.DataFrame,  # + realistic costs
) -> pd.DataFrame:
    """
    Разбивает суммарный return на компоненты:
      • model_signal       = ret(base)
      • execution_improve  = ret(atr) - ret(base)
      • sizing_improve     = ret(kelly) - ret(atr)
      • realism_cost       = ret(realistic) - ret(kelly)  (обычно отрицательно)

    Используется compound equity (single-position) на каждой стадии.
    """
    def _final(trades, col="net_return"):
        if trades is None or trades.empty:
            return 0.0
        eq, _ = equity_compounded_sequential(trades, return_col=col)
        return float(eq[-1] - 1.0)

    r_base   = _final(trades_base)
    r_atr    = _final(trades_with_atr)
    r_kelly  = _final(trades_with_kelly, col=("kelly_net_return"
                if (trades_with_kelly is not None
                    and not trades_with_kelly.empty
                    and "kelly_net_return" in trades_with_kelly.columns)
                else "net_return"))
    r_real   = _final(trades_with_realistic_costs)

    rows = [
        {"component": "model_signal",      "delta": r_base},
        {"component": "execution_atr",     "delta": r_atr   - r_base},
        {"component": "sizing_kelly",      "delta": r_kelly - r_atr},
        {"component": "realism_costs",     "delta": r_real  - r_kelly},
        {"component": "TOTAL_REALISTIC",   "delta": r_real},
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# 9. AUDIT LOG
# ===========================================================================

def write_decision_audit_log(
    decisions: pd.DataFrame,
    features_cols: List[str],
    output_path: Path,
    model_version: str = "v23",
) -> int:
    """
    JSON Lines: одна строка на день, для каждого сигнального дня — снимок фичей.
    Регуляторный/инвесторский аудит-трейл.
    """
    if decisions.empty:
        output_path.write_text("", encoding="utf-8")
        return 0
    n_written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for dt, row in decisions.iterrows():
            sig_long  = row.get("signal_long",  row.get("signal", "HOLD"))
            sig_short = row.get("signal_short", "HOLD")
            if sig_long != "BUY" and sig_short != "SHORT":
                continue
            features = {}
            for c in features_cols:
                if c in row.index:
                    v = row[c]
                    if pd.notna(v):
                        features[c] = float(v) if isinstance(v, (int, float, np.floating)) else str(v)
            rec = {
                "ts":             pd.Timestamp(dt).isoformat(),
                "model_version":  model_version,
                "signal_long":    str(sig_long),
                "signal_short":   str(sig_short),
                "p_up":           float(row.get("p_up", float("nan"))),
                "p_short":        float(row.get("p_short", float("nan"))),
                "split":          str(row.get("split", "")),
                "regime":         str(row.get("regime", "")),
                "close":          float(row.get("silver_close", float("nan"))),
                "features":       features,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    return n_written


# ===========================================================================
# 10. DRIFT DETECTION (KS-test)
# ===========================================================================

def ks_2samp_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Чистая реализация Kolmogorov-Smirnov 2-sample p-value (без scipy)."""
    a = np.sort(a[~np.isnan(a)])
    b = np.sort(b[~np.isnan(b)])
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    data = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, data, side="right") / len(a)
    cdf_b = np.searchsorted(b, data, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    # Серия Marsaglia-Tsang
    p = 0.0
    for j in range(1, 101):
        term = 2 * (-1) ** (j - 1) * math.exp(-2 * (lam * j) ** 2)
        p += term
        if abs(term) < 1e-10:
            break
    return float(max(0.0, min(1.0, p)))


def feature_drift_report(
    features_train: pd.DataFrame,
    features_recent: pd.DataFrame,
    feature_cols: Sequence[str],
    alert_p: float = 0.01,
) -> pd.DataFrame:
    """
    KS-test для каждой фичи: train vs recent. Возвращает таблицу с p-value
    и флагом drift (p < alert_p).
    """
    rows = []
    for c in feature_cols:
        if c not in features_train.columns or c not in features_recent.columns:
            continue
        a = features_train[c].dropna().values
        b = features_recent[c].dropna().values
        p = ks_2samp_pvalue(a, b)
        rows.append({
            "feature": c,
            "n_train": len(a),
            "n_recent": len(b),
            "ks_pvalue": round(p, 6) if not math.isnan(p) else None,
            "drift":     (p < alert_p) if not math.isnan(p) else None,
            "mean_train": round(float(np.mean(a)), 6) if len(a) > 0 else None,
            "mean_recent": round(float(np.mean(b)), 6) if len(b) > 0 else None,
        })
    return pd.DataFrame(rows).sort_values("ks_pvalue", na_position="last")


# ===========================================================================
# 11. AUDIT v22 — пересчёт всех v22 trades в честной математике
# ===========================================================================

def _load_full_data() -> pd.DataFrame:
    """Загружает v22_full_data.csv (silver OHLC + features + split)."""
    p = V22_DIR / "v22_full_data.csv"
    if not p.exists():
        raise FileNotFoundError(f"Нет {p} — сначала запустите v22.")
    df = pd.read_csv(p, parse_dates=[0])
    df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    return df


def _split_dates(df: pd.DataFrame, split: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    d = df[df["split"] == split]
    if d.empty:
        return (pd.NaT, pd.NaT)
    return (d.index.min(), d.index.max())


def audit_v22_outputs(
    n_boot: int = 2000,
    block_len: float = 5.0,
    n_trials_dsr: int = 9,
    use_realistic_costs: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Главная точка входа: пересчитывает все v22 trades в честной математике.
    Сохраняет:
      v23_honest_pnl_summary.csv
      v23_honest_risk_metrics.csv
      v23_bootstrap_ci.csv
      v23_dsr_psr.csv
      v23_realistic_costs_impact.csv
      v23_apples_to_apples_bnh.csv
    """
    print("\n" + "=" * 70)
    print("v23 AUDIT: пересчёт v22 trades в честной математике")
    print("=" * 70)

    full_df = _load_full_data()
    print(f"  Загружено v22_full_data.csv: {len(full_df)} строк, "
          f"{full_df.index.min().date()} → {full_df.index.max().date()}")

    if "silver_atr_14d" not in full_df.columns:
        if {"silver_high", "silver_low", "silver_close"}.issubset(full_df.columns):
            close_prev = full_df["silver_close"].shift(1)
            tr = pd.concat([
                full_df["silver_high"] - full_df["silver_low"],
                (full_df["silver_high"] - close_prev).abs(),
                (full_df["silver_low"]  - close_prev).abs(),
            ], axis=1).max(axis=1)
            full_df["silver_atr_14d"] = tr.ewm(span=14, adjust=False).mean()
        else:
            full_df["silver_atr_14d"] = full_df["silver_close"] * 0.015

    costs_model = RealisticCosts()

    variants = ["v22_base", "v22_atr", "v22_kelly", "v22_mh", "v22_wf", "v22_all"]
    splits   = ["valid", "test", "forward"]

    rows_pnl:    List[dict] = []
    rows_risk:   List[dict] = []
    rows_dsr:    List[dict] = []
    rows_boot:   List[dict] = []
    rows_costs:  List[dict] = []
    rows_bnh:    List[dict] = []

    for variant in variants:
        for split in splits:
            trade_path = V22_DIR / f"{variant}_{split}_trades.csv"
            if not trade_path.exists():
                continue
            trades = pd.read_csv(trade_path)
            if trades.empty:
                continue
            trades["entry_date"] = pd.to_datetime(trades["entry_date"])
            trades["exit_date"]  = pd.to_datetime(trades["exit_date"])

            # --- realistic costs ---
            if use_realistic_costs:
                trades_real = recompute_trades_with_realistic_costs(
                    trades, full_df, costs_model,
                )
                v22_cost_total = float(0.0005 * 2 * len(trades))  # approx old cost
                new_cost_total = float(trades_real["realistic_cost"].sum())
                rows_costs.append({
                    "variant": variant, "split": split, "n_trades": len(trades),
                    "v22_cost_total":   round(v22_cost_total, 4),
                    "v23_cost_total":   round(new_cost_total, 4),
                    "cost_increase":    round(new_cost_total - v22_cost_total, 4),
                    "median_atr_pct":   round(
                        float((trades_real["realistic_cost"]).median()), 5,
                    ),
                })
                trades_used = trades_real
                ret_col = "net_return"
            else:
                trades_used = trades
                ret_col = "kelly_net_return" if (
                    "kelly_net_return" in trades.columns and trades["kelly_net_return"].notna().any()
                ) else "net_return"

            # --- compounded sequential equity ---
            eq_seq, seq_trades = equity_compounded_sequential(trades_used, return_col=ret_col)
            n_kept = len(seq_trades)
            n_drop = len(trades) - n_kept
            if n_kept >= 2:
                trade_days_total = int(
                    (seq_trades["exit_date"].max() - seq_trades["entry_date"].min()).days
                )
            else:
                trade_days_total = 0
            honest_metrics = risk_metrics_honest(
                eq_seq, n_trades=n_kept, trade_days_total=trade_days_total,
            )

            # --- compounded with $1-per-trade (overlap allowed) ---
            eq_unit = equity_compounded_overlapping_unit(trades_used, return_col=ret_col)
            unit_total = float(eq_unit[-1] - 1.0)

            # --- apples-to-apples BnH ---
            bnh_dates = _split_dates(full_df, split)
            true_bnh = true_buy_and_hold(full_df["silver_close"], bnh_dates) \
                if pd.notna(bnh_dates[0]) else float("nan")

            rows_bnh.append({
                "variant": variant, "split": split,
                "true_bnh_compound":   round(true_bnh, 4) if not math.isnan(true_bnh) else None,
                "v22_bnh_in_summary":  None,  # заполним ниже
                "split_start":         bnh_dates[0].date() if pd.notna(bnh_dates[0]) else None,
                "split_end":           bnh_dates[1].date() if pd.notna(bnh_dates[1]) else None,
            })

            # --- bootstrap CI ---
            seq_rets = seq_trades[ret_col].astype(float).values if not seq_trades.empty else np.array([])
            ci = bootstrap_ci_metrics(
                seq_rets, n_boot=n_boot, block_len=block_len,
            ) if len(seq_rets) >= 3 else {}

            # --- DSR / PSR ---
            sharpe_per, skew, kurt = sharpe_stats(seq_rets)
            n_obs = len(seq_rets)
            if not math.isnan(sharpe_per) and n_obs >= 4:
                psr = probabilistic_sharpe_ratio(sharpe_per, n_obs, skew, kurt, 0.0)
                dsr = deflated_sharpe_ratio(
                    sharpe_per, n_obs, n_trials=n_trials_dsr,
                    sharpe_variance=0.5 * (sharpe_per ** 2 + 1e-6),
                    skew=skew, kurt=kurt,
                )
            else:
                psr = float("nan")
                dsr = float("nan")
            rows_dsr.append({
                "variant": variant, "split": split, "n_obs": n_obs,
                "sharpe_per_trade":   round(sharpe_per, 4) if not math.isnan(sharpe_per) else None,
                "skew":  round(skew, 3),
                "kurt":  round(kurt, 3),
                "psr_vs_zero":        round(psr, 4) if not math.isnan(psr) else None,
                "dsr_n_trials":       n_trials_dsr,
                "dsr":                round(dsr, 4) if not math.isnan(dsr) else None,
            })

            # --- сохраняем сводки ---
            rows_pnl.append({
                "variant": variant, "split": split,
                "n_trades_v22":         len(trades),
                "n_trades_sequential":  n_kept,
                "n_dropped_overlapping": n_drop,
                "v22_sum_returns":      round(float(trades_used[ret_col].sum()), 4),
                "honest_total_return":  round(honest_metrics["total_return"], 4),
                "unit_$1_pertrade":     round(unit_total, 4),
                "honest_cagr":          round(honest_metrics["cagr"], 4) if honest_metrics["cagr"] is not None else None,
                "honest_max_dd":        round(honest_metrics["max_drawdown"], 4) if honest_metrics["max_drawdown"] is not None else None,
                "honest_sharpe_ann":    round(honest_metrics["sharpe"], 3) if honest_metrics["sharpe"] is not None else None,
                "honest_calmar":        round(honest_metrics["calmar"], 3) if honest_metrics["calmar"] is not None else None,
                "true_bnh":             round(true_bnh, 4) if not math.isnan(true_bnh) else None,
                "vs_true_bnh":          round(honest_metrics["total_return"] - true_bnh, 4)
                                            if not math.isnan(true_bnh) else None,
            })
            rm = {"variant": variant, "split": split, **honest_metrics}
            rows_risk.append(rm)

            if ci:
                rows_boot.append({
                    "variant": variant, "split": split, "n_obs": len(seq_rets),
                    "tr_lower":    round(ci["total_return"]["lower"], 4),
                    "tr_median":   round(ci["total_return"]["median"], 4),
                    "tr_upper":    round(ci["total_return"]["upper"], 4),
                    "shr_lower":   round(ci["sharpe"]["lower"], 3),
                    "shr_median":  round(ci["sharpe"]["median"], 3),
                    "shr_upper":   round(ci["sharpe"]["upper"], 3),
                    "mdd_lower":   round(ci["max_drawdown"]["lower"], 4),
                    "mdd_median":  round(ci["max_drawdown"]["median"], 4),
                    "mdd_upper":   round(ci["max_drawdown"]["upper"], 4),
                })

    pnl_df    = pd.DataFrame(rows_pnl)
    risk_df   = pd.DataFrame(rows_risk)
    boot_df   = pd.DataFrame(rows_boot)
    dsr_df    = pd.DataFrame(rows_dsr)
    costs_df  = pd.DataFrame(rows_costs)
    bnh_df    = pd.DataFrame(rows_bnh)

    pnl_df.to_csv  (V23_DIR / "v23_honest_pnl_summary.csv",      index=False)
    risk_df.to_csv (V23_DIR / "v23_honest_risk_metrics.csv",     index=False)
    boot_df.to_csv (V23_DIR / "v23_bootstrap_ci.csv",            index=False)
    dsr_df.to_csv  (V23_DIR / "v23_dsr_psr.csv",                 index=False)
    costs_df.to_csv(V23_DIR / "v23_realistic_costs_impact.csv",  index=False)
    bnh_df.to_csv  (V23_DIR / "v23_apples_to_apples_bnh.csv",    index=False)

    print("\n=== Сводка P&L (compounded, single-position) ===")
    print(pnl_df.to_string(index=False))
    print("\n=== Bootstrap 95% CI ===")
    if not boot_df.empty:
        print(boot_df.to_string(index=False))
    print("\n=== Deflated Sharpe Ratio (n_trials=9) ===")
    print(dsr_df.to_string(index=False))
    if not costs_df.empty:
        print("\n=== Влияние реалистичных costs ===")
        print(costs_df.to_string(index=False))

    return {
        "pnl":    pnl_df,
        "risk":   risk_df,
        "boot":   boot_df,
        "dsr":    dsr_df,
        "costs":  costs_df,
        "bnh":    bnh_df,
    }


# ===========================================================================
# 12. CPCV DEMO
# ===========================================================================

def cpcv_demo() -> None:
    """Демонстрация CPCV на v22_full_data."""
    print("\n=== CPCV DEMO ===")
    df = _load_full_data()
    n = len(df)
    end_t = df.index.values
    splits = cpcv_splits(n, end_t, n_groups=6, k_test=2, embargo_frac=0.01)
    print(f"  Получено {len(splits)} CPCV-разбиений на {n} образцах")
    if not splits:
        print("  WARN: данных мало для CPCV")
        return
    rows = []
    for i, s in enumerate(splits[:5]):
        rows.append({
            "fold": i, "test_groups": str(s.test_groups),
            "n_train": len(s.train_idx), "n_test": len(s.test_idx),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"  ...всего {len(splits)} fold'ов сохранено в v23_cpcv_folds.json")
    summary = [{
        "fold": i, "test_groups": list(s.test_groups),
        "n_train": int(len(s.train_idx)), "n_test": int(len(s.test_idx)),
    } for i, s in enumerate(splits)]
    (V23_DIR / "v23_cpcv_folds.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )


# ===========================================================================
# 13. ATTRIBUTION REPORT
# ===========================================================================

def build_attribution_report() -> None:
    """Performance attribution: model / execution / sizing / costs."""
    print("\n=== PERFORMANCE ATTRIBUTION ===")
    rows = []
    full_df = _load_full_data()
    costs_model = RealisticCosts()

    for split in ["valid", "test", "forward"]:
        base_p = V22_DIR / f"v22_base_{split}_trades.csv"
        atr_p  = V22_DIR / f"v22_atr_{split}_trades.csv"
        kel_p  = V22_DIR / f"v22_kelly_{split}_trades.csv"
        if not (base_p.exists() and atr_p.exists() and kel_p.exists()):
            continue
        base = pd.read_csv(base_p)
        atr  = pd.read_csv(atr_p)
        kel  = pd.read_csv(kel_p)

        if "silver_atr_14d" not in full_df.columns and {"silver_high", "silver_low", "silver_close"}.issubset(full_df.columns):
            close_prev = full_df["silver_close"].shift(1)
            tr = pd.concat([
                full_df["silver_high"] - full_df["silver_low"],
                (full_df["silver_high"] - close_prev).abs(),
                (full_df["silver_low"]  - close_prev).abs(),
            ], axis=1).max(axis=1)
            full_df["silver_atr_14d"] = tr.ewm(span=14, adjust=False).mean()

        real = recompute_trades_with_realistic_costs(kel, full_df, costs_model)

        attr = performance_attribution(base, atr, kel, real)
        attr["split"] = split
        rows.append(attr)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(V23_DIR / "v23_performance_attribution.csv", index=False)
        print(out.to_string(index=False))
    else:
        print("  WARN: нет данных для attribution")


# ===========================================================================
# 14. FETCH EXTENDED FEATURES
# ===========================================================================

def fetch_and_save_extended_features() -> None:
    print("\n=== FETCH EXTENDED FEATURES (DXY/VIX/MOVE/GLD) ===")
    full_df = _load_full_data()
    start = full_df.index.min().strftime("%Y-%m-%d")
    end   = (full_df.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"  Период: {start} → {end}")
    ext = fetch_extended_features(start, end)
    if ext.empty:
        print("  Не удалось скачать ни одного тикера")
        return
    ext.to_csv(V23_DIR / "v23_extended_features_raw.csv")
    extended = add_extended_features(full_df, ext)
    new_cols = [c for c in EXTENDED_FEATURE_NAMES if c in extended.columns]
    print(f"  Добавлено фичей: {new_cols}")
    extended[["silver_close", *new_cols]].to_csv(V23_DIR / "v23_extended_features_panel.csv")


# ===========================================================================
# 15. DRIFT DEMO
# ===========================================================================

def drift_demo() -> None:
    """Демонстрация KS drift detection: train vs forward."""
    print("\n=== DRIFT DETECTION DEMO (KS-test, train vs forward) ===")
    df = _load_full_data()
    if "split" not in df.columns:
        print("  Нет колонки split — пропускаем")
        return
    f_cols = [c for c in df.columns if c not in ("split",) and df[c].dtype.kind in "fi"]
    train  = df[df["split"] == "train"]
    fwd    = df[df["split"] == "forward"]
    if train.empty or fwd.empty:
        print("  Нет train/forward — пропускаем")
        return
    report = feature_drift_report(train, fwd, f_cols, alert_p=0.01)
    report.to_csv(V23_DIR / "v23_feature_drift_train_vs_forward.csv", index=False)
    drifted = report[report["drift"] == True]
    print(f"  Всего фичей проверено: {len(report)}")
    print(f"  Фичей с drift (p<0.01): {len(drifted)}")
    if not drifted.empty:
        print(drifted.head(20).to_string(index=False))


# ===========================================================================
# 16. AUDIT LOG GENERATION (на v22 решениях)
# ===========================================================================

def write_audit_log_v22() -> None:
    """Создаёт JSONL audit log из v22_base_decisions.csv."""
    print("\n=== AUDIT LOG (JSONL) ===")
    p = V22_DIR / "v22_base_decisions.csv"
    if not p.exists():
        print(f"  Нет {p}")
        return
    dec = pd.read_csv(p, parse_dates=[0])
    dec = dec.set_index(dec.columns[0])
    dec.index = pd.to_datetime(dec.index)
    fcols = [c for c in dec.columns if c not in ("signal", "signal_long", "signal_short",
             "p_up", "p_short", "split", "regime")]
    out_path = V23_DIR / "v23_decision_audit_log.jsonl"
    n = write_decision_audit_log(dec, fcols, out_path, model_version="v22_base")
    print(f"  Записано {n} решений в {out_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="v23 honest math + statistics")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--block-len", type=float, default=5.0)
    ap.add_argument("--n-trials-dsr", type=int, default=9)
    ap.add_argument("--no-realistic-costs", action="store_true")
    ap.add_argument("--fetch-features", action="store_true",
                    help="Скачать DXY/VIX/MOVE/GLD")
    ap.add_argument("--cpcv-demo", action="store_true")
    ap.add_argument("--drift-demo", action="store_true")
    ap.add_argument("--audit-log", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="Запустить все шаги")
    args = ap.parse_args()

    if args.all or not any([
        args.fetch_features, args.cpcv_demo, args.drift_demo, args.audit_log,
    ]):
        audit_v22_outputs(
            n_boot=args.bootstrap, block_len=args.block_len,
            n_trials_dsr=args.n_trials_dsr,
            use_realistic_costs=(not args.no_realistic_costs),
        )
        build_attribution_report()

    if args.fetch_features or args.all:
        fetch_and_save_extended_features()

    if args.cpcv_demo or args.all:
        cpcv_demo()

    if args.drift_demo or args.all:
        drift_demo()

    if args.audit_log or args.all:
        write_audit_log_v22()


if __name__ == "__main__":
    main()
