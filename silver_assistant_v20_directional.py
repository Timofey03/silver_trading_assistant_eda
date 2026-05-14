"""
Silver Trading Assistant v20 — Directional (LONG + SHORT)

Изменения vs v19:

ДИАГНОЗ: BUY-only стратегия упускает DOWN-движения.
Серебро 2013-2022: ~35% DOWN-баров (15-дневный горизонт).
Без SHORT-сигналов капитал стоит в стороне в медвежьи периоды.

1. DOWN binary классификатор (DOWN vs NOT_DOWN):
   - tb_label_down = (tb_label == "DOWN") — из оригинального triple-barrier
   - Тот же RegimeEnsembleV18, expanding window, time decay
   - Adaptive NOT_DOWN_weight = clip(2.0 × hist_down_rate / recent_down_rate, 1.0, 3.5)
     * Бычий рынок (DOWN_rate↓): weight↑ → консервативнее, меньше ложных шортов
     * Медвежий рынок (DOWN_rate↑): weight↓ → агрессивнее
   - Отдельный поиск политики: down_threshold + short_cooldown
   - Отдельные guardrails: Wilson 95% CI для DOWN-precision

2. Инвертированный trailing stop для SHORT:
   - Отслеживаем минимум (trough) вместо пика
   - Выход: high ≥ trough × (1 + trail_pct_short)
   - trail_pct_short + max_hold_short подбираются на valid по Sharpe

3. State Machine (FLAT → LONG/SHORT → FLAT):
   - Нельзя быть одновременно в LONG и SHORT
   - При конфликте сигналов в один день: LONG имеет приоритет
   - Cooldown применён на этапе генерации сигналов

4. Три состояния: BUY / SHORT / HOLD

Запуск:
  python silver_assistant_v20_directional.py
  streamlit run dashboard_app.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# UTF-8 stdout fix
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    from sklearn.metrics import balanced_accuracy_score
except ImportError:
    raise ImportError("pip install scikit-learn>=1.3.0")

# ---- Импорты из v18 (весь ML-пайплайн) ----
try:
    from silver_assistant_v18_adaptive import (
        RegimeEnsembleV18,
        compute_sample_weights,
        train_expanding_model,
        select_policy_v18,
        purged_cv_v18,
        EXPAND_CUTOFFS, HALFLIFE_YEARS, RECENT_WEIGHT_YEARS,
        HISTORICAL_UP_RATE, HORIZON_V18, EMBARGO_V18,
    )
    print("  v18 функции загружены.")
except ImportError as e:
    raise ImportError(f"v18 не найден: {e}")

# ---- Импорты из v19 ----
try:
    from silver_assistant_v19_trailing import (
        TRAIL_PCT_GRID, MAX_HOLD_GRID,
        TRAIL_PCT_DEFAULT, MAX_HOLD_DEFAULT,
        COST_PER_TRADE,
        trailing_summary,
        _sharpe_from_trades,
    )
    print("  v19 функции загружены.")
except ImportError as e:
    raise ImportError(f"v19 не найден: {e}")

# ---- Импорты из v17 ----
try:
    from silver_assistant_v17_fred import (
        fetch_macro_yfinance, add_macro_features,
        MACRO_FEATURE_NAMES,
    )
    print("  v17 функции загружены.")
except ImportError as e:
    raise ImportError(f"v17 не найден: {e}")

# ---- Импорты из v16 ----
try:
    from silver_assistant_v16_binary import (
        binarize_labels, label_report_binary,
        select_top_features, evaluate_split_v16,
        compute_guardrails_binary,
        apply_policy_v16,
        _get_regimes, split_name,
        TOP_FEATURES_N, NOT_UP_WEIGHT,
        MAX_DEPTH, MAX_LEAF_NODES, LEARNING_RATE,
        MIN_SAMP_LEAF, L2_REG,
    )
    print("  v16 функции загружены.")
except ImportError as e:
    raise ImportError(f"v16 не найден: {e}")

# ---- Импорты из v15 ----
try:
    from silver_assistant_v15_regime_cot import (
        fetch_cot_silver, merge_cot_to_daily,
        add_vol_trend_regime,
        COT_FEATURES, COT_RELEASE_LAG, MIN_REGIME_SAMPLES,
    )
    print("  v15 функции загружены.")
except ImportError as e:
    raise ImportError(f"v15 не найден: {e}")

# ---- Импорты из v14 ----
try:
    from silver_assistant_v14_main import (
        fetch_ohlc, build_features,
        add_triple_barrier_labels, get_feature_cols,
        buy_and_hold_return, backtest_strategy,
        purged_walk_forward_splits,
        wilson_ci, pct,
    )
    print("  v14 функции загружены.")
except ImportError as e:
    raise ImportError(f"v14 не найден: {e}")


# ---------------------------------------------------------------------------
# Константы v20
# ---------------------------------------------------------------------------

HORIZON_V20         = HORIZON_V18        # 15 дней
EMBARGO_V20         = EMBARGO_V18        # 15 дней
HISTORICAL_DOWN_RATE = 0.35              # базовый DOWN_rate 2013-2022
NOT_DOWN_WEIGHT_BASE = NOT_UP_WEIGHT     # 2.0

# Trailing stop для SHORT (тот же диапазон, подбирается отдельно)
TRAIL_PCT_GRID_SHORT = TRAIL_PCT_GRID
MAX_HOLD_GRID_SHORT  = MAX_HOLD_GRID


# ---------------------------------------------------------------------------
# 1. DOWN-метка и дистрибуция
# ---------------------------------------------------------------------------

def add_down_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет tb_label_down = 1 если triple-barrier выбил нижний барьер
    за HORIZON дней (tb_label == "DOWN"), иначе 0.
    NaN там, где оригинальный tb_label == NaN (последние HORIZON строк).
    """
    if "tb_label" not in df.columns:
        raise ValueError("tb_label не найден — запустите add_triple_barrier_labels()")
    df["tb_label_down"] = np.where(
        df["tb_label"].isna(), np.nan,
        (df["tb_label"] == "DOWN").astype(float),
    )
    return df


def label_report_directional(df: pd.DataFrame) -> pd.DataFrame:
    """Распределение UP / NEUTRAL / DOWN по выборкам."""
    rows = []
    for sp in ["train", "valid", "test", "forward"]:
        d = df[df["split"] == sp]
        labeled = d[d["tb_label"].notna()]
        n = len(labeled)
        if n == 0:
            continue
        up      = int((labeled["tb_label"] == "UP").sum())
        down    = int((labeled["tb_label"] == "DOWN").sum())
        neutral = int((labeled["tb_label"] == "NEUTRAL").sum())
        rows.append({
            "split":       sp,
            "n":           n,
            "UP":          up,
            "NEUTRAL":     neutral,
            "DOWN":        down,
            "UP_rate":     pct(up / n),
            "DOWN_rate":   pct(down / n),
            "NEUTRAL_rate": pct(neutral / n),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Адаптивный вес и обучение DOWN-модели
# ---------------------------------------------------------------------------

def compute_adaptive_weight_down(
    train_df: pd.DataFrame,
    recent_years: int   = RECENT_WEIGHT_YEARS,
    base_weight: float  = NOT_DOWN_WEIGHT_BASE,
    hist_down_rate: float = HISTORICAL_DOWN_RATE,
) -> float:
    """
    Адаптирует NOT_DOWN_weight к текущему рыночному режиму.

    Бычий рынок (DOWN_rate↓): weight↑ → консервативнее (меньше ложных шортов).
    Медвежий рынок (DOWN_rate↑): weight↓ → агрессивнее (больше шортов).

    Формула: weight = clip(base_weight × hist_down_rate / recent_down_rate, 1.0, 3.5)
    """
    cutoff = train_df.index.max() - pd.Timedelta(days=recent_years * 365)
    recent = train_df[train_df.index >= cutoff]
    if len(recent) < 50 or "tb_label_down" not in recent.columns:
        recent_down_rate = hist_down_rate
    else:
        valid_down = recent["tb_label_down"].dropna()
        recent_down_rate = float(valid_down.mean()) if len(valid_down) >= 10 else hist_down_rate

    weight = float(np.clip(
        base_weight * hist_down_rate / max(recent_down_rate, 0.15),
        1.0, 3.5,
    ))
    return round(weight, 2)


def train_expanding_model_down(
    df: pd.DataFrame,
    cutoff_date: str,
    feature_cols: List[str],
    embargo_days: int = EMBARGO_V20,
) -> Tuple["RegimeEnsembleV18", float, np.ndarray]:
    """
    Обучает DOWN-классификатор (RegimeEnsembleV18) на tb_label_down.
    Возвращает (model_down, adaptive_weight_down, sample_weights).
    """
    cutoff   = pd.Timestamp(cutoff_date) - pd.Timedelta(days=embargo_days)
    train_df = df[(df.index <= cutoff) & df["tb_label_down"].notna()].copy()

    if len(train_df) < 200:
        raise RuntimeError(f"Недостаточно данных для DOWN-модели, cutoff={cutoff.date()}")

    X       = train_df[feature_cols]
    y       = train_df["tb_label_down"].values.astype(int)
    regimes = _get_regimes(train_df)

    adaptive_wt = compute_adaptive_weight_down(train_df)
    sw          = compute_sample_weights(train_df)

    print(f"\n  DOWN-модель: cutoff={cutoff.date()}, n={len(X)}, "
          f"DOWN_rate={y.mean():.3f}, NOT_DOWN_w={adaptive_wt:.2f}")

    model = RegimeEnsembleV18(not_up_weight=adaptive_wt)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X, y, regimes, sample_weight=sw)

    print(f"    Классы: {model.classes_}, режимы: {list(model.models.keys())}")
    return model, adaptive_wt, sw


# ---------------------------------------------------------------------------
# 3. SHORT-политика и сигналы
# ---------------------------------------------------------------------------

def apply_policy_short(
    df: pd.DataFrame,
    model_down: "RegimeEnsembleV18",
    feature_cols: List[str],
    down_threshold: float,
    short_cooldown: int,
) -> pd.DataFrame:
    """
    Генерирует SHORT-сигналы на основе DOWN-модели.
    Добавляет к DataFrame колонки: signal_short, p_short, reason_short.
    """
    out     = df.copy()
    X       = out[feature_cols]
    regimes = _get_regimes(out)
    proba   = model_down.predict_proba(X, regimes)

    # P(DOWN=1)
    down_idx = list(model_down.classes_).index(1)
    p_short  = proba[:, down_idx]

    out["p_short"]      = p_short
    out["signal_short"] = "HOLD"
    out["reason_short"] = "hold"

    last_pos = -short_cooldown
    for i in range(len(out)):
        if p_short[i] >= down_threshold and i - last_pos >= short_cooldown:
            out.iloc[i, out.columns.get_loc("signal_short")] = "SHORT"
            out.iloc[i, out.columns.get_loc("reason_short")] = (
                f"p_short={p_short[i]:.3f}>={down_threshold}"
            )
            last_pos = i

    return out


def select_policy_short(
    valid_df: pd.DataFrame,
    model_down: "RegimeEnsembleV18",
    feature_cols: List[str],
) -> dict:
    """
    Подбирает down_threshold и short_cooldown на valid-наборе.
    Критерий: (Wilson CI low - base_down_rate) + бонус за кол-во сигналов.
    """
    X       = valid_df[feature_cols]
    regimes = _get_regimes(valid_df)
    proba   = model_down.predict_proba(X, regimes)
    down_idx = list(model_down.classes_).index(1)
    p_short  = proba[:, down_idx]

    labeled        = valid_df[valid_df["tb_label_down"].notna()]
    base_down_rate = float(labeled["tb_label_down"].mean()) if len(labeled) > 0 else HISTORICAL_DOWN_RATE
    tb_down_arr    = valid_df["tb_label_down"].values

    best_obj    = -float("inf")
    best_params: dict = {}

    for thr in [0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]:
        for cooldown in [7, 10, 15, 20]:
            sigs     = []
            last_pos = -cooldown
            for i, p in enumerate(p_short):
                if p >= thr and i - last_pos >= cooldown:
                    sigs.append(i)
                    last_pos = i

            n = len(sigs)
            if n < 3:
                continue

            correct = sum(
                1 for i in sigs
                if not np.isnan(tb_down_arr[i]) and tb_down_arr[i] == 1
            )
            prec  = correct / n
            lo, _ = wilson_ci(correct, n)
            lift  = prec - base_down_rate
            obj   = (lo - base_down_rate) + 0.005 * min(n, 20) if lift > 0 else -999

            if obj > best_obj:
                best_obj    = obj
                best_params = {"down_threshold": thr, "short_cooldown": cooldown}

    if not best_params:
        print("  WARN: DOWN-edge не найден на valid — используем fallback")
        best_params = {"down_threshold": 0.52, "short_cooldown": 15}

    return best_params


def compute_guardrails_down(df: pd.DataFrame, split: str) -> dict:
    """
    Guardrails для SHORT-сигналов:
    Wilson 95% CI для DOWN-precision должна превышать базовый DOWN-rate.
    """
    d      = df[df["split"] == split]
    shorts = d[d["signal_short"] == "SHORT"]
    labeled = d[d["tb_label_down"].notna()]

    base_down_rate = float(labeled["tb_label_down"].mean()) if len(labeled) > 0 else HISTORICAL_DOWN_RATE

    if len(shorts) == 0:
        return {
            "split": split, "n_signals": 0,
            "precision": None, "wilson_95_low": None,
            "base_down_rate": round(base_down_rate, 6), "warning": "no_signal",
        }

    short_labeled = shorts[shorts["tb_label_down"].notna()]
    n_labeled     = len(short_labeled)
    correct       = int((short_labeled["tb_label_down"] == 1).sum())

    if n_labeled == 0:
        return {
            "split": split, "n_signals": len(shorts),
            "precision": None, "wilson_95_low": None,
            "base_down_rate": round(base_down_rate, 6), "warning": "no_labeled_signals",
        }

    precision = correct / n_labeled
    lo, hi    = wilson_ci(correct, n_labeled)
    lift      = precision - base_down_rate
    warning   = ("OK" if lo > base_down_rate else
                 ("negative_lift" if lift < 0 else "ci_lower_not_above_base"))

    return {
        "split":          split,
        "n_signals":      len(shorts),
        "correct_over_n": f"{correct}/{n_labeled}",
        "precision":      round(precision, 6),
        "wilson_95_low":  round(lo, 6),
        "base_down_rate": round(base_down_rate, 6),
        "lift_vs_base":   round(lift, 6),
        "warning":        warning,
    }


# ---------------------------------------------------------------------------
# 4. SHORT trailing stop — параметры и бэктест
# ---------------------------------------------------------------------------

def backtest_strategy_short_only(
    df: pd.DataFrame,
    split: str,
    trail_pct: float  = TRAIL_PCT_DEFAULT,
    max_hold: int     = MAX_HOLD_DEFAULT,
    cost: float       = COST_PER_TRADE,
) -> pd.DataFrame:
    """
    SHORT-only бэктест для подбора параметров.
    Используется ТОЛЬКО в select_trailing_params_short().
    Не учитывает state machine (позиции могут «перекрываться»).

    Trailing stop для SHORT: выход когда high ≥ trough × (1 + trail_pct).
    P&L = entry_price / exit_price - 1  (позитивный при падении цены).
    """
    d = df[df["split"] == split].sort_index()
    if d.empty:
        return pd.DataFrame()

    has_high = "silver_high" in d.columns
    has_low  = "silver_low"  in d.columns
    short_dates = d[d["signal_short"] == "SHORT"].index.tolist()

    if not short_dates:
        return pd.DataFrame()

    trades = []
    for entry_date in short_dates:
        entry_pos   = d.index.get_loc(entry_date)
        entry_price = float(d.loc[entry_date, "silver_close"])
        trough      = entry_price
        trail_stop  = entry_price * (1.0 + trail_pct)   # stop выше входа
        exit_price  = entry_price
        exit_date   = entry_date
        exit_reason = "max_hold"

        for j in range(1, max_hold + 1):
            pos = entry_pos + j
            if pos >= len(d):
                break

            row = d.iloc[pos]
            hi  = float(row["silver_high"]) if has_high else float(row["silver_close"])
            lo  = float(row["silver_low"])  if has_low  else float(row["silver_close"])
            cl  = float(row["silver_close"])

            # Обновляем тро‌ф и трейлер
            if lo < trough:
                trough     = lo
                trail_stop = trough * (1.0 + trail_pct)

            # Срабатывание трейлера: цена поднялась выше стопа
            if hi >= trail_stop:
                exit_price  = max(cl, trail_stop)   # консервативно
                exit_date   = d.index[pos]
                exit_reason = "trail_stop"
                break

            exit_price = cl
            exit_date  = d.index[pos]

        gross_ret = entry_price / exit_price - 1.0   # позитивен при падении
        net_ret   = gross_ret - cost
        hold_days = d.index.get_loc(exit_date) - entry_pos

        trades.append({
            "signal_date":   entry_date,
            "entry_date":    entry_date,
            "exit_date":     exit_date,
            "entry_price":   round(entry_price, 3),
            "exit_price":    round(exit_price,  3),
            "trough_price":  round(trough,      3),
            "trail_stop":    round(trail_stop,  3),
            "gross_return":  round(gross_ret,   6),
            "net_return":    round(net_ret,      6),
            "hold_days":     hold_days,
            "exit_reason":   exit_reason,
            "tb_label_down": d.loc[entry_date, "tb_label_down"]
                             if "tb_label_down" in d.columns else None,
        })

    return pd.DataFrame(trades)


def select_trailing_params_short(
    valid_df: pd.DataFrame,
    trail_pct_grid: list = TRAIL_PCT_GRID_SHORT,
    max_hold_grid:  list = MAX_HOLD_GRID_SHORT,
    cost: float          = COST_PER_TRADE,
) -> Tuple[float, int]:
    """
    Подбирает trail_pct_short + max_hold_short на valid по Sharpe.
    """
    best_sharpe = float("-inf")
    best_trail  = TRAIL_PCT_DEFAULT
    best_hold   = MAX_HOLD_DEFAULT

    print("\n  Поиск trailing-параметров для SHORT на valid:")
    print(f"  {'trail_pct':>10}  {'max_hold':>9}  {'n_trades':>9}  {'sharpe':>8}")

    for trail_pct in trail_pct_grid:
        for max_hold in max_hold_grid:
            trades = backtest_strategy_short_only(
                valid_df, "valid", trail_pct=trail_pct, max_hold=max_hold, cost=cost
            )
            sharpe = _sharpe_from_trades(trades)
            n      = len(trades) if not trades.empty else 0
            mark   = " ← best" if sharpe > best_sharpe and n >= 3 else ""
            print(f"  {trail_pct:>10.2%}  {max_hold:>9}  {n:>9}  {sharpe:>8.4f}{mark}")
            if sharpe > best_sharpe and n >= 3:
                best_sharpe = sharpe
                best_trail  = trail_pct
                best_hold   = max_hold

    print(f"\n  >>> SHORT параметры: trail_pct={best_trail:.2%}, "
          f"max_hold={best_hold}, sharpe={best_sharpe:.4f}")
    return best_trail, best_hold


# ---------------------------------------------------------------------------
# 5. Directional бэктест (LONG + SHORT, state machine)
# ---------------------------------------------------------------------------

def backtest_strategy_directional(
    df: pd.DataFrame,
    split: str,
    trail_pct_long:  float = TRAIL_PCT_DEFAULT,
    max_hold_long:   int   = MAX_HOLD_DEFAULT,
    trail_pct_short: float = TRAIL_PCT_DEFAULT,
    max_hold_short:  int   = MAX_HOLD_DEFAULT,
    cost: float            = COST_PER_TRADE,
) -> pd.DataFrame:
    """
    Комбинированный LONG+SHORT бэктест с state machine.

    Правила:
    - Нельзя одновременно держать LONG и SHORT.
    - При конфликте (BUY и SHORT в один день): LONG приоритет.
    - Вход на Close дня сигнала, выход на Close дня триггера.
    - P&L LONG:  exit / entry - 1
    - P&L SHORT: entry / exit - 1  (прибыль при падении)
    """
    d = df[df["split"] == split].sort_index()
    if d.empty:
        return pd.DataFrame()

    has_high = "silver_high" in d.columns
    has_low  = "silver_low"  in d.columns

    # Собираем все сигналы с приоритетом LONG при конфликте
    signal_dates: List[Tuple] = []   # (date, direction)
    for date, row in d.iterrows():
        sl = row.get("signal_long",  "HOLD")
        ss = row.get("signal_short", "HOLD")
        if sl == "BUY":
            signal_dates.append((date, "LONG"))
        elif ss == "SHORT":
            signal_dates.append((date, "SHORT"))

    trades: List[dict] = []
    in_position_until_idx = -1   # индекс последнего дня текущей позиции

    for sig_date, direction in signal_dates:
        sig_idx = d.index.get_loc(sig_date)

        # Пропускаем если ещё в позиции
        if sig_idx <= in_position_until_idx:
            continue

        entry_price = float(d.loc[sig_date, "silver_close"])
        exit_price  = entry_price
        exit_idx    = sig_idx
        exit_reason = "max_hold"

        if direction == "LONG":
            peak       = entry_price
            trail_stop = entry_price * (1.0 - trail_pct_long)
            max_hold   = max_hold_long

            for j in range(1, max_hold + 1):
                pos = sig_idx + j
                if pos >= len(d):
                    break
                hi = float(d.iloc[pos]["silver_high"]) if has_high else float(d.iloc[pos]["silver_close"])
                lo = float(d.iloc[pos]["silver_low"])  if has_low  else float(d.iloc[pos]["silver_close"])
                cl = float(d.iloc[pos]["silver_close"])

                if hi > peak:
                    peak       = hi
                    trail_stop = peak * (1.0 - trail_pct_long)

                if lo <= trail_stop:
                    exit_price  = min(cl, trail_stop)
                    exit_idx    = pos
                    exit_reason = "trail_stop"
                    break
                exit_price = cl
                exit_idx   = pos

            gross_ret = exit_price / entry_price - 1.0
            extra = {
                "peak_price":  round(peak, 3),
                "trough_price": None,
            }

        else:  # SHORT
            trough     = entry_price
            trail_stop = entry_price * (1.0 + trail_pct_short)
            max_hold   = max_hold_short

            for j in range(1, max_hold + 1):
                pos = sig_idx + j
                if pos >= len(d):
                    break
                hi = float(d.iloc[pos]["silver_high"]) if has_high else float(d.iloc[pos]["silver_close"])
                lo = float(d.iloc[pos]["silver_low"])  if has_low  else float(d.iloc[pos]["silver_close"])
                cl = float(d.iloc[pos]["silver_close"])

                if lo < trough:
                    trough     = lo
                    trail_stop = trough * (1.0 + trail_pct_short)

                if hi >= trail_stop:
                    exit_price  = max(cl, trail_stop)
                    exit_idx    = pos
                    exit_reason = "trail_stop"
                    break
                exit_price = cl
                exit_idx   = pos

            gross_ret = entry_price / exit_price - 1.0    # положителен при падении
            extra = {
                "peak_price":   None,
                "trough_price": round(trough, 3),
            }

        net_ret   = gross_ret - cost
        hold_days = exit_idx - sig_idx
        in_position_until_idx = exit_idx

        trade = {
            "direction":    direction,
            "signal_date":  sig_date,
            "entry_date":   sig_date,
            "exit_date":    d.index[exit_idx],
            "entry_price":  round(entry_price, 3),
            "exit_price":   round(exit_price,  3),
            "trail_stop":   round(trail_stop,  3),
            "gross_return": round(gross_ret,   6),
            "net_return":   round(net_ret,     6),
            "hold_days":    hold_days,
            "exit_reason":  exit_reason,
            "tb_label_bin":  d.loc[sig_date, "tb_label_bin"]  if "tb_label_bin"  in d.columns else None,
            "tb_label_down": d.loc[sig_date, "tb_label_down"] if "tb_label_down" in d.columns else None,
        }
        trade.update(extra)
        trades.append(trade)

    return pd.DataFrame(trades)


def directional_summary(trades: pd.DataFrame, split: str, bnh: float) -> dict:
    """Расширенный отчёт: LONG + SHORT статистика раздельно и совместно."""
    base = {
        "split": split, "n_trades": 0,
        "n_long": 0, "n_short": 0,
        "sum_net_return": 0.0,
        "long_net": 0.0, "short_net": 0.0,
        "win_rate": None, "win_rate_long": None, "win_rate_short": None,
        "profit_factor": None,
        "buy_and_hold": round(bnh, 4), "vs_bnh": None,
        "avg_hold_days": None,
    }
    if trades.empty:
        return base

    rets  = trades["net_return"].values
    longs = trades[trades["direction"] == "LONG"]
    shorts = trades[trades["direction"] == "SHORT"]

    wins   = rets[rets > 0]
    losses = rets[rets < 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

    long_net  = float(longs["net_return"].sum())  if not longs.empty  else 0.0
    short_net = float(shorts["net_return"].sum()) if not shorts.empty else 0.0

    return {
        "split":           split,
        "n_trades":        len(trades),
        "n_long":          len(longs),
        "n_short":         len(shorts),
        "sum_net_return":  round(float(rets.sum()), 4),
        "long_net":        round(long_net,  4),
        "short_net":       round(short_net, 4),
        "win_rate":        round(float((rets > 0).mean()), 4),
        "win_rate_long":   round(float((longs["net_return"] > 0).mean()), 4)  if not longs.empty  else None,
        "win_rate_short":  round(float((shorts["net_return"] > 0).mean()), 4) if not shorts.empty else None,
        "profit_factor":   round(pf, 3) if np.isfinite(pf) else None,
        "buy_and_hold":    round(bnh, 4),
        "vs_bnh":          round(float(rets.sum()) - bnh, 4),
        "avg_hold_days":   round(float(trades["hold_days"].mean()), 1),
        "trail_exits":     int((trades["exit_reason"] == "trail_stop").sum()),
        "max_hold_exits":  int((trades["exit_reason"] == "max_hold").sum()),
    }


# ---------------------------------------------------------------------------
# 6. Purged CV для DOWN-классификатора
# ---------------------------------------------------------------------------

def purged_cv_down(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_train_years: int  = 3,
    n_test_months: int  = 6,
) -> pd.DataFrame:
    """Purged walk-forward CV для DOWN-классификатора."""
    labeled   = df[df["tb_label_down"].notna()].copy()
    wf_splits = purged_walk_forward_splits(
        labeled.index,
        n_train_years=n_train_years, n_test_months=n_test_months,
        embargo_days=EMBARGO_V20, horizon=HORIZON_V20,
    )
    rows = []
    for i, (tr_idx, te_idx) in enumerate(wf_splits):
        train_fold = labeled.loc[tr_idx]
        Xtr  = train_fold[feature_cols]
        ytr  = train_fold["tb_label_down"].values.astype(int)
        Xte  = labeled.loc[te_idx, feature_cols]
        yte  = labeled.loc[te_idx, "tb_label_down"].values.astype(int)
        rgtr = _get_regimes(train_fold)
        rgte = _get_regimes(labeled.loc[te_idx])

        if len(Xtr) < MIN_REGIME_SAMPLES or Xte.empty:
            continue
        try:
            sw  = compute_sample_weights(train_fold)
            m   = RegimeEnsembleV18()
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xtr, ytr, rgtr, sample_weight=sw)
                pred = m.predict(Xte, rgte)
            ba = balanced_accuracy_score(yte, pred)
            rows.append({
                "fold":         i,
                "train_end":    tr_idx[-1].date(),
                "test_start":   te_idx[0].date(),
                "n_train":      len(Xtr),
                "n_test":       len(Xte),
                "balanced_acc": ba,
            })
        except Exception as e:
            print(f"  fold {i} пропущен: {type(e).__name__}: {e}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Основной pipeline v20
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",   default="2013-01-01")
    ap.add_argument("--end",     default="2099-12-31")
    ap.add_argument("--out-dir", default="baseline_outputs_v20")
    args = ap.parse_args(argv)

    end = min(args.end, pd.Timestamp.today().strftime("%Y-%m-%d"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== v20: directional (LONG + SHORT) ===")
    print(f"  ML: depth={MAX_DEPTH}, leaf={MAX_LEAF_NODES}, lr={LEARNING_RATE}, "
          f"l2={L2_REG}, halflife={HALFLIFE_YEARS}y, top_feat={TOP_FEATURES_N}")

    # ---- OHLC ----
    print("\n=== v20: загрузка OHLC ===")
    try:
        df = fetch_ohlc(args.start, end)
        print(f"  OHLC: {len(df)} строк, {df.index[0].date()} — {df.index[-1].date()}")
    except Exception as e:
        print(f"  WARN: fetch_ohlc ошибка ({e}) — пробуем кэш v19/v18...")
        df = None
        for cache_path in [
            Path("baseline_outputs_v19/v19_full_data.csv"),
            Path("baseline_outputs_v18/v18_full_data.csv"),
        ]:
            if cache_path.exists():
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                print(f"  Кэш загружен: {cache_path} ({len(df)} строк)")
                break
        if df is None:
            raise RuntimeError("OHLC недоступен и кэш не найден. Запустите v19 сначала.")

        # Убедимся что индекс — DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Фильтруем по диапазону дат
        df = df[(df.index >= args.start) & (df.index <= end)]
        print(f"  OHLC из кэша: {len(df)} строк, {df.index[0].date()} — {df.index[-1].date()}")

        # Если кэш уже содержит все фичи/метки — пропускаем feature engineering
        _cache_has_features = "tb_label" in df.columns
        if _cache_has_features:
            print("  Кэш содержит признаки и метки — пропускаем build_features / labels")
    else:
        _cache_has_features = False

    # ---- Макро ----
    print("\n=== v20: загрузка макро-данных ===")
    macro = fetch_macro_yfinance(args.start, end)
    if not macro.empty:
        macro.to_csv(out / "v20_macro_raw.csv")

    # ---- COT ----
    print("\n=== v20: загрузка COT ===")
    cot = fetch_cot_silver(pd.Timestamp(args.start).year, pd.Timestamp(end).year)
    if not cot.empty:
        print(f"  COT: {len(cot)} записей")
        cot.to_csv(out / "v20_cot_raw.csv")
    else:
        print("  COT: недоступны")

    # ---- Признаки ----
    if _cache_has_features:
        # Кэш из v19/v18 уже содержит всё — только добавляем DOWN-метку если её нет
        print("\n=== v20: кэш — пропускаем feature engineering ===")
        present_cot = [c for c in COT_FEATURES if c in df.columns]
        if present_cot:
            print(f"  COT признаки из кэша: {present_cot}")
        macro_features = [c for c in MACRO_FEATURE_NAMES if c in df.columns]
    else:
        print("\n=== v20: инженерия признаков ===")
        df = build_features(df, pd.DataFrame())
        df = add_vol_trend_regime(df)
        df = add_macro_features(df, macro)

        if not cot.empty:
            df = merge_cot_to_daily(df, cot, lag_days=COT_RELEASE_LAG)
        present_cot = [c for c in COT_FEATURES if c in df.columns]
        if present_cot:
            print(f"  COT признаки: {present_cot}")
        macro_features = [c for c in MACRO_FEATURE_NAMES if c in df.columns]

    # ---- Labels: UP + DOWN ----
    if _cache_has_features:
        print("\n=== v20: метки из кэша ===")
        if "tb_label_bin" not in df.columns:
            df = binarize_labels(df)
        if "tb_label_down" not in df.columns:
            df = add_down_label(df)
        if "split" not in df.columns:
            df["split"] = df.index.map(split_name)
    else:
        print("\n=== v20: triple-barrier + UP + DOWN метки ===")
        df = add_triple_barrier_labels(df, horizon=HORIZON_V20)
        df = binarize_labels(df)          # tb_label_bin (UP)
        df = add_down_label(df)           # tb_label_down (DOWN)
        df["split"] = df.index.map(split_name)
    df.to_csv(out / "v20_full_data.csv")

    lb3 = label_report_directional(df)
    lb3.to_csv(out / "v20_label_distribution_3class.csv", index=False)
    print("  3-классовое распределение UP / NEUTRAL / DOWN:")
    print(lb3.to_string(index=False))

    # ---- Список признаков ----
    base_features = get_feature_cols(df)
    # macro_features уже определён в ветке выше
    extra_cot     = [c for c in present_cot if c not in base_features]
    all_features: List[str] = list(dict.fromkeys(base_features + macro_features + extra_cot))
    print(f"\n  Всего признаков: {len(all_features)}")

    # ================================================================
    # PASS 1: Отбор признаков (на valid-окне, UP-модель)
    # ================================================================
    print("\n=== v20: pass 1 — отбор признаков (UP-модель, valid-окно) ===")
    valid_cutoff = pd.Timestamp(EXPAND_CUTOFFS["valid"]) - pd.Timedelta(days=EMBARGO_V20)
    p1_train     = df[(df.index <= valid_cutoff) & df["tb_label_bin"].notna()].copy()
    p1_valid     = df[(df["split"] == "valid")   & df["tb_label_bin"].notna()].copy()

    X_p1_tr  = p1_train[all_features]
    y_p1_tr  = p1_train["tb_label_bin"].values.astype(int)
    r_p1_tr  = _get_regimes(p1_train)
    X_p1_val = p1_valid[all_features]
    y_p1_val = p1_valid["tb_label_bin"].values.astype(int)
    sw_p1    = compute_sample_weights(p1_train)

    model_p1 = RegimeEnsembleV18()
    with contextlib.redirect_stdout(io.StringIO()):
        model_p1.fit(X_p1_tr, y_p1_tr, r_p1_tr, sample_weight=sw_p1)

    selected, imp_series = select_top_features(model_p1, X_p1_val, y_p1_val, n_top=TOP_FEATURES_N)

    imp_df = imp_series.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df["source"] = imp_df["feature"].apply(
        lambda f: "macro_yf" if f in MACRO_FEATURE_NAMES
                  else ("cot" if f in COT_FEATURES else "base")
    )
    imp_df.to_csv(out / "v20_feature_importance.csv", index=False)
    pd.DataFrame({"feature": selected, "rank": range(1, len(selected)+1)}).to_csv(
        out / "v20_selected_features.csv", index=False
    )
    macro_in_top = [f for f in selected if f in MACRO_FEATURE_NAMES]
    print(f"  Отобрано: {len(selected)} из {len(all_features)}, "
          f"macro в top: {len(macro_in_top)} → {macro_in_top}")

    # ================================================================
    # PASS 2a: UP-модели (expanding window, как в v18/v19)
    # ================================================================
    print("\n=== v20: pass 2a — UP-модели (expanding window) ===")
    split_models_up:  Dict[str, RegimeEnsembleV18] = {}
    split_weights_up: Dict[str, float]              = {}

    for split_key, cutoff in EXPAND_CUTOFFS.items():
        print(f"\n  [UP / {split_key.upper()}] — expanding до {cutoff}")
        model, wt, _ = train_expanding_model(df, cutoff, selected)
        split_models_up[split_key]  = model
        split_weights_up[split_key] = wt

    # ================================================================
    # PASS 2b: DOWN-модели (expanding window)
    # ================================================================
    print("\n=== v20: pass 2b — DOWN-модели (expanding window) ===")
    split_models_down:  Dict[str, RegimeEnsembleV18] = {}
    split_weights_down: Dict[str, float]              = {}

    for split_key, cutoff in EXPAND_CUTOFFS.items():
        print(f"\n  [DOWN / {split_key.upper()}] — expanding до {cutoff}")
        model_d, wt_d, _ = train_expanding_model_down(df, cutoff, selected)
        split_models_down[split_key]  = model_d
        split_weights_down[split_key] = wt_d

    # ================================================================
    # Выбор политик
    # ================================================================
    print("\n=== v20: выбор UP-политики (valid-модель) ===")
    valid_full    = df[df["split"] == "valid"].copy()
    policy_up     = select_policy_v18(valid_full, split_models_up["valid"], selected)
    print(f"  UP параметры: {policy_up}")

    print("\n=== v20: выбор SHORT-политики (valid DOWN-модель) ===")
    policy_short  = select_policy_short(valid_full, split_models_down["valid"], selected)
    print(f"  SHORT параметры: {policy_short}")

    # ================================================================
    # Применение политик: signal_long + signal_short по каждому split
    # ================================================================
    print("\n=== v20: применение политик ===")
    signal_parts = []
    for split_key in ["train", "valid", "test", "forward"]:
        split_df = df[df["split"] == split_key].copy()
        if split_df.empty:
            continue

        model_up_s   = split_models_up.get(split_key,   split_models_up["valid"])
        model_down_s = split_models_down.get(split_key, split_models_down["valid"])

        # UP-сигналы
        part = apply_policy_v16(
            split_df, model_up_s, selected,
            policy_up["up_threshold"], policy_up["cooldown"],
        )
        # Переименовываем signal → signal_long
        part["signal_long"] = part["signal"]

        # DOWN-сигналы (добавляем к тому же DataFrame)
        part = apply_policy_short(
            part, model_down_s, selected,
            policy_short["down_threshold"], policy_short["short_cooldown"],
        )
        signal_parts.append(part)

    all_df = pd.concat(signal_parts).sort_index()
    all_df.to_csv(out / "v20_decisions_all.csv")

    # ================================================================
    # Метрики классификаторов
    # ================================================================
    print("\n=== v20: метрики UP-классификатора ===")
    cls_rows_up = []
    for split_key in ["train", "valid", "test", "forward"]:
        model_up_s = split_models_up.get(split_key, split_models_up["valid"])
        row = evaluate_split_v16(df, split_key, model_up_s, selected)
        row["direction"] = "UP"
        cls_rows_up.append(row)
    cls_up_df = pd.DataFrame(cls_rows_up)
    cls_up_df.to_csv(out / "v20_classifier_metrics_up.csv", index=False)
    cols = [c for c in ["split", "direction", "n", "balanced_accuracy", "auc", "brier"]
            if c in cls_up_df.columns]
    print(cls_up_df[cols].to_string(index=False))

    # Guardrails LONG
    print("\n=== v20: guardrails LONG ===")
    # Для guardrails используем signal_long как signal
    all_df_long = all_df.copy()
    all_df_long["signal"] = all_df_long["signal_long"]
    grd_long = pd.DataFrame([
        compute_guardrails_binary(all_df_long, s)
        for s in ["valid", "test", "forward"]
    ])
    grd_long.to_csv(out / "v20_guardrails_long.csv", index=False)
    cols = ["split", "n_signals", "correct_over_n", "precision",
            "wilson_95_low", "base_up_rate", "lift_vs_base", "warning"]
    cols = [c for c in cols if c in grd_long.columns]
    print(grd_long[cols].to_string(index=False))

    # Guardrails SHORT
    print("\n=== v20: guardrails SHORT ===")
    grd_short = pd.DataFrame([
        compute_guardrails_down(all_df, s)
        for s in ["valid", "test", "forward"]
    ])
    grd_short.to_csv(out / "v20_guardrails_short.csv", index=False)
    cols = ["split", "n_signals", "correct_over_n", "precision",
            "wilson_95_low", "base_down_rate", "lift_vs_base", "warning"]
    cols = [c for c in cols if c in grd_short.columns]
    print(grd_short[cols].to_string(index=False))

    # ================================================================
    # Подбор trailing параметров
    # ================================================================
    print("\n=== v20: подбор trailing-параметров LONG (valid) ===")
    valid_signal_df = all_df[all_df["split"] == "valid"].copy()

    # LONG trailing (переиспользуем v19 логику через backtest_strategy_trailing-совместимый df)
    # Временно делаем signal = signal_long для v19-функции
    valid_signal_df["signal"] = valid_signal_df["signal_long"]

    from silver_assistant_v19_trailing import select_trailing_params as _select_trail_long
    best_trail_long, best_hold_long = _select_trail_long(valid_signal_df)

    print("\n=== v20: подбор trailing-параметров SHORT (valid) ===")
    best_trail_short, best_hold_short = select_trailing_params_short(valid_signal_df)

    print(f"\n  >>> LONG:  trail_pct={best_trail_long:.2%}, max_hold={best_hold_long}d")
    print(f"  >>> SHORT: trail_pct={best_trail_short:.2%}, max_hold={best_hold_short}d")

    # ================================================================
    # Directional бэктест
    # ================================================================
    print("\n=== v20: directional бэктест (LONG + SHORT) ===")
    dir_bt_rows  = []
    for s in ["valid", "test", "forward"]:
        trades = backtest_strategy_directional(
            all_df, s,
            trail_pct_long=best_trail_long,   max_hold_long=best_hold_long,
            trail_pct_short=best_trail_short, max_hold_short=best_hold_short,
            cost=COST_PER_TRADE,
        )
        trades.to_csv(out / f"{s}_trades_directional_v20.csv", index=False)
        all_df[all_df["split"] == s].to_csv(out / f"{s}_decisions_v20.csv")

        bnh     = buy_and_hold_return(all_df, s)
        summary = directional_summary(trades, s, bnh)
        dir_bt_rows.append(summary)

        n      = summary["n_trades"]
        n_long = summary["n_long"]
        n_sh   = summary["n_short"]
        print(f"\n  [{s.upper()}]  n={n} (long={n_long}, short={n_sh}), "
              f"net_sum={pct(summary['sum_net_return'])}, "
              f"long={pct(summary['long_net'])}, short={pct(summary['short_net'])}, "
              f"win={pct(summary['win_rate'])}, BnH={pct(bnh)}, "
              f"vs_BnH={pct(summary['vs_bnh'])}")

    dir_bt_df = pd.DataFrame(dir_bt_rows)
    dir_bt_df.to_csv(out / "v20_backtest_directional.csv", index=False)

    # ================================================================
    # Adaptive weights summary
    # ================================================================
    print("\n  Адаптивные веса:")
    for sp in EXPAND_CUTOFFS:
        print(f"    {sp:8s}: UP_w={split_weights_up.get(sp,'?'):.2f}, "
              f"DOWN_w={split_weights_down.get(sp,'?'):.2f}")

    # ================================================================
    # Сравнение v20 vs v19
    # ================================================================
    v19_gr_path = Path("baseline_outputs_v19/v19_guardrails.csv")
    if v19_gr_path.exists():
        v19_gr    = pd.read_csv(v19_gr_path)
        comp_rows = []
        for s in ["valid", "test", "forward"]:
            r19  = v19_gr[v19_gr["split"] == s]
            r20l = grd_long[grd_long["split"] == s]
            r20s = grd_short[grd_short["split"] == s]
            comp_rows.append({
                "split":          s,
                "v19_prec_long":  pct(r19["precision"].values[0])  if not r19.empty  else "-",
                "v20_prec_long":  pct(r20l["precision"].values[0]) if not r20l.empty else "-",
                "v20_prec_short": pct(r20s["precision"].values[0]) if not r20s.empty else "-",
                "v20_n_long":     r20l["n_signals"].values[0]       if not r20l.empty else "-",
                "v20_n_short":    r20s["n_signals"].values[0]       if not r20s.empty else "-",
                "v20_short_edge": "✅" if (not r20s.empty and
                                           "OK" == str(r20s["warning"].values[0])) else "❌",
            })
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(out / "v20_vs_v19_comparison.csv", index=False)
        print("\n=== Сравнение v19 vs v20 ===")
        print(comp_df.to_string(index=False))

    # ================================================================
    # Policy JSON
    # ================================================================
    policy_params = {
        "version":           "v20",
        "horizon_days":      HORIZON_V20,
        "top_features_n":    TOP_FEATURES_N,
        # LONG
        "up_threshold":      policy_up["up_threshold"],
        "cooldown":          policy_up["cooldown"],
        "trail_pct_long":    best_trail_long,
        "max_hold_long":     best_hold_long,
        "not_up_weight_adaptive": split_weights_up,
        # SHORT
        "down_threshold":    policy_short["down_threshold"],
        "short_cooldown":    policy_short["short_cooldown"],
        "trail_pct_short":   best_trail_short,
        "max_hold_short":    best_hold_short,
        "not_down_weight_adaptive": split_weights_down,
        # Общее
        "halflife_years":    HALFLIFE_YEARS,
        "macro_features":    macro_features,
        "macro_in_top_n":    macro_in_top,
        "expand_cutoffs":    EXPAND_CUTOFFS,
        "regularization": {
            "max_depth": MAX_DEPTH, "max_leaf_nodes": MAX_LEAF_NODES,
            "learning_rate": LEARNING_RATE, "l2_reg": L2_REG,
            "min_samples_leaf": MIN_SAMP_LEAF,
        },
    }
    with open(out / "v20_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_params, f, indent=2, ensure_ascii=False)

    # ================================================================
    # Последние сигнальные карточки
    # ================================================================
    cards = []
    for s in ["valid", "test", "forward"]:
        d = all_df[all_df["split"] == s].sort_index()
        if d.empty:
            continue
        r = d.iloc[-1]
        cards.append({
            "split":           s,
            "date":            r.name.date(),
            "silver_close":    round(float(r.get("silver_close", float("nan"))), 2),
            "signal_long":     r.get("signal_long",  "HOLD"),
            "signal_short":    r.get("signal_short", "HOLD"),
            "p_up":            round(float(r.get("p_up",    float("nan"))), 4),
            "p_short":         round(float(r.get("p_short", float("nan"))), 4),
            "trend_regime":    r.get("trend_regime", r.get("regime", "")),
            "up_weight":       split_weights_up.get(s, "?"),
            "down_weight":     split_weights_down.get(s, "?"),
            "trail_pct_long":  best_trail_long,
            "trail_pct_short": best_trail_short,
        })
    pd.DataFrame(cards).to_csv(out / "v20_latest_signal_cards.csv", index=False)

    # ================================================================
    # Purged CV — DOWN-классификатор
    # ================================================================
    print("\n=== v20: purged CV DOWN-классификатора (baseline=0.50) ===")
    wf_down = purged_cv_down(df, selected)
    wf_down.to_csv(out / "v20_purged_wf_cv_down.csv", index=False)
    if not wf_down.empty:
        mean_ba = wf_down["balanced_acc"].mean()
        n_above = (wf_down["balanced_acc"] > 0.50).sum()
        print(f"  Фолдов: {len(wf_down)}, mean BA: {mean_ba:.3f}")
        print(f"  Выше 0.50: {n_above}/{len(wf_down)}")
        print(wf_down.to_string(index=False))
    else:
        print("  CV пуст")

    print(f"\n=== v20 завершён. Результаты: {out} ===")
    print(f"  LONG:  up_threshold={policy_up['up_threshold']}, "
          f"trail_pct={best_trail_long:.2%}, max_hold={best_hold_long}d")
    print(f"  SHORT: down_threshold={policy_short['down_threshold']}, "
          f"trail_pct={best_trail_short:.2%}, max_hold={best_hold_short}d")
    print("  Дашборд: streamlit run dashboard_app.py")


if __name__ == "__main__":
    main()
