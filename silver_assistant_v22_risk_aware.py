"""
Silver Trading Assistant v22 — Risk-Aware: пять модулей улучшений

Модули vs v21 (базовая линия):
  A) ATR-based trailing stops  — trail_dist = k × ATR_14d (вместо фиксированного %)
  B) Kelly position sizing     — размер позиции ∝ p_signal (вместо flat $1)
  C) Risk metrics              — drawdown, Sharpe, Calmar, time-underwater, Ulcer
  D) Multi-horizon ensemble    — 5d + 15d + 30d UP-классификаторы (prob averaging)
  E) Walk-forward retraining   — переобучение каждые 60 торговых дней (реалистичнее)

Каждый вариант тестируется изолированно И все вместе для измерения вклада.

Варианты:
  v22_base  = v21-сигналы + fixed trail   (baseline для сравнения)
  v22_atr   = v21-сигналы + ATR trail     (+A)
  v22_kelly = v21-сигналы + ATR trail + Kelly  (+A+B)
  v22_mh    = MultiHorizon-сигналы + ATR + Kelly  (+A+B+D)
  v22_wf    = WalkForward-сигналы + ATR + Kelly   (+A+B+E)
  v22_all   = WF+MH-сигналы + ATR + Kelly         (+A+B+D+E)

Risk metrics (C) вычисляются для ВСЕХ вариантов.

Запуск:
  python silver_assistant_v22_risk_aware.py
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

# ---- Импорты из v21 ----
try:
    from silver_assistant_v21_regime_short import (
        SHORT_BLOCKED_REGIMES,
        SHORT_COOLDOWN_GRID_V21,
        apply_policy_short_v21,
        select_policy_short_v21,
        backtest_strategy_independent,
        independent_summary,
    )
    print("  v21 функции загружены.")
except ImportError as e:
    raise ImportError(f"v21 не найден: {e}")

# ---- Импорты из v20 ----
try:
    from silver_assistant_v20_directional import (
        add_down_label,
        label_report_directional,
        train_expanding_model_down,
        compute_guardrails_down,
        HISTORICAL_DOWN_RATE,
    )
    print("  v20 функции загружены.")
except ImportError as e:
    raise ImportError(f"v20 не найден: {e}")

# ---- Импорты из v19 ----
try:
    from silver_assistant_v19_trailing import (
        TRAIL_PCT_GRID, MAX_HOLD_GRID,
        TRAIL_PCT_DEFAULT, MAX_HOLD_DEFAULT,
        COST_PER_TRADE,
        _sharpe_from_trades,
        select_trailing_params,
    )
    print("  v19 функции загружены.")
except ImportError as e:
    raise ImportError(f"v19 не найден: {e}")

# ---- Импорты из v18 ----
try:
    from silver_assistant_v18_adaptive import (
        RegimeEnsembleV18,
        compute_sample_weights,
        train_expanding_model,
        select_policy_v18,
        EXPAND_CUTOFFS, HALFLIFE_YEARS, RECENT_WEIGHT_YEARS,
        HISTORICAL_UP_RATE, HORIZON_V18, EMBARGO_V18,
    )
    print("  v18 функции загружены.")
except ImportError as e:
    raise ImportError(f"v18 не найден: {e}")

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
        binarize_labels,
        select_top_features,
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
        COT_FEATURES, COT_RELEASE_LAG,
    )
    print("  v15 функции загружены.")
except ImportError as e:
    raise ImportError(f"v15 не найден: {e}")

# ---- Импорты из v14 ----
try:
    from silver_assistant_v14_main import (
        fetch_ohlc, build_features,
        add_triple_barrier_labels, get_feature_cols,
        buy_and_hold_return,
        wilson_ci, pct,
    )
    print("  v14 функции загружены.")
except ImportError as e:
    raise ImportError(f"v14 не найден: {e}")


# ---------------------------------------------------------------------------
# Константы v22
# ---------------------------------------------------------------------------

ATR_K_GRID          = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
ATR_K_DEFAULT       = 3.5
ATR_PERIOD_V22      = 14
MAX_KELLY_FRACTION  = 0.25
KELLY_HALF          = 0.5       # half-Kelly для безопасности
HORIZONS_MULTI      = [5, 15, 30]
RETRAIN_FREQ_DAYS   = 60        # рабочих дней между переобучениями WF
EMBARGO_V22         = EMBARGO_V18
HORIZON_V22         = HORIZON_V18
V22_DIR             = "baseline_outputs_v22"


# ===========================================================================
# MODULE A: ATR computation
# ===========================================================================

def ensure_atr(df: pd.DataFrame, period: int = ATR_PERIOD_V22) -> pd.DataFrame:
    """
    Добавляет silver_atr_{period}d если ещё нет.
    Использует EWM True Range (более гладкий, чем SMA).
    Fallback на volatility×price если OHLC недоступен.
    """
    atr_col = f"silver_atr_{period}d"
    if atr_col in df.columns:
        non_nan = df[atr_col].notna().sum()
        print(f"  ATR: {atr_col} уже существует ({non_nan} строк), median={df[atr_col].median():.3f}")
        return df

    if "silver_high" not in df.columns or "silver_low" not in df.columns:
        print(f"  WARN: OHLC нет → ATR fallback = silver_volatility × silver_close")
        if "silver_volatility" in df.columns and "silver_close" in df.columns:
            df[atr_col] = df["silver_volatility"] * df["silver_close"]
        else:
            df[atr_col] = df["silver_close"] * 0.015
        return df

    close_prev = df["silver_close"].shift(1)
    tr = pd.concat([
        df["silver_high"] - df["silver_low"],
        (df["silver_high"] - close_prev).abs(),
        (df["silver_low"]  - close_prev).abs(),
    ], axis=1).max(axis=1)

    df[atr_col] = tr.ewm(span=period, min_periods=max(period // 2, 5)).mean()
    print(f"  ATR: {atr_col} вычислен. Median={df[atr_col].median():.3f}, "
          f"Min={df[atr_col].min():.3f}, Max={df[atr_col].max():.3f}")
    return df


# ===========================================================================
# MODULE D: Multi-horizon labels & ensemble
# ===========================================================================

def add_multilabels(
    df: pd.DataFrame,
    horizons: List[int] = HORIZONS_MULTI,
    base_horizon: int = 15,
) -> pd.DataFrame:
    """
    Добавляет triple-barrier метки для каждого горизонта (помимо базового 15d).
    Колонки: tb_label_{h}d, tb_label_bin_{h}d  (h ∈ horizons, h ≠ base_horizon)
    Базовый горизонт: tb_label / tb_label_bin (уже существуют).
    """
    for h in horizons:
        label_col = f"tb_label_{h}d"
        bin_col   = f"tb_label_bin_{h}d"

        if h == base_horizon:
            # Алиасы для единообразия
            if "tb_label" in df.columns and label_col not in df.columns:
                df[label_col] = df["tb_label"]
            if "tb_label_bin" in df.columns and bin_col not in df.columns:
                df[bin_col] = df["tb_label_bin"]
            continue

        if bin_col in df.columns:
            print(f"  MultiLabel {h}d: уже существует.")
            continue

        print(f"  MultiLabel {h}d: вычисление triple-barrier (горизонт={h}d)...")
        # Сохраняем оригинальную метку
        orig_tb = df["tb_label"].copy() if "tb_label" in df.columns else None

        df_temp = add_triple_barrier_labels(df.copy(), horizon=h)
        df[label_col] = df_temp["tb_label"]
        df[bin_col] = np.where(
            df[label_col].isna(), np.nan,
            (df[label_col] == "UP").astype(float)
        )

        # Восстанавливаем оригинальную tb_label
        if orig_tb is not None:
            df["tb_label"] = orig_tb

        dist = df[label_col].value_counts(dropna=True).to_dict()
        n_bin = int(df[bin_col].notna().sum())
        up_rate = float(df[bin_col].mean()) if n_bin > 0 else float("nan")
        print(f"  MultiLabel {h}d: {dist}, UP_rate={up_rate:.3f}, n_labeled={n_bin}")

    return df


def train_multi_horizon_split(
    df: pd.DataFrame,
    cutoff_date: str,
    feature_cols: List[str],
    horizons: List[int] = HORIZONS_MULTI,
    embargo_days: int = EMBARGO_V22,
) -> Dict[int, "RegimeEnsembleV18"]:
    """
    Обучает отдельный UP-классификатор для каждого горизонта.
    Возвращает dict {horizon: model}.
    """
    models: Dict[int, RegimeEnsembleV18] = {}
    cutoff_dt = pd.Timestamp(cutoff_date) - pd.Timedelta(days=embargo_days)

    for h in horizons:
        bin_col = f"tb_label_bin_{h}d" if h != 15 else "tb_label_bin"
        if bin_col not in df.columns:
            print(f"  WARN MH {h}d: {bin_col} не найден, пропускаем.")
            continue

        train = df[(df.index <= cutoff_dt) & df[bin_col].notna()].copy()
        if len(train) < 80:
            print(f"  WARN MH {h}d: мало данных ({len(train)}), пропускаем.")
            continue

        sw = compute_sample_weights(train)
        model = RegimeEnsembleV18()
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(
                train[feature_cols],
                train[bin_col].values.astype(int),
                _get_regimes(train),
                sample_weight=sw,
            )
        models[h] = model

    print(f"  MH models: обучено {len(models)} горизонтов {list(models.keys())} "
          f"на cutoff={cutoff_date}")
    return models


def apply_mh_proba(
    df: pd.DataFrame,
    models_by_horizon: Dict[int, "RegimeEnsembleV18"],
    feature_cols: List[str],
) -> pd.Series:
    """
    Усредняет P(UP) от всех горизонтных моделей.
    Возвращает Series p_up_mh (индекс как у df).
    """
    if not models_by_horizon:
        # Fallback: возвращаем p_up если есть
        if "p_up" in df.columns:
            return df["p_up"].rename("p_up_mh")
        return pd.Series(0.5, index=df.index, name="p_up_mh")

    X       = df[feature_cols]
    regimes = _get_regimes(df)
    probas  = []
    for h, model in models_by_horizon.items():
        proba   = model.predict_proba(X, regimes)
        up_idx  = list(model.classes_).index(1)
        probas.append(proba[:, up_idx])

    avg_p = np.mean(probas, axis=0)
    return pd.Series(avg_p, index=df.index, name="p_up_mh")


def apply_policy_mh(
    df: pd.DataFrame,
    models_by_horizon: Dict[int, "RegimeEnsembleV18"],
    feature_cols: List[str],
    up_threshold: float,
    cooldown: int,
) -> pd.DataFrame:
    """
    Генерирует сигналы BUY используя ensemble MH вероятностей.
    Кладёт p_up_mh в df; signal_long = BUY / HOLD на основе p_up_mh.
    """
    out = df.copy()
    p_up_mh = apply_mh_proba(out, models_by_horizon, feature_cols)
    out["p_up_mh"]    = p_up_mh.values
    out["p_up"]       = p_up_mh.values  # переписываем для единообразия
    out["signal_long"] = "HOLD"

    last_pos  = -cooldown
    p_arr     = p_up_mh.values
    for i in range(len(out)):
        if p_arr[i] >= up_threshold and i - last_pos >= cooldown:
            out.iloc[i, out.columns.get_loc("signal_long")] = "BUY"
            last_pos = i

    n_buy = (out["signal_long"] == "BUY").sum()
    print(f"    MH policy: thr={up_threshold}, cd={cooldown} → {n_buy} BUY")
    return out


# ===========================================================================
# MODULE A continued: ATR-based trailing stop backtest
# ===========================================================================

def _atr_at(df: pd.DataFrame, date, atr_col: str) -> float:
    """Безопасно берёт ATR из датафрейма."""
    try:
        v = float(df.loc[date, atr_col])
        return v if not math.isnan(v) and v > 0 else float(df["silver_close"].loc[date]) * 0.015
    except Exception:
        return float(df["silver_close"].iloc[0]) * 0.015


def backtest_atr_independent(
    df: pd.DataFrame,
    split: str,
    atr_k_long:    float = ATR_K_DEFAULT,
    max_hold_long:  int   = MAX_HOLD_DEFAULT,
    atr_k_short:   float = ATR_K_DEFAULT,
    max_hold_short: int   = MAX_HOLD_DEFAULT,
    cost: float           = COST_PER_TRADE,
    atr_col: str          = f"silver_atr_{ATR_PERIOD_V22}d",
) -> pd.DataFrame:
    """
    Бэктест с ATR-based trailing stop.
    LONG:  trail_dist = atr_k_long  × ATR_at_entry; stop = peak  - trail_dist
    SHORT: trail_dist = atr_k_short × ATR_at_entry; stop = trough + trail_dist
    """
    d = df[df["split"] == split].sort_index()
    if d.empty:
        return pd.DataFrame()

    has_high = "silver_high" in d.columns
    has_low  = "silver_low"  in d.columns
    has_atr  = atr_col in d.columns

    all_trades: List[dict] = []

    # ---- LONG ----
    for entry_date in d[d.get("signal_long", d.get("signal", pd.Series(dtype=str))) == "BUY"].index:
        ep  = d.index.get_loc(entry_date)
        ep0 = float(d.loc[entry_date, "silver_close"])

        atr_val    = _atr_at(d, entry_date, atr_col) if has_atr else ep0 * 0.015
        trail_dist = atr_k_long * atr_val
        trail_pct_eq = trail_dist / ep0  # для справки
        peak       = ep0
        trail_stop = ep0 - trail_dist
        exit_price = ep0
        exit_idx   = ep
        exit_rsn   = "max_hold"

        for j in range(1, max_hold_long + 1):
            pos = ep + j
            if pos >= len(d):
                break
            hi = float(d.iloc[pos]["silver_high"]) if has_high else float(d.iloc[pos]["silver_close"])
            lo = float(d.iloc[pos]["silver_low"])  if has_low  else float(d.iloc[pos]["silver_close"])
            cl = float(d.iloc[pos]["silver_close"])
            if hi > peak:
                peak       = hi
                trail_stop = peak - trail_dist
            if lo <= trail_stop:
                exit_price = min(cl, trail_stop)
                exit_idx   = pos
                exit_rsn   = "trail_stop"
                break
            exit_price = cl
            exit_idx   = pos

        gr = exit_price / ep0 - 1.0
        nr = gr - cost
        p_sig = float(d.loc[entry_date, "p_up"]) if "p_up" in d.columns else 0.5

        all_trades.append({
            "direction":    "LONG",
            "signal_date":  entry_date,
            "entry_date":   entry_date,
            "exit_date":    d.index[exit_idx],
            "entry_price":  round(ep0,        3),
            "exit_price":   round(exit_price, 3),
            "peak_price":   round(peak,       3),
            "trail_stop":   round(trail_stop, 3),
            "atr_at_entry": round(atr_val,    4),
            "atr_k":        atr_k_long,
            "trail_pct_eq": round(trail_pct_eq, 4),
            "gross_return": round(gr, 6),
            "net_return":   round(nr, 6),
            "hold_days":    exit_idx - ep,
            "exit_reason":  exit_rsn,
            "p_signal":     round(p_sig, 4),
            "tb_label_bin": d.loc[entry_date, "tb_label_bin"] if "tb_label_bin" in d.columns else None,
        })

    # ---- SHORT ----
    if "signal_short" in d.columns:
        for entry_date in d[d["signal_short"] == "SHORT"].index:
            ep  = d.index.get_loc(entry_date)
            ep0 = float(d.loc[entry_date, "silver_close"])

            atr_val    = _atr_at(d, entry_date, atr_col) if has_atr else ep0 * 0.015
            trail_dist = atr_k_short * atr_val
            trough     = ep0
            trail_stop = ep0 + trail_dist
            exit_price = ep0
            exit_idx   = ep
            exit_rsn   = "max_hold"

            for j in range(1, max_hold_short + 1):
                pos = ep + j
                if pos >= len(d):
                    break
                hi = float(d.iloc[pos]["silver_high"]) if has_high else float(d.iloc[pos]["silver_close"])
                lo = float(d.iloc[pos]["silver_low"])  if has_low  else float(d.iloc[pos]["silver_close"])
                cl = float(d.iloc[pos]["silver_close"])
                if lo < trough:
                    trough     = lo
                    trail_stop = trough + trail_dist
                if hi >= trail_stop:
                    exit_price = max(cl, trail_stop)
                    exit_idx   = pos
                    exit_rsn   = "trail_stop"
                    break
                exit_price = cl
                exit_idx   = pos

            gr = ep0 / exit_price - 1.0
            nr = gr - cost
            p_sig = float(d.loc[entry_date, "p_short"]) if "p_short" in d.columns else 0.5

            all_trades.append({
                "direction":    "SHORT",
                "signal_date":  entry_date,
                "entry_date":   entry_date,
                "exit_date":    d.index[exit_idx],
                "entry_price":  round(ep0,        3),
                "exit_price":   round(exit_price, 3),
                "trough_price": round(trough,     3),
                "trail_stop":   round(trail_stop, 3),
                "atr_at_entry": round(atr_val,    4),
                "atr_k":        atr_k_short,
                "trail_pct_eq": round(trail_dist / ep0, 4),
                "gross_return": round(gr, 6),
                "net_return":   round(nr, 6),
                "hold_days":    exit_idx - ep,
                "exit_reason":  exit_rsn,
                "p_signal":     round(p_sig, 4),
                "tb_label_down": d.loc[entry_date, "tb_label_down"] if "tb_label_down" in d.columns else None,
            })

    if not all_trades:
        return pd.DataFrame()
    return pd.DataFrame(all_trades).sort_values("entry_date").reset_index(drop=True)


def select_atr_k(
    valid_df: pd.DataFrame,
    direction: str = "LONG",
    atr_k_grid: List[float] = ATR_K_GRID,
    max_hold_grid: List[int] = MAX_HOLD_GRID,
    cost: float = COST_PER_TRADE,
) -> Tuple[float, int]:
    """Подбирает ATR k + max_hold по Sharpe на valid."""
    best_sharpe = float("-inf")
    best_k, best_hold = ATR_K_DEFAULT, MAX_HOLD_DEFAULT

    print(f"\n  Поиск ATR-k для {direction} (valid):")
    print(f"  {'atr_k':>8}  {'max_hold':>9}  {'n':>5}  {'trail_pct_eq':>13}  {'sharpe':>8}")

    for k in atr_k_grid:
        for max_hold in max_hold_grid:
            if direction == "LONG":
                trades = backtest_atr_independent(
                    valid_df, "valid",
                    atr_k_long=k, max_hold_long=max_hold,
                    atr_k_short=ATR_K_DEFAULT, max_hold_short=MAX_HOLD_DEFAULT,
                    cost=cost,
                )
                t = trades[trades["direction"] == "LONG"] if not trades.empty else trades
            else:
                trades = backtest_atr_independent(
                    valid_df, "valid",
                    atr_k_long=ATR_K_DEFAULT, max_hold_long=MAX_HOLD_DEFAULT,
                    atr_k_short=k, max_hold_short=max_hold,
                    cost=cost,
                )
                t = trades[trades["direction"] == "SHORT"] if not trades.empty else trades

            sharpe = _sharpe_from_trades(t)
            n = len(t) if not t.empty else 0
            eq = t["trail_pct_eq"].mean() if not t.empty and "trail_pct_eq" in t.columns else float("nan")
            mark = " ← best" if sharpe > best_sharpe and n >= 3 else ""
            print(f"  {k:>8.1f}  {max_hold:>9}  {n:>5}  "
                  f"{eq:>13.2%}  {sharpe:>8.4f}{mark}")
            if sharpe > best_sharpe and n >= 3:
                best_sharpe = sharpe
                best_k, best_hold = k, max_hold

    print(f"\n  >>> {direction} ATR: k={best_k}, max_hold={best_hold}d, sharpe={best_sharpe:.4f}")
    return best_k, best_hold


# ===========================================================================
# MODULE B: Kelly position sizing
# ===========================================================================

def kelly_fraction(
    p_signal: float,
    mean_p: float = 0.48,
    max_frac_ratio: float = 2.5,
) -> float:
    """
    Пропорциональное sizing: kelly_frac = p_signal / mean_p
    Означает: высококонфидентные сигналы получают бо́льший размер позиции.
    clip([1/max_frac_ratio, max_frac_ratio]) — не даём экстремальных весов.

    Интерпретация:
      p_signal = mean_p  → frac = 1.0  (как flat sizing)
      p_signal = 0.65    → frac ≈ 1.35 (на 35% больше среднего)
      p_signal = 0.42    → frac ≈ 0.88 (на 12% меньше среднего)
    """
    if mean_p <= 0:
        return 1.0
    raw = p_signal / mean_p
    return float(np.clip(raw, 1.0 / max_frac_ratio, max_frac_ratio))


def apply_kelly_sizing(
    trades: pd.DataFrame,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Пропорциональный sizing по уверенности модели (Kelly-inspired).
    Размер позиции ∝ p_signal, нормирован так, что mean_frac = 1.0.

    Добавляет колонки: kelly_frac, kelly_net_return.
    Сумма kelly_net_return сопоставима с flat net_return (тот же капитал).
    """
    if trades.empty or "p_signal" not in trades.columns:
        return trades

    t = trades.copy()
    p = t["p_signal"].fillna(0.5).values.astype(float)
    mean_p = float(p.mean()) if len(p) > 0 else 0.48

    fracs = np.array([kelly_fraction(float(pi), mean_p=mean_p) for pi in p])

    if normalize:
        mf = fracs.mean()
        if mf > 0:
            fracs = fracs / mf  # mean_frac = 1.0

    t["kelly_frac"] = fracs
    t["kelly_net_return"] = t["net_return"] * t["kelly_frac"]
    return t


# ===========================================================================
# MODULE C: Risk metrics
# ===========================================================================

def compute_risk_metrics(
    trades: pd.DataFrame,
    split: str,
    kelly_weighted: bool = False,
    label: str = "strategy",
) -> dict:
    """
    Вычисляет полный набор метрик риска.

    Метрики:
      - total_net_return:  суммарный P&L
      - annualized_return: (equity_final)^(1/years) - 1
      - max_drawdown:      макс. просадка от пика (%)
      - sharpe_ratio:      ann. Шарп (по trade-returns)
      - calmar_ratio:      ann_return / |max_drawdown|
      - time_underwater:   доля сделок ниже equity-пика
      - ulcer_index:       RMS просадок
      - win_rate:          доля прибыльных сделок
      - n_trades:          кол-во сделок
    """
    empty = {
        "split": split, "label": label, "n_trades": 0,
        "win_rate": None, "total_net_return": 0.0,
        "annualized_return": None, "max_drawdown": None,
        "sharpe_ratio": None, "calmar_ratio": None,
        "time_underwater": None, "ulcer_index": None,
    }
    if trades.empty:
        return empty

    ret_col = ("kelly_net_return"
               if kelly_weighted and "kelly_net_return" in trades.columns
               else "net_return")

    t = trades.copy()
    t["exit_date"]  = pd.to_datetime(t["exit_date"])
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t_s = t.sort_values("exit_date").reset_index(drop=True)

    n         = len(t_s)
    rets      = t_s[ret_col].values
    total_ret = float(rets.sum())
    win_rate  = float((rets > 0).mean())

    # Equity curve (trade-ordered, additive)
    equity = 1.0 + np.cumsum(rets)

    # Период (лет)
    if n >= 2:
        n_days = (t_s["exit_date"].max() - t_s["entry_date"].min()).days
        years  = max(n_days / 365.25, 0.1)
    else:
        years = 1.0

    # Аннуализированная доходность (CAGR)
    final_eq = equity[-1]
    ann_ret  = (max(final_eq, 0.001) ** (1.0 / years)) - 1.0 if final_eq > 0 else None

    # Drawdown
    running_max = np.maximum.accumulate(equity)
    dd_series   = (equity / running_max) - 1.0
    max_dd      = float(dd_series.min())
    time_uw     = float((dd_series < -0.001).mean())
    ulcer       = float(np.sqrt(np.mean(dd_series ** 2)))

    # Sharpe (trade-level, аннуализированный)
    if n >= 3:
        mean_r = np.mean(rets)
        std_r  = np.std(rets, ddof=1)
        trades_per_year = n / years
        sharpe = (mean_r / std_r * np.sqrt(trades_per_year)) if std_r > 0 else 0.0
    else:
        sharpe = None

    # Calmar
    calmar = (ann_ret / abs(max_dd)) if (ann_ret is not None and max_dd < -0.001) else None

    return {
        "split":             split,
        "label":             label,
        "n_trades":          n,
        "win_rate":          round(win_rate, 4),
        "total_net_return":  round(total_ret, 4),
        "annualized_return": round(ann_ret, 4) if ann_ret is not None else None,
        "max_drawdown":      round(max_dd, 4),
        "sharpe_ratio":      round(sharpe, 3) if sharpe is not None else None,
        "calmar_ratio":      round(calmar, 3) if calmar is not None else None,
        "time_underwater":   round(time_uw, 4),
        "ulcer_index":       round(ulcer, 4),
    }


def print_risk_table(metrics_list: List[dict]) -> None:
    """Красиво печатает таблицу метрик риска для всех вариантов."""
    if not metrics_list:
        return
    df = pd.DataFrame(metrics_list)
    float_cols = [
        "total_net_return", "annualized_return", "max_drawdown",
        "sharpe_ratio", "calmar_ratio", "time_underwater", "ulcer_index",
    ]
    for c in float_cols:
        if c in df.columns:
            if c in ("max_drawdown", "total_net_return", "annualized_return", "time_underwater"):
                df[c] = df[c].apply(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
            else:
                df[c] = df[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    print(df.to_string(index=False))


# ===========================================================================
# MODULE E: Walk-forward retraining
# ===========================================================================

def _get_wf_cutoffs(
    df: pd.DataFrame,
    start_cutoff: str,
    end_date,
    freq: int = RETRAIN_FREQ_DAYS,
) -> List[pd.Timestamp]:
    """Генерирует список дат переобучения с шагом freq рабочих дней."""
    all_dates = df.index.sort_values()
    base = pd.Timestamp(start_cutoff)

    # Находим позицию первой даты >= base
    mask = all_dates >= base
    if not mask.any():
        return []
    first_pos = np.where(mask)[0][0]
    end_ts    = pd.Timestamp(end_date)

    cutoffs = []
    pos = first_pos
    while pos < len(all_dates) and all_dates[pos] <= end_ts:
        cutoffs.append(all_dates[pos])
        pos += freq

    return cutoffs


def generate_wf_signals(
    df: pd.DataFrame,
    feature_cols: List[str],
    policy_up: dict,
    policy_short: dict,
    mh_models_by_cutoff: Optional[Dict[str, Dict[int, "RegimeEnsembleV18"]]] = None,
    retrain_freq: int = RETRAIN_FREQ_DAYS,
) -> pd.DataFrame:
    """
    Генерирует сигналы для test и forward сплитов с walk-forward переобучением.
    Train и valid используют стандартные (уже вычисленные) сигналы из df.

    mh_models_by_cutoff: опционально - MH-модели по дате (для v22_all).
    Если None, использует стандартный single UP-model.

    Возвращает: DataFrame со столбцами signal_long, signal_short, p_up, p_short, split.
    """
    parts = []

    # Train и valid: берём как есть
    for s in ["train", "valid"]:
        part = df[df["split"] == s].copy()
        if not part.empty:
            parts.append(part)

    # Test и forward: walk-forward
    for s in ["test", "forward"]:
        split_df = df[df["split"] == s].copy()
        if split_df.empty:
            continue

        split_start_cutoff = EXPAND_CUTOFFS.get(s, None)
        if split_start_cutoff is None:
            print(f"  WARN WF [{s}]: нет cutoff в EXPAND_CUTOFFS → используем первую дату сплита")
            split_start_cutoff = split_df.index.min().strftime("%Y-%m-%d")

        split_end = split_df.index.max()
        cutoffs = _get_wf_cutoffs(df, split_start_cutoff, split_end, freq=retrain_freq)
        if not cutoffs:
            print(f"  WARN WF [{s}]: не удалось построить cutoffs → стандартные сигналы")
            parts.append(split_df)
            continue

        print(f"\n  WF [{s}]: {len(cutoffs)} переобучений "
              f"({cutoffs[0].date()} → {cutoffs[-1].date()}, freq={retrain_freq}d)")

        wf_parts = []
        for i_c, cutoff_dt in enumerate(cutoffs):
            next_cutoff = (cutoffs[i_c + 1] if i_c + 1 < len(cutoffs)
                           else split_df.index.max() + pd.Timedelta(days=1))
            window = split_df[
                (split_df.index >= cutoff_dt) & (split_df.index < next_cutoff)
            ].copy()
            if window.empty:
                continue

            cutoff_str = cutoff_dt.strftime("%Y-%m-%d")

            # Обучаем UP и DOWN модели
            model_up, wt_up, _   = train_expanding_model(df, cutoff_str, feature_cols)
            model_dn, wt_dn, _   = train_expanding_model_down(df, cutoff_str, feature_cols)

            # Если MH-модели заданы, используем их для p_up_ensemble
            use_mh = (mh_models_by_cutoff is not None)
            if use_mh:
                mh_models = train_multi_horizon_split(df, cutoff_str, feature_cols)
                window = apply_policy_mh(
                    window, mh_models, feature_cols,
                    policy_up["up_threshold"], policy_up["cooldown"],
                )
            else:
                window = apply_policy_v16(
                    window, model_up, feature_cols,
                    policy_up["up_threshold"], policy_up["cooldown"],
                )
                window["signal_long"] = window["signal"]

            # SHORT сигналы с режимным фильтром
            window = apply_policy_short_v21(
                window, model_dn, feature_cols,
                policy_short["down_threshold"], policy_short["short_cooldown"],
                blocked_regimes=SHORT_BLOCKED_REGIMES,
            )

            n_buy   = (window.get("signal_long", window.get("signal", pd.Series())) == "BUY").sum()
            n_short = (window.get("signal_short", pd.Series(dtype=str)) == "SHORT").sum()
            print(f"    WF {cutoff_dt.date()}: window={len(window)}d, "
                  f"BUY={n_buy}, SHORT={n_short}, "
                  f"wt_up={wt_up:.2f}, wt_dn={wt_dn:.2f}")
            wf_parts.append(window)

        if wf_parts:
            parts.append(pd.concat(wf_parts).sort_index())

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


# ===========================================================================
# Вспомогательные: сводка варианта
# ===========================================================================

def variant_summary(trades: pd.DataFrame, split: str, bnh: float, label: str,
                    kelly_weighted: bool = False) -> dict:
    """Краткая сводка результатов варианта."""
    if trades.empty:
        return {
            "variant": label, "split": split, "n_total": 0,
            "n_long": 0, "n_short": 0,
            "total_net": 0.0, "long_net": 0.0, "short_net": 0.0,
            "win_rate": None, "buy_and_hold": round(bnh, 4), "vs_bnh": round(-bnh, 4),
        }

    ret_col = ("kelly_net_return"
               if kelly_weighted and "kelly_net_return" in trades.columns
               else "net_return")

    longs  = trades[trades["direction"] == "LONG"]  if "direction" in trades.columns else trades
    shorts = trades[trades["direction"] == "SHORT"] if "direction" in trades.columns else pd.DataFrame()
    long_net  = float(longs[ret_col].sum())  if not longs.empty  else 0.0
    short_net = float(shorts[ret_col].sum()) if not shorts.empty else 0.0
    total_net = long_net + short_net
    win_rate  = float((trades[ret_col] > 0).mean()) if not trades.empty else None

    return {
        "variant":      label,
        "split":        split,
        "n_total":      len(trades),
        "n_long":       len(longs),
        "n_short":      len(shorts),
        "total_net":    round(total_net, 4),
        "long_net":     round(long_net,  4),
        "short_net":    round(short_net, 4),
        "win_rate":     round(win_rate, 4) if win_rate is not None else None,
        "buy_and_hold": round(bnh, 4),
        "vs_bnh":       round(total_net - bnh, 4),
        "kelly_weighted": kelly_weighted,
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",   default="2013-01-01")
    ap.add_argument("--end",     default="2099-12-31")
    ap.add_argument("--out-dir", default=V22_DIR)
    ap.add_argument("--no-wf",   action="store_true", help="Пропустить walk-forward (быстро)")
    ap.add_argument("--no-mh",   action="store_true", help="Пропустить multi-horizon (быстро)")
    args = ap.parse_args(argv)

    end = min(args.end, pd.Timestamp.today().strftime("%Y-%m-%d"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_wf = not args.no_wf
    run_mh = not args.no_mh

    print("=" * 65)
    print("=== Silver Trading Assistant v22 — Risk-Aware            ===")
    print(f"  Модули: ATR={'✅'} Kelly={'✅'} Risk={'✅'} "
          f"MH={'✅' if run_mh else '⏭'} WF={'✅' if run_wf else '⏭'}")
    print(f"  ATR_K_GRID={ATR_K_GRID}, ATR_PERIOD={ATR_PERIOD_V22}d")
    print(f"  KELLY: half={KELLY_HALF}, max_frac={MAX_KELLY_FRACTION:.0%}")
    print(f"  MH horizons: {HORIZONS_MULTI}d")
    print(f"  WF retrain_freq: {RETRAIN_FREQ_DAYS} торговых дней")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 1. Загрузка OHLC (с fallback на кэш v21/v20/v19)
    # -----------------------------------------------------------------------
    print("\n=== v22: загрузка OHLC ===")
    _cache_has_features = False
    df = None

    try:
        df = fetch_ohlc(args.start, end)
        print(f"  OHLC: {len(df)} строк, {df.index[0].date()} — {df.index[-1].date()}")
    except Exception as e:
        print(f"  WARN fetch_ohlc ({e}) → кэш...")
        for cp in [
            Path("baseline_outputs_v21/v21_full_data.csv"),
            Path("baseline_outputs_v20/v20_full_data.csv"),
            Path("baseline_outputs_v19/v19_full_data.csv"),
        ]:
            if cp.exists():
                df = pd.read_csv(cp, index_col=0, parse_dates=True)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df = df[(df.index >= args.start) & (df.index <= end)]
                print(f"  Кэш: {cp} ({len(df)} строк)")
                break

    if df is None:
        raise RuntimeError("OHLC недоступен и кэш не найден.")

    _cache_has_features = "tb_label" in df.columns

    # -----------------------------------------------------------------------
    # 2. ATR — вычисляем до feature engineering (нужен OHLC)
    # -----------------------------------------------------------------------
    print("\n=== v22: Module A — ATR ===")
    df = ensure_atr(df, period=ATR_PERIOD_V22)

    # -----------------------------------------------------------------------
    # 3. Макро + COT + Feature Engineering
    # -----------------------------------------------------------------------
    if not _cache_has_features:
        print("\n=== v22: загрузка макро ===")
        macro = fetch_macro_yfinance(args.start, end)
        if not macro.empty:
            macro.to_csv(out / "v22_macro_raw.csv")

        print("\n=== v22: загрузка COT ===")
        cot = fetch_cot_silver(
            pd.Timestamp(args.start).year, pd.Timestamp(end).year
        )
        if not cot.empty:
            cot.to_csv(out / "v22_cot_raw.csv")
        else:
            cot = pd.DataFrame()

        print("\n=== v22: инженерия признаков ===")
        df = build_features(df, pd.DataFrame())
        df = add_vol_trend_regime(df)
        df = add_macro_features(df, macro)
        if not cot.empty:
            df = merge_cot_to_daily(df, cot, lag_days=COT_RELEASE_LAG)

        present_cot    = [c for c in COT_FEATURES if c in df.columns]
        macro_features = [c for c in MACRO_FEATURE_NAMES if c in df.columns]

        print("\n=== v22: triple-barrier метки (15d base) ===")
        df = add_triple_barrier_labels(df, horizon=HORIZON_V22)
        df = binarize_labels(df)
        df = add_down_label(df)
        df["split"] = df.index.map(split_name)
    else:
        print("\n=== v22: кэш — пропускаем feature engineering ===")
        present_cot    = [c for c in COT_FEATURES if c in df.columns]
        macro_features = [c for c in MACRO_FEATURE_NAMES if c in df.columns]
        if "tb_label_bin"  not in df.columns:
            df = binarize_labels(df)
        if "tb_label_down" not in df.columns:
            df = add_down_label(df)
        if "split"         not in df.columns:
            df["split"] = df.index.map(split_name)

    # ATR после feature engineering (колонка могла быть сброшена)
    df = ensure_atr(df, period=ATR_PERIOD_V22)

    # -----------------------------------------------------------------------
    # 4. Module D: Multi-horizon метки (5d, 30d)
    # -----------------------------------------------------------------------
    if run_mh:
        print("\n=== v22: Module D — Multi-horizon метки ===")
        df = add_multilabels(df, horizons=HORIZONS_MULTI, base_horizon=15)
    else:
        print("\n  [MH пропущен]")

    df.to_csv(out / "v22_full_data.csv")

    lb = label_report_directional(df)
    lb.to_csv(out / "v22_label_distribution.csv", index=False)
    print(f"  Метки (3-класс):\n{lb.to_string(index=False)}")

    # -----------------------------------------------------------------------
    # 5. Список признаков
    # -----------------------------------------------------------------------
    base_features = get_feature_cols(df)
    extra_cot     = [c for c in present_cot if c not in base_features]
    all_features  = list(dict.fromkeys(base_features + macro_features + extra_cot))
    print(f"\n  Всего признаков: {len(all_features)}")

    # -----------------------------------------------------------------------
    # 6. PASS 1: Отбор признаков (UP-модель, valid-окно)
    # -----------------------------------------------------------------------
    print("\n=== v22: отбор признаков ===")
    valid_cutoff = pd.Timestamp(EXPAND_CUTOFFS["valid"]) - pd.Timedelta(days=EMBARGO_V22)
    p1_train     = df[(df.index <= valid_cutoff) & df["tb_label_bin"].notna()].copy()
    p1_valid     = df[(df["split"] == "valid")   & df["tb_label_bin"].notna()].copy()

    sw_p1    = compute_sample_weights(p1_train)
    model_p1 = RegimeEnsembleV18()
    with contextlib.redirect_stdout(io.StringIO()):
        model_p1.fit(
            p1_train[all_features],
            p1_train["tb_label_bin"].values.astype(int),
            _get_regimes(p1_train),
            sample_weight=sw_p1,
        )
    selected, imp_series = select_top_features(
        model_p1, p1_valid[all_features],
        p1_valid["tb_label_bin"].values.astype(int),
        n_top=TOP_FEATURES_N,
    )
    imp_df = imp_series.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df.to_csv(out / "v22_feature_importance.csv", index=False)
    print(f"  Отобрано: {len(selected)} / {len(all_features)}")

    # -----------------------------------------------------------------------
    # 7. PASS 2: Expanding models (UP + DOWN)
    # -----------------------------------------------------------------------
    print("\n=== v22: expanding models UP + DOWN ===")
    split_models_up:   Dict[str, "RegimeEnsembleV18"] = {}
    split_models_down: Dict[str, "RegimeEnsembleV18"] = {}
    split_weights_up:  Dict[str, float] = {}
    split_weights_down: Dict[str, float] = {}

    for sk, cutoff in EXPAND_CUTOFFS.items():
        m_up, wt_up, _   = train_expanding_model(df, cutoff, selected)
        m_dn, wt_dn, _   = train_expanding_model_down(df, cutoff, selected)
        split_models_up[sk]    = m_up
        split_models_down[sk]  = m_dn
        split_weights_up[sk]   = wt_up
        split_weights_down[sk] = wt_dn
        print(f"  [{sk}] UP wt={wt_up:.2f}, DOWN wt={wt_dn:.2f}")

    # -----------------------------------------------------------------------
    # 8. Multi-horizon models для valid/test/forward
    # -----------------------------------------------------------------------
    mh_models_by_split: Dict[str, Dict[int, "RegimeEnsembleV18"]] = {}
    if run_mh:
        print("\n=== v22: Module D — Multi-horizon models ===")
        for sk, cutoff in EXPAND_CUTOFFS.items():
            mh_models_by_split[sk] = train_multi_horizon_split(
                df, cutoff, selected, horizons=HORIZONS_MULTI
            )

    # -----------------------------------------------------------------------
    # 9. Выбор политик
    # -----------------------------------------------------------------------
    print("\n=== v22: выбор UP-политики (valid) ===")
    valid_full = df[df["split"] == "valid"].copy()
    policy_up  = select_policy_v18(valid_full, split_models_up["valid"], selected)
    print(f"  UP: {policy_up}")

    print("\n=== v22: выбор SHORT-политики (valid, режимный фильтр) ===")
    policy_short = select_policy_short_v21(
        valid_full, split_models_down["valid"], selected,
        blocked_regimes=SHORT_BLOCKED_REGIMES,
    )

    # -----------------------------------------------------------------------
    # 10. Базовые сигналы v21-стиль (для v22_base, v22_atr, v22_kelly)
    # -----------------------------------------------------------------------
    print("\n=== v22: применение базовых сигналов (v21-стиль) ===")
    base_signal_parts = []
    for sk in ["train", "valid", "test", "forward"]:
        sdf = df[df["split"] == sk].copy()
        if sdf.empty:
            continue
        mu = split_models_up.get(sk, split_models_up["valid"])
        md = split_models_down.get(sk, split_models_down["valid"])

        part = apply_policy_v16(
            sdf, mu, selected,
            policy_up["up_threshold"], policy_up["cooldown"],
        )
        part["signal_long"] = part["signal"]
        part = apply_policy_short_v21(
            part, md, selected,
            policy_short["down_threshold"], policy_short["short_cooldown"],
            blocked_regimes=SHORT_BLOCKED_REGIMES,
        )
        base_signal_parts.append(part)

    base_df = pd.concat(base_signal_parts).sort_index()
    base_df.to_csv(out / "v22_base_decisions.csv")

    # -----------------------------------------------------------------------
    # 11. Multi-horizon сигналы (для v22_mh)
    # -----------------------------------------------------------------------
    mh_df = pd.DataFrame()
    if run_mh:
        print("\n=== v22: применение Multi-horizon сигналов ===")
        mh_parts = []
        for sk in ["train", "valid", "test", "forward"]:
            sdf = df[df["split"] == sk].copy()
            if sdf.empty:
                continue
            mh_m  = mh_models_by_split.get(sk, mh_models_by_split.get("valid", {}))
            md_sk = split_models_down.get(sk, split_models_down["valid"])

            part = apply_policy_mh(
                sdf, mh_m, selected,
                policy_up["up_threshold"], policy_up["cooldown"],
            )
            part = apply_policy_short_v21(
                part, md_sk, selected,
                policy_short["down_threshold"], policy_short["short_cooldown"],
                blocked_regimes=SHORT_BLOCKED_REGIMES,
            )
            mh_parts.append(part)

        mh_df = pd.concat(mh_parts).sort_index()
        mh_df.to_csv(out / "v22_mh_decisions.csv")
        print(f"  MH сигналы: {(mh_df.get('signal_long', mh_df.get('signal', pd.Series())) == 'BUY').sum()} BUY")

    # -----------------------------------------------------------------------
    # 12. Walk-forward сигналы (для v22_wf, v22_all)
    # -----------------------------------------------------------------------
    wf_df    = pd.DataFrame()
    wf_mh_df = pd.DataFrame()
    if run_wf:
        print("\n=== v22: Module E — Walk-forward сигналы (без MH) ===")
        wf_df = generate_wf_signals(
            base_df.copy(), selected, policy_up, policy_short,
            mh_models_by_cutoff=None,
            retrain_freq=RETRAIN_FREQ_DAYS,
        )
        wf_df.to_csv(out / "v22_wf_decisions.csv")

        if run_mh:
            print("\n=== v22: Module E+D — Walk-forward сигналы + MH ===")
            wf_mh_df = generate_wf_signals(
                mh_df.copy() if not mh_df.empty else base_df.copy(),
                selected, policy_up, policy_short,
                mh_models_by_cutoff=mh_models_by_split,
                retrain_freq=RETRAIN_FREQ_DAYS,
            )
            wf_mh_df.to_csv(out / "v22_wf_mh_decisions.csv")

    # -----------------------------------------------------------------------
    # 13. Подбор ATR параметров (на valid базовых сигналов)
    # -----------------------------------------------------------------------
    print("\n=== v22: Module A — подбор ATR-k (LONG, valid) ===")
    valid_base_df = base_df[base_df["split"] == "valid"].copy()
    atr_k_long, hold_long = select_atr_k(valid_base_df, direction="LONG")

    print("\n=== v22: Module A — подбор ATR-k (SHORT, valid) ===")
    atr_k_short, hold_short = select_atr_k(valid_base_df, direction="SHORT")

    print(f"\n  ATR: LONG k={atr_k_long}, hold={hold_long}d | "
          f"SHORT k={atr_k_short}, hold={hold_short}d")

    # -----------------------------------------------------------------------
    # 14. Бэктест всех 6 вариантов
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("=== v22: бэктест вариантов ===")
    print("=" * 65)

    all_variant_rows = []
    all_risk_metrics = []

    def _run_variant(source_df: pd.DataFrame, label: str, use_atr: bool, use_kelly: bool):
        """Прогоняет один вариант, возвращает сводку и метрики риска."""
        if source_df.empty:
            print(f"\n  [{label}] DataFrame пуст, пропускаем.")
            return

        for s in ["valid", "test", "forward"]:
            bnh = buy_and_hold_return(source_df, s)

            if use_atr:
                trades = backtest_atr_independent(
                    source_df, s,
                    atr_k_long=atr_k_long,   max_hold_long=hold_long,
                    atr_k_short=atr_k_short, max_hold_short=hold_short,
                    cost=COST_PER_TRADE,
                )
            else:
                trades = backtest_strategy_independent(
                    source_df, s,
                    trail_pct_long=TRAIL_PCT_DEFAULT,  max_hold_long=MAX_HOLD_DEFAULT,
                    trail_pct_short=TRAIL_PCT_DEFAULT, max_hold_short=MAX_HOLD_DEFAULT,
                    cost=COST_PER_TRADE,
                )

            if use_kelly and not trades.empty:
                trades = apply_kelly_sizing(trades, normalize=True)

            trades.to_csv(out / f"{label}_{s}_trades.csv", index=False)

            vs = variant_summary(trades, s, bnh, label, kelly_weighted=use_kelly)
            rm = compute_risk_metrics(trades, s, kelly_weighted=use_kelly, label=label)

            all_variant_rows.append(vs)
            all_risk_metrics.append(rm)

            nl = vs["n_long"]
            ns = vs["n_short"]
            nt = vs["total_net"]
            print(f"\n  [{label} | {s.upper()}] n={vs['n_total']} "
                  f"(L={nl}, S={ns})")
            print(f"    total_net={pct(nt)}, BnH={pct(bnh)}, vs_BnH={pct(nt - bnh)}")
            print(f"    sharpe={rm['sharpe_ratio']}, max_dd={pct(rm['max_drawdown']) if rm['max_drawdown'] else '-'}, "
                  f"calmar={rm['calmar_ratio']}, ulcer={rm['ulcer_index']}")

    # v22_base: v21-стиль, fixed trail
    print("\n--- ВАРИАНТ v22_base (v21-сигналы + fixed trail) ---")
    _run_variant(base_df, "v22_base", use_atr=False, use_kelly=False)

    # v22_atr: v21-сигналы + ATR trail
    print("\n--- ВАРИАНТ v22_atr (+ATR stops) ---")
    _run_variant(base_df, "v22_atr", use_atr=True, use_kelly=False)

    # v22_kelly: v21-сигналы + ATR trail + Kelly
    print("\n--- ВАРИАНТ v22_kelly (+ATR +Kelly) ---")
    _run_variant(base_df, "v22_kelly", use_atr=True, use_kelly=True)

    # v22_mh: MH-сигналы + ATR + Kelly
    if run_mh and not mh_df.empty:
        print("\n--- ВАРИАНТ v22_mh (+MH +ATR +Kelly) ---")
        _run_variant(mh_df, "v22_mh", use_atr=True, use_kelly=True)
    else:
        print("\n  [v22_mh пропущен]")

    # v22_wf: WF-сигналы + ATR + Kelly
    if run_wf and not wf_df.empty:
        print("\n--- ВАРИАНТ v22_wf (+WF +ATR +Kelly) ---")
        _run_variant(wf_df, "v22_wf", use_atr=True, use_kelly=True)
    else:
        print("\n  [v22_wf пропущен]")

    # v22_all: WF+MH-сигналы + ATR + Kelly
    if run_wf and run_mh and not wf_mh_df.empty:
        print("\n--- ВАРИАНТ v22_all (+WF +MH +ATR +Kelly) ---")
        _run_variant(wf_mh_df, "v22_all", use_atr=True, use_kelly=True)
    elif run_wf and not wf_df.empty:
        print("\n--- ВАРИАНТ v22_all (+WF +ATR +Kelly, без MH) ---")
        _run_variant(wf_df, "v22_all", use_atr=True, use_kelly=True)
    else:
        print("\n  [v22_all пропущен]")

    # -----------------------------------------------------------------------
    # 15. Сводные таблицы
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("=== v22: СВОДНАЯ ТАБЛИЦА P&L ===")
    print("=" * 65)

    summary_df = pd.DataFrame(all_variant_rows)
    if not summary_df.empty:
        for c in ["total_net", "long_net", "short_net", "buy_and_hold", "vs_bnh"]:
            if c in summary_df.columns:
                summary_df[c] = summary_df[c].apply(pct)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(out / "v22_pnl_summary.csv", index=False)

    print("\n" + "=" * 65)
    print("=== v22: ТАБЛИЦА РИСК-МЕТРИК ===")
    print("=" * 65)
    risk_df = pd.DataFrame(all_risk_metrics)
    if not risk_df.empty:
        print_risk_table(all_risk_metrics)
        risk_df.to_csv(out / "v22_risk_metrics.csv", index=False)

    # -----------------------------------------------------------------------
    # 16. Сравнение v19/v21/v22
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("=== v22: СРАВНЕНИЕ v19 → v21 → v22 ===")
    print("=" * 65)
    def _safe_pct(v) -> str:
        """pct() с защитой от None."""
        if v is None:
            return "-"
        try:
            return pct(float(v))
        except Exception:
            return "-"

    def _rm(label: str, split: str) -> Optional[float]:
        """Ищет total_net_return для варианта в risk_metrics."""
        for r in all_risk_metrics:
            if r.get("split") == split and r.get("label") == label:
                return r.get("total_net_return", None)
        return None

    comp_rows = []
    v19_path = Path("baseline_outputs_v19/v19_backtest_trailing.csv")
    v21_path = Path("baseline_outputs_v21/v21_backtest_independent.csv")
    v19_bt = pd.read_csv(v19_path) if v19_path.exists() else pd.DataFrame()
    v21_bt = pd.read_csv(v21_path) if v21_path.exists() else pd.DataFrame()

    for s in ["valid", "test", "forward"]:
        bnh = buy_and_hold_return(base_df, s)
        r19 = v19_bt[v19_bt["split"] == s].iloc[0] if not v19_bt.empty and (v19_bt["split"] == s).any() else {}
        r21 = v21_bt[v21_bt["split"] == s].iloc[0] if not v21_bt.empty and (v21_bt["split"] == s).any() else {}

        comp_rows.append({
            "split":     s,
            "bnh":       _safe_pct(bnh),
            "v19_long":  _safe_pct(r19.get("sum_net_return", None)) if hasattr(r19, "get") else "-",
            "v21_total": _safe_pct(r21.get("total_net", None))      if hasattr(r21, "get") else "-",
            "v22_base":  _safe_pct(_rm("v22_base",  s)),
            "v22_atr":   _safe_pct(_rm("v22_atr",   s)),
            "v22_kelly": _safe_pct(_rm("v22_kelly", s)),
            "v22_mh":    _safe_pct(_rm("v22_mh",    s)),
            "v22_wf":    _safe_pct(_rm("v22_wf",    s)),
            "v22_all":   _safe_pct(_rm("v22_all",   s)),
        })

    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))
    comp_df.to_csv(out / "v22_comparison_all.csv", index=False)

    # -----------------------------------------------------------------------
    # 17. Сохранение policy JSON
    # -----------------------------------------------------------------------
    policy_params = {
        "version":           "v22",
        "horizon_days":      HORIZON_V22,
        "top_features_n":    TOP_FEATURES_N,
        "up_threshold":      policy_up["up_threshold"],
        "cooldown":          policy_up["cooldown"],
        "trail_pct_default": TRAIL_PCT_DEFAULT,
        "atr_k_long":        atr_k_long,
        "atr_k_short":       atr_k_short,
        "max_hold_long":     hold_long,
        "max_hold_short":    hold_short,
        "kelly_half":        KELLY_HALF,
        "max_kelly_frac":    MAX_KELLY_FRACTION,
        "horizons_multi":    HORIZONS_MULTI if run_mh else [],
        "retrain_freq_days": RETRAIN_FREQ_DAYS if run_wf else 0,
        "not_up_weight_adaptive":  split_weights_up,
        "not_down_weight_adaptive": split_weights_down,
        "down_threshold":    policy_short["down_threshold"],
        "short_cooldown":    policy_short["short_cooldown"],
        "short_blocked_regimes": list(SHORT_BLOCKED_REGIMES),
        "expand_cutoffs":    EXPAND_CUTOFFS,
        "atr_period":        ATR_PERIOD_V22,
    }
    with open(out / "v22_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_params, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # 18. Latest signal card (v22_base сигналы)
    # -----------------------------------------------------------------------
    cards = []
    for s in ["valid", "test", "forward"]:
        d = base_df[base_df["split"] == s].sort_index()
        if d.empty:
            continue
        r = d.iloc[-1]
        cards.append({
            "split":        s,
            "date":         r.name.date(),
            "silver_close": round(float(r.get("silver_close", float("nan"))), 2),
            "signal_long":  r.get("signal_long",  "HOLD"),
            "signal_short": r.get("signal_short", "HOLD"),
            "p_up":         round(float(r.get("p_up",    float("nan"))), 4),
            "p_short":      round(float(r.get("p_short", float("nan"))), 4),
            "trend_regime": r.get("trend_regime", r.get("regime", "")),
            "atr_k_long":   atr_k_long,
            "atr_k_short":  atr_k_short,
        })
    pd.DataFrame(cards).to_csv(out / "v22_latest_signal_cards.csv", index=False)

    # -----------------------------------------------------------------------
    # Итог
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"=== v22 завершён. Результаты: {out} ===")
    print(f"  ATR trailing:  LONG k={atr_k_long}×ATR, SHORT k={atr_k_short}×ATR")
    print(f"  Kelly sizing:  half={KELLY_HALF}, max={MAX_KELLY_FRACTION:.0%}")
    print(f"  Multi-horizon: {HORIZONS_MULTI if run_mh else '[пропущен]'}")
    print(f"  Walk-forward:  freq={RETRAIN_FREQ_DAYS}d {'[пропущен]' if not run_wf else ''}")
    print("  Дашборд: streamlit run dashboard_app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
