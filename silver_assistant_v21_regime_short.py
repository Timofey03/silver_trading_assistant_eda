"""
Silver Trading Assistant v21 — Режимный SHORT + Независимые потоки

Три доработки vs v20:

1. Режимный фильтр SHORT:
   SHORT разрешён ТОЛЬКО в downtrend/sideways.
   В uptrend SHORT заблокирован → устраняет шортирование бычьего тренда 2025.
   Реализовано через проверку trend_regime перед генерацией сигнала.

2. Независимые потоки LONG/SHORT (вместо state machine):
   Каждый BUY-сигнал → независимая LONG-сделка (как в v19).
   Каждый SHORT-сигнал → независимая SHORT-сделка.
   Потоки не блокируют друг друга.
   P&L суммируются (fixed-$1 sizing per trade).
   Следствие: возвращается полный объём сделок из v19 для LONG.

3. Агрессивный поиск SHORT-cooldown:
   Добавляем cooldown=5 дней (было min=7 в v20).
   Больше SHORT-сигналов → шире Wilson CI → возможен подтверждённый edge.

Запуск:
  python silver_assistant_v21_regime_short.py
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
from typing import Dict, FrozenSet, List, Optional, Tuple

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

# ---- Импорты из v20 (DOWN-инфраструктура) ----
try:
    from silver_assistant_v20_directional import (
        add_down_label,
        label_report_directional,
        compute_adaptive_weight_down,
        train_expanding_model_down,
        compute_guardrails_down,
        backtest_strategy_short_only,   # используем для подбора параметров
        purged_cv_down,
        HISTORICAL_DOWN_RATE,
        NOT_DOWN_WEIGHT_BASE,
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
        purged_cv_v18,
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
        buy_and_hold_return,
        purged_walk_forward_splits,
        wilson_ci, pct,
    )
    print("  v14 функции загружены.")
except ImportError as e:
    raise ImportError(f"v14 не найден: {e}")


# ---------------------------------------------------------------------------
# Константы v21
# ---------------------------------------------------------------------------

HORIZON_V21   = HORIZON_V18       # 15 дней
EMBARGO_V21   = EMBARGO_V18       # 15 дней

# Режимы, в которых SHORT ЗАПРЕЩЁН
SHORT_BLOCKED_REGIMES: FrozenSet[str] = frozenset({"uptrend"})

# Расширенный grid cooldown для SHORT (добавлен 5)
SHORT_COOLDOWN_GRID_V21 = [5, 7, 10, 15, 20]
SHORT_THRESHOLD_GRID_V21 = [
    0.42, 0.44, 0.46, 0.48, 0.50,
    0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65,
]


# ---------------------------------------------------------------------------
# 1. SHORT-политика с режимным фильтром
# ---------------------------------------------------------------------------

def apply_policy_short_v21(
    df: pd.DataFrame,
    model_down: "RegimeEnsembleV18",
    feature_cols: List[str],
    down_threshold: float,
    short_cooldown: int,
    blocked_regimes: FrozenSet[str] = SHORT_BLOCKED_REGIMES,
) -> pd.DataFrame:
    """
    Генерирует SHORT-сигналы с режимным фильтром.

    Изменения vs v20.apply_policy_short:
    - Добавлен blocked_regimes фильтр: в uptrend SHORT не выдаётся.
    - Остальная логика идентична.
    """
    out     = df.copy()
    X       = out[feature_cols]
    regimes = _get_regimes(out)
    proba   = model_down.predict_proba(X, regimes)

    down_idx = list(model_down.classes_).index(1)
    p_short  = proba[:, down_idx]

    out["p_short"]      = p_short
    out["signal_short"] = "HOLD"
    out["reason_short"] = "hold"

    # Определяем колонку режима
    regime_col = ("trend_regime" if "trend_regime" in out.columns else
                  ("regime" if "regime" in out.columns else None))

    last_pos = -short_cooldown
    for i in range(len(out)):
        if p_short[i] < down_threshold:
            continue
        if i - last_pos < short_cooldown:
            continue
        # Режимный фильтр
        if blocked_regimes and regime_col is not None:
            regime_val = out.iloc[i][regime_col]
            if regime_val in blocked_regimes:
                continue   # SHORT заблокирован в uptrend
        out.iloc[i, out.columns.get_loc("signal_short")] = "SHORT"
        out.iloc[i, out.columns.get_loc("reason_short")] = (
            f"p_short={p_short[i]:.3f}>={down_threshold}"
        )
        last_pos = i

    return out


def select_policy_short_v21(
    valid_df: pd.DataFrame,
    model_down: "RegimeEnsembleV18",
    feature_cols: List[str],
    blocked_regimes: FrozenSet[str] = SHORT_BLOCKED_REGIMES,
) -> dict:
    """
    Подбирает down_threshold + short_cooldown с:
    - Расширенным cooldown-гридом: [5, 7, 10, 15, 20]
    - Режимным фильтром на этапе отбора
    Критерий: (Wilson CI low - base_down_rate) + бонус за n_signals.
    """
    X       = valid_df[feature_cols]
    regimes = _get_regimes(valid_df)
    proba   = model_down.predict_proba(X, regimes)
    down_idx = list(model_down.classes_).index(1)
    p_short  = proba[:, down_idx]

    labeled        = valid_df[valid_df["tb_label_down"].notna()]
    base_down_rate = float(labeled["tb_label_down"].mean()) if len(labeled) > 0 else HISTORICAL_DOWN_RATE
    tb_down_arr    = valid_df["tb_label_down"].values

    regime_col = ("trend_regime" if "trend_regime" in valid_df.columns else
                  ("regime" if "regime" in valid_df.columns else None))
    regimes_arr = valid_df[regime_col].values if regime_col else None

    best_obj    = -float("inf")
    best_params: dict = {}

    print(f"\n  Поиск SHORT-политики (режим-фильтр={bool(blocked_regimes)}, "
          f"cooldown={SHORT_COOLDOWN_GRID_V21}):")
    print(f"  {'thr':>5}  {'cd':>4}  {'n':>5}  {'prec':>7}  {'wilson_lo':>10}  {'base':>7}  {'obj':>8}")

    for thr in SHORT_THRESHOLD_GRID_V21:
        for cooldown in SHORT_COOLDOWN_GRID_V21:
            sigs, last_pos = [], -cooldown
            for i, p in enumerate(p_short):
                if p >= thr and i - last_pos >= cooldown:
                    if blocked_regimes and regimes_arr is not None:
                        if regimes_arr[i] in blocked_regimes:
                            continue
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
            obj   = (lo - base_down_rate) + 0.005 * min(n, 25) if lift > 0 else -999
            mark  = " ←" if obj > best_obj else ""
            print(f"  {thr:>5.2f}  {cooldown:>4}  {n:>5}  {prec:>7.3f}  "
                  f"{lo:>10.4f}  {base_down_rate:>7.4f}  {obj:>8.4f}{mark}")

            if obj > best_obj:
                best_obj    = obj
                best_params = {"down_threshold": thr, "short_cooldown": cooldown,
                               "n_signals_valid": n, "wilson_low_valid": round(lo, 4),
                               "precision_valid": round(prec, 4)}

    if not best_params:
        print("  WARN: DOWN-edge не найден → SHORT отключены (fallback hold)")
        best_params = {"down_threshold": 0.65, "short_cooldown": 20,
                       "n_signals_valid": 0, "wilson_low_valid": 0.0, "precision_valid": 0.0}

    print(f"\n  >>> SHORT параметры: {best_params}")
    return best_params


# ---------------------------------------------------------------------------
# 2. Независимые потоки LONG + SHORT (без state machine)
# ---------------------------------------------------------------------------

def backtest_strategy_independent(
    df: pd.DataFrame,
    split: str,
    trail_pct_long:  float = TRAIL_PCT_DEFAULT,
    max_hold_long:   int   = MAX_HOLD_DEFAULT,
    trail_pct_short: float = TRAIL_PCT_DEFAULT,
    max_hold_short:  int   = MAX_HOLD_DEFAULT,
    cost: float            = COST_PER_TRADE,
) -> pd.DataFrame:
    """
    Независимые потоки LONG и SHORT без state machine.

    Каждый BUY-сигнал → отдельная LONG-сделка (trailing stop).
    Каждый SHORT-сигнал → отдельная SHORT-сделка (инверт. trailing stop).
    Потоки не блокируют друг друга.
    P&L суммируются (equal $1 allocation per trade).

    LONG:  выход при low ≤ peak × (1 - trail_pct_long)
    SHORT: выход при high ≥ trough × (1 + trail_pct_short)
    """
    d = df[df["split"] == split].sort_index()
    if d.empty:
        return pd.DataFrame()

    has_high = "silver_high" in d.columns
    has_low  = "silver_low"  in d.columns

    all_trades: List[dict] = []

    # ---------- LONG-поток ----------
    buy_dates = d[d["signal_long"] == "BUY"].index.tolist()
    for entry_date in buy_dates:
        entry_pos   = d.index.get_loc(entry_date)
        entry_price = float(d.loc[entry_date, "silver_close"])
        peak        = entry_price
        trail_stop  = entry_price * (1.0 - trail_pct_long)
        exit_price  = entry_price
        exit_idx    = entry_pos
        exit_reason = "max_hold"

        for j in range(1, max_hold_long + 1):
            pos = entry_pos + j
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
        net_ret   = gross_ret - cost
        all_trades.append({
            "direction":    "LONG",
            "signal_date":  entry_date,
            "entry_date":   entry_date,
            "exit_date":    d.index[exit_idx],
            "entry_price":  round(entry_price, 3),
            "exit_price":   round(exit_price,  3),
            "peak_price":   round(peak,        3),
            "trough_price": None,
            "trail_stop":   round(trail_stop,  3),
            "gross_return": round(gross_ret,   6),
            "net_return":   round(net_ret,     6),
            "hold_days":    exit_idx - entry_pos,
            "exit_reason":  exit_reason,
            "tb_label_bin":  d.loc[entry_date, "tb_label_bin"]  if "tb_label_bin"  in d.columns else None,
            "tb_label_down": None,
        })

    # ---------- SHORT-поток ----------
    short_dates = d[d["signal_short"] == "SHORT"].index.tolist()
    for entry_date in short_dates:
        entry_pos   = d.index.get_loc(entry_date)
        entry_price = float(d.loc[entry_date, "silver_close"])
        trough      = entry_price
        trail_stop  = entry_price * (1.0 + trail_pct_short)
        exit_price  = entry_price
        exit_idx    = entry_pos
        exit_reason = "max_hold"

        for j in range(1, max_hold_short + 1):
            pos = entry_pos + j
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

        gross_ret = entry_price / exit_price - 1.0   # положителен при падении
        net_ret   = gross_ret - cost
        all_trades.append({
            "direction":    "SHORT",
            "signal_date":  entry_date,
            "entry_date":   entry_date,
            "exit_date":    d.index[exit_idx],
            "entry_price":  round(entry_price, 3),
            "exit_price":   round(exit_price,  3),
            "peak_price":   None,
            "trough_price": round(trough,      3),
            "trail_stop":   round(trail_stop,  3),
            "gross_return": round(gross_ret,   6),
            "net_return":   round(net_ret,     6),
            "hold_days":    exit_idx - entry_pos,
            "exit_reason":  exit_reason,
            "tb_label_bin":  None,
            "tb_label_down": d.loc[entry_date, "tb_label_down"] if "tb_label_down" in d.columns else None,
        })

    if not all_trades:
        return pd.DataFrame()

    return pd.DataFrame(all_trades).sort_values("entry_date").reset_index(drop=True)


def independent_summary(trades: pd.DataFrame, split: str, bnh: float) -> dict:
    """
    Сводка для independent-потоков.
    LONG и SHORT вклады раздельно + суммарно.
    """
    base = {
        "split": split, "n_long": 0, "n_short": 0, "n_total": 0,
        "long_net": 0.0, "short_net": 0.0, "total_net": 0.0,
        "long_win_rate": None, "short_win_rate": None, "total_win_rate": None,
        "long_pf": None, "short_pf": None,
        "buy_and_hold": round(bnh, 4), "vs_bnh": round(-bnh, 4),
        "avg_hold_long": None, "avg_hold_short": None,
        "short_regime_filter": "uptrend_blocked",
    }
    if trades.empty:
        return base

    longs  = trades[trades["direction"] == "LONG"]
    shorts = trades[trades["direction"] == "SHORT"]

    def _win_rate(t): return round(float((t["net_return"] > 0).mean()), 4) if len(t) else None
    def _pf(t):
        r = t["net_return"].values
        w, l = r[r > 0].sum(), abs(r[r < 0].sum())
        return round(w / l, 3) if l > 0 else (None if w == 0 else float("inf"))

    long_net  = float(longs["net_return"].sum())  if not longs.empty  else 0.0
    short_net = float(shorts["net_return"].sum()) if not shorts.empty else 0.0
    total_net = long_net + short_net

    return {
        "split":          split,
        "n_long":         len(longs),
        "n_short":        len(shorts),
        "n_total":        len(trades),
        "long_net":       round(long_net,  4),
        "short_net":      round(short_net, 4),
        "total_net":      round(total_net, 4),
        "long_win_rate":  _win_rate(longs),
        "short_win_rate": _win_rate(shorts),
        "total_win_rate": _win_rate(trades),
        "long_pf":        _pf(longs),
        "short_pf":       _pf(shorts),
        "buy_and_hold":   round(bnh, 4),
        "vs_bnh":         round(total_net - bnh, 4),
        "avg_hold_long":  round(float(longs["hold_days"].mean()),  1) if not longs.empty  else None,
        "avg_hold_short": round(float(shorts["hold_days"].mean()), 1) if not shorts.empty else None,
        "short_regime_filter": "uptrend_blocked",
    }


# ---------------------------------------------------------------------------
# 3. Подбор trailing для SHORT (тот же grid, но работает на новых сигналах)
# ---------------------------------------------------------------------------

def select_trailing_params_short_v21(
    valid_df: pd.DataFrame,
    trail_pct_grid: list = TRAIL_PCT_GRID,
    max_hold_grid:  list = MAX_HOLD_GRID,
    cost: float          = COST_PER_TRADE,
) -> Tuple[float, int]:
    """Подбирает trail_pct + max_hold для SHORT на valid (Sharpe)."""
    best_sharpe = float("-inf")
    best_trail  = TRAIL_PCT_DEFAULT
    best_hold   = MAX_HOLD_DEFAULT

    print("\n  Поиск trailing-параметров SHORT (valid, с режимным фильтром):")
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

    print(f"\n  >>> SHORT trail: trail_pct={best_trail:.2%}, "
          f"max_hold={best_hold}d, sharpe={best_sharpe:.4f}")
    return best_trail, best_hold


# ---------------------------------------------------------------------------
# 4. Основной pipeline v21
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",   default="2013-01-01")
    ap.add_argument("--end",     default="2099-12-31")
    ap.add_argument("--out-dir", default="baseline_outputs_v21")
    args = ap.parse_args(argv)

    end = min(args.end, pd.Timestamp.today().strftime("%Y-%m-%d"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== v21: режимный SHORT + независимые потоки ===")
    print(f"  SHORT заблокирован в режимах: {SHORT_BLOCKED_REGIMES}")
    print(f"  SHORT cooldown grid: {SHORT_COOLDOWN_GRID_V21}")

    # ---- OHLC (с fallback на кэш v19/v20) ----
    print("\n=== v21: загрузка OHLC ===")
    try:
        df = fetch_ohlc(args.start, end)
        print(f"  OHLC: {len(df)} строк, {df.index[0].date()} — {df.index[-1].date()}")
        _cache_has_features = False
    except Exception as e:
        print(f"  WARN: fetch_ohlc ({e}) → кэш...")
        df = None
        for cp in [
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

    # ---- Макро ----
    if not _cache_has_features:
        print("\n=== v21: загрузка макро-данных ===")
        macro = fetch_macro_yfinance(args.start, end)
        if not macro.empty:
            macro.to_csv(out / "v21_macro_raw.csv")
    else:
        macro = pd.DataFrame()

    # ---- COT ----
    if not _cache_has_features:
        print("\n=== v21: загрузка COT ===")
        cot = fetch_cot_silver(pd.Timestamp(args.start).year, pd.Timestamp(end).year)
        if not cot.empty:
            print(f"  COT: {len(cot)} записей")
            cot.to_csv(out / "v21_cot_raw.csv")
        else:
            print("  COT: недоступны")
            cot = pd.DataFrame()
    else:
        cot = pd.DataFrame()

    # ---- Признаки + Метки ----
    if _cache_has_features:
        print("\n=== v21: кэш — пропускаем feature engineering ===")
        present_cot    = [c for c in COT_FEATURES if c in df.columns]
        macro_features = [c for c in MACRO_FEATURE_NAMES if c in df.columns]
        if "tb_label_bin"  not in df.columns: df = binarize_labels(df)
        if "tb_label_down" not in df.columns: df = add_down_label(df)
        if "split"         not in df.columns: df["split"] = df.index.map(split_name)
    else:
        print("\n=== v21: инженерия признаков ===")
        df = build_features(df, pd.DataFrame())
        df = add_vol_trend_regime(df)
        df = add_macro_features(df, macro)
        if not cot.empty:
            df = merge_cot_to_daily(df, cot, lag_days=COT_RELEASE_LAG)
        present_cot    = [c for c in COT_FEATURES if c in df.columns]
        macro_features = [c for c in MACRO_FEATURE_NAMES if c in df.columns]

        print("\n=== v21: triple-barrier + UP + DOWN метки ===")
        df = add_triple_barrier_labels(df, horizon=HORIZON_V21)
        df = binarize_labels(df)
        df = add_down_label(df)
        df["split"] = df.index.map(split_name)

    df.to_csv(out / "v21_full_data.csv")

    lb3 = label_report_directional(df)
    lb3.to_csv(out / "v21_label_distribution_3class.csv", index=False)
    print("  3-классовое распределение:")
    print(lb3.to_string(index=False))

    # ---- Список признаков ----
    base_features = get_feature_cols(df)
    extra_cot     = [c for c in present_cot if c not in base_features]
    all_features  = list(dict.fromkeys(base_features + macro_features + extra_cot))
    print(f"\n  Всего признаков: {len(all_features)}")

    # ================================================================
    # PASS 1: Отбор признаков (UP-модель, valid-окно)
    # ================================================================
    print("\n=== v21: pass 1 — отбор признаков ===")
    valid_cutoff = pd.Timestamp(EXPAND_CUTOFFS["valid"]) - pd.Timedelta(days=EMBARGO_V21)
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
    imp_df["source"] = imp_df["feature"].apply(
        lambda f: "macro_yf" if f in MACRO_FEATURE_NAMES
                  else ("cot" if f in COT_FEATURES else "base")
    )
    imp_df.to_csv(out / "v21_feature_importance.csv", index=False)
    pd.DataFrame({"feature": selected, "rank": range(1, len(selected)+1)}).to_csv(
        out / "v21_selected_features.csv", index=False
    )
    macro_in_top = [f for f in selected if f in MACRO_FEATURE_NAMES]
    print(f"  Отобрано: {len(selected)} / {len(all_features)}, macro: {macro_in_top}")

    # ================================================================
    # PASS 2a: UP-модели (expanding window)
    # ================================================================
    print("\n=== v21: pass 2a — UP-модели ===")
    split_models_up:  Dict[str, RegimeEnsembleV18] = {}
    split_weights_up: Dict[str, float]              = {}
    for split_key, cutoff in EXPAND_CUTOFFS.items():
        model, wt, _ = train_expanding_model(df, cutoff, selected)
        split_models_up[split_key]  = model
        split_weights_up[split_key] = wt

    # ================================================================
    # PASS 2b: DOWN-модели (expanding window)
    # ================================================================
    print("\n=== v21: pass 2b — DOWN-модели ===")
    split_models_down:  Dict[str, RegimeEnsembleV18] = {}
    split_weights_down: Dict[str, float]              = {}
    for split_key, cutoff in EXPAND_CUTOFFS.items():
        model_d, wt_d, _ = train_expanding_model_down(df, cutoff, selected)
        split_models_down[split_key]  = model_d
        split_weights_down[split_key] = wt_d

    # ================================================================
    # Выбор политик
    # ================================================================
    print("\n=== v21: UP-политика (valid) ===")
    valid_full = df[df["split"] == "valid"].copy()
    policy_up  = select_policy_v18(valid_full, split_models_up["valid"], selected)
    print(f"  UP: {policy_up}")

    print("\n=== v21: SHORT-политика (valid, с режимным фильтром) ===")
    policy_short = select_policy_short_v21(
        valid_full, split_models_down["valid"], selected,
        blocked_regimes=SHORT_BLOCKED_REGIMES,
    )

    # ================================================================
    # Применение политик
    # ================================================================
    print("\n=== v21: применение политик ===")
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
        part["signal_long"] = part["signal"]

        # SHORT-сигналы с режимным фильтром
        part = apply_policy_short_v21(
            part, model_down_s, selected,
            policy_short["down_threshold"], policy_short["short_cooldown"],
            blocked_regimes=SHORT_BLOCKED_REGIMES,
        )
        signal_parts.append(part)

    all_df = pd.concat(signal_parts).sort_index()
    all_df.to_csv(out / "v21_decisions_all.csv")

    # Диагностика режимного фильтра
    for s in ["valid", "test", "forward"]:
        d = all_df[all_df["split"] == s]
        regime_col = "trend_regime" if "trend_regime" in d.columns else "regime"
        if regime_col in d.columns:
            regime_dist  = d[regime_col].value_counts().to_dict()
            short_regimes = d[d["signal_short"] == "SHORT"][regime_col].value_counts().to_dict()
            print(f"  [{s}] режимы: {regime_dist} | SHORT по режимам: {short_regimes}")

    # ================================================================
    # Guardrails
    # ================================================================
    print("\n=== v21: guardrails LONG ===")
    all_df_long = all_df.copy()
    all_df_long["signal"] = all_df_long["signal_long"]
    grd_long = pd.DataFrame([
        compute_guardrails_binary(all_df_long, s)
        for s in ["valid", "test", "forward"]
    ])
    grd_long.to_csv(out / "v21_guardrails_long.csv", index=False)
    cols = ["split", "n_signals", "correct_over_n", "precision",
            "wilson_95_low", "base_up_rate", "lift_vs_base", "warning"]
    print(grd_long[[c for c in cols if c in grd_long.columns]].to_string(index=False))

    print("\n=== v21: guardrails SHORT (после режимного фильтра) ===")
    grd_short = pd.DataFrame([
        compute_guardrails_down(all_df, s)
        for s in ["valid", "test", "forward"]
    ])
    grd_short.to_csv(out / "v21_guardrails_short.csv", index=False)
    cols_s = ["split", "n_signals", "correct_over_n", "precision",
              "wilson_95_low", "base_down_rate", "lift_vs_base", "warning"]
    print(grd_short[[c for c in cols_s if c in grd_short.columns]].to_string(index=False))

    # ================================================================
    # Подбор trailing параметров
    # ================================================================
    print("\n=== v21: trailing LONG (valid) ===")
    valid_signal_df = all_df[all_df["split"] == "valid"].copy()
    valid_signal_df["signal"] = valid_signal_df["signal_long"]
    best_trail_long, best_hold_long = select_trailing_params(valid_signal_df)

    print("\n=== v21: trailing SHORT (valid, режим-фильтр) ===")
    best_trail_short, best_hold_short = select_trailing_params_short_v21(valid_signal_df)

    print(f"\n  LONG:  trail_pct={best_trail_long:.2%}, max_hold={best_hold_long}d")
    print(f"  SHORT: trail_pct={best_trail_short:.2%}, max_hold={best_hold_short}d")

    # ================================================================
    # Independent бэктест (главный результат v21)
    # ================================================================
    print("\n=== v21: independent бэктест (LONG + SHORT независимо) ===")
    ind_bt_rows = []
    for s in ["valid", "test", "forward"]:
        trades = backtest_strategy_independent(
            all_df, s,
            trail_pct_long=best_trail_long,   max_hold_long=best_hold_long,
            trail_pct_short=best_trail_short, max_hold_short=best_hold_short,
            cost=COST_PER_TRADE,
        )
        trades.to_csv(out / f"{s}_trades_v21.csv", index=False)
        all_df[all_df["split"] == s].to_csv(out / f"{s}_decisions_v21.csv")

        bnh     = buy_and_hold_return(all_df, s)
        summary = independent_summary(trades, s, bnh)
        ind_bt_rows.append(summary)

        nl, ns = summary["n_long"], summary["n_short"]
        print(f"\n  [{s.upper()}]  n={summary['n_total']} "
              f"(long={nl}, short={ns})")
        print(f"    LONG:  net={pct(summary['long_net'])}, "
              f"win={pct(summary['long_win_rate'])}, "
              f"pf={summary['long_pf']}, avg_hold={summary['avg_hold_long']}d")
        if ns > 0:
            print(f"    SHORT: net={pct(summary['short_net'])}, "
                  f"win={pct(summary['short_win_rate'])}, "
                  f"pf={summary['short_pf']}, avg_hold={summary['avg_hold_short']}d")
        else:
            print(f"    SHORT: нет сигналов (все в uptrend, заблокированы)")
        print(f"    TOTAL: net={pct(summary['total_net'])}, "
              f"BnH={pct(bnh)}, vs_BnH={pct(summary['vs_bnh'])}")

    ind_bt_df = pd.DataFrame(ind_bt_rows)
    ind_bt_df.to_csv(out / "v21_backtest_independent.csv", index=False)

    # ================================================================
    # Сравнение v19 / v20 / v21
    # ================================================================
    v19_trail  = Path("baseline_outputs_v19/v19_backtest_trailing.csv")
    v20_dir_   = Path("baseline_outputs_v20/v20_backtest_directional.csv")
    comp_rows  = []
    v19_bt_df  = pd.read_csv(v19_trail)  if v19_trail.exists()  else pd.DataFrame()
    v20_bt_df  = pd.read_csv(v20_dir_)   if v20_dir_.exists()   else pd.DataFrame()

    for s in ["valid", "test", "forward"]:
        bnh = buy_and_hold_return(all_df, s)
        r19 = v19_bt_df[v19_bt_df["split"] == s].iloc[0] if not v19_bt_df.empty else {}
        r20 = v20_bt_df[v20_bt_df["split"] == s].iloc[0] if not v20_bt_df.empty else {}
        r21 = ind_bt_df[ind_bt_df["split"] == s].iloc[0] if not ind_bt_df.empty else {}

        comp_rows.append({
            "split":           s,
            "bnh":             pct(bnh),
            "v19_long_only":   pct(r19.get("sum_net_return", None)) if r19 is not None and hasattr(r19, 'get') else "-",
            "v20_state_mach":  pct(r20.get("sum_net_return", None)) if r20 is not None and hasattr(r20, 'get') else "-",
            "v21_independent": pct(r21.get("total_net", None))      if r21 is not None and hasattr(r21, 'get') else "-",
            "v21_long":        pct(r21.get("long_net",  None))      if r21 is not None and hasattr(r21, 'get') else "-",
            "v21_short":       pct(r21.get("short_net", None))      if r21 is not None and hasattr(r21, 'get') else "-",
            "v21_n_short":     int(r21.get("n_short",   0))         if r21 is not None and hasattr(r21, 'get') else "-",
        })

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out / "v21_vs_v19_v20.csv", index=False)
    print("\n=== Сравнение v19 / v20 / v21 ===")
    print(comp_df.to_string(index=False))

    # ================================================================
    # Guardrails comparison SHORT
    # ================================================================
    v20_gs = Path("baseline_outputs_v20/v20_guardrails_short.csv")
    if v20_gs.exists():
        g20 = pd.read_csv(v20_gs)
        short_comp = []
        for s in ["valid", "test", "forward"]:
            r20s = g20[g20["split"] == s]
            r21s = grd_short[grd_short["split"] == s]
            short_comp.append({
                "split":           s,
                "v20_n_short":     r20s["n_signals"].values[0]   if not r20s.empty else "-",
                "v21_n_short":     r21s["n_signals"].values[0]   if not r21s.empty else "-",
                "v20_prec_short":  pct(r20s["precision"].values[0]) if not r20s.empty else "-",
                "v21_prec_short":  pct(r21s["precision"].values[0]) if not r21s.empty else "-",
                "v20_lift":        pct(r20s["lift_vs_base"].values[0]) if not r20s.empty else "-",
                "v21_lift":        pct(r21s["lift_vs_base"].values[0]) if not r21s.empty else "-",
                "v21_edge":        "✅" if (not r21s.empty and r21s["warning"].values[0] == "OK") else "❌",
            })
        short_comp_df = pd.DataFrame(short_comp)
        short_comp_df.to_csv(out / "v21_short_guardrails_compare.csv", index=False)
        print("\n=== Guardrails SHORT: v20 vs v21 ===")
        print(short_comp_df.to_string(index=False))

    # ================================================================
    # Policy JSON + signal cards + purged CV
    # ================================================================
    policy_params = {
        "version":           "v21",
        "horizon_days":      HORIZON_V21,
        "top_features_n":    TOP_FEATURES_N,
        "up_threshold":      policy_up["up_threshold"],
        "cooldown":          policy_up["cooldown"],
        "trail_pct_long":    best_trail_long,
        "max_hold_long":     best_hold_long,
        "not_up_weight_adaptive": split_weights_up,
        "down_threshold":    policy_short["down_threshold"],
        "short_cooldown":    policy_short["short_cooldown"],
        "trail_pct_short":   best_trail_short,
        "max_hold_short":    best_hold_short,
        "not_down_weight_adaptive": split_weights_down,
        "short_blocked_regimes": list(SHORT_BLOCKED_REGIMES),
        "short_cooldown_grid": SHORT_COOLDOWN_GRID_V21,
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
    with open(out / "v21_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_params, f, indent=2, ensure_ascii=False)

    cards = []
    for s in ["valid", "test", "forward"]:
        d = all_df[all_df["split"] == s].sort_index()
        if d.empty: continue
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
            "short_regime_filter": "uptrend_blocked",
        })
    pd.DataFrame(cards).to_csv(out / "v21_latest_signal_cards.csv", index=False)

    print("\n=== v21: purged CV DOWN (baseline=0.50) ===")
    wf_down = purged_cv_down(df, selected)
    wf_down.to_csv(out / "v21_purged_wf_cv_down.csv", index=False)
    if not wf_down.empty:
        mean_ba = wf_down["balanced_acc"].mean()
        n_above = (wf_down["balanced_acc"] > 0.50).sum()
        print(f"  Фолдов: {len(wf_down)}, mean BA: {mean_ba:.3f}, "
              f"выше 0.50: {n_above}/{len(wf_down)}")

    print(f"\n=== v21 завершён. Результаты: {out} ===")
    print(f"  LONG:  up_thr={policy_up['up_threshold']}, "
          f"trail={best_trail_long:.2%}, hold={best_hold_long}d")
    print(f"  SHORT: down_thr={policy_short['down_threshold']}, "
          f"trail={best_trail_short:.2%}, hold={best_hold_short}d, "
          f"blocked={SHORT_BLOCKED_REGIMES}")
    print("  Дашборд: streamlit run dashboard_app.py")


if __name__ == "__main__":
    main()
