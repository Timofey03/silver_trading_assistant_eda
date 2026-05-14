"""
Silver Trading Assistant v19 — Trailing Stop (динамический выход)

Изменения vs v18:

ДИАГНОЗ: Структурный потолок доходности.
Горизонт=15 дней → стратегия принудительно закрывает позиции,
даже если серебро продолжает расти (2025: BnH +190%, стратегия +15%).
Модель видит правильные входы, но упускает трендовую доходность.

1. Trailing stop вместо фиксированного горизонта:
   - Держать позицию, пока цена не упадёт trail_pct% от пика
   - Максимальный удержания max_hold торговых дней (страховка от боковика)
   - Параметры подбираются на valid-наборе по критерию Шарпа
   - Используем OHLC: выход триггерится по silver_low ≤ peak * (1 - trail_pct)

2. Весь ML-пайплайн v18 сохранён без изменений:
   - Расширяющееся окно (expanding window)
   - Адаптивный NOT_UP_weight (adaptive class weight)
   - Экспоненциальный временной декай (time decay)
   - Макро-признаки yfinance (^TNX, ^IRX, TIP, HYG)
   - RegimeEnsembleV18 (режимная ансамблевая модель)

3. Сохраняются оба бэктеста:
   - fixed-horizon (как в v18, для сравнения)
   - trailing-stop (новый)

Запуск:
  python silver_assistant_v19_trailing.py
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
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss
    from sklearn.inspection import permutation_importance
except ImportError:
    raise ImportError("pip install scikit-learn>=1.3.0")

# ---- Импорты из v18 (весь ML-пайплайн) ----
try:
    from silver_assistant_v18_adaptive import (
        RegimeEnsembleV18,
        compute_sample_weights, compute_adaptive_weight,
        train_expanding_model, select_policy_v18, purged_cv_v18,
        EXPAND_CUTOFFS, HALFLIFE_YEARS, RECENT_WEIGHT_YEARS, HISTORICAL_UP_RATE,
        HORIZON_V18, EMBARGO_V18,
    )
    print("  v18 функции загружены.")
except ImportError as e:
    raise ImportError(f"v18 не найден: {e}")

# ---- Импорты из v17 ----
try:
    from silver_assistant_v17_fred import (
        fetch_macro_yfinance, add_macro_features,
        MACRO_FEATURE_NAMES, MACRO_TICKERS,
    )
    print("  v17 функции загружены.")
except ImportError as e:
    raise ImportError(f"v17 не найден: {e}")

# ---- Импорты из v16 ----
try:
    from silver_assistant_v16_binary import (
        RegimeEnsembleBinary,
        binarize_labels, label_report_binary,
        select_top_features, evaluate_split_v16,
        compute_guardrails_binary,
        apply_policy_v16,
        _get_regimes, split_name,
        TOP_FEATURES_N, NOT_UP_WEIGHT,
        MAX_DEPTH, MAX_LEAF_NODES, LEARNING_RATE,
        MIN_SAMP_LEAF, L2_REG, N_ITER_NO_CHANGE,
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
        backtest_strategy, buy_and_hold_return, backtest_summary,
        purged_walk_forward_splits,
        wilson_ci, pct,
    )
    print("  v14 функции загружены.")
except ImportError as e:
    raise ImportError(f"v14 не найден: {e}")


# ---------------------------------------------------------------------------
# Гиперпараметры v19
# ---------------------------------------------------------------------------

HORIZON_V19        = HORIZON_V18        # 15 дней (для ML-меток, без изменений)
EMBARGO_V19        = EMBARGO_V18        # 15 дней

# Trailing stop — диапазон поиска
TRAIL_PCT_GRID     = [0.05, 0.07, 0.08, 0.10, 0.12]   # 5-12% от пика
MAX_HOLD_GRID      = [30,   45,   60,   90]             # торговых дней

TRAIL_PCT_DEFAULT  = 0.08   # fallback, если valid-поиск не нашёл лучших
MAX_HOLD_DEFAULT   = 45

COST_PER_TRADE     = 0.0005   # 5 bp на вход+выход


# ---------------------------------------------------------------------------
# 1. Trailing-stop бэктест
# ---------------------------------------------------------------------------

def backtest_strategy_trailing(
    df: pd.DataFrame,
    split: str,
    trail_pct: float = TRAIL_PCT_DEFAULT,
    max_hold: int    = MAX_HOLD_DEFAULT,
    cost: float      = COST_PER_TRADE,
) -> pd.DataFrame:
    """
    Trailing-stop бэктест на основе сигналов BUY из apply_policy_v16.

    Для каждого входа:
      - Держим позицию, пока silver_low не пробьёт peak * (1 - trail_pct)
        (используем внутридневной Low, если доступен, иначе Close)
      - Принудительный выход на max_hold торговый день
      - Цена выхода: min(Close, trail_stop) при срабатывании трейлера

    Возвращает DataFrame с колонками:
      signal_date, entry_date, exit_date, entry_price, exit_price,
      peak_price, trail_stop, gross_return, net_return,
      hold_days, exit_reason, tb_label_bin
    """
    d = df[df["split"] == split].sort_index()
    if d.empty:
        return pd.DataFrame()

    has_high = "silver_high" in d.columns
    has_low  = "silver_low"  in d.columns
    if not (has_high and has_low):
        print(f"  WARN [{split}]: OHLC High/Low недоступны → используем Close")

    buy_dates = d[d["signal"] == "BUY"].index.tolist()
    if not buy_dates:
        return pd.DataFrame()

    trades = []
    for entry_date in buy_dates:
        entry_pos    = d.index.get_loc(entry_date)
        entry_price  = float(d.loc[entry_date, "silver_close"])
        peak         = entry_price
        trail_stop   = entry_price * (1.0 - trail_pct)
        exit_price   = entry_price
        exit_date    = entry_date
        exit_reason  = "max_hold"

        for j in range(1, max_hold + 1):
            pos = entry_pos + j
            if pos >= len(d):
                break

            row = d.iloc[pos]
            hi  = float(row["silver_high"]) if has_high else float(row["silver_close"])
            lo  = float(row["silver_low"])  if has_low  else float(row["silver_close"])
            cl  = float(row["silver_close"])

            # Обновляем пик и трейлер
            if hi > peak:
                peak        = hi
                trail_stop  = peak * (1.0 - trail_pct)

            # Проверяем срабатывание трейлера
            if lo <= trail_stop:
                exit_price  = min(cl, trail_stop)   # консервативная оценка
                exit_date   = d.index[pos]
                exit_reason = "trail_stop"
                break

            # Конец дня без триггера → обновляем exit как текущий Close
            exit_price = cl
            exit_date  = d.index[pos]

        gross_ret = exit_price / entry_price - 1.0
        net_ret   = gross_ret - cost
        hold_days = d.index.get_loc(exit_date) - entry_pos

        trades.append({
            "signal_date":  entry_date,
            "entry_date":   entry_date,
            "exit_date":    exit_date,
            "entry_price":  round(entry_price, 3),
            "exit_price":   round(exit_price,  3),
            "peak_price":   round(peak,         3),
            "trail_stop":   round(trail_stop,   3),
            "gross_return": round(gross_ret,     6),
            "net_return":   round(net_ret,       6),
            "hold_days":    hold_days,
            "exit_reason":  exit_reason,
            "tb_label_bin": d.loc[entry_date, "tb_label_bin"]
                            if "tb_label_bin" in d.columns else None,
        })

    return pd.DataFrame(trades)


def trailing_summary(trades: pd.DataFrame, split: str, bnh: float) -> dict:
    """
    Расширенный отчёт для trailing-stop бэктеста.
    Совместим с backtest_summary() из v14 + дополнительные метрики.
    """
    if trades.empty:
        return {
            "split": split, "n_trades": 0,
            "sum_net_return": 0.0, "win_rate": None,
            "profit_factor": None, "buy_and_hold": round(bnh, 4),
            "vs_bnh": None, "avg_hold_days": None,
            "trail_exits": 0, "max_hold_exits": 0,
            "trail_exit_pct": None, "avg_peak_gain": None,
        }

    rets    = trades["net_return"].values
    gross_r = trades["gross_return"].values
    wins    = rets[rets > 0]
    losses  = rets[rets < 0]

    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

    trail_exits   = int((trades["exit_reason"] == "trail_stop").sum())
    max_hold_exit = int((trades["exit_reason"] == "max_hold").sum())
    n             = len(trades)

    avg_peak_gain = float(
        ((trades["peak_price"] / trades["entry_price"] - 1.0).mean())
    ) if "peak_price" in trades.columns else None

    return {
        "split":           split,
        "n_trades":        n,
        "sum_net_return":  round(float(rets.sum()), 4),
        "win_rate":        round(float((rets > 0).mean()), 4),
        "profit_factor":   round(pf, 3) if np.isfinite(pf) else None,
        "buy_and_hold":    round(bnh, 4),
        "vs_bnh":          round(float(rets.sum()) - bnh, 4),
        "avg_hold_days":   round(float(trades["hold_days"].mean()), 1),
        "trail_exits":     trail_exits,
        "max_hold_exits":  max_hold_exit,
        "trail_exit_pct":  round(trail_exits / n, 3) if n > 0 else None,
        "avg_peak_gain":   round(avg_peak_gain, 4) if avg_peak_gain is not None else None,
    }


# ---------------------------------------------------------------------------
# 2. Подбор параметров trailing stop на valid-наборе
# ---------------------------------------------------------------------------

def _sharpe_from_trades(trades: pd.DataFrame) -> float:
    """Простой Шарп из сделок: mean / std net_return (nan если <3 сделок)."""
    if trades.empty or len(trades) < 3:
        return float("-inf")
    rets = trades["net_return"].values
    std  = rets.std()
    if std < 1e-9:
        return float("-inf")
    return float(rets.mean() / std)


def select_trailing_params(
    valid_df: pd.DataFrame,
    trail_pct_grid: list = TRAIL_PCT_GRID,
    max_hold_grid:  list = MAX_HOLD_GRID,
    cost: float          = COST_PER_TRADE,
) -> Tuple[float, int]:
    """
    Перебирает (trail_pct, max_hold) на valid-наборе.
    Критерий: максимальный Sharpe ratio по сделкам.
    Требует >= 3 сделок; при ничье — выбирает большее trail_pct.

    Возвращает (best_trail_pct, best_max_hold).
    """
    best_sharpe = float("-inf")
    best_trail  = TRAIL_PCT_DEFAULT
    best_hold   = MAX_HOLD_DEFAULT

    print("\n  Поиск trailing-параметров на valid:")
    print(f"  {'trail_pct':>10}  {'max_hold':>9}  {'n_trades':>9}  {'sharpe':>8}")

    for trail_pct in trail_pct_grid:
        for max_hold in max_hold_grid:
            trades = backtest_strategy_trailing(
                valid_df, "valid", trail_pct=trail_pct, max_hold=max_hold, cost=cost
            )
            sharpe = _sharpe_from_trades(trades)
            n      = len(trades) if not trades.empty else 0
            mark   = " ← best" if sharpe > best_sharpe and n >= 3 else ""
            print(f"  {trail_pct:>10.2%}  {max_hold:>9}  {n:>9}  "
                  f"{sharpe:>8.4f}{mark}")
            if sharpe > best_sharpe and n >= 3:
                best_sharpe = sharpe
                best_trail  = trail_pct
                best_hold   = max_hold

    print(f"\n  >>> Лучшие параметры: trail_pct={best_trail:.2%}, "
          f"max_hold={best_hold}, sharpe={best_sharpe:.4f}")
    return best_trail, best_hold


# ---------------------------------------------------------------------------
# 3. Основной pipeline v19
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",   default="2013-01-01")
    ap.add_argument("--end",     default="2099-12-31")
    ap.add_argument("--out-dir", default="baseline_outputs_v19")
    args = ap.parse_args(argv)

    end = min(args.end, pd.Timestamp.today().strftime("%Y-%m-%d"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== v19: trailing stop + expanding window + adaptive weight ===")
    print(f"  depth={MAX_DEPTH}, leaf={MAX_LEAF_NODES}, lr={LEARNING_RATE}, "
          f"l2={L2_REG}, halflife={HALFLIFE_YEARS}y, top_feat={TOP_FEATURES_N}")
    print(f"  trail_pct_grid={TRAIL_PCT_GRID}, max_hold_grid={MAX_HOLD_GRID}")

    # ---- OHLC ----
    print("\n=== v19: загрузка OHLC ===")
    df = fetch_ohlc(args.start, end)
    print(f"  OHLC: {len(df)} строк, {df.index[0].date()} — {df.index[-1].date()}")
    ohlc_cols = [c for c in ["silver_open", "silver_high", "silver_low", "silver_close"]
                 if c in df.columns]
    print(f"  OHLC колонки: {ohlc_cols}")

    # ---- Макро (yfinance) ----
    print("\n=== v19: загрузка макро-данных ===")
    macro = fetch_macro_yfinance(args.start, end)
    if not macro.empty:
        macro.to_csv(out / "v19_macro_raw.csv")
    else:
        print("  WARN: макро недоступны")

    # ---- COT ----
    print("\n=== v19: загрузка COT ===")
    cot = fetch_cot_silver(pd.Timestamp(args.start).year, pd.Timestamp(end).year)
    if cot.empty:
        print("  COT: недоступны")
    else:
        print(f"  COT: {len(cot)} записей")
        cot.to_csv(out / "v19_cot_raw.csv")

    # ---- Признаки ----
    print("\n=== v19: инженерия признаков ===")
    df = build_features(df, pd.DataFrame())
    df = add_vol_trend_regime(df)
    df = add_macro_features(df, macro)

    if not cot.empty:
        df = merge_cot_to_daily(df, cot, lag_days=COT_RELEASE_LAG)
        present_cot = [c for c in COT_FEATURES if c in df.columns]
        print(f"  COT признаки: {present_cot}")
    else:
        present_cot = []

    # ---- Labels ----
    print("\n=== v19: triple-barrier + бинаризация ===")
    df = add_triple_barrier_labels(df, horizon=HORIZON_V19)
    df = binarize_labels(df)
    df["split"] = df.index.map(split_name)
    df.to_csv(out / "v19_full_data.csv")

    # Проверяем OHLC в финальном DataFrame
    has_ohlc = ("silver_high" in df.columns) and ("silver_low" in df.columns)
    print(f"  OHLC High/Low доступны: {has_ohlc}")

    lb = label_report_binary(df)
    lb.to_csv(out / "v19_label_distribution.csv", index=False)
    print("  UP/NOT_UP распределение:")
    print(lb.to_string(index=False))

    # ---- Список всех признаков ----
    base_features  = get_feature_cols(df)
    macro_features = [c for c in MACRO_FEATURE_NAMES if c in df.columns]
    extra_cot      = [c for c in present_cot if c not in base_features]
    all_features: List[str] = list(dict.fromkeys(base_features + macro_features + extra_cot))
    print(f"\n  Всего признаков: {len(all_features)} "
          f"(base={len(base_features)}, macro={len(macro_features)}, COT={len(extra_cot)})")

    # ---- PASS 1: отбор признаков (на valid-окне для fair selection) ----
    print("\n=== v19: pass 1 — отбор признаков (valid-модель 2013–2022) ===")
    valid_cutoff = pd.Timestamp(EXPAND_CUTOFFS["valid"]) - pd.Timedelta(days=EMBARGO_V19)
    p1_train     = df[(df.index <= valid_cutoff) & df["tb_label_bin"].notna()].copy()
    p1_valid     = df[(df["split"] == "valid") & df["tb_label_bin"].notna()].copy()

    X_p1_tr   = p1_train[all_features]
    y_p1_tr   = p1_train["tb_label_bin"].values.astype(int)
    r_p1_tr   = _get_regimes(p1_train)
    X_p1_val  = p1_valid[all_features]
    y_p1_val  = p1_valid["tb_label_bin"].values.astype(int)
    sw_p1     = compute_sample_weights(p1_train)

    print(f"  Pass-1 train: {len(X_p1_tr)} строк (UP={y_p1_tr.mean():.3f})")
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
    imp_df.to_csv(out / "v19_feature_importance.csv", index=False)
    pd.DataFrame({"feature": selected, "rank": range(1, len(selected)+1)}).to_csv(
        out / "v19_selected_features.csv", index=False
    )
    macro_in_top = [f for f in selected if f in MACRO_FEATURE_NAMES]
    print(f"  Отобрано: {len(selected)} из {len(all_features)}, "
          f"macro в top: {len(macro_in_top)} → {macro_in_top}")

    # ---- PASS 2: адаптивные модели по расширяющемуся окну ----
    print("\n=== v19: pass 2 — адаптивные модели (expanding window) ===")
    split_models:  Dict[str, RegimeEnsembleV18] = {}
    split_weights: Dict[str, float]             = {}

    for split_key, cutoff in EXPAND_CUTOFFS.items():
        print(f"\n  [{split_key.upper()}] — expanding до {cutoff}")
        model, wt, _ = train_expanding_model(df, cutoff, selected)
        split_models[split_key]  = model
        split_weights[split_key] = wt

    # ---- Выбор политики на valid (valid-модель) ----
    print("\n=== v19: выбор ML-политики (valid-модель, 2023) ===")
    valid_full    = df[df["split"] == "valid"].copy()
    policy_params = select_policy_v18(valid_full, split_models["valid"], selected)
    print(f"  Параметры: {policy_params}")

    # ---- Применение политики (каждый split — своя модель) ----
    print("\n=== v19: применение политики ===")
    signal_parts = []
    for split_key in ["train", "valid", "test", "forward"]:
        split_df = df[df["split"] == split_key].copy()
        if split_df.empty:
            continue
        model_for_split = split_models.get(split_key, split_models["valid"])
        part = apply_policy_v16(
            split_df, model_for_split, selected,
            policy_params["up_threshold"],
            policy_params["cooldown"],
        )
        signal_parts.append(part)

    all_df = pd.concat(signal_parts).sort_index()
    all_df.to_csv(out / "v19_decisions_all.csv")

    # ---- Метрики (каждый split — своя модель) ----
    print("\n=== v19: метрики (бинарные, baseline=0.50) ===")
    cls_rows = []
    for split_key in ["train", "valid", "test", "forward"]:
        model_for_split = split_models.get(split_key, split_models["valid"])
        row = evaluate_split_v16(df, split_key, model_for_split, selected)
        cls_rows.append(row)
    cls_df = pd.DataFrame(cls_rows)
    cls_df.to_csv(out / "v19_classifier_metrics.csv", index=False)
    cols = [c for c in ["split", "n", "balanced_accuracy", "auc", "brier"]
            if c in cls_df.columns]
    print(cls_df[cols].to_string(index=False))

    # ---- Guardrails ----
    print("\n=== v19: guardrails ===")
    grd_rows   = [compute_guardrails_binary(all_df, s) for s in ["valid", "test", "forward"]]
    guardrails = pd.DataFrame(grd_rows)
    guardrails.to_csv(out / "v19_guardrails.csv", index=False)
    cols = ["split", "n_signals", "correct_over_n", "precision",
            "wilson_95_low", "base_up_rate", "lift_vs_base", "warning"]
    cols = [c for c in cols if c in guardrails.columns]
    print(guardrails[cols].to_string(index=False))

    # ---- Adaptive weights summary ----
    print("\n  Адаптивные веса по окнам:")
    for sp, wt in split_weights.items():
        cutoff_date = EXPAND_CUTOFFS[sp]
        cutoff_ts   = pd.Timestamp(cutoff_date) - pd.Timedelta(days=EMBARGO_V19)
        train_tmp   = df[(df.index <= cutoff_ts) & df["tb_label_bin"].notna()]
        if len(train_tmp) >= 50:
            recent_cut = train_tmp.index.max() - pd.Timedelta(days=RECENT_WEIGHT_YEARS * 365)
            recent_tmp = train_tmp[train_tmp.index >= recent_cut]
            recent_up  = recent_tmp["tb_label_bin"].mean() if len(recent_tmp) >= 50 else float("nan")
        else:
            recent_up = float("nan")
        print(f"    {sp:8s}: NOT_UP_w={wt:.2f}, recent_2y_UP={recent_up:.3f}, "
              f"train_n={len(train_tmp)}")

    # ================================================================
    # TRAILING STOP BACKTEST
    # ================================================================
    print("\n=== v19: подбор trailing-параметров (valid) ===")
    valid_signal_df = all_df[all_df["split"] == "valid"].copy()
    best_trail_pct, best_max_hold = select_trailing_params(valid_signal_df)

    # ---- Бэктест trailing (все splits) ----
    print("\n=== v19: trailing-stop бэктест ===")
    print(f"  Параметры: trail_pct={best_trail_pct:.2%}, max_hold={best_max_hold}")

    trail_bt_rows   = []
    trail_trade_dfs = {}
    for s in ["valid", "test", "forward"]:
        trades  = backtest_strategy_trailing(
            all_df, s,
            trail_pct=best_trail_pct,
            max_hold=best_max_hold,
            cost=COST_PER_TRADE,
        )
        trades.to_csv(out / f"{s}_trades_trailing_v19.csv", index=False)
        all_df[all_df["split"] == s].to_csv(out / f"{s}_decisions_v19.csv")

        bnh     = buy_and_hold_return(all_df, s)
        summary = trailing_summary(trades, s, bnh)
        trail_bt_rows.append(summary)
        trail_trade_dfs[s] = trades

        # Удобная печать
        n = summary["n_trades"]
        print(f"\n  [{s.upper()}]  n={n}, "
              f"net_sum={pct(summary['sum_net_return'])}, "
              f"win_rate={pct(summary['win_rate']) if summary['win_rate'] else '-'}, "
              f"avg_hold={summary['avg_hold_days']}d, "
              f"trail_exits={summary['trail_exits']}/{n}, "
              f"BnH={pct(bnh)}, vs_BnH={pct(summary['vs_bnh']) if summary['vs_bnh'] else '-'}")
        if not trades.empty:
            print(f"    avg_peak_gain={pct(summary.get('avg_peak_gain'))}, "
                  f"pf={summary['profit_factor']}")

    trail_bt_df = pd.DataFrame(trail_bt_rows)
    trail_bt_df.to_csv(out / "v19_backtest_trailing.csv", index=False)

    # ---- Бэктест fixed (для сравнения v18→v19) ----
    print("\n=== v19: fixed-horizon бэктест (для сравнения с trailing) ===")
    fixed_bt_rows = []
    for s in ["valid", "test", "forward"]:
        trades  = backtest_strategy(all_df, s, HORIZON_V19)
        trades.to_csv(out / f"{s}_trades_fixed_v19.csv", index=False)
        bnh     = buy_and_hold_return(all_df, s)
        summary = backtest_summary(trades, s, bnh)
        fixed_bt_rows.append(summary)
    fixed_bt_df = pd.DataFrame(fixed_bt_rows)
    fixed_bt_df.to_csv(out / "v19_backtest_fixed.csv", index=False)
    cols = [c for c in ["split", "n_trades", "sum_net_return", "win_rate",
                         "profit_factor", "buy_and_hold", "vs_bnh"]
            if c in fixed_bt_df.columns]
    print(fixed_bt_df[cols].to_string(index=False))

    # ---- Сравнение trailing vs fixed ----
    print("\n=== Сравнение trailing vs fixed (v19) ===")
    comp_rows = []
    for s in ["valid", "test", "forward"]:
        ft = fixed_bt_df[fixed_bt_df["split"] == s].iloc[0] if not fixed_bt_df.empty else {}
        tr = trail_bt_df[trail_bt_df["split"] == s].iloc[0] if not trail_bt_df.empty else {}
        comp_rows.append({
            "split":              s,
            "fixed_sum_net":      ft.get("sum_net_return", None),
            "trail_sum_net":      tr.get("sum_net_return", None),
            "fixed_win_rate":     ft.get("win_rate",       None),
            "trail_win_rate":     tr.get("win_rate",       None),
            "fixed_avg_hold":     "15d",
            "trail_avg_hold":     f"{tr.get('avg_hold_days', '?')}d",
            "trail_exit_pct":     tr.get("trail_exit_pct", None),
            "avg_peak_gain":      tr.get("avg_peak_gain",  None),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out / "v19_trailing_vs_fixed.csv", index=False)
    print(comp_df.to_string(index=False))

    # ---- Policy JSON ----
    policy_params.update({
        "version":            "v19",
        "horizon_days":       HORIZON_V19,
        "top_features_n":     TOP_FEATURES_N,
        "not_up_weight_adaptive": split_weights,
        "halflife_years":     HALFLIFE_YEARS,
        "macro_features":     macro_features,
        "macro_in_top_n":     macro_in_top,
        "trail_pct":          best_trail_pct,
        "max_hold_days":      best_max_hold,
        "trail_pct_grid":     TRAIL_PCT_GRID,
        "max_hold_grid":      MAX_HOLD_GRID,
        "regularization": {
            "max_depth":       MAX_DEPTH,
            "max_leaf_nodes":  MAX_LEAF_NODES,
            "learning_rate":   LEARNING_RATE,
            "l2_reg":          L2_REG,
            "min_samples_leaf": MIN_SAMP_LEAF,
        },
        "expand_cutoffs":     EXPAND_CUTOFFS,
    })
    with open(out / "v19_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_params, f, indent=2, ensure_ascii=False)

    # ---- Последние сигнальные карточки ----
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
            "signal":          r.get("signal", "HOLD"),
            "reason":          r.get("reason", ""),
            "p_up":            round(float(r.get("p_up", float("nan"))), 4),
            "p_down":          round(float(r.get("p_down", float("nan"))), 4),
            "trend_regime":    r.get("trend_regime", r.get("regime", "")),
            "adaptive_weight": split_weights.get(s, NOT_UP_WEIGHT),
            "trail_pct":       best_trail_pct,
            "max_hold_days":   best_max_hold,
        })
    pd.DataFrame(cards).to_csv(out / "v19_latest_signal_cards.csv", index=False)

    # ---- Purged CV (наследуем v18) ----
    print("\n=== v19: purged CV (с time-decay, baseline=0.50) ===")
    wf_df = purged_cv_v18(df, selected)
    wf_df.to_csv(out / "v19_purged_wf_cv.csv", index=False)
    if not wf_df.empty:
        mean_ba = wf_df["balanced_acc"].mean()
        std_ba  = wf_df["balanced_acc"].std()
        n_above = (wf_df["balanced_acc"] > 0.50).sum()
        print(f"  Фолдов: {len(wf_df)}, mean BA: {mean_ba:.3f} ± {std_ba:.3f}")
        print(f"  Выше 0.50: {n_above}/{len(wf_df)}")
        print(wf_df.to_string(index=False))
    else:
        print("  CV пуст")

    # ---- Сравнение v18 vs v19 ----
    v18_gr_path = Path("baseline_outputs_v18/v18_guardrails.csv")
    if v18_gr_path.exists():
        v18_gr    = pd.read_csv(v18_gr_path)
        comp_rows = []
        for s in ["valid", "test", "forward"]:
            r18 = v18_gr[v18_gr["split"] == s]
            r19 = guardrails[guardrails["split"] == s]
            comp_rows.append({
                "split":          s,
                "v18_precision":  pct(r18["precision"].values[0])     if not r18.empty else "-",
                "v19_precision":  pct(r19["precision"].values[0])     if not r19.empty else "-",
                "v18_wilson_low": pct(r18["wilson_95_low"].values[0]) if not r18.empty else "-",
                "v19_wilson_low": pct(r19["wilson_95_low"].values[0]) if not r19.empty else "-",
                "v19_n_signals":  r19["n_signals"].values[0]          if not r19.empty else "-",
            })
        comp_df2 = pd.DataFrame(comp_rows)
        comp_df2.to_csv(out / "v19_vs_v18_comparison.csv", index=False)
        print("\n=== Сравнение v18 vs v19 (guardrails) ===")
        print(comp_df2.to_string(index=False))

    print(f"\n=== v19 завершён. Результаты: {out} ===")
    print(f"  trail_pct={best_trail_pct:.2%}, max_hold={best_max_hold}d")
    print("  Дашборд: streamlit run dashboard_app.py")


if __name__ == "__main__":
    main()
