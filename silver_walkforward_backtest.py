"""
silver_walkforward_backtest.py — Honest multi-year walk-forward backtest

Цель: превратить 11 forward trades в 80+ trades через rolling train/test.

Метод:
  Для каждого года [2018, 2019, ..., 2025]:
    train = data[year < cutoff]      # модель учится только на прошлом
    test  = data[year == cutoff]     # бэктест на этом году
    trades_year = backtest(model, test)
    all_trades += trades_year

Итого:
  ~88 honest OOS trades в РАЗНЫХ режимах рынка:
  - 2018 sideways
  - 2019 recovery
  - 2020 covid crash + recovery
  - 2021 inflation
  - 2022 bear (rates ↑)
  - 2023 sideways
  - 2024 mild bull
  - 2025 strong bull

Это даёт честную оценку: работает ли стратегия в любых режимах,
или только в bull markets как 2025.

Запуск:
  python silver_walkforward_backtest.py             # все годы
  python silver_walkforward_backtest.py --year 2020 # один год
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from silver_assistant_v18_adaptive import (
    RegimeEnsembleV18, compute_sample_weights, _get_regimes,
    HISTORICAL_UP_RATE,
)
from silver_assistant_v16_binary import TOP_FEATURES_N
from silver_signal_modes import (
    OPTIMAL_PARAMS, PRESETS,
    generate_signals_with_exits, backtest_with_model_exits,
)
from silver_assistant_v23_honest import (
    equity_compounded_sequential, risk_metrics_honest,
    recompute_trades_with_realistic_costs, RealisticCosts,
    bootstrap_ci_metrics, sharpe_stats,
    probabilistic_sharpe_ratio, deflated_sharpe_ratio,
)
from silver_assistant_v19_trailing import COST_PER_TRADE

V22_DIR = Path("baseline_outputs_v22")
WF_DIR  = Path("baseline_outputs_walkforward")
WF_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. LOAD DATA + FEATURES
# =============================================================================

def load_data() -> pd.DataFrame:
    p = V22_DIR / "v22_full_data.csv"
    df = pd.read_csv(p, parse_dates=[0]).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def load_feature_cols() -> List[str]:
    fi = pd.read_csv(V22_DIR / "v22_feature_importance.csv")
    return fi.sort_values("importance", ascending=False).head(TOP_FEATURES_N)["feature"].tolist()


# =============================================================================
# 2. TRAIN/TEST FOR ONE YEAR
# =============================================================================

def train_for_year(
    df: pd.DataFrame, cutoff_year: int, feature_cols: List[str],
) -> RegimeEnsembleV18:
    """Тренирует модель на данных СТРОГО ДО cutoff_year."""
    train_data = df[df.index.year < cutoff_year]
    labeled = train_data[
        train_data["tb_label_bin"].notna()
        & train_data[feature_cols].notna().all(axis=1)
    ].copy()

    if len(labeled) < 200:
        raise ValueError(f"Слишком мало training data для {cutoff_year}: {len(labeled)}")

    X = labeled[feature_cols]
    y = labeled["tb_label_bin"].astype(int).values
    regimes = _get_regimes(labeled)
    sw = compute_sample_weights(labeled, halflife_years=1.5)

    recent_up = float(labeled.tail(252)["tb_label_bin"].mean())
    not_up_w = HISTORICAL_UP_RATE / max(recent_up, 0.05)

    model = RegimeEnsembleV18(not_up_weight=not_up_w)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X, y, regimes, sample_weight=sw)
    return model


def predict_year(
    df: pd.DataFrame, model: RegimeEnsembleV18,
    cutoff_year: int, feature_cols: List[str],
) -> pd.Series:
    """Predict p_up на данных IN cutoff_year."""
    test_data = df[df.index.year == cutoff_year]
    valid = test_data[test_data[feature_cols].notna().all(axis=1)]
    if valid.empty:
        return pd.Series(dtype=float)

    X = valid[feature_cols]
    regimes = _get_regimes(valid)
    with contextlib.redirect_stdout(io.StringIO()):
        p_up = model.p_up(X, regimes)
    return pd.Series(p_up, index=valid.index, name="p_up")


def backtest_year(
    df: pd.DataFrame, p_up: pd.Series, cutoff_year: int,
    mode_name: str = "optimal",
    enable_short: bool = False,
    enable_kelly: bool = False,
) -> pd.DataFrame:
    """
    Бэктест с выбранным mode на один год.

    mode_name:    пресет из PRESETS (default "optimal" = MaxReturn v28)
    enable_short: торговать в обе стороны (LONG + SHORT)
    enable_kelly: пропорциональный сайзинг на основе p_up
    """
    mode = PRESETS.get(mode_name, OPTIMAL_PARAMS)

    test_data = df[df.index.year == cutoff_year].copy()
    test_data["p_up"]  = p_up.reindex(test_data.index)
    test_data["split"] = f"wf_{cutoff_year}"

    if "silver_atr_14d" not in test_data.columns:
        if {"silver_high", "silver_low", "silver_close"}.issubset(test_data.columns):
            cp = test_data["silver_close"].shift(1)
            tr = pd.concat([
                test_data["silver_high"] - test_data["silver_low"],
                (test_data["silver_high"] - cp).abs(),
                (test_data["silver_low"]  - cp).abs(),
            ], axis=1).max(axis=1)
            test_data["silver_atr_14d"] = tr.ewm(span=14, adjust=False).mean()

    signaled = generate_signals_with_exits(
        test_data, test_data["p_up"], mode,
        enable_short=enable_short,
        enable_kelly=enable_kelly,
    )
    trades = backtest_with_model_exits(
        signaled, f"wf_{cutoff_year}", mode,
        cost=COST_PER_TRADE,
        enable_short=enable_short,
        enable_kelly=enable_kelly,
    )

    if not trades.empty:
        costs = RealisticCosts()
        trades = recompute_trades_with_realistic_costs(trades, signaled, costs)
        trades["wf_year"] = cutoff_year
    return trades


# =============================================================================
# 3. AGGREGATE METRICS
# =============================================================================

def compute_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    """Метрики на trades."""
    if trades.empty:
        return {"label": label, "n_trades": 0, "total_return": 0,
                "sharpe": None, "max_dd": None, "win_rate": None, "dsr": None}

    eq, seq = equity_compounded_sequential(trades, return_col="net_return")
    n_kept = len(seq)
    trade_days = int((seq["exit_date"].max() - seq["entry_date"].min()).days) \
        if n_kept >= 2 else 0
    rm = risk_metrics_honest(eq, n_trades=n_kept, trade_days_total=trade_days)

    rets = seq["net_return"].astype(float).values if not seq.empty else np.array([])
    sharpe_per, skew, kurt = sharpe_stats(rets) if len(rets) >= 4 else (float("nan"), 0, 3)
    # PSR/DSR: sqrt внутри формулы может уйти в минус при малом n_obs + отриц. Sharpe
    # → ловим ValueError и возвращаем nan вместо краша
    try:
        psr = probabilistic_sharpe_ratio(sharpe_per, len(rets), skew, kurt, 0.0) \
            if not np.isnan(sharpe_per) else float("nan")
    except (ValueError, ZeroDivisionError):
        psr = float("nan")
    try:
        dsr = deflated_sharpe_ratio(
            sharpe_per, len(rets), n_trials=8,
            sharpe_variance=0.5 * (sharpe_per**2 + 1e-6),
            skew=skew, kurt=kurt,
        ) if not np.isnan(sharpe_per) else float("nan")
    except (ValueError, ZeroDivisionError):
        dsr = float("nan")

    return {
        "label":         label,
        "n_trades":      len(trades),
        "n_sequential":  n_kept,
        "total_return":  round(rm["total_return"], 4),
        "win_rate":      round(float((trades["net_return"] > 0).mean()), 4),
        "sharpe":        round(rm["sharpe"], 3) if rm["sharpe"] is not None else None,
        "max_dd":        round(rm["max_drawdown"], 4) if rm["max_drawdown"] is not None else None,
        "calmar":        round(rm["calmar"], 3) if rm["calmar"] is not None else None,
        "psr":           round(psr, 4) if not np.isnan(psr) else None,
        "dsr":           round(dsr, 4) if not np.isnan(dsr) else None,
        "sharpe_per":    round(sharpe_per, 4) if not np.isnan(sharpe_per) else None,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=str, default="2018,2019,2020,2021,2022,2023,2024,2025",
                    help="Comma-separated years to backtest")
    ap.add_argument("--year",  type=int, default=None, help="Only one year")
    ap.add_argument(
        "--mode", choices=list(PRESETS.keys()), default="optimal",
        help="Торговый пресет: optimal/max_return (default), balanced, consistent, "
             "conservative, aggressive, ultra",
    )
    ap.add_argument("--short",  action="store_true", help="Включить SHORT позиции")
    ap.add_argument("--kelly",  action="store_true", help="Включить Kelly sizing")
    args = ap.parse_args()

    print(f"  Режим: {args.mode}"
          + (" + SHORT" if args.short else "")
          + (" + Kelly" if args.kelly else ""))

    print("=" * 70)
    print(" Walk-forward backtest: 8 независимых OOS периодов")
    print("=" * 70)

    df = load_data()
    feature_cols = [c for c in load_feature_cols() if c in df.columns]
    print(f"  Data: {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Features: {len(feature_cols)}")

    years = [args.year] if args.year else [int(y) for y in args.years.split(",")]

    per_year = {}
    all_trades = []

    for year in years:
        print(f"\n=== Year {year} ===")
        train_size = len(df[df.index.year < year])
        test_size  = len(df[df.index.year == year])
        print(f"  Train: {train_size} rows (data < {year})")
        print(f"  Test:  {test_size} rows ({year})")

        try:
            model  = train_for_year(df, year, feature_cols)
            p_up   = predict_year(df, model, year, feature_cols)
            trades = backtest_year(
                df, p_up, year,
                mode_name=args.mode,
                enable_short=args.short,
                enable_kelly=args.kelly,
            )
        except Exception as e:
            print(f"  ERR: {type(e).__name__}: {e}")
            continue

        if trades.empty:
            print(f"  📊 0 trades в {year}")
            per_year[year] = {"label": str(year), "n_trades": 0}
            continue

        metrics = compute_metrics(trades, label=str(year))
        per_year[year] = metrics
        all_trades.append(trades)

        print(f"  📊 Trades: {metrics['n_trades']}, "
              f"Total: {metrics['total_return']*100:+.1f}%, "
              f"Win: {metrics['win_rate']*100:.0f}%, "
              f"Sharpe: {metrics['sharpe']}, "
              f"MaxDD: {metrics['max_dd']*100:.1f}%")

        trades.to_csv(WF_DIR / f"trades_{year}.csv", index=False)

    # Aggregate
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(WF_DIR / "trades_all.csv", index=False)

        print("\n" + "=" * 70)
        print(f" АГРЕГИРОВАННЫЕ МЕТРИКИ ({len(combined)} trades)")
        print("=" * 70)
        agg = compute_metrics(combined, label="all_walk_forward")
        for k, v in agg.items():
            print(f"  {k:18s}: {v}")

        # By year breakdown
        print("\n" + "=" * 70)
        print(" BREAKDOWN ПО ГОДАМ")
        print("=" * 70)
        rows = []
        for y in years:
            m = per_year.get(y, {"label": str(y), "n_trades": 0})
            rows.append({
                "year":     y,
                "n_trades": m.get("n_trades", 0),
                "return":   f"{(m.get('total_return') or 0)*100:+.1f}%",
                "win_rate": f"{(m.get('win_rate') or 0)*100:.0f}%",
                "sharpe":   m.get("sharpe", "—"),
                "max_dd":   f"{(m.get('max_dd') or 0)*100:.1f}%",
            })
        breakdown_df = pd.DataFrame(rows)
        print(breakdown_df.to_string(index=False))
        breakdown_df.to_csv(WF_DIR / "year_breakdown.csv", index=False)

        # Consistency analysis
        positive_years = sum(1 for y in years if per_year.get(y, {}).get("total_return", 0) > 0)
        print(f"\n  ✅ Положительных лет: {positive_years}/{len(years)} = {positive_years/len(years)*100:.0f}%")
        win_years_rate = positive_years / len(years)
        if win_years_rate >= 0.65:
            print("  🟢 ВЕРДИКТ: стратегия консистентна (≥65% годов плюсовые)")
        elif win_years_rate >= 0.50:
            print("  🟡 ВЕРДИКТЬ: стратегия неоднозначна (50-64% годов плюсовые)")
        else:
            print("  🔴 ВЕРДИКТ: стратегия НЕ работает на разных рынках")

        # Save summary JSON
        summary = {
            "ts":             datetime.now(timezone.utc).isoformat(),
            "years_tested":   years,
            "aggregated":     agg,
            "per_year":       {str(y): per_year[y] for y in years if y in per_year},
            "positive_years": int(positive_years),
            "total_years":    len(years),
            "consistency":    round(positive_years / len(years), 4),
        }
        (WF_DIR / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\n  ✅ Saved: {WF_DIR / 'summary.json'}")
        print(f"  ✅ Saved: {WF_DIR / 'year_breakdown.csv'}")
        print(f"  ✅ Saved: {WF_DIR / 'trades_all.csv'}")


if __name__ == "__main__":
    main()
