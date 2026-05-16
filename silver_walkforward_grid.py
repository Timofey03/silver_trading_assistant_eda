"""
silver_walkforward_grid.py — Consistency-aware grid search

Грид-поиск параметров стратегии по walk-forward данным (2018-2025).

КЛЮЧЕВОЕ ОТЛИЧИЕ от silver_signal_grid_search.py:
  Старый: оптимизировал по forward 2025 → overfit к bull market
  Новый:  оптимизирует по CONSISTENCY across 8 лет

Scoring (consistency-first):
  score = 0.5 × (n_pos / 8)        ← доля прибыльных лет
        + 0.3 × median_return        ← устойчивая прибыль
        + 0.2 × max(worst_year, -0.20)  ← защита от катастроф

  Штрафы:
   × 0   если worst_year < -0.30 (catastrophe)
   × 0.5 если worst_year < -0.20
   × 0.7 если n_positive < 4 (меньше половины)

OPTIMIZATION TRICK:
  Модель тренируется ОДИН раз для каждого года (8 моделей в кэш).
  Predictions тоже кэшируются. Только signal-generation + backtest перебираются.
  Это экономит ~95% времени.

Запуск:
  python silver_walkforward_grid.py            # default grid (~100 combos)
  python silver_walkforward_grid.py --quick    # быстрый грид (~30 combos)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from silver_walkforward_backtest import (
    load_data, load_feature_cols, train_for_year, predict_year,
)
from silver_signal_modes import SignalMode, generate_signals_with_exits, backtest_with_model_exits
from silver_assistant_v23_honest import (
    equity_compounded_sequential, risk_metrics_honest,
    recompute_trades_with_realistic_costs, RealisticCosts,
)
from silver_assistant_v19_trailing import COST_PER_TRADE

V22_DIR = Path("baseline_outputs_v22")
WF_DIR  = Path("baseline_outputs_walkforward")
WF_DIR.mkdir(exist_ok=True)
CACHE_DIR = WF_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# =============================================================================
# GRID DEFINITIONS
# =============================================================================

def build_grid(mode: str = "default") -> List[Dict]:
    """Параметры с уклоном в consistency: wider stops, longer cooldowns."""
    if mode == "quick":
        entries  = [0.50, 0.55, 0.60]
        exits    = [0.35, 0.40]
        cds      = [15, 20]
        trails   = [0.08, 0.10]
        max_holds = [30]
    else:
        entries  = [0.48, 0.52, 0.55, 0.58, 0.62]
        exits    = [0.30, 0.35, 0.40, 0.42]
        cds      = [10, 15, 20, 25]
        trails   = [0.07, 0.10, 0.12]
        max_holds = [30, 45]

    grid = []
    for e_in, e_out, cd, tr, mh in itertools.product(
        entries, exits, cds, trails, max_holds,
    ):
        if e_out >= e_in:
            continue
        grid.append({
            "p_up_entry": e_in,
            "p_up_exit":  e_out,
            "cooldown":   cd,
            "trail_pct":  tr,
            "max_hold":   mh,
        })
    return grid


# =============================================================================
# CACHE MODELS + PREDICTIONS
# =============================================================================

def cache_models_and_predictions(
    df: pd.DataFrame, feature_cols: List[str], years: List[int],
) -> Dict[int, pd.Series]:
    """Тренирует модели и предсказания (cache one-time per year)."""
    cache = {}
    for year in years:
        cache_file = CACHE_DIR / f"p_up_{year}.pkl"
        if cache_file.exists():
            print(f"  [{year}] загрузка из кэша...")
            with open(cache_file, "rb") as f:
                cache[year] = pickle.load(f)
        else:
            print(f"  [{year}] обучение модели...")
            try:
                model = train_for_year(df, year, feature_cols)
                p_up = predict_year(df, model, year, feature_cols)
                with open(cache_file, "wb") as f:
                    pickle.dump(p_up, f)
                cache[year] = p_up
            except Exception as e:
                print(f"    ERR: {e}")
                cache[year] = pd.Series(dtype=float)
    return cache


# =============================================================================
# BACKTEST ONE YEAR WITH GIVEN PARAMS
# =============================================================================

def backtest_year_with_params(
    df: pd.DataFrame, p_up: pd.Series, year: int, params: Dict,
    costs: RealisticCosts,
) -> Dict:
    """Backtest год с конкретными params. Возвращает метрики."""
    mode = SignalMode(
        name="grid", description="",
        p_up_entry=params["p_up_entry"],
        p_up_exit=params["p_up_exit"],
        cooldown=params["cooldown"],
        trail_pct=params["trail_pct"],
        max_hold=params["max_hold"],
        expected_trades_per_year=0,
    )

    test_data = df[df.index.year == year].copy()
    test_data["p_up"] = p_up.reindex(test_data.index)
    test_data["split"] = f"wf_{year}"

    if "silver_atr_14d" not in test_data.columns:
        if {"silver_high","silver_low","silver_close"}.issubset(test_data.columns):
            cp = test_data["silver_close"].shift(1)
            tr = pd.concat([
                test_data["silver_high"] - test_data["silver_low"],
                (test_data["silver_high"] - cp).abs(),
                (test_data["silver_low"]  - cp).abs(),
            ], axis=1).max(axis=1)
            test_data["silver_atr_14d"] = tr.ewm(span=14, adjust=False).mean()

    signaled = generate_signals_with_exits(test_data, test_data["p_up"], mode)
    trades = backtest_with_model_exits(signaled, f"wf_{year}", mode, cost=COST_PER_TRADE)

    if trades.empty:
        return {"n_trades": 0, "total_return": 0.0, "win_rate": 0.0, "max_dd": 0.0}

    trades = recompute_trades_with_realistic_costs(trades, signaled, costs)
    eq, seq = equity_compounded_sequential(trades, return_col="net_return")
    n_kept = len(seq)
    trade_days = int((seq["exit_date"].max() - seq["entry_date"].min()).days) \
        if n_kept >= 2 else 0
    rm = risk_metrics_honest(eq, n_trades=n_kept, trade_days_total=trade_days)

    return {
        "n_trades":     len(trades),
        "total_return": float(rm["total_return"]),
        "win_rate":     float((trades["net_return"] > 0).mean()),
        "max_dd":       float(rm["max_drawdown"]) if rm["max_drawdown"] else 0.0,
        "sharpe":       float(rm["sharpe"]) if rm["sharpe"] else 0.0,
    }


# =============================================================================
# CONSISTENCY SCORING
# =============================================================================

def consistency_score(year_results: List[Dict]) -> Dict:
    """
    Composite scoring с уклоном в consistency.

    Возвращает:
      score: composite (выше = лучше)
      n_positive: положительных лет
      median_return: медианная доходность
      worst_year: худший год
      mean_return: средняя
    """
    returns = [r["total_return"] for r in year_results]
    n = len(returns)
    if n == 0:
        return {"score": -999, "n_positive": 0, "median_return": 0,
                "worst_year": 0, "mean_return": 0, "no_trade_years": 0}

    no_trade_years = sum(1 for r in year_results if r["n_trades"] == 0)
    n_positive = sum(1 for r in returns if r > 0)
    median_return = float(np.median(returns))
    mean_return = float(np.mean(returns))
    worst_year = float(min(returns))

    # ШТРАФЫ
    if worst_year < -0.30:
        catastrophe_mult = 0.0  # катастрофа — score = 0
    elif worst_year < -0.20:
        catastrophe_mult = 0.5
    elif worst_year < -0.10:
        catastrophe_mult = 0.8
    else:
        catastrophe_mult = 1.0

    if n_positive < n / 2:
        majority_mult = 0.7
    else:
        majority_mult = 1.0

    if no_trade_years >= n / 2:  # >= половина лет без сделок = плохо
        activity_mult = 0.5
    else:
        activity_mult = 1.0

    # CORE SCORE
    pos_factor = (n_positive / n) ** 1.5  # squared to weight positivity
    base = pos_factor * 0.5 + median_return * 0.3 + max(worst_year, -0.20) * 0.2

    score = base * catastrophe_mult * majority_mult * activity_mult

    return {
        "score":          round(score, 4),
        "n_positive":     n_positive,
        "median_return":  round(median_return, 4),
        "mean_return":    round(mean_return, 4),
        "worst_year":     round(worst_year, 4),
        "best_year":      round(max(returns), 4),
        "no_trade_years": no_trade_years,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick",  action="store_true")
    ap.add_argument("--top",    type=int, default=15)
    args = ap.parse_args()

    print("=" * 70)
    print(" Consistency-aware grid search (walk-forward 8 лет)")
    print("=" * 70)

    df = load_data()
    feature_cols = [c for c in load_feature_cols() if c in df.columns]
    print(f"  Data: {len(df)} rows, features: {len(feature_cols)}")

    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

    # Phase 1: cache models + predictions
    print(f"\n=== Phase 1: train + predict для {len(years)} лет ===")
    p_up_cache = cache_models_and_predictions(df, feature_cols, years)

    # Phase 2: grid search
    grid = build_grid(mode="quick" if args.quick else "default")
    print(f"\n=== Phase 2: grid search ({len(grid)} комбинаций) ===")

    costs = RealisticCosts()
    all_results = []

    for i, params in enumerate(grid):
        year_results = []
        for year in years:
            if p_up_cache[year].empty:
                continue
            res = backtest_year_with_params(df, p_up_cache[year], year, params, costs)
            res["year"] = year
            year_results.append(res)

        cs = consistency_score(year_results)
        all_results.append({**params, **cs, "year_results": year_results})

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(grid)}] best so far: "
                  f"score={max(r['score'] for r in all_results):.3f}")

    # Sort by score
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # TOP
    print("\n" + "=" * 70)
    print(f" TOP-{args.top} по consistency score")
    print("=" * 70)
    print(f"{'entry':>6} {'exit':>5} {'cd':>4} {'trail':>6} {'mh':>4} "
          f"{'n_pos':>5} {'med':>7} {'worst':>7} {'mean':>7} {'score':>7}")
    print("-" * 70)
    for r in all_results[:args.top]:
        print(f"{r['p_up_entry']:>6.2f} {r['p_up_exit']:>5.2f} "
              f"{r['cooldown']:>4d} {r['trail_pct']:>6.2f} {r['max_hold']:>4d} "
              f"{r['n_positive']:>5d}/8 "
              f"{r['median_return']*100:>+6.1f}% "
              f"{r['worst_year']*100:>+6.1f}% "
              f"{r['mean_return']*100:>+6.1f}% "
              f"{r['score']:>7.3f}")

    # Best
    best = all_results[0]
    print("\n" + "=" * 70)
    print("  🏆 BEST PARAMS (consistency-aware)")
    print("=" * 70)
    print(f"  entry={best['p_up_entry']}, exit={best['p_up_exit']}, "
          f"cd={best['cooldown']}, trail={best['trail_pct']}, mh={best['max_hold']}")
    print(f"  Положительных лет: {best['n_positive']}/8")
    print(f"  Медианная доходность: {best['median_return']*100:+.1f}%")
    print(f"  Худший год: {best['worst_year']*100:+.1f}%")
    print(f"  Лучший год: {best['best_year']*100:+.1f}%")
    print(f"  Composite score: {best['score']:.3f}")
    print()
    print("  Year-by-year breakdown:")
    for yr in best["year_results"]:
        print(f"    {yr['year']}: trades={yr['n_trades']:2d} "
              f"return={yr['total_return']*100:+6.1f}% "
              f"win={yr['win_rate']*100:>3.0f}% "
              f"dd={yr['max_dd']*100:+6.1f}%")

    # Save
    save_results = [{k: v for k, v in r.items() if k != "year_results"} for r in all_results]
    pd.DataFrame(save_results).to_csv(WF_DIR / "consistency_grid_results.csv", index=False)
    (WF_DIR / "consistency_best.json").write_text(
        json.dumps(best, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Saved: {WF_DIR / 'consistency_best.json'}")
    print(f"  Saved: {WF_DIR / 'consistency_grid_results.csv'}")


if __name__ == "__main__":
    main()
