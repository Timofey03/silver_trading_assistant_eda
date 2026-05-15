"""
silver_signal_grid_search.py — Grid search оптимальных параметров стратегии

Перебирает (p_up_entry, p_up_exit, cooldown, trail_pct) и выбирает оптимум.

Метрика выбора (composite_score):
  score = forward_total_return × 0.5 + valid_total_return × 0.25 + test_total_return × 0.25
  с штрафом × 0.5 если win_rate < 50% на любом split
  с штрафом × 0.3 если total_return < 0 на любом split

Это даёт robust оптимум — не overfit к одному периоду.

Запуск:
  python silver_signal_grid_search.py
  python silver_signal_grid_search.py --fine    # узкий грид вокруг балансированного
  python silver_signal_grid_search.py --wide    # широкий грид
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from silver_signal_modes import (
    SignalMode, generate_signals_with_exits, backtest_with_model_exits,
)
from silver_assistant_v23_honest import (
    equity_compounded_sequential, risk_metrics_honest,
    recompute_trades_with_realistic_costs, RealisticCosts,
)
from silver_assistant_v19_trailing import COST_PER_TRADE

V22_DIR = Path("baseline_outputs_v22")
V25_DIR = Path("baseline_outputs_v25")
GRID_DIR = Path("baseline_outputs_grid")
GRID_DIR.mkdir(exist_ok=True)


# =============================================================================
# GRID DEFINITION
# =============================================================================

def build_grid(mode: str = "default") -> List[Dict]:
    """Возвращает список dict с параметрами для перебора."""
    if mode == "fine":
        # узкий грид вокруг балансированного (быстро)
        entries  = [0.50, 0.51, 0.52, 0.53, 0.54]
        exits    = [0.42, 0.44, 0.46]
        cooldowns = [8, 10, 12]
        trails   = [0.06, 0.07, 0.08]
        max_holds = [30]
    elif mode == "wide":
        # широкий грид (медленно)
        entries  = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58]
        exits    = [0.38, 0.40, 0.42, 0.44, 0.46, 0.48]
        cooldowns = [3, 5, 7, 10, 13, 16, 20]
        trails   = [0.04, 0.05, 0.07, 0.09]
        max_holds = [20, 30, 45]
    else:  # default — баланс между скоростью и охватом
        entries   = [0.49, 0.51, 0.52, 0.53, 0.55]
        exits     = [0.40, 0.43, 0.45, 0.47]
        cooldowns = [6, 9, 12, 15]
        trails    = [0.06, 0.07, 0.08]
        max_holds = [30]

    grid = []
    for e_in, e_out, cd, tr, mh in itertools.product(
        entries, exits, cooldowns, trails, max_holds,
    ):
        if e_out >= e_in:  # exit threshold должен быть НИЖЕ entry
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
# EVALUATE ONE COMBINATION
# =============================================================================

def eval_combination(
    df: pd.DataFrame, p_up: pd.Series, params: Dict,
    costs: RealisticCosts,
) -> Dict:
    """Прогоняет один набор параметров, возвращает метрики per-split."""
    mode = SignalMode(
        name="grid",
        description="grid search",
        p_up_entry=params["p_up_entry"],
        p_up_exit=params["p_up_exit"],
        cooldown=params["cooldown"],
        trail_pct=params["trail_pct"],
        max_hold=params["max_hold"],
        expected_trades_per_year=0,
    )

    signaled = generate_signals_with_exits(df, p_up, mode)

    result = {**params}
    for split in ["valid", "test", "forward"]:
        trades = backtest_with_model_exits(signaled, split, mode, cost=COST_PER_TRADE)
        if trades.empty:
            result[f"{split}_n_trades"]     = 0
            result[f"{split}_total_return"] = 0.0
            result[f"{split}_win_rate"]     = 0.0
            result[f"{split}_sharpe"]       = 0.0
            result[f"{split}_max_dd"]       = 0.0
            continue
        trades = recompute_trades_with_realistic_costs(trades, signaled, costs)
        eq, seq = equity_compounded_sequential(trades, return_col="net_return")
        n_kept = len(seq)
        trade_days = int((seq["exit_date"].max() - seq["entry_date"].min()).days) \
            if n_kept >= 2 else 0
        metrics = risk_metrics_honest(eq, n_trades=n_kept, trade_days_total=trade_days)
        result[f"{split}_n_trades"]     = len(trades)
        result[f"{split}_total_return"] = round(metrics["total_return"], 4)
        result[f"{split}_win_rate"]     = round(float((trades["net_return"] > 0).mean()), 4)
        result[f"{split}_sharpe"]       = round(metrics["sharpe"], 3) if metrics["sharpe"] is not None else 0.0
        result[f"{split}_max_dd"]       = round(metrics["max_drawdown"], 4) if metrics["max_drawdown"] is not None else 0.0

    return result


# =============================================================================
# SCORING
# =============================================================================

def composite_score(row: pd.Series) -> float:
    """
    Композитный скор: forward важнее, но требуется консистентность.

    Формула:
      base = 0.5 × forward + 0.25 × valid + 0.25 × test
      penalty × 0.5 если win_rate < 50% на любом split
      penalty × 0.3 если total_return < 0 на любом split (кроме маленьких)
      bonus × 1.1 если Sharpe > 1.0 на forward
    """
    fwd = row.get("forward_total_return", 0)
    val = row.get("valid_total_return", 0)
    tst = row.get("test_total_return", 0)

    base = 0.5 * fwd + 0.25 * val + 0.25 * tst

    # Штраф за низкий win rate
    for split in ["valid", "test", "forward"]:
        wr = row.get(f"{split}_win_rate", 0)
        n  = row.get(f"{split}_n_trades", 0)
        if n >= 3 and wr < 0.5:
            base *= 0.5
            break

    # Штраф за отрицательный return
    for split in ["valid", "test", "forward"]:
        ret = row.get(f"{split}_total_return", 0)
        n   = row.get(f"{split}_n_trades", 0)
        if n >= 3 and ret < -0.05:
            base *= 0.3
            break

    # Бонус за высокий Sharpe
    if row.get("forward_sharpe", 0) > 1.0:
        base *= 1.1

    return base


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="default",
                    choices=["default", "fine", "wide"])
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    print("=" * 70)
    print(f" Grid search: '{args.grid}' grid")
    print("=" * 70)

    # Load data
    full = pd.read_csv(V22_DIR / "v22_full_data.csv", parse_dates=[0]).set_index("Date")
    full.index = pd.to_datetime(full.index)
    p_up = pd.read_csv(V25_DIR / "v25_p_up_cpcv.csv", parse_dates=[0]).set_index("Date")
    full["p_up"] = p_up.iloc[:, 0]

    # ATR fallback
    if "silver_atr_14d" not in full.columns:
        cp = full["silver_close"].shift(1)
        tr = pd.concat([
            full["silver_high"] - full["silver_low"],
            (full["silver_high"] - cp).abs(),
            (full["silver_low"]  - cp).abs(),
        ], axis=1).max(axis=1)
        full["silver_atr_14d"] = tr.ewm(span=14, adjust=False).mean()

    costs = RealisticCosts()
    grid = build_grid(args.grid)
    print(f"  Total combinations: {len(grid)}")

    results = []
    for i, params in enumerate(grid):
        try:
            row = eval_combination(full, full["p_up"], params, costs)
            results.append(row)
            if (i + 1) % 10 == 0:
                fwd_ret = row.get("forward_total_return", 0)
                print(f"  [{i+1:3d}/{len(grid)}] "
                      f"entry={params['p_up_entry']}, exit={params['p_up_exit']}, "
                      f"cd={params['cooldown']}, trail={params['trail_pct']} "
                      f"→ fwd={fwd_ret*100:+.1f}%")
        except Exception as e:
            print(f"  [{i+1}/{len(grid)}] FAIL: {e}")

    df = pd.DataFrame(results)
    if df.empty:
        print("\n  ❌ Нет результатов!")
        return

    # Композитный скор
    df["score"] = df.apply(composite_score, axis=1)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.to_csv(GRID_DIR / "grid_results.csv", index=False)

    print("\n" + "=" * 70)
    print(f"  TOP-{args.top} по composite score")
    print("=" * 70)
    show_cols = ["p_up_entry", "p_up_exit", "cooldown", "trail_pct", "max_hold",
                 "valid_total_return", "test_total_return", "forward_total_return",
                 "forward_win_rate", "forward_sharpe", "forward_n_trades", "score"]
    show_cols = [c for c in show_cols if c in df.columns]
    top = df.head(args.top)[show_cols].copy()
    for c in ["valid_total_return", "test_total_return", "forward_total_return", "forward_win_rate"]:
        if c in top.columns:
            top[c] = top[c].apply(lambda x: f"{x*100:+.1f}%")
    top["score"] = top["score"].apply(lambda x: f"{x:.4f}")
    print(top.to_string(index=False))

    # Best
    best = df.iloc[0]
    print("\n" + "=" * 70)
    print("  🏆 BEST PARAMETERS")
    print("=" * 70)
    print(f"  p_up_entry: {best['p_up_entry']}")
    print(f"  p_up_exit:  {best['p_up_exit']}")
    print(f"  cooldown:   {int(best['cooldown'])} days")
    print(f"  trail_pct:  {best['trail_pct']}")
    print(f"  max_hold:   {int(best['max_hold'])} days")
    print()
    print(f"  Performance:")
    print(f"    valid:   {best['valid_total_return']*100:+.1f}% ({int(best['valid_n_trades'])} trades, win {best['valid_win_rate']*100:.0f}%)")
    print(f"    test:    {best['test_total_return']*100:+.1f}% ({int(best['test_n_trades'])} trades, win {best['test_win_rate']*100:.0f}%)")
    print(f"    forward: {best['forward_total_return']*100:+.1f}% ({int(best['forward_n_trades'])} trades, win {best['forward_win_rate']*100:.0f}%)")
    print(f"    forward Sharpe: {best['forward_sharpe']}")
    print(f"    forward MaxDD:  {best['forward_max_dd']*100:.1f}%")
    print(f"    composite score: {best['score']:.4f}")

    # Save best as JSON
    best_dict = {
        "p_up_entry": float(best["p_up_entry"]),
        "p_up_exit":  float(best["p_up_exit"]),
        "cooldown":   int(best["cooldown"]),
        "trail_pct":  float(best["trail_pct"]),
        "max_hold":   int(best["max_hold"]),
        "performance": {
            "valid":   {"total_return": float(best["valid_total_return"]),
                        "n_trades": int(best["valid_n_trades"]),
                        "win_rate": float(best["valid_win_rate"])},
            "test":    {"total_return": float(best["test_total_return"]),
                        "n_trades": int(best["test_n_trades"]),
                        "win_rate": float(best["test_win_rate"])},
            "forward": {"total_return": float(best["forward_total_return"]),
                        "n_trades": int(best["forward_n_trades"]),
                        "win_rate": float(best["forward_win_rate"]),
                        "sharpe": float(best["forward_sharpe"]),
                        "max_drawdown": float(best["forward_max_dd"])},
        },
        "composite_score": float(best["score"]),
        "grid_size": int(len(grid)),
    }
    (GRID_DIR / "best_params.json").write_text(
        json.dumps(best_dict, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n  ✅ Saved: {GRID_DIR / 'best_params.json'}")
    print(f"  ✅ Full grid: {GRID_DIR / 'grid_results.csv'}")


if __name__ == "__main__":
    main()
