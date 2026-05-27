"""
scripts/grid_search_exits.py — grid search exit-параметров на ffill=5 predictions.

После обнаружения что OLD backtest был артефактом дыр в walk-forward,
honest model (ffill=5) показывает Sharpe 0.07 / +0.8% за 11 лет.

Гипотеза: модель exit'ит слишком рано на трендах. Тестируем разные параметры:
1. Baseline:        trail=0.12, max_hold=30, exit_thr=0.35 (current)
2. No model_exit:   exit_thr=0.0 (только trail+max_hold)
3. Loose trail:     trail=0.20
4. Long hold:       max_hold=90, trail=0.15
5. Trend friendly:  trail=0.18, max_hold=60, exit_thr=0.20
6. Ultra patient:   trail=0.25, max_hold=180, exit_thr=0.10
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def main() -> int:
    from app.multi_asset.simulator import simulate_trades, TradeConfig
    from app.multi_asset.metrics import compute_all_metrics
    from app.multi_asset.metal_loader import load_metals

    preds = pd.read_parquet("baseline_outputs_multiasset/e3b_adaptive/predictions.parquet")
    print(f"Predictions: {len(preds):,} rows, "
          f"{preds.index.min().date()} -> {preds.index.max().date()}")
    print(f"Columns: {list(preds.columns)}")

    silver = load_metals()["silver"]

    # Шесть стратегий exit-логики
    configs = {
        "baseline":       dict(trail_pct=0.12, max_hold_days=30,  exit_threshold=0.35),
        "no_model_exit":  dict(trail_pct=0.12, max_hold_days=30,  exit_threshold=0.0),
        "loose_trail":    dict(trail_pct=0.20, max_hold_days=30,  exit_threshold=0.35),
        "long_hold":      dict(trail_pct=0.15, max_hold_days=90,  exit_threshold=0.35),
        "trend_friendly": dict(trail_pct=0.18, max_hold_days=60,  exit_threshold=0.20),
        "ultra_patient":  dict(trail_pct=0.25, max_hold_days=180, exit_threshold=0.10),
    }

    rows = []
    for name, params in configs.items():
        cfg = TradeConfig(
            entry_threshold=0.48,
            cooldown_days=25,
            commission_pct=0.0005,
            spread_pct=0.0,
            slippage_pct=0.0005,
            direction_label=1,
            **params,
        )
        trades, _equity = simulate_trades(preds, silver, cfg)
        if not trades:
            metrics = {"n_trades": 0, "total_return": 0, "sharpe": 0,
                       "max_dd": 0, "win_rate": 0}
        else:
            tdf = pd.DataFrame([{
                "entry_date":  t.entry_date,
                "exit_date":   t.exit_date,
                "net_return":  t.net_return,
                "gross_return": t.gross_return,
                "hold_days":   t.hold_days,
                "exit_reason": t.exit_reason,
            } for t in trades])
            metrics = compute_all_metrics(tdf, n_trials=1)

        rows.append({
            "config":      name,
            "trail":       params["trail_pct"],
            "max_hold":    params["max_hold_days"],
            "exit_thr":    params["exit_threshold"],
            "n_trades":    metrics.get("n_trades", 0),
            "total_ret%":  round(metrics.get("total_return", 0) * 100, 1),
            "sharpe":      round(metrics.get("sharpe", 0), 3),
            "max_dd%":     round(metrics.get("max_dd", 0) * 100, 1),
            "win%":        round(metrics.get("win_rate", 0) * 100, 1),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe", ascending=False)
    print()
    print("=" * 110)
    print(" EXIT PARAMS GRID SEARCH (те же predictions, разная exit-логика)")
    print("=" * 110)
    print(df.to_string(index=False))

    out = Path("baseline_outputs_multiasset/exit_grid.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
