"""
scripts/grid_search_smoothed.py — тестируем сглаженный signal + asymmetric thresholds.

Найденная проблема: p_up осциллирует в шумовой зоне 0.4-0.6, что приводит к ложным
1-дневным входам/выходам. Например 2026-03-05 (p=0.61, entry) → 2026-03-06 (p=0.29, exit).
А через 2 недели модель кричит p=0.89-0.98 19 дней подряд — но в cooldown.

Тестируем:
1. Smoothing p_up через 3-дневное rolling mean (фильтр шума)
2. Asymmetric thresholds: entry=0.60 (строгий), exit=0.20 (мягкий — terpимый к диапу)
3. Короткий cooldown (5 дней) — чтобы успеть на следующую сильную серию
4. Combinations
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

    preds_raw = pd.read_parquet("baseline_outputs_multiasset/e3b_adaptive/predictions.parquet")
    silver = load_metals()["silver"]

    configs = {
        # Reference: current optimal
        "current_optimal": dict(
            p_up_smooth=1, entry=0.48, exit=0.35, trail=0.20, hold=30, cool=25,
        ),
        # Asymmetric — exit on really weak signal only
        "asym_low_exit":   dict(
            p_up_smooth=1, entry=0.48, exit=0.20, trail=0.20, hold=30, cool=25,
        ),
        # 3-day smooth — filter noise
        "smooth3":         dict(
            p_up_smooth=3, entry=0.48, exit=0.35, trail=0.20, hold=30, cool=25,
        ),
        # 5-day smooth
        "smooth5":         dict(
            p_up_smooth=5, entry=0.48, exit=0.35, trail=0.20, hold=30, cool=25,
        ),
        # Smooth + asymmetric + short cooldown — комбо
        "best_combo":      dict(
            p_up_smooth=3, entry=0.55, exit=0.20, trail=0.20, hold=60, cool=10,
        ),
        # Aggressive trend follower
        "trend_chaser":    dict(
            p_up_smooth=3, entry=0.65, exit=0.25, trail=0.25, hold=90, cool=5,
        ),
        # Same as best_combo but enter on strong signal only
        "strong_only":     dict(
            p_up_smooth=3, entry=0.70, exit=0.30, trail=0.20, hold=60, cool=10,
        ),
    }

    rows = []
    for name, params in configs.items():
        # Применяем сглаживание к p_up
        preds = preds_raw.copy()
        if params["p_up_smooth"] > 1:
            preds["p_1"] = preds["p_1"].rolling(
                window=params["p_up_smooth"], min_periods=1,
            ).mean()

        cfg = TradeConfig(
            entry_threshold=params["entry"],
            exit_threshold=params["exit"],
            trail_pct=params["trail"],
            max_hold_days=params["hold"],
            cooldown_days=params["cool"],
            commission_pct=0.0005,
            spread_pct=0.0,
            slippage_pct=0.0005,
            direction_label=1,
        )
        trades, _ = simulate_trades(preds, silver, cfg)
        if not trades:
            metrics = {"n_trades": 0, "total_return": 0, "sharpe": 0,
                       "max_dd": 0, "win_rate": 0}
        else:
            tdf = pd.DataFrame([{
                "entry_date":   t.entry_date,
                "exit_date":    t.exit_date,
                "net_return":   t.net_return,
                "gross_return": t.gross_return,
                "hold_days":    t.hold_days,
                "exit_reason":  t.exit_reason,
            } for t in trades])
            metrics = compute_all_metrics(tdf, n_trials=len(configs))

        # Сколько успело войти в 2025-07 → 2026-04?
        in_rally = sum(1 for t in trades if pd.Timestamp("2025-07-01")
                       <= t.entry_date <= pd.Timestamp("2026-04-30"))
        avg_hold = sum(t.hold_days for t in trades) / max(1, len(trades))

        rows.append({
            "config":        name,
            "smooth":        params["p_up_smooth"],
            "entry":         params["entry"],
            "exit":          params["exit"],
            "trail":         params["trail"],
            "hold":          params["hold"],
            "cool":          params["cool"],
            "n":             metrics.get("n_trades", 0),
            "ret%":          round(metrics.get("total_return", 0) * 100, 1),
            "sharpe":        round(metrics.get("sharpe", 0), 3),
            "dd%":           round(metrics.get("max_dd", 0) * 100, 1),
            "win%":          round(metrics.get("win_rate", 0) * 100, 1),
            "rally_trades":  in_rally,
            "avg_hold":      round(avg_hold, 1),
        })

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print()
    print("=" * 130)
    print(" Grid search: smoothing + asymmetric thresholds + cooldown")
    print(" rally_trades = number of trades that entered between 2025-07 and 2026-04")
    print("=" * 130)
    print(df.to_string(index=False))

    out = Path("baseline_outputs_multiasset/grid_smoothed.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
