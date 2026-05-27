"""
scripts/grid_search_regime.py — тест regime фильтров поверх optimal exit config.

Базовый конфиг (current best): smoothing=3, entry=0.70, exit=0.30,
trail=0.20, hold=60, cool=10. Дает Sharpe 0.515 / +91% на cleaned data.

Тестируем поверх 5 вариантов:
1. baseline — без фильтров
2. trend — SMA200 trend filter
3. vol — ATR volatility filter (skip top 10%)
4. trend + vol — оба
5. trend + vol + gmm — все + GMM regime detector
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
    from app.multi_asset.regime_filters import apply_filters

    preds = pd.read_parquet("baseline_outputs_multiasset/e3b_adaptive/predictions.parquet")
    silver = load_metals()["silver"]

    # Optimal config (smoothed)
    SMOOTH = 3
    cfg = TradeConfig(
        entry_threshold=0.70, exit_threshold=0.30,
        trail_pct=0.20, max_hold_days=60, cooldown_days=10,
        commission_pct=0.0005, spread_pct=0.0, slippage_pct=0.0005,
        direction_label=1,
    )

    variants = [
        ("baseline",        False, False, False),
        ("trend",           True,  False, False),
        ("vol",             False, True,  False),
        ("trend+vol",       True,  True,  False),
        ("trend+vol+gmm",   True,  True,  True),
    ]

    rows = []
    for name, use_trend, use_vol, use_hmm in variants:
        print(f"\n[{name}]")
        # Smooth p_up
        preds_smooth = preds.copy()
        preds_smooth["p_1"] = preds_smooth["p_1"].rolling(SMOOTH, min_periods=1).mean()

        # Apply regime filters
        p_filtered = apply_filters(
            preds_smooth["p_1"], silver,
            use_trend=use_trend, use_vol=use_vol, use_hmm=use_hmm,
            hmm_train_window=500,
        )
        # Inject filtered p_1 back
        preds_smooth["p_1"] = p_filtered.reindex(preds_smooth.index)

        trades, _ = simulate_trades(preds_smooth, silver, cfg)
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
            metrics = compute_all_metrics(tdf, n_trials=len(variants))

        rows.append({
            "filter":     name,
            "n":          metrics.get("n_trades", 0),
            "ret%":       round(metrics.get("total_return", 0) * 100, 1),
            "sharpe":     round(metrics.get("sharpe", 0), 3),
            "dd%":        round(metrics.get("max_dd", 0) * 100, 1),
            "win%":       round(metrics.get("win_rate", 0) * 100, 1),
        })

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print()
    print("=" * 90)
    print(" REGIME FILTER GRID (поверх optimal exit config)")
    print("=" * 90)
    print(df.to_string(index=False))

    df.to_csv("baseline_outputs_multiasset/regime_grid.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
