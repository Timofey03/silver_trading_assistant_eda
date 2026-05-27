"""
scripts/grid_search_regime_v2.py — гипотеза contrarian-режимов.

Открытие из grid_search_regime.py: стандартные trend/vol filters ХУЖЕ baseline.
Гипотеза: модель ловит mean-reversion (bottoms), не trends.
Тогда нам нужны ИНВЕРТИРОВАННЫЕ фильтры:
- Trade ТОЛЬКО когда price < SMA200 (oversold)
- Trade ТОЛЬКО когда ATR HIGH (volatility/panic)
- Trade ТОЛЬКО когда regime НЕ bull (по GMM)
"""
from __future__ import annotations
import os, sys, warnings
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
    from app.multi_asset.regime_filters import (
        trend_filter, volatility_filter, hmm_filter,
    )

    preds = pd.read_parquet("baseline_outputs_multiasset/e3b_adaptive/predictions.parquet")
    silver = load_metals()["silver"]
    SMOOTH = 3
    cfg = TradeConfig(
        entry_threshold=0.70, exit_threshold=0.30,
        trail_pct=0.20, max_hold_days=60, cooldown_days=10,
        commission_pct=0.0005, spread_pct=0.0, slippage_pct=0.0005,
        direction_label=1,
    )

    common = preds.index.intersection(silver.index)
    px = silver.reindex(common)

    # Precompute filters — ensure bool dtype
    m_trend_up = trend_filter(px["close"]).reindex(common).fillna(False).astype(bool)
    m_vol_low = volatility_filter(px["high"], px["low"], px["close"]).reindex(common).fillna(True).astype(bool)
    returns = px["close"].pct_change()
    in_bull_raw, _ = hmm_filter(returns, train_window=500)
    in_bull = in_bull_raw.reindex(common).fillna(True).astype(bool)

    variants = {
        "baseline":              pd.Series(True, index=common),
        "trend_DOWN_only":       ~m_trend_up,           # contrarian
        "vol_HIGH_only":         ~m_vol_low,            # contrarian
        "non_bull_only":         ~in_bull,              # contrarian
        "down_AND_volhigh":      (~m_trend_up) & (~m_vol_low),
        "down_OR_volhigh":       (~m_trend_up) | (~m_vol_low),
        "non_bull_AND_down":     (~in_bull) & (~m_trend_up),
    }

    rows = []
    for name, mask in variants.items():
        preds_smooth = preds.copy()
        preds_smooth["p_1"] = preds_smooth["p_1"].rolling(SMOOTH, min_periods=1).mean()
        mask_aligned = mask.reindex(preds_smooth.index, fill_value=True).astype(bool)
        preds_smooth["p_1"] = preds_smooth["p_1"].where(mask_aligned, 0.0)

        trades, _ = simulate_trades(preds_smooth, silver, cfg)
        if not trades:
            metrics = {"n_trades": 0, "total_return": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0}
        else:
            tdf = pd.DataFrame([{
                "entry_date": t.entry_date, "exit_date": t.exit_date,
                "net_return": t.net_return, "gross_return": t.gross_return,
                "hold_days": t.hold_days, "exit_reason": t.exit_reason,
            } for t in trades])
            metrics = compute_all_metrics(tdf, n_trials=len(variants))

        rows.append({
            "filter":     name,
            "days_allow": int(mask.sum()),
            "n":          metrics.get("n_trades", 0),
            "ret%":       round(metrics.get("total_return", 0) * 100, 1),
            "sharpe":     round(metrics.get("sharpe", 0), 3),
            "dd%":        round(metrics.get("max_dd", 0) * 100, 1),
            "win%":       round(metrics.get("win_rate", 0) * 100, 1),
        })

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print()
    print("=" * 95)
    print(" CONTRARIAN REGIME FILTERS")
    print("=" * 95)
    print(df.to_string(index=False))
    df.to_csv("baseline_outputs_multiasset/regime_grid_v2.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
