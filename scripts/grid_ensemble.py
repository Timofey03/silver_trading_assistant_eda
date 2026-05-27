"""
scripts/grid_ensemble.py — ensemble E3b (mean-reversion) + momentum.

Тестируем 5 стратегий объединения:
1. e3b_only (baseline current)
2. momentum_only
3. max(e3b, momentum) — most confident wins
4. mean(e3b, momentum) — average
5. e3b OR (momentum AND breakout) — context-aware
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

    e3b = pd.read_parquet("baseline_outputs_multiasset/e3b_adaptive/predictions.parquet")
    mom = pd.read_parquet("baseline_outputs_multiasset/momentum/predictions.parquet")
    silver = load_metals()["silver"]

    cfg = TradeConfig(
        entry_threshold=0.70, exit_threshold=0.30,
        trail_pct=0.20, max_hold_days=60, cooldown_days=10,
        commission_pct=0.0005, spread_pct=0.0, slippage_pct=0.0005,
        direction_label=1,
    )

    # Align predictions
    common_idx = e3b.index.intersection(mom.index)
    p_e3b = e3b["p_1"].reindex(common_idx).rolling(3, min_periods=1).mean()
    p_mom = mom["p_1"].reindex(common_idx).rolling(3, min_periods=1).mean()

    # Compute breakout (для context-aware)
    close = silver.reindex(common_idx)["close"]
    high120 = close.rolling(120, min_periods=60).max().shift(1)
    breakout_120 = (close > high120).fillna(False)

    variants = {
        # Только E3b (текущая)
        "e3b_strong":           p_e3b >= 0.85,
        # Только momentum
        "momentum_strong":      p_mom >= 0.85,
        # Max: кто-то уверен — берём
        "max_e3b_mom":          (pd.concat([p_e3b, p_mom], axis=1).max(axis=1)) >= 0.85,
        # Mean
        "mean_e3b_mom":         ((p_e3b + p_mom) / 2) >= 0.70,
        # E3b + momentum-on-breakout
        "e3b_OR_mom_breakout":  (p_e3b >= 0.85) | ((p_mom >= 0.70) & breakout_120),
        # Both agree (conservative)
        "both_agree":           (p_e3b >= 0.60) & (p_mom >= 0.60),
        # E3b strong OR both moderate
        "e3b_strong_OR_both":   (p_e3b >= 0.85) | ((p_e3b >= 0.50) & (p_mom >= 0.50)),
    }

    rows = []
    for name, mask in variants.items():
        mask = mask.reindex(common_idx).fillna(False).astype(bool)

        # Use max(p_e3b, p_mom) as the effective p_up for entry
        p_eff = pd.concat([p_e3b, p_mom], axis=1).max(axis=1)
        p_use = p_eff.where(mask, 0.0)
        # Force to 1.0 if mask but p_eff < 0.85 (mask says enter regardless)
        force = mask & (p_eff < 0.85)
        p_use = p_use.where(~force, 1.0)

        # Build predictions DataFrame for simulator
        preds_filt = pd.DataFrame({"p_1": p_use.reindex(e3b.index).fillna(0.0)})
        preds_filt.index = e3b.index

        trades, _ = simulate_trades(preds_filt, silver, cfg)
        if not trades:
            metrics = {"n_trades": 0, "total_return": 0, "sharpe": 0,
                       "max_dd": 0, "win_rate": 0}
            in_rally = 0
            last_exit = "—"
        else:
            tdf = pd.DataFrame([{
                "entry_date": t.entry_date, "exit_date": t.exit_date,
                "net_return": t.net_return, "gross_return": t.gross_return,
                "hold_days": t.hold_days, "exit_reason": t.exit_reason,
            } for t in trades])
            metrics = compute_all_metrics(tdf, n_trials=len(variants))
            in_rally = sum(1 for t in trades if pd.Timestamp("2025-04-15")
                           <= t.entry_date <= pd.Timestamp("2026-03-23"))
            last_exit = str(pd.to_datetime(tdf["exit_date"]).max().date())

        rows.append({
            "config":     name,
            "days_allow": int(mask.sum()),
            "n":          metrics.get("n_trades", 0),
            "ret%":       round(metrics.get("total_return", 0) * 100, 1),
            "sharpe":     round(metrics.get("sharpe", 0), 3),
            "dd%":        round(metrics.get("max_dd", 0) * 100, 1),
            "win%":       round(metrics.get("win_rate", 0) * 100, 1),
            "rally_n":    in_rally,
            "last_exit":  last_exit,
        })

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print()
    print("=" * 130)
    print(" ENSEMBLE: E3b (mean-reversion) + Momentum")
    print("=" * 130)
    print(df.to_string(index=False))
    df.to_csv("baseline_outputs_multiasset/grid_ensemble.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
