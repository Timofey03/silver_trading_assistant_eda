"""
Финальный тест: ensemble + breakout для покрытия rally.
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

    common_idx = e3b.index.intersection(mom.index)
    p_e3b = e3b["p_1"].reindex(common_idx).rolling(3, min_periods=1).mean()
    p_mom = mom["p_1"].reindex(common_idx).rolling(3, min_periods=1).mean()

    close = silver.reindex(common_idx)["close"]
    high120 = close.rolling(120, min_periods=60).max().shift(1)
    high60  = close.rolling(60,  min_periods=30).max().shift(1)
    sma50   = close.rolling(50,  min_periods=25).mean()
    breakout_120 = (close > high120).fillna(False)
    breakout_60_uptrend = ((close > high60) & (close > sma50)).fillna(False)

    variants = {
        "e3b_strong (current best Sharpe)":
            p_e3b >= 0.85,
        "e3b OR mom_strong":
            (p_e3b >= 0.85) | (p_mom >= 0.85),
        "e3b OR mom_strong OR breakout_120":
            (p_e3b >= 0.85) | (p_mom >= 0.85) | breakout_120,
        "e3b OR mom_strong OR breakout_60_uptrend":
            (p_e3b >= 0.85) | (p_mom >= 0.85) | breakout_60_uptrend,
        "e3b OR (mom AND breakout)":
            (p_e3b >= 0.85) | ((p_mom >= 0.70) & breakout_120),
        "e3b OR (mom_loose AND breakout)":
            (p_e3b >= 0.85) | ((p_mom >= 0.50) & breakout_120),
        "FULL ENSEMBLE":
            (p_e3b >= 0.85) | (p_mom >= 0.85) | ((p_mom >= 0.50) & breakout_120),
    }

    rows = []
    for name, mask in variants.items():
        mask = mask.reindex(common_idx).fillna(False).astype(bool)
        p_eff = pd.concat([p_e3b, p_mom], axis=1).max(axis=1)
        p_use = p_eff.where(mask, 0.0)
        force = mask & (p_eff < 0.85)
        p_use = p_use.where(~force, 1.0)
        preds_filt = pd.DataFrame({"p_1": p_use.reindex(e3b.index).fillna(0.0)},
                                   index=e3b.index)

        trades, _ = simulate_trades(preds_filt, silver, cfg)
        if not trades:
            metrics = {"n_trades": 0, "total_return": 0, "sharpe": 0,
                       "max_dd": 0, "win_rate": 0}
            in_rally = 0; last_exit = "—"
        else:
            tdf = pd.DataFrame([{
                "entry_date": t.entry_date, "exit_date": t.exit_date,
                "net_return": t.net_return, "hold_days": t.hold_days,
                "gross_return": t.gross_return,
            } for t in trades])
            metrics = compute_all_metrics(tdf, n_trials=len(variants))
            in_rally = sum(1 for t in trades if pd.Timestamp("2025-04-15")
                           <= t.entry_date <= pd.Timestamp("2026-03-23"))
            last_exit = str(pd.to_datetime(tdf["exit_date"]).max().date())

        rows.append({
            "config":   name,
            "n":        metrics.get("n_trades", 0),
            "ret%":     round(metrics.get("total_return", 0) * 100, 1),
            "sharpe":   round(metrics.get("sharpe", 0), 3),
            "dd%":      round(metrics.get("max_dd", 0) * 100, 1),
            "win%":     round(metrics.get("win_rate", 0) * 100, 1),
            "rally":    in_rally,
            "last":     last_exit,
        })

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print()
    print("=" * 130)
    print(" FULL ENSEMBLE — E3b + momentum + breakout")
    print("=" * 130)
    print(df.to_string(index=False))
    df.to_csv("baseline_outputs_multiasset/grid_ensemble_v2.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
