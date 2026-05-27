"""
scripts/grid_breakout_v2.py — улучшенный breakout без зависимости от p_up.

Учим из ошибки v1: модель E3b strongly bearish во время ралли (mean p_up=0.25),
поэтому requiring p_up>0.5 блокировал все trend entries. Делаем breakout
INDEPENDENTLY от модели.

Варианты:
- breakout_solid: close > 120d high (4 месяца ATH) + momentum 20d > +10%
- breakout_uptrend: close > SMA50 > SMA200 (golden cross style) + новый 60d high
- breakout_confirmed: 5 consecutive days close > 60d high (no 1-day fakeouts)
- combo: strong_only + breakout_confirmed
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

    preds = pd.read_parquet("baseline_outputs_multiasset/e3b_adaptive/predictions.parquet")
    silver = load_metals()["silver"]
    cfg = TradeConfig(
        entry_threshold=0.70, exit_threshold=0.30,
        trail_pct=0.20, max_hold_days=60, cooldown_days=10,
        commission_pct=0.0005, spread_pct=0.0, slippage_pct=0.0005,
        direction_label=1,
    )

    common = preds.index.intersection(silver.index)
    px = silver.reindex(common)
    close = px["close"]

    # Trend features
    sma50  = close.rolling(50,  min_periods=25).mean()
    sma200 = close.rolling(200, min_periods=100).mean()
    high60 = close.rolling(60, min_periods=30).max().shift(1)
    high120 = close.rolling(120, min_periods=60).max().shift(1)
    mom20 = close.pct_change(20)
    p_smoothed = preds["p_1"].rolling(3, min_periods=1).mean()
    p_aligned = p_smoothed.reindex(common)

    # Breakout flavors
    breakout_60 = (close > high60).fillna(False)
    breakout_60_confirmed = breakout_60.rolling(3, min_periods=3).min().fillna(0) == 1
    breakout_120 = (close > high120).fillna(False)
    uptrend = ((close > sma50) & (sma50 > sma200)).fillna(False)
    strong_mom = (mom20 > 0.10).fillna(False)

    variants = {
        "baseline (strong_only)": p_aligned >= 0.85,
        "+breakout_60_confirmed": (p_aligned >= 0.85) | breakout_60_confirmed,
        "+breakout_120_pure":     (p_aligned >= 0.85) | breakout_120,
        "+uptrend_strong_mom":    (p_aligned >= 0.85) | (uptrend & strong_mom),
        "+golden_cross_breakout": (p_aligned >= 0.85) | (uptrend & breakout_60),
        "pure_breakout_60_conf":  breakout_60_confirmed,
        "pure_uptrend_strong":    uptrend & strong_mom,
        "ALL_THREE":              (p_aligned >= 0.85) | breakout_60_confirmed | (uptrend & strong_mom),
    }

    rows = []
    for name, mask in variants.items():
        mask = mask.reindex(common).fillna(False).astype(bool)
        mask_aligned = mask.reindex(preds.index, fill_value=False).astype(bool)

        # Если в strong_only mode, используем raw p_up; иначе forced "1.0"
        preds_filt = preds.copy()
        if "baseline" in name or name.startswith("+"):
            # Use raw smoothed p_up if it's high enough, ELSE force 1.0 if in breakout zone
            base_strong = p_aligned >= 0.85
            # p = max(smoothed, 1.0 if breakout else 0)
            p_use = p_smoothed.reindex(preds.index).copy()
            override = mask_aligned & ~base_strong.reindex(preds.index, fill_value=False).astype(bool)
            p_use = p_use.where(~override, 1.0)
            # Zero out where mask is False (don't trade)
            p_use = p_use.where(mask_aligned, 0.0)
            preds_filt["p_1"] = p_use
        else:
            # Pure breakout mode
            preds_filt["p_1"] = pd.Series(
                [1.0 if m else 0.0 for m in mask_aligned], index=preds.index,
            )

        trades, _ = simulate_trades(preds_filt, silver, cfg)
        if not trades:
            metrics = {"n_trades": 0, "total_return": 0, "sharpe": 0,
                       "max_dd": 0, "win_rate": 0}
            last_exit_str = "—"
            in_rally = 0
        else:
            tdf = pd.DataFrame([{
                "entry_date": t.entry_date, "exit_date": t.exit_date,
                "net_return": t.net_return, "gross_return": t.gross_return,
                "hold_days": t.hold_days, "exit_reason": t.exit_reason,
            } for t in trades])
            metrics = compute_all_metrics(tdf, n_trials=len(variants))
            last_exit_str = str(pd.to_datetime(tdf["exit_date"]).max().date())
            in_rally = sum(1 for t in trades if pd.Timestamp("2025-04-15")
                           <= t.entry_date <= pd.Timestamp("2026-03-23"))

        rows.append({
            "config":     name,
            "days_allow": int(mask.sum()),
            "n":          metrics.get("n_trades", 0),
            "ret%":       round(metrics.get("total_return", 0) * 100, 1),
            "sharpe":     round(metrics.get("sharpe", 0), 3),
            "dd%":        round(metrics.get("max_dd", 0) * 100, 1),
            "win%":       round(metrics.get("win_rate", 0) * 100, 1),
            "rally_n":    in_rally,
            "last_exit":  last_exit_str,
        })

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print()
    print("=" * 130)
    print(" BREAKOUT v2 — independent от модели trend signals")
    print("=" * 130)
    print(df.to_string(index=False))
    df.to_csv("baseline_outputs_multiasset/grid_breakout_v2.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
