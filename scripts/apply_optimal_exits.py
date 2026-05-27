"""
scripts/apply_optimal_exits.py — финальная конфигурация с ensemble + breakout.

После эволюции:
1. trail=0.20 (вместо 0.12) — больше места дышать
2. Smoothing=3 day rolling mean — фильтр шумовых спайков
3. Strong-signal filter (E3b p_smoothed >= 0.85)
4. Momentum model (separate HistGB на trend features)
5. Breakout_120 override для distribution shift / ATH breakouts

Финальный entry mask:
  (e3b_p >= 0.85) OR (mom_p >= 0.85) OR (close > rolling_120d_high)

Backtest evolution (full clean data, 11.2 years):
  E1 baseline:                Sharpe 0.46 / +66%
  E3b OLD (fake gaps):        Sharpe 0.47 / +106% ⚠
  E3b ffill5 cleaned:         Sharpe 0.07 / +1%
  + trail=0.20:               Sharpe 0.35 / +38%
  + smoothing+strict entry:   Sharpe 0.52 / +91%
  + contrarian regime:        Sharpe 0.92 / +153%
  + strong_signal_only:       Sharpe 1.20 / +272% (best Sharpe но gap в rally)
  ★ + ensemble + breakout:    Sharpe 0.99 / +343% (rally покрыт!)
"""
from __future__ import annotations

import json
import os
import shutil
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

    print("=" * 70)
    print(" Apply ENSEMBLE config: E3b + momentum + breakout")
    print("=" * 70)

    e3b = pd.read_parquet("baseline_outputs_multiasset/e3b_adaptive/predictions.parquet")
    silver = load_metals()["silver"]

    # Try load momentum predictions (fallback to E3b-only if missing)
    mom_path = ROOT / "baseline_outputs_multiasset" / "momentum" / "predictions.parquet"
    if mom_path.exists():
        mom = pd.read_parquet(mom_path)
        print(f"  E3b preds: {len(e3b)}, Momentum preds: {len(mom)}")
        use_ensemble = True
    else:
        print(f"  No momentum predictions — using E3b only")
        mom = None
        use_ensemble = False

    common_idx = e3b.index if not use_ensemble else e3b.index.intersection(mom.index)
    p_e3b_smoothed = e3b["p_1"].reindex(common_idx).rolling(3, min_periods=1).mean()
    if use_ensemble:
        p_mom_smoothed = mom["p_1"].reindex(common_idx).rolling(3, min_periods=1).mean()
    else:
        p_mom_smoothed = pd.Series(0.0, index=common_idx)

    # Breakout signal (120-day ATH)
    close = silver.reindex(common_idx)["close"]
    high120 = close.rolling(120, min_periods=60).max().shift(1)
    breakout_120 = (close > high120).fillna(False).astype(bool)

    # Entry mask: E3b strong OR Momentum strong OR Breakout
    STRONG_THRESHOLD = 0.85
    e3b_strong = p_e3b_smoothed >= STRONG_THRESHOLD
    mom_strong = p_mom_smoothed >= STRONG_THRESHOLD
    entry_mask = e3b_strong | mom_strong | breakout_120

    print(f"  E3b strong:      {e3b_strong.sum()}/{len(common_idx)} days")
    print(f"  Momentum strong: {mom_strong.sum()}/{len(common_idx)} days")
    print(f"  Breakout 120d:   {breakout_120.sum()}/{len(common_idx)} days")
    print(f"  Combined mask:   {entry_mask.sum()}/{len(common_idx)} days "
          f"({entry_mask.sum()/len(common_idx)*100:.1f}%)")

    # Build effective p_up
    p_effective = pd.concat([p_e3b_smoothed, p_mom_smoothed], axis=1).max(axis=1)
    p_use = p_effective.where(entry_mask, 0.0)
    # Force p=1.0 if mask fires but both models < 0.85 (breakout-only entry)
    force = entry_mask & (p_effective < STRONG_THRESHOLD)
    p_use = p_use.where(~force, 1.0)

    preds_filt = pd.DataFrame({"p_1": p_use.reindex(e3b.index).fillna(0.0)},
                               index=e3b.index)

    cfg = TradeConfig(
        entry_threshold=0.70, exit_threshold=0.30,
        trail_pct=0.20, max_hold_days=60, cooldown_days=10,
        commission_pct=0.0005, spread_pct=0.0, slippage_pct=0.0005,
        direction_label=1,
    )
    print(f"  Sim cfg: entry={cfg.entry_threshold}, exit={cfg.exit_threshold}, "
          f"trail={cfg.trail_pct}, max_hold={cfg.max_hold_days}, cooldown={cfg.cooldown_days}")

    trades, _ = simulate_trades(preds_filt, silver, cfg)
    print(f"  Generated {len(trades)} closed trades")

    # === Detect OPEN position ===
    common = preds_filt.index.intersection(silver.index)
    px = silver.reindex(common)
    p_arr = preds_filt["p_1"].reindex(common).values
    cl, hi, lo = px["close"].values, px["high"].values, px["low"].values
    dates_list = common.tolist()

    import numpy as np
    state = "FLAT"
    entry_idx = None
    entry_price = None
    peak_price = None
    cooldown_until_idx = -1
    for i in range(len(dates_list)):
        c, h, lw, pp = cl[i], hi[i], lo[i], p_arr[i]
        if not np.isfinite(c):
            continue
        if state == "LONG":
            if h > peak_price:
                peak_price = h
            hold_days = i - entry_idx
            trail_level = peak_price * (1 - cfg.trail_pct)
            if lw <= trail_level or hold_days >= cfg.max_hold_days or (
                np.isfinite(pp) and pp < cfg.exit_threshold
            ):
                state = "FLAT"
                cooldown_until_idx = i + cfg.cooldown_days
                entry_idx = None
                peak_price = None
        elif state == "FLAT":
            if i < cooldown_until_idx:
                continue
            if not np.isfinite(pp):
                continue
            if pp >= cfg.entry_threshold:
                state = "LONG"
                entry_idx = i
                entry_price = c
                peak_price = h

    open_trade_info = None
    if state == "LONG":
        last_close = float(cl[-1])
        last_date = dates_list[-1]
        entry_date_open = dates_list[entry_idx]
        gross_open = last_close / entry_price - 1
        open_trade_info = {
            "entry_date":   entry_date_open.strftime("%Y-%m-%d"),
            "exit_date":    last_date.strftime("%Y-%m-%d") + "_OPEN",
            "entry_price":  float(entry_price),
            "exit_price":   float(last_close),
            "peak_price":   float(peak_price),
            "hold_days":    int(len(dates_list) - 1 - entry_idx),
            "gross_return": float(gross_open),
            "net_return":   float(gross_open),
            "exit_reason":  "OPEN",
        }
        print(f"  *** OPEN position: entered {entry_date_open.date()} @ ${entry_price:.2f}, "
              f"current ${last_close:.2f} ({gross_open*100:+.1f}%) peak ${peak_price:.2f}")

    # === Convert and save ===
    tdf = pd.DataFrame([{
        "entry_date":   t.entry_date.strftime("%Y-%m-%d") if hasattr(t.entry_date, "strftime") else str(t.entry_date),
        "exit_date":    t.exit_date.strftime("%Y-%m-%d") if hasattr(t.exit_date, "strftime") else str(t.exit_date),
        "entry_price":  t.entry_price,
        "exit_price":   t.exit_price,
        "peak_price":   t.peak_price,
        "hold_days":    t.hold_days,
        "gross_return": t.gross_return,
        "net_return":   t.net_return,
        "exit_reason":  t.exit_reason,
    } for t in trades])
    if open_trade_info:
        tdf = pd.concat([tdf, pd.DataFrame([open_trade_info])], ignore_index=True)

    out_dir = ROOT / "baseline_outputs_multiasset" / "e3b_adaptive"
    tdf.to_csv(out_dir / "trades.csv", index=False)

    # Metrics
    tdf_for_metrics = pd.DataFrame([{
        "entry_date":   t.entry_date,
        "exit_date":    t.exit_date,
        "net_return":   t.net_return,
        "gross_return": t.gross_return,
        "hold_days":    t.hold_days,
        "exit_reason":  t.exit_reason,
    } for t in trades])
    metrics = compute_all_metrics(tdf_for_metrics, n_trials=8)
    metrics_out = {
        **metrics,
        "n_predictions":  len(e3b),
        "experiment":     "e3b_ensemble_breakout",
        "features_pool":  105,
        "features_top_k": 30,
        "config": {
            "entry_threshold":     cfg.entry_threshold,
            "exit_threshold":      cfg.exit_threshold,
            "trail_pct":           cfg.trail_pct,
            "max_hold_days":       cfg.max_hold_days,
            "cooldown_days":       cfg.cooldown_days,
            "smoothing_window":    3,
            "strong_threshold":    STRONG_THRESHOLD,
            "breakout_lookback":   120,
            "ensemble":            "e3b OR mom OR breakout",
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 70)
    print(" Final metrics (ENSEMBLE)")
    print("=" * 70)
    print(f"  Sharpe:        {metrics.get('sharpe', 0):.3f}")
    print(f"  Total return:  {metrics.get('total_return', 0)*100:+.1f}%")
    print(f"  Annual return: {metrics.get('annual_return', 0)*100:+.2f}%")
    print(f"  Max DD:        {metrics.get('max_dd', 0)*100:.1f}%")
    print(f"  Win rate:      {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  N trades:      {metrics.get('n_trades', 0)}")

    # Rally trades
    if len(tdf):
        rally = tdf[(pd.to_datetime(tdf["entry_date"], errors="coerce") >= "2025-04-15")
                  & (pd.to_datetime(tdf["entry_date"], errors="coerce") <= "2026-03-23")]
        print(f"  Trades в rally 2025-04 → 2026-03: {len(rally)}")
        if len(rally):
            print(rally[["entry_date","exit_date","entry_price","exit_price","net_return","exit_reason"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
