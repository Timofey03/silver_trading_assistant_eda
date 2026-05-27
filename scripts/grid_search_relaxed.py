"""
scripts/grid_search_relaxed.py — поиск баланса между Sharpe и частотой сделок.

Проблема: contrarian filter (current) даёт Sharpe 0.92 но только 37 сделок за 11 лет
и большие окна без активности (last trade 2025-04-11, потом тишина почти год).

Тестируем более гибкие варианты:
1. no_filter           — голый smoothing (без regime)
2. contrarian_current  — текущий (price<SMA200 AND non-bull) [TOO STRICT]
3. price_under_sma200  — только trend filter (без GMM regime)
4. dual_mode           — contrarian OR (very_strong_p_up AND price>SMA50) [hybrid]
5. relaxed_above_sma   — позволять trade если price<SMA200*1.10 (10% буфер)
6. strong_signal_only  — без regime, но требовать smoothed p_up > 0.85
7. dual_mode_v2        — contrarian OR (p_up>0.80 AND not in chaos)

Цель: 50-80 сделок за 8-11 лет, Sharpe >= 0.6
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def main() -> int:
    from app.multi_asset.simulator import simulate_trades, TradeConfig
    from app.multi_asset.metrics import compute_all_metrics
    from app.multi_asset.metal_loader import load_metals
    from app.multi_asset.regime_filters import trend_filter, volatility_filter, hmm_filter

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
    close = px["close"]
    sma200 = close.rolling(200, min_periods=100).mean()
    sma50  = close.rolling(50,  min_periods=25).mean()
    m_price_under_sma200 = (close < sma200).reindex(common).fillna(False).astype(bool)
    m_price_under_sma50  = (close < sma50).reindex(common).fillna(False).astype(bool)
    m_price_under_sma200_relaxed = (close < sma200 * 1.10).reindex(common).fillna(False).astype(bool)
    m_vol_low = volatility_filter(px["high"], px["low"], close).reindex(common).fillna(True).astype(bool)
    returns = close.pct_change()
    in_bull_raw, _ = hmm_filter(returns, train_window=500)
    in_bull = in_bull_raw.reindex(common).fillna(True).astype(bool)

    # Compute smoothed p_up
    p_smoothed = preds["p_1"].rolling(SMOOTH, min_periods=1).mean()

    # Variants
    very_strong = (p_smoothed >= 0.85).reindex(common).fillna(False).astype(bool)
    strong      = (p_smoothed >= 0.80).reindex(common).fillna(False).astype(bool)

    variants = {
        "no_filter":            pd.Series(True,  index=common),
        "contrarian_current":   (~m_price_under_sma200.eq(False)) & (~in_bull),  # = price<SMA200 AND non-bull
        # ВАРИАНТЫ:
        "price_under_sma200":   m_price_under_sma200,
        "relaxed_above_sma":    m_price_under_sma200_relaxed,
        "strong_signal_only":   very_strong,
        "dual_mode":            (m_price_under_sma200 & ~in_bull) | (very_strong & ~m_price_under_sma50),
        "dual_mode_v2":         (m_price_under_sma200 & ~in_bull) | (strong & m_vol_low),
        "non_bull_GMM_only":    ~in_bull,
    }

    rows = []
    for name, mask in variants.items():
        mask_aligned = mask.reindex(preds.index, fill_value=True).astype(bool)
        preds_filt = preds.copy()
        preds_filt["p_1"] = p_smoothed.reindex(preds.index).where(mask_aligned, 0.0)

        trades, _ = simulate_trades(preds_filt, silver, cfg)
        if not trades:
            metrics = {"n_trades": 0, "total_return": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0}
            avg_gap_days = 0
        else:
            tdf = pd.DataFrame([{
                "entry_date": t.entry_date, "exit_date": t.exit_date,
                "net_return": t.net_return, "gross_return": t.gross_return,
                "hold_days": t.hold_days, "exit_reason": t.exit_reason,
            } for t in trades])
            metrics = compute_all_metrics(tdf, n_trials=len(variants))

            # Average gap between trades (calendar days)
            tdf_sorted = tdf.sort_values("entry_date")
            gaps = (
                pd.to_datetime(tdf_sorted["entry_date"]).shift(-1)
                - pd.to_datetime(tdf_sorted["exit_date"])
            ).dt.days.dropna()
            avg_gap_days = round(gaps.mean(), 0) if len(gaps) else 0
            # Last trade date
            last_exit = tdf_sorted["exit_date"].iloc[-1]

        # In rally period (2025-07 to 2026-04)
        in_rally = sum(1 for t in trades if pd.Timestamp("2025-07-01")
                       <= t.entry_date <= pd.Timestamp("2026-04-30"))

        last_exit_str = str(pd.to_datetime(last_exit).date()) if trades else "—"

        rows.append({
            "config":      name,
            "days_allow":  int(mask.sum()),
            "n":           metrics.get("n_trades", 0),
            "ret%":        round(metrics.get("total_return", 0) * 100, 1),
            "sharpe":      round(metrics.get("sharpe", 0), 3),
            "dd%":         round(metrics.get("max_dd", 0) * 100, 1),
            "win%":        round(metrics.get("win_rate", 0) * 100, 1),
            "avg_gap_d":   avg_gap_days,
            "rally_n":     in_rally,
            "last_exit":   last_exit_str,
        })

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print()
    print("=" * 130)
    print(" RELAXED FILTERS — balance Sharpe vs trade frequency")
    print(" avg_gap_d = средний разрыв между сделками; rally_n = сделок в период 2025-07..2026-04 рalли")
    print("=" * 130)
    print(df.to_string(index=False))
    df.to_csv("baseline_outputs_multiasset/grid_relaxed.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
