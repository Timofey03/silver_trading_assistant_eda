"""
scripts/grid_trend_override.py — добавить trend-following override к strong_signal.

Проблема: mean-reversion E3b пропустил 100%+ ралли 2025-04 → 2026-03 потому
что features вышли за training distribution. p_up был bearish весь период.

Решение A: добавить второй entry path — breakout signal:
  ENTER если (smoothed p_up >= 0.85) OR (
      close > rolling_60d_high AND price > SMA50 AND
      momentum_5d > +3% AND p_up >= 0.50  -- модель хотя бы не bearish
  )

Тестируем разные пороги breakout.
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

    # Breakout signals
    high60 = close.rolling(60, min_periods=30).max().shift(1)  # вчерашний max (no leak)
    sma50  = close.rolling(50, min_periods=25).mean()
    mom5d  = close.pct_change(5)
    breakout = (close > high60).fillna(False)
    above_sma50 = (close > sma50).fillna(False)

    p_smoothed = preds["p_1"].rolling(3, min_periods=1).mean()
    p_aligned = p_smoothed.reindex(common)

    # Конфигурации
    variants = {
        "strong_only (baseline)": pd.Series(p_aligned >= 0.85, index=common),
        "trend_override_loose": (
            (p_aligned >= 0.85)
            | (breakout & above_sma50 & (mom5d > 0.02) & (p_aligned >= 0.40))
        ),
        "trend_override_medium": (
            (p_aligned >= 0.85)
            | (breakout & above_sma50 & (mom5d > 0.03) & (p_aligned >= 0.50))
        ),
        "trend_override_strict": (
            (p_aligned >= 0.85)
            | (breakout & above_sma50 & (mom5d > 0.05) & (p_aligned >= 0.50))
        ),
        "breakout_only": breakout & above_sma50 & (mom5d > 0.03),
    }

    rows = []
    for name, mask in variants.items():
        mask = mask.reindex(common).fillna(False).astype(bool)
        mask_aligned = mask.reindex(preds.index, fill_value=False).astype(bool)

        # Для breakout-only — используем p=1 если pass mask else 0
        # Для override — используем raw smoothed p_up если passed, иначе 0
        preds_filt = preds.copy()
        if name == "breakout_only":
            preds_filt["p_1"] = pd.Series(
                [1.0 if m else 0.0 for m in mask_aligned],
                index=preds.index,
            )
        else:
            preds_filt["p_1"] = p_smoothed.reindex(preds.index).where(mask_aligned, 0.0)

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
    print(" TREND OVERRIDE — добавляем breakout entry path к strong_signal")
    print(" rally_n = сделок в период 2025-04 → 2026-03 ралли (был 0!)")
    print("=" * 130)
    print(df.to_string(index=False))
    df.to_csv("baseline_outputs_multiasset/grid_trend_override.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
