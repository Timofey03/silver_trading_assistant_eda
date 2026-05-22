"""Сравнение E3b (наш новый winner) с V25 (production в Streamlit).

Три уровня сравнения:
1. Метрики на общих метриках (apples-to-apples где возможно)
2. Per-period breakdown (что когда лучше работает)
3. Overlap-period comparison (E3b vs V25 в одной временной шкале 2025-2026)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.multi_asset.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "baseline_outputs_multiasset" / "comparison_v25"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trades(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    return t


def per_period_metrics(trades: pd.DataFrame, periods: list) -> pd.DataFrame:
    """Compute metrics for each period."""
    rows = []
    for start, end, label in periods:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        chunk = trades[(trades["entry_date"] >= s) & (trades["entry_date"] < e)]
        if len(chunk) > 0:
            nr = chunk["net_return"]
            rows.append({
                "period":      label,
                "n_trades":    len(chunk),
                "win_rate":    float((nr > 0).mean()),
                "compound":    float(((1 + nr).prod() - 1)),
                "mean_return": float(nr.mean()),
                "best":        float(nr.max()),
                "worst":       float(nr.min()),
            })
        else:
            rows.append({
                "period": label, "n_trades": 0, "win_rate": np.nan,
                "compound": np.nan, "mean_return": np.nan,
                "best": np.nan, "worst": np.nan,
            })
    return pd.DataFrame(rows)


def run_comparison():
    logger.info("=" * 70)
    logger.info("E3b vs V25 (Streamlit production) COMPARISON")
    logger.info("=" * 70)

    # Загрузка
    e3b = load_trades(REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv")
    v25_fwd = load_trades(REPO_ROOT / "baseline_outputs_v25" / "v25_forward_trades.csv")
    v25_wf = load_trades(REPO_ROOT / "baseline_outputs_walkforward" / "trades_all.csv")

    logger.info(f"E3b:        {len(e3b)} trades, {e3b['entry_date'].min().date()} → {e3b['exit_date'].max().date()}")
    logger.info(f"V25 forward: {len(v25_fwd)} trades, {v25_fwd['entry_date'].min().date()} → {v25_fwd['exit_date'].max().date()}")
    logger.info(f"V25 WF 8y:   {len(v25_wf)} trades, {v25_wf['entry_date'].min().date()} → {v25_wf['exit_date'].max().date()}")

    # ===== 1. Full metrics =====
    logger.info("\n[1/3] Computing full metrics for each model...")
    metrics = {}
    for name, df in [("e3b", e3b), ("v25_forward", v25_fwd), ("v25_walkforward", v25_wf)]:
        metrics[name] = compute_all_metrics(df, n_trials=1)
        logger.info(f"  {name}: Sharpe={metrics[name]['sharpe']:.3f}, "
                    f"Annual={metrics[name]['annual_return']*100:+.1f}%, "
                    f"Win={metrics[name]['win_rate']*100:.0f}%, "
                    f"MaxDD={metrics[name]['max_dd']*100:.0f}%")

    # ===== 2. Per-period breakdown =====
    logger.info("\n[2/3] Per-period breakdown for E3b vs V25 walk-forward...")
    periods = [
        ("2014-01-01", "2017-01-01", "2014-2016: Mild bull"),
        ("2017-01-01", "2019-01-01", "2017-2018: Sideways"),
        ("2019-01-01", "2021-01-01", "2019-2020: COVID"),
        ("2021-01-01", "2023-01-01", "2021-2022: Hike"),
        ("2023-01-01", "2025-01-01", "2023-2024: Normal"),
        ("2025-01-01", "2026-06-01", "2025-2026: BULL"),
    ]

    e3b_periods = per_period_metrics(e3b, periods)
    v25_fwd_periods = per_period_metrics(v25_fwd, periods)
    v25_wf_periods = per_period_metrics(v25_wf, periods)

    # Combine для удобства
    combined = pd.DataFrame({
        "period":             e3b_periods["period"],
        "e3b_trades":         e3b_periods["n_trades"],
        "e3b_win":            e3b_periods["win_rate"],
        "e3b_compound":       e3b_periods["compound"],
        "v25_wf_trades":      v25_wf_periods["n_trades"],
        "v25_wf_win":         v25_wf_periods["win_rate"],
        "v25_wf_compound":    v25_wf_periods["compound"],
        "v25_fwd_trades":     v25_fwd_periods["n_trades"],
        "v25_fwd_win":        v25_fwd_periods["win_rate"],
        "v25_fwd_compound":   v25_fwd_periods["compound"],
    })

    # ===== 3. Overlap period analysis =====
    logger.info("\n[3/3] Overlap-period (apples-to-apples)...")
    overlap_start = max(e3b["entry_date"].min(), v25_wf["entry_date"].min())
    overlap_end = min(e3b["exit_date"].max(), v25_wf["exit_date"].max())
    logger.info(f"  Overlap: {overlap_start.date()} → {overlap_end.date()}")

    e3b_overlap = e3b[(e3b["entry_date"] >= overlap_start) & (e3b["exit_date"] <= overlap_end)]
    v25_wf_overlap = v25_wf[(v25_wf["entry_date"] >= overlap_start) & (v25_wf["exit_date"] <= overlap_end)]

    overlap_e3b = compute_all_metrics(e3b_overlap, n_trials=1) if not e3b_overlap.empty else {}
    overlap_v25 = compute_all_metrics(v25_wf_overlap, n_trials=1) if not v25_wf_overlap.empty else {}

    # ===== Save results =====
    logger.info("\nSaving artifacts...")
    combined.to_csv(OUT_DIR / "per_period_breakdown.csv", index=False)

    summary = {
        "comparison_date": pd.Timestamp.now().isoformat(),
        "overlap_period": {
            "start": overlap_start.isoformat(),
            "end":   overlap_end.isoformat(),
        },
        "e3b_full":            metrics["e3b"],
        "v25_forward_full":    metrics["v25_forward"],
        "v25_walkforward_full": metrics["v25_walkforward"],
        "e3b_in_overlap":      overlap_e3b,
        "v25_wf_in_overlap":   overlap_v25,
    }

    with open(OUT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ===== Print summary =====
    print()
    print("=" * 90)
    print("СВОДКА — Полные метрики каждой модели")
    print("=" * 90)
    print(f"{'Metric':<18} {'E3b (new)':>15} {'V25 forward':>15} {'V25 WF 8 лет':>15}")
    print("-" * 90)
    for key, label in [
        ("n_trades",      "Сделок"),
        ("period_years",  "Лет покрытия"),
        ("total_return",  "Total return"),
        ("annual_return", "Annual return"),
        ("sharpe",        "Sharpe"),
        ("sortino",       "Sortino"),
        ("max_dd",        "Max drawdown"),
        ("profit_factor", "Profit factor"),
        ("win_rate",      "Win rate"),
        ("mean_return",   "Mean per trade"),
        ("best_trade",    "Best trade"),
        ("worst_trade",   "Worst trade"),
        ("psr",           "PSR"),
        ("dsr",           "DSR"),
    ]:
        e3b_v = metrics["e3b"].get(key, "—")
        v25f_v = metrics["v25_forward"].get(key, "—")
        v25w_v = metrics["v25_walkforward"].get(key, "—")
        fmt_row = []
        for v in [e3b_v, v25f_v, v25w_v]:
            if isinstance(v, float):
                if key in ("total_return", "annual_return", "max_dd", "win_rate",
                           "mean_return", "best_trade", "worst_trade"):
                    fmt_row.append(f"{v*100:+.1f}%")
                else:
                    fmt_row.append(f"{v:.3f}")
            else:
                fmt_row.append(str(v))
        print(f"{label:<18} {fmt_row[0]:>15} {fmt_row[1]:>15} {fmt_row[2]:>15}")

    print()
    print("=" * 90)
    print("ПЕРИОДЫ — что когда лучше работает")
    print("=" * 90)
    print(f"{'Period':<25} {'E3b':<22} {'V25 walk-forward':<22} {'V25 forward':<22}")
    print("-" * 90)
    for _, r in combined.iterrows():
        e3b_str = (f"{r['e3b_trades']:>2}t  win {r['e3b_win']*100:.0f}%  "
                   f"{r['e3b_compound']*100:+.0f}%") if r["e3b_trades"] > 0 else "—"
        v25w_str = (f"{r['v25_wf_trades']:>2}t  win {r['v25_wf_win']*100:.0f}%  "
                    f"{r['v25_wf_compound']*100:+.0f}%") if r["v25_wf_trades"] > 0 else "—"
        v25f_str = (f"{r['v25_fwd_trades']:>2}t  win {r['v25_fwd_win']*100:.0f}%  "
                    f"{r['v25_fwd_compound']*100:+.0f}%") if r["v25_fwd_trades"] > 0 else "—"
        print(f"{r['period']:<25} {e3b_str:<22} {v25w_str:<22} {v25f_str:<22}")

    print()
    print("=" * 90)
    print(f"OVERLAP PERIOD ({overlap_start.date()} → {overlap_end.date()})")
    print("=" * 90)
    if overlap_e3b and overlap_v25:
        print(f"{'Metric':<18} {'E3b':>14} {'V25 WF':>14} {'Δ':>14}")
        print("-" * 60)
        for key, label in [
            ("n_trades",      "Trades"),
            ("total_return",  "Total return"),
            ("annual_return", "Annual return"),
            ("sharpe",        "Sharpe"),
            ("win_rate",      "Win rate"),
            ("max_dd",        "Max DD"),
            ("profit_factor", "Profit factor"),
        ]:
            v_e = overlap_e3b.get(key)
            v_v = overlap_v25.get(key)
            if isinstance(v_e, float) and isinstance(v_v, float):
                if key in ("total_return", "annual_return", "max_dd", "win_rate"):
                    diff = v_e - v_v
                    print(f"{label:<18} {v_e*100:+13.1f}% {v_v*100:+13.1f}% {diff*100:+13.1f}pp")
                else:
                    diff = v_e - v_v
                    print(f"{label:<18} {v_e:>14.3f} {v_v:>14.3f} {diff:>+14.3f}")
            else:
                print(f"{label:<18} {str(v_e):>14} {str(v_v):>14}")

    print()
    print(f"Saved: {OUT_DIR}")


if __name__ == "__main__":
    run_comparison()
