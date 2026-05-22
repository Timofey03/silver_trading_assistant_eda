"""GRID SEARCH на forward test 2025-2026 — настройка torсhу входа и cooldown.

Цель: понять trade-offs между активностью модели и качеством сделок.

Перебираем:
  - entry_threshold ∈ {0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55}
  - cooldown_days  ∈ {10, 15, 20, 25, 30}

Итого 7 × 5 = 35 комбинаций. Не нужно переобучать модель — используем уже
сохранённые predictions.parquet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.multi_asset.metal_loader import load_metals
from app.multi_asset.simulator import simulate_trades, trades_to_df, TradeConfig
from app.multi_asset.metrics import compute_all_metrics

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

FWD_DIR = REPO_ROOT / "baseline_outputs_multiasset" / "forward_test_2025"
OUT_DIR = REPO_ROOT / "baseline_outputs_multiasset" / "forward_grid_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO_ROOT / "data" / "multi_asset" / "figures"

ENTRY_THRESHOLDS = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55]
COOLDOWN_DAYS_GRID = [10, 15, 20, 25, 30]
# Фиксированные параметры
EXIT_THRESHOLD = 0.35
TRAIL_PCT = 0.12
MAX_HOLD = 30


def run_grid():
    print("=" * 80)
    print(f"GRID SEARCH: {len(ENTRY_THRESHOLDS)} × {len(COOLDOWN_DAYS_GRID)} = "
          f"{len(ENTRY_THRESHOLDS) * len(COOLDOWN_DAYS_GRID)} комбинаций")
    print("=" * 80)

    predictions = pd.read_parquet(FWD_DIR / "predictions.parquet")
    silver = load_metals()["silver"]
    prices = silver[["close", "high", "low"]].reindex(predictions.index)

    results = []
    for entry_th in ENTRY_THRESHOLDS:
        for cool in COOLDOWN_DAYS_GRID:
            cfg = TradeConfig(
                entry_threshold=entry_th,
                exit_threshold=EXIT_THRESHOLD,
                trail_pct=TRAIL_PCT,
                max_hold_days=MAX_HOLD,
                cooldown_days=cool,
                commission_pct=0.001,
                direction_label=1,
            )
            trades, _ = simulate_trades(predictions, prices, cfg)
            trades_df = trades_to_df(trades)

            if trades_df.empty:
                metrics = {"n_trades": 0, "total_return": 0, "sharpe": 0,
                           "max_dd": 0, "win_rate": 0, "profit_factor": 0,
                           "annual_return": 0}
            else:
                metrics = compute_all_metrics(trades_df, n_trials=1)

            row = {
                "entry_threshold": entry_th,
                "cooldown_days": cool,
                "n_trades": metrics.get("n_trades", 0),
                "total_return": metrics.get("total_return", 0),
                "annual_return": metrics.get("annual_return", 0),
                "sharpe": metrics.get("sharpe", 0),
                "max_dd": metrics.get("max_dd", 0),
                "win_rate": metrics.get("win_rate", 0),
                "profit_factor": metrics.get("profit_factor", 0),
            }
            results.append(row)
            print(f"  thr={entry_th:.2f} cool={cool:2d}: "
                  f"trades={row['n_trades']:2d}, "
                  f"return={row['total_return']*100:+6.1f}%, "
                  f"Sharpe={row['sharpe']:5.2f}, "
                  f"DD={row['max_dd']*100:5.1f}%, "
                  f"win={row['win_rate']*100:4.0f}%")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "grid_results.csv", index=False)

    # ===== Best configurations =====
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)

    print("\n🏆 Best by Sharpe Ratio:")
    top_sharpe = df.nlargest(5, "sharpe")
    for _, r in top_sharpe.iterrows():
        print(f"  thr={r['entry_threshold']:.2f} cool={r['cooldown_days']:2.0f}: "
              f"Sharpe={r['sharpe']:5.2f}, return={r['total_return']*100:+6.1f}%, "
              f"trades={r['n_trades']:.0f}, DD={r['max_dd']*100:5.1f}%")

    print("\n💰 Best by Total Return:")
    top_return = df.nlargest(5, "total_return")
    for _, r in top_return.iterrows():
        print(f"  thr={r['entry_threshold']:.2f} cool={r['cooldown_days']:2.0f}: "
              f"return={r['total_return']*100:+6.1f}%, Sharpe={r['sharpe']:5.2f}, "
              f"trades={r['n_trades']:.0f}, DD={r['max_dd']*100:5.1f}%")

    print("\n🛡 Best by min(|DD|) при positive return:")
    safe = df[df["total_return"] > 0].copy()
    safe["abs_dd"] = safe["max_dd"].abs()
    top_safe = safe.nsmallest(5, "abs_dd")
    for _, r in top_safe.iterrows():
        print(f"  thr={r['entry_threshold']:.2f} cool={r['cooldown_days']:2.0f}: "
              f"DD={r['max_dd']*100:5.1f}%, Sharpe={r['sharpe']:5.2f}, "
              f"return={r['total_return']*100:+6.1f}%, trades={r['n_trades']:.0f}")

    # ===== Visualization =====
    print("\n📊 Создаю heatmap визуализации...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics_to_plot = [
        ("n_trades", "Кол-во сделок", "viridis", ".0f"),
        ("total_return", "Total return, %", "RdYlGn", "+.1f"),
        ("sharpe", "Sharpe Ratio", "RdYlGn", ".2f"),
        ("max_dd", "Max Drawdown, %", "RdYlGn", ".1f"),
        ("win_rate", "Win Rate, %", "RdYlGn", ".0f"),
        ("profit_factor", "Profit Factor (cap=10)", "RdYlGn", ".2f"),
    ]

    for ax, (metric, title, cmap, fmt) in zip(axes.flat, metrics_to_plot):
        pivot = df.pivot(index="entry_threshold", columns="cooldown_days",
                         values=metric).astype(float)
        if metric in ("total_return", "win_rate", "max_dd"):
            pivot = pivot * 100
        if metric == "profit_factor":
            pivot = pivot.clip(upper=10)

        sns.heatmap(pivot, annot=True, fmt=fmt,
                    cmap=cmap, ax=ax, cbar_kws={"shrink": 0.8},
                    center=0 if metric == "max_dd" else None)
        ax.set_title(title)
        ax.set_xlabel("Cooldown (дней)")
        ax.set_ylabel("Порог входа")

        # Highlight baseline cell (0.48, 25)
        try:
            row_idx = ENTRY_THRESHOLDS.index(0.48)
            col_idx = COOLDOWN_DAYS_GRID.index(25)
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1,
                                        fill=False, edgecolor="black",
                                        linewidth=3))
        except (ValueError, IndexError):
            pass

    fig.suptitle("Grid search на forward test 2025-2026 "
                 "(чёрная рамка = baseline thr=0.48, cool=25)",
                 fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    path = FIG_DIR / "09_forward_grid_search.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

    # ===== Summary table =====
    baseline = df[(df["entry_threshold"] == 0.48) &
                  (df["cooldown_days"] == 25)].iloc[0]
    best_sharpe = df.loc[df["sharpe"].idxmax()]
    best_return = df.loc[df["total_return"].idxmax()]

    summary = {
        "baseline": {
            "entry_threshold": float(baseline["entry_threshold"]),
            "cooldown_days": int(baseline["cooldown_days"]),
            "n_trades": int(baseline["n_trades"]),
            "total_return": float(baseline["total_return"]),
            "sharpe": float(baseline["sharpe"]),
            "max_dd": float(baseline["max_dd"]),
            "win_rate": float(baseline["win_rate"]),
        },
        "best_by_sharpe": {
            "entry_threshold": float(best_sharpe["entry_threshold"]),
            "cooldown_days": int(best_sharpe["cooldown_days"]),
            "n_trades": int(best_sharpe["n_trades"]),
            "total_return": float(best_sharpe["total_return"]),
            "sharpe": float(best_sharpe["sharpe"]),
            "max_dd": float(best_sharpe["max_dd"]),
            "win_rate": float(best_sharpe["win_rate"]),
        },
        "best_by_return": {
            "entry_threshold": float(best_return["entry_threshold"]),
            "cooldown_days": int(best_return["cooldown_days"]),
            "n_trades": int(best_return["n_trades"]),
            "total_return": float(best_return["total_return"]),
            "sharpe": float(best_return["sharpe"]),
            "max_dd": float(best_return["max_dd"]),
            "win_rate": float(best_return["win_rate"]),
        },
    }

    with open(OUT_DIR / "grid_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {OUT_DIR}")
    return df, summary


if __name__ == "__main__":
    run_grid()
