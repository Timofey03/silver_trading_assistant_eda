"""E1 BASELINE — воспроизведение текущей V25 модели на новой инфраструктуре.

Сетап:
- Features: ТОЛЬКО silver per-asset технические (~14 фичей)
- Labels: horizon=20, non-adaptive (фиксированные ±1.5 sigma)
- Model: HistGradientBoosting (как сейчас)
- Walk-forward: sliding 1000 / step 30 / test 30
- Trade execution: trailing 12%, max_hold 30, cooldown 25

Цель: зафиксировать benchmark Sharpe / Win Rate для сравнения с E2-E5.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.multi_asset.config import REPO_ROOT as CONFIG_ROOT, METALS
from app.multi_asset.metal_loader import load_single_metal
from app.multi_asset.features import per_asset_features
from app.multi_asset.labels import build_multi_horizon_labels
from app.multi_asset.walkforward import WalkForwardEngine, WFConfig, accuracy_metrics
from app.multi_asset.simulator import simulate_trades, trades_to_df, TradeConfig
from app.multi_asset.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = CONFIG_ROOT / "baseline_outputs_multiasset" / "e1_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_e1() -> dict:
    """Запустить E1 baseline эксперимент."""
    logger.info("=" * 70)
    logger.info("E1 BASELINE — silver-only features, non-adaptive labels, horizon=20")
    logger.info("=" * 70)

    # Step 1: загрузка silver данных
    logger.info("\n[1/5] Loading silver data...")
    silver = load_single_metal("silver")
    logger.info(f"  Silver: {len(silver):,} rows, {silver.index.min().date()} → {silver.index.max().date()}")

    # Step 2: features (только silver per-asset, без cross-asset/macro)
    logger.info("\n[2/5] Building silver-only features...")
    features = per_asset_features(silver, prefix="silver")
    features = features.dropna()
    logger.info(f"  Features: {len(features.columns)} cols × {len(features):,} clean rows")

    # Step 3: labels (non-adaptive, horizon=20)
    logger.info("\n[3/5] Building labels (horizon=20, non-adaptive)...")
    labels_full = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
        horizons=[20],
        adaptive=False,  # ← BASELINE: фиксированные барьеры
    )
    labels = labels_full["label_20"]
    logger.info(f"  Labels: {labels.value_counts(dropna=False).to_dict()}")

    # Step 4: walk-forward
    logger.info("\n[4/5] Running walk-forward...")
    config = WFConfig(
        train_window=1000,
        test_window=30,
        step=30,
        horizon=20,
        max_depth=6,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=30,
        random_state=42,
    )
    engine = WalkForwardEngine(features, labels, config)
    predictions, folds = engine.run(verbose=True)

    # Сохраняем predictions
    predictions.to_parquet(OUTPUT_DIR / "predictions.parquet", compression="snappy")

    # Accuracy на out-of-sample
    acc = accuracy_metrics(predictions)
    logger.info(f"  OOS accuracy: {acc['accuracy']:.3f} (n={acc['n']})")
    if "per_class" in acc:
        for cls, info in acc["per_class"].items():
            logger.info(f"    class={cls}: recall {info['recall']:.3f} (n={info['n']})")

    # Step 5: simulation
    logger.info("\n[5/5] Simulating trades...")
    # Подготовим цены для симулятора
    prices = silver[["close", "high", "low"]].reindex(predictions.index)

    # В labels у нас классы {-1, 0, 1}. В predictions колонки p_-1, p_0, p_1
    # Для simulator укажем direction_label=1 (TP)
    trade_cfg = TradeConfig(
        entry_threshold=0.48,
        exit_threshold=0.35,
        trail_pct=0.12,
        max_hold_days=30,
        cooldown_days=25,
        commission_pct=0.001,
        direction_label=1,
    )
    trades, equity = simulate_trades(predictions, prices, trade_cfg)
    trades_df = trades_to_df(trades)

    if not trades_df.empty:
        trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
        logger.info(f"  Trades: {len(trades_df)} executed")
    else:
        logger.warning("  No trades generated!")

    # Step 6: metrics
    metrics = compute_all_metrics(trades_df, n_trials=1) if not trades_df.empty else {}

    # Add accuracy info
    metrics["oos_accuracy"] = acc.get("accuracy", 0)
    metrics["n_predictions"] = acc.get("n", 0)
    metrics["n_folds"] = len(folds)
    metrics["experiment"] = "E1_baseline"
    metrics["features_used"] = len(features.columns)
    metrics["adaptive_barriers"] = False
    metrics["cross_asset"] = False
    metrics["multi_horizon"] = False

    # Save metrics
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("E1 RESULTS")
    logger.info("=" * 70)
    for key in ["n_trades", "trades_per_year", "period_years", "total_return",
                "annual_return", "sharpe", "sortino", "max_dd", "profit_factor",
                "win_rate", "mean_return", "best_trade", "worst_trade",
                "psr", "dsr", "oos_accuracy"]:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                if key.endswith("return") or key in ("max_dd", "mean_win", "mean_loss",
                                                     "best_trade", "worst_trade",
                                                     "mean_return", "median_return"):
                    logger.info(f"  {key:20s}: {val*100:+.2f}%")
                else:
                    logger.info(f"  {key:20s}: {val:.3f}")
            else:
                logger.info(f"  {key:20s}: {val}")

    logger.info(f"\nSaved to: {OUTPUT_DIR}")
    return metrics


if __name__ == "__main__":
    run_e1()
