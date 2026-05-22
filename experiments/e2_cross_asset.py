"""E2 CROSS-ASSET — добавляем фичи из gold/platinum/palladium/copper + ratios.

Изменения от E1:
- Features: silver per-asset (14) + другие 4 металла (4×14=56) + ratios (10) + corr (2) + composite (2) = ~84
- Labels: те же (horizon=20, non-adaptive) для честного сравнения
- Model: HistGradientBoosting (без изменений)
- WF: те же параметры

Цель: показать что cross-asset контекст улучшает предсказательную силу.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.multi_asset.config import REPO_ROOT as CONFIG_ROOT, METALS
from app.multi_asset.metal_loader import load_metals
from app.multi_asset.features import per_asset_features, cross_asset_features
from app.multi_asset.labels import build_multi_horizon_labels
from app.multi_asset.walkforward import WalkForwardEngine, WFConfig, accuracy_metrics
from app.multi_asset.simulator import simulate_trades, trades_to_df, TradeConfig
from app.multi_asset.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = CONFIG_ROOT / "baseline_outputs_multiasset" / "e2_cross_asset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_cross_asset_features() -> pd.DataFrame:
    """Все per-asset фичи 5 металлов + cross-asset ratios. Без macro."""
    metals = load_metals()
    if "silver" not in metals or metals["silver"].empty:
        raise RuntimeError("Silver data missing")

    target_index = metals["silver"].index

    per_asset = []
    for metal, df in metals.items():
        if df.empty:
            continue
        per_asset.append(per_asset_features(df, prefix=metal))

    all_per_asset = pd.concat(per_asset, axis=1, sort=False).reindex(target_index)
    cross = cross_asset_features(metals).reindex(target_index)

    result = pd.concat([all_per_asset, cross], axis=1, sort=False)
    result.index.name = "date"
    return result


def run_e2() -> dict:
    """Запустить E2 cross-asset эксперимент."""
    logger.info("=" * 70)
    logger.info("E2 CROSS-ASSET — 5 metals + ratios, no macro")
    logger.info("=" * 70)

    # Step 1: features
    logger.info("\n[1/5] Building cross-asset features...")
    features = build_cross_asset_features()
    features = features.dropna()
    logger.info(f"  Features: {len(features.columns)} cols × {len(features):,} clean rows")

    # Step 2: labels (те же что в E1 — для честного сравнения)
    logger.info("\n[2/5] Building labels (horizon=20, non-adaptive)...")
    metals = load_metals()
    silver = metals["silver"]
    labels_full = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
        horizons=[20],
        adaptive=False,  # ← FIXED для честного сравнения с E1
    )
    labels = labels_full["label_20"]

    # Step 3: walk-forward
    logger.info("\n[3/5] Running walk-forward...")
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
    predictions.to_parquet(OUTPUT_DIR / "predictions.parquet", compression="snappy")

    acc = accuracy_metrics(predictions)
    logger.info(f"  OOS accuracy: {acc['accuracy']:.3f} (n={acc['n']})")
    if "per_class" in acc:
        for cls, info in acc["per_class"].items():
            logger.info(f"    class={cls}: recall {info['recall']:.3f} (n={info['n']})")

    # Step 4: simulation
    logger.info("\n[4/5] Simulating trades...")
    prices = silver[["close", "high", "low"]].reindex(predictions.index)
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

    # Step 5: metrics
    metrics = compute_all_metrics(trades_df, n_trials=2) if not trades_df.empty else {}
    metrics["oos_accuracy"] = acc.get("accuracy", 0)
    metrics["n_predictions"] = acc.get("n", 0)
    metrics["n_folds"] = len(folds)
    metrics["experiment"] = "E2_cross_asset"
    metrics["features_used"] = len(features.columns)
    metrics["adaptive_barriers"] = False
    metrics["cross_asset"] = True
    metrics["multi_horizon"] = False
    metrics["macro_features"] = False

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("E2 RESULTS")
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

    # E1 vs E2 quick comparison
    logger.info("\n" + "=" * 70)
    logger.info("E1 vs E2 COMPARISON")
    logger.info("=" * 70)
    e1_path = CONFIG_ROOT / "baseline_outputs_multiasset" / "e1_baseline" / "metrics.json"
    if e1_path.exists():
        with open(e1_path, encoding="utf-8") as f:
            e1 = json.load(f)
        for key in ["sharpe", "win_rate", "annual_return", "max_dd", "profit_factor",
                    "n_trades", "oos_accuracy", "dsr"]:
            v1, v2 = e1.get(key, "n/a"), metrics.get(key, "n/a")
            if isinstance(v1, float) and isinstance(v2, float):
                delta = v2 - v1
                arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
                if key in ("annual_return", "max_dd", "win_rate"):
                    logger.info(f"  {key:20s}: {v1*100:+7.2f}%  →  {v2*100:+7.2f}%  "
                                f"{arrow} {delta*100:+.2f}pp")
                else:
                    logger.info(f"  {key:20s}: {v1:8.3f}  →  {v2:8.3f}  {arrow} {delta:+.3f}")

    logger.info(f"\nSaved to: {OUTPUT_DIR}")
    return metrics


if __name__ == "__main__":
    run_e2()
