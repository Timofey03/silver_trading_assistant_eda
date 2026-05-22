"""FORWARD TEST 2025-2026 — фактическая OOS проверка E3b на свежих данных.

Это **самый честный** способ проверить модель:
1. Train: 2010-01-01 → 2024-12-31 (всё что было до 2025)
2. Test: 2025-01-01 → сегодня (≈16 месяцев, OOS)
3. Модель ОДИН РАЗ обучается на train, потом fixed → predict на test
4. Никаких обновлений во время теста (как реальный production без retraining)

Сравниваем с:
- V25 forward (тот же период, та же экспериментальная установка)
- Buy & Hold silver (купил 2025-01-01, держим до сегодня)

Это apples-to-apples сравнение E3b vs V25 в одинаковых условиях.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.multi_asset.config import REPO_ROOT as CONFIG_ROOT
from app.multi_asset.metal_loader import load_metals
from app.multi_asset.macro_loader import load_macro, assemble_macro_frame
from app.multi_asset.features import per_asset_features, cross_asset_features
from app.multi_asset.labels import build_multi_horizon_labels
from app.multi_asset.simulator import simulate_trades, trades_to_df, TradeConfig
from app.multi_asset.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = CONFIG_ROOT / "baseline_outputs_multiasset" / "forward_test_2025"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END_DATE = pd.Timestamp("2024-12-31")
TEST_START_DATE = pd.Timestamp("2025-01-01")


def build_features_for_test():
    """Полные features (cross-asset + macro) для всего периода."""
    metals = load_metals()
    target_index = metals["silver"].index

    per_asset = [per_asset_features(df, prefix=m) for m, df in metals.items()
                 if not df.empty]
    all_per_asset = pd.concat(per_asset, axis=1, sort=False).reindex(target_index)
    cross = cross_asset_features(metals).reindex(target_index)
    macro = load_macro()
    macro_frame = assemble_macro_frame(macro, target_index=target_index)
    result = pd.concat([all_per_asset, cross, macro_frame], axis=1, sort=False)
    result.index.name = "date"
    return result, metals["silver"]


def run_forward_test():
    logger.info("=" * 70)
    logger.info("FORWARD TEST 2025-2026 — Out-of-sample проверка E3b")
    logger.info("=" * 70)

    # Step 1: data
    logger.info("\n[1/5] Loading data...")
    features, silver = build_features_for_test()
    features_clean = features.dropna()
    logger.info(f"  Features: {len(features_clean):,} rows × {len(features_clean.columns)} cols")
    logger.info(f"  Period: {features_clean.index.min().date()} → {features_clean.index.max().date()}")

    # Step 2: labels (adaptive)
    logger.info("\n[2/5] Building adaptive labels (horizon=20)...")
    labels_all = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
        horizons=[20], adaptive=True,
    )["label_20"]

    # Step 3: split train/test
    logger.info(f"\n[3/5] Splitting train (≤{TRAIN_END_DATE.date()}) / "
                f"test (≥{TEST_START_DATE.date()})...")

    common = features_clean.index.intersection(labels_all.index)
    X = features_clean.loc[common]
    y = labels_all.loc[common]
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    train_mask = X.index <= TRAIN_END_DATE
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[~train_mask], y.loc[~train_mask]

    # Purge: убираем последние 20+1 дней train (horizon=20 + embargo=1)
    purge_idx = X_train.index < (TEST_START_DATE - pd.Timedelta(days=30))
    X_train, y_train = X_train.loc[purge_idx], y_train.loc[purge_idx]

    logger.info(f"  Train: {len(X_train):,} rows ({X_train.index.min().date()} → "
                f"{X_train.index.max().date()})")
    logger.info(f"  Test:  {len(X_test):,} rows ({X_test.index.min().date()} → "
                f"{X_test.index.max().date()})")
    logger.info(f"  Test label distribution: {y_test.value_counts().to_dict()}")

    # Step 4: feature selection + train model
    logger.info("\n[4/5] Training model (feature selection top-30)...")
    selector = SelectKBest(
        score_func=lambda Xs, ys: mutual_info_classif(Xs, ys, random_state=42),
        k=min(30, X_train.shape[1]),
    )
    selector.fit(X_train, y_train)
    sel_cols = X_train.columns[selector.get_support()].tolist()
    logger.info(f"  Selected: {len(sel_cols)} features")
    logger.info(f"  Top 10: {sel_cols[:10]}")

    model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=200,
        min_samples_leaf=30, random_state=42,
    )
    model.fit(X_train[sel_cols], y_train)
    logger.info("  Model trained")

    # Step 5: predict on test
    proba = model.predict_proba(X_test[sel_cols])
    classes = model.classes_

    predictions = pd.DataFrame(index=X_test.index)
    for i, cls in enumerate(classes):
        predictions[f"p_{int(cls)}"] = proba[:, i]
    predictions["y_true"] = y_test.values
    predictions["pred"] = classes[np.argmax(proba, axis=1)]

    predictions.to_parquet(OUTPUT_DIR / "predictions.parquet", compression="snappy")
    logger.info(f"  Saved {len(predictions):,} predictions")

    # Accuracy
    acc = (predictions["y_true"] == predictions["pred"]).mean()
    logger.info(f"  OOS accuracy: {acc:.3f}")

    # Step 6: simulate trades
    logger.info("\n[5/5] Simulating trades...")
    prices = silver[["close", "high", "low"]].reindex(predictions.index)
    trade_cfg = TradeConfig(
        entry_threshold=0.48, exit_threshold=0.35,
        trail_pct=0.12, max_hold_days=30, cooldown_days=25,
        commission_pct=0.001, direction_label=1,
    )
    trades, _ = simulate_trades(predictions, prices, trade_cfg)
    trades_df = trades_to_df(trades)
    if not trades_df.empty:
        trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
        logger.info(f"  Trades: {len(trades_df)}")

    # Metrics
    metrics = compute_all_metrics(trades_df, n_trials=1) if not trades_df.empty else {}
    metrics["test_period_start"] = TEST_START_DATE.isoformat()
    metrics["test_period_end"] = X_test.index.max().isoformat()
    metrics["train_size"] = len(X_train)
    metrics["test_size"] = len(X_test)
    metrics["features_selected"] = len(sel_cols)
    metrics["selected_features"] = sel_cols
    metrics["oos_accuracy"] = float(acc)
    metrics["experiment"] = "forward_test_2025"

    # Buy & Hold benchmark
    silver_test = silver.loc[TEST_START_DATE:X_test.index.max()]
    bh_return = (silver_test["close"].iloc[-1] / silver_test["close"].iloc[0]) - 1
    metrics["bh_silver_return"] = float(bh_return)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FORWARD TEST RESULTS")
    logger.info("=" * 70)
    for key in ["n_trades", "trades_per_year", "period_years", "total_return",
                "annual_return", "sharpe", "max_dd", "profit_factor",
                "win_rate", "best_trade", "worst_trade", "oos_accuracy"]:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                if key in ("total_return", "annual_return", "max_dd", "win_rate",
                           "best_trade", "worst_trade"):
                    logger.info(f"  {key:20s}: {val*100:+.2f}%")
                else:
                    logger.info(f"  {key:20s}: {val:.3f}")

    # Comparison with V25 forward
    logger.info("\n" + "=" * 70)
    logger.info("E3b forward test vs V25 forward (SAME PERIOD 2025-2026)")
    logger.info("=" * 70)
    v25_path = REPO_ROOT / "baseline_outputs_v25" / "v25_forward_trades.csv"
    if v25_path.exists():
        v25 = pd.read_csv(v25_path)
        v25["entry_date"] = pd.to_datetime(v25["entry_date"])
        v25["exit_date"] = pd.to_datetime(v25["exit_date"])
        v25_metrics = compute_all_metrics(v25, n_trials=1)

        logger.info(f"  {'Metric':<20} {'E3b':>15} {'V25 forward':>15} {'B&H silver':>15}")
        logger.info("-" * 70)
        for key, label in [
            ("n_trades",       "Trades"),
            ("total_return",   "Total return"),
            ("annual_return",  "Annual return"),
            ("sharpe",         "Sharpe"),
            ("max_dd",         "Max DD"),
            ("win_rate",       "Win rate"),
            ("profit_factor",  "Profit factor"),
        ]:
            e3b_v = metrics.get(key, 0)
            v25_v = v25_metrics.get(key, 0)
            bh_v = metrics["bh_silver_return"] if key == "total_return" else "—"

            if key in ("total_return", "annual_return", "max_dd", "win_rate"):
                e3b_s = f"{e3b_v*100:+.1f}%"
                v25_s = f"{v25_v*100:+.1f}%"
                bh_s = f"{bh_v*100:+.1f}%" if isinstance(bh_v, float) else bh_v
            else:
                e3b_s = f"{e3b_v:.3f}" if isinstance(e3b_v, float) else str(e3b_v)
                v25_s = f"{v25_v:.3f}" if isinstance(v25_v, float) else str(v25_v)
                bh_s = bh_v if isinstance(bh_v, str) else f"{bh_v:.3f}"
            logger.info(f"  {label:<20} {e3b_s:>15} {v25_s:>15} {bh_s:>15}")

    logger.info(f"\nB&H silver returned {metrics['bh_silver_return']*100:+.1f}% за тот же период")
    logger.info(f"\nSaved to: {OUTPUT_DIR}")
    return metrics


if __name__ == "__main__":
    run_forward_test()
