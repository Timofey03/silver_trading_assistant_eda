"""E2b CROSS-ASSET + FEATURE SELECTION — устранение curse of dimensionality.

После E2 выяснилось: наивное добавление 84 фичей ухудшает результат
(curse of dimensionality, overfit). Здесь применяем feature selection через
mutual information к ограниченному набору ~25 топ-фичей.

Изменения от E2:
- Применяем SelectKBest с mutual_info_classif на каждом фолде train
- Выбираем top-K=25 фичей на каждом фолде (модель адаптируется к режиму)
- Это даёт лучшую обобщающую способность при сохранении cross-asset информации
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
from app.multi_asset.features import per_asset_features, cross_asset_features
from app.multi_asset.labels import build_multi_horizon_labels
from app.multi_asset.walkforward import WFConfig, FoldResult, accuracy_metrics
from app.multi_asset.simulator import simulate_trades, trades_to_df, TradeConfig
from app.multi_asset.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = CONFIG_ROOT / "baseline_outputs_multiasset" / "e2b_feature_selected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_cross_asset_features() -> pd.DataFrame:
    metals = load_metals()
    target_index = metals["silver"].index
    per_asset = [per_asset_features(df, prefix=m) for m, df in metals.items() if not df.empty]
    all_per_asset = pd.concat(per_asset, axis=1, sort=False).reindex(target_index)
    cross = cross_asset_features(metals).reindex(target_index)
    result = pd.concat([all_per_asset, cross], axis=1, sort=False)
    result.index.name = "date"
    return result


def walkforward_with_feature_selection(
    features: pd.DataFrame,
    labels: pd.Series,
    config: WFConfig,
    top_k: int = 25,
) -> tuple[pd.DataFrame, list[FoldResult], pd.DataFrame]:
    """Walk-forward с feature selection на каждом фолде.

    Returns:
        predictions, folds, feature_importance_history
    """
    common = features.index.intersection(labels.index)
    X = features.loc[common].copy()
    y = labels.loc[common].copy()
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    logger.info(f"WF with FS: {len(X):,} samples × {len(X.columns)} features → top-{top_k}")

    cfg = config
    n = len(X)
    embargo = max(1, int(cfg.embargo_ratio * cfg.horizon))

    folds_meta = []
    all_preds = []
    feature_counts = {col: 0 for col in X.columns}

    test_start_idx = max(cfg.min_train_size, cfg.train_window)
    fold_idx = 0

    while test_start_idx + cfg.test_window <= n:
        test_end_idx = test_start_idx + cfg.test_window
        train_end_idx = test_start_idx - cfg.horizon - embargo
        train_start_idx = max(0, train_end_idx - cfg.train_window)

        if train_end_idx - train_start_idx < cfg.min_train_size:
            test_start_idx += cfg.step
            continue

        X_train = X.iloc[train_start_idx:train_end_idx]
        y_train = y.iloc[train_start_idx:train_end_idx]
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]

        # === Feature selection через mutual information ===
        selector = SelectKBest(
            score_func=lambda Xs, ys: mutual_info_classif(Xs, ys, random_state=42),
            k=min(top_k, X_train.shape[1]),
        )
        selector.fit(X_train, y_train)
        selected_mask = selector.get_support()
        selected_cols = X_train.columns[selected_mask].tolist()

        # Track features
        for col in selected_cols:
            feature_counts[col] = feature_counts.get(col, 0) + 1

        # === Train model on selected features ===
        X_train_sel = X_train[selected_cols]
        X_test_sel = X_test[selected_cols]

        model = HistGradientBoostingClassifier(
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            max_iter=cfg.max_iter,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )
        model.fit(X_train_sel, y_train)
        proba = model.predict_proba(X_test_sel)
        classes = model.classes_

        preds = pd.DataFrame(index=X_test.index)
        for i, cls in enumerate(classes):
            preds[f"p_{int(cls)}"] = proba[:, i]
        preds["y_true"] = y_test.values
        preds["pred"] = classes[np.argmax(proba, axis=1)]

        folds_meta.append(FoldResult(
            fold_idx=fold_idx,
            train_start=X_train.index.min(),
            train_end=X_train.index.max(),
            test_start=X_test.index.min(),
            test_end=X_test.index.max(),
            n_train=len(X_train),
            n_test=len(X_test),
            predictions=preds,
        ))
        all_preds.append(preds)

        if fold_idx % 10 == 0:
            logger.info(
                f"  fold {fold_idx + 1}: train [{X_train.index.min().date()} → {X_train.index.max().date()}] | "
                f"selected top features: {selected_cols[:3]}..."
            )

        fold_idx += 1
        test_start_idx += cfg.step

    predictions = pd.concat(all_preds, axis=0).sort_index()
    predictions = predictions[~predictions.index.duplicated(keep="first")]

    # Feature importance: how often each feature was selected
    n_folds = len(folds_meta)
    fi = pd.DataFrame({
        "feature": list(feature_counts.keys()),
        "n_selected": list(feature_counts.values()),
        "frequency": [c / n_folds for c in feature_counts.values()],
    }).sort_values("frequency", ascending=False)

    return predictions, folds_meta, fi


def run_e2b() -> dict:
    logger.info("=" * 70)
    logger.info("E2b CROSS-ASSET + FEATURE SELECTION (mutual_info top-25)")
    logger.info("=" * 70)

    logger.info("\n[1/5] Building features...")
    features = build_cross_asset_features().dropna()
    logger.info(f"  Features: {len(features.columns)} cols × {len(features):,} rows")

    logger.info("\n[2/5] Building labels (horizon=20, non-adaptive)...")
    metals = load_metals()
    silver = metals["silver"]
    labels_full = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
        horizons=[20], adaptive=False,
    )
    labels = labels_full["label_20"]

    logger.info("\n[3/5] WF with feature selection (top-25 mutual info)...")
    cfg = WFConfig(
        train_window=1000, test_window=30, step=30, horizon=20,
        max_depth=6, learning_rate=0.05, max_iter=200,
        min_samples_leaf=30, random_state=42,
    )
    predictions, folds, fi = walkforward_with_feature_selection(
        features, labels, cfg, top_k=25,
    )
    predictions.to_parquet(OUTPUT_DIR / "predictions.parquet", compression="snappy")
    fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    acc = accuracy_metrics(predictions)
    logger.info(f"  OOS accuracy: {acc['accuracy']:.3f} (n={acc['n']})")

    logger.info("\n[4/5] Simulating trades...")
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

    metrics = compute_all_metrics(trades_df, n_trials=3) if not trades_df.empty else {}
    metrics["oos_accuracy"] = acc.get("accuracy", 0)
    metrics["n_predictions"] = acc.get("n", 0)
    metrics["n_folds"] = len(folds)
    metrics["experiment"] = "E2b_feature_selected"
    metrics["features_pool"] = len(features.columns)
    metrics["features_selected_per_fold"] = 25
    metrics["adaptive_barriers"] = False
    metrics["cross_asset"] = True

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Summary + comparison E1 vs E2 vs E2b
    logger.info("\n" + "=" * 70)
    logger.info("E2b RESULTS")
    logger.info("=" * 70)
    for key in ["n_trades", "trades_per_year", "period_years", "total_return",
                "annual_return", "sharpe", "sortino", "max_dd", "profit_factor",
                "win_rate", "mean_return", "best_trade", "worst_trade",
                "psr", "dsr", "oos_accuracy"]:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                if key.endswith("return") or key in ("max_dd", "best_trade", "worst_trade",
                                                     "mean_return", "median_return"):
                    logger.info(f"  {key:20s}: {val*100:+.2f}%")
                else:
                    logger.info(f"  {key:20s}: {val:.3f}")

    # Comparison
    logger.info("\n" + "=" * 70)
    logger.info("E1 → E2 → E2b PROGRESSION")
    logger.info("=" * 70)
    e1_path = CONFIG_ROOT / "baseline_outputs_multiasset" / "e1_baseline" / "metrics.json"
    e2_path = CONFIG_ROOT / "baseline_outputs_multiasset" / "e2_cross_asset" / "metrics.json"
    e1 = json.load(open(e1_path, encoding="utf-8")) if e1_path.exists() else {}
    e2 = json.load(open(e2_path, encoding="utf-8")) if e2_path.exists() else {}

    logger.info(f"  {'Metric':<20} {'E1':>10} {'E2':>10} {'E2b':>10}")
    for key in ["sharpe", "annual_return", "max_dd", "profit_factor",
                "win_rate", "n_trades", "oos_accuracy", "dsr"]:
        v1 = e1.get(key, None)
        v2 = e2.get(key, None)
        v3 = metrics.get(key, None)
        if isinstance(v1, float) and isinstance(v2, float) and isinstance(v3, float):
            if key in ("annual_return", "max_dd", "win_rate"):
                logger.info(f"  {key:<20} {v1*100:>9.1f}% {v2*100:>9.1f}% {v3*100:>9.1f}%")
            else:
                logger.info(f"  {key:<20} {v1:>10.3f} {v2:>10.3f} {v3:>10.3f}")
        else:
            logger.info(f"  {key:<20} {str(v1):>10} {str(v2):>10} {str(v3):>10}")

    logger.info(f"\nTop 10 selected features:")
    for _, row in fi.head(10).iterrows():
        logger.info(f"  {row['feature']:<40} {row['frequency']*100:5.1f}%")

    logger.info(f"\nSaved to: {OUTPUT_DIR}")
    return metrics


if __name__ == "__main__":
    run_e2b()
