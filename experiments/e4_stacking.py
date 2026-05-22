"""E4 STACKING ENSEMBLE — финальный эксперимент.

Архитектура:
  Level 0 (base models):
    - HistGradientBoostingClassifier (sklearn)
    - LightGBMClassifier
    - CatBoostClassifier

  Level 1 (meta-learner):
    - LogisticRegression на out-of-fold probabilities базовых моделей

Setup как в E3b winner:
  - Cross-asset features (102 cols)
  - Adaptive barriers (vol-scaled)
  - Feature selection top-30 (mutual info)
  - Horizon = 20 дней

Stacking использует internal 5-fold CV внутри каждого train fold,
чтобы получить unbiased OOF predictions для meta-learner.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.multi_asset.config import REPO_ROOT as CONFIG_ROOT
from app.multi_asset.metal_loader import load_metals
from app.multi_asset.macro_loader import load_macro, assemble_macro_frame
from app.multi_asset.features import per_asset_features, cross_asset_features
from app.multi_asset.labels import build_multi_horizon_labels
from app.multi_asset.walkforward import WFConfig, FoldResult, accuracy_metrics
from app.multi_asset.simulator import simulate_trades, trades_to_df, TradeConfig
from app.multi_asset.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = CONFIG_ROOT / "baseline_outputs_multiasset" / "e4_stacking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Base model factory
# =============================================================================

def make_base_models(random_state: int = 42):
    """Три разнородные модели для diversity."""
    return {
        "histgb": HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=200,
            min_samples_leaf=30, random_state=random_state,
        ),
        "lgbm": LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=30,
            random_state=random_state, verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            min_data_in_leaf=30, random_seed=random_state,
            verbose=False, allow_writing_files=False,
        ),
    }


# =============================================================================
# Build features (same as E3b)
# =============================================================================

def build_full_features() -> pd.DataFrame:
    metals = load_metals()
    target_index = metals["silver"].index

    per_asset = [per_asset_features(df, prefix=m)
                 for m, df in metals.items() if not df.empty]
    all_per_asset = pd.concat(per_asset, axis=1, sort=False).reindex(target_index)
    cross = cross_asset_features(metals).reindex(target_index)

    macro = load_macro()
    macro_frame = assemble_macro_frame(macro, target_index=target_index)

    result = pd.concat([all_per_asset, cross, macro_frame], axis=1, sort=False)
    result.index.name = "date"
    return result


# =============================================================================
# Stacking walk-forward
# =============================================================================

def walkforward_stacking(
    features: pd.DataFrame,
    labels: pd.Series,
    config: WFConfig,
    top_k: int = 30,
    n_inner_folds: int = 5,
) -> tuple[pd.DataFrame, list[FoldResult], pd.DataFrame]:
    """Walk-forward stacking ensemble.

    На каждом outer fold:
      1. Feature selection (top-K mutual info) на train
      2. Internal K-fold CV на train для получения OOF predictions
         от каждой base model
      3. Train meta-LR на OOF predictions
      4. Re-train base models на полном train fold
      5. Predict test fold через base models → meta-LR → final probability
    """
    common = features.index.intersection(labels.index)
    X = features.loc[common].copy()
    y = labels.loc[common].copy()
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    logger.info(
        f"Stacking WF: {len(X):,} samples × {len(X.columns)} → top-{top_k} | "
        f"{n_inner_folds}-fold inner CV"
    )

    cfg = config
    n = len(X)
    embargo = max(1, int(cfg.embargo_ratio * cfg.horizon))
    test_start_idx = max(cfg.min_train_size, cfg.train_window)
    fold_idx = 0

    folds_meta = []
    all_preds = []
    feature_counts = {col: 0 for col in X.columns}

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

        # ===== Step 1: Feature selection =====
        selector = SelectKBest(
            score_func=lambda Xs, ys: mutual_info_classif(Xs, ys, random_state=42),
            k=min(top_k, X_train.shape[1]),
        )
        selector.fit(X_train, y_train)
        sel_cols = X_train.columns[selector.get_support()].tolist()
        for c in sel_cols:
            feature_counts[c] = feature_counts.get(c, 0) + 1

        X_train_sel = X_train[sel_cols]
        X_test_sel = X_test[sel_cols]

        # Ensure label space is consistent
        classes = sorted(y_train.unique())
        n_classes = len(classes)
        class_map = {c: i for i, c in enumerate(classes)}

        # ===== Step 2: OOF predictions for meta-learner =====
        # KFold (без shuffle для сохранения временного порядка)
        kf = KFold(n_splits=n_inner_folds, shuffle=False)
        oof_preds = {name: np.zeros((len(X_train_sel), n_classes))
                     for name in ["histgb", "lgbm", "catboost"]}

        for inner_train_idx, inner_val_idx in kf.split(X_train_sel):
            X_inner_train = X_train_sel.iloc[inner_train_idx]
            y_inner_train = y_train.iloc[inner_train_idx]
            X_inner_val = X_train_sel.iloc[inner_val_idx]

            base_models = make_base_models(random_state=cfg.random_state)
            for name, model in base_models.items():
                try:
                    model.fit(X_inner_train, y_inner_train)
                    proba = model.predict_proba(X_inner_val)
                    # Выровнять proba под все возможные классы
                    aligned = np.zeros((len(X_inner_val), n_classes))
                    for j, cls in enumerate(model.classes_):
                        if cls in class_map:
                            aligned[:, class_map[cls]] = proba[:, j]
                    oof_preds[name][inner_val_idx] = aligned
                except Exception as e:
                    logger.warning(f"  inner fold {name} failed: {e}")

        # ===== Step 3: Train meta-LR on OOF =====
        # Stacking matrix: [n_train, n_base × n_classes]
        meta_X_train = np.hstack([oof_preds[name] for name in ["histgb", "lgbm", "catboost"]])
        meta_lr = LogisticRegression(
            max_iter=1000,
            random_state=cfg.random_state,
        )
        try:
            meta_lr.fit(meta_X_train, y_train)
        except Exception as e:
            logger.warning(f"  meta-LR failed: {e}")
            test_start_idx += cfg.step
            continue

        # ===== Step 4: Re-train base models on full train =====
        final_base_models = make_base_models(random_state=cfg.random_state)
        test_preds_per_model = []
        for name, model in final_base_models.items():
            try:
                model.fit(X_train_sel, y_train)
                proba = model.predict_proba(X_test_sel)
                aligned = np.zeros((len(X_test_sel), n_classes))
                for j, cls in enumerate(model.classes_):
                    if cls in class_map:
                        aligned[:, class_map[cls]] = proba[:, j]
                test_preds_per_model.append(aligned)
            except Exception as e:
                logger.warning(f"  base {name} test failed: {e}")
                test_preds_per_model.append(np.full((len(X_test_sel), n_classes),
                                                     1.0 / n_classes))

        # ===== Step 5: Meta-LR predict on test =====
        meta_X_test = np.hstack(test_preds_per_model)
        final_proba = meta_lr.predict_proba(meta_X_test)

        # Align final to canonical class order
        meta_classes = meta_lr.classes_
        aligned_final = np.zeros((len(X_test_sel), n_classes))
        for j, cls in enumerate(meta_classes):
            if cls in class_map:
                aligned_final[:, class_map[cls]] = final_proba[:, j]

        preds = pd.DataFrame(index=X_test.index)
        for i, cls in enumerate(classes):
            preds[f"p_{int(cls)}"] = aligned_final[:, i]
        preds["y_true"] = y_test.values
        preds["pred"] = np.array(classes)[np.argmax(aligned_final, axis=1)]

        folds_meta.append(FoldResult(
            fold_idx=fold_idx,
            train_start=X_train.index.min(), train_end=X_train.index.max(),
            test_start=X_test.index.min(), test_end=X_test.index.max(),
            n_train=len(X_train), n_test=len(X_test), predictions=preds,
        ))
        all_preds.append(preds)

        if fold_idx % 5 == 0:
            logger.info(
                f"  fold {fold_idx + 1}: train [{X_train.index.min().date()} → "
                f"{X_train.index.max().date()}] ({len(X_train)}) | "
                f"test [{X_test.index.min().date()} → {X_test.index.max().date()}]"
            )
        fold_idx += 1
        test_start_idx += cfg.step

    predictions = pd.concat(all_preds, axis=0).sort_index()
    predictions = predictions[~predictions.index.duplicated(keep="first")]

    fi = pd.DataFrame({
        "feature": list(feature_counts.keys()),
        "n_selected": list(feature_counts.values()),
        "frequency": [c / max(fold_idx, 1) for c in feature_counts.values()],
    }).sort_values("frequency", ascending=False)

    return predictions, folds_meta, fi


def run_e4() -> dict:
    logger.info("=" * 70)
    logger.info("E4 STACKING ENSEMBLE — HistGB + LGBM + CatBoost → meta-LR")
    logger.info("=" * 70)

    logger.info("\n[1/5] Building features (cross-asset + macro)...")
    features = build_full_features().dropna()
    logger.info(f"  Features: {len(features.columns)} cols × {len(features):,} clean rows")

    logger.info("\n[2/5] Building adaptive labels (horizon=20)...")
    silver = load_metals()["silver"]
    labels = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
        horizons=[20], adaptive=True,
    )["label_20"]

    logger.info("\n[3/5] Running stacking walk-forward...")
    cfg = WFConfig(
        train_window=1000, test_window=30, step=30, horizon=20,
        max_depth=6, learning_rate=0.05, max_iter=200,
        min_samples_leaf=30, random_state=42,
    )
    predictions, folds, fi = walkforward_stacking(
        features, labels, cfg, top_k=30, n_inner_folds=5,
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

    logger.info("\n[5/5] Computing metrics...")
    metrics = compute_all_metrics(trades_df, n_trials=7) if not trades_df.empty else {}
    metrics["oos_accuracy"] = acc.get("accuracy", 0)
    metrics["n_predictions"] = acc.get("n", 0)
    metrics["n_folds"] = len(folds)
    metrics["experiment"] = "E4_stacking"
    metrics["base_models"] = ["histgb", "lgbm", "catboost"]
    metrics["meta_learner"] = "logistic_regression"
    metrics["inner_cv_folds"] = 5
    metrics["features_pool"] = len(features.columns)
    metrics["features_top_k"] = 30
    metrics["adaptive_barriers"] = True

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("E4 RESULTS")
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

    # Full progression
    logger.info("\n" + "=" * 70)
    logger.info("FULL PROGRESSION E1 → E2b → E3b → E4")
    logger.info("=" * 70)

    all_metrics = {}
    for exp_name in ["e1_baseline", "e2b_feature_selected",
                     "e3b_adaptive", "e4_stacking"]:
        path = CONFIG_ROOT / "baseline_outputs_multiasset" / exp_name / "metrics.json"
        if path.exists():
            all_metrics[exp_name] = json.load(open(path, encoding="utf-8"))

    logger.info(f"  {'Metric':<18}" + " ".join(f"{n[:12]:>14}" for n in all_metrics.keys()))
    for key in ["sharpe", "annual_return", "max_dd", "profit_factor",
                "win_rate", "n_trades", "oos_accuracy", "psr", "dsr"]:
        row = [f"  {key:<18}"]
        for exp_name, m in all_metrics.items():
            v = m.get(key, "—")
            if isinstance(v, float):
                if key in ("annual_return", "max_dd", "win_rate"):
                    row.append(f"{v*100:>13.1f}%")
                else:
                    row.append(f"{v:>14.3f}")
            else:
                row.append(f"{str(v):>14}")
        logger.info("".join(row))

    logger.info(f"\nTop 10 features in E4:")
    for _, r in fi.head(10).iterrows():
        logger.info(f"  {r['feature']:<40} {r['frequency']*100:5.1f}%")

    logger.info(f"\nSaved to: {OUTPUT_DIR}")
    return metrics


if __name__ == "__main__":
    run_e4()
