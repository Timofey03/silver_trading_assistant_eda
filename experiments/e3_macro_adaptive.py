"""E3 — Macro features + Adaptive barriers + Meta-labeling.

Три инкрементальных эксперимента для атрибуции каждого компонента:

E3a: + macro features (TIPS, DXY, USDRUB, COT, VIX, oil, INDPRO, CPI, breakeven)
     Сравниваем с E2b — даёт ли макроконтекст реальный буст.

E3b: + adaptive barriers (volatility-scaled, regime-aware asymmetric)
     Метки динамически адаптируются под текущую vol.

E3c: + meta-labeling (фильтр weak signals второй моделью)
     López de Prado meta-labeling pattern.
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

from app.multi_asset.config import REPO_ROOT as CONFIG_ROOT, MACRO
from app.multi_asset.metal_loader import load_metals
from app.multi_asset.macro_loader import load_macro, assemble_macro_frame
from app.multi_asset.features import per_asset_features, cross_asset_features
from app.multi_asset.labels import build_multi_horizon_labels
from app.multi_asset.walkforward import WFConfig, FoldResult, accuracy_metrics
from app.multi_asset.simulator import simulate_trades, trades_to_df, TradeConfig
from app.multi_asset.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_ROOT = CONFIG_ROOT / "baseline_outputs_multiasset"


# =============================================================================
# Helpers
# =============================================================================

def build_full_features(include_macro: bool) -> pd.DataFrame:
    """Все cross-asset фичи + (опционально) macro."""
    metals = load_metals()
    target_index = metals["silver"].index

    per_asset = [per_asset_features(df, prefix=m) for m, df in metals.items() if not df.empty]
    all_per_asset = pd.concat(per_asset, axis=1, sort=False).reindex(target_index)
    cross = cross_asset_features(metals).reindex(target_index)

    parts = [all_per_asset, cross]
    if include_macro:
        macro = load_macro()
        macro_frame = assemble_macro_frame(macro, target_index=target_index)
        parts.append(macro_frame)

    result = pd.concat(parts, axis=1, sort=False)
    result.index.name = "date"
    return result


def walkforward_with_fs(
    features: pd.DataFrame,
    labels: pd.Series,
    config: WFConfig,
    top_k: int = 30,
) -> tuple[pd.DataFrame, list[FoldResult], pd.DataFrame]:
    """Walk-forward с feature selection top-K на каждом фолде."""
    common = features.index.intersection(labels.index)
    X = features.loc[common].copy()
    y = labels.loc[common].copy()
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    logger.info(f"WF+FS: {len(X):,} samples × {len(X.columns)} → top-{top_k}")

    n = len(X)
    embargo = max(1, int(config.embargo_ratio * config.horizon))
    test_start_idx = max(config.min_train_size, config.train_window)
    fold_idx = 0

    folds_meta = []
    all_preds = []
    feature_counts = {col: 0 for col in X.columns}

    while test_start_idx + config.test_window <= n:
        test_end_idx = test_start_idx + config.test_window
        train_end_idx = test_start_idx - config.horizon - embargo
        train_start_idx = max(0, train_end_idx - config.train_window)

        if train_end_idx - train_start_idx < config.min_train_size:
            test_start_idx += config.step
            continue

        X_train = X.iloc[train_start_idx:train_end_idx]
        y_train = y.iloc[train_start_idx:train_end_idx]
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]

        selector = SelectKBest(
            score_func=lambda Xs, ys: mutual_info_classif(Xs, ys, random_state=42),
            k=min(top_k, X_train.shape[1]),
        )
        selector.fit(X_train, y_train)
        sel_cols = X_train.columns[selector.get_support()].tolist()
        for c in sel_cols:
            feature_counts[c] = feature_counts.get(c, 0) + 1

        model = HistGradientBoostingClassifier(
            max_depth=config.max_depth, learning_rate=config.learning_rate,
            max_iter=config.max_iter, min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
        )
        model.fit(X_train[sel_cols], y_train)
        proba = model.predict_proba(X_test[sel_cols])
        classes = model.classes_

        preds = pd.DataFrame(index=X_test.index)
        for i, cls in enumerate(classes):
            preds[f"p_{int(cls)}"] = proba[:, i]
        preds["y_true"] = y_test.values
        preds["pred"] = classes[np.argmax(proba, axis=1)]

        folds_meta.append(FoldResult(
            fold_idx=fold_idx,
            train_start=X_train.index.min(), train_end=X_train.index.max(),
            test_start=X_test.index.min(), test_end=X_test.index.max(),
            n_train=len(X_train), n_test=len(X_test),
            predictions=preds,
        ))
        all_preds.append(preds)
        fold_idx += 1
        test_start_idx += config.step

    predictions = pd.concat(all_preds, axis=0).sort_index()
    predictions = predictions[~predictions.index.duplicated(keep="first")]

    fi = pd.DataFrame({
        "feature": list(feature_counts.keys()),
        "n_selected": list(feature_counts.values()),
        "frequency": [c / max(fold_idx, 1) for c in feature_counts.values()],
    }).sort_values("frequency", ascending=False)

    return predictions, folds_meta, fi


def walkforward_with_metalabel(
    features: pd.DataFrame,
    labels: pd.Series,
    config: WFConfig,
    top_k: int = 30,
    primary_threshold: float = 0.48,
    meta_threshold: float = 0.55,
) -> tuple[pd.DataFrame, list[FoldResult], pd.DataFrame]:
    """Двухэтапная схема López de Prado.

    Primary model: предсказывает направление {-1, 0, 1}.
    Meta model: на тех точках, где primary p_1 >= threshold, предсказывает
                является ли сигнал прибыльным (binary).

    Финальный сигнал: BUY iff (primary p_1 >= primary_thr) AND (meta p_win >= meta_thr).
    """
    common = features.index.intersection(labels.index)
    X = features.loc[common].copy()
    y = labels.loc[common].copy()
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    logger.info(f"WF+meta: {len(X):,} samples × {len(X.columns)} | "
                f"primary≥{primary_threshold} ∧ meta≥{meta_threshold}")

    n = len(X)
    embargo = max(1, int(config.embargo_ratio * config.horizon))
    test_start_idx = max(config.min_train_size, config.train_window)
    fold_idx = 0

    folds_meta = []
    all_preds = []
    feature_counts = {col: 0 for col in X.columns}

    while test_start_idx + config.test_window <= n:
        test_end_idx = test_start_idx + config.test_window
        train_end_idx = test_start_idx - config.horizon - embargo
        train_start_idx = max(0, train_end_idx - config.train_window)

        if train_end_idx - train_start_idx < config.min_train_size:
            test_start_idx += config.step
            continue

        X_train = X.iloc[train_start_idx:train_end_idx]
        y_train = y.iloc[train_start_idx:train_end_idx]
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]

        # === Step 1: Primary model with feature selection ===
        selector = SelectKBest(
            score_func=lambda Xs, ys: mutual_info_classif(Xs, ys, random_state=42),
            k=min(top_k, X_train.shape[1]),
        )
        selector.fit(X_train, y_train)
        sel_cols = X_train.columns[selector.get_support()].tolist()
        for c in sel_cols:
            feature_counts[c] = feature_counts.get(c, 0) + 1

        primary = HistGradientBoostingClassifier(
            max_depth=config.max_depth, learning_rate=config.learning_rate,
            max_iter=config.max_iter, min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
        )
        primary.fit(X_train[sel_cols], y_train)

        # Primary predictions для train (для построения meta-label) и test
        primary_train_proba = primary.predict_proba(X_train[sel_cols])
        primary_classes = primary.classes_
        p1_idx = list(primary_classes).index(1) if 1 in primary_classes else None
        if p1_idx is None:
            # Skip fold — нет положительного класса
            test_start_idx += config.step
            continue
        p1_train = primary_train_proba[:, p1_idx]

        # === Step 2: Build meta-training set ===
        # Берём только те train-наблюдения, где primary сказал BUY
        meta_train_mask = p1_train >= primary_threshold
        if meta_train_mask.sum() < 50:  # мало примеров — fallback
            primary_test_proba = primary.predict_proba(X_test[sel_cols])
            p1_test = primary_test_proba[:, p1_idx]
            preds = pd.DataFrame(index=X_test.index)
            for i, cls in enumerate(primary_classes):
                preds[f"p_{int(cls)}"] = primary_test_proba[:, i]
            preds["meta_p_win"] = 1.0  # без фильтра
            preds["y_true"] = y_test.values
            preds["pred"] = primary_classes[np.argmax(primary_test_proba, axis=1)]
            folds_meta.append(FoldResult(
                fold_idx=fold_idx, train_start=X_train.index.min(), train_end=X_train.index.max(),
                test_start=X_test.index.min(), test_end=X_test.index.max(),
                n_train=len(X_train), n_test=len(X_test), predictions=preds,
            ))
            all_preds.append(preds)
            fold_idx += 1
            test_start_idx += config.step
            continue

        # Meta-label: was the sygnal на самом деле прибыльным?
        # (1 if y_train == 1 при primary BUY signal)
        X_meta_train = X_train[sel_cols].iloc[np.where(meta_train_mask)[0]].copy()
        # Добавляем primary p_1 как feature
        X_meta_train["primary_p_up"] = p1_train[meta_train_mask]
        y_meta_train = (y_train.iloc[np.where(meta_train_mask)[0]] == 1).astype(int)

        meta = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=150,
            min_samples_leaf=20, random_state=config.random_state,
        )
        if y_meta_train.nunique() < 2:
            # Дегенерат — все signals в train одного класса
            meta_p_win_test = np.ones(len(X_test))
        else:
            meta.fit(X_meta_train, y_meta_train)
            primary_test_proba = primary.predict_proba(X_test[sel_cols])
            p1_test = primary_test_proba[:, p1_idx]

            X_meta_test = X_test[sel_cols].copy()
            X_meta_test["primary_p_up"] = p1_test
            meta_p_win_test = meta.predict_proba(X_meta_test)[:, 1]
            # ↑ вероятность класса 1 = "будет прибыльный"

        # Final test predictions
        primary_test_proba = primary.predict_proba(X_test[sel_cols])
        preds = pd.DataFrame(index=X_test.index)
        for i, cls in enumerate(primary_classes):
            preds[f"p_{int(cls)}"] = primary_test_proba[:, i]
        preds["meta_p_win"] = meta_p_win_test
        preds["y_true"] = y_test.values
        preds["pred"] = primary_classes[np.argmax(primary_test_proba, axis=1)]

        folds_meta.append(FoldResult(
            fold_idx=fold_idx, train_start=X_train.index.min(), train_end=X_train.index.max(),
            test_start=X_test.index.min(), test_end=X_test.index.max(),
            n_train=len(X_train), n_test=len(X_test), predictions=preds,
        ))
        all_preds.append(preds)
        fold_idx += 1
        test_start_idx += config.step

    predictions = pd.concat(all_preds, axis=0).sort_index()
    predictions = predictions[~predictions.index.duplicated(keep="first")]

    fi = pd.DataFrame({
        "feature": list(feature_counts.keys()),
        "n_selected": list(feature_counts.values()),
        "frequency": [c / max(fold_idx, 1) for c in feature_counts.values()],
    }).sort_values("frequency", ascending=False)

    return predictions, folds_meta, fi


def run_one_experiment(
    name: str,
    features: pd.DataFrame,
    labels: pd.Series,
    silver: pd.DataFrame,
    use_metalabel: bool = False,
    meta_threshold: float = 0.55,
    top_k: int = 30,
    n_trials: int = 5,
) -> dict:
    """Запустить один вариант эксперимента и сохранить результаты."""
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = WFConfig(
        train_window=1000, test_window=30, step=30, horizon=20,
        max_depth=6, learning_rate=0.05, max_iter=200,
        min_samples_leaf=30, random_state=42,
    )
    if use_metalabel:
        predictions, folds, fi = walkforward_with_metalabel(
            features, labels, cfg, top_k=top_k, meta_threshold=meta_threshold,
        )
        # Сигнал: BUY только если оба прохода прошли
        predictions["p_combined"] = predictions["p_1"] * predictions["meta_p_win"]
        # Симулятор использует p_1 — но фильтруем через meta:
        # эффективная вероятность = p_1 если meta>=thr, иначе 0
        predictions["p_1_effective"] = np.where(
            predictions["meta_p_win"] >= meta_threshold,
            predictions["p_1"],
            0.0,
        )
        # Переопределяем p_1 для simulator
        predictions["p_1_orig"] = predictions["p_1"]
        predictions["p_1"] = predictions["p_1_effective"]
    else:
        predictions, folds, fi = walkforward_with_fs(features, labels, cfg, top_k=top_k)

    predictions.to_parquet(out_dir / "predictions.parquet", compression="snappy")
    fi.to_csv(out_dir / "feature_importance.csv", index=False)

    acc = accuracy_metrics(predictions)
    prices = silver[["close", "high", "low"]].reindex(predictions.index)
    trade_cfg = TradeConfig(
        entry_threshold=0.48, exit_threshold=0.35,
        trail_pct=0.12, max_hold_days=30, cooldown_days=25,
        commission_pct=0.001, direction_label=1,
    )
    trades, _ = simulate_trades(predictions, prices, trade_cfg)
    trades_df = trades_to_df(trades)
    if not trades_df.empty:
        trades_df.to_csv(out_dir / "trades.csv", index=False)

    metrics = compute_all_metrics(trades_df, n_trials=n_trials) if not trades_df.empty else {}
    metrics["oos_accuracy"] = acc.get("accuracy", 0)
    metrics["n_predictions"] = acc.get("n", 0)
    metrics["n_folds"] = len(folds)
    metrics["experiment"] = name
    metrics["features_pool"] = len(features.columns)
    metrics["features_top_k"] = top_k

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


def run_all() -> dict[str, dict]:
    """Запустить E3a / E3b / E3c — поэтапное добавление компонентов."""
    results = {}

    silver = load_metals()["silver"]

    # ============================================================
    # E3a — Cross-asset + Macro (без adaptive, без meta)
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("E3a — Cross-asset + MACRO features (non-adaptive labels)")
    logger.info("=" * 70)

    features_full = build_full_features(include_macro=True).dropna()
    logger.info(f"  Features pool: {len(features_full.columns)} cols × {len(features_full):,} rows")

    labels_nonadaptive = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
        horizons=[20], adaptive=False,
    )["label_20"]

    results["e3a_macro"] = run_one_experiment(
        "e3a_macro", features_full, labels_nonadaptive, silver,
        use_metalabel=False, top_k=30, n_trials=4,
    )

    # ============================================================
    # E3b — + Adaptive barriers
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("E3b — Macro + ADAPTIVE barriers")
    logger.info("=" * 70)

    labels_adaptive = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
        horizons=[20], adaptive=True,
    )["label_20"]

    results["e3b_adaptive"] = run_one_experiment(
        "e3b_adaptive", features_full, labels_adaptive, silver,
        use_metalabel=False, top_k=30, n_trials=5,
    )

    # ============================================================
    # E3c — + Meta-labeling
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("E3c — Macro + Adaptive + META-LABELING")
    logger.info("=" * 70)

    results["e3c_metalabel"] = run_one_experiment(
        "e3c_metalabel", features_full, labels_adaptive, silver,
        use_metalabel=True, meta_threshold=0.55, top_k=30, n_trials=6,
    )

    # ============================================================
    # Final comparison
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("FULL PROGRESSION E1 → E2b → E3a → E3b → E3c")
    logger.info("=" * 70)

    all_metrics = {}
    for exp_name in ["e1_baseline", "e2b_feature_selected", "e3a_macro", "e3b_adaptive", "e3c_metalabel"]:
        path = OUTPUT_ROOT / exp_name / "metrics.json"
        if path.exists():
            all_metrics[exp_name] = json.load(open(path, encoding="utf-8"))

    header = "  " + " ".join(f"{n[:11]:>12}" for n in all_metrics.keys())
    logger.info(f"  {'Metric':<18}" + " ".join(f"{n[:11]:>12}" for n in all_metrics.keys()))
    for key in ["sharpe", "annual_return", "max_dd", "profit_factor",
                "win_rate", "n_trades", "oos_accuracy", "dsr", "psr"]:
        row = [f"  {key:<18}"]
        for exp_name, m in all_metrics.items():
            v = m.get(key, "—")
            if isinstance(v, float):
                if key in ("annual_return", "max_dd", "win_rate"):
                    row.append(f"{v*100:>11.1f}%")
                else:
                    row.append(f"{v:>12.3f}")
            else:
                row.append(f"{str(v):>12}")
        logger.info("".join(row))

    return results


if __name__ == "__main__":
    run_all()
