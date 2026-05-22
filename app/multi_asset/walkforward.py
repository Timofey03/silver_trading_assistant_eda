"""Walk-forward engine с purging и embargo.

Логика:
- Sliding window training (фиксированный размер train окна)
- Прямая хронологическая последовательность (нет leakage)
- Purging: убираем из train последние H дней перед test (H = horizon метки)
- Embargo: убираем из train первые K дней после test (K ~ 5% от horizon)

Использование:
    engine = WalkForwardEngine(features, labels, ...)
    predictions, fold_info = engine.run()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger(__name__)


@dataclass
class WFConfig:
    train_window: int = 1000           # размер обучающего окна (дней)
    test_window: int = 30              # размер тестового окна (дней)
    step: int = 30                     # шаг сдвига между фолдами
    horizon: int = 20                  # длительность label (для purging)
    embargo_ratio: float = 0.05        # доля horizon для embargo
    min_train_size: int = 500          # минимальный train для warm-up

    # Model hyperparameters
    max_depth: int = 6
    learning_rate: float = 0.05
    max_iter: int = 200
    min_samples_leaf: int = 30
    random_state: int = 42


@dataclass
class FoldResult:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


class WalkForwardEngine:
    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        config: WFConfig | None = None,
    ):
        """
        Args:
            features: DataFrame с фичами, индексирован по дате. БЕЗ NaN.
            labels: Series меток (-1/0/+1), совпадающий по индексу с features.
            config: параметры walk-forward
        """
        self.config = config or WFConfig()

        # Выравнивание features и labels по общему индексу
        common = features.index.intersection(labels.index)
        self.features = features.loc[common].copy()
        self.labels = labels.loc[common].copy()

        # Убираем строки где label = NaN
        mask = self.labels.notna()
        self.features = self.features.loc[mask]
        self.labels = self.labels.loc[mask]

        logger.info(
            f"WalkForward initialized: "
            f"{len(self.features):,} samples × {len(self.features.columns)} features"
        )
        logger.info(f"  Label distribution: {self.labels.value_counts().to_dict()}")

    def _generate_fold_indices(self) -> list[dict]:
        """Сгенерировать индексы для всех фолдов."""
        cfg = self.config
        n = len(self.features)
        embargo = max(1, int(cfg.embargo_ratio * cfg.horizon))

        folds = []
        # Начинаем с того момента, когда есть хотя бы min_train_size данных
        test_start_idx = max(cfg.min_train_size, cfg.train_window)

        fold_idx = 0
        while test_start_idx + cfg.test_window <= n:
            test_end_idx = test_start_idx + cfg.test_window

            # Train window: фиксированный sliding [test_start - train_window, test_start)
            train_end_idx = test_start_idx - cfg.horizon - embargo  # PURGE
            train_start_idx = max(0, train_end_idx - cfg.train_window)

            if train_end_idx - train_start_idx < cfg.min_train_size:
                test_start_idx += cfg.step
                continue

            folds.append({
                "fold_idx": fold_idx,
                "train_start_idx": train_start_idx,
                "train_end_idx": train_end_idx,
                "test_start_idx": test_start_idx,
                "test_end_idx": test_end_idx,
            })

            fold_idx += 1
            test_start_idx += cfg.step

        return folds

    def _train_predict_fold(self, fold: dict) -> FoldResult:
        """Обучить модель и сделать предсказания для одного фолда."""
        cfg = self.config

        X = self.features
        y = self.labels

        X_train = X.iloc[fold["train_start_idx"]:fold["train_end_idx"]]
        y_train = y.iloc[fold["train_start_idx"]:fold["train_end_idx"]]
        X_test = X.iloc[fold["test_start_idx"]:fold["test_end_idx"]]
        y_test = y.iloc[fold["test_start_idx"]:fold["test_end_idx"]]

        model = HistGradientBoostingClassifier(
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            max_iter=cfg.max_iter,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )
        model.fit(X_train, y_train)

        # predict_proba — массив [n_test, n_classes]
        proba = model.predict_proba(X_test)
        classes = model.classes_

        # Собираем результат
        preds = pd.DataFrame(index=X_test.index)
        for i, cls in enumerate(classes):
            preds[f"p_{int(cls)}"] = proba[:, i]
        preds["y_true"] = y_test.values
        preds["pred"] = classes[np.argmax(proba, axis=1)]

        return FoldResult(
            fold_idx=fold["fold_idx"],
            train_start=X_train.index.min(),
            train_end=X_train.index.max(),
            test_start=X_test.index.min(),
            test_end=X_test.index.max(),
            n_train=len(X_train),
            n_test=len(X_test),
            predictions=preds,
        )

    def run(self, verbose: bool = True) -> tuple[pd.DataFrame, list[FoldResult]]:
        """Запустить полный walk-forward.

        Returns:
            predictions: DataFrame со всеми предсказаниями
            fold_results: список FoldResult с метаданными
        """
        folds = self._generate_fold_indices()
        logger.info(f"Total folds: {len(folds)}")

        results = []
        all_preds = []

        for fold in folds:
            result = self._train_predict_fold(fold)
            results.append(result)
            all_preds.append(result.predictions)
            if verbose and result.fold_idx % 10 == 0:
                logger.info(
                    f"  fold {result.fold_idx + 1}/{len(folds)}: "
                    f"train [{result.train_start.date()} → {result.train_end.date()}] "
                    f"({result.n_train}) | "
                    f"test [{result.test_start.date()} → {result.test_end.date()}] "
                    f"({result.n_test})"
                )

        predictions = pd.concat(all_preds, axis=0).sort_index()
        # Уберём дубли по индексу (если step < test_window — могут перекрываться)
        predictions = predictions[~predictions.index.duplicated(keep="first")]

        logger.info(f"WalkForward complete: {len(predictions):,} predictions")
        return predictions, results


def accuracy_metrics(predictions: pd.DataFrame) -> dict:
    """Базовые классификационные метрики на out-of-sample predictions."""
    y_true = predictions["y_true"]
    y_pred = predictions["pred"]
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {"accuracy": 0, "n": 0}

    acc = (y_true[mask] == y_pred[mask]).mean()

    # Per-class accuracy
    per_class = {}
    for cls in sorted(y_true.dropna().unique()):
        cls_mask = mask & (y_true == cls)
        if cls_mask.sum() > 0:
            per_class[int(cls)] = {
                "n": int(cls_mask.sum()),
                "recall": float((y_pred[cls_mask] == cls).mean()),
            }

    return {
        "accuracy": float(acc),
        "n": int(mask.sum()),
        "per_class": per_class,
    }
