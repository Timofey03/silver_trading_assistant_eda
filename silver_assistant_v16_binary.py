"""
Silver Trading Assistant v16 — Бинарная классификация + регуляризация + отбор признаков

Изменения vs v15 (на основании анализа дашборда):

Проблема: переобучение train 0.58 → valid 0.36, purged CV mean 0.353 ≈ baseline 0.333.

1. Бинарная задача: UP (1) vs NOT_UP (0) — DOWN+NEUTRAL слиты.
   Почему: 3-класс. задача переусложнена при 2608 строках и 65 признаках.
   Бинарный baseline = 0.50 (честнее измерить реальный edge).

2. Сильная регуляризация HistGradientBoosting:
   max_depth 4→3, max_leaf_nodes 15, learning_rate 0.04→0.02,
   min_samples_leaf 20→40, l2_reg 1.5→3.0, n_iter_no_change 30→40.

3. Асимметричная стоимость: class_weight={NOT_UP: 2.0, UP: 1.0}.
   Модель консервативнее предсказывает UP → меньше ложных сигналов → выше precision.

4. Отбор признаков: top-30 по permutation importance (valid, ROC AUC).
   65 признаков → 30: убирает шумовые факторы, снижает дисперсию.

Запуск:
  python silver_assistant_v16_binary.py
  streamlit run dashboard_app.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# UTF-8 stdout fix for Windows cp1251
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        balanced_accuracy_score, roc_auc_score, brier_score_loss,
    )
except ImportError:
    raise ImportError("pip install scikit-learn>=1.3.0")

# Переиспользуем из v15 (который подтягивает v14)
try:
    from silver_assistant_v15_regime_cot import (
        fetch_cot_silver, merge_cot_to_daily,
        add_vol_trend_regime,
        COT_FEATURES, COT_RELEASE_LAG,
        MIN_REGIME_SAMPLES, HORIZON_V15, EMBARGO_V15,
    )
    from silver_assistant_v14_main import (
        fetch_ohlc, fetch_fred, build_features,
        add_triple_barrier_labels, get_feature_cols,
        backtest_strategy, buy_and_hold_return, backtest_summary,
        purged_walk_forward_splits,
        wilson_ci, md_table, pct,
        SPLITS,
    )
    print("  v15/v14 функции загружены.")
except ImportError as e:
    raise ImportError(
        "Запустите сначала: python silver_assistant_v15_regime_cot.py\n"
        f"Детали: {e}"
    )

# ---------------------------------------------------------------------------
# Гиперпараметры v16
# ---------------------------------------------------------------------------

HORIZON_V16      = HORIZON_V15     # 15 дней (наследуем)
EMBARGO_V16      = EMBARGO_V15     # 15 дней
MAX_DEPTH        = 3               # было 4
MAX_LEAF_NODES   = 15              # было не ограничено (31 по умолчанию)
LEARNING_RATE    = 0.02            # было 0.04
MIN_SAMP_LEAF    = 40              # было 20
L2_REG           = 3.0             # было 1.5
N_ITER_NO_CHANGE = 40              # было 30
NOT_UP_WEIGHT    = 2.0             # вес класса NOT_UP (0) для асимметричной стоимости
TOP_FEATURES_N   = 30              # отбираем top-N признаков по permutation importance


# ---------------------------------------------------------------------------
# 1. Бинаризация меток
# ---------------------------------------------------------------------------

def binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет колонку tb_label_bin: UP=1, NOT_UP=0 (DOWN+NEUTRAL объединены).
    Оригинальный tb_label сохраняется для совместимости с backtest_strategy.
    """
    df = df.copy()
    mask = df["tb_label"].notna()
    df.loc[mask, "tb_label_bin"] = (df.loc[mask, "tb_label"] == "UP").astype(int)
    return df


def label_report_binary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in ["train", "valid", "test", "forward"]:
        d  = df[(df["split"] == s) & df["tb_label_bin"].notna()]
        n  = len(d)
        up = int(d["tb_label_bin"].sum()) if n else 0
        rows.append({
            "split":   s, "n": n,
            "UP":      up, "NOT_UP": n - up,
            "UP_rate": f"{up/n:.3f}" if n else "?",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Режимная ансамблевая модель (бинарная)
# ---------------------------------------------------------------------------

class RegimeEnsembleBinary:
    """
    Mixture-of-Experts для бинарной задачи UP vs NOT_UP.

    Ключевые отличия от v15:
    - Бинарные метки 0/1 вместо UP/NEUTRAL/DOWN
    - Усиленная регуляризация (depth=3, lr=0.02, l2=3.0)
    - Асимметричная стоимость: NOT_UP весит в NOT_UP_WEIGHT раз больше
      → модель требует больше уверенности для предсказания UP
      → ниже recall, выше precision (что важно для трейдинга)
    """

    def __init__(self, not_up_weight: float = NOT_UP_WEIGHT):
        self.not_up_weight = not_up_weight
        self.models:   Dict[str, CalibratedClassifierCV] = {}
        self.fallback: Optional[CalibratedClassifierCV]  = None
        self.classes_: Optional[np.ndarray]              = None

    def _build_base(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter          = 600,
            max_depth         = MAX_DEPTH,
            max_leaf_nodes    = MAX_LEAF_NODES,
            learning_rate     = LEARNING_RATE,
            min_samples_leaf  = MIN_SAMP_LEAF,
            l2_regularization = L2_REG,
            early_stopping    = True,
            validation_fraction = 0.15,
            n_iter_no_change  = N_ITER_NO_CHANGE,
            class_weight      = {0: self.not_up_weight, 1: 1.0},
            random_state      = 42,
        )

    def _calibrate(self, X: pd.DataFrame, y: np.ndarray) -> CalibratedClassifierCV:
        cal = CalibratedClassifierCV(self._build_base(), method="isotonic", cv=3)
        cal.fit(X, y)
        return cal

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, regimes: pd.Series,
    ) -> "RegimeEnsembleBinary":
        print("  Обучение: глобальный fallback (бинарный)...")
        self.fallback = self._calibrate(X, y)
        self.classes_ = self.fallback.classes_

        unique_regimes = regimes.value_counts()
        print(f"  Режимы в train: {dict(unique_regimes)}")
        for regime, n in unique_regimes.items():
            if n >= MIN_REGIME_SAMPLES:
                print(f"  Обучение: режим '{regime}' ({n} строк)...")
                mask = regimes == regime
                self.models[regime] = self._calibrate(X[mask], y[mask.values])
            else:
                print(f"  Режим '{regime}': {n} < {MIN_REGIME_SAMPLES} → fallback")
        return self

    def predict_proba(self, X: pd.DataFrame, regimes: pd.Series) -> np.ndarray:
        classes = list(self.classes_)
        result  = np.zeros((len(X), 2))
        for regime in set(regimes.unique()) | {"__fallback__"}:
            mask = (regimes == regime).values if regime != "__fallback__" \
                   else ~regimes.isin(self.models).values
            if not mask.any():
                continue
            model   = self.models.get(regime, self.fallback)
            proba   = model.predict_proba(X[mask])
            m_cls   = list(model.classes_)
            aligned = np.zeros((proba.shape[0], 2))
            for i, cls in enumerate(m_cls):
                if cls in classes:
                    aligned[:, classes.index(cls)] = proba[:, i]
            result[mask] = aligned
        return result

    def predict(self, X: pd.DataFrame, regimes: pd.Series) -> np.ndarray:
        proba = self.predict_proba(X, regimes)
        return self.classes_[np.argmax(proba, axis=1)]

    def p_up(self, X: pd.DataFrame, regimes: pd.Series) -> np.ndarray:
        """Вероятность UP (класс 1)."""
        proba = self.predict_proba(X, regimes)
        up_i  = list(self.classes_).index(1) if 1 in self.classes_ else 1
        return proba[:, up_i]


# ---------------------------------------------------------------------------
# 3. Отбор признаков по permutation importance
# ---------------------------------------------------------------------------

def select_top_features(
    model: RegimeEnsembleBinary,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_top: int = TOP_FEATURES_N,
) -> Tuple[List[str], pd.Series]:
    """
    Использует fallback-модель (глобальный CalibratedClassifierCV) для
    вычисления permutation importance на валидации.
    Возвращает (top-n_top признаков, Series всех важностей).
    """
    print(f"  Permutation importance: {X_val.shape[1]} признаков, 10 повторений...")
    result = permutation_importance(
        model.fallback, X_val, y_val,
        n_repeats=10, random_state=42,
        scoring="roc_auc",
    )
    imp    = pd.Series(result.importances_mean, index=X_val.columns)
    sorted_imp = imp.sort_values(ascending=False)
    top    = sorted_imp.head(n_top).index.tolist()
    print(f"  Топ-5: {top[:5]}")
    return top, sorted_imp


# ---------------------------------------------------------------------------
# 4. Метрики (бинарные)
# ---------------------------------------------------------------------------

def _get_regimes(df: pd.DataFrame) -> pd.Series:
    col = "trend_regime" if "trend_regime" in df.columns else "regime"
    if col in df.columns:
        return df[col].fillna("sideways")
    return pd.Series("sideways", index=df.index)


def evaluate_split_v16(
    df: pd.DataFrame, split_name: str,
    model: RegimeEnsembleBinary, feature_cols: List[str],
) -> Dict[str, object]:
    d = df[(df["split"] == split_name) & df["tb_label_bin"].notna()].copy()
    if d.empty:
        return {"split": split_name, "n": 0}

    X       = d[feature_cols]
    regimes = _get_regimes(d)
    y       = d["tb_label_bin"].values.astype(int)
    p       = model.p_up(X, regimes)
    pred    = (p >= 0.50).astype(int)

    return {
        "split":             split_name,
        "n":                 len(d),
        "label_up_rate":     float(y.mean()),
        "accuracy":          float((pred == y).mean()),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "auc":               float(roc_auc_score(y, p)),
        "brier":             float(brier_score_loss(y, p)),
    }


# ---------------------------------------------------------------------------
# 5. Guardrails (бинарные, логика аналогична v14/v15)
# ---------------------------------------------------------------------------

def compute_guardrails_binary(df: pd.DataFrame, split: str) -> Dict[str, object]:
    d       = df[df["split"] == split].copy()
    labeled = d[d["tb_label_bin"].notna()]
    sigs    = d[(d["signal"] == "BUY") & d["tb_label_bin"].notna()]
    n       = len(sigs)
    base    = float((labeled["tb_label_bin"] == 1).mean()) if len(labeled) else float("nan")

    if n == 0:
        return {
            "split": split, "n_signals": 0, "precision": float("nan"),
            "wilson_95_low": float("nan"), "base_up_rate": base,
            "lift_vs_base": float("nan"), "warning": "no_signals",
        }

    correct   = int((sigs["tb_label_bin"] == 1).sum())
    precision = correct / n
    lo, hi    = wilson_ci(correct, n)
    lift      = precision - base

    warns = []
    if n < 20:
        warns.append("small_sample")
    if lo <= base:
        warns.append("ci_lower_not_above_base")

    return {
        "split":          split,
        "n_signals":      n,
        "correct_over_n": f"{correct}/{n}",
        "precision":      round(precision, 6),
        "wilson_95_low":  round(lo, 6),
        "wilson_95_high": round(hi, 6),
        "base_up_rate":   round(base, 6),
        "lift_vs_base":   round(lift, 6),
        "warning":        ";".join(warns) if warns else "OK",
    }


# ---------------------------------------------------------------------------
# 6. Политика (упрощённая: только up_threshold + cooldown)
# ---------------------------------------------------------------------------

def apply_policy_v16(
    df: pd.DataFrame,
    model: RegimeEnsembleBinary,
    feature_cols: List[str],
    up_threshold: float = 0.55,
    cooldown: int = 15,
) -> pd.DataFrame:
    out     = df.copy().sort_index()
    X       = out[feature_cols]
    regimes = _get_regimes(out)

    p_up_arr = model.p_up(X, regimes)
    out["p_up"]         = p_up_arr
    out["p_down"]       = 1.0 - p_up_arr     # бинарный complement
    out["p_neutral"]    = 0.0                 # нет в бинарной модели
    out["regime_model"] = regimes.values

    raw = p_up_arr >= up_threshold

    signals, reasons = [], []
    last_buy = -9999
    for i, ok in enumerate(raw):
        if ok and i - last_buy > cooldown:
            signals.append("BUY");  reasons.append("binary_edge"); last_buy = i
        elif ok:
            signals.append("HOLD"); reasons.append("cooldown")
        else:
            signals.append("HOLD"); reasons.append("no_edge")

    out["signal"] = signals
    out["reason"] = reasons
    return out


def select_policy_v16(
    valid_df: pd.DataFrame,
    model: RegimeEnsembleBinary,
    feature_cols: List[str],
) -> dict:
    best_obj    = -float("inf")
    best_params: dict = {}

    for up_thr in [0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]:
        for cooldown in [5, 10, 15, 20]:
            tmp  = apply_policy_v16(valid_df, model, feature_cols, up_thr, cooldown)
            sigs = tmp[(tmp["signal"] == "BUY") & tmp["tb_label_bin"].notna()]
            n    = len(sigs)
            if n < 4:
                continue
            labeled  = tmp[tmp["tb_label_bin"].notna()]
            correct  = int((sigs["tb_label_bin"] == 1).sum())
            base     = float((labeled["tb_label_bin"] == 1).mean())
            prec     = correct / n
            lo, _    = wilson_ci(correct, n)
            lift     = prec - base
            obj      = (lo - base) + 0.003 * min(n, 20) if lift > 0 else -999
            if obj > best_obj:
                best_obj    = obj
                best_params = {"up_threshold": up_thr, "cooldown": cooldown}

    return best_params if best_params else {"up_threshold": 0.55, "cooldown": 15}


# ---------------------------------------------------------------------------
# 7. Purged CV (бинарный)
# ---------------------------------------------------------------------------

def purged_cv_binary(
    df: pd.DataFrame, feature_cols: List[str],
    n_train_years: int = 3, n_test_months: int = 6,
) -> pd.DataFrame:
    labeled   = df[df["tb_label_bin"].notna()].copy()
    wf_splits = purged_walk_forward_splits(
        labeled.index,
        n_train_years=n_train_years,
        n_test_months=n_test_months,
        embargo_days=EMBARGO_V16,
        horizon=HORIZON_V16,
    )
    rows = []
    for i, (tr_idx, te_idx) in enumerate(wf_splits):
        Xtr  = labeled.loc[tr_idx, feature_cols]
        ytr  = labeled.loc[tr_idx, "tb_label_bin"].values.astype(int)
        Xte  = labeled.loc[te_idx, feature_cols]
        yte  = labeled.loc[te_idx, "tb_label_bin"].values.astype(int)
        rgtr = _get_regimes(labeled.loc[tr_idx])
        rgte = _get_regimes(labeled.loc[te_idx])

        if len(Xtr) < MIN_REGIME_SAMPLES or Xte.empty:
            continue
        try:
            m = RegimeEnsembleBinary()
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xtr, ytr, rgtr)
                pred = m.predict(Xte, rgte)
            ba = balanced_accuracy_score(yte, pred)
            rows.append({
                "fold":         i,
                "train_end":    tr_idx[-1].date(),
                "test_start":   te_idx[0].date(),
                "n_train":      len(Xtr),
                "n_test":       len(Xte),
                "balanced_acc": ba,
            })
        except Exception as e:
            print(f"  fold {i} skipped: {type(e).__name__}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Утилиты
# ---------------------------------------------------------------------------

def split_name(date: pd.Timestamp) -> str:
    if date.year <= 2022: return "train"
    if date.year == 2023: return "valid"
    if date.year == 2024: return "test"
    return "forward"


# ---------------------------------------------------------------------------
# 9. Основной pipeline
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",   default="2013-01-01")
    ap.add_argument("--end",     default="2099-12-31")
    ap.add_argument("--out-dir", default="baseline_outputs_v16")
    args = ap.parse_args(argv)

    end = min(args.end, pd.Timestamp.today().strftime("%Y-%m-%d"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== v16: binary classification + регуляризация + отбор признаков ===")
    print(f"  depth={MAX_DEPTH}, leaf={MAX_LEAF_NODES}, lr={LEARNING_RATE}, "
          f"l2={L2_REG}, min_leaf={MIN_SAMP_LEAF}, NOT_UP_w={NOT_UP_WEIGHT}, top_feat={TOP_FEATURES_N}")

    # ---- Данные ----
    print("\n=== v16: загрузка OHLC ===")
    df = fetch_ohlc(args.start, end)
    print(f"  OHLC: {len(df)} строк, {df.index[0].date()} — {df.index[-1].date()}")

    print("\n=== v16: загрузка FRED ===")
    try:
        fred = fetch_fred(args.start, end)
        print(f"  FRED: {len(fred)} строк" if not fred.empty else "  FRED: пропущены")
    except Exception:
        fred = pd.DataFrame()

    print("\n=== v16: загрузка COT (CFTC) ===")
    start_year = pd.Timestamp(args.start).year
    end_year   = pd.Timestamp(end).year
    cot = fetch_cot_silver(start_year, end_year)
    if cot.empty:
        print("  COT: недоступны")
    else:
        print(f"  COT: {len(cot)} записей, {cot.index[0].date()} — {cot.index[-1].date()}")
        cot.to_csv(out / "v16_cot_raw.csv")

    # ---- Признаки ----
    print("\n=== v16: инженерия признаков ===")
    df = build_features(df, fred)
    df = add_vol_trend_regime(df)
    regime_counts = df["trend_regime"].value_counts().to_dict()
    print(f"  trend_regime: {regime_counts}")

    if not cot.empty:
        df = merge_cot_to_daily(df, cot, lag_days=COT_RELEASE_LAG)
        present_cot = [c for c in COT_FEATURES if c in df.columns]
        print(f"  COT признаки: {present_cot}")
    else:
        present_cot = []

    # ---- Triple-barrier + бинаризация ----
    print("\n=== v16: triple-barrier (horizon=15) + бинаризация ===")
    df = add_triple_barrier_labels(df, horizon=HORIZON_V16)
    df = binarize_labels(df)
    df["split"] = df.index.map(split_name)
    df.to_csv(out / "v16_full_data.csv")

    lb = label_report_binary(df)
    lb.to_csv(out / "v16_label_distribution.csv", index=False)
    print("  Бинарные метки (UP=1, NOT_UP=0):")
    print(lb.to_string(index=False))

    # ---- Список признаков ----
    base_features = get_feature_cols(df)
    all_features  = base_features + [c for c in present_cot if c not in base_features]
    print(f"\n  Признаков до отбора: {len(all_features)} (COT: {len(present_cot)})")

    train_df = df[(df["split"] == "train") & df["tb_label_bin"].notna()].copy()
    valid_df_raw = df[(df["split"] == "valid") & df["tb_label_bin"].notna()].copy()

    if len(train_df) < 200:
        raise RuntimeError(f"Недостаточно обучающих данных: {len(train_df)}")

    X_tr_all  = train_df[all_features]
    y_tr_bin  = train_df["tb_label_bin"].values.astype(int)
    r_tr      = _get_regimes(train_df)
    X_val_all = valid_df_raw[all_features]
    y_val_bin = valid_df_raw["tb_label_bin"].values.astype(int)

    # ---- Pass 1: обучение для отбора признаков ----
    print("\n=== v16: pass 1 — отбор признаков ===")
    print(f"  Обучение pass-1 на {len(X_tr_all)} строках...")
    model_p1 = RegimeEnsembleBinary()
    with contextlib.redirect_stdout(io.StringIO()):
        model_p1.fit(X_tr_all, y_tr_bin, r_tr)

    selected, imp_series = select_top_features(model_p1, X_val_all, y_val_bin, n_top=TOP_FEATURES_N)
    # Сохраняем все важности для дашборда
    imp_df = imp_series.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df.to_csv(out / "v16_feature_importance.csv", index=False)
    pd.DataFrame({"feature": selected, "rank": range(1, len(selected)+1)}).to_csv(
        out / "v16_selected_features.csv", index=False
    )
    print(f"  Отобрано: {len(selected)} признаков (из {len(all_features)})")

    # ---- Pass 2: финальная модель с top признаками ----
    print(f"\n=== v16: pass 2 — финальная модель (top-{TOP_FEATURES_N}) ===")
    X_tr_sel = train_df[selected]
    print(f"  Обучение: {len(X_tr_sel)} строк ({X_tr_sel.index[0].date()} — {X_tr_sel.index[-1].date()})")
    model = RegimeEnsembleBinary()
    model.fit(X_tr_sel, y_tr_bin, r_tr)
    print(f"  Классы: {model.classes_}")
    print(f"  Режимные модели: {list(model.models.keys())}")

    # ---- Метрики ----
    print("\n=== v16: метрики (бинарные, baseline=0.50) ===")
    cls_metrics = [evaluate_split_v16(df, s, model, selected)
                   for s in ["train", "valid", "test", "forward"]]
    cls_df = pd.DataFrame(cls_metrics)
    cls_df.to_csv(out / "v16_classifier_metrics.csv", index=False)
    cols = [c for c in ["split", "n", "balanced_accuracy", "auc", "brier"] if c in cls_df.columns]
    print(cls_df[cols].to_string(index=False))

    # ---- Политика ----
    print("\n=== v16: выбор политики (valid 2023) ===")
    valid_full    = df[df["split"] == "valid"].copy()
    policy_params = select_policy_v16(valid_full, model, selected)
    print(f"  Параметры: {policy_params}")
    policy_params.update({
        "horizon_days":    HORIZON_V16,
        "top_features_n":  TOP_FEATURES_N,
        "not_up_weight":   NOT_UP_WEIGHT,
        "regularization": {
            "max_depth": MAX_DEPTH, "max_leaf_nodes": MAX_LEAF_NODES,
            "learning_rate": LEARNING_RATE, "l2_reg": L2_REG,
            "min_samples_leaf": MIN_SAMP_LEAF,
        },
        "regime_models": list(model.models.keys()),
    })
    with open(out / "v16_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_params, f, indent=2, ensure_ascii=False)

    # ---- Применение политики ----
    all_df = apply_policy_v16(
        df, model, selected,
        policy_params["up_threshold"],
        policy_params["cooldown"],
    )
    all_df.to_csv(out / "v16_decisions_all.csv")

    # ---- Guardrails ----
    print("\n=== v16: guardrails ===")
    grd_rows   = [compute_guardrails_binary(all_df, s) for s in ["valid", "test", "forward"]]
    guardrails = pd.DataFrame(grd_rows)
    guardrails.to_csv(out / "v16_guardrails.csv", index=False)
    cols = ["split", "n_signals", "correct_over_n", "precision",
            "wilson_95_low", "base_up_rate", "lift_vs_base", "warning"]
    cols = [c for c in cols if c in guardrails.columns]
    print(guardrails[cols].to_string(index=False))

    # ---- Бэктест ----
    print("\n=== v16: бэктест + buy-and-hold ===")
    bt_rows = []
    for s in ["valid", "test", "forward"]:
        trades = backtest_strategy(all_df, s, HORIZON_V16)
        trades.to_csv(out / f"{s}_trades_v16.csv", index=False)
        all_df[all_df["split"] == s].to_csv(out / f"{s}_decisions_v16.csv")
        bnh     = buy_and_hold_return(all_df, s)
        summary = backtest_summary(trades, s, bnh)
        bt_rows.append(summary)
    bt_df = pd.DataFrame(bt_rows)
    bt_df.to_csv(out / "v16_backtest_report.csv", index=False)
    cols = [c for c in ["split", "n_trades", "sum_net_return", "win_rate",
                         "profit_factor", "buy_and_hold", "vs_bnh"] if c in bt_df.columns]
    print(bt_df[cols].to_string(index=False))

    # ---- Последние карточки ----
    cards = []
    for s in ["valid", "test", "forward"]:
        d = all_df[all_df["split"] == s].sort_index()
        if d.empty:
            continue
        r = d.iloc[-1]
        cards.append({
            "split":        s,
            "date":         r.name.date(),
            "silver_close": round(float(r.get("silver_close", float("nan"))), 2),
            "signal":       r.get("signal", "HOLD"),
            "reason":       r.get("reason", ""),
            "p_up":         round(float(r.get("p_up", float("nan"))), 4),
            "p_down":       round(float(r.get("p_down", float("nan"))), 4),
            "trend_regime": r.get("trend_regime", r.get("regime", "")),
        })
    pd.DataFrame(cards).to_csv(out / "v16_latest_signal_cards.csv", index=False)

    # ---- Purged CV ----
    print("\n=== v16: purged CV (бинарный, baseline=0.50) ===")
    wf_df = purged_cv_binary(df, selected)
    wf_df.to_csv(out / "v16_purged_wf_cv.csv", index=False)
    if not wf_df.empty:
        mean_ba = wf_df["balanced_acc"].mean()
        std_ba  = wf_df["balanced_acc"].std()
        n_above = (wf_df["balanced_acc"] > 0.50).sum()
        print(f"  Фолдов: {len(wf_df)}, mean balanced_acc: {mean_ba:.3f} +/- {std_ba:.3f}")
        print(f"  Фолдов выше 0.50 (baseline): {n_above}/{len(wf_df)}")
        print(wf_df.to_string(index=False))
    else:
        print("  CV пуст — все фолды пропущены")

    # ---- Сравнение v15 vs v16 ----
    v15_gr_path = Path("baseline_outputs_v15/v15_guardrails.csv")
    if v15_gr_path.exists():
        v15_gr = pd.read_csv(v15_gr_path)
        comp_rows = []
        for s in ["valid", "test", "forward"]:
            r15 = v15_gr[v15_gr["split"] == s]
            r16 = guardrails[guardrails["split"] == s]
            comp_rows.append({
                "split":           s,
                "v15_precision":   pct(r15["precision"].values[0])    if not r15.empty else "-",
                "v16_precision":   pct(r16["precision"].values[0])    if not r16.empty else "-",
                "v15_wilson_low":  pct(r15["wilson_95_low"].values[0]) if not r15.empty else "-",
                "v16_wilson_low":  pct(r16["wilson_95_low"].values[0]) if not r16.empty else "-",
                "v15_warning":     r15["warning"].values[0]            if not r15.empty else "-",
                "v16_warning":     r16["warning"].values[0]            if not r16.empty else "-",
            })
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(out / "v16_vs_v15_comparison.csv", index=False)
        print("\n=== Сравнение v15 vs v16 ===")
        print(comp_df.to_string(index=False))

    print(f"\n=== v16 завершён. Результаты: {out} ===")
    print("  Дашборд: streamlit run dashboard_app.py")


if __name__ == "__main__":
    main()
