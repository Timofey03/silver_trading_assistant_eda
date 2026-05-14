"""
Silver Trading Assistant v18 — Адаптивное расширяющееся окно + временной декай

Изменения vs v17 (на основании анализа дашборда):

ДИАГНОЗ: Distribution shift — модель обучена при UP_rate=43% (2013–2022),
а рынок 2024 (UP=55%) и 2025 (UP=66%) — устойчивый бычий тренд.
Следствие: NOT_UP_weight=2.0 подавляет P(UP) → модель пропускает рост,
на test lift=-5.3pp (хуже случайного!).

1. Расширяющееся окно (expanding window):
   - valid-модель:   train на 2013-2022  (без изменений)
   - test-модель:    train на 2013-2023  (+1 год valid)
   - forward-модель: train на 2013-2024  (+2 года valid+test)
   Каждый период получает модель, обученную на самых свежих данных.

2. Адаптивный класс-вес NOT_UP_weight:
   Вместо фиксированного 2.0:
   adaptive_weight = clip(2.0 × 0.43 / recent_2y_UP_rate, 1.0, 3.5)
   - При UP=43%: weight=2.0 (базовый)
   - При UP=55%: weight≈1.56  ← test 2024
   - При UP=66%: weight≈1.30  ← forward 2025
   Модель становится менее пессимистичной в бычьем режиме.

3. Экспоненциальный временной декай (sample_weight):
   w_i = exp(-λ × days_back),  λ = ln(2) / (3y × 252 дня)
   Последние 3 года весят в 2× больше чем данные 6-летней давности.
   Адаптирует к смене режима без выброса истории.

4. Расширенный поиск политики:
   Пороги 0.42–0.65 (было 0.48–0.65), cooldown 7-20 (было 5-20).
   Минимум сигналов = 3 (было 4) → больший охват при малых выборках.

Запуск:
  python silver_assistant_v18_adaptive.py
  streamlit run dashboard_app.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# UTF-8 stdout fix
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss
    from sklearn.inspection import permutation_importance
except ImportError:
    raise ImportError("pip install scikit-learn>=1.3.0")

# ---- Импорты из v17 ----
try:
    from silver_assistant_v17_fred import (
        fetch_macro_yfinance, add_macro_features,
        MACRO_FEATURE_NAMES, MACRO_TICKERS,
        HORIZON_V17, EMBARGO_V17,
    )
    print("  v17 функции загружены.")
except ImportError as e:
    raise ImportError(f"v17 не найден: {e}")

# ---- Импорты из v16 ----
try:
    from silver_assistant_v16_binary import (
        RegimeEnsembleBinary,
        binarize_labels, label_report_binary,
        select_top_features, evaluate_split_v16,
        compute_guardrails_binary,
        apply_policy_v16,
        _get_regimes, split_name,
        TOP_FEATURES_N, NOT_UP_WEIGHT,
        MAX_DEPTH, MAX_LEAF_NODES, LEARNING_RATE,
        MIN_SAMP_LEAF, L2_REG, N_ITER_NO_CHANGE,
    )
    print("  v16 функции загружены.")
except ImportError as e:
    raise ImportError(f"v16 не найден: {e}")

# ---- Импорты из v15 ----
try:
    from silver_assistant_v15_regime_cot import (
        fetch_cot_silver, merge_cot_to_daily,
        add_vol_trend_regime,
        COT_FEATURES, COT_RELEASE_LAG, MIN_REGIME_SAMPLES,
    )
    print("  v15 функции загружены.")
except ImportError as e:
    raise ImportError(f"v15 не найден: {e}")

# ---- Импорты из v14 ----
try:
    from silver_assistant_v14_main import (
        fetch_ohlc, build_features,
        add_triple_barrier_labels, get_feature_cols,
        backtest_strategy, buy_and_hold_return, backtest_summary,
        purged_walk_forward_splits,
        wilson_ci, pct,
    )
    print("  v14 функции загружены.")
except ImportError as e:
    raise ImportError(f"v14 не найден: {e}")


# ---------------------------------------------------------------------------
# Гиперпараметры v18
# ---------------------------------------------------------------------------

HORIZON_V18         = HORIZON_V17        # 15 дней
EMBARGO_V18         = EMBARGO_V17        # 15 дней
HALFLIFE_YEARS      = 3.0               # полураспад весов (3 года ≈ 756 торговых дней)
RECENT_WEIGHT_YEARS = 2                 # период для adaptive NOT_UP_weight
HISTORICAL_UP_RATE  = 0.43             # базовый UP_rate из обучающей выборки 2013-2022

# Граничные даты расширяющегося окна (включительно)
EXPAND_CUTOFFS: Dict[str, str] = {
    "valid":   "2022-12-31",   # модель видит только 2013-2022
    "test":    "2023-12-31",   # + valid 2023
    "forward": "2024-12-31",   # + test 2024
}


# ---------------------------------------------------------------------------
# 1. RegimeEnsembleV18 — расширение v16 с поддержкой sample_weight
# ---------------------------------------------------------------------------

class RegimeEnsembleV18(RegimeEnsembleBinary):
    """
    v18: добавляет поддержку sample_weight (временной декай) и
    принимает adaptive not_up_weight при создании.

    Всё остальное наследуется от RegimeEnsembleBinary (v16).
    """

    def _calibrate(
        self, X: pd.DataFrame, y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> CalibratedClassifierCV:
        cal = CalibratedClassifierCV(self._build_base(), method="isotonic", cv=3)
        if sample_weight is not None:
            cal.fit(X, y, sample_weight=sample_weight)
        else:
            cal.fit(X, y)
        return cal

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, regimes: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "RegimeEnsembleV18":
        print("  Обучение: глобальный fallback (v18 + weight)...")
        self.fallback = self._calibrate(X, y, sample_weight)
        self.classes_ = self.fallback.classes_

        unique_regimes = regimes.value_counts()
        print(f"  Режимы: {dict(unique_regimes)}")
        for regime, n in unique_regimes.items():
            if n >= MIN_REGIME_SAMPLES:
                mask      = regimes == regime
                mask_arr  = mask.values
                sw_regime = sample_weight[mask_arr] if sample_weight is not None else None
                self.models[regime] = self._calibrate(
                    X[mask], y[mask_arr], sw_regime
                )
                print(f"  Режим '{regime}': {n} строк, weight={'decay' if sw_regime is not None else 'none'}")
            else:
                print(f"  Режим '{regime}': {n} < {MIN_REGIME_SAMPLES} → fallback")
        return self


# ---------------------------------------------------------------------------
# 2. Временной декай и адаптивный вес
# ---------------------------------------------------------------------------

def compute_sample_weights(df: pd.DataFrame, halflife_years: float = HALFLIFE_YEARS) -> np.ndarray:
    """
    Экспоненциальный временной декай: w_i = exp(-λ × days_back).
    λ = ln(2) / (halflife_years × 252).
    Нормировка к среднему=1 для стабильности обучения.
    """
    max_date  = df.index.max()
    days_back = (max_date - df.index).days.astype(float).values
    lambda_   = math.log(2) / (halflife_years * 252)
    weights   = np.exp(-lambda_ * days_back)
    weights  /= weights.mean()          # normalize → mean=1
    return weights.astype(float)


def compute_adaptive_weight(
    train_df: pd.DataFrame,
    recent_years: int = RECENT_WEIGHT_YEARS,
    base_weight: float = NOT_UP_WEIGHT,
    hist_up_rate: float = HISTORICAL_UP_RATE,
) -> float:
    """
    Адаптирует NOT_UP_weight к текущему рыночному режиму.

    Формула: weight = clip(base_weight × hist_up_rate / recent_up_rate, 1.0, 3.5)
    - Бычий рынок (UP_rate↑) → weight↓ → меньше штраф за NOT_UP
    - Медвежий рынок (UP_rate↓) → weight↑ → больше штраф за NOT_UP
    """
    cutoff = train_df.index.max() - pd.Timedelta(days=recent_years * 365)
    recent = train_df[train_df.index >= cutoff]
    if len(recent) < 50:
        recent_up_rate = hist_up_rate
    else:
        recent_up_rate = float(recent["tb_label_bin"].mean())
    weight = float(np.clip(base_weight * hist_up_rate / max(recent_up_rate, 0.25), 1.0, 3.5))
    return round(weight, 2)


# ---------------------------------------------------------------------------
# 3. Обучение модели с расширяющимся окном
# ---------------------------------------------------------------------------

def train_expanding_model(
    df: pd.DataFrame,
    cutoff_date: str,
    feature_cols: List[str],
    embargo_days: int = EMBARGO_V18,
) -> Tuple[RegimeEnsembleV18, float, np.ndarray]:
    """
    Обучает RegimeEnsembleV18 на всех данных до cutoff_date - embargo.
    Возвращает (model, adaptive_weight, sample_weights).
    """
    cutoff   = pd.Timestamp(cutoff_date) - pd.Timedelta(days=embargo_days)
    train_df = df[(df.index <= cutoff) & df["tb_label_bin"].notna()].copy()

    if len(train_df) < 200:
        raise RuntimeError(f"Недостаточно данных для cutoff={cutoff.date()}: {len(train_df)}")

    X       = train_df[feature_cols]
    y       = train_df["tb_label_bin"].values.astype(int)
    regimes = _get_regimes(train_df)

    adaptive_wt = compute_adaptive_weight(train_df)
    sw          = compute_sample_weights(train_df)

    print(f"\n  — cutoff={cutoff.date()}, n={len(X)}, "
          f"UP_rate={y.mean():.3f}, adaptive_NOT_UP_w={adaptive_wt:.2f}")

    model = RegimeEnsembleV18(not_up_weight=adaptive_wt)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X, y, regimes, sample_weight=sw)

    print(f"    Классы: {model.classes_}, режимы: {list(model.models.keys())}")
    return model, adaptive_wt, sw


# ---------------------------------------------------------------------------
# 4. Расширенный поиск политики v18
# ---------------------------------------------------------------------------

def select_policy_v18(
    valid_df: pd.DataFrame,
    model: RegimeEnsembleV18,
    feature_cols: List[str],
) -> dict:
    """
    Расширенная сетка: пороги 0.42–0.65, cooldown 7–20.
    Минимальное число сигналов = 3 (было 4 в v16/v17).
    """
    best_obj: float = -float("inf")
    best_params: dict = {}

    for up_thr in [0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]:
        for cooldown in [7, 10, 15, 20]:
            tmp     = apply_policy_v16(valid_df, model, feature_cols, up_thr, cooldown)
            sigs    = tmp[(tmp["signal"] == "BUY") & tmp["tb_label_bin"].notna()]
            n       = len(sigs)
            if n < 3:
                continue
            labeled = tmp[tmp["tb_label_bin"].notna()]
            correct = int((sigs["tb_label_bin"] == 1).sum())
            base    = float((labeled["tb_label_bin"] == 1).mean())
            prec    = correct / n
            lo, _   = wilson_ci(correct, n)
            lift    = prec - base
            # Цель: (Wilson low - base) + бонус за количество сигналов
            obj     = (lo - base) + 0.005 * min(n, 20) if lift > 0 else -999
            if obj > best_obj:
                best_obj    = obj
                best_params = {"up_threshold": up_thr, "cooldown": cooldown}

    return best_params if best_params else {"up_threshold": 0.48, "cooldown": 15}


# ---------------------------------------------------------------------------
# 5. Purged CV v18 (с декай-весами)
# ---------------------------------------------------------------------------

def purged_cv_v18(
    df: pd.DataFrame, feature_cols: List[str],
    n_train_years: int = 3, n_test_months: int = 6,
) -> pd.DataFrame:
    labeled   = df[df["tb_label_bin"].notna()].copy()
    wf_splits = purged_walk_forward_splits(
        labeled.index,
        n_train_years=n_train_years, n_test_months=n_test_months,
        embargo_days=EMBARGO_V18, horizon=HORIZON_V18,
    )
    rows = []
    for i, (tr_idx, te_idx) in enumerate(wf_splits):
        train_fold = labeled.loc[tr_idx]
        Xtr  = train_fold[feature_cols]
        ytr  = train_fold["tb_label_bin"].values.astype(int)
        Xte  = labeled.loc[te_idx, feature_cols]
        yte  = labeled.loc[te_idx, "tb_label_bin"].values.astype(int)
        rgtr = _get_regimes(train_fold)
        rgte = _get_regimes(labeled.loc[te_idx])

        if len(Xtr) < MIN_REGIME_SAMPLES or Xte.empty:
            continue
        try:
            sw  = compute_sample_weights(train_fold)
            m   = RegimeEnsembleV18()
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xtr, ytr, rgtr, sample_weight=sw)
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
            print(f"  fold {i} пропущен: {type(e).__name__}: {e}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Основной pipeline
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",   default="2013-01-01")
    ap.add_argument("--end",     default="2099-12-31")
    ap.add_argument("--out-dir", default="baseline_outputs_v18")
    args = ap.parse_args(argv)

    end = min(args.end, pd.Timestamp.today().strftime("%Y-%m-%d"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== v18: expanding window + adaptive weight + time decay ===")
    print(f"  depth={MAX_DEPTH}, leaf={MAX_LEAF_NODES}, lr={LEARNING_RATE}, "
          f"l2={L2_REG}, halflife={HALFLIFE_YEARS}y, top_feat={TOP_FEATURES_N}")

    # ---- OHLC ----
    print("\n=== v18: загрузка OHLC ===")
    df = fetch_ohlc(args.start, end)
    print(f"  OHLC: {len(df)} строк, {df.index[0].date()} — {df.index[-1].date()}")

    # ---- Макро (yfinance) ----
    print("\n=== v18: загрузка макро-данных ===")
    macro = fetch_macro_yfinance(args.start, end)
    if not macro.empty:
        macro.to_csv(out / "v18_macro_raw.csv")
    else:
        print("  WARN: макро недоступны")

    # ---- COT ----
    print("\n=== v18: загрузка COT ===")
    cot = fetch_cot_silver(pd.Timestamp(args.start).year, pd.Timestamp(end).year)
    if cot.empty:
        print("  COT: недоступны")
    else:
        print(f"  COT: {len(cot)} записей")
        cot.to_csv(out / "v18_cot_raw.csv")

    # ---- Признаки ----
    print("\n=== v18: инженерия признаков ===")
    df = build_features(df, pd.DataFrame())
    df = add_vol_trend_regime(df)
    df = add_macro_features(df, macro)

    if not cot.empty:
        df = merge_cot_to_daily(df, cot, lag_days=COT_RELEASE_LAG)
        present_cot = [c for c in COT_FEATURES if c in df.columns]
        print(f"  COT признаки: {present_cot}")
    else:
        present_cot = []

    # ---- Labels ----
    print("\n=== v18: triple-barrier + бинаризация ===")
    df = add_triple_barrier_labels(df, horizon=HORIZON_V18)
    df = binarize_labels(df)
    df["split"] = df.index.map(split_name)
    df.to_csv(out / "v18_full_data.csv")

    lb = label_report_binary(df)
    lb.to_csv(out / "v18_label_distribution.csv", index=False)
    print("  UP/NOT_UP распределение:")
    print(lb.to_string(index=False))

    # ---- Список всех признаков ----
    base_features  = get_feature_cols(df)
    macro_features = [c for c in MACRO_FEATURE_NAMES if c in df.columns]
    extra_cot      = [c for c in present_cot if c not in base_features]
    all_features: List[str] = list(dict.fromkeys(base_features + macro_features + extra_cot))
    print(f"\n  Всего признаков: {len(all_features)} "
          f"(base={len(base_features)}, macro={len(macro_features)}, COT={len(extra_cot)})")

    # ---- PASS 1: отбор признаков (на valid-окне для fair selection) ----
    print("\n=== v18: pass 1 — отбор признаков (valid-модель 2013–2022) ===")
    valid_cutoff = pd.Timestamp(EXPAND_CUTOFFS["valid"]) - pd.Timedelta(days=EMBARGO_V18)
    p1_train     = df[(df.index <= valid_cutoff) & df["tb_label_bin"].notna()].copy()
    p1_valid     = df[(df["split"] == "valid") & df["tb_label_bin"].notna()].copy()

    X_p1_tr   = p1_train[all_features]
    y_p1_tr   = p1_train["tb_label_bin"].values.astype(int)
    r_p1_tr   = _get_regimes(p1_train)
    X_p1_val  = p1_valid[all_features]
    y_p1_val  = p1_valid["tb_label_bin"].values.astype(int)
    sw_p1     = compute_sample_weights(p1_train)

    print(f"  Pass-1 train: {len(X_p1_tr)} строк (UP={y_p1_tr.mean():.3f})")
    model_p1 = RegimeEnsembleV18()
    with contextlib.redirect_stdout(io.StringIO()):
        model_p1.fit(X_p1_tr, y_p1_tr, r_p1_tr, sample_weight=sw_p1)

    selected, imp_series = select_top_features(model_p1, X_p1_val, y_p1_val, n_top=TOP_FEATURES_N)

    imp_df = imp_series.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df["source"] = imp_df["feature"].apply(
        lambda f: "macro_yf" if f in MACRO_FEATURE_NAMES
                  else ("cot" if f in COT_FEATURES else "base")
    )
    imp_df.to_csv(out / "v18_feature_importance.csv", index=False)
    pd.DataFrame({"feature": selected, "rank": range(1, len(selected)+1)}).to_csv(
        out / "v18_selected_features.csv", index=False
    )
    macro_in_top = [f for f in selected if f in MACRO_FEATURE_NAMES]
    print(f"  Отобрано: {len(selected)} из {len(all_features)}, "
          f"macro в top: {len(macro_in_top)} → {macro_in_top}")

    # ---- PASS 2: адаптивные модели по расширяющемуся окну ----
    print("\n=== v18: pass 2 — адаптивные модели (expanding window) ===")
    split_models: Dict[str, RegimeEnsembleV18] = {}
    split_weights: Dict[str, float] = {}

    for split_key, cutoff in EXPAND_CUTOFFS.items():
        print(f"\n  [{split_key.upper()}] — expanding до {cutoff}")
        model, wt, _ = train_expanding_model(df, cutoff, selected)
        split_models[split_key]  = model
        split_weights[split_key] = wt

    # ---- Выбор политики на valid (valid-модель) ----
    print("\n=== v18: выбор политики (valid-модель, 2023) ===")
    valid_full    = df[df["split"] == "valid"].copy()
    policy_params = select_policy_v18(valid_full, split_models["valid"], selected)
    print(f"  Параметры: {policy_params}")

    # ---- Применение политики (каждый split — своя модель) ----
    print("\n=== v18: применение политики (split-specific models) ===")
    signal_parts = []
    for split_key in ["train", "valid", "test", "forward"]:
        split_df = df[df["split"] == split_key].copy()
        if split_df.empty:
            continue
        model_for_split = split_models.get(split_key, split_models["valid"])
        part = apply_policy_v16(
            split_df, model_for_split, selected,
            policy_params["up_threshold"],
            policy_params["cooldown"],
        )
        signal_parts.append(part)

    all_df = pd.concat(signal_parts).sort_index()
    all_df.to_csv(out / "v18_decisions_all.csv")

    # ---- Метрики (каждый split — своя модель) ----
    print("\n=== v18: метрики (бинарные, baseline=0.50) ===")
    cls_rows = []
    for split_key in ["train", "valid", "test", "forward"]:
        model_for_split = split_models.get(split_key, split_models["valid"])
        row = evaluate_split_v16(df, split_key, model_for_split, selected)
        cls_rows.append(row)
    cls_df = pd.DataFrame(cls_rows)
    cls_df.to_csv(out / "v18_classifier_metrics.csv", index=False)
    cols = [c for c in ["split", "n", "balanced_accuracy", "auc", "brier"] if c in cls_df.columns]
    print(cls_df[cols].to_string(index=False))

    # ---- Adaptive weights summary ----
    print("\n  Адаптивные веса по окнам:")
    for sp, wt in split_weights.items():
        cutoff_date = EXPAND_CUTOFFS[sp]
        cutoff_ts   = pd.Timestamp(cutoff_date) - pd.Timedelta(days=EMBARGO_V18)
        train_tmp   = df[(df.index <= cutoff_ts) & df["tb_label_bin"].notna()]
        if len(train_tmp) >= 50:
            recent_cut  = train_tmp.index.max() - pd.Timedelta(days=RECENT_WEIGHT_YEARS * 365)
            recent_tmp  = train_tmp[train_tmp.index >= recent_cut]
            recent_up   = recent_tmp["tb_label_bin"].mean() if len(recent_tmp) >= 50 else float("nan")
        else:
            recent_up = float("nan")
        print(f"    {sp:8s}: NOT_UP_w={wt:.2f}, recent_2y_UP={recent_up:.3f}, "
              f"train_n={len(train_tmp)}")

    # ---- Guardrails ----
    print("\n=== v18: guardrails ===")
    grd_rows   = [compute_guardrails_binary(all_df, s) for s in ["valid", "test", "forward"]]
    guardrails = pd.DataFrame(grd_rows)
    guardrails.to_csv(out / "v18_guardrails.csv", index=False)
    cols = ["split", "n_signals", "correct_over_n", "precision",
            "wilson_95_low", "base_up_rate", "lift_vs_base", "warning"]
    cols = [c for c in cols if c in guardrails.columns]
    print(guardrails[cols].to_string(index=False))

    # ---- Policy JSON ----
    policy_params.update({
        "version":         "v18",
        "horizon_days":    HORIZON_V18,
        "top_features_n":  TOP_FEATURES_N,
        "not_up_weight_adaptive": split_weights,
        "halflife_years":  HALFLIFE_YEARS,
        "macro_features":  macro_features,
        "macro_in_top_n":  macro_in_top,
        "regularization": {
            "max_depth": MAX_DEPTH, "max_leaf_nodes": MAX_LEAF_NODES,
            "learning_rate": LEARNING_RATE, "l2_reg": L2_REG,
            "min_samples_leaf": MIN_SAMP_LEAF,
        },
        "expand_cutoffs": EXPAND_CUTOFFS,
    })
    with open(out / "v18_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_params, f, indent=2, ensure_ascii=False)

    # ---- Бэктест ----
    print("\n=== v18: бэктест + buy-and-hold ===")
    bt_rows = []
    for s in ["valid", "test", "forward"]:
        trades = backtest_strategy(all_df, s, HORIZON_V18)
        trades.to_csv(out / f"{s}_trades_v18.csv", index=False)
        all_df[all_df["split"] == s].to_csv(out / f"{s}_decisions_v18.csv")
        bnh     = buy_and_hold_return(all_df, s)
        summary = backtest_summary(trades, s, bnh)
        bt_rows.append(summary)
    bt_df = pd.DataFrame(bt_rows)
    bt_df.to_csv(out / "v18_backtest_report.csv", index=False)
    cols = [c for c in ["split", "n_trades", "sum_net_return", "win_rate",
                         "profit_factor", "buy_and_hold", "vs_bnh"] if c in bt_df.columns]
    print(bt_df[cols].to_string(index=False))

    # ---- Последние сигнальные карточки ----
    cards = []
    for s in ["valid", "test", "forward"]:
        d = all_df[all_df["split"] == s].sort_index()
        if d.empty:
            continue
        r = d.iloc[-1]
        cards.append({
            "split":          s,
            "date":           r.name.date(),
            "silver_close":   round(float(r.get("silver_close", float("nan"))), 2),
            "signal":         r.get("signal", "HOLD"),
            "reason":         r.get("reason", ""),
            "p_up":           round(float(r.get("p_up", float("nan"))), 4),
            "p_down":         round(float(r.get("p_down", float("nan"))), 4),
            "trend_regime":   r.get("trend_regime", r.get("regime", "")),
            "adaptive_weight": split_weights.get(s, NOT_UP_WEIGHT),
        })
    pd.DataFrame(cards).to_csv(out / "v18_latest_signal_cards.csv", index=False)

    # ---- Purged CV ----
    print("\n=== v18: purged CV (с time-decay, baseline=0.50) ===")
    wf_df = purged_cv_v18(df, selected)
    wf_df.to_csv(out / "v18_purged_wf_cv.csv", index=False)
    if not wf_df.empty:
        mean_ba = wf_df["balanced_acc"].mean()
        std_ba  = wf_df["balanced_acc"].std()
        n_above = (wf_df["balanced_acc"] > 0.50).sum()
        print(f"  Фолдов: {len(wf_df)}, mean BA: {mean_ba:.3f} ± {std_ba:.3f}")
        print(f"  Выше 0.50: {n_above}/{len(wf_df)}")
        print(wf_df.to_string(index=False))
    else:
        print("  CV пуст")

    # ---- Сравнение v17 vs v18 ----
    v17_gr_path = Path("baseline_outputs_v17/v17_guardrails.csv")
    if v17_gr_path.exists():
        v17_gr    = pd.read_csv(v17_gr_path)
        comp_rows = []
        for s in ["valid", "test", "forward"]:
            r17 = v17_gr[v17_gr["split"] == s]
            r18 = guardrails[guardrails["split"] == s]
            comp_rows.append({
                "split":           s,
                "v17_precision":   pct(r17["precision"].values[0])     if not r17.empty else "-",
                "v18_precision":   pct(r18["precision"].values[0])     if not r18.empty else "-",
                "v17_wilson_low":  pct(r17["wilson_95_low"].values[0]) if not r17.empty else "-",
                "v18_wilson_low":  pct(r18["wilson_95_low"].values[0]) if not r18.empty else "-",
                "v18_n_signals":   r18["n_signals"].values[0]          if not r18.empty else "-",
                "v17_warning":     r17["warning"].values[0]             if not r17.empty else "-",
                "v18_warning":     r18["warning"].values[0]             if not r18.empty else "-",
            })
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(out / "v18_vs_v17_comparison.csv", index=False)
        print("\n=== Сравнение v17 vs v18 ===")
        print(comp_df.to_string(index=False))

    print(f"\n=== v18 завершён. Результаты: {out} ===")
    print("  Дашборд: streamlit run dashboard_app.py")


if __name__ == "__main__":
    main()
