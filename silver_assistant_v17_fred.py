"""
Silver Trading Assistant v17 — Макро-данные через yfinance (FRED-прокси)

Изменения vs v16:

Проблема: FRED недоступен напрямую (TimeoutError) и pandas-datareader несовместим с Python 3.14.
Решение: yfinance ETF/фьючерсы как прокси FRED-переменных.

Новые признаки (15 штук, итого ~80 до отбора):
  ^TNX  → us10y_yf, us10y_chg_5d, us10y_chg_20d   (10-летняя доходность)
  ^IRX  → us3m_yf, yield_curve_10y3m               (3-месячная / кривая)
  TIP   → tip_ret_5d, tip_ret_20d, tip_zscore_20d, tip_above_ma50   (реальные ставки)
  RINF  → rinf_ret_5d, rinf_ret_20d, rinf_zscore_20d               (инфляционные ожидания)
  HYG   → hyg_ret_5d, hyg_ret_20d, hyg_above_ma50                  (кредитный риск)

Почему именно эти:
  - Реальные ставки (TIP) — сильнейший фундаментальный драйвер золота/серебра.
  - Инфляционные ожидания (RINF) — вторичный драйвер: рост breakeven → рост silver.
  - Кривая доходности (10y-3m) — индикатор рецессии / risk-off.
  - Кредитный риск (HYG) — risk-on/off сигнал, коррелирует с промышленным спросом.

Запуск:
  python silver_assistant_v17_fred.py
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
    import yfinance as yf
except ImportError:
    raise ImportError("pip install yfinance>=0.2.40")

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

# Переиспользуем весь binary-стек из v16
try:
    from silver_assistant_v16_binary import (
        RegimeEnsembleBinary,
        binarize_labels,
        label_report_binary,
        select_top_features,
        evaluate_split_v16,
        compute_guardrails_binary,
        apply_policy_v16,
        select_policy_v16,
        purged_cv_binary,
        _get_regimes,
        split_name,
        MAX_DEPTH, MAX_LEAF_NODES, LEARNING_RATE,
        MIN_SAMP_LEAF, L2_REG, NOT_UP_WEIGHT, TOP_FEATURES_N,
        HORIZON_V16, EMBARGO_V16,
    )
    print("  v16 функции загружены.")
except ImportError as e:
    raise ImportError(
        "Запустите сначала: python silver_assistant_v16_binary.py\n"
        f"Детали: {e}"
    )

# COT + режимная инженерия из v15
try:
    from silver_assistant_v15_regime_cot import (
        fetch_cot_silver, merge_cot_to_daily,
        add_vol_trend_regime,
        COT_FEATURES, COT_RELEASE_LAG,
        MIN_REGIME_SAMPLES,
    )
    print("  v15 функции загружены.")
except ImportError as e:
    raise ImportError(f"v15 не найден: {e}")

# OHLC + признаки + утилиты из v14
try:
    from silver_assistant_v14_main import (
        fetch_ohlc, build_features,
        add_triple_barrier_labels, get_feature_cols,
        backtest_strategy, buy_and_hold_return, backtest_summary,
        purged_walk_forward_splits,
        wilson_ci, md_table, pct,
        SPLITS,
    )
    print("  v14 функции загружены.")
except ImportError as e:
    raise ImportError(f"v14 не найден: {e}")


# ---------------------------------------------------------------------------
# Версия v17
# ---------------------------------------------------------------------------

HORIZON_V17 = HORIZON_V16   # 15 дней
EMBARGO_V17 = EMBARGO_V16   # 15 дней


# ---------------------------------------------------------------------------
# 1. Макро-тикеры (yfinance-прокси для FRED)
# ---------------------------------------------------------------------------

MACRO_TICKERS: Dict[str, str] = {
    "^TNX":  "tnx",   # 10-летняя доходность UST (= FRED DGS10)
    "^IRX":  "irx",   # 13-недельная T-Bill доходность (= FRED DGS3MO)
    "TIP":   "tip",   # iShares TIPS ETF (прокси реальных ставок DFII10)
    "RINF":  "rinf",  # ProShares Inflation Expectations ETF (прокси T10YIE)
    "HYG":   "hyg",   # iShares High Yield ETF (кредитный риск / risk-on)
}

# Итоговые имена macro-признаков (добавляются в feature list)
MACRO_FEATURE_NAMES: List[str] = [
    "us10y_yf",
    "us10y_chg_5d",
    "us10y_chg_20d",
    "us3m_yf",
    "yield_curve_10y3m",
    "tip_ret_5d",
    "tip_ret_20d",
    "tip_zscore_20d",
    "tip_above_ma50",
    "rinf_ret_5d",
    "rinf_ret_20d",
    "rinf_zscore_20d",
    "hyg_ret_5d",
    "hyg_ret_20d",
    "hyg_above_ma50",
]


# ---------------------------------------------------------------------------
# 2. Загрузка макро-данных
# ---------------------------------------------------------------------------

def _yf_close_series(ticker: str, start: str, end: str) -> pd.Series:
    """Загружает серию закрытий по одному тикеру, обрабатывает MultiIndex."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    close.name = ticker
    return close


def fetch_macro_yfinance(start: str, end: str) -> pd.DataFrame:
    """
    Скачивает макро-тикеры через yfinance.
    Возвращает DataFrame с колонками по именам из MACRO_TICKERS.values().
    ETF-цены (TIP, RINF, HYG) и yield-индексы (^TNX, ^IRX) — рыночные данные,
    доступны в тот же торговый день → сдвиг не нужен.
    """
    frames: Dict[str, pd.Series] = {}
    for ticker, col in MACRO_TICKERS.items():
        try:
            s = _yf_close_series(ticker, start, end)
            if not s.empty:
                frames[col] = s
                print(f"    OK  {ticker:6s} ({col}): {len(s)} строк")
            else:
                print(f"    WARN {ticker}: пустые данные")
        except Exception as e:
            print(f"    FAIL {ticker}: {type(e).__name__}: {e}")

    if not frames:
        return pd.DataFrame()

    macro = pd.DataFrame(frames)
    macro.index.name = "Date"
    macro = macro.sort_index().ffill()
    return macro


# ---------------------------------------------------------------------------
# 3. Инженерия макро-признаков
# ---------------------------------------------------------------------------

def add_macro_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет 15 макро-признаков в df, выравнивая по индексу серебра.
    Все признаки формируются ТОЛЬКО из доступных строк macro.
    """
    if macro.empty:
        print("  WARN: macro DataFrame пуст — признаки пропущены.")
        return df

    out   = df.copy()
    # Выровняем по индексу серебра через forward-fill (заполним выходные)
    mac   = macro.reindex(out.index, method="ffill")

    # --- ^TNX: 10-летняя UST доходность ---
    if "tnx" in mac.columns:
        tnx = mac["tnx"]
        out["us10y_yf"]      = tnx
        out["us10y_chg_5d"]  = tnx.diff(5)
        out["us10y_chg_20d"] = tnx.diff(20)

    # --- ^IRX: 13w T-Bill доходность ---
    if "irx" in mac.columns:
        irx = mac["irx"]
        out["us3m_yf"] = irx
        # Кривая доходности: 10y − 3m (отрицательная = инверсия = риск рецессии)
        if "tnx" in mac.columns:
            out["yield_curve_10y3m"] = mac["tnx"] - irx

    # --- TIP: TIPS ETF (прокси реальной ставки) ---
    if "tip" in mac.columns:
        tip = mac["tip"]
        ma50_tip = tip.rolling(50).mean()
        out["tip_ret_5d"]     = tip.pct_change(5)
        out["tip_ret_20d"]    = tip.pct_change(20)
        out["tip_zscore_20d"] = (tip - tip.rolling(20).mean()) / tip.rolling(20).std()
        out["tip_above_ma50"] = (tip > ma50_tip).astype(float)

    # --- RINF: Breakeven inflation ETF ---
    if "rinf" in mac.columns:
        rinf = mac["rinf"]
        out["rinf_ret_5d"]     = rinf.pct_change(5)
        out["rinf_ret_20d"]    = rinf.pct_change(20)
        out["rinf_zscore_20d"] = (rinf - rinf.rolling(20).mean()) / rinf.rolling(20).std()

    # --- HYG: High-yield bonds (кредитный спред / risk-on) ---
    if "hyg" in mac.columns:
        hyg = mac["hyg"]
        ma50_hyg = hyg.rolling(50).mean()
        out["hyg_ret_5d"]     = hyg.pct_change(5)
        out["hyg_ret_20d"]    = hyg.pct_change(20)
        out["hyg_above_ma50"] = (hyg > ma50_hyg).astype(float)

    present = [c for c in MACRO_FEATURE_NAMES if c in out.columns]
    missing = [c for c in MACRO_FEATURE_NAMES if c not in out.columns]
    print(f"  Макро-признаки добавлены: {len(present)} / {len(MACRO_FEATURE_NAMES)}")
    if missing:
        print(f"  Пропущены: {missing}")

    return out


# ---------------------------------------------------------------------------
# 4. Основной pipeline
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",   default="2013-01-01")
    ap.add_argument("--end",     default="2099-12-31")
    ap.add_argument("--out-dir", default="baseline_outputs_v17")
    args = ap.parse_args(argv)

    end = min(args.end, pd.Timestamp.today().strftime("%Y-%m-%d"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== v17: macro yfinance + binary + регуляризация + отбор признаков ===")
    print(f"  depth={MAX_DEPTH}, leaf={MAX_LEAF_NODES}, lr={LEARNING_RATE}, "
          f"l2={L2_REG}, min_leaf={MIN_SAMP_LEAF}, NOT_UP_w={NOT_UP_WEIGHT}, "
          f"top_feat={TOP_FEATURES_N}, horizon={HORIZON_V17}")

    # ---- OHLC ----
    print("\n=== v17: загрузка OHLC ===")
    df = fetch_ohlc(args.start, end)
    print(f"  OHLC: {len(df)} строк, {df.index[0].date()} — {df.index[-1].date()}")

    # ---- Макро-данные (yfinance) ----
    print("\n=== v17: загрузка макро-данных (yfinance) ===")
    macro = fetch_macro_yfinance(args.start, end)
    if macro.empty:
        print("  WARN: макро-данные недоступны — продолжаем без них")
    else:
        print(f"  Macro: {len(macro)} строк, {macro.index[0].date()} — {macro.index[-1].date()}")
        macro.to_csv(out / "v17_macro_raw.csv")

    # ---- COT ----
    print("\n=== v17: загрузка COT (CFTC) ===")
    start_year = pd.Timestamp(args.start).year
    end_year   = pd.Timestamp(end).year
    cot = fetch_cot_silver(start_year, end_year)
    if cot.empty:
        print("  COT: недоступны")
    else:
        print(f"  COT: {len(cot)} записей, {cot.index[0].date()} — {cot.index[-1].date()}")
        cot.to_csv(out / "v17_cot_raw.csv")

    # ---- Признаки ----
    print("\n=== v17: инженерия признаков ===")
    # build_features вызываем без FRED (пустой DataFrame — FRED недоступен)
    df = build_features(df, pd.DataFrame())
    df = add_vol_trend_regime(df)
    regime_counts = df["trend_regime"].value_counts().to_dict()
    print(f"  trend_regime: {regime_counts}")

    # Добавляем макро-признаки (yfinance-прокси)
    df = add_macro_features(df, macro)

    if not cot.empty:
        df = merge_cot_to_daily(df, cot, lag_days=COT_RELEASE_LAG)
        present_cot = [c for c in COT_FEATURES if c in df.columns]
        print(f"  COT признаки: {present_cot}")
    else:
        present_cot = []

    # ---- Triple-barrier + бинаризация ----
    print("\n=== v17: triple-barrier (horizon=15) + бинаризация ===")
    df = add_triple_barrier_labels(df, horizon=HORIZON_V17)
    df = binarize_labels(df)
    df["split"] = df.index.map(split_name)
    df.to_csv(out / "v17_full_data.csv")

    lb = label_report_binary(df)
    lb.to_csv(out / "v17_label_distribution.csv", index=False)
    print("  Бинарные метки (UP=1, NOT_UP=0):")
    print(lb.to_string(index=False))

    # ---- Список признаков (v16 base + macro + COT) ----
    base_features  = get_feature_cols(df)
    macro_features = [c for c in MACRO_FEATURE_NAMES if c in df.columns]
    extra_cot      = [c for c in present_cot if c not in base_features]

    # Объединяем без дублей
    all_features: List[str] = []
    seen = set()
    for feat in base_features + macro_features + extra_cot:
        if feat not in seen:
            all_features.append(feat)
            seen.add(feat)

    print(f"\n  Признаков до отбора: {len(all_features)}")
    print(f"    base: {len(base_features)}, macro_yf: {len(macro_features)}, COT: {len(extra_cot)}")

    train_df     = df[(df["split"] == "train") & df["tb_label_bin"].notna()].copy()
    valid_df_raw = df[(df["split"] == "valid") & df["tb_label_bin"].notna()].copy()

    if len(train_df) < 200:
        raise RuntimeError(f"Недостаточно обучающих данных: {len(train_df)}")

    X_tr_all  = train_df[all_features]
    y_tr_bin  = train_df["tb_label_bin"].values.astype(int)
    r_tr      = _get_regimes(train_df)
    X_val_all = valid_df_raw[all_features]
    y_val_bin = valid_df_raw["tb_label_bin"].values.astype(int)

    # ---- Pass 1: обучение для отбора признаков ----
    print("\n=== v17: pass 1 — отбор признаков ===")
    print(f"  Обучение pass-1 на {len(X_tr_all)} строках ({len(all_features)} признаков)...")
    model_p1 = RegimeEnsembleBinary()
    with contextlib.redirect_stdout(io.StringIO()):
        model_p1.fit(X_tr_all, y_tr_bin, r_tr)

    selected, imp_series = select_top_features(model_p1, X_val_all, y_val_bin, n_top=TOP_FEATURES_N)

    imp_df = imp_series.reset_index()
    imp_df.columns = ["feature", "importance"]
    # Помечаем источник признака
    imp_df["source"] = imp_df["feature"].apply(
        lambda f: "macro_yf" if f in MACRO_FEATURE_NAMES
                  else ("cot" if f in COT_FEATURES else "base")
    )
    imp_df.to_csv(out / "v17_feature_importance.csv", index=False)
    pd.DataFrame({"feature": selected, "rank": range(1, len(selected)+1)}).to_csv(
        out / "v17_selected_features.csv", index=False
    )

    # Сколько macro признаков попало в top-N
    macro_in_top = [f for f in selected if f in MACRO_FEATURE_NAMES]
    print(f"  Отобрано: {len(selected)} признаков из {len(all_features)}")
    print(f"  Macro-признаков в top-{TOP_FEATURES_N}: {len(macro_in_top)} → {macro_in_top}")

    # ---- Pass 2: финальная модель с top признаками ----
    print(f"\n=== v17: pass 2 — финальная модель (top-{TOP_FEATURES_N}) ===")
    X_tr_sel = train_df[selected]
    print(f"  Обучение: {len(X_tr_sel)} строк ({X_tr_sel.index[0].date()} — {X_tr_sel.index[-1].date()})")
    model = RegimeEnsembleBinary()
    model.fit(X_tr_sel, y_tr_bin, r_tr)
    print(f"  Классы: {model.classes_}")
    print(f"  Режимные модели: {list(model.models.keys())}")

    # ---- Метрики ----
    print("\n=== v17: метрики (бинарные, baseline=0.50) ===")
    cls_metrics = [evaluate_split_v16(df, s, model, selected)
                   for s in ["train", "valid", "test", "forward"]]
    cls_df = pd.DataFrame(cls_metrics)
    cls_df.to_csv(out / "v17_classifier_metrics.csv", index=False)
    cols = [c for c in ["split", "n", "balanced_accuracy", "auc", "brier"] if c in cls_df.columns]
    print(cls_df[cols].to_string(index=False))

    # ---- Политика ----
    print("\n=== v17: выбор политики (valid 2023) ===")
    valid_full    = df[df["split"] == "valid"].copy()
    policy_params = select_policy_v16(valid_full, model, selected)
    print(f"  Параметры: {policy_params}")
    policy_params.update({
        "version":         "v17",
        "horizon_days":    HORIZON_V17,
        "top_features_n":  TOP_FEATURES_N,
        "not_up_weight":   NOT_UP_WEIGHT,
        "macro_features":  macro_features,
        "macro_in_top_n":  macro_in_top,
        "regularization": {
            "max_depth": MAX_DEPTH, "max_leaf_nodes": MAX_LEAF_NODES,
            "learning_rate": LEARNING_RATE, "l2_reg": L2_REG,
            "min_samples_leaf": MIN_SAMP_LEAF,
        },
        "regime_models": list(model.models.keys()),
    })
    with open(out / "v17_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_params, f, indent=2, ensure_ascii=False)

    # ---- Применение политики ----
    all_df = apply_policy_v16(
        df, model, selected,
        policy_params["up_threshold"],
        policy_params["cooldown"],
    )
    all_df.to_csv(out / "v17_decisions_all.csv")

    # ---- Guardrails ----
    print("\n=== v17: guardrails ===")
    grd_rows   = [compute_guardrails_binary(all_df, s) for s in ["valid", "test", "forward"]]
    guardrails = pd.DataFrame(grd_rows)
    guardrails.to_csv(out / "v17_guardrails.csv", index=False)
    cols = ["split", "n_signals", "correct_over_n", "precision",
            "wilson_95_low", "base_up_rate", "lift_vs_base", "warning"]
    cols = [c for c in cols if c in guardrails.columns]
    print(guardrails[cols].to_string(index=False))

    # ---- Бэктест ----
    print("\n=== v17: бэктест + buy-and-hold ===")
    bt_rows = []
    for s in ["valid", "test", "forward"]:
        trades = backtest_strategy(all_df, s, HORIZON_V17)
        trades.to_csv(out / f"{s}_trades_v17.csv", index=False)
        all_df[all_df["split"] == s].to_csv(out / f"{s}_decisions_v17.csv")
        bnh     = buy_and_hold_return(all_df, s)
        summary = backtest_summary(trades, s, bnh)
        bt_rows.append(summary)
    bt_df = pd.DataFrame(bt_rows)
    bt_df.to_csv(out / "v17_backtest_report.csv", index=False)
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
    pd.DataFrame(cards).to_csv(out / "v17_latest_signal_cards.csv", index=False)

    # ---- Purged CV ----
    print("\n=== v17: purged CV (бинарный, baseline=0.50) ===")
    wf_df = purged_cv_binary(df, selected)
    wf_df.to_csv(out / "v17_purged_wf_cv.csv", index=False)
    if not wf_df.empty:
        mean_ba = wf_df["balanced_acc"].mean()
        std_ba  = wf_df["balanced_acc"].std()
        n_above = (wf_df["balanced_acc"] > 0.50).sum()
        print(f"  Фолдов: {len(wf_df)}, mean balanced_acc: {mean_ba:.3f} +/- {std_ba:.3f}")
        print(f"  Фолдов выше 0.50 (baseline): {n_above}/{len(wf_df)}")
        print(wf_df.to_string(index=False))
    else:
        print("  CV пуст — все фолды пропущены")

    # ---- Сравнение v16 vs v17 ----
    v16_gr_path = Path("baseline_outputs_v16/v16_guardrails.csv")
    if v16_gr_path.exists():
        v16_gr    = pd.read_csv(v16_gr_path)
        comp_rows = []
        for s in ["valid", "test", "forward"]:
            r16 = v16_gr[v16_gr["split"] == s]
            r17 = guardrails[guardrails["split"] == s]
            comp_rows.append({
                "split":           s,
                "v16_precision":   pct(r16["precision"].values[0])     if not r16.empty else "-",
                "v17_precision":   pct(r17["precision"].values[0])     if not r17.empty else "-",
                "v16_wilson_low":  pct(r16["wilson_95_low"].values[0]) if not r16.empty else "-",
                "v17_wilson_low":  pct(r17["wilson_95_low"].values[0]) if not r17.empty else "-",
                "v16_warning":     r16["warning"].values[0]             if not r16.empty else "-",
                "v17_warning":     r17["warning"].values[0]             if not r17.empty else "-",
            })
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(out / "v17_vs_v16_comparison.csv", index=False)
        print("\n=== Сравнение v16 vs v17 ===")
        print(comp_df.to_string(index=False))

    # ---- Сводка macro-вклада ----
    print("\n=== v17: вклад macro-признаков ===")
    top_imp = imp_df[imp_df["feature"].isin(selected)].sort_values("importance", ascending=False)
    macro_top = top_imp[top_imp["source"] == "macro_yf"]
    if not macro_top.empty:
        print("  Macro-признаки в top-30 (по importance):")
        print(macro_top[["feature", "importance", "source"]].to_string(index=False))
    else:
        print("  Ни один macro-признак не попал в top-30.")

    print(f"\n=== v17 завершён. Результаты: {out} ===")
    print("  Дашборд: streamlit run dashboard_app.py")


if __name__ == "__main__":
    main()
