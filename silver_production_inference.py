"""
silver_production_inference.py — Production inference для актуального сигнала на сегодня

Отличие от CPCV:
  CPCV     → тренирует 15 моделей на разных fold'ах, предсказывает только на labeled данных
             → используется для ВАЛИДАЦИИ (DSR, PSR, bootstrap CI)
  Production → тренирует ОДНУ модель на ВСЕХ labeled данных, предсказывает на самых свежих
               features → используется для РЕАЛЬНОЙ ТОРГОВЛИ

Почему так:
  - Triple-barrier label требует 15d forward данных → последние 15 дней без labels
  - CPCV предсказывает только на labeled → последние ~30 дней без CPCV-сигналов
  - Production inference: train на ВСЁМ labeled, infer на features последнего дня

Запуск:
  python silver_production_inference.py            # обучить + инференс на сегодня
  python silver_production_inference.py --predict  # только инференс (если модель уже есть)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from silver_assistant_v18_adaptive import (
    RegimeEnsembleV18, compute_sample_weights, _get_regimes,
    HISTORICAL_UP_RATE,
)
from silver_assistant_v16_binary import TOP_FEATURES_N

V22_DIR  = Path("baseline_outputs_v22")
V25_DIR  = Path("baseline_outputs_v25")
PROD_DIR = Path("baseline_outputs_prod")
PROD_DIR.mkdir(exist_ok=True)

MODEL_PATH = PROD_DIR / "production_model.pkl"
SIGNAL_PATH = PROD_DIR / "production_signal_today.json"


def load_features_list() -> List[str]:
    """Top-30 фичей из v22 feature importance."""
    p = V22_DIR / "v22_feature_importance.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    fi = pd.read_csv(p)
    return fi.sort_values("importance", ascending=False).head(TOP_FEATURES_N)["feature"].tolist()


def load_data() -> pd.DataFrame:
    p = V22_DIR / "v22_full_data.csv"
    df = pd.read_csv(p, parse_dates=[0]).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def train_production_model(
    df: pd.DataFrame, feature_cols: List[str],
    halflife_years: float = 1.5,
) -> RegimeEnsembleV18:
    """
    Обучает ОДНУ модель на ВСЕХ labeled данных (без train/test split).
    Production paradigm: максимальное использование данных для serving today's signal.
    """
    labeled = df[df["tb_label_bin"].notna() & df[feature_cols].notna().all(axis=1)].copy()
    print(f"  Train on {len(labeled)} labeled rows "
          f"({labeled.index.min().date()} → {labeled.index.max().date()})")

    X = labeled[feature_cols]
    y = labeled["tb_label_bin"].astype(int).values
    regimes = _get_regimes(labeled)
    sw = compute_sample_weights(labeled, halflife_years=halflife_years)

    recent_up = float(labeled.tail(252)["tb_label_bin"].mean())
    not_up_w = HISTORICAL_UP_RATE / max(recent_up, 0.05)
    print(f"  UP rate (last 252 days): {recent_up:.3f}, not_up_weight: {not_up_w:.3f}")

    model = RegimeEnsembleV18(not_up_weight=not_up_w)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X, y, regimes, sample_weight=sw)
    print(f"  Model trained. Regimes: {list(model.models.keys())}")
    return model


def save_model(model: RegimeEnsembleV18, path: Path = MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {path}")


def load_model(path: Path = MODEL_PATH) -> Optional[RegimeEnsembleV18]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_recent(
    df: pd.DataFrame, model: RegimeEnsembleV18,
    feature_cols: List[str], n_recent: int = 30,
) -> pd.DataFrame:
    """
    Предсказывает p_up на последних n_recent днях (включая unlabeled).
    Возвращает DataFrame с p_up, signal_long, и контекстом.
    """
    valid = df[df[feature_cols].notna().all(axis=1)]
    if valid.empty:
        return pd.DataFrame()

    last = valid.tail(n_recent).copy()
    X = last[feature_cols]
    regimes = _get_regimes(last)
    with contextlib.redirect_stdout(io.StringIO()):
        p_up = model.p_up(X, regimes)
    last["p_up_prod"] = p_up

    # ⭐ MAX_RETURN params (v28) — цель: обогнать buy-and-hold
    #
    # ИСТОРИЯ ИЗМЕНЕНИЙ:
    #   OPTIMAL_V1: entry=0.49, exit=0.43, cooldown=15 — оверфит к 2025
    #   OPTIMAL_V2: entry=0.48, exit=0.35, cooldown=25 — "консистентность"
    #               → 25d cooldown: пропускали большую часть тренда
    #               → exit=0.35: держали убыточные позиции слишком долго
    #   MAX_RETURN: entry=0.52, exit=0.45, cooldown=10 — balanced mode
    #               → forward +44.6% Sharpe 1.311 vs BnH (было +17.8% WF / +53.4% CPCV)
    #               → выход симметричен входу (gap=0.07 vs 0.13 в v2)
    #               → cooldown 2.5x короче → захватываем больше движения
    threshold       = 0.52    # p_up_entry: было 0.48 → точнее (69% WR vs 80%)
    exit_threshold  = 0.45    # p_up_exit:  было 0.35 → выходим при ослаблении сигнала
    short_threshold = 0.35    # p_up<0.35 → SHORT (зарабатываем в медвежьих рынках)
    cooldown        = 10      # было 25 → в 2.5x больше возможностей войти

    # Применяем policy: BUY / SHORT / HOLD с cooldown
    p_vals = last["p_up_prod"].values
    signals      = []
    kelly_fracs  = []
    last_long_i  = -10**9
    last_short_i = -10**9

    for i, p in enumerate(p_vals):
        sig  = "HOLD"
        frac = 0.0

        if p >= threshold and (i - last_long_i) > cooldown:
            sig  = "BUY"
            # Kelly fraction: минимум 25% при входе, растёт до 100% при p_up→1.0
            frac = round(0.25 + 0.75 * (p - threshold) / (1.0 - threshold), 4)
            frac = min(frac, 1.0)
            last_long_i = i
        elif p < short_threshold and (i - last_short_i) > cooldown:
            sig  = "SHORT"
            # Kelly fraction для SHORT: минимум 25% при p_up=short_threshold, растёт к 0
            frac = round(0.25 + 0.75 * (short_threshold - p) / short_threshold, 4)
            frac = min(frac, 1.0)
            last_short_i = i

        signals.append(sig)
        kelly_fracs.append(frac)

    last["signal_prod"] = signals
    last["kelly_frac"]  = kelly_fracs
    return last[["silver_close", "p_up_prod", "signal_prod", "kelly_frac"] +
                (["regime"] if "regime" in last.columns else [])]


def emit_today_signal(predictions: pd.DataFrame, policy: dict) -> dict:
    """
    Извлекает сегодняшний сигнал + контекст для UI.
    Возвращает dict, который пишется в production_signal_today.json
    """
    if predictions.empty:
        return {
            "ok": False,
            "error": "no recent valid features",
        }

    today = predictions.iloc[-1]
    today_date = predictions.index[-1]

    # Тренд p_up за последние 5/10/20 дней
    p_history = predictions["p_up_prod"].astype(float)
    trend_5 = float(p_history.tail(5).mean())
    trend_10 = float(p_history.tail(10).mean())
    trend_20 = float(p_history.tail(20).mean())

    threshold = float(policy.get("up_threshold", 0.52))    # было 0.55 → v28 default
    cooldown  = int(policy.get("cooldown", 10))             # было 15 → v28 default

    # Сколько дней до возможного сигнала: если cooldown активен, оценим
    last_buy_idx = None
    for i in range(len(predictions) - 1, -1, -1):
        if predictions["signal_prod"].iloc[i] == "BUY":
            last_buy_idx = i
            break
    days_since_buy = (len(predictions) - 1 - last_buy_idx) if last_buy_idx is not None else None
    cooldown_remaining = max(0, cooldown - days_since_buy) if days_since_buy is not None else 0

    p_today = float(today["p_up_prod"])

    # ⭐ MAX_RETURN exit/short thresholds (v28)
    # Было exit=0.35 → держали убыточные позиции до почти нейтрального сигнала
    # Теперь exit=0.45 → выходим при первом ослаблении тренда
    exit_threshold  = 0.45    # было 0.35
    short_threshold = 0.35    # SHORT ниже нейтральной зоны (новое)

    sell_recommended  = p_today < exit_threshold
    short_recommended = p_today < short_threshold

    # Kelly fraction для текущего p_up
    if p_today >= threshold:
        kelly_frac = round(0.25 + 0.75 * (p_today - threshold) / (1.0 - threshold), 4)
        kelly_frac = min(kelly_frac, 1.0)
    elif p_today < short_threshold:
        kelly_frac = round(0.25 + 0.75 * (short_threshold - p_today) / short_threshold, 4)
        kelly_frac = min(kelly_frac, 1.0)
    else:
        kelly_frac = 0.0

    # Определяем тип сигнала (BUY / SELL / SHORT / COVER / HOLD)
    if p_today >= threshold and cooldown_remaining == 0:
        primary_signal = "BUY"
    elif sell_recommended and not short_recommended:
        primary_signal = "SELL"
    elif short_recommended and cooldown_remaining == 0:
        primary_signal = "SHORT"
    else:
        primary_signal = "HOLD"

    # ⚠️ Drift alert: модель работает вне дистрибуции обучения
    # При drift_rate > 0.7 — результаты ненадёжны, нужен ретрейн
    drift_path = Path("daily_reports/training") / datetime.now().strftime("%Y-%m-%d") / "summary.json"
    drift_rate = None
    drift_alert = False
    if drift_path.exists():
        try:
            drift_data = json.loads(drift_path.read_text(encoding="utf-8"))
            drift_rate = drift_data.get("drift", {}).get("drift_rate")
            drift_alert = (drift_rate is not None and drift_rate > 0.70)
        except Exception:
            pass

    return {
        "ok":                True,
        "date":              today_date.strftime("%Y-%m-%d"),
        "ts_utc":            datetime.now(timezone.utc).isoformat(),
        "signal":            primary_signal,
        "p_up":              p_today,
        "p_up_trend_5d":     round(trend_5, 4),
        "p_up_trend_10d":    round(trend_10, 4),
        "p_up_trend_20d":    round(trend_20, 4),
        "above_threshold":   p_today >= threshold,
        "below_exit":        sell_recommended,
        "short_zone":        short_recommended,
        "threshold":         threshold,
        "exit_threshold":    exit_threshold,
        "short_threshold":   short_threshold,
        "kelly_fraction":    kelly_frac,
        "cooldown_days":     cooldown,
        "cooldown_remaining": cooldown_remaining,
        "regime":            str(today.get("regime", "unknown")),
        "silver_close":      float(today["silver_close"]),
        "n_predictions":     len(predictions),
        "predictions_window": [
            predictions.index[0].strftime("%Y-%m-%d"),
            predictions.index[-1].strftime("%Y-%m-%d"),
        ],
        "drift_alert":       drift_alert,
        "drift_rate":        drift_rate,
        "exit_recommendation": {
            "action": "SELL" if sell_recommended else "HOLD_POSITION",
            "reason": (f"p_up={p_today:.3f} < exit_threshold={exit_threshold}" if sell_recommended
                       else f"p_up={p_today:.3f} >= exit_threshold={exit_threshold}"),
        },
        "short_recommendation": {
            "action": "SHORT" if short_recommended else "NO_SHORT",
            "reason": (f"p_up={p_today:.3f} < short_threshold={short_threshold}" if short_recommended
                       else f"p_up={p_today:.3f} >= short_threshold={short_threshold}"),
            "kelly_fraction": kelly_frac if short_recommended else 0.0,
        },
    }


def save_predictions_history(predictions: pd.DataFrame) -> None:
    """Сохраняет полную таблицу production predictions для UI/charts."""
    if predictions.empty:
        return
    out = predictions.copy()
    out.index.name = "Date"
    out.to_csv(PROD_DIR / "production_predictions.csv")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true",
                    help="Только инференс, без переобучения (если модель уже на диске)")
    ap.add_argument("--n-recent", type=int, default=30,
                    help="Сколько последних дней предсказывать")
    args = ap.parse_args()

    print("=" * 70)
    print(" Production inference — сигнал на сегодня")
    print("=" * 70)

    feature_cols = load_features_list()
    df = load_data()
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Features: {len(feature_cols)}")

    if args.predict and MODEL_PATH.exists():
        print("\n→ Загружаю существующую модель...")
        model = load_model()
    else:
        print("\n→ Обучаю production-модель...")
        model = train_production_model(df, feature_cols)
        save_model(model)

    print(f"\n→ Predict на последних {args.n_recent} днях...")
    preds = predict_recent(df, model, feature_cols, n_recent=args.n_recent)
    save_predictions_history(preds)

    policy_path = V25_DIR / "v25_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}

    today_sig = emit_today_signal(preds, policy)
    SIGNAL_PATH.write_text(
        json.dumps(today_sig, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Сегодняшний сигнал: {today_sig.get('signal', '?')}")
    print(f"  p_up:              {today_sig.get('p_up', 'n/a'):.4f}")
    print(f"  p_up 5d trend:     {today_sig.get('p_up_trend_5d', 'n/a')}")
    print(f"  p_up 10d trend:    {today_sig.get('p_up_trend_10d', 'n/a')}")
    print(f"  Above threshold:   {today_sig.get('above_threshold')}")
    print(f"  Cooldown remaining: {today_sig.get('cooldown_remaining')}d")
    print(f"  Date:              {today_sig.get('date')}")
    print(f"\n  ✅ Saved: {SIGNAL_PATH}")
    print(f"  ✅ Saved: {PROD_DIR / 'production_predictions.csv'}")


if __name__ == "__main__":
    main()
