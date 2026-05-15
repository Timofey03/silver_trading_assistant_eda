"""
gold_production_inference.py — Production inference для золота (аналог silver)

Обучает gold-модель на ВСЕХ labeled данных и делает predict на свежих features.
Использует те же optimal params (entry 0.49, exit 0.43, cooldown 15) но с
gold-specific фичами и labels (1% barriers вместо 2%).

Запуск:
  python gold_production_inference.py            # train + predict
  python gold_production_inference.py --predict  # только predict
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

V26_DIR  = Path("baseline_outputs_v26")
PROD_DIR = Path("baseline_outputs_prod")
PROD_DIR.mkdir(exist_ok=True)

MODEL_PATH  = PROD_DIR / "production_model_gold.pkl"
SIGNAL_PATH = PROD_DIR / "gold_signal_today.json"


def load_data() -> pd.DataFrame:
    p = V26_DIR / "gold_full_data.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} — run v26_multiasset.py --fetch first")
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("gold_ret_")
            or c.startswith("gold_zscore_")
            or c.startswith("gold_dist_")
            or c.startswith("gold_ma")
            or c.startswith("gold_atr")
            or c.startswith("gold_rsi")
            or c.startswith("gold_macd")
            or c.startswith("gold_realized_vol")]


def train_production_model(
    df: pd.DataFrame, feature_cols: List[str],
    halflife_years: float = 1.5,
) -> RegimeEnsembleV18:
    """Обучает gold модель на ВСЕХ labeled данных."""
    labeled = df[df["gold_tb_label_bin"].notna() & df[feature_cols].notna().all(axis=1)].copy()
    print(f"  Train on {len(labeled)} labeled gold rows "
          f"({labeled.index.min().date()} → {labeled.index.max().date()})")

    # Простой regime для gold (по MA60)
    if "regime" not in labeled.columns:
        ma60 = labeled["gold_close"].rolling(60).mean()
        labeled["regime"] = np.where(
            labeled["gold_close"] > ma60 * 1.02, "uptrend_medium",
            np.where(labeled["gold_close"] < ma60 * 0.98, "downtrend_medium",
                     "sideways_medium"),
        )

    X = labeled[feature_cols]
    y = labeled["gold_tb_label_bin"].astype(int).values
    regimes = _get_regimes(labeled)
    sw = compute_sample_weights(labeled, halflife_years=halflife_years)

    recent_up = float(labeled.tail(252)["gold_tb_label_bin"].mean())
    not_up_w = HISTORICAL_UP_RATE / max(recent_up, 0.05)
    print(f"  UP rate (last 252): {recent_up:.3f}, not_up_weight: {not_up_w:.3f}")

    model = RegimeEnsembleV18(not_up_weight=not_up_w)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X, y, regimes, sample_weight=sw)
    print(f"  Gold model trained. Regimes: {list(model.models.keys())}")
    return model


def predict_recent(
    df: pd.DataFrame, model: RegimeEnsembleV18,
    feature_cols: List[str], n_recent: int = 30,
) -> pd.DataFrame:
    valid = df[df[feature_cols].notna().all(axis=1)]
    if valid.empty:
        return pd.DataFrame()

    last = valid.tail(n_recent).copy()
    # regime
    if "regime" not in last.columns:
        ma60 = df["gold_close"].rolling(60).mean().reindex(last.index)
        last["regime"] = np.where(
            last["gold_close"] > ma60 * 1.02, "uptrend_medium",
            np.where(last["gold_close"] < ma60 * 0.98, "downtrend_medium",
                     "sideways_medium"),
        )

    X = last[feature_cols]
    regimes = _get_regimes(last)
    with contextlib.redirect_stdout(io.StringIO()):
        p_up = model.p_up(X, regimes)
    last["p_up_prod"] = p_up

    # Optimal mode params (gold uses same thresholds as silver)
    threshold      = 0.49
    exit_threshold = 0.43
    cooldown       = 15

    raw = last["p_up_prod"] >= threshold
    signals = []
    last_buy_i = -10**9
    for i, ok in enumerate(raw.values):
        if ok and (i - last_buy_i) > cooldown:
            signals.append("BUY")
            last_buy_i = i
        else:
            signals.append("HOLD")
    last["signal_prod"] = signals
    return last[["gold_close", "p_up_prod", "signal_prod", "regime"]]


def emit_today_signal(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {"ok": False, "error": "no recent valid features"}

    today = predictions.iloc[-1]
    today_date = predictions.index[-1]

    p_history = predictions["p_up_prod"].astype(float)
    trend_5 = float(p_history.tail(5).mean())
    trend_10 = float(p_history.tail(10).mean())
    trend_20 = float(p_history.tail(20).mean())

    threshold = 0.49
    exit_threshold = 0.43
    cooldown = 15

    last_buy_idx = None
    for i in range(len(predictions) - 1, -1, -1):
        if predictions["signal_prod"].iloc[i] == "BUY":
            last_buy_idx = i
            break
    days_since_buy = (len(predictions) - 1 - last_buy_idx) if last_buy_idx is not None else None
    cooldown_remaining = max(0, cooldown - days_since_buy) if days_since_buy is not None else 0

    p_today = float(today["p_up_prod"])
    sell_recommended = p_today < exit_threshold

    if p_today >= threshold and cooldown_remaining == 0:
        primary_signal = "BUY"
    elif sell_recommended:
        primary_signal = "SELL"
    else:
        primary_signal = "HOLD"

    return {
        "ok":                 True,
        "asset":              "gold",
        "date":               today_date.strftime("%Y-%m-%d"),
        "ts_utc":             datetime.now(timezone.utc).isoformat(),
        "signal":             primary_signal,
        "p_up":               p_today,
        "p_up_trend_5d":      round(trend_5, 4),
        "p_up_trend_10d":     round(trend_10, 4),
        "p_up_trend_20d":     round(trend_20, 4),
        "above_threshold":    p_today >= threshold,
        "below_exit":         sell_recommended,
        "threshold":          threshold,
        "exit_threshold":     exit_threshold,
        "cooldown_days":      cooldown,
        "cooldown_remaining": cooldown_remaining,
        "regime":             str(today.get("regime", "unknown")),
        "gold_close":         float(today["gold_close"]),
        "n_predictions":      len(predictions),
        "predictions_window": [
            predictions.index[0].strftime("%Y-%m-%d"),
            predictions.index[-1].strftime("%Y-%m-%d"),
        ],
        "exit_recommendation": {
            "action": "SELL" if sell_recommended else "HOLD_POSITION",
            "reason": (f"p_up={p_today:.3f} < exit_threshold={exit_threshold}" if sell_recommended
                       else f"p_up={p_today:.3f} >= exit_threshold={exit_threshold}"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true",
                    help="Только predict, без переобучения")
    ap.add_argument("--n-recent", type=int, default=30)
    args = ap.parse_args()

    print("=" * 70)
    print(" Gold production inference — сигнал на сегодня")
    print("=" * 70)

    df = load_data()
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}")

    if args.predict and MODEL_PATH.exists():
        print("\n→ Загружаю gold модель...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        print("\n→ Обучаю gold модель...")
        model = train_production_model(df, feature_cols)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved: {MODEL_PATH}")

    print(f"\n→ Predict на последних {args.n_recent} днях...")
    preds = predict_recent(df, model, feature_cols, n_recent=args.n_recent)
    if not preds.empty:
        preds.index.name = "Date"
        preds.to_csv(PROD_DIR / "gold_predictions.csv")

    today_sig = emit_today_signal(preds)
    SIGNAL_PATH.write_text(
        json.dumps(today_sig, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Сегодняшний gold сигнал: {today_sig.get('signal', '?')}")
    p_up = today_sig.get('p_up', 'n/a')
    print(f"  p_up:               {p_up:.4f}" if isinstance(p_up, float) else f"  p_up: {p_up}")
    print(f"  Cooldown remaining: {today_sig.get('cooldown_remaining')}d")
    print(f"  Date:               {today_sig.get('date')}")
    print(f"\n  ✅ Saved: {SIGNAL_PATH}")


if __name__ == "__main__":
    main()
