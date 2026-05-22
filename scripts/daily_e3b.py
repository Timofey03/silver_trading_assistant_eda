"""scripts/daily_e3b.py — ежедневный run новой модели E3b.

Запускается 3 раза в день через .github/workflows/daily_e3b.yml.
Аналог scripts/daily_run.py, но использует E3b (multi-asset + adaptive barriers).

Последовательность:
1. Refresh данных:
   - 5 металлов через yfinance (silver/gold/platinum/palladium/copper)
   - 9 макрорядов через FRED + USDRUB
2. Re-train E3b walk-forward — обновляет baseline_outputs_multiasset/e3b_adaptive/
3. Production inference на сегодня:
   - Использует features с ffill_limit=5 (заполняет gaps в palladium для свежих дней)
   - Обучает финальную модель на ВСЕХ данных до вчерашнего дня
   - Генерирует p_up на сегодня
4. Формирует сигнал (BUY/HOLD/SELL) с теми же параметрами что E3b:
   entry≥0.48, exit≤0.35, trail=12%, max_hold=30d, cooldown=25d
5. Создаёт два отчёта:
   - daily_reports/e3b/training/YYYY-MM-DD/ — walk-forward метрики
   - daily_reports/e3b/trading/YYYY-MM-DD/  — сегодняшний сигнал
6. (опционально) Telegram уведомление

Запуск:
  python scripts/daily_e3b.py                  # полный цикл
  python scripts/daily_e3b.py --skip-training  # только инференс на сегодня
  python scripts/daily_e3b.py --quick          # без обновления данных
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# UTF-8 для stdout (особенно важно на Windows-консоли)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from app.multi_asset.metal_loader import refresh_metals_cache, load_metals
from app.multi_asset.macro_loader import load_macro
from app.multi_asset.features import build_feature_frame
from app.multi_asset.labels import build_multi_horizon_labels

# E3B output dirs
E3B_DIR = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive"
REPORTS_DIR = REPO_ROOT / "daily_reports" / "e3b"
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
TRAINING_DIR = REPORTS_DIR / "training" / TODAY
TRADING_DIR = REPORTS_DIR / "trading" / TODAY
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
TRADING_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Step 1: Refresh data
# =============================================================================
def refresh_data(force: bool = True) -> dict:
    """Скачать свежие metal + macro данные."""
    print("\n" + "=" * 70)
    print("[1/4] REFRESH DATA")
    print("=" * 70)
    diagnostics = {}
    try:
        diag = refresh_metals_cache(verbose=True)
        diagnostics["metals"] = diag
        load_macro(force_refresh=force)
        diagnostics["macro"] = "refreshed"
        diagnostics["status"] = "ok"
    except Exception as e:
        diagnostics["status"] = "failed"
        diagnostics["error"] = str(e)
        traceback.print_exc()
    return diagnostics


# =============================================================================
# Step 2: Walk-forward retraining (E3b)
# =============================================================================
def retrain_walkforward() -> dict:
    """Перезапустить E3b walk-forward — обновит trades.csv, predictions, metrics."""
    print("\n" + "=" * 70)
    print("[2/4] RETRAIN E3b WALK-FORWARD")
    print("=" * 70)
    try:
        from experiments.e3_macro_adaptive import run_one_experiment
        from app.multi_asset.metal_loader import load_metals
        silver = load_metals()["silver"]
        # Полный feature frame (с macro, без ffill — для academic walk-forward)
        features = build_feature_frame(target="silver", ffill_limit=0).dropna()
        labels = build_multi_horizon_labels(
            silver["close"], silver["high"], silver["low"],
            horizons=[20], adaptive=True,
        )["label_20"]

        metrics = run_one_experiment(
            "e3b_adaptive", features, labels, silver,
            use_metalabel=False, top_k=30, n_trials=5,
        )
        print(f"  ✅ Walk-forward complete: Sharpe={metrics.get('sharpe', 0):.3f}, "
              f"trades={metrics.get('n_trades', 0)}")
        return metrics
    except Exception as e:
        print(f"  ❌ Walk-forward failed: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


# =============================================================================
# Step 3: Production inference (signal for today)
# =============================================================================
def generate_today_signal() -> dict:
    """Обучить финальную модель на всех данных и сгенерировать p_up на сегодня.

    Использует ffill_limit=5 для production — заполняет gaps в palladium,
    чтобы рядом со свежими датами не было NaN.
    """
    print("\n" + "=" * 70)
    print("[3/4] GENERATE TODAY'S SIGNAL")
    print("=" * 70)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    try:
        metals = load_metals()
        silver = metals["silver"]

        # Features с ffill для production (захват свежих дней)
        features = build_feature_frame(target="silver", metals=metals,
                                       ffill_limit=5).dropna()
        print(f"  Features: {len(features.columns)} cols × {len(features):,} rows")
        print(f"  Period: {features.index.min().date()} → {features.index.max().date()}")

        # Labels с adaptive barriers
        labels = build_multi_horizon_labels(
            silver["close"], silver["high"], silver["low"],
            horizons=[20], adaptive=True,
        )["label_20"]

        # Train: всё что было до 20 дней назад (purge + embargo)
        purge_date = features.index.max() - pd.Timedelta(days=21)
        train_mask = features.index <= purge_date
        common = features.index.intersection(labels.index)
        X = features.loc[common]
        y = labels.loc[common]
        mask = y.notna() & (X.index <= purge_date)
        X_train = X.loc[mask]
        y_train = y.loc[mask]
        print(f"  Train: {len(X_train):,} samples (до {X_train.index.max().date()})")

        # Feature selection top-30
        selector = SelectKBest(
            score_func=lambda Xs, ys: mutual_info_classif(Xs, ys, random_state=42),
            k=min(30, X_train.shape[1]),
        )
        selector.fit(X_train, y_train)
        sel_cols = X_train.columns[selector.get_support()].tolist()
        print(f"  Selected: {len(sel_cols)} features")

        # Train HistGB
        model = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=200,
            min_samples_leaf=30, random_state=42,
        )
        model.fit(X_train[sel_cols], y_train)

        # Predict на последние 5 доступных дней (включая сегодня)
        X_recent = features.iloc[-5:][sel_cols]
        proba = model.predict_proba(X_recent)
        classes = model.classes_

        # Извлекаем p_up (вероятность TP=1)
        p1_idx = list(classes).index(1) if 1 in classes else None
        p_up_today = float(proba[-1, p1_idx]) if p1_idx is not None else 0.0
        today_date = features.index[-1]

        recent_df = pd.DataFrame(proba, index=X_recent.index,
                                  columns=[f"p_{int(c)}" for c in classes])
        recent_df["close"] = silver["close"].reindex(recent_df.index)

        print(f"\n  Recent predictions:")
        print(recent_df.to_string())

        # Сигнал на сегодня
        ENTRY_THR = 0.48
        EXIT_THR = 0.35
        if p_up_today >= ENTRY_THR:
            signal = "BUY"
        elif p_up_today < EXIT_THR:
            signal = "SELL"
        else:
            signal = "HOLD"

        print(f"\n  → Date:    {today_date.date()}")
        print(f"  → Close:    {silver['close'].iloc[-1]:.2f}")
        print(f"  → p_up:     {p_up_today:.4f}")
        print(f"  → Signal:   {signal}")

        return {
            "date":      today_date.isoformat(),
            "close":     float(silver["close"].iloc[-1]),
            "p_up":      p_up_today,
            "signal":    signal,
            "entry_threshold": ENTRY_THR,
            "exit_threshold": EXIT_THR,
            "trail_pct": 0.12,
            "max_hold_days": 30,
            "cooldown_days": 25,
            "n_features_used": len(sel_cols),
            "selected_features": sel_cols,
            "train_samples": int(len(X_train)),
        }
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


# =============================================================================
# Step 4: Save reports
# =============================================================================
def save_training_report(metrics: dict) -> None:
    """Отчёт обучения — Sharpe, trades, etc."""
    path = TRAINING_DIR / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Также копируем актуальный trades.csv
    src = E3B_DIR / "trades.csv"
    if src.exists():
        dst = TRAINING_DIR / "trades.csv"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Plain-text сводка
    summary = TRAINING_DIR / "summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"E3b Daily Training Report — {TODAY}\n")
        f.write("=" * 50 + "\n\n")
        for key in ["sharpe", "annual_return", "max_dd", "win_rate",
                    "n_trades", "profit_factor", "oos_accuracy", "psr"]:
            v = metrics.get(key)
            if isinstance(v, float):
                if key in ("annual_return", "max_dd", "win_rate"):
                    f.write(f"  {key:<20s}: {v*100:+7.2f}%\n")
                else:
                    f.write(f"  {key:<20s}: {v:.3f}\n")
            else:
                f.write(f"  {key:<20s}: {v}\n")
    print(f"  ✅ Training report → {TRAINING_DIR}")


def save_trading_report(signal_info: dict) -> None:
    """Отчёт торгового сигнала на сегодня."""
    path = TRADING_DIR / "signal.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(signal_info, f, indent=2, default=str)

    summary = TRADING_DIR / "summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"E3b Daily Trading Signal — {TODAY}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"  Date:           {signal_info.get('date', '—')}\n")
        f.write(f"  Silver close:   ${signal_info.get('close', 0):.2f}\n")
        f.write(f"  Probability:    p_up = {signal_info.get('p_up', 0):.4f}\n")
        f.write(f"  Signal:         {signal_info.get('signal', '—')}\n\n")
        f.write(f"Trade execution params:\n")
        f.write(f"  Entry threshold:  {signal_info.get('entry_threshold', 0.48)}\n")
        f.write(f"  Exit threshold:   {signal_info.get('exit_threshold', 0.35)}\n")
        f.write(f"  Trailing stop:    {signal_info.get('trail_pct', 0.12)*100:.0f}%\n")
        f.write(f"  Max hold:         {signal_info.get('max_hold_days', 30)} days\n")
        f.write(f"  Cooldown:         {signal_info.get('cooldown_days', 25)} days\n\n")
        f.write(f"Model params:\n")
        f.write(f"  Features used:    {signal_info.get('n_features_used', 0)}\n")
        f.write(f"  Train samples:    {signal_info.get('train_samples', 0)}\n")
    print(f"  ✅ Trading report → {TRADING_DIR}")


# =============================================================================
# Optional: Telegram notification
# =============================================================================
def send_telegram(signal_info: dict, metrics: dict) -> None:
    """Опционально отправить уведомление в Telegram."""
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")
    if not token or not chat_id:
        print("  (Telegram credentials отсутствуют, skip)")
        return

    try:
        import urllib.request
        import urllib.parse

        sig = signal_info.get("signal", "—")
        sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")
        sig_date = str(signal_info.get('date', '—')).split('T')[0]
        text = (
            f"{sig_emoji} <b>E3b Daily Signal</b>\n\n"
            f"Date: <code>{sig_date}</code>\n"
            f"Silver: <code>${signal_info.get('close', 0):.2f}</code>\n"
            f"Probability: <code>{signal_info.get('p_up', 0):.3f}</code>\n"
            f"Signal: <b>{sig}</b>\n\n"
            f"<i>Walk-forward metrics:</i>\n"
            f"Sharpe <code>{metrics.get('sharpe', 0):.2f}</code> · "
            f"Win <code>{metrics.get('win_rate', 0)*100:.0f}%</code> · "
            f"Trades <code>{metrics.get('n_trades', 0)}</code>"
        )
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=15) as r:
            ok = r.status == 200
        print(f"  Telegram: {'✅ sent' if ok else '❌ failed'}")
    except Exception as e:
        print(f"  Telegram failed: {e}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="E3b daily run")
    parser.add_argument("--skip-training", action="store_true",
                        help="Пропустить walk-forward, только сигнал")
    parser.add_argument("--quick", action="store_true",
                        help="Не обновлять данные, использовать кеш")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Не отправлять Telegram")
    args = parser.parse_args()

    print("=" * 70)
    print(f"E3b DAILY RUN — {TODAY}")
    print("=" * 70)
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"Training output: {TRAINING_DIR}")
    print(f"Trading output:  {TRADING_DIR}")

    # Step 1
    if not args.quick:
        refresh_data()

    # Step 2
    if args.skip_training:
        print("\n[2/4] Skipping training (--skip-training)")
        # Загружаем последние известные метрики
        try:
            with open(E3B_DIR / "metrics.json", encoding="utf-8") as f:
                metrics = json.load(f)
        except FileNotFoundError:
            metrics = {}
    else:
        metrics = retrain_walkforward()

    # Step 3
    signal_info = generate_today_signal()

    # Step 4
    print("\n" + "=" * 70)
    print("[4/4] SAVE REPORTS")
    print("=" * 70)
    save_training_report(metrics)
    save_trading_report(signal_info)

    # Telegram
    if not args.no_telegram:
        print("\n  Sending Telegram notification...")
        send_telegram(signal_info, metrics)

    print("\n" + "=" * 70)
    print("DAILY E3b RUN COMPLETE ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
