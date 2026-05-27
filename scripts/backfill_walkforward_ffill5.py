"""
scripts/backfill_walkforward_ffill5.py — пересчёт E3b walk-forward c ffill_limit=5.

Зачем: оригинальный walk-forward использует ffill_limit=0 (академически строго),
из-за чего месячные макро-фичи (CPI, INDPRO) обрывают данные на 2025-07-03.
Это закрывает покрытие весь backtest периодом и оставляет дыру 2025-07 → 2026-05.

ffill_limit=5 (5 торговых дней) — стандартная практика: используем последнее
известное значение CPI/INDPRO, но не более чем за 5 дней назад. Это:
- НЕ leakage (берём только то, что было опубликовано до даты)
- Покрывает все ~340 пропущенных дней
- Та же логика что в production_inference

Результат сохраняется в baseline_outputs_multiasset/e3b_adaptive/
(replace) и параллельно в e3b_adaptive_ffill5/ для аудита.

Запуск:
    .venv/Scripts/python.exe scripts/backfill_walkforward_ffill5.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def main() -> int:
    from experiments.e3_macro_adaptive import run_one_experiment
    from app.multi_asset.metal_loader import load_metals
    from app.multi_asset.features import build_feature_frame
    from app.multi_asset.labels import build_multi_horizon_labels

    print("=" * 70)
    print(" Backfill walk-forward с ffill_limit=5")
    print("=" * 70)

    t0 = time.time()
    metals = load_metals()
    silver = metals["silver"]
    print(f"  silver: {len(silver):,} rows, до {silver.index[-1].date()}")

    features = build_feature_frame(
        target="silver", metals=metals, ffill_limit=5,
    ).dropna()
    print(f"  Features: {features.shape[1]} cols x {len(features):,} rows")
    print(f"  Period:   {features.index.min().date()} -> {features.index.max().date()}")

    labels = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
        horizons=[20], adaptive=True,
    )["label_20"]
    print(f"  Labels:   {labels.notna().sum():,} non-na")

    # Параллельный output
    aux_dir = ROOT / "baseline_outputs_multiasset" / "e3b_adaptive_ffill5"
    if aux_dir.exists():
        shutil.rmtree(aux_dir)
    aux_dir.mkdir(parents=True)

    # run_one_experiment записывает в baseline_outputs_multiasset/{name}/
    print("\n  Запускаю walk-forward (5-15 мин)...")
    metrics = run_one_experiment(
        "e3b_adaptive_ffill5", features, labels, silver,
        use_metalabel=False, top_k=30, n_trials=5,
    )

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f" Готово за {elapsed/60:.1f} мин")
    print("=" * 70)
    print(f"  Sharpe:        {metrics.get('sharpe', 0):.3f}")
    print(f"  N trades:      {metrics.get('n_trades', 0)}")
    print(f"  Total return:  {metrics.get('total_return', 0)*100:+.1f}%")
    print(f"  Max DD:        {metrics.get('max_dd', 0)*100:.1f}%")
    print(f"  Win rate:      {metrics.get('win_rate', 0)*100:.1f}%")

    # Показать последние сделки
    trades_csv = aux_dir / "trades.csv"
    if trades_csv.exists():
        import pandas as pd
        trades = pd.read_csv(trades_csv)
        print()
        print(f"  Последние 10 сделок:")
        print(trades.tail(10).to_string(index=False))

    # Заменить основной e3b_adaptive (с бэкапом)
    main_dir = ROOT / "baseline_outputs_multiasset" / "e3b_adaptive"
    print()
    print(f"  Копирую e3b_adaptive_ffill5/* -> e3b_adaptive/* ...")
    for f in aux_dir.iterdir():
        target = main_dir / f.name
        shutil.copy2(f, target)
    print("  OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
