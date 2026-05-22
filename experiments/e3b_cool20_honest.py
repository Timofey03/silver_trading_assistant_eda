"""HONEST CHECK: E3b с cool=20 на train period 2014-2024 (БЕЗ forward data).

Методологически корректно:
- Используем уже сохранённые E3b walk-forward predictions (модель не пересматриваем)
- Re-simulate trade execution с двумя config: cool=25 (baseline) и cool=20
- Срезаем последний 1.3 года (~2025) чтобы НЕ смотреть на forward данные
- Сравниваем: если cool=20 побеждает на train period → оптимизация защитима
- Если cool=20 проигрывает на train period → это был cherry-pick на forward

Это академически правильный подход:
  Сначала оптимизируем cool на train (2014-2024).
  Потом фиксируем и тестируем на forward (2025-2026).
  Если получается «честно» с улучшением — пишем в диплом как validated finding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.multi_asset.metal_loader import load_metals
from app.multi_asset.simulator import simulate_trades, trades_to_df, TradeConfig
from app.multi_asset.metrics import compute_all_metrics

OUT_DIR = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_cool20_honest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

E3B_PREDS = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "predictions.parquet"

# Train period — НЕ включаем forward (2025-2026)
TRAIN_END = pd.Timestamp("2024-12-31")


def simulate_with_config(preds, prices, cool_days, label):
    """Симулировать сделки с заданным cooldown."""
    cfg = TradeConfig(
        entry_threshold=0.48,
        exit_threshold=0.35,
        trail_pct=0.12,
        max_hold_days=30,
        cooldown_days=cool_days,
        commission_pct=0.001,
        direction_label=1,
    )
    trades, _ = simulate_trades(preds, prices, cfg)
    trades_df = trades_to_df(trades)
    metrics = compute_all_metrics(trades_df, n_trials=1) if not trades_df.empty else {}
    return trades_df, metrics


def main():
    print("=" * 80)
    print("HONEST CHECK: E3b cool=20 vs cool=25 на TRAIN period 2014-2024")
    print("(без forward данных — чтобы не было cherry-picking)")
    print("=" * 80)

    # ===== Load predictions and silver =====
    preds_full = pd.read_parquet(E3B_PREDS)
    silver = load_metals()["silver"]
    prices_full = silver[["close", "high", "low"]].reindex(preds_full.index)

    # ===== Cut to train-only period =====
    preds_train = preds_full[preds_full.index <= TRAIN_END]
    prices_train = prices_full[prices_full.index <= TRAIN_END]
    print(f"\nPeriod cut: {preds_train.index.min().date()} → "
          f"{preds_train.index.max().date()} ({len(preds_train)} predictions)")

    # ===== Test multiple cooldown values =====
    print("\nТестирую cooldown от 15 до 30 дней:")
    print(f"{'cool':>6} {'trades':>8} {'return':>10} {'sharpe':>8} "
          f"{'win':>6} {'maxDD':>8} {'PF':>6}")

    results = {}
    for cool in [15, 18, 20, 22, 25, 28, 30]:
        trades_df, metrics = simulate_with_config(
            preds_train, prices_train, cool, f"cool={cool}"
        )
        results[cool] = {
            "trades_df": trades_df,
            "metrics": metrics,
        }
        if metrics:
            print(f"{cool:>6} {metrics['n_trades']:>8} "
                  f"{metrics['total_return']*100:>+9.1f}% "
                  f"{metrics['sharpe']:>7.3f} "
                  f"{metrics['win_rate']*100:>5.1f}% "
                  f"{metrics['max_dd']*100:>7.1f}% "
                  f"{metrics['profit_factor']:>5.2f}")
        else:
            print(f"{cool:>6} {'0':>8} {'—':>10} {'—':>8} {'—':>6} {'—':>8} {'—':>6}")

    # ===== Baseline (cool=25) vs winner (cool=20) =====
    baseline = results[25]["metrics"]
    challenger = results[20]["metrics"]

    print("\n" + "=" * 80)
    print("ВЫВОД")
    print("=" * 80)

    if not baseline or not challenger:
        print("⚠ Не получилось обе симуляции")
        return

    print(f"\n📊 Baseline cool=25:")
    print(f"   Sharpe:        {baseline['sharpe']:.3f}")
    print(f"   Total return:  {baseline['total_return']*100:+.2f}%")
    print(f"   Trades:        {baseline['n_trades']}")
    print(f"   Win rate:      {baseline['win_rate']*100:.1f}%")
    print(f"   Max DD:        {baseline['max_dd']*100:+.1f}%")

    print(f"\n🎯 Challenger cool=20:")
    print(f"   Sharpe:        {challenger['sharpe']:.3f}")
    print(f"   Total return:  {challenger['total_return']*100:+.2f}%")
    print(f"   Trades:        {challenger['n_trades']}")
    print(f"   Win rate:      {challenger['win_rate']*100:.1f}%")
    print(f"   Max DD:        {challenger['max_dd']*100:+.1f}%")

    sharpe_diff = challenger['sharpe'] - baseline['sharpe']
    return_diff = (challenger['total_return'] - baseline['total_return']) * 100

    print(f"\n📈 Изменения (challenger vs baseline):")
    print(f"   ΔSharpe:       {sharpe_diff:+.3f}")
    print(f"   ΔReturn:       {return_diff:+.2f}pp")

    if sharpe_diff > 0.05 and challenger['total_return'] > baseline['total_return']:
        verdict = "✅ ВАЛИДИРОВАНО: cool=20 действительно лучше cool=25"
        verdict_detail = (
            "Улучшение видно даже на train period (2014-2024) БЕЗ доступа\n"
            "к forward данным. Это означает что cool=20 — реальное улучшение,\n"
            "а не cherry-pick на forward. Можно использовать в финальной модели."
        )
    elif sharpe_diff < -0.05:
        verdict = "❌ ОТКЛОНЕНО: cool=20 хуже на train period"
        verdict_detail = (
            "На train period (2014-2024) cool=25 лучше cool=20.\n"
            "Преимущество cool=20 на forward (2025-2026) — это cherry-pick,\n"
            "не статистически robust. Остаёмся с cool=25 как baseline."
        )
    else:
        verdict = "⚠ НЕОДНОЗНАЧНО: cool=20 примерно равен cool=25 на train"
        verdict_detail = (
            "На train period (2014-2024) cool=20 и cool=25 дают похожий результат.\n"
            "Преимущество cool=20 на forward (2025-2026) может быть случайностью.\n"
            "Безопаснее остаться с cool=25 ради академической prudency."
        )

    print(f"\n{verdict}")
    print(verdict_detail)

    # ===== Save =====
    summary = {
        "train_period_start": preds_train.index.min().isoformat(),
        "train_period_end": preds_train.index.max().isoformat(),
        "results": {
            str(c): r["metrics"] for c, r in results.items()
        },
        "verdict": verdict,
        "sharpe_change": sharpe_diff,
        "return_change_pp": return_diff,
    }
    with open(OUT_DIR / "honest_check_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save cool=20 trades для интеграции (если победил)
    if not results[20]["trades_df"].empty:
        results[20]["trades_df"].to_csv(OUT_DIR / "trades_cool20.csv", index=False)
    if not results[25]["trades_df"].empty:
        results[25]["trades_df"].to_csv(OUT_DIR / "trades_cool25.csv", index=False)

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
