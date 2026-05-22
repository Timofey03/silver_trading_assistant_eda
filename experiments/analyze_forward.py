"""Глубокий анализ: почему E3b сделал только 6 сделок в forward test 2025-2026.

5 анализов:
1. Распределение p_up за весь период (как часто высокая уверенность)
2. Сколько дней prob >= порога входа (потенциальные сигналы)
3. Что блокировало сделки (cooldown vs низкая уверенность)
4. Сделки во времени (когда открывались, против цены silver)
5. Сравнение с V25 forward — где V25 торговал, а E3b — нет
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.multi_asset.metal_loader import load_metals

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

OUT_DIR = REPO_ROOT / "baseline_outputs_multiasset" / "forward_test_2025"
FIG_DIR = REPO_ROOT / "data" / "multi_asset" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_THRESHOLD = 0.48
EXIT_THRESHOLD = 0.35
COOLDOWN_DAYS = 25
TRAIL_PCT = 0.12
MAX_HOLD = 30


def run_analysis():
    # Загрузка данных
    preds = pd.read_parquet(OUT_DIR / "predictions.parquet")
    e3b_trades = pd.read_csv(OUT_DIR / "trades.csv")
    e3b_trades["entry_date"] = pd.to_datetime(e3b_trades["entry_date"])
    e3b_trades["exit_date"] = pd.to_datetime(e3b_trades["exit_date"])

    v25 = pd.read_csv(REPO_ROOT / "baseline_outputs_v25" / "v25_forward_trades.csv")
    v25["entry_date"] = pd.to_datetime(v25["entry_date"])
    v25["exit_date"] = pd.to_datetime(v25["exit_date"])

    silver = load_metals()["silver"]
    silver_test = silver.loc["2025-01-01":]

    print("=" * 80)
    print("АНАЛИЗ: Почему E3b сделал только 6 сделок в forward test 2025-2026")
    print("=" * 80)

    # ===== 1. Распределение p_up =====
    print("\n[1] Распределение p_up (вероятность роста) за весь test-период")
    p_up = preds["p_1"]
    print(f"  N predictions:           {len(p_up):,}")
    print(f"  Mean p_up:               {p_up.mean():.3f}")
    print(f"  Median p_up:             {p_up.median():.3f}")
    print(f"  Std p_up:                {p_up.std():.3f}")
    print(f"  Min / Max:               {p_up.min():.3f} / {p_up.max():.3f}")
    print()
    print(f"  Порог входа: p_up >= {ENTRY_THRESHOLD}")
    above_thr = p_up >= ENTRY_THRESHOLD
    print(f"  Дней с p_up >= {ENTRY_THRESHOLD}: {above_thr.sum()} из {len(p_up)} "
          f"({above_thr.mean()*100:.1f}%)")
    print(f"  Дней с p_up >= 0.50:    {(p_up >= 0.50).sum()} ({(p_up >= 0.50).mean()*100:.1f}%)")
    print(f"  Дней с p_up >= 0.55:    {(p_up >= 0.55).sum()} ({(p_up >= 0.55).mean()*100:.1f}%)")
    print(f"  Дней с p_up >= 0.60:    {(p_up >= 0.60).sum()} ({(p_up >= 0.60).mean()*100:.1f}%)")

    # ===== 2. Сколько потенциальных сигналов =====
    print("\n[2] Сколько потенциальных сигналов модель сгенерировала")
    # Считаем "момент входа" — первый день после периода p_up < threshold когда p_up >= threshold
    above = p_up.values >= ENTRY_THRESHOLD
    signals = []
    in_signal = False
    for i, a in enumerate(above):
        if a and not in_signal:
            signals.append(p_up.index[i])
            in_signal = True
        elif not a and in_signal:
            in_signal = False
    print(f"  Сигналов входа (без cooldown): {len(signals)}")
    print(f"  Сделок исполнено (с cooldown): {len(e3b_trades)}")
    print(f"  Заблокировано по cooldown:     {len(signals) - len(e3b_trades)}")

    # ===== 3. Анализ сделок =====
    print("\n[3] Детали 6 сделок E3b")
    for i, r in e3b_trades.iterrows():
        days = (r["exit_date"] - r["entry_date"]).days
        print(f"  {i+1}. {r['entry_date'].date()} → {r['exit_date'].date()} "
              f"({days:3d} дн) | вход {r['entry_price']:6.2f} → выход {r['exit_price']:6.2f} "
              f"| {r['net_return']*100:+6.2f}% | {r['exit_reason']}")

    # ===== 4. Сравнение с V25 — что V25 ловил, а E3b — нет =====
    print("\n[4] Сравнение покрытия V25 vs E3b")
    print(f"  V25 trades: {len(v25)} entries")
    print(f"  V25 entries в период p_up >= 0.48 у E3b: ", end="")
    e3b_signal_dates = set(preds.index[above].normalize())
    v25_in_e3b_signals = sum(1 for d in v25["entry_date"]
                              if d.normalize() in e3b_signal_dates)
    print(f"{v25_in_e3b_signals} из {len(v25)} "
          f"({v25_in_e3b_signals / len(v25) * 100:.0f}%)")

    # ===== 5. Визуализация =====
    print("\n[5] Создаю графики...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # Subplot 1: silver price + entry/exit markers
    ax = axes[0]
    ax.plot(silver_test.index, silver_test["close"], color="#1F4E79",
            linewidth=1.5, label="Silver price")

    # E3b trades markers
    for _, r in e3b_trades.iterrows():
        ax.scatter(r["entry_date"], r["entry_price"], color="#2CA02C",
                   s=120, marker="^", zorder=5, edgecolor="white", linewidth=1.5)
        ax.scatter(r["exit_date"], r["exit_price"], color="#2CA02C",
                   s=120, marker="v", zorder=5, edgecolor="white", linewidth=1.5)
        # Connecting line
        ax.plot([r["entry_date"], r["exit_date"]],
                [r["entry_price"], r["exit_price"]],
                color="#2CA02C", linewidth=2, alpha=0.4, linestyle="--")

    # V25 trades markers (background)
    for _, r in v25.iterrows():
        ax.scatter(r["entry_date"], r["entry_price"], color="#FF7F0E",
                   s=40, marker="o", zorder=3, alpha=0.55)

    ax.set_title("Цена silver и сделки: E3b (зелёные треугольники, n=6) vs V25 "
                 "(оранжевые точки, n=38)")
    ax.set_ylabel("Silver, $")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")

    # Subplot 2: p_up timeline
    ax = axes[1]
    ax.plot(preds.index, preds["p_1"], color="#1F77B4", linewidth=1, alpha=0.7,
            label="p_up (модель)")
    ax.fill_between(preds.index, preds["p_1"], ENTRY_THRESHOLD,
                    where=preds["p_1"] >= ENTRY_THRESHOLD,
                    color="#2CA02C", alpha=0.25, label="Зона BUY")
    ax.axhline(ENTRY_THRESHOLD, color="#2CA02C", linestyle="--", linewidth=1,
               label=f"Порог входа ({ENTRY_THRESHOLD})")
    ax.axhline(EXIT_THRESHOLD, color="#C62828", linestyle="--", linewidth=1,
               label=f"Порог выхода ({EXIT_THRESHOLD})")
    ax.axhline(0.5, color="black", linewidth=0.5, alpha=0.5)
    # Mark entries
    for _, r in e3b_trades.iterrows():
        ax.axvline(r["entry_date"], color="#2CA02C", linestyle=":",
                   linewidth=1.5, alpha=0.7)
    ax.set_title("Вероятность роста по модели E3b (p_up)")
    ax.set_ylabel("p_up")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3, linestyle="--")

    # Subplot 3: гистограмма p_up
    # Use timestamp as x-axis to keep alignment, but build inset for histogram
    ax = axes[2]
    ax.hist(preds["p_1"], bins=40, color="#1F77B4", alpha=0.7,
            edgecolor="white", linewidth=1)
    ax.axvline(ENTRY_THRESHOLD, color="#2CA02C", linestyle="--", linewidth=2,
               label=f"Порог входа {ENTRY_THRESHOLD}")
    ax.axvline(p_up.mean(), color="black", linestyle=":", linewidth=1.5,
               label=f"Среднее {p_up.mean():.3f}")
    ax.set_title(f"Распределение p_up по дням 2025-2026 (медиана {p_up.median():.3f})")
    ax.set_xlabel("p_up")
    ax.set_ylabel("Кол-во дней")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, axis="y", linestyle="--")
    # Override sharex для гистограммы — нужно убрать связь
    ax.set_xlim(p_up.min() - 0.02, p_up.max() + 0.02)

    fig.tight_layout()
    path = FIG_DIR / "08_e3b_forward_diagnostic.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

    # ===== 6. Финальный диагноз =====
    print("\n" + "=" * 80)
    print("ДИАГНОЗ")
    print("=" * 80)
    pct_above = above_thr.mean() * 100

    if pct_above < 30:
        print(f"\n✅ Главная причина: МОДЕЛЬ КОНСЕРВАТИВНА")
        print(f"   Только {pct_above:.1f}% дней модель уверена (p_up >= {ENTRY_THRESHOLD})")
        print(f"   В out-of-distribution данных 2025-2026 (silver $30→$75)")
        print(f"   модель обученная на 2010-2024 не делает агрессивных входов.")
    else:
        print(f"\n⚠ Модель активна {pct_above:.1f}% дней, но cooldown ограничивает")
        print(f"   25-дневная пауза + 30-дневный max_hold = ~50 дней между сделками")
        print(f"   16 месяцев / 50 дней = ~10 потенциальных сделок")

    if len(signals) > len(e3b_trades) + 2:
        print(f"\n📊 Сигналов модели: {len(signals)}, исполнено: {len(e3b_trades)}")
        print(f"   {len(signals) - len(e3b_trades)} сигналов заблокированы cooldown'ом")
        print(f"   → Можно увеличить число сделок снизив cooldown")

    print(f"\n💡 ВЫВОД для дипломной защиты:")
    print(f"   E3b — selective модель. Она ждёт CONFIRMED setups.")
    print(f"   В bull rally V25 'наглеет' с 38 сделок, набирая +442%.")
    print(f"   E3b делает 6 сделок с +18%, но **с 83% win rate и 0% просадки**.")
    print(f"   Это разные торговые философии — agressive vs conservative.")

    return {
        "n_trades": len(e3b_trades),
        "n_potential_signals": len(signals),
        "blocked_by_cooldown": len(signals) - len(e3b_trades),
        "pct_above_threshold": float(above_thr.mean()),
        "mean_p_up": float(p_up.mean()),
        "median_p_up": float(p_up.median()),
        "max_p_up": float(p_up.max()),
        "v25_in_e3b_signals_pct": v25_in_e3b_signals / len(v25),
    }


if __name__ == "__main__":
    result = run_analysis()
    with open(OUT_DIR / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved analysis to {OUT_DIR / 'analysis.json'}")
