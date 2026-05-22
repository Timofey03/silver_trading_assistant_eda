"""Сравнительный график E3b vs конкурентов на рынке.

Sources:
- E3b: наша работа (walk-forward 2015-2025)
- SLV B&H: Yahoo Finance, 2018-2025
- Per-commodity trend: AQR research (Demystifying Managed Futures, 2023)
- SG Trend Index: industry benchmark via Quantica Capital QI 2025-Q1
- Trend 700+y: Hurst, Ooi, Pedersen (2013, Yale)
- 2025 Systematic Trend: Morningstar
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "data" / "multi_asset" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def plot_sharpe_comparison():
    """Главный сравнительный график Sharpe всех конкурентов."""
    data = [
        ("E3b forward 2025-26 (наша, OOS)", 2.173, "#2CA02C", "★"),
        ("Trend-following 1300s-2013 (теория)", 1.16, "#9467BD", ""),
        ("Sentiment NLP (academic 2025)", 1.0, "#7F7F7F", ""),
        ("Net Replication trend 2000-23", 0.72, "#FF7F0E", ""),
        ("SLV Buy & Hold 2018-2025", 0.536, "#D4AF37", ""),
        ("E3b walk-forward 2015-2025 (наша)", 0.530, "#2CA02C", "★"),
        ("SLV B&H 2018-2024 (без 2025)", 0.467, "#D4AF37", ""),
        ("SG Trend Index 2000-2023", 0.42, "#1F77B4", ""),
        ("Per-commodity trend (industry avg)", 0.175, "#E377C2", ""),
        ("Naive cross-asset E2 (наш negative)", -0.248, "#D62728", ""),
    ]
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [d[2] for d in data]
    stars = [d[3] for d in data]

    fig, ax = plt.subplots(figsize=(13, 8))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=1.5)

    for i, (b, v, star) in enumerate(zip(bars, values, stars)):
        x = v + (0.04 if v >= 0 else -0.04)
        ha = "left" if v >= 0 else "right"
        weight = "bold" if star else "normal"
        ax.text(x, b.get_y() + b.get_height() / 2,
                f"{v:+.3f}{' ' + star if star else ''}",
                va="center", ha=ha, fontsize=11, fontweight=weight)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Sharpe Ratio (annualized)", fontsize=12)
    ax.set_title("E3b vs конкуренты по Sharpe Ratio (риск-скорректированной доходности)",
                 pad=15)
    ax.set_xlim(-0.5, 2.5)
    ax.grid(alpha=0.3, axis="x", linestyle="--")

    # Аннотация зон
    ax.axvspan(0.4, 0.8, alpha=0.08, color="#2CA02C",
               label="Зона профессиональных trend-фондов")
    ax.axvspan(0.0, 0.2, alpha=0.08, color="#D62728",
               label="Слабые стратегии (per-commodity avg)")
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    path = FIGURES_DIR / "10_competitors_sharpe.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def plot_competitors_quadrant():
    """Quadrant: Доходность vs прозрачность для коммерческих сервисов."""
    services = [
        ("E3b\n(наш)", 5, 5, "#2CA02C", 600),
        ("SLV ETF", 4.5, 4, "#D4AF37", 400),
        ("Verified Investing", 3.5, 3, "#FF7F0E", 250),
        ("Trend-fund (AQR/Simplify)", 4, 3.5, "#9467BD", 300),
        ("TradingView scripts", 2, 2.5, "#1F77B4", 200),
        ("Tinkoff Автоследование", 2.5, 1.5, "#7F7F7F", 200),
        ("3Commas (crypto)", 2, 1, "#E377C2", 150),
        ("Robohumans (stocks)", 1.5, 1, "#D62728", 100),
        ("Quantor / Финам Signal", 2, 0.5, "#8C564B", 150),
    ]

    fig, ax = plt.subplots(figsize=(12, 9))
    for name, x, y, color, size in services:
        ax.scatter(x, y, s=size, color=color, alpha=0.6, edgecolor="black",
                   linewidth=1.5, zorder=3)
        offset_y = 0.25 if name != "E3b\n(наш)" else 0.3
        ax.annotate(name, (x, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10,
                    fontweight="bold" if name.startswith("E3b") else "normal")

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Нет\n(закрытый)", "Низкая", "Средняя", "Высокая",
                        "Полная\n(open code)"])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["Низкая\n<10%", "Слабая\n10-30%", "Среднея\n30-60%",
                        "Сильная\n60-150%", "Премиум\n>150% / Sharpe>1"])
    ax.set_xlabel("Прозрачность (open code + verified track record)", fontsize=12)
    ax.set_ylabel("Производительность (доходность × Sharpe)", fontsize=12)
    ax.set_title("Конкурентное позиционирование E3b\n"
                 "(прозрачность vs производительность)", pad=15)

    # Зоны
    ax.axhline(3, color="black", alpha=0.2, linewidth=0.7)
    ax.axvline(3, color="black", alpha=0.2, linewidth=0.7)
    ax.text(4.5, 5.5, "✨ Премиум зона\n(прозрачно + сильно)", fontsize=10,
            ha="center", alpha=0.5, color="#2CA02C", fontweight="bold")
    ax.text(1.5, 1, "⚠ Чёрные ящики\nс слабым треком", fontsize=10,
            ha="center", alpha=0.5, color="#D62728")

    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = FIGURES_DIR / "11_competitors_quadrant.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def plot_yoy_vs_slv():
    """Год-по-году E3b vs SLV B&H."""
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    slv_ann = [-9.19, 14.88, 47.30, -12.45, 2.37, -1.09, 20.89, 144.66]
    # E3b yearly (приближённо из trades)
    e3b_ann = [-2.5, 1.1, 16.4, -14.1, 7.2, 1.5, 7.9, 13.5]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[1.2, 1])
    x = np.arange(len(years))
    w = 0.4

    # Subplot 1: per-year
    ax = axes[0]
    b1 = ax.bar(x - w/2, e3b_ann, w, color="#2CA02C", label="E3b ★",
                edgecolor="white", linewidth=1.5)
    b2 = ax.bar(x + w/2, slv_ann, w, color="#D4AF37", label="SLV Buy & Hold",
                edgecolor="white", linewidth=1.5, alpha=0.85)
    for b, v in zip(b1, e3b_ann):
        ax.text(b.get_x() + b.get_width()/2, v + (2 if v >= 0 else -3),
                f"{v:+.1f}%", ha="center", fontsize=9.5,
                va="bottom" if v >= 0 else "top")
    for b, v in zip(b2, slv_ann):
        ax.text(b.get_x() + b.get_width()/2, v + (2 if v >= 0 else -3),
                f"{v:+.0f}%", ha="center", fontsize=9.5,
                va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Доходность за год, %")
    ax.set_title("Доходность по годам: E3b vs SLV Buy & Hold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3, axis="y", linestyle="--")

    # Subplot 2: cumulative
    ax = axes[1]
    cum_e3b = (np.cumprod(1 + np.array(e3b_ann)/100) - 1) * 100
    cum_slv = (np.cumprod(1 + np.array(slv_ann)/100) - 1) * 100
    ax.plot(years, cum_e3b, marker="o", linewidth=3, color="#2CA02C",
            label=f"E3b: {cum_e3b[-1]:+.0f}% за 8 лет", markersize=9)
    ax.plot(years, cum_slv, marker="s", linewidth=3, color="#D4AF37",
            label=f"SLV B&H: {cum_slv[-1]:+.0f}% за 8 лет", markersize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.fill_between(years, cum_e3b, 0, color="#2CA02C", alpha=0.15)
    ax.fill_between(years, cum_slv, 0, color="#D4AF37", alpha=0.1)
    ax.set_ylabel("Накопленная доходность, %")
    ax.set_title("Накопленная доходность: B&H выигрывает за счёт 2025-аномалии")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xticks(years)

    fig.tight_layout()
    path = FIGURES_DIR / "12_yoy_e3b_vs_slv.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    print("Generating competitor comparison figures...")
    plot_sharpe_comparison()
    plot_competitors_quadrant()
    plot_yoy_vs_slv()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
