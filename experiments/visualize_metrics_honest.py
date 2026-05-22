"""Честные визуализации risk-adjusted метрик E3b vs реальных конкурентов.

Использует **публично подтверждённые** годовые доходности:
- E3b: наш walk-forward
- SLV: Yahoo Finance
- AQMIX: Yahoo Finance (AQR Managed Futures, $3B AUM)
- Medallion: net returns by year (publicly disclosed estimates)
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

# Цветовая палитра
COLORS = {
    "e3b":       "#2CA02C",
    "slv":       "#D4AF37",
    "aqr":       "#1F77B4",
    "medallion": "#9467BD",
}

# Year-by-year returns
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
e3b   = [-2.5, +1.1, +16.4, -14.1, +7.2, +1.5, +7.9, +13.5]
slv   = [-9.19, +14.88, +47.30, -12.45, +2.37, -1.09, +20.89, +144.66]
aqmix = [-8.88, +1.93, -0.41, -1.06, +35.38, +1.80, +8.41, None]  # 2025 N/A
medal = [+30,   +30,   +76,   +48,   +21,   +30,   +30,   +30]    # estimates


# =============================================================================
# 13. Calmar comparison — главная разоблачающая метрика
# =============================================================================
def plot_calmar_comparison():
    strategies = ["AQR Managed\nFutures", "SLV ETF\nBuy & Hold", "E3b\n(наша)",
                  "Per-commodity\ntrend (industry)"]
    cagrs = [4.57, 19.03, 3.47, 5.0]      # %
    max_dds = [-1.5, -12.5, -14.1, -25.0]  # %
    calmars = [c / abs(d) for c, d in zip(cagrs, max_dds)]
    colors = [COLORS["aqr"], COLORS["slv"], COLORS["e3b"], "#9E9E9E"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # 1: CAGR
    ax = axes[0]
    bars = ax.bar(strategies, cagrs, color=colors, edgecolor="white", linewidth=2)
    for b, v in zip(bars, cagrs):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f"{v:+.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("CAGR, % годовых")
    ax.set_title("Доходность (CAGR)")
    ax.grid(alpha=0.3, axis="y", linestyle="--")
    ax.set_ylim(0, max(cagrs) * 1.2)

    # 2: Max DD
    ax = axes[1]
    bars = ax.bar(strategies, max_dds, color=colors, edgecolor="white", linewidth=2)
    for b, v in zip(bars, max_dds):
        ax.text(b.get_x() + b.get_width()/2, v - 0.5,
                f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold",
                va="top")
    ax.set_ylabel("Max Drawdown, %")
    ax.set_title("Максимальная просадка (меньше = лучше)")
    ax.grid(alpha=0.3, axis="y", linestyle="--")
    ax.invert_yaxis()

    # 3: Calmar
    ax = axes[2]
    bars = ax.bar(strategies, calmars, color=colors, edgecolor="white", linewidth=2)
    for b, v in zip(bars, calmars):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Calmar Ratio (CAGR / |Max DD|)")
    ax.set_title("Calmar Ratio — return-per-risk")
    ax.grid(alpha=0.3, axis="y", linestyle="--")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(3.5, 1.05, "Calmar = 1\n(точка безубытка)",
            color="red", fontsize=9, alpha=0.7, ha="right")

    fig.suptitle("Calmar Ratio — где E3b проигрывает AQR Managed Futures",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = FIGURES_DIR / "13_calmar_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 14. Полный radar — 6 метрик × 4 стратегии
# =============================================================================
def plot_metrics_radar_extended():
    metric_labels = ["Sharpe", "Sortino", "Calmar", "Hit Rate", "Low DD", "Stability"]
    # Нормализация в 0..1 (выше = лучше)
    # Sharpe: 0-2.5 → 0-1 (Medallion отметаем как outlier)
    # Sortino: 0-5 → 0-1
    # Calmar: 0-3 → 0-1
    # Hit Rate: 0-1
    # Low DD: 1 - |DD|/0.3 (clamp 0-1)
    # Stability: 1 - std/0.5 (lower std = higher)

    def normalize(strategy_data):
        s, sor, cal, hit, dd, std = strategy_data
        return [
            min(s / 1.5, 1),                     # Sharpe (1.5 = excellent)
            min(sor / 5, 1),                      # Sortino (5 = excellent)
            min(cal / 3, 1),                      # Calmar (3 = excellent)
            hit,                                   # Hit rate already 0-1
            max(0, 1 - abs(dd) / 0.30),           # Low DD (30% = bad)
            max(0, 1 - std / 0.50),               # Low std (50% = bad)
        ]

    data = {
        "E3b (наша)":     normalize([0.429, 0.668, 0.246, 0.75, -0.141, 0.090]),
        "SLV Buy & Hold": normalize([0.536, 5.427, 1.528, 0.62, -0.125, 0.483]),
        "AQR Managed Futures": normalize([0.404, 1.380, 3.119, 0.57, -0.015, 0.132]),
    }

    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    color_map = {"E3b (наша)": COLORS["e3b"],
                 "SLV Buy & Hold": COLORS["slv"],
                 "AQR Managed Futures": COLORS["aqr"]}

    for name, vals in data.items():
        v = vals + vals[:1]
        is_e3b = name.startswith("E3b")
        ax.plot(angles, v, color=color_map[name],
                linewidth=3 if is_e3b else 2,
                label=name,
                alpha=1.0 if is_e3b else 0.7)
        ax.fill(angles, v, color=color_map[name],
                alpha=0.20 if is_e3b else 0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], color="gray", fontsize=9)
    ax.set_title("Многомерное сравнение метрик (выше = лучше)\n"
                 "Каждая метрика нормализована в 0..1",
                 pad=25, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=11)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    path = FIGURES_DIR / "14_metrics_radar_extended.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 15. Year-by-year heatmap — кто когда лучше
# =============================================================================
def plot_year_winner_heatmap():
    strategies = ["E3b", "SLV B&H", "AQR Managed", "Medallion"]
    data = np.array([e3b, slv,
                     [v if v is not None else np.nan for v in aqmix],
                     medal])  # 4 × 8

    fig, ax = plt.subplots(figsize=(13, 6))
    # Используем diverging colormap (красный для негатива, зелёный для позитива)
    vmax = 50  # Cap for readability
    im = ax.imshow(np.clip(data, -vmax, vmax), cmap="RdYlGn",
                   aspect="auto", vmin=-30, vmax=30)

    # Annotate cells with actual values
    for i in range(len(strategies)):
        for j in range(len(years)):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=11, color="gray")
            else:
                color = "white" if abs(v) > 20 else "black"
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_title("Год-по-году по 4 стратегиям (% годовая доходность)",
                 pad=15, fontsize=14)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Годовая доходность %")

    # Add winners row
    winners_idx = np.nanargmax(data, axis=0)
    ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(strategies), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)

    # Mark winners with star
    for j, idx in enumerate(winners_idx):
        ax.scatter(j, idx, marker="*", s=200, color="#FFD700",
                   edgecolor="black", linewidth=1.5, zorder=5)

    fig.tight_layout()
    path = FIGURES_DIR / "15_yearly_winners.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 16. Distribution of returns — skewness анализ
# =============================================================================
def plot_returns_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    data = [
        ("E3b\nSkew = -0.53", e3b, COLORS["e3b"]),
        ("SLV B&H\nSkew = +1.71", slv, COLORS["slv"]),
        ("AQR Managed Futures\nSkew = +1.49", [v for v in aqmix if v is not None],
         COLORS["aqr"]),
    ]

    for ax, (title, returns, color) in zip(axes, data):
        ax.hist(returns, bins=8, color=color, alpha=0.75, edgecolor="black", linewidth=1.2)
        ax.axvline(np.mean(returns), color="black", linewidth=2,
                   linestyle="--", label=f"Mean {np.mean(returns):+.1f}%")
        ax.axvline(0, color="red", linewidth=1, alpha=0.5)
        ax.set_xlabel("Годовая доходность, %")
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Количество лет")
    fig.suptitle("Распределение годовых доходностей: E3b имеет хвост в минус (плохо)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = FIGURES_DIR / "16_returns_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 17. Final dashboard — 4-panel summary
# =============================================================================
def plot_final_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1: Cumulative wealth
    ax = axes[0, 0]
    x = list(range(2018, 2026))
    cum_e3b   = np.cumprod(1 + np.array(e3b)/100)
    cum_slv   = np.cumprod(1 + np.array(slv)/100)
    cum_aqr   = np.cumprod(1 + np.array([v if v is not None else 0 for v in aqmix])/100)

    ax.plot(x, (cum_e3b - 1) * 100, marker="o", linewidth=3,
            color=COLORS["e3b"], label="E3b (наша) ★", markersize=9)
    ax.plot(x, (cum_slv - 1) * 100, marker="s", linewidth=2.5,
            color=COLORS["slv"], label="SLV B&H ETF", markersize=8)
    ax.plot(x, (cum_aqr - 1) * 100, marker="^", linewidth=2.5,
            color=COLORS["aqr"], label="AQR Managed Futures", markersize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Накопленная доходность 2018-2025")
    ax.set_ylabel("Total return, %")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")

    # 2: Sharpe / Sortino / Calmar comparison
    ax = axes[0, 1]
    metric_names = ["Sharpe", "Sortino", "Calmar"]
    e3b_metrics = [0.429, 0.668, 0.246]
    slv_metrics = [0.536, 5.427, 1.528]
    aqr_metrics = [0.404, 1.380, 3.119]

    x_pos = np.arange(len(metric_names))
    w = 0.27
    ax.bar(x_pos - w, e3b_metrics, w, color=COLORS["e3b"], label="E3b ★")
    ax.bar(x_pos, slv_metrics, w, color=COLORS["slv"], label="SLV B&H")
    ax.bar(x_pos + w, aqr_metrics, w, color=COLORS["aqr"], label="AQR")
    for i, vals in enumerate([e3b_metrics, slv_metrics, aqr_metrics]):
        offset = (i - 1) * w
        for j, v in enumerate(vals):
            ax.text(j + offset, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names)
    ax.set_title("Risk-adjusted метрики (3 ключевых)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y", linestyle="--")
    ax.set_ylim(0, 6)

    # 3: Drawdown over time
    ax = axes[1, 0]
    for label, returns, color in [("E3b", e3b, COLORS["e3b"]),
                                    ("SLV B&H", slv, COLORS["slv"]),
                                    ("AQR", [v if v is not None else 0 for v in aqmix],
                                     COLORS["aqr"])]:
        cum = np.cumprod(1 + np.array(returns) / 100)
        peaks = np.maximum.accumulate(cum)
        dd = (cum / peaks - 1) * 100
        is_e3b = label == "E3b"
        ax.plot(x, dd, marker="o", linewidth=3 if is_e3b else 2,
                color=color, label=f"{label} (min: {dd.min():.1f}%)",
                markersize=7)
        ax.fill_between(x, dd, 0, color=color, alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Drawdown over time (меньше = лучше)")
    ax.set_ylabel("Drawdown, %")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3, linestyle="--")

    # 4: Hit rate + Win/loss size
    ax = axes[1, 1]
    e3b_pos = [r for r in e3b if r > 0]
    e3b_neg = [r for r in e3b if r < 0]
    slv_pos = [r for r in slv if r > 0]
    slv_neg = [r for r in slv if r < 0]
    aqr_pos = [r for r in aqmix if r is not None and r > 0]
    aqr_neg = [r for r in aqmix if r is not None and r < 0]

    metrics = ["Avg win year", "Avg loss year"]
    e3b_avg = [np.mean(e3b_pos), np.mean(e3b_neg)]
    slv_avg = [np.mean(slv_pos), np.mean(slv_neg)]
    aqr_avg = [np.mean(aqr_pos), np.mean(aqr_neg)]

    x_pos = np.arange(2)
    w = 0.27
    ax.bar(x_pos - w, e3b_avg, w, color=COLORS["e3b"], label="E3b ★")
    ax.bar(x_pos, slv_avg, w, color=COLORS["slv"], label="SLV")
    ax.bar(x_pos + w, aqr_avg, w, color=COLORS["aqr"], label="AQR")
    for i, vals in enumerate([e3b_avg, slv_avg, aqr_avg]):
        offset = (i - 1) * w
        for j, v in enumerate(vals):
            ax.text(j + offset, v + (1 if v >= 0 else -1),
                    f"{v:+.1f}%", ha="center", fontsize=10,
                    va="bottom" if v >= 0 else "top")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Размер типичного выигрышного / убыточного года")
    ax.set_ylabel("Доходность, %")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y", linestyle="--")

    fig.suptitle("Финальный dashboard: E3b vs SLV B&H vs AQR Managed Futures (2018-2025)",
                 fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout()
    path = FIGURES_DIR / "17_final_dashboard.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    print("Generating honest metric comparisons...")
    plot_calmar_comparison()
    plot_metrics_radar_extended()
    plot_year_winner_heatmap()
    plot_returns_distribution()
    plot_final_dashboard()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
