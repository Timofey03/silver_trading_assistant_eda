"""Полный набор визуализаций для дипломной главы 4 + защиты.

Генерирует 6 ключевых графиков в data/multi_asset/figures/:
1. equity_curves.png       — 6 экспериментов на одной картинке
2. sharpe_progression.png  — bar chart прогресса
3. metrics_radar.png       — radar chart (5 метрик × 4 модели)
4. yoy_comparison.png      — год-по-году E3b vs V25 WF
5. drawdown_underwater.png — underwater drawdown plot
6. feature_importance.png  — heatmap топ фичей E3b
+ е3b_vs_v25.png           — фокусированное сравнение для defense
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Стиль
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})
sns.set_palette("colorblind")

OUT_ROOT = REPO_ROOT / "baseline_outputs_multiasset"
FIGURES_DIR = REPO_ROOT / "data" / "multi_asset" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Цветовая палитра экспериментов
EXPERIMENT_COLORS = {
    "e1_baseline":         "#7F7F7F",   # серый — baseline
    "e2_cross_asset":      "#D62728",   # красный — провал
    "e2b_feature_selected": "#FF7F0E",  # оранжевый
    "e3a_macro":           "#9467BD",   # фиолетовый — снова провал
    "e3b_adaptive":        "#2CA02C",   # ЗЕЛЁНЫЙ — WINNER
    "e4_stacking":         "#1F77B4",   # синий
    "v25_forward":         "#E377C2",   # розовый — outlier
    "v25_walkforward":     "#8C564B",   # коричневый
}

EXPERIMENT_LABELS = {
    "e1_baseline":          "E1: baseline (silver-only)",
    "e2_cross_asset":       "E2: naive cross-asset",
    "e2b_feature_selected": "E2b: + feature selection",
    "e3a_macro":            "E3a: + macro features",
    "e3b_adaptive":         "E3b: + adaptive barriers ★",
    "e4_stacking":          "E4: stacking ensemble",
    "v25_forward":          "V25 forward (Streamlit)",
    "v25_walkforward":      "V25 walk-forward 8y",
}


def load_metrics(name: str) -> dict:
    p = OUT_ROOT / name / "metrics.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    return {}


def load_trades(name: str) -> pd.DataFrame:
    if name == "v25_walkforward":
        p = REPO_ROOT / "baseline_outputs_walkforward" / "trades_all.csv"
    elif name == "v25_forward":
        p = REPO_ROOT / "baseline_outputs_v25" / "v25_forward_trades.csv"
    else:
        p = OUT_ROOT / name / "trades.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df


def equity_from_trades(trades: pd.DataFrame) -> pd.Series:
    """Накопительный equity на каждую дату выхода сделки."""
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades.sort_values("exit_date")
    cum = (1 + t["net_return"]).cumprod()
    cum.index = t["exit_date"]
    # Начальная точка
    start = t["entry_date"].min()
    cum_with_start = pd.concat([pd.Series([1.0], index=[start]), cum])
    return cum_with_start.sort_index()


# =============================================================================
# 1. EQUITY CURVES
# =============================================================================

def plot_equity_curves():
    experiments = ["e1_baseline", "e2_cross_asset", "e2b_feature_selected",
                   "e3a_macro", "e3b_adaptive", "e4_stacking", "v25_walkforward"]

    fig, ax = plt.subplots(figsize=(14, 7))
    for exp in experiments:
        trades = load_trades(exp)
        if trades.empty:
            continue
        eq = equity_from_trades(trades)
        is_winner = (exp == "e3b_adaptive")
        ax.plot(eq.index, (eq.values - 1) * 100,
                color=EXPERIMENT_COLORS[exp],
                linewidth=2.8 if is_winner else 1.8,
                alpha=1.0 if is_winner else 0.75,
                label=EXPERIMENT_LABELS[exp],
                zorder=10 if is_winner else 5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_title("Накопленная доходность всех экспериментов (walk-forward 2014–2025)",
                 pad=15)
    ax.set_ylabel("Накоплено, %")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    path = FIGURES_DIR / "01_equity_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 2. SHARPE PROGRESSION
# =============================================================================

def plot_sharpe_progression():
    order = ["e1_baseline", "e2_cross_asset", "e2b_feature_selected",
             "e3a_macro", "e3b_adaptive", "e4_stacking"]
    sharpes, labels, colors = [], [], []
    for exp in order:
        m = load_metrics(exp)
        if m:
            sharpes.append(m.get("sharpe", 0))
            labels.append(EXPERIMENT_LABELS[exp].split(":")[0])
            colors.append(EXPERIMENT_COLORS[exp])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(range(len(sharpes)), sharpes, color=colors,
                  edgecolor="white", linewidth=2)
    # Highlight winner
    winner_idx = order.index("e3b_adaptive")
    if winner_idx < len(bars):
        bars[winner_idx].set_edgecolor("#1B5E20")
        bars[winner_idx].set_linewidth(3)

    for i, (b, v) in enumerate(zip(bars, sharpes)):
        y = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2,
                y + (0.02 if y >= 0 else -0.04),
                f"{v:.3f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontweight="bold", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=11)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title("Прогрессия Sharpe Ratio по экспериментам", pad=15)
    ax.set_ylim(min(sharpes) - 0.15, max(sharpes) + 0.15)
    ax.grid(alpha=0.3, axis="y", linestyle="--")

    # Annotate winner
    if winner_idx < len(bars):
        b = bars[winner_idx]
        ax.annotate("WINNER ★",
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(b.get_x() + b.get_width() / 2, b.get_height() + 0.18),
                    ha="center", fontsize=11, color="#1B5E20", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.5))

    fig.tight_layout()
    path = FIGURES_DIR / "02_sharpe_progression.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 3. METRICS RADAR
# =============================================================================

def plot_metrics_radar():
    """Radar chart — 5 ключевых метрик × top-4 эксперимента."""
    metrics_keys = ["sharpe", "annual_return", "win_rate", "profit_factor", "max_dd"]
    metric_labels = ["Sharpe", "Annual %", "Win Rate", "Profit Factor", "Max DD (inverse)"]

    experiments = ["e1_baseline", "e2b_feature_selected", "e3b_adaptive", "v25_walkforward"]
    data = {}
    for exp in experiments:
        m = load_metrics(exp) if not exp.startswith("v25") else None
        if exp == "v25_walkforward":
            from app.multi_asset.metrics import compute_all_metrics
            t = load_trades("v25_walkforward")
            m = compute_all_metrics(t, n_trials=1)
        if m:
            # Нормализация в 0..1
            raw = {
                "sharpe": (m.get("sharpe", 0) + 1) / 3,  # -1..2 → 0..1
                "annual_return": min(max(m.get("annual_return", 0) * 10, 0), 1),
                "win_rate": m.get("win_rate", 0),
                "profit_factor": min(m.get("profit_factor", 0) / 3, 1),
                "max_dd": min(max(1 - abs(m.get("max_dd", 0)) * 2.5, 0), 1),
            }
            data[exp] = [raw[k] for k in metrics_keys]

    angles = np.linspace(0, 2 * np.pi, len(metrics_keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for exp, values in data.items():
        v = values + values[:1]
        is_winner = (exp == "e3b_adaptive")
        ax.plot(angles, v, color=EXPERIMENT_COLORS[exp],
                linewidth=3 if is_winner else 1.8,
                label=EXPERIMENT_LABELS[exp],
                alpha=1.0 if is_winner else 0.7)
        ax.fill(angles, v, color=EXPERIMENT_COLORS[exp],
                alpha=0.18 if is_winner else 0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], color="gray", fontsize=9)
    ax.set_title("Многомерное сравнение моделей (выше = лучше)\n",
                 pad=20, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    path = FIGURES_DIR / "03_metrics_radar.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 4. YEAR-OVER-YEAR COMPARISON
# =============================================================================

def plot_yoy_comparison():
    e3b = load_trades("e3b_adaptive")
    v25 = load_trades("v25_walkforward")

    years = sorted(set(e3b["entry_date"].dt.year.tolist() + v25["entry_date"].dt.year.tolist()))

    e3b_returns, v25_returns = [], []
    for y in years:
        e_chunk = e3b[e3b["entry_date"].dt.year == y]
        v_chunk = v25[v25["entry_date"].dt.year == y]
        e3b_returns.append(((1 + e_chunk["net_return"]).prod() - 1) * 100 if len(e_chunk) else 0)
        v25_returns.append(((1 + v_chunk["net_return"]).prod() - 1) * 100 if len(v_chunk) else 0)

    x = np.arange(len(years))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 7))
    b1 = ax.bar(x - width / 2, e3b_returns, width,
                color=EXPERIMENT_COLORS["e3b_adaptive"],
                label="E3b (новая модель) ★",
                edgecolor="white", linewidth=1.5)
    b2 = ax.bar(x + width / 2, v25_returns, width,
                color=EXPERIMENT_COLORS["v25_walkforward"],
                alpha=0.7,
                label="V25 walk-forward (текущий production)",
                edgecolor="white", linewidth=1.5)

    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if abs(h) < 0.5:
                continue
            ax.text(b.get_x() + b.get_width() / 2,
                    h + (1 if h >= 0 else -1.5),
                    f"{h:+.0f}%",
                    ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=9.5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_title("Год-по-году: E3b vs V25 walk-forward", pad=15)
    ax.set_ylabel("Доходность за год, %")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3, axis="y", linestyle="--")
    fig.tight_layout()
    path = FIGURES_DIR / "04_yoy_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 5. DRAWDOWN UNDERWATER
# =============================================================================

def plot_drawdown_underwater():
    experiments = ["e1_baseline", "e2b_feature_selected", "e3b_adaptive", "v25_walkforward"]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    for exp in experiments:
        trades = load_trades(exp)
        if trades.empty:
            continue
        eq = equity_from_trades(trades)
        peak = eq.cummax()
        dd = (eq / peak - 1) * 100
        is_winner = (exp == "e3b_adaptive")
        ax.plot(dd.index, dd.values,
                color=EXPERIMENT_COLORS[exp],
                linewidth=2.5 if is_winner else 1.5,
                alpha=1.0 if is_winner else 0.7,
                label=EXPERIMENT_LABELS[exp])
        ax.fill_between(dd.index, dd.values, 0,
                        color=EXPERIMENT_COLORS[exp],
                        alpha=0.15 if is_winner else 0.05)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Просадки (drawdown) — глубже = хуже", pad=15)
    ax.set_ylabel("Просадка от пика, %")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    path = FIGURES_DIR / "05_drawdown_underwater.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 6. FEATURE IMPORTANCE HEATMAP
# =============================================================================

def plot_feature_importance():
    fi_path = OUT_ROOT / "e3b_adaptive" / "feature_importance.csv"
    if not fi_path.exists():
        # E3b не пишет feature_importance — берём из E4 (та же FS логика)
        fi_path = OUT_ROOT / "e4_stacking" / "feature_importance.csv"
    if not fi_path.exists():
        # Fallback на E2b
        fi_path = OUT_ROOT / "e2b_feature_selected" / "feature_importance.csv"
    if not fi_path.exists():
        print("No feature_importance file found, skipping")
        return

    fi = pd.read_csv(fi_path).head(25)
    fi = fi.sort_values("frequency")

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#2CA02C" if r["frequency"] >= 0.95 else
              "#FF7F0E" if r["frequency"] >= 0.7 else "#1F77B4"
              for _, r in fi.iterrows()]

    bars = ax.barh(fi["feature"], fi["frequency"] * 100, color=colors,
                   edgecolor="white", linewidth=1.0)
    for b, v in zip(bars, fi["frequency"]):
        ax.text(v * 100 + 1, b.get_y() + b.get_height() / 2,
                f"{v*100:.0f}%", va="center", fontsize=10)

    ax.set_xlim(0, 110)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title(f"Top-25 фичей по частоте отбора (источник: {fi_path.parent.name})",
                 pad=15)
    ax.set_xlabel("% фолдов где фича была выбрана top-30")
    ax.grid(alpha=0.3, axis="x", linestyle="--")
    fig.tight_layout()
    path = FIGURES_DIR / "06_feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 7. E3b vs V25 — fokuсный график для defence
# =============================================================================

def plot_e3b_vs_v25_focused():
    e3b = load_trades("e3b_adaptive")
    v25_wf = load_trades("v25_walkforward")
    v25_fwd = load_trades("v25_forward")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Subplot 1: Equity curves
    ax = axes[0, 0]
    for exp, trades in [("e3b_adaptive", e3b),
                        ("v25_walkforward", v25_wf),
                        ("v25_forward", v25_fwd)]:
        eq = equity_from_trades(trades)
        ax.plot(eq.index, (eq.values - 1) * 100,
                color=EXPERIMENT_COLORS[exp],
                linewidth=2.5 if exp == "e3b_adaptive" else 2,
                label=EXPERIMENT_LABELS[exp])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Накопленная доходность")
    ax.set_ylabel("%")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

    # Subplot 2: bar chart key metrics
    ax = axes[0, 1]
    metrics_e3b = load_metrics("e3b_adaptive")
    from app.multi_asset.metrics import compute_all_metrics
    metrics_v25_wf = compute_all_metrics(v25_wf, n_trials=1)
    metrics_v25_fwd = compute_all_metrics(v25_fwd, n_trials=1)

    metric_names = ["Sharpe", "Annual %", "Win Rate %", "Profit Factor"]
    e3b_vals = [metrics_e3b.get("sharpe", 0),
                metrics_e3b.get("annual_return", 0) * 100,
                metrics_e3b.get("win_rate", 0) * 100,
                metrics_e3b.get("profit_factor", 0)]
    v25_wf_vals = [metrics_v25_wf.get("sharpe", 0),
                   metrics_v25_wf.get("annual_return", 0) * 100,
                   metrics_v25_wf.get("win_rate", 0) * 100,
                   metrics_v25_wf.get("profit_factor", 0)]
    v25_fwd_vals = [metrics_v25_fwd.get("sharpe", 0),
                    metrics_v25_fwd.get("annual_return", 0) * 100,
                    metrics_v25_fwd.get("win_rate", 0) * 100,
                    metrics_v25_fwd.get("profit_factor", 0)]

    x = np.arange(len(metric_names))
    w = 0.27
    ax.bar(x - w, e3b_vals, w, color=EXPERIMENT_COLORS["e3b_adaptive"], label="E3b ★")
    ax.bar(x, v25_wf_vals, w, color=EXPERIMENT_COLORS["v25_walkforward"], label="V25 WF")
    ax.bar(x + w, v25_fwd_vals, w, color=EXPERIMENT_COLORS["v25_forward"], label="V25 fwd")
    for i, (e, w_, f) in enumerate(zip(e3b_vals, v25_wf_vals, v25_fwd_vals)):
        for j, v in enumerate([e, w_, f]):
            offset = (j - 1) * w
            ax.text(i + offset, v + (max(e, w_, f, 0.5) * 0.02 if v > 0 else -1),
                    f"{v:.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_title("Ключевые метрики (V25 forward — bull-market sample, не репрезентативно)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y", linestyle="--")

    # Subplot 3: drawdown comparison
    ax = axes[1, 0]
    for exp, trades in [("e3b_adaptive", e3b), ("v25_walkforward", v25_wf)]:
        eq = equity_from_trades(trades)
        peak = eq.cummax()
        dd = (eq / peak - 1) * 100
        ax.plot(dd.index, dd.values,
                color=EXPERIMENT_COLORS[exp],
                linewidth=2.5 if exp == "e3b_adaptive" else 1.8,
                label=EXPERIMENT_LABELS[exp])
        ax.fill_between(dd.index, dd.values, 0,
                        color=EXPERIMENT_COLORS[exp], alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Просадки (E3b vs V25 walk-forward)")
    ax.set_ylabel("Drawdown %")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

    # Subplot 4: year-by-year (только overlap 2018-2025)
    ax = axes[1, 1]
    years = sorted({y for y in e3b["entry_date"].dt.year if y >= 2018 and y <= 2025} |
                   {y for y in v25_wf["entry_date"].dt.year if y >= 2018 and y <= 2025})
    e_yoy, v_yoy = [], []
    for y in years:
        e = e3b[e3b["entry_date"].dt.year == y]
        v = v25_wf[v25_wf["entry_date"].dt.year == y]
        e_yoy.append(((1 + e["net_return"]).prod() - 1) * 100 if len(e) else 0)
        v_yoy.append(((1 + v["net_return"]).prod() - 1) * 100 if len(v) else 0)

    x = np.arange(len(years))
    w = 0.4
    ax.bar(x - w / 2, e_yoy, w, color=EXPERIMENT_COLORS["e3b_adaptive"], label="E3b ★")
    ax.bar(x + w / 2, v_yoy, w, color=EXPERIMENT_COLORS["v25_walkforward"], alpha=0.75,
           label="V25 walk-forward")
    for i, (e, v) in enumerate(zip(e_yoy, v_yoy)):
        if abs(e) > 0.5:
            ax.text(i - w / 2, e + (1 if e >= 0 else -1.5), f"{e:+.0f}%",
                    ha="center", va="bottom" if e >= 0 else "top", fontsize=8)
        if abs(v) > 0.5:
            ax.text(i + w / 2, v + (1 if v >= 0 else -1.5), f"{v:+.0f}%",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=10)
    ax.set_title("Год-по-году в overlap-периоде (2018-2025)")
    ax.set_ylabel("% за год")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y", linestyle="--")

    fig.suptitle("E3b (новая) vs V25 (production) — полная картина для защиты",
                 fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout()
    path = FIGURES_DIR / "07_e3b_vs_v25_focused.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    print("Generating diploma visualizations...")
    plot_equity_curves()
    plot_sharpe_progression()
    plot_metrics_radar()
    plot_yoy_comparison()
    plot_drawdown_underwater()
    plot_feature_importance()
    plot_e3b_vs_v25_focused()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
