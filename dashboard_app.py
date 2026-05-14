"""
Silver Trading Assistant — Streamlit Dashboard

Запуск:
  streamlit run dashboard_app.py

Автоматически показывает наиболее свежую версию: v19 > v18 > v17 > v16 > v15 > v14.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Настройки страницы
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Silver Trading Assistant",
    page_icon="🥈",
    layout="wide",
)

V22_DIR = Path("baseline_outputs_v22")
V19_DIR = Path("baseline_outputs_v19")
V18_DIR = Path("baseline_outputs_v18")
V17_DIR = Path("baseline_outputs_v17")
V16_DIR = Path("baseline_outputs_v16")
V15_DIR = Path("baseline_outputs_v15")
V14_DIR = Path("baseline_outputs_v14")
SPLITS       = ["valid", "test", "forward"]
SPLIT_LABELS = {"valid": "Valid 2023", "test": "Test 2024", "forward": "Forward 2025+"}

# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_best() -> dict:
    """Загружает наиболее свежую версию: v19 > v18 > v17 > v16 > v15 > v14."""
    if (V19_DIR / "v19_guardrails.csv").exists():
        ver_dir, ver = V19_DIR, "v19"
    elif (V18_DIR / "v18_guardrails.csv").exists():
        ver_dir, ver = V18_DIR, "v18"
    elif (V17_DIR / "v17_guardrails.csv").exists():
        ver_dir, ver = V17_DIR, "v17"
    elif (V16_DIR / "v16_guardrails.csv").exists():
        ver_dir, ver = V16_DIR, "v16"
    elif (V15_DIR / "v15_guardrails.csv").exists():
        ver_dir, ver = V15_DIR, "v15"
    else:
        ver_dir, ver = V14_DIR, "v14"

    if not ver_dir.exists():
        return {}

    data: dict = {"version": ver, "ver_dir": str(ver_dir)}

    file_map = {
        f"{ver}_classifier_metrics.csv":  "metrics",
        f"{ver}_guardrails.csv":          "guardrails",
        f"{ver}_backtest_report.csv":     "backtest",
        f"{ver}_label_distribution.csv":  "labels",
        f"{ver}_latest_signal_cards.csv": "cards",
        f"{ver}_purged_wf_cv.csv":        "wf_cv",
        f"{ver}_decisions_all.csv":       "all_decisions",
        f"{ver}_feature_importance.csv":  "feature_importance",
        f"{ver}_selected_features.csv":   "selected_features",
    }
    # Дополнительные файлы
    extras = {
        "v15_cot_raw.csv": "v15_cot_raw",
        f"{ver}_vs_v14_comparison.csv": f"{ver}_vs_v14_comparison",
        f"{ver}_vs_v15_comparison.csv": f"{ver}_vs_v15_comparison",
    }
    file_map.update(extras)

    for fname, key in file_map.items():
        path = ver_dir / fname
        if path.exists():
            try:
                data[key] = pd.read_csv(path, index_col=0, parse_dates=True) \
                    if key == "all_decisions" else pd.read_csv(path)
            except Exception:
                pass

    # Сделки по выборкам (trailing для v19, fixed иначе)
    data["trades"] = {}
    data["trades_trailing"] = {}
    data["trades_fixed"]    = {}
    for s in SPLITS:
        # Trailing-stop сделки (v19)
        path_trail = ver_dir / f"{s}_trades_trailing_{ver}.csv"
        if path_trail.exists():
            try:
                data["trades_trailing"][s] = pd.read_csv(
                    path_trail, parse_dates=["signal_date", "entry_date", "exit_date"]
                )
            except Exception:
                pass
        # Fixed-horizon сделки (v19 comparison / v18 и ранее — основные)
        path_fixed = ver_dir / f"{s}_trades_fixed_{ver}.csv"
        if path_fixed.exists():
            try:
                data["trades_fixed"][s] = pd.read_csv(
                    path_fixed, parse_dates=["signal_date", "entry_date", "exit_date"]
                )
            except Exception:
                pass
        # Основные сделки (для обратной совместимости с v14-v18)
        path = ver_dir / f"{s}_trades_{ver}.csv"
        if path.exists():
            try:
                data["trades"][s] = pd.read_csv(
                    path, parse_dates=["signal_date", "entry_date", "exit_date"]
                )
            except Exception:
                pass
        # v19: основные сделки = trailing
        if ver == "v19" and s in data["trades_trailing"]:
            data["trades"][s] = data["trades_trailing"][s]

    # Trailing vs fixed сравнение (v19)
    trail_vs_fixed_path = ver_dir / f"{ver}_trailing_vs_fixed.csv"
    if trail_vs_fixed_path.exists():
        try:
            data["trailing_vs_fixed"] = pd.read_csv(trail_vs_fixed_path)
        except Exception:
            pass

    # Trailing-stop бэктест-отчёт (v19)
    trail_bt_path = ver_dir / f"{ver}_backtest_trailing.csv"
    if trail_bt_path.exists():
        try:
            data["backtest_trailing"] = pd.read_csv(trail_bt_path)
        except Exception:
            pass

    # Fixed бэктест-отчёт (v19 fixed сравнение)
    fixed_bt_path = ver_dir / f"{ver}_backtest_fixed.csv"
    if fixed_bt_path.exists():
        try:
            data["backtest_fixed"] = pd.read_csv(fixed_bt_path)
        except Exception:
            pass

    # Политика
    policy_path = ver_dir / f"{ver}_policy.json"
    if policy_path.exists():
        with open(policy_path, encoding="utf-8") as f:
            data["policy"] = json.load(f)

    # Guardrails предыдущих версий для сравнения
    for v, vdir in [("v14", V14_DIR), ("v15", V15_DIR), ("v16", V16_DIR),
                    ("v17", V17_DIR), ("v18", V18_DIR)]:
        p = vdir / f"{v}_guardrails.csv"
        if p.exists():
            data[f"{v}_guardrails"] = pd.read_csv(p)

    # v13 guardrails
    v13_path = Path("baseline_outputs_v13/v13_guardrails.csv")
    if v13_path.exists():
        data["v13_guardrails"] = pd.read_csv(v13_path)

    # v19: также загружаем backtest_trailing в "backtest" для совместимости вкладки
    if ver == "v19" and "backtest_trailing" in data:
        data["backtest"] = data["backtest_trailing"]

    return data


@st.cache_data(ttl=3600)
def load_v22() -> dict:
    """Загружает результаты v22 (Risk-Aware) если доступны."""
    if not V22_DIR.exists():
        return {}

    d: dict = {}
    for fname, key in [
        ("v22_comparison_all.csv",  "comparison"),
        ("v22_risk_metrics.csv",    "risk_metrics"),
        ("v22_pnl_summary.csv",     "pnl_summary"),
        ("v22_label_distribution.csv", "labels"),
        ("v22_feature_importance.csv", "feature_importance"),
        ("v22_latest_signal_cards.csv", "cards"),
    ]:
        p = V22_DIR / fname
        if p.exists():
            try:
                d[key] = pd.read_csv(p)
            except Exception:
                pass

    p = V22_DIR / "v22_policy.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            d["policy"] = json.load(f)

    # Сделки по вариантам и выборкам
    variants = ["v22_base", "v22_atr", "v22_kelly", "v22_mh", "v22_wf", "v22_all"]
    d["trades"] = {}
    for var in variants:
        d["trades"][var] = {}
        for s in SPLITS:
            tp = V22_DIR / f"{var}_{s}_trades.csv"
            if tp.exists():
                try:
                    df_t = pd.read_csv(
                        tp, parse_dates=["entry_date", "exit_date"]
                    )
                    d["trades"][var][s] = df_t
                except Exception:
                    pass

    return d


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def color_warning(val: str) -> str:
    s = str(val)
    if s == "OK":                             return "background-color: #d4edda"
    if "ci_lower_not_above_base" in s:        return "background-color: #fff3cd"
    if "negative_lift" in s or "no_signal" in s: return "background-color: #f8d7da"
    return ""


def pct_str(x, d: int = 2) -> str:
    try:
        return f"{float(x)*100:.{d}f}%"
    except Exception:
        return str(x)


def fmt_num(x, d: int = 3) -> str:
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def is_binary(data: dict) -> bool:
    """True если текущая версия использует бинарную классификацию (v16+)."""
    return data.get("version", "v14") >= "v16"


def has_macro(data: dict) -> bool:
    """True если текущая версия использует yfinance macro-признаки (v17+)."""
    return data.get("version", "v14") >= "v17"


def has_adaptive(data: dict) -> bool:
    """True если версия использует expanding window + adaptive weight (v18+)."""
    return data.get("version", "v14") >= "v18"


def has_trailing(data: dict) -> bool:
    """True если версия использует trailing stop бэктест (v19+)."""
    return data.get("version", "v14") >= "v19"


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

def plot_equity_curves(data: dict, split: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})
    ax, ax2 = axes

    decisions = data.get("all_decisions")
    trades    = data.get("trades", {}).get(split, pd.DataFrame())

    if decisions is None or decisions.empty:
        ax.text(0.5, 0.5, "Нет данных", transform=ax.transAxes, ha="center")
        return fig

    d = decisions[decisions["split"] == split].sort_index()
    if d.empty:
        ax.text(0.5, 0.5, "Нет данных для выборки", transform=ax.transAxes, ha="center")
        return fig

    start_price = d["silver_close"].iloc[0]
    ax.plot(d.index, d["silver_close"] / start_price,
            color="#888", lw=1.2, label="Silver (норм.)", alpha=0.6)

    eq = pd.Series(1.0, index=d.index)
    if not trades.empty:
        equity = 1.0
        for _, tr in trades.sort_values("entry_date").iterrows():
            mask = d.index >= tr["exit_date"]
            if mask.any():
                eq[mask] = equity * (1 + tr["net_return"])
                equity    = eq[mask].iloc[0]
    ax.plot(d.index, eq, color="#1f77b4", lw=2, label="Стратегия (equity)")

    buy_days = d[d["signal"] == "BUY"]
    if not buy_days.empty:
        ax.scatter(buy_days.index, buy_days["silver_close"] / start_price,
                   marker="^", color="green", s=80, zorder=5, label="BUY сигнал")

    ax.set_title(f"{SPLIT_LABELS.get(split, split)} — Equity curve vs Silver", fontsize=13)
    ax.set_ylabel("Относительная доходность")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.3)

    if "p_up" in d.columns:
        ax2.fill_between(d.index, d["p_up"], alpha=0.4, color="#ff7f0e", label="P(UP)")
        if "policy" in data:
            thr = data["policy"].get("up_threshold", 0.50)
            ax2.axhline(thr, color="red", ls="--", lw=1, label=f"Порог {thr:.2f}")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("P(UP)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.2)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    return fig


def plot_wf_cv(wf_df: pd.DataFrame, binary: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    mean_ba = wf_df["balanced_acc"].mean()
    colors  = ["#2ca02c" if v > 0.50 else "#d62728" for v in wf_df["balanced_acc"]]
    ax.bar(range(len(wf_df)), wf_df["balanced_acc"], color=colors, alpha=0.8)
    ax.axhline(0.5,     color="red",   ls="--", lw=1.5, label="Baseline случайного (0.50)")
    ax.axhline(mean_ba, color="green", ls="--", lw=1.5,
               label=f"Среднее: {mean_ba:.3f}")
    if not binary:
        ax.axhline(1/3, color="gray", ls=":", lw=1, label="3-класс. baseline (0.33)")
    ax.set_xticks(range(len(wf_df)))
    ax.set_xticklabels([f"F{i}" for i in range(len(wf_df))], rotation=45, fontsize=8)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Purged Walk-Forward CV — balanced accuracy по фолдам")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(fi: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    if "importance" not in fi.columns:
        # Только список без значений (selected_features)
        fi = fi.head(top_n).copy()
        fi["importance"] = range(len(fi), 0, -1)
    top = fi.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in top["importance"]]
    ax.barh(top["feature"], top["importance"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Permutation Importance (mean, ROC AUC)")
    ax.set_title(f"Топ-{top_n} признаков по важности")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar(data: dict) -> None:
    ver = data.get("version", "v14")
    st.sidebar.title(f"Silver Assistant {ver.upper()}")
    st.sidebar.markdown("---")

    if "policy" in data:
        p = data["policy"]
        st.sidebar.subheader("Параметры политики")
        keep_keys = {"up_threshold", "margin_threshold", "down_cap", "cooldown",
                     "horizon_days", "tb_mode", "train_window", "regime_models",
                     "top_features_n", "not_up_weight"}
        display = {k: v for k, v in p.items() if k in keep_keys}
        st.sidebar.json(display)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Запуск pipeline:**\n"
        "```\n# v14 (базовый)\npython silver_assistant_v14_main.py\n\n"
        "# v15 (COT + режимная)\npython silver_assistant_v15_regime_cot.py\n\n"
        "# v16 (бинарная + регуляризация)\npython silver_assistant_v16_binary.py\n\n"
        "# v17 (+ макро yfinance)\npython silver_assistant_v17_fred.py\n\n"
        "# v18 (expanding window)\npython silver_assistant_v18_adaptive.py\n\n"
        "# v19 (trailing stop)\npython silver_assistant_v19_trailing.py\n```"
    )
    if has_trailing(data):
        policy = data.get("policy", {})
        st.sidebar.markdown("---")
        st.sidebar.subheader("Trailing Stop (v19)")
        trail_pct  = policy.get("trail_pct", "?")
        max_hold   = policy.get("max_hold_days", "?")
        st.sidebar.markdown(
            f"**trail_pct**: {trail_pct:.1%}" if isinstance(trail_pct, float) else f"**trail_pct**: {trail_pct}"
        )
        st.sidebar.markdown(f"**max_hold**: {max_hold} дней")
        adaptive_wts = policy.get("not_up_weight_adaptive", {})
        for sp, wt in adaptive_wts.items():
            st.sidebar.markdown(f"**{sp}**: NOT_UP_w = {wt:.2f}")
        st.sidebar.markdown(f"**Halflife**: {policy.get('halflife_years', '?')} лет")
    elif has_adaptive(data):
        policy = data.get("policy", {})
        adaptive_wts = policy.get("not_up_weight_adaptive", {})
        st.sidebar.markdown("---")
        st.sidebar.subheader("Adaptive window (v18)")
        for sp, wt in adaptive_wts.items():
            st.sidebar.markdown(f"**{sp}**: NOT_UP_w = {wt:.2f}")
        st.sidebar.markdown(f"**Halflife**: {policy.get('halflife_years', '?')} лет")
    elif has_macro(data):
        policy = data.get("policy", {})
        macro_in_top = policy.get("macro_in_top_n", [])
        st.sidebar.markdown("---")
        st.sidebar.subheader("Macro-признаки (v17)")
        st.sidebar.markdown(
            f"**Тикеры:** ^TNX, ^IRX, TIP, RINF, HYG\n\n"
            f"**В top-{policy.get('top_features_n', '?')}:** {len(macro_in_top)} шт.\n\n"
            + (f"*{', '.join(macro_in_top[:5])}{'...' if len(macro_in_top) > 5 else ''}*" if macro_in_top else "*(none)*")
        )


# ---------------------------------------------------------------------------
# Вкладки
# ---------------------------------------------------------------------------

def tab_overview(data: dict) -> None:
    ver = data.get("version", "v14")
    st.header(f"Обзор результатов {ver.upper()}")

    if "cards" in data and not data["cards"].empty:
        st.subheader("Последние сигналы")
        cols = st.columns(3)
        for i, s in enumerate(SPLITS):
            row = data["cards"][data["cards"]["split"] == s]
            if row.empty:
                continue
            r   = row.iloc[0]
            sig = r.get("signal", "HOLD")
            color = "#28a745" if sig == "BUY" else "#6c757d"
            trend_col = "trend_regime" if "trend_regime" in r.index else "regime"
            with cols[i]:
                st.markdown(
                    f"""<div style="border:2px solid {color}; border-radius:8px; padding:12px; text-align:center;">
                    <b>{SPLIT_LABELS.get(s, s)}</b><br>
                    <span style="font-size:1.4em; color:{color};"><b>{sig}</b></span><br>
                    <small>P(UP): {fmt_num(r.get('p_up',''), 3)}</small><br>
                    <small>Silver: {fmt_num(r.get('silver_close',''), 2)}</small><br>
                    <small>Тренд: {r.get(trend_col,'')}</small>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    if "guardrails" in data and not data["guardrails"].empty:
        st.subheader("Guardrails (статистический edge)")
        gr = data["guardrails"].copy()
        st.dataframe(
            gr.style.map(color_warning, subset=["warning"]),
            use_container_width=True,
        )
        st.caption(
            "**OK** = Wilson 95% CI нижняя граница > базовый уровень = доказанный edge. "
            "**ci_lower_not_above_base** = нет доказанного edge — торговать нельзя."
        )


def tab_metrics(data: dict) -> None:
    ver    = data.get("version", "v14")
    binary = is_binary(data)
    st.header("Метрики классификатора")

    if "metrics" not in data or data["metrics"].empty:
        st.warning(f"Файл {ver}_classifier_metrics.csv не найден. Запустите pipeline.")
        return

    m = data["metrics"]
    st.dataframe(m, use_container_width=True)

    st.markdown("---")
    st.subheader("Balanced Accuracy по выборкам")
    fig, ax = plt.subplots(figsize=(8, 4))
    splits = m["split"].tolist()
    values = m["balanced_accuracy"].tolist()
    colors = ["#2ca02c" if v > 0.50 else "#d62728" for v in values]
    ax.bar(splits, values, color=colors, alpha=0.8)
    ax.axhline(0.5, color="black", ls="--", lw=1.5, label="Случайный (0.50)")
    if not binary:
        ax.axhline(1/3, color="gray", ls=":", lw=1, label="3-класс. случайный (0.33)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced Accuracy")
    baseline_note = "0.50 (бинарный)" if binary else "0.50 / 0.33 (3-класс.)"
    ax.set_title(f"Balanced Accuracy: train / valid / test / forward  [baseline={baseline_note}]")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.caption(
        "Красный = хуже случайного выбора. Зелёный = лучше случайного. "
        "Разрыв train→valid/test — признак переобучения."
        + (" Для v16 (бинарная задача) baseline = 0.50." if binary else "")
    )

    # Дополнительные метрики для v16: AUC и Brier
    if binary and "auc" in m.columns:
        st.markdown("---")
        st.subheader("ROC AUC и Brier Score (бинарная задача)")
        fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(splits, m["auc"], color=["#2ca02c" if v > 0.55 else "#d62728" for v in m["auc"]], alpha=0.8)
        axes[0].axhline(0.5, color="black", ls="--", lw=1, label="Baseline 0.50")
        axes[0].set_ylim(0, 1); axes[0].set_title("ROC AUC (UP vs NOT_UP)")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        if "brier" in m.columns:
            axes[1].bar(splits, m["brier"], color="#ff7f0e", alpha=0.8)
            axes[1].axhline(0.25, color="black", ls="--", lw=1, label="Baseline 0.25 (no-skill)")
            axes[1].set_title("Brier Score (ниже = лучше)")
            axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    if "wf_cv" in data and not data["wf_cv"].empty:
        st.markdown("---")
        st.subheader("Purged Walk-Forward CV")
        wf = data["wf_cv"]
        st.pyplot(plot_wf_cv(wf, binary=binary))
        embargo = 15 if data.get("version", "v14") >= "v15" else 5
        st.caption(
            f"Каждый фолд: 3 года обучения, 6 месяцев теста, embargo={embargo} дней. "
            + ("Baseline = 0.50 (бинарная задача)." if binary else "Baseline = 0.33 (3 класса).")
        )


def tab_backtest(data: dict) -> None:
    st.header("Бэктест vs Buy-and-Hold")

    # ---- v19: Trailing Stop — основной результат ----
    if has_trailing(data):
        policy  = data.get("policy", {})
        trail_p = policy.get("trail_pct", "?")
        max_h   = policy.get("max_hold_days", "?")
        tp_str  = f"{trail_p:.1%}" if isinstance(trail_p, float) else str(trail_p)
        st.info(
            f"**v19 — Trailing Stop**: trail_pct={tp_str}, max_hold={max_h}d  "
            f"(параметры подобраны на valid по критерию Шарпа)"
        )

        # Trailing бэктест
        if "backtest_trailing" in data and not data["backtest_trailing"].empty:
            bt_t = data["backtest_trailing"].copy()
            st.subheader("Trailing Stop — сводка")
            disp_t = bt_t.copy()
            for col in ["sum_net_return", "buy_and_hold", "vs_bnh", "avg_peak_gain"]:
                if col in disp_t.columns:
                    disp_t[col] = disp_t[col].apply(lambda x: pct_str(x) if pd.notna(x) else "-")
            for col in ["win_rate", "trail_exit_pct"]:
                if col in disp_t.columns:
                    disp_t[col] = disp_t[col].apply(lambda x: pct_str(x, 1) if pd.notna(x) else "-")
            st.dataframe(disp_t, use_container_width=True)
            st.caption(
                "**trail_exits** = позиций закрытых трейлером (цена упала на trail_pct от пика). "
                "**max_hold_exits** = позиций удержанных до max_hold дней. "
                "**avg_peak_gain** = средний прирост от входа до пика."
            )

        # Сравнение trailing vs fixed
        if "trailing_vs_fixed" in data and not data["trailing_vs_fixed"].empty:
            st.markdown("---")
            st.subheader("Trailing vs Fixed-horizon (15d) — сравнение")
            comp = data["trailing_vs_fixed"].copy()
            for col in ["fixed_sum_net", "trail_sum_net", "avg_peak_gain"]:
                if col in comp.columns:
                    comp[col] = comp[col].apply(lambda x: pct_str(x) if pd.notna(x) else "-")
            for col in ["fixed_win_rate", "trail_win_rate", "trail_exit_pct"]:
                if col in comp.columns:
                    comp[col] = comp[col].apply(lambda x: pct_str(x, 1) if pd.notna(x) else "-")
            st.dataframe(comp, use_container_width=True)
            st.caption(
                "**trail_avg_hold** — среднее время в позиции (trailing). "
                "**fixed_avg_hold** = 15d (фиксированный горизонт). "
                "Trailing позволяет удерживать тренд дольше."
            )
        st.markdown("---")

    # ---- Стандартный бэктест (v14-v18 или fixed v19) ----
    bt_key = "backtest_fixed" if (has_trailing(data) and "backtest_fixed" in data) else "backtest"
    if bt_key not in data or data[bt_key] is None or (hasattr(data[bt_key], 'empty') and data[bt_key].empty):
        if not has_trailing(data):
            st.warning("backtest_report.csv не найден.")
            return
    else:
        bt = data[bt_key].copy()
        label = "Fixed Horizon (15d) — для сравнения" if has_trailing(data) else "Бэктест"
        st.subheader(label)
        display = bt.copy()
        for col in ["sum_net_return", "avg_net_return", "buy_and_hold", "vs_bnh"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: pct_str(x) if pd.notna(x) else "-")
        for col in ["win_rate", "take_profit_pct", "stop_loss_pct"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: pct_str(x, 1) if pd.notna(x) else "-")
        st.dataframe(display, use_container_width=True)
        st.caption(
            "**vs_bnh** — разница суммарного возврата стратегии и buy-and-hold за период. "
            "Отрицательное значение = пассивное удержание выгоднее."
        )

    st.markdown("---")
    st.subheader("Equity curves")
    tabs = st.tabs([SPLIT_LABELS.get(s, s) for s in SPLITS])
    for i, s in enumerate(SPLITS):
        with tabs[i]:
            st.pyplot(plot_equity_curves(data, s))
            trades = data.get("trades", {}).get(s, pd.DataFrame())
            if not trades.empty:
                st.subheader("Сделки" + (" (trailing)" if has_trailing(data) else ""))
                display_t = trades.copy()
                for col in ["gross_return", "net_return"]:
                    if col in display_t.columns:
                        display_t[col] = display_t[col].apply(pct_str)
                st.dataframe(display_t, use_container_width=True)


def tab_features(data: dict) -> None:
    ver    = data.get("version", "v14")
    binary = is_binary(data)
    st.header("Важность признаков")

    if binary:
        fi  = data.get("feature_importance")
        sel = data.get("selected_features")
        ver = data.get("version", "v16")
        if fi is None or fi.empty:
            st.warning(f"{ver}_feature_importance.csv не найден. Запустите pipeline.")
            return
        st.info(
            f"{ver} использует отбор признаков: top-{len(sel) if sel is not None else '?'} "
            f"из {len(fi)} по permutation importance (ROC AUC на valid). "
            "Неотобранные признаки не используются в финальной модели."
        )

        # v17: показываем фильтр по источнику признака
        if has_macro(data) and "source" in fi.columns:
            src_filter = st.multiselect(
                "Источник признака", fi["source"].unique().tolist(),
                default=fi["source"].unique().tolist(),
            )
            fi_show = fi[fi["source"].isin(src_filter)]
        else:
            fi_show = fi

        top_n = st.slider("Топ-N признаков", 5, min(50, len(fi_show)), 30)
        st.pyplot(plot_feature_importance(fi_show, top_n))

        if has_macro(data) and "source" in fi.columns:
            st.subheader("Вклад источников в top-30")
            sel_names = set(sel["feature"].tolist()) if sel is not None and not sel.empty else set()
            top30 = fi[fi["feature"].isin(sel_names)] if sel_names else fi.head(30)
            src_counts = top30.groupby("source")["importance"].agg(["count", "sum", "mean"]).reset_index()
            src_counts.columns = ["Источник", "Кол-во в top", "Сумм. важность", "Сред. важность"]
            st.dataframe(src_counts, use_container_width=True)

        st.subheader("Все важности")
        st.dataframe(fi_show.head(top_n), use_container_width=True)
        if sel is not None and not sel.empty:
            st.subheader("Отобранные признаки (используются в модели)")
            st.dataframe(sel, use_container_width=True)
    else:
        # v14/v15: feature_importance из файла
        fi = data.get("feature_importance")
        if fi is None or fi.empty:
            st.warning(f"Файл {ver}_feature_importance.csv не найден.")
            return
        top_n = st.slider("Топ-N признаков", 5, min(40, len(fi)), 20)
        st.pyplot(plot_feature_importance(fi, top_n))
        st.dataframe(fi.head(top_n), use_container_width=True)

    st.caption(
        "Permutation importance: насколько снижается ROC AUC при случайном перемешивании признака. "
        "Отрицательные значения = признак добавляет шум."
    )


def tab_labels(data: dict) -> None:
    binary = is_binary(data)
    st.header("Распределение меток triple-barrier")

    if "labels" not in data:
        st.warning("label_distribution.csv не найден.")
        return

    lb = data["labels"]
    st.dataframe(lb, use_container_width=True)

    fig, axes = plt.subplots(1, len(lb), figsize=(12, 4))
    if len(lb) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, lb.iterrows()):
        if binary:
            # UP / NOT_UP
            vals   = [row.get("UP", 0), row.get("NOT_UP", 0)]
            labels = ["UP", "NOT_UP"]
            colors = ["#2ca02c", "#d62728"]
        else:
            vals   = [row.get("UP", 0), row.get("NEUTRAL", 0), row.get("DOWN", 0)]
            labels = ["UP", "NEUTRAL", "DOWN"]
            colors = ["#2ca02c", "#ff7f0e", "#d62728"]
        ax.pie([v for v in vals if v > 0],
               labels=[l for l, v in zip(labels, vals) if v > 0],
               colors=[c for c, v in zip(colors, vals) if v > 0],
               autopct="%1.0f%%", startangle=90)
        ax.set_title(row["split"])
    st.pyplot(fig)

    if binary:
        st.markdown("""
**v16 (бинарная):** UP = цена достигла верхнего барьера за 15 дней.
NOT_UP = объединение DOWN + NEUTRAL (не достигла верхнего барьера).
Упрощение задачи снижает переобучение при ограниченном датасете.
""")
    else:
        st.markdown("""
**Замечание:** если доля NEUTRAL < 5% — triple-barrier вырожден.
Нормальное распределение: 30–50% NEUTRAL при 5-дневном горизонте и 0.75× vol барьере.
""")


def tab_raw(data: dict) -> None:
    st.header("Сырые данные")
    sel = st.selectbox("Выборка", SPLITS, format_func=lambda s: SPLIT_LABELS.get(s, s))
    decisions = data.get("all_decisions")
    if decisions is None or decisions.empty:
        st.warning("decisions_all.csv не найден.")
        return

    d = decisions[decisions["split"] == sel].sort_index()
    if st.checkbox("Только BUY сигналы"):
        d = d[d["signal"] == "BUY"]

    ver = data.get("version", "v14")
    st.dataframe(d, use_container_width=True)
    st.download_button(
        "Скачать CSV",
        d.to_csv().encode("utf-8"),
        file_name=f"{ver}_{sel}_decisions.csv",
        mime="text/csv",
    )


def tab_cot(data: dict) -> None:
    st.header("COT — Commitments of Traders (CFTC)")

    # Ищем COT данные в v19/v18/v17/v16/v15 директории
    cot = data.get("v15_cot_raw") or data.get("v16_cot_raw") or data.get("v17_cot_raw")
    if cot is None:
        for cot_path in [
            V19_DIR / "v19_cot_raw.csv",
            V18_DIR / "v18_cot_raw.csv",
            V17_DIR / "v17_cot_raw.csv",
            V16_DIR / "v16_cot_raw.csv",
            V15_DIR / "v15_cot_raw.csv",
        ]:
            if cot_path.exists():
                try:
                    cot = pd.read_csv(cot_path, index_col=0, parse_dates=True)
                    break
                except Exception:
                    pass

    if cot is None:
        st.info(
            "COT данные не найдены. Запустите v19:\n"
            "```\npython silver_assistant_v19_trailing.py\n```"
        )
        return

    st.subheader("Чистая позиция спекулянтов (net_spec)")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    if "net_spec" in cot.columns:
        idx = pd.to_datetime(cot.index)
        axes[0].plot(idx, cot["net_spec"], color="#1f77b4", lw=1.2)
        axes[0].axhline(0, color="black", lw=0.8, ls="--")
        axes[0].set_ylabel("Net Speculative")
        axes[0].set_title("Чистая позиция крупных спекулянтов (long - short)")
        axes[0].grid(alpha=0.3)

    if "cot_index_52w" in cot.columns:
        idx = pd.to_datetime(cot.index)
        axes[1].fill_between(idx, cot["cot_index_52w"], alpha=0.5, color="#ff7f0e")
        axes[1].axhline(80, color="red",   ls="--", lw=1, label="Экстремально бычий (80)")
        axes[1].axhline(20, color="green", ls="--", lw=1, label="Экстремально медвежий (20)")
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("COT Index (52w)")
        axes[1].set_title("COT Index — нормированная позиция за 52 недели")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)
    st.dataframe(cot.tail(20), use_container_width=True)
    st.caption(
        "COT Index > 80: спекулянты максимально бычьи → возможный разворот вниз. "
        "COT Index < 20: спекулянты максимально медвежьи → возможный разворот вверх. "
        "Данные публикуются CFTC еженедельно с задержкой ~4 дня."
    )


def tab_version_compare(data: dict) -> None:
    st.header("Сравнение версий")

    v13_gr  = data.get("v13_guardrails")
    v14_gr  = data.get("v14_guardrails")
    v15_gr  = data.get("v15_guardrails")
    v16_gr  = data.get("v16_guardrails")
    v17_gr  = data.get("v17_guardrails")
    v18_gr  = data.get("v18_guardrails")
    cur_gr  = data.get("guardrails")
    cur_ver = data.get("version", "v14")

    rows = []
    for split_label, split_key in [("Valid 2023", "valid"), ("Test 2024", "test"), ("Forward 2025+", "forward")]:
        row: dict = {"Split": split_label}

        for label, gr in [("v13", v13_gr), ("v14", v14_gr), ("v15", v15_gr),
                          ("v16", v16_gr), ("v17", v17_gr), ("v18", v18_gr)]:
            if gr is not None:
                r = gr[gr["split"].str.contains(split_key, case=False, na=False)]
                if not r.empty:
                    row[f"{label} prec"] = pct_str(r["precision"].values[0])
                    row[f"{label} edge"] = "❌" if "ci_lower_not_above_base" in str(r["warning"].values[0]) else "✅"
                else:
                    row[f"{label} prec"] = "-"
                    row[f"{label} edge"] = "-"

        # Текущая версия (v19 или любая новее v18)
        if cur_gr is not None and cur_ver not in ("v13","v14","v15","v16","v17","v18"):
            r = cur_gr[cur_gr["split"] == split_key]
            if not r.empty:
                row[f"{cur_ver} prec"] = pct_str(r["precision"].values[0])
                row[f"{cur_ver} edge"] = "❌" if "ci_lower_not_above_base" in str(r["warning"].values[0]) else "✅"

        rows.append(row)

    comp_df = pd.DataFrame(rows).set_index("Split")
    st.dataframe(comp_df, use_container_width=True)
    st.caption(
        "✅ = Wilson 95% CI нижняя граница > базовый уровень = доказанный edge. "
        "❌ = edge не доказан статистически."
    )

    st.markdown("---")
    st.subheader("Что изменилось в каждой версии")
    st.markdown("""
| Версия | Ключевые изменения |
|---|---|
| **v13** | Close-only triple-barrier, 250 строк обучения, LogisticRegression |
| **v14** | OHLC H/L triple-barrier, 10 лет обучения, HistGradientBoosting, purged CV, FRED, BnH бенчмарк |
| **v15** | + COT (CFTC), горизонт 15 дней, режимная ансамблевая модель (uptrend/sideways/downtrend) |
| **v16** | Бинарная задача (UP vs NOT_UP), усиленная регуляризация, асимметричная стоимость, отбор top-30 признаков |
| **v17** | + Макро-данные через yfinance: ^TNX, ^IRX, TIP, RINF, HYG (15 новых признаков, ~80 до отбора) |
| **v18** | Expanding window (разные модели для valid/test/forward) · Adaptive NOT_UP_weight · Exp. time decay |
| **v19** | **Trailing stop** вместо fixed 15d · trail_pct + max_hold подбираются на valid по Шарпу · Удерживает тренд |
""")

    # Сравнение hyperparameters v15 vs v16
    st.markdown("---")
    st.subheader("Изменения гиперпараметров v15 → v16")
    st.table(pd.DataFrame({
        "Параметр":  ["Задача",       "max_depth", "max_leaf_nodes", "learning_rate", "min_samples_leaf", "l2_regularization", "Вес NOT_UP",    "Отбор признаков"],
        "v15":       ["3-класс.",     "4",         "31 (default)",   "0.04",          "20",               "1.5",               "balanced",      "нет (65 шт.)"],
        "v16":       ["Бинарная",     "3",         "15",             "0.02",          "40",               "3.0",               "2.0",           "top-30 (perm. imp.)"],
        "Цель":      ["Упростить задачу", "Меньше глубина", "Меньше листьев", "Медленнее обучение", "Больше мин. выборка", "Сильнее штраф", "Снизить false positive", "Убрать шум"],
    }))

    if cur_ver == "v19":
        st.markdown("---")
        st.subheader("Изменения v18 → v19 (Trailing Stop)")
        st.table(pd.DataFrame({
            "Аспект":         ["Выход из позиции", "Горизонт", "Параметры выхода", "Использование OHLC", "Бэктест"],
            "v18":            ["Fixed 15 дней", "Фиксированный горизонт=15d", "Нет (принудительный выход)", "Только close-to-close", "backtest_strategy()"],
            "v19":            ["Trailing stop", "Переменный (до max_hold=45d)", "trail_pct + max_hold по Sharpe", "H/L для триггера, close для P&L", "backtest_strategy_trailing()"],
            "Цель":           ["Удержать тренд", "Выйти когда тренд завершается", "Data-driven оптимизация", "Реалистичная симуляция", "Trailing P&L"],
        }))
    elif cur_ver == "v18":
        st.markdown("---")
        st.subheader("Изменения v17 → v18 (Adaptive Window)")
        st.table(pd.DataFrame({
            "Аспект":        ["Окно обучения", "NOT_UP_weight", "Веса выборок", "Поиск порога", "Мин. сигналов"],
            "v17":           ["Фиксированное 2013–2022", "2.0 (фиксированный)", "Равные", "0.48–0.65", "4"],
            "v18":           ["Expanding: +2023 (test), +2024 (fwd)", "Adaptive = f(recent_UP_rate)", "Exp. decay halflife=3y", "0.42–0.65", "3"],
            "Цель":          ["Адаптация к бычьему тренду", "При UP=66% → weight≈1.3", "Recent 3y весят 2× больше", "Больше сигналов", "Шире CI"],
        }))
    elif cur_ver == "v17":
        st.markdown("---")
        st.subheader("Изменения v16 → v17 (Macro yfinance)")
        st.table(pd.DataFrame({
            "Аспект":    ["Признаки", "Источник macro", "Реальные ставки", "Инф. ожидания", "Кривая доходности", "Кредитный риск"],
            "v16":       ["65 (без macro)", "FRED (недоступен)", "❌", "❌", "❌", "❌"],
            "v17":       ["~80 до отбора", "yfinance ETF/индексы", "TIP ETF (tip_ret_5/20d)", "RINF ETF (rinf_ret_5/20d)", "^TNX - ^IRX (yield_curve_10y3m)", "HYG ETF (hyg_ret_5/20d)"],
        }))


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    data = load_best()
    ver  = data.get("version", "v14")

    if not data or "guardrails" not in data:
        st.error("Данные не найдены. Запустите pipeline:")
        st.code(
            "python silver_assistant_v14_main.py\n"
            "# или\npython silver_assistant_v15_regime_cot.py\n"
            "# или\npython silver_assistant_v16_binary.py\n"
            "# или\npython silver_assistant_v18_adaptive.py\n"
            "# или (рекомендуется)\npython silver_assistant_v19_trailing.py",
            language="bash",
        )
        st.stop()

    sidebar(data)

    # Заголовок по версии
    titles = {
        "v19": "v19 (Trailing Stop + Expanding Window + Adaptive Weight)",
        "v18": "v18 (Expanding Window + Adaptive Weight + Time Decay)",
        "v17": "v17 (Macro yfinance + Бинарная + Регуляризация + Отбор признаков)",
        "v16": "v16 (Бинарная + Регуляризация + Отбор признаков)",
        "v15": "v15 (COT + Режимная модель)",
        "v14": "v14",
    }
    captions = {
        "v19": ("OHLC triple-barrier · Trailing stop (trail_pct + max_hold по Sharpe) · "
                "Expanding window (valid/test/forward — разные модели) · "
                "Adaptive NOT_UP_weight · Exp. time decay (halflife=3y) · "
                "Macro yfinance · Top-30 · Purged CV · COT"),
        "v18": ("OHLC triple-barrier · Expanding window (valid/test/forward — разные модели) · "
                "Adaptive NOT_UP_weight (бычий рынок → меньше штраф) · "
                "Exp. time decay (halflife=3y) · Macro yfinance · Top-30 · Purged CV · COT"),
        "v17": ("OHLC triple-barrier · 10 лет обучения · Бинарная задача UP/NOT_UP · "
                "Макро yfinance: ^TNX, ^IRX, TIP, RINF, HYG (15 признаков) · "
                "Регуляризация (depth=3, l2=3) · Top-30 признаков · Purged CV · COT (CFTC)"),
        "v16": ("OHLC triple-barrier · 10 лет обучения · Бинарная задача UP/NOT_UP · "
                "Асимм. стоимость NOT_UP×2 · Регуляризация (depth=3, l2=3) · "
                "Top-30 признаков · Purged CV · COT (CFTC) · BnH бенчмарк"),
        "v15": ("OHLC triple-barrier · 10 лет обучения · Режимная ансамблевая модель · "
                "COT (CFTC) · Горизонт 15 дней · Purged CV · Buy-and-Hold бенчмарк"),
        "v14": ("OHLC triple-barrier · 10 лет обучения · HistGradientBoosting · "
                "Purged CV · Калиброванные вероятности · Buy-and-Hold бенчмарк"),
    }
    st.title(f"🥈 Silver Trading Assistant {titles.get(ver, ver)}")
    st.caption(captions.get(ver, ""))

    tab_names = ["Обзор", "Метрики", "Бэктест", "COT", "Признаки", "Метки", "Версии", "Сырые данные"]
    tabs = st.tabs(tab_names)
    with tabs[0]: tab_overview(data)
    with tabs[1]: tab_metrics(data)
    with tabs[2]: tab_backtest(data)
    with tabs[3]: tab_cot(data)
    with tabs[4]: tab_features(data)
    with tabs[5]: tab_labels(data)
    with tabs[6]: tab_version_compare(data)
    with tabs[7]: tab_raw(data)


if __name__ == "__main__":
    main()
