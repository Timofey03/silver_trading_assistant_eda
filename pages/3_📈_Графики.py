"""Страница: интерактивные графики цены + сигналы + drawdown."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils import (
    load_decisions, load_full_data, load_trades, get_current_signal,
    inject_styles, top_signal_badge,
)
from app.charts import candlestick_with_signals, drawdown_chart, equity_curve

st.set_page_config(page_title="Графики", page_icon="📈", layout="wide")
inject_styles()

st.markdown("# 📈 Графики")

top_signal_badge(get_current_signal())


# =============================================================================
# Контролы
# =============================================================================

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    period = st.selectbox("Период",
                          ["1 месяц", "3 месяца", "6 месяцев", "1 год", "Всё"],
                          index=2)
with col2:
    show_split = st.selectbox(
        "Split", ["forward", "test", "valid", "train", "Все"],
        index=0,
        help="Временной сплит данных:\n"
             "• train — данные на которых модель училась (2013-2022)\n"
             "• valid — для подбора гиперпараметров (2023)\n"
             "• test — чистый OOS-тест (2024)\n"
             "• forward — самые свежие данные (2025+)\n"
             "• Все — объединение",
    )

# Карточка-объяснение под выбранным сплитом
split_info = {
    "forward": ("🟢 Forward — данные 2025+",
                 "Самый важный сплит: модель НИКОГДА не видела эти данные. "
                 "Реальная проверка работоспособности. Сюда попадают live-сигналы."),
    "test":    ("🟡 Test — данные 2024",
                 "Финальная out-of-sample проверка. Модель училась ДО, "
                 "а потом мы её тестировали на этом периоде."),
    "valid":   ("🟠 Validation — данные 2023",
                 "Сплит для подбора гиперпараметров (порог 0.55, cooldown 15). "
                 "Не показатель реального edge — модель сюда «подгонялась»."),
    "train":   ("⚪ Train — данные 2013-2022",
                 "Обучающая выборка. Бесполезна для оценки результата — "
                 "модель ВИДЕЛА эти данные."),
    "Все":     ("⚫ Все периоды",
                 "Полный диапазон: обучающие + валидация + тест + forward. "
                 "Удобно для общего обзора, не для оценки edge."),
}
title, desc = split_info.get(show_split, ("", ""))
if title:
    st.info(f"**{title}** — {desc}")


# =============================================================================
# Data
# =============================================================================

dec = load_decisions()
full = load_full_data()

if dec.empty or full.empty:
    st.error("Данные не загружены.")
    st.stop()

# Объединяем decisions с OHLC из full
merged = full.copy()
merged["signal_long"] = dec["signal_long"].reindex(merged.index)
merged["signal_short"] = dec["signal_short"].reindex(merged.index) if "signal_short" in dec.columns else "HOLD"
merged["p_up"] = dec["p_up"].reindex(merged.index)

# Фильтр по периоду
period_days = {"1 месяц": 30, "3 месяца": 90, "6 месяцев": 180,
               "1 год": 365, "Всё": 99999}[period]
cutoff = merged.index.max() - pd.Timedelta(days=period_days)
view = merged[merged.index >= cutoff].copy()

# Фильтр по split
if show_split != "Все":
    view = view[view["split"] == show_split] if "split" in view.columns else view


# =============================================================================
# Candlestick
# =============================================================================

st.markdown("### 🕯 Свечной график SLV (silver close) + сигналы модели")
st.caption("🟢 = BUY signal, 🔴 = SHORT signal · Нижний график: уверенность модели (p_up)")

if not view.empty and {"silver_open", "silver_high", "silver_low", "silver_close"}.issubset(view.columns):
    fig = candlestick_with_signals(view, view, height=600)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("OHLC данные отсутствуют для выбранного периода.")


# =============================================================================
# Equity curve — переключатель E3b / V25 + маркеры BUY/SELL
# =============================================================================

st.markdown("---")
st.markdown("### 📈 Equity curve — наша стратегия vs Buy-and-Hold")

# Выбор модели
chart_model_g = st.radio(
    "Модель для графика:",
    options=["🏆 E3b (новая, диплом)", "🟢 V25 (legacy)"],
    horizontal=True,
    help=(
        "**E3b** — walk-forward 10.3 года (2015-2025), 48 сделок. Реалистичные результаты.\n\n"
        "**V25** — forward test только 1.3 года (2025-2026 bull rally), 24-38 сделок. "
        "Compound 7.6x не воспроизводим в обычных условиях."
    ),
    key="chart_model_graphs",
)

if chart_model_g.startswith("🏆"):
    # E3b
    e3b_path = ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv"
    if e3b_path.exists():
        trades = pd.read_csv(e3b_path)
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        strategy_name = "Стратегия E3b ★"
        # BnH за E3b период
        period_start = trades["entry_date"].min()
        period_end = trades["exit_date"].max()
        try:
            from app.multi_asset.metal_loader import load_single_metal
            silver = load_single_metal("silver")
            bnh = silver["close"].loc[period_start:period_end]
        except Exception:
            bnh = pd.Series(dtype=float)
    else:
        trades = pd.DataFrame()
        bnh = pd.Series(dtype=float)
        strategy_name = "E3b"
else:
    trades = load_trades("forward")
    fwd_full = full[full["split"] == "forward"] if "split" in full.columns else full
    bnh = fwd_full["silver_close"] if not fwd_full.empty else pd.Series(dtype=float)
    strategy_name = "Стратегия V25"

if not trades.empty:
    fig = equity_curve(trades, bnh, strategy_name=strategy_name, show_buy_sell_markers=True)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 BUY — момент входа в позицию · 🔴 SELL — момент выхода (наведите для деталей)")
else:
    st.info("Нет данных для equity curve.")


# =============================================================================
# Drawdown
# =============================================================================

st.markdown("---")
st.markdown("### 📉 Просадка стратегии")
st.caption("Drawdown = текущая equity / максимальная historic equity − 1")

if not trades.empty:
    t = trades.sort_values("exit_date").copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    eq = np.cumprod(1.0 + t["net_return"].astype(float).values)
    fig = drawdown_chart(eq, t["exit_date"])
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1
    with col1:
        st.metric("Max Drawdown", f"{dd.min()*100:.2f}%")
    with col2:
        time_uw = (dd < -0.001).mean()
        st.metric("Time underwater", f"{time_uw:.0%}")
    with col3:
        ulcer = np.sqrt(np.mean(dd ** 2)) * 100
        st.metric("Ulcer Index", f"{ulcer:.2f}%")
