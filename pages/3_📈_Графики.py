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
    show_split = st.selectbox("Split",
                              ["forward", "test", "valid", "Все"], index=0)


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
# Equity curve
# =============================================================================

st.markdown("---")
st.markdown("### 📈 Equity curve — наша стратегия vs Buy-and-Hold")

trades = load_trades("forward")
fwd_full = full[full["split"] == "forward"] if "split" in full.columns else full
bnh = fwd_full["silver_close"] if not fwd_full.empty else pd.Series(dtype=float)

if not trades.empty:
    fig = equity_curve(trades, bnh)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Нет данных для equity curve (forward split пустой).")


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
