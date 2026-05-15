"""Страница: история сигналов v25."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils import (
    load_decisions, load_trades, get_current_signal,
    pct, signal_emoji, inject_styles, top_signal_badge,
)
from app.charts import trades_scatter

st.set_page_config(page_title="Сигналы", page_icon="📊", layout="wide")
inject_styles()

st.markdown("# 📊 История сигналов")

top_signal_badge(get_current_signal())

# =============================================================================
# Фильтры
# =============================================================================

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    period = st.selectbox("Период",
                          ["7 дней", "30 дней", "90 дней", "1 год", "Всё"],
                          index=2)
with col2:
    signal_filter = st.selectbox("Тип сигнала",
                                  ["Все", "Только BUY", "Только SHORT",
                                   "Только сработавшие"])
with col3:
    only_active = st.checkbox("Показать только активные дни (не HOLD)", value=True)

# =============================================================================
# Load + filter
# =============================================================================

dec = load_decisions()
if dec.empty:
    st.info("v25 решения не загружены. Запустите `python silver_assistant_v25_cpcv.py`")
    st.stop()

period_days = {"7 дней": 7, "30 дней": 30, "90 дней": 90,
               "1 год": 365, "Всё": 99999}[period]
cutoff = dec.index.max() - pd.Timedelta(days=period_days)
filtered = dec[dec.index >= cutoff].copy()

if only_active:
    filtered = filtered[
        (filtered["signal_long"] == "BUY") |
        (filtered.get("signal_short", "HOLD") == "SHORT")
    ]

if signal_filter == "Только BUY":
    filtered = filtered[filtered["signal_long"] == "BUY"]
elif signal_filter == "Только SHORT":
    filtered = filtered[filtered.get("signal_short", "HOLD") == "SHORT"]


# =============================================================================
# Stats
# =============================================================================

st.markdown("### 📈 Статистика")

trades = load_trades("forward")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    n_buys = (filtered["signal_long"] == "BUY").sum() if "signal_long" in filtered else 0
    st.metric("BUY сигналов", n_buys)
with col2:
    n_shorts = (filtered.get("signal_short", "HOLD") == "SHORT").sum()
    st.metric("SHORT сигналов", n_shorts)
with col3:
    if not trades.empty:
        n_wins = (trades["net_return"] > 0).sum()
        win_rate = n_wins / len(trades)
        st.metric("Win rate", f"{win_rate:.0%}",
                  delta=f"{n_wins}/{len(trades)}")
    else:
        st.metric("Win rate", "—")
with col4:
    if not trades.empty:
        avg_ret = trades["net_return"].mean()
        st.metric("Avg P&L", pct(avg_ret))
    else:
        st.metric("Avg P&L", "—")
with col5:
    if not trades.empty:
        eq_final = np.prod(1 + trades["net_return"].values) - 1
        st.metric("Total compound", pct(eq_final))
    else:
        st.metric("Total compound", "—")


# =============================================================================
# Trade P&L bar chart
# =============================================================================

if not trades.empty:
    st.markdown("### 📊 P&L по каждой сделке")
    fig = trades_scatter(trades)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Таблица сигналов
# =============================================================================

st.markdown("### 📋 Таблица сигналов")

if filtered.empty:
    st.info("Нет сигналов под выбранные фильтры.")
else:
    show = filtered[["signal_long", "p_up", "silver_close", "regime"]].copy()
    if "signal_short" in filtered.columns:
        show["signal_short"] = filtered["signal_short"]

    # Эмодзи в колонке сигнала
    show["Сигнал"] = show["signal_long"].apply(lambda s: f"{signal_emoji(s)} {s}")
    if "signal_short" in show.columns:
        short_emoji = show["signal_short"].apply(
            lambda s: f"{signal_emoji(s)} SHORT" if s == "SHORT" else ""
        )
        show["Шорт"] = short_emoji

    show.index = show.index.strftime("%Y-%m-%d (%a)")
    display = show[["Сигнал", "p_up", "silver_close", "regime"]].copy()
    display.columns = ["Сигнал", "p_up", "Цена ($)", "Режим"]
    display["p_up"] = display["p_up"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "n/a"
    )
    display["Цена ($)"] = display["Цена ($)"].apply(lambda x: f"${x:.2f}")

    st.dataframe(display.iloc[::-1], use_container_width=True, height=500)


# =============================================================================
# Trade-by-trade detail
# =============================================================================

if not trades.empty:
    st.markdown("---")
    st.markdown("### 🔍 Детализация сделок (forward split)")

    t = trades.copy().sort_values("entry_date", ascending=False)
    t["Длит. (дн)"] = (t["exit_date"] - t["entry_date"]).dt.days
    t["P&L %"] = t["net_return"].apply(lambda x: f"{x*100:+.2f}%")
    t["Entry"] = t["entry_date"].dt.strftime("%Y-%m-%d")
    t["Exit"] = t["exit_date"].dt.strftime("%Y-%m-%d")
    t["Result"] = t["net_return"].apply(lambda x: "✅" if x > 0 else "❌")

    show_cols = ["Result", "Entry", "Exit", "Длит. (дн)",
                 "direction", "entry_price", "exit_price", "P&L %"]
    show_cols = [c for c in show_cols if c in t.columns]
    st.dataframe(t[show_cols], use_container_width=True, hide_index=True,
                 column_config={
                     "direction":   "Сторона",
                     "entry_price": st.column_config.NumberColumn("Entry $", format="%.3f"),
                     "exit_price":  st.column_config.NumberColumn("Exit $",  format="%.3f"),
                 })
