"""
Silver Trading Assistant — Streamlit Web UI

Запуск:
    streamlit run dashboard_app.py

Главная страница (Dashboard). Остальные страницы — в pages/.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.utils import (
    load_decisions, load_pnl_summary, load_trades, load_dsr_psr,
    load_full_data, get_current_signal, get_kpis, get_tinkoff_status,
    pct, rub, usd, signal_emoji,
    inject_styles, top_signal_badge,
)
from app.charts import equity_curve

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Silver Assistant",
    page_icon="🥈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("# 🥈 Silver Assistant")
    st.markdown("*ML-помощник для торговли серебром*")
    st.markdown("---")
    st.caption(f"v25 CPCV · обновлено {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Перезагрузить данные", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# =============================================================================
# Header
# =============================================================================

st.markdown("# 🏠 Главная")
st.caption("Текущий сигнал, баланс, история — всё в одном месте")

sig = get_current_signal()
kpis = get_kpis()
tinkoff = get_tinkoff_status()

# Большая карточка сигнала
top_signal_badge(sig)


# =============================================================================
# KPI cards
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    if tinkoff.get("ok"):
        total = tinkoff["total"]["value"]
        st.metric("💰 Баланс", rub(total),
                  delta=rub(tinkoff["expected_yield"]["value"]),
                  delta_color="normal")
    else:
        st.metric("💰 Баланс", "—",
                  help=f"Tinkoff недоступен: {tinkoff.get('error', '')}")

with col2:
    if tinkoff.get("ok"):
        n_pos = len([p for p in tinkoff["positions"] if p["qty"] != 0])
        futures_val = tinkoff["futures"]["value"]
        st.metric("📊 Позиции", f"{n_pos}",
                  delta=f"{rub(futures_val)} в futures",
                  delta_color="off")
    else:
        st.metric("📊 Позиции", "—")

with col3:
    price = kpis.get("last_price", 0)
    ret_7d = kpis.get("ret_7d", 0)
    st.metric("💲 Серебро (SLV)", usd(price),
              delta=pct(ret_7d), delta_color="normal")

with col4:
    ret_30d = kpis.get("ret_30d", 0)
    st.metric("📈 За 30 дней", pct(ret_30d),
              delta=None)


# =============================================================================
# Объяснение сигнала
# =============================================================================

st.markdown("### 💬 Что это значит и что делать")

s = sig.get("signal", "HOLD")
p_up = sig.get("p_up")
p_up_pct = f"{p_up:.0%}" if p_up is not None else "—"

if s == "BUY":
    st.success(f"""
**🟢 ПОКУПАТЬ серебро**

Модель видит сигнал на повышение. Уверенность модели: **{p_up_pct}** (это выше порога 55%).

**Что делает paper trading**:
- Покупает 1-2 лота SLVRUBF в Tinkoff sandbox
- Trailing stop 7% от пика
- Максимальный holding period: 45 торговых дней

**Что делать вам**:
- Если согласны — на следующий рабочий день paper trading исполнит сделку автоматически
- Если не согласны — никаких действий, модель не использует ваши деньги
""")
elif s == "SHORT" or sig.get("signal_short") == "SHORT":
    st.error(f"""
**🔴 ПРОДАВАТЬ серебро (шорт)**

Модель видит сигнал на снижение. Уверенность модели на падение: **{p_up_pct}**.

**Что делает paper trading**:
- Открывает SHORT позицию в SLVRUBF
- Inverted trailing stop +7% от минимума
""")
else:
    st.info(f"""
**⚪ ДЕРЖАТЬ — нет нового сигнала**

Модель не видит достаточно сильного сигнала для входа в позицию.

- **Уверенность модели**: {p_up_pct} (нужно >55%)
- **Режим рынка**: `{sig.get('regime', '—')}`
- **Текущая цена**: ${sig.get('current_price', 0):.2f}

**Что делать**: ничего. Стратегия v25 даёт ~3-5 сигналов в год — это нормально.
Следующий сигнал ожидается когда `p_up` превысит порог.
""")


# =============================================================================
# Equity curve
# =============================================================================

st.markdown("### 📈 Историческая производительность")
st.caption("Сравнение нашей стратегии (compound, single-position) с обычным buy-and-hold")

trades = load_trades("forward")
full = load_full_data()

if not trades.empty and not full.empty:
    # BnH серия за период forward
    fwd = full[full["split"] == "forward"]
    bnh = fwd["silver_close"]
    fig = equity_curve(trades, bnh)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Данных для графика пока недостаточно. Запустите v25 retrain.")


# =============================================================================
# Quick stats
# =============================================================================

pnl = load_pnl_summary()
dsr = load_dsr_psr()

if not pnl.empty:
    st.markdown("### 📊 Сводка модели (v25 CPCV)")

    cols = st.columns(len(pnl))
    for i, (_, row) in enumerate(pnl.iterrows()):
        with cols[i]:
            split = row["split"]
            our   = float(row["v25_honest_total"]) if pd.notna(row["v25_honest_total"]) else 0
            bnh   = float(row["true_bnh"]) if pd.notna(row["true_bnh"]) else 0
            delta = our - bnh

            split_label = {
                "valid":   "Валидация (2023)",
                "test":    "Тест (2024)",
                "forward": "Forward (2025+)",
            }.get(split, split)

            st.markdown(f"**{split_label}**")
            st.metric("Стратегия", pct(our),
                      delta=f"{pct(delta)} vs BnH",
                      delta_color="normal")
            st.caption(f"BnH: {pct(bnh)} · сделок: {int(row.get('n_sequential', 0))}")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.caption(f"""
**Внимание**: это исследовательский проект. Все сигналы — paper trading в Tinkoff sandbox,
без реальных денег. Перед любым real-money использованием — минимум 6 месяцев живого OOS.

GitHub: [Timofey03/silver_trading_assistant_eda](https://github.com/Timofey03/silver_trading_assistant_eda) ·
Обновлено: {kpis.get('last_update', '—')}
""")
