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

    auto_refresh = st.checkbox("⏱ Авто-обновление (60 сек)", value=False,
                                 help="Перезагружает данные каждые 60 секунд")
    if auto_refresh:
        import time
        time.sleep(60)
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
        non_cash = [p for p in tinkoff["positions"]
                    if p["qty"] != 0 and p["instrument_type"] != "currency"]
        n_pos = len(non_cash)
        futures_val = tinkoff["futures"]["value"]
        st.metric(
            "📊 Открытых позиций", f"{n_pos}",
            delta=f"{rub(futures_val)} notional",
            delta_color="off",
            help="Кол-во открытых позиций (без cash). "
                 "Notional = квот.цена × множитель × кол-во лотов. "
                 "Это ваша **экспозиция** на рынок, а не реальные деньги — "
                 "маржа реально занимает 10-15% от notional.",
        )
    else:
        st.metric("📊 Открытых позиций", "—")

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

if s == "SELL":
    st.error(f"""
**🔴 ПРОДАТЬ открытые позиции**

Модель **рекомендует выход**: p_up = **{p_up_pct}** упало ниже exit-порога 45%.
Это сигнал что вероятность дальнейшего роста снизилась — лучше зафиксировать прибыль/убыток.

**Что делает paper trading**:
- Закрывает все открытые LONG позиции по SLVRUBF
- Освобождает margin для следующих сигналов

**Что делать вам**:
- Если есть открытые позиции — рекомендуется выход
- Если позиций нет — просто ждём следующий BUY-сигнал
""")
elif s == "BUY":
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
    # HOLD — разделяем 2 кейса: cooldown (модель уверена, но ждём) и no_edge
    above_thr = sig.get("above_threshold", False)
    cooldown = sig.get("cooldown_remaining", 0)
    trend_5d = sig.get("p_up_trend_5d")
    trend_5d_pct = f"{trend_5d:.0%}" if trend_5d is not None else "—"

    if above_thr and cooldown > 0:
        st.warning(f"""
**⏳ ДЕРЖАТЬ — модель уверена, но cooldown ещё активен**

Модель **видит UP-сигнал**: p_up = **{p_up_pct}** (выше порога 49%).
Тренд за последние 5 дней: **{trend_5d_pct}**.

Но **cooldown ещё {cooldown} торговых дней** от последнего BUY-сигнала.

**Что будет дальше**: если p_up останется выше 49% — через {cooldown} дней
paper trading автоматически откроет позицию.

- **Режим рынка**: `{sig.get('regime', '—')}`
- **Текущая цена**: ${sig.get('current_price', 0):.2f}
""")
    else:
        st.info(f"""
**⚪ ДЕРЖАТЬ — нет нового сигнала**

Модель не видит достаточно сильного сигнала для входа в позицию.

- **Уверенность модели**: {p_up_pct} (нужно >49%)
- **Тренд 5d**: {trend_5d_pct}
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

# =============================================================================
# 💡 Калькулятор размера позиции (интегрирован в Главную)
# =============================================================================

st.markdown("---")
st.markdown("### 💡 На какую сумму покупать?")
st.caption("Risk-based калькулятор: вы выбираете риск % счёта на сделку — "
            "получаете точный размер позиции")

from app.sizing import SizingInputs, calculate_position_size, explain_sizing, INSTRUMENTS

calc_col1, calc_col2 = st.columns(2)

with calc_col1:
    st.markdown("**Ваш капитал и риск**")
    default_acc = tinkoff.get("cash", {}).get("value", 1_000_000) if tinkoff.get("ok") else 1_000_000
    calc_account = st.number_input(
        "Размер счёта (RUB)", value=int(default_acc),
        min_value=10_000, step=10_000, key="calc_acc",
        help="Текущая стоимость вашего счёта (cash + позиции)",
    )
    calc_risk_pct = st.slider(
        "Риск на сделку (%)", 0.5, 5.0, 1.5, 0.1, key="calc_risk",
        help="% капитала которые готовы потерять если сработает stop. "
             "Профи: 0.5-2%. Розница: до 5%.",
    )
    calc_stop_pct = st.slider(
        "Стоп-дистанция (%)", 2.0, 15.0, 8.0, 0.5, key="calc_stop",
        help="Optimal mode использует trailing stop 8%",
    ) / 100.0

with calc_col2:
    st.markdown("**Инструмент**")
    calc_instr_name = st.selectbox(
        "Что покупаем", list(INSTRUMENTS.keys()), index=1, key="calc_instr",
    )
    calc_instrument = INSTRUMENTS[calc_instr_name]

    if calc_instrument.currency == "USD":
        calc_price = st.number_input(
            f"Цена {calc_instr_name} (USD)", value=85.0,
            min_value=0.01, step=0.5, key="calc_price",
        )
        calc_usd = st.number_input("Курс USD/RUB", value=80.0, step=0.5, key="calc_usd")
    else:
        default_price = 200.0 if calc_instr_name == "SLVRUBF" else 14000.0
        calc_price = st.number_input(
            f"Цена {calc_instr_name} (RUB)", value=default_price,
            min_value=0.01, step=1.0, key="calc_price",
        )
        calc_usd = 80.0

    calc_p_up = sig.get("p_up", 0.5) if sig.get("p_up") is not None else 0.5
    st.metric("Текущая p_up модели", f"{calc_p_up:.0%}",
              help="Используется для Kelly tilt — масштаб позиции по уверенности")

# Расчёт
sizing_result = calculate_position_size(SizingInputs(
    account_value_rub=calc_account,
    risk_per_trade_pct=calc_risk_pct,
    stop_distance_pct=calc_stop_pct,
    instrument=calc_instrument,
    current_price=calc_price,
    usd_rub_rate=calc_usd,
    p_up=calc_p_up,
    confidence_tilt=True,
))

if sizing_result.lots > 0:
    st.success(explain_sizing(sizing_result, SizingInputs(
        account_value_rub=calc_account,
        risk_per_trade_pct=calc_risk_pct,
        stop_distance_pct=calc_stop_pct,
        instrument=calc_instrument,
        current_price=calc_price,
        usd_rub_rate=calc_usd,
        p_up=calc_p_up,
        confidence_tilt=True,
    )))
else:
    st.error(f"❌ {sizing_result.reason}")

with st.expander("📚 Принципы position sizing"):
    st.markdown("""
**Главное правило**: размер позиции выбирается **исходя из риска**, а не из «хочу больше прибыли».

**Пример**: у вас 1М ₽ счёт, риск 1.5%, стоп 8% (optimal mode).
- Максимальный убыток = 15,000 ₽ (1.5% × 1М)
- Stop loss на лот SLVRUBF ≈ 20,000 × 0.08 = 1,600 ₽
- → размер позиции = 15,000 / 1,600 = **9 лотов**

**Kelly tilt**: при высокой уверенности (p_up=0.7+) размер ×1.4, при p_up=0.5 — ×0.5.

**Почему 1-3% риска**: 10 убыточных сделок подряд при 5% риска = -40% drawdown.
При 1.5% = только -14%, восстановимо.
""")


# =============================================================================
# Сводка модели
# =============================================================================

if not pnl.empty:
    st.markdown("---")
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
