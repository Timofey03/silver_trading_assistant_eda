"""Страница: 📍 Сейчас — текущее состояние и действие пользователя."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils import get_current_signal, get_kpis
from app.simple_storage import (
    add_open_trade, close_open_trade, get_capital,
    get_open_trade, update_max_price,
)


# Параметры исполнения OptimalV2 (захардкожены)
TRAIL_PCT = 0.12          # 12% trailing stop
MAX_HOLD_DAYS = 30


# =============================================================================
# Загрузка состояния
# =============================================================================
signal = get_current_signal()
kpis = get_kpis()
open_trade = get_open_trade()
capital = get_capital()

current_price = signal.get("current_price") or kpis.get("last_price") or 0


# =============================================================================
# Шапка
# =============================================================================
st.title("📍 Что сейчас делать")

today_str = datetime.now().strftime("%d.%m.%Y")
source = signal.get("source", "—")
model_label_map = {
    "e3b_daily":     "🏆 E3b (новая модель диплома)",
    "production":    "🟢 V25 production",
    "cpcv_fallback": "🔵 V25 CPCV fallback",
    "none":          "⚪ нет данных",
}
model_label = model_label_map.get(source, source)

# Переключатель уровня сложности (сохраняется в session_state)
if "user_level" not in st.session_state:
    st.session_state.user_level = "🌱 Новичок"
level = st.radio(
    "Уровень опыта:",
    options=["🌱 Новичок", "📊 Продвинутый", "🔬 Эксперт"],
    horizontal=True,
    key="user_level",
    help=(
        "**🌱 Новичок** — только «что делать сейчас», объяснения простым языком, "
        "никаких профессиональных терминов.\n\n"
        "**📊 Продвинутый** — добавлены ключевые метрики (Win Rate, прибыль), "
        "видны параметры стратегии.\n\n"
        "**🔬 Эксперт** — полная техническая информация: Sharpe, p_up, источник "
        "модели, OOS metrics."
    ),
)
is_novice = level.startswith("🌱")
is_advanced = level.startswith("📊")
is_expert = level.startswith("🔬")

if is_expert:
    st.caption(
        f"Сегодня {today_str} · модель: **{model_label}** · "
        f"цена серебра (фьючерс): **{current_price:,.0f} ₽** за лот".replace(",", " ")
    )
elif is_advanced:
    st.caption(
        f"Сегодня {today_str} · цена серебра: **{current_price:,.0f} ₽** за лот".replace(",", " ")
    )
else:  # novice
    st.caption(f"Сегодня {today_str} · цена серебра: **{current_price:,.0f} ₽**".replace(",", " "))

# Если сигнал устарел — предупреждаем
if signal.get("is_stale"):
    if is_novice:
        st.warning("⚠ Данные не самые свежие. Можете подождать обновления.")
    else:
        st.warning(signal.get("stale_reason", "⚠ Данные устарели"))

# Бейдж: action vs info (если есть метаданные дедупликации)
if signal.get("alert_type") == "info" and signal.get("is_repeat"):
    prev_sig = signal.get("previous_signal", "—")
    st.info(
        f"ℹ **Сигнал не изменился** — это повторное уведомление дня (предыдущий: {prev_sig}). "
        f"Если ты уже отреагировал утром — повторно ничего делать не нужно."
    )
elif signal.get("alert_type") == "action" and signal.get("previous_signal"):
    prev_sig = signal.get("previous_signal", "—")
    new_sig = signal.get("signal", "—")
    st.success(
        f"📢 **НОВЫЙ СИГНАЛ:** {prev_sig} → **{new_sig}** — действовать сейчас."
    )


# =============================================================================
# СОСТОЯНИЕ 1: Есть открытая позиция
# =============================================================================
if open_trade is not None:
    entry_price = open_trade["entry_price"]
    lots = open_trade["lots"]
    entry_date = pd.to_datetime(open_trade["entry_date"]).date()
    days_held = (datetime.now().date() - entry_date).days
    days_left = max(0, MAX_HOLD_DAYS - days_held)

    # Обновляем максимум цены для trailing stop
    if current_price > 0:
        update_max_price(current_price)
        open_trade = get_open_trade()

    max_price = open_trade["max_price_seen"]
    trail_stop_level = max_price * (1 - TRAIL_PCT)
    pnl_pct = (current_price / entry_price - 1) * 100 if current_price > 0 else 0
    pnl_rub_per_lot = (current_price - entry_price) * 100  # SLVRUBF multiplier = 100
    pnl_rub_total = pnl_rub_per_lot * lots

    # Цвет фона: зелёный или красный
    bg_color = "#E8F5E9" if pnl_pct >= 0 else "#FFEBEE"
    txt_color = "#2E7D32" if pnl_pct >= 0 else "#C62828"

    st.markdown(
        f"""
        <div style="background:{bg_color}; padding:24px; border-radius:12px;
                    border-left: 6px solid {txt_color};">
            <h2 style="color:{txt_color}; margin:0;">
                🟢 Позиция открыта · {lots} лот{'а' if 1<lots<5 else 'ов' if lots>4 else ''}
            </h2>
            <p style="font-size:18px; margin:8px 0 0 0; color:#333;">
                Куплено {entry_date.strftime('%d.%m.%Y')} по цене {entry_price:,.0f} ₽
            </p>
        </div>
        """.replace(",", " "),
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Прибыль/убыток", f"{pnl_pct:+.2f}%")
    with c2:
        rub_label = f"{pnl_rub_total:+,.0f} ₽".replace(",", " ")
        st.metric("В рублях", rub_label,
                  help=f"На {lots} лот(ов). 1 лот = 100 ед. серебра.")
    with c3:
        st.metric("Дней в позиции", days_held, help=f"Закрытие через {days_left} дней")
    with c4:
        stop_dist_pct = (current_price / trail_stop_level - 1) * 100 if trail_stop_level > 0 else 0
        st.metric("До стопа",
                  f"{stop_dist_pct:.1f}%",
                  help=f"Защита сработает при цене ниже {trail_stop_level:,.0f} ₽"
                       .replace(",", " "))

    st.divider()
    st.markdown("### 🛡 Защита от убытков (trailing stop)")
    st.markdown(
        f"""
        Защитный уровень автоматически поднимается вслед за ценой:
        - Максимум цены с момента покупки: **{max_price:,.0f} ₽**
        - Защитный уровень сейчас: **{trail_stop_level:,.0f} ₽**
            ({TRAIL_PCT*100:.0f}% ниже максимума)
        - Если цена опустится ниже — нужно закрыть позицию.
        """.replace(",", " ")
    )

    # Автоматическое предупреждение о срабатывании
    if current_price > 0 and current_price < trail_stop_level:
        st.error(
            f"⚠ Цена ({current_price:,.0f} ₽) опустилась ниже защитного уровня! "
            f"Рекомендуется закрыть позицию.".replace(",", " ")
        )
    elif days_left == 0:
        st.warning("⏰ Истёк срок удержания (30 дней). Рекомендуется закрыть.")

    st.divider()
    st.markdown("### Закрыть позицию")
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        exit_price = st.number_input(
            "По какой цене закрываете?",
            min_value=0.0, value=float(current_price), step=10.0,
            help="По умолчанию — текущая рыночная цена",
        )
    with cc2:
        st.write("")
        st.write("")
        if st.button("✅ Закрыть сделку", type="primary", use_container_width=True):
            closed = close_open_trade(exit_price, reason="manual")
            if closed:
                st.success(f"Сделка закрыта. P&L: {closed['pnl_pct']:+.2f}%")
                st.balloons()
                st.rerun()


# =============================================================================
# СОСТОЯНИЕ 2: Свежий сигнал на покупку
# =============================================================================
elif signal.get("signal") == "BUY":
    p_up = signal.get("p_up", 0)
    confidence = p_up * 100 if p_up else 0

    # === Главный блок BUY (адаптируется под уровень) ===
    if is_novice:
        # 🌱 НОВИЧОК — большая зелёная карточка с эмодзи и одной фразой
        st.markdown(
            f"""
            <div style="background:#E8F5E9; padding:32px; border-radius:16px;
                        border-left: 8px solid #2E7D32; text-align:center;">
                <div style="font-size:72px; margin:0;">🟢 ✋</div>
                <h1 style="color:#2E7D32; margin:8px 0 0 0; font-size:36px;">
                    Можно покупать
                </h1>
                <p style="font-size:20px; margin:12px 0 0 0; color:#333;">
                    Помощник уверен в росте серебра в ближайшие 2-4 недели.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        # Простая шкала уверенности вместо процентов
        if confidence >= 65:
            conf_text = "💪 Уверенность: **Высокая**"
        elif confidence >= 50:
            conf_text = "👌 Уверенность: **Средняя**"
        else:
            conf_text = "🤏 Уверенность: **Слабая**"
        st.info(conf_text)
    else:
        # 📊 ПРОДВИНУТЫЙ / 🔬 ЭКСПЕРТ
        st.markdown(
            f"""
            <div style="background:#E8F5E9; padding:24px; border-radius:12px;
                        border-left: 6px solid #2E7D32;">
                <h2 style="color:#2E7D32; margin:0;">
                    🟢 Помощник рекомендует купить
                </h2>
                <p style="font-size:18px; margin:8px 0 0 0; color:#333;">
                    Уверенность модели: <b>{confidence:.0f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")

    # === Параметры сделки ===
    trail_stop = current_price * (1 - TRAIL_PCT)
    if is_novice:
        # 🌱 НОВИЧОК — только 2 главных числа в простых словах
        c1, c2 = st.columns(2)
        with c1:
            st.metric("💵 Купить по цене", f"{current_price:,.0f} ₽".replace(",", " "))
        with c2:
            loss_pct = TRAIL_PCT * 100
            st.metric(f"🛡️ Помощник продаст если упадёт на {loss_pct:.0f}%",
                      f"{trail_stop:,.0f} ₽".replace(",", " "),
                      help="Это «стоп» — автоматическая защита от больших потерь")
    else:
        # 📊 / 🔬 — стандартные 3 метрики
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Цена входа", f"{current_price:,.0f} ₽".replace(",", " "))
        with c2:
            st.metric("Защитный уровень",
                      f"{trail_stop:,.0f} ₽".replace(",", " "),
                      help=f"{TRAIL_PCT*100:.0f}% ниже текущей цены, "
                           "поднимается за движением")
        with c3:
            st.metric("Срок", f"до {MAX_HOLD_DAYS} дней",
                      help="Автоматическое закрытие если стоп не сработает")

    if capital == 0:
        st.info(
            "💡 Укажите свой капитал в **⚙ Настройки**, чтобы помощник "
            "рассчитал размер позиции в рублях."
        )
    else:
        # Простой расчёт лотов: 30% капитала по умолчанию
        notional_per_lot = current_price * 100  # SLVRUBF multiplier
        target_notional = capital * 0.3
        suggested_lots = max(1, int(target_notional / notional_per_lot))
        st.info(
            f"💡 При вашем капитале **{capital:,.0f} ₽** рекомендуется "
            f"купить **{suggested_lots} лот{'а' if 1<suggested_lots<5 else 'ов' if suggested_lots>4 else ''}** "
            f"(~30% от счёта). Подробный расчёт — в **🧮 Калькулятор**."
            .replace(",", " ")
        )

    st.divider()
    st.markdown("### Я открыл сделку")
    cc1, cc2, cc3 = st.columns([2, 1, 1])
    with cc1:
        actual_price = st.number_input(
            "По какой цене купили?",
            min_value=0.0, value=float(current_price), step=10.0,
        )
    with cc2:
        actual_lots = st.number_input(
            "Сколько лотов?",
            min_value=1, value=1, step=1,
        )
    with cc3:
        st.write("")
        st.write("")
        if st.button("📝 Записать", type="primary", use_container_width=True):
            try:
                trade = add_open_trade(
                    entry_price=actual_price,
                    lots=int(actual_lots),
                    signal_date=str(signal.get("signal_date", "")),
                    trail_pct=TRAIL_PCT,
                    max_hold_days=MAX_HOLD_DAYS,
                )
                st.success(f"Сделка записана. ID: {trade['id']}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))


# =============================================================================
# СОСТОЯНИЕ 3: Ожидание (HOLD или SELL)
# =============================================================================
else:
    cooldown = signal.get("cooldown_remaining", 0)
    p_up = signal.get("p_up", 0) or 0
    signal_status = signal.get("signal", "HOLD")

    if signal_status == "SELL":
        title = "🔴 Помощник не рекомендует покупать"
        sub = f"Уверенность в росте упала до {p_up*100:.0f}%"
        color = "#C62828"
        bg = "#FFEBEE"
    elif cooldown > 0:
        title = "⏳ Помощник ждёт"
        sub = f"После предыдущей сделки осталось ещё {cooldown} торговых дней до возможного нового входа"
        color = "#F9A825"
        bg = "#FFF8E1"
    else:
        title = "⚪ Помощник ждёт подходящий момент"
        sub = f"Текущая уверенность в росте: {p_up*100:.0f}% (нужно ≥48%)"
        color = "#616161"
        bg = "#F5F5F5"

    st.markdown(
        f"""
        <div style="background:{bg}; padding:24px; border-radius:12px;
                    border-left: 6px solid {color};">
            <h2 style="color:{color}; margin:0;">{title}</h2>
            <p style="font-size:18px; margin:8px 0 0 0; color:#333;">{sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.info(
        "💡 Помощник торгует выборочно — открывает позицию только когда условия "
        "благоприятные. В среднем это 4–5 сделок в год. Свободное время "
        "можно использовать для других дел — программа сама проверяет рынок "
        "три раза в день."
    )

    # Полезная информация в ожидании
    st.divider()
    st.markdown("### Что происходит на рынке")
    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        st.metric("Цена сейчас",
                  f"{current_price:,.0f} ₽".replace(",", " ") if current_price else "—")
    with cm2:
        ret_7d = kpis.get("ret_7d", 0) * 100
        st.metric("За неделю", f"{ret_7d:+.1f}%")
    with cm3:
        ret_30d = kpis.get("ret_30d", 0) * 100
        st.metric("За месяц", f"{ret_30d:+.1f}%")


# =============================================================================
# Ручная запись сделки (всегда доступна — для тестов и пропущенных сигналов)
# =============================================================================
if open_trade is None:
    st.divider()
    with st.expander("✍ Записать сделку вручную (если открыли без сигнала или для теста)"):
        st.caption(
            "Используйте если уже купили серебро самостоятельно или хотите "
            "протестировать раздел «💼 Мои сделки»."
        )
        mc1, mc2, mc3, mc4 = st.columns([2, 1, 1, 1])
        with mc1:
            manual_price = st.number_input(
                "Цена покупки (₽)",
                min_value=0.0,
                value=float(current_price) if current_price > 0 else 8000.0,
                step=10.0,
                key="manual_entry_price",
            )
        with mc2:
            manual_lots = st.number_input(
                "Лотов", min_value=1, value=1, step=1,
                key="manual_entry_lots",
            )
        with mc3:
            manual_date = st.date_input(
                "Дата", value=datetime.now().date(),
                key="manual_entry_date",
            )
        with mc4:
            st.write("")
            st.write("")
            if st.button("📝 Записать", type="primary", use_container_width=True,
                         key="manual_record_btn"):
                try:
                    trade = add_open_trade(
                        entry_price=manual_price,
                        lots=int(manual_lots),
                        signal_date=manual_date.strftime("%Y-%m-%d"),
                        trail_pct=TRAIL_PCT,
                        max_hold_days=MAX_HOLD_DAYS,
                    )
                    # Подменяем entry_date чтобы соответствовать введённой
                    from app.simple_storage import _load_trades_raw, _save_trades_raw
                    all_t = _load_trades_raw()
                    for t in all_t:
                        if t["id"] == trade["id"]:
                            t["entry_date"] = manual_date.strftime("%Y-%m-%d")
                    _save_trades_raw(all_t)
                    st.success(f"Сделка записана: {manual_lots} лот(ов) по {manual_price:,.0f} ₽".replace(",", " "))
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))


# =============================================================================
# Подвал — когда было последнее обновление модели
# =============================================================================
st.divider()
sig_date = signal.get("signal_date")
if sig_date is not None:
    try:
        sig_str = pd.to_datetime(sig_date).strftime("%d.%m.%Y")
        st.caption(f"Последнее обновление сигнала: {sig_str} · "
                   f"автообновление 3 раза в день (8:00, 14:00, 22:00 МСК)")
    except Exception:
        pass
