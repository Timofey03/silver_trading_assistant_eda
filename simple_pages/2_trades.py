"""Страница: 💼 Мои сделки — история пользовательских сделок + бэктест."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils import load_trades
from app.simple_storage import get_all_trades, get_capital, remove_trade


st.title("💼 Мои сделки")

capital = get_capital()
user_trades = get_all_trades()

tab_user, tab_backtest = st.tabs([
    f"🟢 Мои сделки ({len(user_trades)})",
    "📚 История на бэктесте",
])


# =============================================================================
# Вкладка 1: пользовательские сделки
# =============================================================================
with tab_user:
    if not user_trades:
        st.info(
            "Здесь будет история ваших сделок. Чтобы добавить первую — "
            "перейдите на **📍 Сейчас** и нажмите «Записать», когда "
            "появится сигнал на покупку."
        )
    else:
        # Формируем таблицу
        rows = []
        for t in user_trades:
            entry = t.get("entry_price", 0)
            exit_p = t.get("exit_price")
            lots = t.get("lots", 1)
            status = t.get("status", "")

            if status == "closed" and exit_p:
                pnl_pct = (exit_p / entry - 1) * 100
                pnl_rub = (exit_p - entry) * 100 * lots  # SLVRUBF mult=100
            elif status == "open":
                pnl_pct = None
                pnl_rub = None
            else:
                pnl_pct = None
                pnl_rub = None

            rows.append({
                "ID":           t["id"],
                "Открыто":      t.get("entry_date", ""),
                "Цена входа":   f"{entry:,.0f} ₽".replace(",", " "),
                "Лотов":        lots,
                "Закрыто":      t.get("exit_date", "—"),
                "Цена выхода":  f"{exit_p:,.0f} ₽".replace(",", " ") if exit_p else "—",
                "Прибыль %":    f"{pnl_pct:+.2f}%" if pnl_pct is not None else "в работе",
                "Прибыль ₽":    f"{pnl_rub:+,.0f}".replace(",", " ") if pnl_rub is not None else "—",
                "Статус":       "🟢 открыта" if status == "open" else "✓ закрыта",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Сводка
        closed = [t for t in user_trades if t.get("status") == "closed"]
        if closed:
            st.divider()
            st.markdown("### 📊 Итог по закрытым сделкам")
            total_pct = sum(t["pnl_pct"] for t in closed)
            total_rub = sum(
                (t["exit_price"] - t["entry_price"]) * 100 * t.get("lots", 1)
                for t in closed
            )
            wins = sum(1 for t in closed if t["pnl_pct"] > 0)
            win_rate = wins / len(closed) * 100

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Закрытых сделок", len(closed))
            with c2:
                st.metric("Сумма прибыли", f"{total_pct:+.2f}%")
            with c3:
                st.metric("В рублях", f"{total_rub:+,.0f} ₽".replace(",", " "))
            with c4:
                st.metric("Прибыльных", f"{wins}/{len(closed)} ({win_rate:.0f}%)")

            if capital > 0:
                pnl_of_capital = (total_rub / capital) * 100
                st.success(
                    f"💰 Это **{pnl_of_capital:+.2f}%** от вашего капитала "
                    f"({capital:,.0f} ₽)".replace(",", " ")
                )

        # Удаление сделки
        st.divider()
        with st.expander("⚠ Удалить запись о сделке (например, если ошиблись)"):
            if user_trades:
                ids = [t["id"] for t in user_trades]
                to_remove = st.selectbox("ID сделки", ids)
                if st.button("Удалить навсегда", type="secondary"):
                    if remove_trade(to_remove):
                        st.success("Удалено")
                        st.rerun()


# =============================================================================
# Вкладка 2: бэктест (что было бы если бы все сигналы исполнялись)
# =============================================================================
with tab_backtest:
    # Источник данных — переключатель моделей
    source = st.radio(
        "Какую модель показать?",
        options=[
            "🏆 E3b — новая модель диплома (10.3 года walk-forward)",
            "🟢 V25 — текущая production (1.3 года 2025-2026)",
            "🔵 Базовая walk-forward — 8 лет старой модели",
        ],
        horizontal=False,
        help=(
            "**E3b** — финальная модель диплома, прошла walk-forward валидацию на 10+ лет.\n\n"
            "**V25** — текущая production-модель в Streamlit (тестировалась только на 1.3 года "
            "экстремального бычьего рынка).\n\n"
            "**Базовая** — walk-forward 8 лет старой OptimalV2."
        ),
    )

    if source.startswith("🏆"):
        # E3b — загружаем напрямую из multi_asset
        e3b_path = ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv"
        if e3b_path.exists():
            trades = pd.read_csv(e3b_path)
            trades["entry_date"] = pd.to_datetime(trades["entry_date"])
            trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        else:
            trades = pd.DataFrame()
        period_label = "10.3 года walk-forward (2015–2025)"
        period_short = "10.3 года"
        st.markdown(
            f"Все сделки финальной модели E3b за **{period_label}**. "
            "Cross-asset 5 металлов + adaptive barriers. "
            "🟢 — прибыльные, 🔴 — убыточные."
        )
    elif source.startswith("🟢"):
        trades = load_trades("forward")
        period_label = "1.3 года forward-test (2025-2026)"
        period_short = "1.3 года"
        st.markdown(
            f"Сделки V25 production за **{period_label}** — период экстремального роста серебра "
            "(silver +136% YoY). ⚠ Результаты не репрезентативны для обычных рынков."
        )
    else:
        # Базовая walk-forward 8 лет
        wf_path = ROOT / "baseline_outputs_walkforward" / "trades_all.csv"
        if wf_path.exists():
            trades = pd.read_csv(wf_path)
            trades["entry_date"] = pd.to_datetime(trades["entry_date"])
            trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        else:
            trades = pd.DataFrame()
        period_label = "8 лет walk-forward (2018–2025)"
        period_short = "8 лет"
        st.markdown(
            f"Сделки старой базовой модели OptimalV2 за **{period_label}**. "
            "Результат отрицательный — модель оставлена для академического сравнения."
        )

    if trades.empty:
        st.warning(f"Файл сделок для выбранной модели не найден.")
    else:
        # Найдём колонку с доходностью (схема может отличаться)
        pnl_col = next(
            (c for c in ("pnl_pct", "pnl_pct_net", "net_return", "gross_return")
             if c in trades.columns),
            None,
        )
        if pnl_col is None:
            st.error("Колонка доходности не найдена в файле trades.")
            st.stop()

        pnl_pct_series = (trades[pnl_col] * 100).round(2)

        view = pd.DataFrame({
            "Открыто":      pd.to_datetime(trades["entry_date"]).dt.strftime("%d.%m.%Y"),
            "Цена входа":   trades["entry_price"].round(2),
            "Закрыто":      pd.to_datetime(trades["exit_date"]).dt.strftime("%d.%m.%Y"),
            "Цена выхода":  trades["exit_price"].round(2),
            "Дней":         (pd.to_datetime(trades["exit_date"])
                             - pd.to_datetime(trades["entry_date"])).dt.days,
            "Прибыль %":    pnl_pct_series,
        })

        # Цветная подсветка по знаку прибыли
        def color_row(row):
            v = row["Прибыль %"]
            if pd.isna(v):
                return [""] * len(row)
            if v > 0:
                return ["background-color: #E8F5E9"] * len(row)
            elif v < 0:
                return ["background-color: #FFEBEE"] * len(row)
            return [""] * len(row)

        st.dataframe(
            view.style.apply(color_row, axis=1)
                      .format({"Прибыль %": "{:+.2f}%"}),
            use_container_width=True, hide_index=True, height=420,
        )

        # Сводка
        st.divider()
        st.markdown(f"### 📊 Итог за {period_short}")
        total_pct = (1 + trades[pnl_col]).prod() - 1
        wins = (trades[pnl_col] > 0).sum()
        win_rate = wins / len(trades) * 100 if len(trades) else 0
        avg = trades[pnl_col].mean() * 100
        mean_win = trades[pnl_col][trades[pnl_col] > 0].mean() * 100 if wins else 0
        mean_loss = trades[pnl_col][trades[pnl_col] <= 0].mean() * 100 if (len(trades) - wins) else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Всего сделок", len(trades))
        with c2:
            st.metric("Итоговая прибыль", f"{total_pct*100:+.1f}%")
        with c3:
            st.metric("Прибыльных", f"{wins}/{len(trades)} ({win_rate:.0f}%)")
        with c4:
            st.metric("В среднем за сделку", f"{avg:+.2f}%",
                      help=f"Формула: {win_rate:.0f}% × {mean_win:+.1f}% + "
                           f"{100-win_rate:.0f}% × {mean_loss:+.1f}% = {avg:+.2f}%")
