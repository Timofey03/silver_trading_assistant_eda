"""Страница: Tinkoff portfolio — live."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils import (
    get_current_signal, get_tinkoff_status, load_paper_trading_log,
    rub, pct, inject_styles, top_signal_badge,
)
from app.charts import portfolio_donut

st.set_page_config(page_title="Портфель", page_icon="💰", layout="wide")
inject_styles()

st.markdown("# 💰 Портфель Tinkoff")

# Топ-сигнал
top_signal_badge(get_current_signal())

tinkoff = get_tinkoff_status()

if not tinkoff.get("ok"):
    st.error(f"❌ Tinkoff недоступен: {tinkoff.get('error', '')}")
    st.info("""
**Что нужно проверить:**
1. Файл `.env` существует и содержит `TINKOFF_TOKEN=...`
2. Sandbox-счёт создан: `python silver_paper_tinkoff.py --setup`
3. Токен валиден (sandbox-only права)
""")
    st.stop()

# =============================================================================
# Header
# =============================================================================

st.caption(f"Sandbox account: `{tinkoff['account_id']}`")

if st.button("🔄 Обновить из Tinkoff", use_container_width=False):
    st.cache_data.clear()
    st.rerun()


# =============================================================================
# Big numbers
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💵 Всего", rub(tinkoff["total"]["value"]),
              delta=rub(tinkoff["expected_yield"]["value"]))

with col2:
    st.metric("💰 Cash", rub(tinkoff["cash"]["value"]))

with col3:
    st.metric("📊 Futures", rub(tinkoff["futures"]["value"]))

with col4:
    n_pos = len([p for p in tinkoff["positions"]
                 if p["qty"] != 0 and p["instrument_type"] != "currency"])
    st.metric("Открытых позиций", n_pos)


# =============================================================================
# Donut chart + positions
# =============================================================================

col_chart, col_table = st.columns([1, 1])

with col_chart:
    fig = portfolio_donut(
        cash=tinkoff["cash"]["value"],
        futures=tinkoff["futures"]["value"],
        shares=tinkoff["shares"]["value"],
        etf=tinkoff["etf"]["value"],
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.markdown("### 📋 Открытые позиции")
    positions = [p for p in tinkoff["positions"] if p["qty"] != 0]
    if not positions:
        st.info("Нет открытых позиций")
    else:
        rows = []
        for p in positions:
            qty = p["qty"]
            avg = p["avg_price"]
            cur = p["current_price"]
            unreal = (cur - avg) * qty if (cur and avg) else 0
            rows.append({
                "Тип":      p["instrument_type"],
                "FIGI":     p["figi"][:15] + "..." if len(p["figi"]) > 15 else p["figi"],
                "Кол-во":   f"{qty:.0f}",
                "Avg цена": f"{avg:.2f}" if avg else "—",
                "Текущая":  f"{cur:.2f}" if cur else "—",
                "P&L":      f"{unreal:+.2f} ₽" if unreal else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# Paper trading log (история сделок)
# =============================================================================

st.markdown("---")
st.markdown("### 📜 История ордеров")

log = load_paper_trading_log()
if log.empty:
    st.info("Лог paper trading пустой. Сделки появятся после первого `--replay` или `--live`.")
else:
    log_display = log.copy()
    if "ts_signal" in log_display.columns:
        log_display["ts_signal"] = log_display["ts_signal"].dt.strftime("%Y-%m-%d")

    # Раскрашиваем по signal
    show_cols = ["ts_signal", "signal", "ticker", "direction", "lots",
                 "price", "executed", "error"]
    show_cols = [c for c in show_cols if c in log_display.columns]

    st.dataframe(
        log_display[show_cols].iloc[::-1],   # последние сверху
        use_container_width=True, hide_index=True,
        column_config={
            "ts_signal": "Дата сигнала",
            "signal":    "Сигнал",
            "ticker":    "Тикер",
            "direction": "Направление",
            "lots":      "Лоты",
            "price":     "Цена",
            "executed":  st.column_config.CheckboxColumn("Исполнен"),
            "error":     "Ошибка",
        },
    )

    st.caption(f"Всего записей: {len(log)} · "
               f"Исполнено: {log['executed'].sum() if 'executed' in log.columns else '—'}")


# =============================================================================
# Actions
# =============================================================================

st.markdown("---")
st.markdown("### ⚡ Действия")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Закрыть все позиции SLVRUBF**")
    st.caption("Полезно перед тестированием стратегии")
    if st.button("🛑 Sell all SLVRUBF", type="secondary"):
        st.warning("Функция доступна в CLI: `python silver_paper_tinkoff.py` "
                   "с custom скриптом")
with col2:
    st.markdown("**Пополнить sandbox**")
    st.caption("Добавить виртуальные RUB")
    amount = st.number_input("RUB", min_value=1000, value=100000, step=10000)
    if st.button("💵 Pay-in"):
        try:
            import os
            from silver_paper_tinkoff import TinkoffClient, _load_account_id
            client = TinkoffClient(os.getenv("TINKOFF_TOKEN"))
            account = _load_account_id(client)
            client.sandbox_pay_in(account, amount)
            st.success(f"✅ Зачислено {rub(amount)}")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка: {e}")
with col3:
    st.markdown("**Запустить daily run**")
    st.caption("Полный цикл: retrain + paper trade")
    if st.button("🚀 Run daily"):
        import subprocess
        with st.spinner("Запуск daily_run.py..."):
            r = subprocess.run(
                ["python", "scripts/daily_run.py", "--skip-training"],
                capture_output=True, text=True, timeout=180,
            )
            if r.returncode == 0:
                st.success("✅ Готово!")
            else:
                st.error(f"Ошибка: {r.stderr[-500:]}")
        st.cache_data.clear()
