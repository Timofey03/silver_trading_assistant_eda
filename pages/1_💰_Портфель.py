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

# Считаем реальный realized P&L (только non-cash positions)
unrealized = 0.0
for p in tinkoff["positions"]:
    if p["instrument_type"] != "currency" and p["qty"] != 0:
        avg = p.get("avg_price", 0)
        cur = p.get("current_price", 0)
        if avg and cur:
            unrealized += (cur - avg) * p["qty"]

# "Реальная" цифра счёта = Cash + unrealized P&L (без раздутого notional)
real_value = tinkoff["cash"]["value"] + unrealized

with col1:
    st.metric(
        "💵 Всего (notional)", rub(tinkoff["total"]["value"]),
        delta=rub(tinkoff["expected_yield"]["value"]),
        help="Tinkoff показывает Cash + НОМИНАЛ futures. Это **не реальные деньги** — "
             "реальное состояние счёта см. в карточке 'Реальный счёт'.",
    )

with col2:
    st.metric("💰 Cash", rub(tinkoff["cash"]["value"]),
              help="Доступные деньги. Можно снять или открыть новые позиции.")

with col3:
    st.metric(
        "📊 Futures (notional)", rub(tinkoff["futures"]["value"]),
        help="**Notional value** ваших фьючерсных позиций. "
             "Это НЕ деньги, а ЭКСПОЗИЦИЯ на рынок: "
             "notional = quote × multiplier × лоты. "
             "Margin (реально заблокированные деньги) ≈ 10-15% от notional.",
    )

with col4:
    st.metric(
        "📈 Реальный счёт", rub(real_value),
        delta=rub(unrealized),
        delta_color="normal" if unrealized >= 0 else "inverse",
        help="Cash + текущий unrealized P&L по открытым позициям. "
             "Это реальная стоимость вашего счёта.",
    )


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
st.markdown("### 🎯 Рекомендации по открытым позициям")

sig = get_current_signal()
open_positions = [p for p in tinkoff["positions"]
                  if p["qty"] != 0 and p["instrument_type"] != "currency"]

if not open_positions:
    st.info("Нет открытых позиций. Помощник ждёт нового BUY-сигнала.")
else:
    p_up = sig.get("p_up", 0.5)
    exit_threshold = 0.45

    if p_up is None:
        st.warning("⚠ Нет актуального p_up для рекомендации.")
    elif p_up < exit_threshold:
        st.error(f"""
🔴 **РЕКОМЕНДАЦИЯ: ЗАКРЫТЬ ПОЗИЦИИ**

p_up = **{p_up:.0%}** упало ниже exit-порога **{exit_threshold:.0%}**.
Модель **уже не уверена в росте** — лучше зафиксировать результат.

- Открытых позиций: **{len(open_positions)}**
- Текущий p_up: **{p_up:.0%}** (нужно ≥ {exit_threshold:.0%})
""")
        if st.button("⛔ Закрыть все LONG позиции"):
            try:
                from silver_paper_tinkoff import TinkoffClient, _load_account_id
                import os
                client = TinkoffClient(os.getenv("TINKOFF_TOKEN"))
                account = _load_account_id(client)
                closed = 0
                for p in open_positions:
                    if p["instrument_type"] == "futures" and p["qty"] > 0:
                        client.sandbox_post_order(
                            account, p["figi"], int(p["qty"]),
                            "ORDER_DIRECTION_SELL",
                        )
                        closed += 1
                st.success(f"✅ Закрыто {closed} позиций")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")
    elif p_up < 0.55:
        st.warning(f"""
⚠ **РЕКОМЕНДАЦИЯ: ДЕРЖАТЬ, но мониторить**

p_up = **{p_up:.0%}** — в нейтральной зоне.
Не пора закрывать, но и не входим новыми.
""")
    else:
        st.success(f"""
✅ **РЕКОМЕНДАЦИЯ: ДЕРЖАТЬ позиции**

p_up = **{p_up:.0%}** — модель **уверена в продолжении роста**.
Trailing stop сам закроет позицию при развороте.
""")


st.markdown("---")
st.markdown("### 📜 История ордеров")
st.info(
    "🤖 **Это РЕАЛЬНЫЕ API-вызовы в Tinkoff sandbox.** "
    "Каждая строка = ордер был отправлен в Tinkoff. Может включать ошибки "
    "(блокировки тикеров, нехватка средств) и успешные исполнения. "
    "Прогнозы модели на каждый день → страница **📊 Сигналы**."
)

log = load_paper_trading_log()
if log.empty:
    st.info("Лог paper trading пустой. Сделки появятся после первого `--replay` или `--live`.")
else:
    # Фильтр + кнопка очистки
    col_filter, col_clear = st.columns([3, 1])
    with col_filter:
        filter_mode = st.radio(
            "Показать", ["Все", "Только сработавшие", "Только ошибки"],
            horizontal=True, label_visibility="collapsed",
        )
    with col_clear:
        if st.button("🗑 Очистить лог", help="Удалить ВСЕ записи paper trading"):
            from pathlib import Path
            log_path = Path("baseline_outputs_v23") / "v23_paper_trading_log.csv"
            try:
                log_path.unlink()
                st.success("✅ Лог очищен")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")

    log_display = log.copy()
    if "ts_signal" in log_display.columns:
        log_display["ts_signal"] = log_display["ts_signal"].dt.strftime("%Y-%m-%d")

    # Применяем фильтр
    if "executed" in log_display.columns:
        if filter_mode == "Только сработавшие":
            log_display = log_display[log_display["executed"] == True]
        elif filter_mode == "Только ошибки":
            log_display = log_display[log_display["executed"] != True]

    show_cols = ["ts_signal", "signal", "ticker", "direction", "lots",
                 "price", "executed", "error"]
    show_cols = [c for c in show_cols if c in log_display.columns]

    st.dataframe(
        log_display[show_cols].iloc[::-1],
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

    n_exec = int(log["executed"].sum()) if "executed" in log.columns else 0
    n_err = len(log) - n_exec
    st.caption(f"Всего: {len(log)} · ✅ Исполнено: {n_exec} · ❌ Ошибок: {n_err} "
               f"· Показано: {len(log_display)}")


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
        import sys as _sys
        import os as _os
        with st.spinner("Запуск daily_run.py..."):
            # Forces UTF-8 в subprocess (Windows default = cp1251 → mojibake)
            env = _os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            r = subprocess.run(
                [_sys.executable, "scripts/daily_run.py", "--skip-training"],
                capture_output=True, text=True, timeout=300,
                cwd=str(ROOT),
                encoding="utf-8", errors="replace",
                env=env,
            )
            if r.returncode == 0:
                st.success("✅ Готово!")
                if r.stdout:
                    with st.expander("📋 Output (последние 2KB)"):
                        st.code(r.stdout[-2000:])
            else:
                st.error(f"Ошибка (rc={r.returncode}):")
                st.code((r.stderr or r.stdout or "")[-1500:])
        st.cache_data.clear()
