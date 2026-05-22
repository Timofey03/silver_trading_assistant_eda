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
    load_gold_signal,
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
    st.caption(f"🏆 E3b multi-asset · обновлено {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Перезагрузить данные", use_container_width=True,
                 help="Только обновить UI из существующих файлов"):
        st.cache_data.clear()
        st.rerun()

    # Полный refresh с пересчётом модели на свежих данных
    st.markdown("---")
    st.markdown("**🔬 Свежий сигнал**")
    st.caption("Скачивает новые цены → пересчитывает модель → новый прогноз. ~3-5 минут.")
    if st.button("🔬 Refresh signal (полный)", use_container_width=True, type="primary"):
        import subprocess
        import sys as _sys
        import os as _os

        env = _os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        ph = st.empty()
        with st.spinner("⏳ Шаг 1/4: Refresh silver data (yfinance)..."):
            ph.text("Шаг 1/4: Силвер data...")
            r1 = subprocess.run(
                [_sys.executable, "silver_assistant_v22_risk_aware.py", "--no-wf", "--no-mh"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=env, cwd=str(ROOT), timeout=600,
            )
        with st.spinner("⏳ Шаг 2/4: Силвер inference..."):
            ph.text("Шаг 2/4: Силвер inference...")
            r2 = subprocess.run(
                [_sys.executable, "silver_production_inference.py"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=env, cwd=str(ROOT), timeout=300,
            )
        with st.spinner("⏳ Шаг 3/4: Gold data..."):
            ph.text("Шаг 3/4: Gold data...")
            r3 = subprocess.run(
                [_sys.executable, "silver_assistant_v26_multiasset.py", "--fetch"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=env, cwd=str(ROOT), timeout=600,
            )
        with st.spinner("⏳ Шаг 4/4: Gold inference..."):
            ph.text("Шаг 4/4: Gold inference...")
            r4 = subprocess.run(
                [_sys.executable, "gold_production_inference.py"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=env, cwd=str(ROOT), timeout=300,
            )
        ph.empty()

        all_ok = all(r.returncode == 0 for r in [r1, r2, r3, r4])
        if all_ok:
            st.success("✅ Сигналы обновлены на свежих данных")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("❌ Ошибки при refresh:")
            for name, r in [("v22", r1), ("silver inf", r2), ("gold data", r3), ("gold inf", r4)]:
                if r.returncode != 0:
                    st.code(f"{name}: {r.stderr[-500:]}")

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

# Метка источника сигнала
source = sig.get("source", "—")
model_label_map = {
    "e3b_daily":     "🏆 **E3b** (multi-asset + adaptive barriers) — финальная модель диплома",
    "production":    "🟢 **V25 production** — legacy CPCV модель",
    "cpcv_fallback": "🔵 **V25 CPCV fallback** — резервный источник",
    "none":          "⚪ Сигнал не загружен",
}
st.info(f"Источник сигнала: {model_label_map.get(source, source)}")

# Warning если данные устарели
if sig.get("is_stale"):
    st.warning(
        f"⚠ **{sig.get('stale_reason', 'Данные устарели')}**\n\n"
        f"Сигнал показан как HOLD (консервативно). "
        f"В sidebar нажмите **🔬 Refresh signal** чтобы получить актуальный прогноз "
        f"на сегодняшних ценах (~3-5 минут)."
    )

# Бейдж дедупликации (action vs info)
if sig.get("alert_type") == "info" and sig.get("is_repeat"):
    prev_sig = sig.get("previous_signal", "—")
    st.info(
        f"ℹ **Сигнал не изменился** — это повторное уведомление дня (предыдущий: {prev_sig}). "
        f"Если уже отреагировал утром — повторно ничего делать не нужно."
    )
elif sig.get("alert_type") == "action" and sig.get("previous_signal"):
    prev_sig = sig.get("previous_signal", "—")
    new_sig = sig.get("signal", "—")
    st.success(f"📢 **НОВЫЙ СИГНАЛ:** {prev_sig} → **{new_sig}** — действовать сейчас.")

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

**Что делать**: ничего. Модель E3b даёт ~5 сигналов в год — это нормально (селективная стратегия).
Следующий сигнал ожидается когда `p_up` превысит порог.
""")


# =============================================================================
# Equity curve — теперь с переключателем E3b / V25 и BUY/SELL маркерами
# =============================================================================

st.markdown("### 📈 Историческая производительность")

# Переключатель модели
chart_model = st.radio(
    "Какая модель на графике?",
    options=["🏆 E3b (новая, диплом)", "🟢 V25 (legacy)"],
    horizontal=True,
    help=(
        "**E3b** — финальная модель диплома, walk-forward 2015-2025 (10.3 года).\n\n"
        "**V25** — старая production модель, forward test только 2025-2026 (1.3 года). "
        "Compound 7.6x в bull rally математически корректен, но не репрезентативен."
    ),
)

full = load_full_data()

if chart_model.startswith("🏆"):
    # E3b — загружаем из multi_asset
    e3b_trades_path = ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv"
    if e3b_trades_path.exists():
        trades = pd.read_csv(e3b_trades_path)
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        strategy_name = "Стратегия E3b ★"
        # BnH серия за тот же период что E3b
        period_start = trades["entry_date"].min()
        period_end = trades["exit_date"].max()
        if not full.empty and "silver_close" in full.columns:
            bnh = full["silver_close"].loc[period_start:period_end]
        else:
            # Fallback на parquet
            from app.multi_asset.metal_loader import load_single_metal
            silver = load_single_metal("silver")
            bnh = silver["close"].loc[period_start:period_end]
        caption_text = (
            "Сравнение E3b walk-forward (10.3 года, 48 сделок) с Buy & Hold. "
            "🟢 BUY = вход в позицию, 🔴 SELL = выход. Наведите на маркер для деталей."
        )
    else:
        trades = pd.DataFrame()
        bnh = pd.Series(dtype=float)
        strategy_name = "E3b"
        caption_text = "Файл E3b trades не найден. Запустите daily_e3b.py."
else:
    # V25 legacy
    trades = load_trades("forward")
    if not trades.empty and not full.empty:
        fwd = full[full["split"] == "forward"]
        bnh = fwd["silver_close"]
        strategy_name = "Стратегия V25"
        caption_text = (
            "V25 forward test (1.3 года, 24-38 сделок в bull rally). "
            "Compound 7-8x математически корректен — каждая сделка реинвестируется. "
            "В реальной торговле такая доходность не воспроизводится."
        )
    else:
        bnh = pd.Series(dtype=float)
        strategy_name = "V25"
        caption_text = "Данных V25 пока нет."

st.caption(caption_text)

if not trades.empty:
    # Опционально подгружаем реальные Tinkoff ордера
    tinkoff_orders = None
    tk_path = ROOT / "baseline_outputs_v23" / "v23_paper_trading_log.csv"
    if tk_path.exists():
        try:
            tinkoff_orders = pd.read_csv(tk_path)
        except Exception:
            tinkoff_orders = None

    fig = equity_curve(trades, bnh, strategy_name=strategy_name,
                       show_buy_sell_markers=True, tinkoff_orders=tinkoff_orders)
    st.plotly_chart(fig, use_container_width=True)

    # Подытог под графиком
    final_equity = (1 + trades["net_return"].astype(float)).prod()
    n_buys = len(trades)
    n_sells = len(trades)  # каждая сделка имеет ровно один SELL
    period_days = (trades["exit_date"].max() - trades["entry_date"].min()).days
    period_years = period_days / 365.25

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", f"+{(final_equity - 1) * 100:.1f}%",
              help="Compound каждой сделки. В реальной торговле такая доходность "
                   "не воспроизводится из-за market impact и slippage.")
    c2.metric("Период", f"{period_years:.1f} лет")
    c3.metric("🟢 BUY входов (модель)", n_buys)
    c4.metric("🔴 SELL выходов (модель)", n_sells)

    if tinkoff_orders is not None and not tinkoff_orders.empty:
        n_tk = len(tinkoff_orders[tinkoff_orders.get("executed", True) == True])
        st.caption(
            f"💎 На графике также показаны **{n_tk} реальных Tinkoff sandbox ордеров** "
            "(фиолетовые/оранжевые ромбы). Это paper-trading исполнение через биржу — "
            "связывает теоретические сигналы с реальными ордерами."
        )
else:
    st.info("Данных для графика пока недостаточно.")


# =============================================================================
# Quick stats
# =============================================================================

pnl = load_pnl_summary()
dsr = load_dsr_psr()

# =============================================================================
# Gold signal — multi-asset (Session 3)
# =============================================================================

gold_sig = load_gold_signal()
if gold_sig.get("ok"):
    st.markdown("---")
    st.markdown("### 🥇 Gold сигнал на сегодня")

    g_signal = gold_sig.get("signal", "HOLD")
    g_p_up = gold_sig.get("p_up", 0) or 0
    g_emoji = {"BUY": "🟢", "SHORT": "🔴", "SELL": "🔴", "HOLD": "⚪"}.get(g_signal, "❔")

    gc1, gc2, gc3, gc4 = st.columns(4)
    with gc1:
        st.metric(f"{g_emoji} Gold сигнал", g_signal)
    with gc2:
        st.metric("Уверенность (p_up)", f"{g_p_up:.0%}")
    with gc3:
        st.metric("Gold price", f"${gold_sig.get('gold_close', 0):.2f}")
    with gc4:
        cd = gold_sig.get("cooldown_remaining", 0)
        if cd > 0:
            st.metric("Cooldown", f"{cd}d")
        else:
            st.metric("Cooldown", "—")

    g_above = gold_sig.get("above_threshold", False)
    if g_signal == "BUY":
        st.success(f"🟢 Gold BUY сигнал — модель уверена на {g_p_up:.0%}. "
                    "При portfolio strategy paper trading купит GLDRUBF.")
    elif g_signal == "SELL":
        st.error(f"🔴 Gold SELL — p_up={g_p_up:.0%} ниже порога. Закрываем gold позиции.")
    elif g_above:
        st.warning(f"⏳ Gold — модель уверена ({g_p_up:.0%}), но cooldown {gold_sig.get('cooldown_remaining', 0)}d.")
    else:
        st.info(f"⚪ Gold — нет сигнала. p_up={g_p_up:.0%}, порог 49%.")

# =============================================================================
# Multi-asset portfolio results
# =============================================================================

import json as _json
portfolio_path = ROOT / "baseline_outputs_v27" / "portfolio_summary.json"
if portfolio_path.exists():
    portfolio_data = _json.loads(portfolio_path.read_text(encoding="utf-8"))
    fwd = portfolio_data.get("forward", {})

    st.markdown("---")
    st.markdown("### 🪙 Multi-asset результат (Silver + Gold, 50/50)")
    st.caption("Forward 2025+ backtest. Диверсификация снижает риск.")

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.metric("Total return", f"{fwd.get('total_return', 0)*100:+.1f}%",
                  help="Суммарная доходность портфеля")
    with pcol2:
        st.metric("CAGR", f"{fwd.get('cagr', 0)*100:+.1f}%",
                  help="Среднегодовая доходность")
    with pcol3:
        st.metric("Max Drawdown", f"{fwd.get('max_drawdown', 0)*100:.1f}%",
                  help="Глубина худшей просадки")
    with pcol4:
        st.metric("Sharpe", f"{fwd.get('sharpe_ann', 0):.2f}",
                  help="Доходность / риск")

    pc1, pc2 = st.columns(2)
    with pc1:
        sv_final = fwd.get('silver_final', 0.5)
        st.metric("Silver bucket (start 0.5)", f"{sv_final:.3f}",
                  delta=f"{(sv_final/0.5 - 1)*100:+.1f}%")
    with pc2:
        gd_final = fwd.get('gold_final', 0.5)
        st.metric("Gold bucket (start 0.5)", f"{gd_final:.3f}",
                  delta=f"{(gd_final/0.5 - 1)*100:+.1f}%")

    st.caption(f"💎 Silver bucket вырос с 0.5 до {sv_final:.3f} (+{(sv_final/0.5 - 1)*100:.1f}%) · "
                f"🥇 Gold bucket вырос с 0.5 до {gd_final:.3f} (+{(gd_final/0.5 - 1)*100:.1f}%)")

# =============================================================================
# Подсказка про калькулятор
# =============================================================================

st.markdown("---")
st.info("💡 **Хотите рассчитать на какую сумму вкладываться?** "
        "Откройте страницу **🧮 Калькулятор** в левом меню — там пошаговый расчёт "
        "с понятным объяснением сценариев.")


# =============================================================================
# Сводка модели
# =============================================================================

# Финальная сводка — E3b метрики (наша главная модель в дипломе)
st.markdown("---")
st.markdown("### 🏆 Сводка финальной модели E3b")
st.caption("Multi-asset cross-asset + adaptive volatility-scaled barriers + feature selection")

e3b_metrics_path = ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "metrics.json"
if e3b_metrics_path.exists():
    e3b_m = _json.loads(e3b_metrics_path.read_text(encoding="utf-8"))

    em_col1, em_col2, em_col3, em_col4 = st.columns(4)
    with em_col1:
        st.metric("Total return (10.3 года)",
                  f"+{e3b_m.get('total_return', 0)*100:.1f}%",
                  help="Walk-forward 2015-2025, 48 сделок")
    with em_col2:
        st.metric("Sharpe Ratio",
                  f"{e3b_m.get('sharpe', 0):.3f}",
                  delta=f"PSR {e3b_m.get('psr', 0):.2f}",
                  help="Annualized Sharpe + Probabilistic Sharpe Ratio")
    with em_col3:
        st.metric("Win Rate",
                  f"{e3b_m.get('win_rate', 0)*100:.0f}%",
                  delta=f"Profit factor {e3b_m.get('profit_factor', 0):.2f}",
                  help="Доля прибыльных сделок")
    with em_col4:
        st.metric("Max Drawdown",
                  f"{e3b_m.get('max_dd', 0)*100:.1f}%",
                  delta=f"Sortino {e3b_m.get('sortino', 0):.2f}",
                  help="Максимальная просадка + Sortino")

    st.caption(
        f"📊 OOS Accuracy {e3b_m.get('oos_accuracy', 0)*100:.1f}% · "
        f"Annual return {e3b_m.get('annual_return', 0)*100:+.1f}% · "
        f"Best trade {e3b_m.get('best_trade', 0)*100:+.1f}% · "
        f"Worst trade {e3b_m.get('worst_trade', 0)*100:+.1f}%"
    )

# Legacy V25 сводка (свернута в expander для совместимости)
if not pnl.empty:
    with st.expander("📊 Legacy V25 CPCV сводка (для сравнения)"):
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
