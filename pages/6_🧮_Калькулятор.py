"""
Страница: 🧮 Калькулятор для обычного пользователя

3-step wizard: Сбережения → Размер ставки → Результат + Apply/View.

Без жаргона. Все тех. термины спрятаны в expanders.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils import (
    get_current_signal, get_tinkoff_status,
    inject_styles, top_signal_badge, rub,
)

CONFIG_PATH = ROOT / "baseline_outputs_prod" / "user_trading_config.json"

# Константы стратегии (захардкожены — Optimal mode из grid search)
LOT_NOTIONAL_RUB = 20_000     # SLVRUBF: 1 лот ≈ 20k RUB notional
STOP_PCT         = 0.08       # trailing stop 8% (из Optimal mode)

# Стратегии: silver-only vs portfolio (silver + gold)
STRATEGIES = {
    "silver_only": {
        "name":             "🥈 Только серебро",
        "description":      "Только SLVRUBF. Максимальная доходность на forward, но и DD больше.",
        "trades_per_year":  11,
        "win_rate":         0.64,
        "scenarios": [
            {"label": "🟢 Хороший день",     "pct": 32, "ret": +0.15, "color": "#00C853"},
            {"label": "🟢 Обычная прибыль",  "pct": 32, "ret": +0.06, "color": "#43A047"},
            {"label": "🟡 Около нуля",       "pct": 10, "ret": -0.005, "color": "#9E9E9E"},
            {"label": "🔴 Небольшой минус",  "pct": 18, "ret": -0.025, "color": "#EF5350"},
            {"label": "🔴 Сработал стоп",    "pct":  8, "ret": -0.080, "color": "#D32F2F"},
        ],
    },
    "portfolio": {
        "name":             "🪙 Портфель (Silver + Gold 50/50)",
        "description":      "Диверсификация: 50% silver + 50% gold. Меньше DD, более стабильно.",
        "trades_per_year":  21,   # 11 silver + 10 gold
        "win_rate":         0.66,
        "scenarios": [
            # Более стабильные сценарии — diversified
            {"label": "🟢 Хороший день",     "pct": 30, "ret": +0.12, "color": "#00C853"},
            {"label": "🟢 Обычная прибыль",  "pct": 36, "ret": +0.05, "color": "#43A047"},
            {"label": "🟡 Около нуля",       "pct": 14, "ret": -0.005, "color": "#9E9E9E"},
            {"label": "🔴 Небольшой минус",  "pct": 14, "ret": -0.025, "color": "#EF5350"},
            {"label": "🔴 Сработал стоп",    "pct":  6, "ret": -0.075, "color": "#D32F2F"},
        ],
    },
}

# Дефолтная стратегия
DEFAULT_STRATEGY = "silver_only"


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(page_title="Калькулятор", page_icon="🧮", layout="wide")
inject_styles()

# Init session state
defaults = {
    "calc_step":          1,
    "calc_savings":       1_000_000,
    "calc_allocation":    30,           # 30% — реальный рекомендуемый уровень
    "calc_risk_level":    "medium",
    "calc_strategy":      DEFAULT_STRATEGY,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# HELPERS
# =============================================================================

def get_active_scenarios() -> list:
    strat = st.session_state.get("calc_strategy", DEFAULT_STRATEGY)
    return STRATEGIES[strat]["scenarios"]


def get_expected_return_per_trade() -> float:
    scenarios = get_active_scenarios()
    return sum(s["pct"] / 100 * s["ret"] for s in scenarios)


def get_trades_per_year() -> int:
    strat = st.session_state.get("calc_strategy", DEFAULT_STRATEGY)
    return STRATEGIES[strat]["trades_per_year"]


def compute_position(savings: float, allocation_pct: int, risk_key: str) -> dict:
    """
    Главный расчёт. Возвращает все числа для отображения.

    Логика:
      1. allocation_rub = savings × allocation_pct% — макс что в серебре
      2. max_loss_rub = savings × risk_pct% — макс потеря на сделку
      3. position_by_risk = max_loss / stop_pct — макс позиция по риску
      4. position_actual = min(position_by_risk, allocation_rub)
      5. lots = floor(position / lot_notional), минимум 1 если положительная сумма
      6. actual_position = lots × lot_notional
      7. actual_loss = actual_position × stop_pct
    """
    risk_pct = {"small": 0.002, "medium": 0.015, "large": 0.05}[risk_key]

    allocation_rub = savings * allocation_pct / 100
    max_loss = savings * risk_pct

    position_by_risk = max_loss / STOP_PCT
    position_actual = min(position_by_risk, allocation_rub)
    lots = max(0, int(position_actual / LOT_NOTIONAL_RUB))
    if lots == 0 and position_actual >= LOT_NOTIONAL_RUB * 0.5:
        lots = 1

    actual_position = lots * LOT_NOTIONAL_RUB
    actual_loss = actual_position * STOP_PCT

    # Годовая доходность (per backtest scenarios)
    expected_ret = get_expected_return_per_trade()
    trades_yr    = get_trades_per_year()
    annual_expected = actual_position * expected_ret * trades_yr
    annual_low = annual_expected * 0.5   # пессимистично
    annual_high = annual_expected * 1.5  # оптимистично

    # Capped реальный risk %
    actual_risk_pct = (actual_loss / savings * 100) if savings > 0 else 0

    return {
        "risk_key":         risk_key,
        "risk_pct_chosen":  risk_pct * 100,
        "max_loss":         max_loss,
        "allocation_rub":   allocation_rub,
        "lots":             lots,
        "actual_position":  actual_position,
        "actual_loss":      actual_loss,
        "actual_risk_pct":  actual_risk_pct,
        "annual_expected":  annual_expected,
        "annual_low":       annual_low,
        "annual_high":      annual_high,
        "capped_by_allocation": position_by_risk > allocation_rub,
    }


def render_progress(current_step: int) -> None:
    dots = []
    labels = ["💰 Деньги", "🎯 Размер ставки", "✅ План"]
    for i in range(1, 4):
        if i < current_step:
            dots.append(f"✅ {labels[i-1]}")
        elif i == current_step:
            dots.append(f"**🔵 Шаг {i}: {labels[i-1]}**")
        else:
            dots.append(f"⚪ {labels[i-1]}")
    st.markdown("   →   ".join(dots))


# =============================================================================
# STEP 1: Сбережения и доля
# =============================================================================

def show_step_1() -> None:
    st.markdown("## 💰 Расскажите про ваши сбережения")
    st.caption("Нужно для расчёта **безопасного** размера ставки")

    # Выбор стратегии (silver-only / portfolio)
    st.markdown("### 0. Какую стратегию использовать?")
    strat_labels = {k: v["name"] for k, v in STRATEGIES.items()}
    strat_descrs = {k: v["description"] for k, v in STRATEGIES.items()}

    cur_strat = st.session_state.get("calc_strategy", DEFAULT_STRATEGY)
    keys = list(STRATEGIES.keys())
    selected_strat = st.radio(
        "Тип стратегии",
        options=keys,
        format_func=lambda k: strat_labels[k],
        index=keys.index(cur_strat),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.calc_strategy = selected_strat
    st.caption(strat_descrs[selected_strat])

    if selected_strat == "portfolio":
        st.success("✨ Multi-asset режим: капитал делится 50/50 между silver и gold. "
                    "Меньше волатильности, более стабильный рост.")

    st.markdown("### 1. Сколько у вас свободных сбережений?")

    tinkoff = get_tinkoff_status()
    if tinkoff.get("ok"):
        suggested = int(tinkoff["cash"]["value"] + tinkoff["futures"]["value"])
        st.caption(f"💡 По данным Tinkoff sandbox, у вас на счёте {rub(suggested)}")

    savings = st.number_input(
        "Введите сумму в рублях",
        value=int(st.session_state.calc_savings),
        min_value=50_000, step=10_000,
        format="%d",
        label_visibility="collapsed",
    )
    st.session_state.calc_savings = savings

    st.warning(
        "⚠ **Важно**: НЕ включайте сюда «подушку безопасности» "
        "(3-6 месяцев расходов). Только те деньги, которые **не критично потерять**."
    )

    st.markdown("### 2. Какую часть готовы попробовать в серебре?")

    options = [
        (10, "10% — осторожно",                                 False),
        (30, "30% — стандарт для торгуемого ML-помощника",      True),  # рекомендуется
        (50, "50% — большая ставка",                            False),
        (80, "80% — почти всё (⚠ рискованно)",                 False),
    ]

    current = st.session_state.calc_allocation
    # Если в session_state старое значение (20 или 5) — мапим на новое
    if current not in [v for v, _, _ in options]:
        current = 30 if current == 20 else 10 if current == 5 else current
        st.session_state.calc_allocation = current
    idx = next((i for i, (v, _, _) in enumerate(options) if v == current), 1)

    def format_option(opt):
        v, label, recommended = opt
        sum_rub = rub(savings * v / 100)
        star = " ⭐" if recommended else ""
        return f"{label} → {sum_rub}{star}"

    selected_opt = st.radio(
        "Доля сбережений в серебре",
        options=options,
        index=idx,
        format_func=format_option,
        label_visibility="collapsed",
    )
    st.session_state.calc_allocation = selected_opt[0]

    if selected_opt[0] == 80:
        st.error(
            "⚠ **80% в одном классе активов — слишком сильная концентрация.**\n\n"
            "Если silver просядет на 20% — потеряете 16% всех сбережений. "
            "Рекомендую 30%."
        )
    elif selected_opt[0] == 50:
        st.warning(
            "💡 50% — серьёзная ставка. Подходит если вы уверены в долгосрочном "
            "росте металлов. При сильной просадке (-15%) потеряете 7.5% сбережений."
        )

    # Info про compounding
    st.success(
        "🔄 **Compounding включён**: размер позиции пересчитывается **от текущего** "
        "баланса каждый раз. Когда счёт растёт — ставки тоже растут пропорционально."
    )

    with st.expander("🔍 Тех. детали: откуда берётся эта рекомендация?"):
        st.markdown("""
**Принцип диверсификации портфеля** (Markowitz Modern Portfolio Theory):
- Распределение между разными классами активов снижает совокупный риск
- Для металлов типичная аллокация в портфеле: 5-15% (золото), 2-5% (серебро)
- Для агрессивной стратегии на металлы (наша): 20-30% оправдано
- Свыше 50% — нарушение принципа диверсификации

**Полная классификация активов**:
- 🟢 Cash, депозиты (low risk, low return)
- 🟡 Облигации (medium)
- 🟠 Акции (medium-high)
- 🔴 Сырьё/металлы (high) — **сюда серебро**
- 🔴 Криптовалюты (very high)
""")

    st.markdown("---")
    col_l, col_r = st.columns([1, 1])
    with col_r:
        if st.button("Далее →", use_container_width=True, type="primary"):
            st.session_state.calc_step = 2
            st.rerun()


# =============================================================================
# STEP 2: Размер ставки
# =============================================================================

def show_step_2() -> None:
    savings = st.session_state.calc_savings
    allocation_pct = st.session_state.calc_allocation
    allocation_rub = savings * allocation_pct / 100

    st.markdown("## 🎯 Какой размер ставки вам комфортен?")
    st.caption(f"У вас в серебре: **{rub(allocation_rub)}** ({allocation_pct}% сбережений)")

    st.markdown(
        "Помощник делает **~11 сделок в год**, выигрывает **64%** из них. "
        "Выберите сколько РУБЛЕЙ готовы потерять на ОДНОЙ проигрышной сделке:"
    )

    # Рассчитываем все 3 варианта
    results = {k: compute_position(savings, allocation_pct, k)
               for k in ["small", "medium", "large"]}

    cards = [
        ("small", "🟢 Маленькая ставка", "осторожное накопление", False),
        ("medium", "🟡 Средняя ставка", "разумная инвестиция", True),
        ("large", "🔴 Большая ставка", "ставка на крупное движение", False),
    ]

    current = st.session_state.calc_risk_level

    for key, name, analogy, recommended in cards:
        r = results[key]
        star = " ⭐ Рекомендуется" if recommended else ""
        selected = (key == current)
        border_color = "#00BCD4" if selected else "rgba(255,255,255,0.1)"

        st.markdown(f"""
        <div style="border: 2px solid {border_color}; border-radius: 12px;
                    padding: 16px; margin-bottom: 12px;">
            <h4 style="margin: 0;">{name}{star}</h4>
            <p style="margin: 8px 0; opacity: 0.85;">
                Похоже на: <i>{analogy}</i>
            </p>
            <table style="width: 100%; margin-top: 12px;">
                <tr>
                    <td style="opacity: 0.7;">Размер позиции:</td>
                    <td><b>{r['lots']} лотов = {rub(r['actual_position'])}</b></td>
                </tr>
                <tr>
                    <td style="opacity: 0.7;">Макс убыток на сделку:</td>
                    <td><b>{rub(r['actual_loss'])}</b> ({r['actual_risk_pct']:.1f}% сбережений)</td>
                </tr>
                <tr>
                    <td style="opacity: 0.7;">Ожидаемая годовая доходность:</td>
                    <td><b>{rub(r['annual_low'])} — {rub(r['annual_high'])}</b></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    selected_key = st.radio(
        "Выберите вариант",
        options=["small", "medium", "large"],
        format_func=lambda k: {"small": "🟢 Маленькая",
                                "medium": "🟡 Средняя (рекомендуется)",
                                "large": "🔴 Большая"}[k],
        index=["small", "medium", "large"].index(current),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.calc_risk_level = selected_key

    if selected_key == "large":
        st.warning(
            "⚠ Возможны просадки −20% капитала и больше при серии проигрышей. "
            "Подходит только если психологически готовы к большим колебаниям."
        )

    with st.expander("🔍 Тех. детали: что такое stop-loss и почему 1-3% — стандарт"):
        st.markdown(f"""
**Risk-based sizing — стандарт профессионального трейдинга**:

1. **Trailing stop** (плавающая защита): когда цена падает на {STOP_PCT*100:.0f}%
   от пика — позиция автоматически закрывается. Это гарантирует что вы не потеряете
   больше выбранного лимита.

2. **Risk per trade**: % капитала который теряете если стоп сработал:
   - 0.2% (Маленькая) — очень осторожно
   - 1.5% (Средняя) — стандарт профи
   - 5% (Большая) — повышенный риск

3. **Формула размера позиции**:
   `position = (savings × risk%) / stop%`

4. **Правило 10 проигрышей подряд** (худший сценарий):
   - При 1.5% риске = −15% drawdown (восстановимо)
   - При 5% риске = −40% drawdown (психологически тяжело)
""")

    st.markdown("---")
    col_l, col_r = st.columns([1, 1])
    with col_l:
        if st.button("← Назад", use_container_width=True):
            st.session_state.calc_step = 1
            st.rerun()
    with col_r:
        if st.button("Далее →", use_container_width=True, type="primary"):
            st.session_state.calc_step = 3
            st.rerun()


# =============================================================================
# STEP 3: Результат
# =============================================================================

def show_step_3() -> None:
    savings = st.session_state.calc_savings
    allocation_pct = st.session_state.calc_allocation
    risk_key = st.session_state.calc_risk_level

    r = compute_position(savings, allocation_pct, risk_key)

    # Главная карточка
    st.markdown(f"""
    <div style="text-align: center; padding: 30px;
                background: linear-gradient(135deg, #00BCD4, #0097A7);
                border-radius: 16px; color: white; margin-bottom: 24px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);">
        <p style="font-size: 20px; margin: 0; opacity: 0.9;">
            КОГДА ПРИДЁТ СИГНАЛ:
        </p>
        <p style="font-size: 42px; margin: 8px 0; font-weight: 700;">
            КУПИТЬ {r['lots']} ЛОТОВ SLVRUBF
        </p>
        <p style="font-size: 17px; margin: 8px 0;">
            Серебряные фьючерсы на Мосбирже
        </p>
        <p style="font-size: 28px; margin: 12px 0 0; font-weight: 600;">
            💰 {rub(r['actual_position'])}
        </p>
        <p style="font-size: 15px; opacity: 0.85; margin: 4px 0 0;">
            ({allocation_pct}% ваших сбережений · риск {r['actual_risk_pct']:.1f}%)
        </p>
    </div>
    """, unsafe_allow_html=True)

    if r['lots'] == 0:
        st.error(
            f"❌ Размер позиции = 0 лотов. Вероятно слишком маленькие сбережения "
            f"({rub(savings)}) или слишком жёсткий лимит риска.\n\n"
            f"Минимальный размер 1 лот SLVRUBF = {rub(LOT_NOTIONAL_RUB)}, "
            f"требует риска ~{rub(LOT_NOTIONAL_RUB * STOP_PCT)} (макс убыток на сделку). "
        )
        if st.button("← Изменить настройки"):
            st.session_state.calc_step = 1
            st.rerun()
        return

    if r['capped_by_allocation']:
        st.info(
            "💡 Размер позиции ограничен **долей сбережений** в серебре "
            "(шаг 1), а не риском. Если хотите больше — увеличьте долю."
        )

    # 5 сценариев
    st.markdown("### 📊 Что может произойти за 30 дней")
    strat_name = STRATEGIES[st.session_state.calc_strategy]["name"]
    st.caption(f"Стратегия: **{strat_name}**. На основе backtest за 2025+ год, "
                "**округлённо «как 100 наблюдений»** для наглядности")

    scenarios = get_active_scenarios()
    pos = r['actual_position']
    scenario_rows = []
    for s in scenarios:
        pnl_rub = pos * s["ret"]
        pnl_pct_savings = pnl_rub / savings * 100
        scenario_rows.append({
            "Сценарий":      s["label"],
            "Частота":       f"{s['pct']}%",
            "P&L в ₽":       f"{pnl_rub:+,.0f} ₽".replace(",", " "),
            "% от сбережений": f"{pnl_pct_savings:+.2f}%",
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, use_container_width=True)

    # Bar chart
    fig = go.Figure()
    for s in scenarios:
        pnl = pos * s["ret"]
        fig.add_trace(go.Bar(
            x=[s["label"]], y=[pnl],
            text=[f"{s['pct']}%<br>{pnl:+,.0f} ₽".replace(",", " ")],
            textposition="auto",
            marker_color=s["color"],
            showlegend=False,
            hovertemplate=f"{s['label']}<br>Частота: {s['pct']}%<br>P&L: {pnl:+,.0f} ₽<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.3)
    fig.update_layout(
        title="P&L по сценариям (одна сделка)",
        yaxis_title="P&L (₽)",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    expected_ret = get_expected_return_per_trade()
    trades_yr = get_trades_per_year()
    with col_a:
        st.metric("Ожидаемая годовая доходность",
                  f"{rub(r['annual_expected'])}",
                  help=f"При {trades_yr} сделках/год и среднем "
                       f"{expected_ret*100:.2f}% на сделку")
    with col_b:
        st.metric("В % от сбережений",
                  f"{r['annual_expected']/savings*100:+.1f}%")

    # Защита
    st.markdown("---")
    st.markdown("### 🛡 Как помощник защищает ваши деньги")
    st.markdown(f"""
- ✅ **Автоматический стоп-лосс** при падении на {STOP_PCT*100:.0f}% от пика
  → ваш максимальный убыток на сделку = **{rub(r['actual_loss'])}**
- ✅ **Модель закрывает позицию при развороте** — не надо следить за рынком
- ✅ **Не больше 1 позиции одновременно** — весь капитал не «висит» в одной сделке
""")

    # Когда покупать
    st.markdown("---")
    st.markdown("### 📅 Когда покупать?")
    sig = get_current_signal()
    sig_type = sig.get("signal", "HOLD")
    p_up = sig.get("p_up", 0) or 0

    if sig_type == "BUY":
        st.success(f"""
🟢 **СЕЙЧАС! Сигнал на покупку**

Модель уверена на **{p_up:.0%}** — выше порога 49%.
Если применить настройки — paper trading купит сегодня же.
""")
    elif sig_type == "SELL":
        st.error("🔴 Сейчас сигнал на ВЫХОД, а не вход. Подождите следующий BUY.")
    else:
        cooldown = sig.get("cooldown_remaining", 0)
        if cooldown > 0:
            st.warning(f"""
⏳ **Скоро будет сигнал — ~{cooldown} торговых дней**

Модель уверена ({p_up:.0%}), но **cooldown ещё активен** —
помощник не входит в новую сделку слишком часто.

Через {cooldown} дней при сохранении уверенности — автоматически купит.
""")
        else:
            st.info(f"""
⚪ **Сейчас сигнала нет**

Уверенность модели: **{p_up:.0%}** (нужно ≥49%).
Сигнал ожидается через 3-7 дней при росте уверенности.

Когда придёт — paper trading автоматически купит **{r['lots']} лотов**.
""")

    # Действия
    st.markdown("---")
    st.markdown("### ⚡ Что делать дальше?")

    col_back, col_view, col_apply = st.columns([1, 1.5, 1.5])

    with col_back:
        if st.button("← Назад", use_container_width=True):
            st.session_state.calc_step = 2
            st.rerun()

    with col_view:
        if st.button("📊 Посмотреть прошлые сделки", use_container_width=True,
                     help="Открыть страницу с историей сигналов и сделок"):
            st.switch_page("pages/2_📊_Сигналы.py")

    with col_apply:
        if st.button("💾 Применить для будущих сделок",
                     use_container_width=True, type="primary",
                     help="Сохранить настройки. Daily run будет использовать "
                          "эти параметры для размещения ордеров"):
            apply_settings(r)

    # Тех детали
    with st.expander("🔍 Тех. детали: формула расчёта, p_up, Sharpe, notional"):
        st.markdown(f"""
**Параметры расчёта** (зашиты, не настраиваются):
- Тикер: SLVRUBF (MOEX silver futures)
- Лот: 1 контракт = 100 oz × текущая цена ≈ {rub(LOT_NOTIONAL_RUB)} notional
- Trailing stop: {STOP_PCT*100:.0f}%
- Maximum holding period: 30 дней

**Текущий сигнал модели**:
- p_up (уверенность UP): {p_up:.4f}
- Источник: {sig.get('source', 'unknown')}
- Дата сигнала: {sig.get('signal_date', 'n/a')}

**Backtest статистика (forward 2025+)**:
- Сделок: 11, Win rate: 64%, Sharpe: 1.69
- Total return: +64.5%, Max DD: −8%

**Формула вашей позиции**:
- savings × risk% = {rub(savings)} × {r['risk_pct_chosen']:.1f}% = {rub(r['max_loss'])}
- max_position_by_risk = {rub(r['max_loss'])} / {STOP_PCT:.2f} = {rub(r['max_loss']/STOP_PCT)}
- max_allocation = savings × {allocation_pct}% = {rub(savings*allocation_pct/100)}
- actual_position = min(by_risk, by_allocation) = {rub(r['actual_position'])}
- lots = floor(actual_position / lot_notional) = {r['lots']}
- actual_loss_on_stop = actual_position × {STOP_PCT:.2f} = {rub(r['actual_loss'])}
""")

    # FAQ
    with st.expander("❓ Частые вопросы"):
        st.markdown("""
**Что такое лот?**
Один лот SLVRUBF = ~20,000 ₽. Это «упаковка» серебра на бирже —
наименьшая единица, которую можно купить.

**Где это всё происходит?**
На Московской бирже через брокера Тинькофф. У вас sandbox-счёт —
бумажная торговля, реальные деньги не задействованы.

**Это безопасно?**
Sandbox абсолютно безопасен — это симулятор. Реальные деньги подключаются
только если ВЫ САМИ переключите режим в production (рекомендуется минимум
6 месяцев успешных тестов).

**Как я узнаю когда покупать?**
Никак — помощник сделает всё сам. Daily run каждый рабочий день
проверяет сигнал и торгует в sandbox. Можно настроить Telegram-уведомления.

**А если я хочу остановить торговлю?**
В Портфеле кнопка «Закрыть все позиции». Также можно изменить долю
сбережений в шаге 1 на 0% и применить — новых сделок не будет.

**Что значит «уверенность модели»?**
Это число от 0 до 1 — вероятность роста серебра в ближайшие 15 дней
по ML-модели. Помощник входит когда >49%, выходит когда <43%.
""")


# =============================================================================
# APPLY ACTION
# =============================================================================

def apply_settings(r: dict) -> None:
    """Сохраняет настройки в user_trading_config.json"""
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    config = {
        "savings_rub":            float(st.session_state.calc_savings),
        "allocation_pct":         int(st.session_state.calc_allocation),
        "allocation_rub":         float(r["allocation_rub"]),
        "risk_level":             st.session_state.calc_risk_level,
        "risk_pct_chosen":        float(r["risk_pct_chosen"]),
        "max_loss_per_trade_rub": float(r["actual_loss"]),
        "lots_target":            int(r["lots"]),
        "position_rub":           float(r["actual_position"]),
        "applied_at":             datetime.now(timezone.utc).isoformat(),
    }
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    st.success(
        f"✅ **Настройки применены!**\n\n"
        f"Daily run будет покупать **{r['lots']} лотов SLVRUBF** "
        f"({rub(r['actual_position'])}) при следующем BUY-сигнале.\n\n"
        f"Файл сохранён: `baseline_outputs_prod/user_trading_config.json`"
    )
    st.balloons()


# =============================================================================
# MAIN
# =============================================================================

st.markdown("# 🧮 Калькулятор позиции для серебра")

top_signal_badge(get_current_signal())

st.markdown("> Помогу понять — стоит ли вкладываться в серебро, и если да, **какой суммой**. "
            "Простой 3-шаговый расчёт. Все технические термины спрятаны в раскрывающихся блоках.")

render_progress(st.session_state.calc_step)
st.markdown("---")

if st.session_state.calc_step == 1:
    show_step_1()
elif st.session_state.calc_step == 2:
    show_step_2()
elif st.session_state.calc_step == 3:
    show_step_3()

# Reset
st.markdown("---")
if st.button("🔄 Начать заново", help="Сбросить все настройки"):
    for k in ["calc_step", "calc_savings", "calc_allocation", "calc_risk_level"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# Загруженный конфиг (если есть)
if CONFIG_PATH.exists():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    st.caption(f"📁 Сейчас применено: **{cfg['lots_target']} лотов** "
                f"({rub(cfg['position_rub'])}), риск {cfg['risk_pct_chosen']:.1f}% "
                f"· последнее обновление: {cfg['applied_at'][:10]}")
