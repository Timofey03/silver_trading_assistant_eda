"""Страница: position sizing calculator + mode comparison + sell signals."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils import (
    get_current_signal, get_tinkoff_status,
    inject_styles, top_signal_badge, rub,
)
from app.sizing import (
    SizingInputs, calculate_position_size, explain_sizing,
    INSTRUMENTS,
)

st.set_page_config(page_title="Калькулятор", page_icon="💡", layout="wide")
inject_styles()

st.markdown("# 💡 Калькулятор размера позиции")
top_signal_badge(get_current_signal())

st.markdown("""
> Этот калькулятор отвечает на главный вопрос: **на какую сумму покупать?**
> Использует **risk-based sizing**: вы выбираете % капитала, который готовы
> потерять на одной сделке, а калькулятор считает размер позиции.
""")

# =============================================================================
# 1. Inputs
# =============================================================================

st.markdown("## 📝 Параметры")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Ваш счёт**")
    tinkoff = get_tinkoff_status()
    default_acc = tinkoff.get("cash", {}).get("value", 1_000_000) if tinkoff.get("ok") else 1_000_000

    account = st.number_input(
        "Размер счёта (RUB)", value=int(default_acc),
        min_value=10_000, step=10_000,
        help="Текущая стоимость вашего счёта (cash + позиции по рыночной)",
    )

    risk_pct = st.slider(
        "Риск на сделку (%)",
        min_value=0.5, max_value=5.0, value=1.5, step=0.1,
        help="% капитала, который вы готовы потерять если сработает стоп. "
             "Профессиональные трейдеры: 0.5-2%. Розничные: до 5%.",
    )

    stop_pct = st.slider(
        "Стоп-дистанция (%)",
        min_value=2.0, max_value=15.0, value=7.0, step=0.5,
        help="На сколько % цена должна упасть от пика чтобы trailing stop сработал. "
             "Меньше = плотнее = чаще выбивает. Больше = риск дольше.",
    ) / 100.0

with col2:
    st.markdown("**Инструмент**")
    instr_name = st.selectbox(
        "Что покупаем", list(INSTRUMENTS.keys()),
        index=1,   # SLVRUBF default
        help="SLVRUBF — единственный реально доступный из РФ для серебра. "
             "SLV — если есть SPB Exchange. PLZL/POLY — золото/металлы.",
    )
    instrument = INSTRUMENTS[instr_name]

    if instrument.currency == "USD":
        current_price = st.number_input(
            f"Текущая цена {instr_name} (USD)",
            value=85.0, min_value=0.01, step=0.5,
        )
        usd_rate = st.number_input("Курс USD/RUB", value=80.0, step=0.5)
    else:
        current_price = st.number_input(
            f"Текущая цена {instr_name} (RUB)",
            value=200.0 if instr_name == "SLVRUBF" else 14000.0,
            min_value=0.01, step=1.0,
        )
        usd_rate = 80.0

    sig = get_current_signal()
    p_up_now = float(sig.get("p_up", 0.5)) if sig.get("p_up") is not None else 0.5
    p_up = st.slider(
        "Уверенность модели (p_up)",
        min_value=0.40, max_value=0.85,
        value=float(p_up_now), step=0.01,
        help="Вероятность UP по модели. По умолчанию — текущая. "
             "Чем выше, тем больше размер (Kelly tilt).",
    )

    use_kelly = st.checkbox("Применять Kelly tilt по p_up", value=True,
                             help="Размер ∝ (0.5 + (p_up-0.5)×2). "
                                  "Без tilt — везде одинаковый размер.")

# =============================================================================
# 2. Calculate
# =============================================================================

inputs = SizingInputs(
    account_value_rub=account,
    risk_per_trade_pct=risk_pct,
    stop_distance_pct=stop_pct,
    instrument=instrument,
    current_price=current_price,
    usd_rub_rate=usd_rate,
    p_up=p_up,
    confidence_tilt=use_kelly,
)
result = calculate_position_size(inputs)

st.markdown("---")
st.markdown("## 📋 Рекомендация")

col_main, col_compare = st.columns([3, 2])

with col_main:
    explanation = explain_sizing(result, inputs)
    if result.lots > 0:
        st.success(explanation)
    else:
        st.error(explanation)

with col_compare:
    st.markdown("**Что если изменить параметры?**")
    st.caption("Sensitivity по риску")
    sens_rows = []
    for r in [0.5, 1.0, 1.5, 2.0, 3.0]:
        test_inputs = SizingInputs(
            account_value_rub=account, risk_per_trade_pct=r,
            stop_distance_pct=stop_pct, instrument=instrument,
            current_price=current_price, usd_rub_rate=usd_rate,
            p_up=p_up, confidence_tilt=use_kelly,
        )
        test_res = calculate_position_size(test_inputs)
        sens_rows.append({
            "Риск %": f"{r}%",
            "Лоты":   test_res.lots,
            "Notional ₽": f"{test_res.notional_rub/1000:.0f}k",
            "Max loss ₽": f"{test_res.max_loss_rub:,.0f}",
        })
    st.dataframe(pd.DataFrame(sens_rows), hide_index=True, use_container_width=True)


# =============================================================================
# 3. Mode selection — frequency vs edge trade-off
# =============================================================================

st.markdown("---")
st.markdown("## 🎚 Стратегические режимы — сколько сигналов в год?")

modes_csv = ROOT / "baseline_outputs_modes" / "modes_comparison.csv"
if modes_csv.exists():
    modes = pd.read_csv(modes_csv)
    fwd = modes[modes["split"] == "forward"].copy()

    st.markdown("""
Помощник может работать в разных режимах. Все режимы используют **одну и ту же ML-модель**,
но разные пороги входа/выхода + кулдаун.
""")

    show = fwd[["mode", "n_trades", "win_rate", "total_return",
                "sharpe_ann", "max_drawdown", "calmar",
                "exit_model_exit", "exit_trail_stop"]].copy()
    show.columns = ["Режим", "Сделок/год", "Win rate", "Total return",
                    "Sharpe", "Max DD", "Calmar",
                    "Выходы по модели", "Выходы по stop"]
    show["Win rate"] = show["Win rate"].apply(lambda x: f"{x*100:.0f}%")
    show["Total return"] = show["Total return"].apply(lambda x: f"{x*100:+.1f}%")
    show["Max DD"] = show["Max DD"].apply(lambda x: f"{x*100:.1f}%")

    # Подсвечиваем лучший
    def highlight_best(row):
        if row["Режим"] == "balanced":
            return ["background-color: rgba(0,200,83,0.15)"] * len(row)
        return [""] * len(row)

    st.dataframe(show.style.apply(highlight_best, axis=1),
                 hide_index=True, use_container_width=True)

    st.markdown("""
**🏆 Победитель — Balanced**: больше сделок (13 vs 5), выше win rate (69% vs 60%),
**вдвое больше profit** (+45% vs +21%). И главное: **используется SELL-сигнал модели**
(8 выходов «по решению модели», когда p_up упал ниже 0.45).

**❌ Ultra проигрывает** — слишком много шума, edge ломается.
""")

    st.info("""
**Как переключить режим**: пока через файл `silver_signal_modes.py` и переменную в
[silver_production_inference.py]. Полноценная UI-настройка — в работе.
""")
else:
    st.warning("Сравнение режимов не сгенерировано. Запустите: "
                "`python silver_signal_modes.py --compare`")


# =============================================================================
# 4. Best practices
# =============================================================================

st.markdown("---")
st.markdown("## 📚 Принципы position sizing")

with st.expander("📖 Зачем нужен risk-based sizing"):
    st.markdown("""
**Главное правило**: размер позиции выбирается **исходя из риска**, а не из «хочу больше прибыли».

**Пример**: у вас 1М ₽ счёт.
- Risk per trade = 1.5% = **15,000 ₽ максимальный убыток**
- Stop distance = 7% (trailing stop)
- → размер позиции = 15000 / 0.07 = **214,000 ₽ notional**

При 7% движении против вас → теряете ровно 15k ₽ (1.5% счёта). Это план.

**Почему 1-3% риска**, а не 10-20%:
- 10 убыточных сделок подряд при 5% риска = **-40% счёта** (drawdown)
- При 1.5% риска = -14% drawdown — восстановимо

**Kelly tilt**: при высокой уверенности модели (p_up=0.7+) размер ×1.5,
при p_up=0.5 (граница) — ×0.5. **Никогда не входим без edge.**
""")

with st.expander("📖 Stop distance — почему 7%"):
    st.markdown("""
Trailing stop 7% — компромисс:
- **Меньше (3-5%)** — чаще выбивает, упускаешь тренды
- **Больше (10-15%)** — позиция дольше живёт, но больше убыток если ошибся

7% — типичная амплитуда **обычной волатильности серебра** на дневках.
ATR_14 ≈ 5-8% от цены, поэтому 7% — outside-of-noise.
""")

with st.expander("📖 Как часто можно входить — risk budget"):
    st.markdown("""
**Простой расчёт**:
- 1.5% риска × **20 одновременных сделок** = 30% общего риска. Слишком много.
- 1.5% × **5 одновременных позиций** = 7.5% — нормально.

Поэтому **cooldown 10-15 дней** = не больше 1-2 одновременных открытых позиций
при max_hold 30-45 дней. Это и есть risk budgeting.

**Aggressive режим** с cooldown=5 и max_hold=20 = до 4 одновременных позиций →
6% общего риска. **Ultra** с cooldown=3, max_hold=15 = до 5 позиций → 7.5%.
Дальше уже опасно.
""")
