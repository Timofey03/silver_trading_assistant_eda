"""Страница: 🧮 Калькулятор — сколько купить и сколько риск."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.utils import get_current_signal, get_kpis
from app.simple_storage import get_capital, set_capital


# Параметры стратегии (захардкожены — OptimalV2)
TRAIL_PCT = 0.12
MAX_HOLD_DAYS = 30
SLVRUBF_MULTIPLIER = 100   # 1 лот = 100 единиц серебра
FUTURES_MARGIN_PCT = 0.12  # маржа ~12% от полной стоимости

WF_TRADES_FILE = ROOT / "baseline_outputs_walkforward" / "trades_all.csv"
V25_TRADES_FILE = ROOT / "baseline_outputs_v25" / "v25_forward_trades.csv"
E3B_TRADES_FILE = ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv"


@st.cache_data(ttl=300)
def compute_stats(trades_path: Path, span_years: float) -> dict:
    """Реальная статистика из конкретного файла сделок."""
    if not trades_path.exists():
        return {}
    t = pd.read_csv(trades_path)
    nr = t["net_return"]
    wins = nr[nr > 0]
    losses = nr[nr <= 0]
    buckets_def = [
        (-1.0,  -0.07, "🔴 Большой убыток (хуже −7%)"),
        (-0.07, -0.02, "🟠 Малый убыток (от −7% до −2%)"),
        (-0.02,  0.02, "⚪ Около нуля (±2%)"),
        ( 0.02,  0.07, "🟢 Малая прибыль (от +2% до +7%)"),
        ( 0.07, 10.0,  "🟢 Большая прибыль (больше +7%)"),
    ]
    buckets = []
    for lo, hi, label in buckets_def:
        mask = (nr >= lo) & (nr < hi)
        prob = float(mask.mean())
        avg = float(nr[mask].mean()) if mask.any() else 0.0
        buckets.append((label, prob, avg))
    return {
        "n":            len(nr),
        "win_rate":     float((nr > 0).mean()),
        "mean_win":     float(wins.mean()) if len(wins) else 0,
        "mean_loss":    float(losses.mean()) if len(losses) else 0,
        "expected":     float(nr.mean()),
        "best":         float(nr.max()),
        "worst":        float(nr.min()),
        "avg_hold":     int(t["hold_days"].mean()) if "hold_days" in t.columns else 24,
        "trades_per_year": len(nr) / span_years,
        "buckets":      buckets,
    }


st.title("🧮 Калькулятор сделки")
st.caption("Рассчитывает сколько лотов купить, сколько денег задействовать и какой риск")


# =============================================================================
# Текущая цена + сигнал
# =============================================================================
signal = get_current_signal()
kpis = get_kpis()
current_price = signal.get("current_price") or kpis.get("last_price") or 0
p_up = signal.get("p_up", 0) or 0


# =============================================================================
# Капитал
# =============================================================================
stored_capital = get_capital()
c1, c2 = st.columns([2, 1])
with c1:
    capital = st.number_input(
        "💰 Ваш капитал (₽)",
        min_value=10_000.0,
        value=float(stored_capital) if stored_capital > 0 else 100_000.0,
        step=10_000.0,
        format="%.0f",
        help="Общая сумма, которую готовы использовать. Сохранится в настройках.",
    )
with c2:
    st.write("")
    st.write("")
    if st.button("💾 Запомнить", use_container_width=True):
        set_capital(capital)
        st.success("Сохранено в настройках")

# Доля от капитала
allocation_pct = st.slider(
    "🎯 Какую долю капитала использовать на эту сделку?",
    min_value=5, max_value=100, value=30, step=5,
    help="Рекомендуется 20–40%. На фьючерсах требуется только маржа (~12%), "
         "так что физически блокируется меньше денег.",
)

st.divider()


# =============================================================================
# Расчёт
# =============================================================================
if current_price <= 0:
    st.error("Текущая цена недоступна. Обновите данные на странице **📍 Сейчас**.")
    st.stop()

notional_per_lot = current_price * SLVRUBF_MULTIPLIER     # стоимость 1 лота
target_notional = capital * (allocation_pct / 100)         # целевой объём
lots = max(1, int(target_notional / notional_per_lot))     # округление вниз
notional_total = lots * notional_per_lot
margin_required = notional_total * FUTURES_MARGIN_PCT      # ~12% маржа

# Риск через trailing stop
max_loss_rub = notional_total * TRAIL_PCT
max_loss_pct_of_capital = (max_loss_rub / capital) * 100

# =============================================================================
# Результат
# =============================================================================
st.markdown("## 📋 Результат")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        "Сколько купить",
        f"{lots} лот{'а' if 1<lots<5 else 'ов' if lots>4 else ''}",
        help=f"1 лот = 100 ед. серебра ≈ {notional_per_lot:,.0f} ₽".replace(",", " "),
    )
with c2:
    st.metric(
        "Маржа (блокируется)",
        f"{margin_required:,.0f} ₽".replace(",", " "),
        help="Фьючерсы требуют только маржу — ~12% от полной стоимости",
    )
with c3:
    st.metric(
        "Объём позиции",
        f"{notional_total:,.0f} ₽".replace(",", " "),
        help="Полная стоимость серебра в позиции",
    )

st.divider()

# Риск-секция
st.markdown("### 🛡 Сколько риск, если что-то пойдёт не так")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"""
        <div style="background:#FFEBEE; padding:18px; border-radius:10px;
                    border-left: 4px solid #C62828;">
            <div style="font-size:14px; color:#666;">Максимальный убыток</div>
            <div style="font-size:32px; font-weight:700; color:#C62828;">
                −{max_loss_rub:,.0f} ₽
            </div>
            <div style="font-size:14px; color:#666; margin-top:6px;">
                Это <b>{max_loss_pct_of_capital:.1f}%</b> от вашего капитала
            </div>
        </div>
        """.replace(",", " "),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div style="background:#FFF8E1; padding:18px; border-radius:10px;
                    border-left: 4px solid #F9A825;">
            <div style="font-size:14px; color:#666;">Защита (trailing stop)</div>
            <div style="font-size:32px; font-weight:700; color:#F9A825;">
                {TRAIL_PCT*100:.0f}%
            </div>
            <div style="font-size:14px; color:#666; margin-top:6px;">
                Защитный уровень автоматически<br/>
                поднимается за ценой
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# =============================================================================
# Прибыль-секция: основано на РЕАЛЬНОЙ статистике бэктеста
# =============================================================================
st.markdown("### 📈 Сколько реально можно заработать")

# Выбор какие данные использовать (E3b — финальная модель по умолчанию)
mode = st.radio(
    "По каким данным считать?",
    options=[
        "🏆 E3b — новая модель диплома (рекомендуется)",
        "🟢 V25 — текущая production-модель",
        "🔵 Базовая walk-forward — для сравнения",
    ],
    horizontal=False,
    help=(
        "**E3b** = финальная улучшенная модель дипломной работы (cross-asset, "
        "adaptive barriers, feature selection). Проверена на 10+ лет walk-forward.\n\n"
        "**V25** = текущая production (Streamlit), но тестировалась только на 1.3 года "
        "экстремального бычьего рынка — статистически нерепрезентативна.\n\n"
        "**Базовая** = walk-forward 8 лет старой модели OptimalV2."
    ),
)

if mode.startswith("🏆"):
    stats = compute_stats(E3B_TRADES_FILE, span_years=10.32)
    src_label = "E3b — финальная модель дипломной работы"
    period_text = "за 10.3 года walk-forward"
elif mode.startswith("🟢"):
    stats = compute_stats(V25_TRADES_FILE, span_years=1.33)
    src_label = "V25 — текущая production-модель"
    period_text = "за 16 месяцев работы"
else:
    stats = compute_stats(WF_TRADES_FILE, span_years=8.0)
    src_label = "Базовая модель (walk-forward 2018–2025)"
    period_text = "за 8 лет проверки"

if not stats:
    st.error("Файл со сделками не найден.")
    st.stop()

st.caption(
    f"Расчёт по фактическим **{stats['n']} сделкам** ({src_label}) {period_text}. "
    f"Каждая сделка длится в среднем **{stats['avg_hold']} дней**, "
    f"частота — около **{stats['trades_per_year']:.1f} сделок в год**. "
    f"На одну позицию идёт весь капитал, который вы выделили выше."
)

# Главные средние
c1, c2, c3 = st.columns(3)
with c1:
    win_rub = notional_total * stats["mean_win"]
    st.markdown(f"""
        <div style="background:#E8F5E9; padding:14px; border-radius:10px;
                    border-left: 4px solid #2E7D32;">
            <div style="font-size:13px; color:#666;">Средняя <b>прибыльная</b> сделка</div>
            <div style="font-size:24px; font-weight:700; color:#2E7D32;">
                +{stats['mean_win']*100:.1f}% &nbsp; ({win_rub:+,.0f} ₽)
            </div>
            <div style="font-size:12px; color:#666;">Случается в {stats['win_rate']*100:.0f}% сделок</div>
        </div>
    """.replace(",", " "), unsafe_allow_html=True)
with c2:
    loss_rub = notional_total * stats["mean_loss"]
    st.markdown(f"""
        <div style="background:#FFEBEE; padding:14px; border-radius:10px;
                    border-left: 4px solid #C62828;">
            <div style="font-size:13px; color:#666;">Средняя <b>убыточная</b> сделка</div>
            <div style="font-size:24px; font-weight:700; color:#C62828;">
                {stats['mean_loss']*100:+.1f}% &nbsp; ({loss_rub:+,.0f} ₽)
            </div>
            <div style="font-size:12px; color:#666;">Случается в {(1-stats['win_rate'])*100:.0f}% сделок</div>
        </div>
    """.replace(",", " "), unsafe_allow_html=True)
with c3:
    exp_rub = notional_total * stats["expected"]
    st.markdown(f"""
        <div style="background:#E3F2FD; padding:14px; border-radius:10px;
                    border-left: 4px solid #1F4E79;">
            <div style="font-size:13px; color:#666;">В среднем <b>за одну сделку</b></div>
            <div style="font-size:24px; font-weight:700; color:#1F4E79;">
                {stats['expected']*100:+.1f}% &nbsp; ({exp_rub:+,.0f} ₽)
            </div>
            <div style="font-size:12px; color:#666;">Уже с учётом и побед, и потерь</div>
        </div>
    """.replace(",", " "), unsafe_allow_html=True)

# Подробное распределение всех 5 сценариев
st.write("")
st.markdown("#### 🎯 Что произошло с каждой сделкой — по фактам")
st.caption(
    "Каждая из проверенных сделок попадает в один из пяти исходов. "
    "Колонка «В рублях» — что вы получили бы от выделенной доли капитала."
)
bucket_rows = []
for label, prob, avg_ret in stats["buckets"]:
    pnl_rub = notional_total * avg_ret
    bucket_rows.append({
        "Исход":            label,
        "Как часто":        f"{prob*100:.0f}% сделок",
        "Средний результат": f"{avg_ret*100:+.1f}%",
        "В рублях":         f"{pnl_rub:+,.0f} ₽".replace(",", " "),
    })

import pandas as _pd
bucket_df = _pd.DataFrame(bucket_rows)

def _color_outcome(row):
    text = row["Исход"]
    if "🔴" in text:
        return ["background-color: #FFEBEE"] * len(row)
    if "🟠" in text:
        return ["background-color: #FFF3E0"] * len(row)
    if "⚪" in text:
        return ["background-color: #F5F5F5"] * len(row)
    if "🟢" in text:
        return ["background-color: #E8F5E9"] * len(row)
    return [""] * len(row)

st.dataframe(
    bucket_df.style.apply(_color_outcome, axis=1),
    use_container_width=True, hide_index=True,
)

# Прогноз на год: диапазон вместо одной цифры
st.write("")
st.markdown("#### 📅 Возможный результат за год")
st.caption(
    "Доходность зависит от рыночных условий. Показываем диапазон: в лучшие, "
    "в типичные и в неудачные годы по реальной истории."
)

tpy = stats["trades_per_year"]
alloc = allocation_pct / 100
capital_alloc = capital * alloc

if mode.startswith("🏆"):
    # E3b — реальные годовые результаты из bаcktest
    # Из E3b walk-forward: 2015-2025, обычно +5-15% в год, без катастроф
    # Лучшие годы: 2021 +49%, 2023 +37%. Худшие: 2016 -8%, 2020 COVID -10%.
    best_year_pct = 0.40     # как 2021/2023 (без аномалий)
    typical_year_pct = 0.08  # средний год (после compounding)
    worst_year_pct = -0.10   # как 2020 COVID
    best_label = "В удачный год (как 2021, 2023)"
    typical_label = "В средний год"
    worst_label = "В неудачный год (как 2020)"
elif mode.startswith("🟢"):
    # Текущая модель V25 — bull-market sample
    best_year_pct = 0.80
    typical_year_pct = 0.30
    worst_year_pct = -0.10
    best_label = "В благоприятный год (как 2025)"
    typical_label = "В обычный год"
    worst_label = "В трудный год"
else:
    # Базовая walk-forward — используем реальный year_breakdown
    yearly_returns = [-10.3, -4.6, 6.3, -28.6, -2.1, -14.5, -1.6, 17.8]
    sorted_y = sorted(yearly_returns)
    best_year_pct = sorted_y[-2] / 100
    typical_year_pct = sorted(yearly_returns)[len(yearly_returns)//2] / 100
    worst_year_pct = sorted_y[1] / 100
    best_label = "В удачный год (как 2020, 2025)"
    typical_label = "В средний год"
    worst_label = "В неудачный год"

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
        <div style="background:#E8F5E9; padding:16px; border-radius:10px;
                    border-left: 4px solid #2E7D32; height: 100%;">
            <div style="font-size:13px; color:#666;">🟢 {best_label}</div>
            <div style="font-size:28px; font-weight:700; color:#2E7D32; line-height:1.1;">
                +{best_year_pct*100:.0f}%
            </div>
            <div style="font-size:15px; color:#2E7D32; font-weight:600;">
                +{capital_alloc * best_year_pct:,.0f} ₽
            </div>
        </div>
    """.replace(",", " "), unsafe_allow_html=True)
with c2:
    typical_color = "#1F4E79" if typical_year_pct >= 0 else "#9E9E9E"
    typical_bg = "#E3F2FD" if typical_year_pct >= 0 else "#F5F5F5"
    sign = "+" if typical_year_pct >= 0 else ""
    st.markdown(f"""
        <div style="background:{typical_bg}; padding:16px; border-radius:10px;
                    border-left: 4px solid {typical_color}; height: 100%;">
            <div style="font-size:13px; color:#666;">🔵 {typical_label}</div>
            <div style="font-size:28px; font-weight:700; color:{typical_color}; line-height:1.1;">
                {sign}{typical_year_pct*100:.0f}%
            </div>
            <div style="font-size:15px; color:{typical_color}; font-weight:600;">
                {sign}{capital_alloc * typical_year_pct:,.0f} ₽
            </div>
        </div>
    """.replace(",", " "), unsafe_allow_html=True)
with c3:
    st.markdown(f"""
        <div style="background:#FFEBEE; padding:16px; border-radius:10px;
                    border-left: 4px solid #C62828; height: 100%;">
            <div style="font-size:13px; color:#666;">🔴 {worst_label}</div>
            <div style="font-size:28px; font-weight:700; color:#C62828; line-height:1.1;">
                {worst_year_pct*100:.0f}%
            </div>
            <div style="font-size:15px; color:#C62828; font-weight:600;">
                {capital_alloc * worst_year_pct:,.0f} ₽
            </div>
        </div>
    """.replace(",", " "), unsafe_allow_html=True)

st.caption(
    f"Сделок в год: **~{tpy:.0f}** · средняя длительность сделки: **{stats['avg_hold']} дней** · "
    f"доля капитала на сделку: **{allocation_pct}%**"
)

if mode.startswith("🏆"):
    st.success(
        "🏆 **E3b — финальная модель диплома.** Использует данные 5 металлов "
        "(silver/gold/platinum/palladium/copper), adaptive volatility-scaled "
        "барьеры, feature selection top-30. Walk-forward проверка на 10.3 года: "
        "Sharpe **0.53**, Win Rate **69%**, накопленная доходность **+115%**. "
        "Подробнее — на странице **🔬 Эволюция модели**."
    )
elif mode.startswith("🟢"):
    st.warning(
        "⚠ **V25 — текущая production-модель** (показывается в основном Streamlit). "
        "В период 2025–2026 фактическая доходность составила +442%, но это совпало "
        "с экстремальным ростом серебра (silver +136% YoY). Для расчёта используем "
        "**более консервативные оценки**. В дипломе E3b превзошла V25 на +148 пп."
    )
else:
    st.info(
        "ℹ Это **базовая walk-forward модель** старой версии OptimalV2 — оставлена для "
        "академического сравнения. Реально в отдельные годы результат колебался от "
        "−29% (2021) до +18% (2025). См. **📊 Как работал** для детального разбора."
    )

# Предупреждения
warnings = []
if max_loss_pct_of_capital > 5:
    warnings.append(
        f"⚠ Риск **{max_loss_pct_of_capital:.1f}%** от капитала — это много. "
        f"Уменьшите долю до 20–30%."
    )
if margin_required > capital * 0.5:
    warnings.append(
        f"⚠ Маржа **{margin_required:,.0f} ₽** > половины капитала. "
        f"Не останется денег на другие сделки.".replace(",", " ")
    )
if lots > 10:
    warnings.append(
        "💡 Большое количество лотов — можно разбить вход на 2–3 части "
        "для лучшей средней цены."
    )

if warnings:
    st.divider()
    for w in warnings:
        st.warning(w)


# =============================================================================
# Если есть свежий сигнал — подсказка перейти
# =============================================================================
if signal.get("signal") == "BUY":
    st.divider()
    st.info(
        f"🟢 **Прямо сейчас есть сигнал на покупку** (уверенность {p_up*100:.0f}%). "
        f"Перейдите на **📍 Сейчас**, чтобы записать сделку с этими параметрами."
    )


# =============================================================================
# Объяснение «что значат цифры» в expander
# =============================================================================
st.divider()
with st.expander("ℹ️ Почему именно эти цифры — простое объяснение"):
    st.markdown(
        f"""
        **Как считается размер позиции:**

        1. Берём ваш капитал: **{capital:,.0f} ₽**
        2. Берём долю на сделку: **{allocation_pct}%** → **{target_notional:,.0f} ₽**
        3. Делим на стоимость одного лота: текущая цена × 100 = {notional_per_lot:,.0f} ₽
        4. Получаем **{lots}** лотов

        **Почему маржа меньше объёма:** фьючерсы — это контракты, а не покупка
        самого серебра. Биржа берёт залог (маржу) около 12% от полной стоимости.
        Остальное играет роль «кредитного плеча».

        **Откуда берётся максимальный убыток:** помощник использует «ползущий стоп»
        ({TRAIL_PCT*100:.0f}%). Если цена пойдёт против вас на 12%, позиция
        автоматически закрывается, и убыток не превысит эту величину.

        **Срок удержания:** если защита не сработает, помощник всё равно
        закроет позицию через {MAX_HOLD_DAYS} дней — чтобы не зависать в
        неработающих сделках.
        """.replace(",", " ")
    )
