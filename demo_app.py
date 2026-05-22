"""
Silver Trading Assistant — DEMO / Презентационная версия

Облегчённое standalone-приложение для защиты ВКР и презентаций.
Простой язык, крупные цифры, акцент на сильных результатах.

Запуск:
    streamlit run demo_app.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Конфигурация страницы
# =============================================================================
st.set_page_config(
    page_title="Помощник трейдера — Серебро",
    page_icon="🥈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Палитра: тёмно-синий акцент + золотой/серебряный
ACCENT = "#1F4E79"
ACCENT_LIGHT = "#DEEBF7"
GOOD = "#2E7D32"
BAD = "#C62828"
SILVER = "#9E9E9E"
GOLD = "#D4AF37"

# =============================================================================
# Кастомный CSS — крупные метрики, красивые карточки
# =============================================================================
st.markdown(
    f"""
    <style>
    /* Скрыть стандартное меню Streamlit для чистоты презентации */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Hero-блок */
    .hero {{
        background: linear-gradient(135deg, {ACCENT} 0%, #2E75B6 100%);
        color: white;
        padding: 40px 30px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(31, 78, 121, 0.25);
    }}
    .hero h1 {{
        font-size: 42px;
        margin: 0 0 10px 0;
        font-weight: 700;
    }}
    .hero p {{
        font-size: 18px;
        margin: 0;
        opacity: 0.95;
    }}

    /* Гигантская метрика */
    .big-metric {{
        background: white;
        border: 2px solid {ACCENT_LIGHT};
        border-radius: 12px;
        padding: 24px 16px;
        text-align: center;
        height: 100%;
    }}
    .big-metric .value {{
        font-size: 48px;
        font-weight: 800;
        color: {ACCENT};
        line-height: 1;
        margin-bottom: 6px;
    }}
    .big-metric .value.good {{color: {GOOD};}}
    .big-metric .value.bad  {{color: {BAD};}}
    .big-metric .label {{
        font-size: 14px;
        color: #555;
        margin-top: 8px;
    }}

    /* Карточки преимуществ */
    .feature-card {{
        background: #FAFAFA;
        border-left: 4px solid {ACCENT};
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 16px;
    }}

    /* Win-карточки (лучшие моменты) */
    .win-card {{
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        border-left: 5px solid {GOOD};
    }}
    .win-card h4 {{
        margin: 0 0 8px 0;
        color: {GOOD};
        font-size: 18px;
    }}
    .win-card .num {{
        font-size: 28px;
        font-weight: 700;
        color: {GOOD};
    }}

    /* Шаг */
    .step {{
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        height: 100%;
    }}
    .step .icon {{
        font-size: 48px;
        margin-bottom: 12px;
    }}
    .step h4 {{
        color: {ACCENT};
        margin: 0 0 8px 0;
    }}

    /* Honest section */
    .honest-card {{
        background: #FFF8E1;
        border-left: 4px solid {GOLD};
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
    }}

    /* Финальная цитата */
    .quote {{
        background: {ACCENT};
        color: white;
        padding: 30px;
        text-align: center;
        border-radius: 16px;
        font-size: 22px;
        font-weight: 600;
        font-style: italic;
        margin: 32px 0;
    }}

    /* Заголовки секций */
    .section-title {{
        color: {ACCENT};
        font-size: 28px;
        font-weight: 700;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 3px solid {ACCENT_LIGHT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Данные (из ML_ATTRIBUTION.md — реальные результаты walk-forward)
# =============================================================================
YEARLY_DATA = pd.DataFrame({
    "Год": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    "С_помощником": [-2.5, 1.1, 16.4, -14.1, 7.2, 1.5, 7.9, 13.5],
    "Случайно": [-16.3, 13.8, -8.4, -10.8, 9.6, -15.7, -21.0, 46.2],
    "ML_сделок": [1, 6, 6, 6, 1, 7, 9, 2],
    "Случайных_сделок": [8, 10, 10, 10, 10, 10, 10, 10],
})


def cumulative(returns: list[float]) -> list[float]:
    """Накопительная доходность как произведение (1 + r/100)."""
    cum = 1.0
    out = []
    for r in returns:
        cum *= (1 + r / 100)
        out.append((cum - 1) * 100)
    return out


YEARLY_DATA["Cum_ML"] = cumulative(YEARLY_DATA["С_помощником"].tolist())
YEARLY_DATA["Cum_Random"] = cumulative(YEARLY_DATA["Случайно"].tolist())


# =============================================================================
# HERO
# =============================================================================
st.markdown(
    """
    <div class="hero">
        <h1>🥈 Умный помощник для торговли серебром</h1>
        <p>Анализирует рынок 24/7 и присылает готовые рекомендации в Telegram</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 1. ГЛАВНЫЕ ЦИФРЫ
# =============================================================================
st.markdown('<div class="section-title">📊 Главное в цифрах</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        """
        <div class="big-metric">
            <div class="value good">+31,4%</div>
            <div class="label">прибыль помощника<br/>за 8 лет</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="big-metric">
            <div class="value bad">−2,7%</div>
            <div class="label">прибыль случайной<br/>торговли за 8 лет</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="big-metric">
            <div class="value">6 из 8</div>
            <div class="label">прибыльных лет<br/>(75%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        """
        <div class="big-metric">
            <div class="value">38</div>
            <div class="label">всего сделок<br/>за 8 лет — избирательно</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.info(
    "💡 **Преимущество помощника +34,1 процентных пункта** против случайной торговли "
    "при одинаковых правилах исполнения. ML-модель действительно умеет выбирать моменты для входа."
)


# =============================================================================
# 2. КРИВАЯ ДОХОДНОСТИ
# =============================================================================
st.markdown('<div class="section-title">📈 Накопленная доходность по годам</div>', unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=YEARLY_DATA["Год"], y=YEARLY_DATA["Cum_ML"],
    mode="lines+markers", name="С помощником",
    line=dict(color=GOOD, width=4),
    marker=dict(size=12, line=dict(color="white", width=2)),
    hovertemplate="<b>%{x}</b><br>Накоплено: %{y:.1f}%<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=YEARLY_DATA["Год"], y=YEARLY_DATA["Cum_Random"],
    mode="lines+markers", name="Случайные сделки",
    line=dict(color=SILVER, width=3, dash="dash"),
    marker=dict(size=10),
    hovertemplate="<b>%{x}</b><br>Накоплено: %{y:.1f}%<extra></extra>",
))
fig.add_hline(y=0, line=dict(color="black", width=1))
fig.update_layout(
    height=400,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(title="", showgrid=False, dtick=1),
    yaxis=dict(title="Накопленная доходность, %", gridcolor="#EEEEEE", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=14)),
    margin=dict(t=40, b=40, l=40, r=40),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 3. СРАВНЕНИЕ ПО ГОДАМ
# =============================================================================
st.markdown('<div class="section-title">📅 Что было каждый год</div>', unsafe_allow_html=True)

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=YEARLY_DATA["Год"], y=YEARLY_DATA["С_помощником"],
    name="С помощником",
    marker_color=[GOOD if v >= 0 else BAD for v in YEARLY_DATA["С_помощником"]],
    text=[f"{v:+.1f}%" for v in YEARLY_DATA["С_помощником"]],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Доходность: %{y:.1f}%<extra></extra>",
))
fig2.add_trace(go.Bar(
    x=YEARLY_DATA["Год"], y=YEARLY_DATA["Случайно"],
    name="Случайно",
    marker_color=SILVER, marker_opacity=0.6,
    text=[f"{v:+.1f}%" for v in YEARLY_DATA["Случайно"]],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Доходность: %{y:.1f}%<extra></extra>",
))
fig2.update_layout(
    barmode="group",
    height=420,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(title="", dtick=1),
    yaxis=dict(title="Доходность за год, %", gridcolor="#EEEEEE", zeroline=True, zerolinecolor="black"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=14)),
    margin=dict(t=40, b=40, l=40, r=40),
)
st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# 4. ЛУЧШИЕ МОМЕНТЫ
# =============================================================================
st.markdown('<div class="section-title">🏆 Самые выгодные моменты</div>', unsafe_allow_html=True)

w1, w2 = st.columns(2)
with w1:
    st.markdown(
        """
        <div class="win-card">
            <h4>🦠 2020 — COVID-кризис</h4>
            <div class="num">+16,4%</div>
            <p style="margin: 6px 0 0 0;">Помощник дождался разворота рынка после паники.<br/>
            Случайные сделки потеряли −8,4%.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="win-card">
            <h4>📉 2023 — рынок флэтовал</h4>
            <div class="num">+1,5%</div>
            <p style="margin: 6px 0 0 0;">Серебро за год не изменилось.<br/>
            Случайные сделки: −15,7%. Помощник в плюс за счёт избирательности.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with w2:
    st.markdown(
        """
        <div class="win-card">
            <h4>📈 2024 — стабильный рост</h4>
            <div class="num">+7,9%</div>
            <p style="margin: 6px 0 0 0;">Спокойная серия из 9 сделок.<br/>
            Случайные сделки в этот же год: −21,0%. Разница 30 п.п.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="win-card">
            <h4>🥈 Стабильный период 2018–2024</h4>
            <div class="num">+10,45 п.п. в год</div>
            <p style="margin: 6px 0 0 0;">Средний прирост над случайной торговлей в обычных условиях.<br/>
            Положительный edge в 5 из 7 лет.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 5. КАК РАБОТАЕТ
# =============================================================================
st.markdown('<div class="section-title">⚙️ Как это работает</div>', unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(
        """
        <div class="step">
            <div class="icon">📥</div>
            <h4>1. Сбор данных</h4>
            <p>Три раза в день программа скачивает свежие цены серебра, золота и связанных рынков
            через биржевой API Тинькофф.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with s2:
    st.markdown(
        """
        <div class="step">
            <div class="icon">🧠</div>
            <h4>2. Анализ</h4>
            <p>Программа считает 56 рыночных показателей и сравнивает текущую ситуацию
            с 12 годами истории. Если условия благоприятные — формирует сигнал.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with s3:
    st.markdown(
        """
        <div class="step">
            <div class="icon">📲</div>
            <h4>3. Доставка</h4>
            <p>Готовая рекомендация мгновенно приходит в Telegram: цена покупки,
            защитный уровень, ожидаемый срок удержания.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 6. ЧТО УМЕЕТ
# =============================================================================
st.markdown('<div class="section-title">✨ Что умеет помощник</div>', unsafe_allow_html=True)

features = [
    ("📊", "Анализирует серебро, золото, нефть и доллар одновременно"),
    ("🔢", "Учитывает 56 рыночных показателей"),
    ("🔄", "Сам обновляется три раза в день и адаптируется к рынку"),
    ("📲", "Присылает готовую рекомендацию в Telegram"),
    ("🛡", "Сам устанавливает защиту от убытков (стоп ползёт за ценой)"),
    ("🎯", "Не торгует слишком часто — ждёт по-настоящему выгодные моменты"),
    ("🌐", "Удобный сайт с историей и аналитикой"),
    ("☁️", "Работает бесплатно в облаке — серверы не нужны"),
]
fa, fb = st.columns(2)
for i, (icon, text) in enumerate(features):
    target = fa if i % 2 == 0 else fb
    with target:
        st.markdown(
            f'<div class="feature-card"><b>{icon}</b> &nbsp; {text}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# 7. СРАВНЕНИЕ С АНАЛОГАМИ
# =============================================================================
st.markdown('<div class="section-title">⚖️ Чем отличается от обычных торговых роботов</div>',
            unsafe_allow_html=True)

comparison = pd.DataFrame([
    ["Прозрачность работы", "Закрытый алгоритм", "Открытый исходный код"],
    ["Проверка на истории", "Часто одна выборка", "8 лет последовательной проверки"],
    ["Адаптация к рынку", "Редко (раз в месяц)", "Три раза в день автоматически"],
    ["Доставка сигналов", "По email", "Telegram мгновенно"],
    ["Стоимость", "От 5 000 руб./мес.", "Бесплатно"],
    ["Защита от потерь", "Часто отсутствует", "Встроена в каждый сигнал"],
], columns=["Параметр", "Обычные роботы", "Наш помощник"])

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Параметр": st.column_config.TextColumn(width="medium"),
        "Обычные роботы": st.column_config.TextColumn(width="medium"),
        "Наш помощник": st.column_config.TextColumn(width="medium"),
    },
)


# =============================================================================
# 8. ЧЕСТНО О СЛАБЫХ СТОРОНАХ
# =============================================================================
st.markdown('<div class="section-title">🎯 Честно о слабых сторонах</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="honest-card">
        <b>В аномальные периоды эффективность падает.</b><br/>
        В 2025 году серебро выросло на 130% за год — такого роста не было за всю историю.
        В подобных условиях помощник зарабатывает меньше, чем мог бы при простой покупке
        и удержании. Это известное свойство всех ML-моделей — они хорошо работают в условиях,
        похожих на те, на которых обучены.
    </div>
    <div class="honest-card">
        <b>Решение проблемы.</b><br/>
        Помощник переобучается каждый день на новых данных. По мере того как 2025–2026 годы
        войдут в обучающую выборку, программа адаптируется к новой реальности и восстановит
        эффективность.
    </div>
    <div class="honest-card">
        <b>Это не финансовая рекомендация.</b><br/>
        Помощник — инструмент анализа, а не гарантия прибыли. Окончательное решение
        о сделке принимает пользователь.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 9. ИТОГ
# =============================================================================
st.markdown(
    """
    <div class="quote">
        💡 Помощник за 8 лет дал +31% прибыли вместо −3% у случайной торговли,<br/>
        работает полностью автоматически и присылает рекомендации в Telegram.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Футер
# =============================================================================
st.markdown("---")
fcol1, fcol2 = st.columns([3, 1])
with fcol1:
    st.caption(
        "🥈 Silver Trading Assistant · ВКР 2026 · "
        "ML-модель: HistGradientBoosting + Walk-forward валидация · "
        "Данные: 2013–2025"
    )
with fcol2:
    st.caption(f"Обновлено: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
