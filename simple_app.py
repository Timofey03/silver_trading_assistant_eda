"""
Silver Trading Assistant — Облегчённое приложение для конечного пользователя.

Запуск:
    streamlit run simple_app.py --server.port 8502

Цель — минимум технических деталей, максимум практической пользы.
5 страниц: Сейчас / Мои сделки / Калькулятор / Как работал / Настройки.

Может работать параллельно с основным dashboard_app.py на другом порту.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Серебро · помощник",
    page_icon="🥈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Кастомные стили (крупные кнопки, чистый минимализм)
st.markdown(
    """
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Brand color tweaks */
    :root {
        --brand: #1F4E79;
        --brand-light: #DEEBF7;
        --good: #2E7D32;
        --bad: #C62828;
        --warn: #F9A825;
    }

    /* Bigger buttons in main area */
    .stButton > button {
        font-size: 16px;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--brand);
        border: none;
    }

    /* Sidebar branding */
    [data-testid="stSidebar"] {
        background-color: #FAFAFA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar шапка
with st.sidebar:
    st.markdown("## 🥈 Помощник")
    st.caption("Серебро · упрощённая версия")
    st.divider()


# Multi-page навигация (Streamlit >= 1.30)
pages = [
    st.Page("simple_pages/1_now.py",        title="Сейчас",          icon="📍", default=True),
    st.Page("simple_pages/2_trades.py",     title="Мои сделки",      icon="💼"),
    st.Page("simple_pages/3_calculator.py", title="Калькулятор",     icon="🧮"),
    st.Page("simple_pages/4_stats.py",      title="Как работал",     icon="📊"),
    st.Page("simple_pages/6_evolution.py",  title="Эволюция модели", icon="🔬"),
    st.Page("simple_pages/5_settings.py",   title="Настройки",       icon="⚙"),
]
pg = st.navigation(pages)
pg.run()
