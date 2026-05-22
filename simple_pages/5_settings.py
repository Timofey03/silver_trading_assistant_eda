"""Страница: ⚙ Настройки — капитал и Telegram."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.notifier import TelegramNotifier
from app.simple_storage import (
    get_capital, set_capital,
    get_telegram, set_telegram, load_config,
)


st.title("⚙ Настройки")


# =============================================================================
# Капитал
# =============================================================================
st.markdown("## 💰 Капитал")
st.caption("Используется в калькуляторе и для расчёта прибыли в рублях")

current_capital = get_capital()
c1, c2 = st.columns([2, 1])
with c1:
    new_cap = st.number_input(
        "Ваш капитал, ₽",
        min_value=0.0,
        value=float(current_capital) if current_capital > 0 else 100_000.0,
        step=10_000.0,
        format="%.0f",
    )
with c2:
    st.write("")
    st.write("")
    if st.button("💾 Сохранить капитал", use_container_width=True, type="primary"):
        set_capital(new_cap)
        st.success(f"Сохранено: {new_cap:,.0f} ₽".replace(",", " "))

if current_capital > 0:
    st.caption(f"Сейчас сохранено: {current_capital:,.0f} ₽".replace(",", " "))
else:
    st.caption("Капитал ещё не задан")


st.divider()


# =============================================================================
# Telegram
# =============================================================================
st.markdown("## 📲 Telegram-уведомления")

st.markdown("""
Помощник может присылать уведомления о новых сигналах прямо в Telegram.
Для этого нужно настроить бота. Это 2 простых шага.
""")

with st.expander("📖 Как получить токен бота и chat_id (если ещё нет)"):
    st.markdown("""
    **Шаг 1. Создать бота:**
    1. Откройте в Telegram чат с **@BotFather**
    2. Отправьте команду `/newbot`
    3. Придумайте имя (например, «Мой помощник по серебру»)
    4. Придумайте username (должен заканчиваться на `_bot`, например `silver_helper_bot`)
    5. Скопируйте **токен** — длинная строка вида `1234567890:AAEh...`

    **Шаг 2. Получить chat_id:**
    1. Откройте в Telegram чат с **@userinfobot** (или **@getmyid_bot**)
    2. Отправьте любое сообщение
    3. Бот пришлёт ваш **chat_id** — это число (например `123456789`)

    После этого вставьте оба значения в форму ниже и нажмите «Проверить».
    """)

tg = get_telegram()

c1, c2 = st.columns(2)
with c1:
    bot_token = st.text_input(
        "Токен бота",
        value=tg["bot_token"],
        type="password",
        placeholder="1234567890:AAEh...",
        help="Получите у @BotFather",
    )
with c2:
    chat_id = st.text_input(
        "Ваш chat_id",
        value=tg["chat_id"],
        placeholder="123456789",
        help="Получите у @userinfobot",
    )

cc1, cc2, cc3 = st.columns([1, 1, 2])
with cc1:
    if st.button("💾 Сохранить", use_container_width=True):
        set_telegram(bot_token, chat_id)
        st.success("Сохранено")

with cc2:
    if st.button("📨 Проверить", use_container_width=True, type="primary"):
        if not bot_token or not chat_id:
            st.error("Заполните оба поля")
        else:
            # Сохраняем перед тестом
            set_telegram(bot_token, chat_id)
            notifier = TelegramNotifier(token=bot_token, chat_id=chat_id)
            ok = notifier.send(
                "🥈 *Тест подключения*\n\n"
                "Если вы это видите — бот настроен и готов присылать "
                "сигналы от помощника по серебру."
            )
            if ok:
                st.success("✅ Сообщение отправлено! Проверьте Telegram.")
            else:
                st.error(
                    "❌ Не удалось отправить. Проверьте:\n"
                    "- правильность токена\n"
                    "- правильность chat_id\n"
                    "- что вы написали боту хотя бы одно сообщение (иначе он не может писать первым)"
                )

with cc3:
    if tg["bot_token"] and tg["chat_id"]:
        st.success(f"✅ Telegram настроен (chat_id: {tg['chat_id']})")
    else:
        st.info("ℹ Telegram пока не настроен")


st.divider()


# =============================================================================
# О файлах настроек
# =============================================================================
st.markdown("## 📁 Где хранятся настройки")

cfg = load_config()
storage_path = Path.home() / ".silver_simple"

st.markdown(f"""
Все настройки и история сделок хранятся локально в папке:

```
{storage_path}
```

- `config.json` — капитал и Telegram
- `trades.json` — история ваших сделок

Никакие данные не отправляются на сторонние серверы.
""")

with st.expander("🔍 Показать содержимое конфигурации"):
    if cfg:
        safe = dict(cfg)
        if "tg_bot_token" in safe and safe["tg_bot_token"]:
            safe["tg_bot_token"] = safe["tg_bot_token"][:10] + "..."
        st.json(safe)
    else:
        st.info("Конфигурация пуста")


st.divider()


# =============================================================================
# Полная версия приложения
# =============================================================================
st.markdown("## 🔬 Полная версия")
st.markdown("""
Это **облегчённая** версия для повседневного использования. Есть также
**полная** версия с техническими деталями для аналитики:

- Метрики качества модели (Sharpe, DSR, walk-forward folds)
- Сравнение версий стратегии
- Drift-мониторинг признаков
- Управление портфелем через Tinkoff API
- Ручной запуск переобучения

Запуск полной версии (в отдельном терминале):

```bash
streamlit run dashboard_app.py
```
""")
