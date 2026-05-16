"""
scripts/telegram_setup.py — интерактивная настройка Telegram-бота

Использование:
  1. Создать бота: написать @BotFather → /newbot → получить TOKEN
  2. Найти своего бота, написать ему /start
  3. Запустить: python scripts/telegram_setup.py
  4. Скрипт сам найдёт ваш chat_id и пропишет в .env

Альтернатива (вручную):
  https://api.telegram.org/bot<TOKEN>/getUpdates → найти "chat":{"id": ...}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"


def main() -> None:
    print("=" * 60)
    print(" Telegram bot setup для Silver Assistant")
    print("=" * 60)

    # Step 1: Получаем токен
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    if not token:
        print("\n📋 Шаг 1: создайте бота")
        print("   1. Откройте @BotFather в Telegram")
        print("   2. Команда: /newbot")
        print("   3. Введите имя (например: My Silver Helper)")
        print("   4. Введите username (например: my_silver_helper_bot)")
        print("   5. Скопируйте Bot Token (формат 1234567890:ABC...)")
        print()
        token = input("🔑 Вставьте TOKEN бота: ").strip()
        if not token:
            print("❌ Токен пустой. Отмена.")
            sys.exit(1)

    # Validate token
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        if r.status_code != 200:
            print(f"❌ Невалидный токен: HTTP {r.status_code}")
            sys.exit(1)
        bot_info = r.json().get("result", {})
        print(f"✅ Токен валиден. Bot: @{bot_info.get('username', '?')} ({bot_info.get('first_name', '?')})")
    except Exception as e:
        print(f"❌ Ошибка проверки токена: {e}")
        sys.exit(1)

    # Step 2: Получаем chat_id
    chat_id = os.getenv("TG_CHAT_ID", "").strip()
    if not chat_id:
        print("\n📋 Шаг 2: напишите своему боту")
        print(f"   1. Откройте @{bot_info.get('username', '<your_bot>')} в Telegram")
        print("   2. Нажмите Start или напишите любое сообщение")
        print("   3. Затем нажмите Enter здесь, чтобы я нашёл ваш chat_id")
        input("\n⏎ Готово? Нажмите Enter...")

        try:
            r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
            data = r.json()
            if not data.get("ok"):
                print(f"❌ Ошибка: {data}")
                sys.exit(1)
            updates = data.get("result", [])
            if not updates:
                print("❌ Нет сообщений от вас. Напишите боту что-нибудь и запустите снова.")
                sys.exit(1)
            chat_id = str(updates[-1].get("message", {}).get("chat", {}).get("id", ""))
            if not chat_id:
                print("❌ Не нашёл chat_id в getUpdates")
                sys.exit(1)
            user = updates[-1].get("message", {}).get("from", {})
            print(f"✅ chat_id найден: {chat_id}")
            print(f"   Пользователь: {user.get('first_name', '')} (@{user.get('username', '?')})")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            sys.exit(1)

    # Step 3: Test message
    print(f"\n📨 Шаг 3: тестовое сообщение...")
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={
            "chat_id": chat_id,
            "text":    "🥈 *Silver Assistant подключён!*\n\nС этого момента вы будете получать уведомления о BUY/SELL сигналах.",
            "parse_mode": "Markdown",
        }, timeout=10)
        if r.status_code == 200:
            print("✅ Сообщение отправлено! Проверьте Telegram.")
        else:
            print(f"⚠ Ошибка отправки: HTTP {r.status_code}")
            print(r.text)
    except Exception as e:
        print(f"⚠ Ошибка: {e}")

    # Step 4: Сохраняем в .env
    print(f"\n💾 Шаг 4: записываю в {ENV_PATH}")
    current = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                current[k.strip()] = v.strip()
    current["TG_BOT_TOKEN"] = token
    current["TG_CHAT_ID"]   = chat_id

    lines = [f"{k}={v}" for k, v in current.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ Сохранено в {ENV_PATH}")

    print()
    print("=" * 60)
    print(" Setup завершён!")
    print("=" * 60)
    print()
    print("Следующие шаги:")
    print(f"  1. Для GitHub Actions добавьте 2 secrets:")
    print(f"     TG_BOT_TOKEN = {token[:20]}...")
    print(f"     TG_CHAT_ID   = {chat_id}")
    print(f"     → https://github.com/Timofey03/silver_trading_assistant_eda/settings/secrets/actions")
    print(f"  2. Перезапустите Streamlit чтобы он увидел новый .env")
    print(f"  3. В Настройках появится кнопка 'Test notification'")


if __name__ == "__main__":
    main()
