"""scripts/test_telegram_e3b.py — тестовое сообщение от E3b в Telegram.

Использует существующий сигнал из daily_reports/e3b/trading/<latest>/,
не запускает retraining. Если daily report ещё не создан — берёт
данные из baseline_outputs_multiasset/e3b_adaptive/metrics.json.

Запуск:
  python scripts/test_telegram_e3b.py
  python scripts/test_telegram_e3b.py --force   # пошлём даже если последний сигнал старый
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

E3B_TRADING = REPO_ROOT / "daily_reports" / "e3b" / "trading"
E3B_BASELINE = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive"


def load_latest_signal() -> dict | None:
    """Загрузить последний E3b сигнал."""
    if not E3B_TRADING.exists():
        return None
    dirs = sorted([d for d in E3B_TRADING.iterdir() if d.is_dir()], reverse=True)
    for d in dirs:
        sig_file = d / "signal.json"
        if sig_file.exists():
            data = json.loads(sig_file.read_text(encoding="utf-8"))
            data["report_dir"] = d.name
            return data
    return None


def load_walkforward_metrics() -> dict:
    p = E3B_BASELINE / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def send_telegram(text: str, token: str, chat_id: str,
                  parse_mode: str = "Markdown") -> tuple[bool, str]:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode("utf-8")
            return (r.status == 200, body)
    except urllib.error.HTTPError as e:
        return (False, f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")


def format_message(signal: dict, metrics: dict) -> str:
    """Красивое Telegram-сообщение в HTML (проще чем MarkdownV2)."""
    sig = signal.get("signal", "HOLD")
    sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")

    sig_date = signal.get("date", "—")
    if isinstance(sig_date, str) and "T" in sig_date:
        sig_date = sig_date.split("T")[0]

    close = float(signal.get("close", 0))
    p_up = float(signal.get("p_up", 0))
    thr = float(signal.get("entry_threshold", 0.48))

    confidence_bar = "█" * int(p_up * 10) + "░" * (10 - int(p_up * 10))

    sharpe = metrics.get("sharpe", 0)
    win_rate = metrics.get("win_rate", 0) * 100
    n_trades = metrics.get("n_trades", 0)
    annual = metrics.get("annual_return", 0) * 100
    max_dd = metrics.get("max_dd", 0) * 100

    return f"""🧪 <b>Test E3b — Silver Assistant</b>

{sig_emoji} <b>Сигнал: {sig}</b>

📅 Дата:    <code>{sig_date}</code>
💰 Цена:    <code>${close:.2f}</code>
🎯 Порог:   <code>{thr:.2f}</code>
📊 p_up:    <code>{p_up:.4f}</code>

<code>{confidence_bar}</code> {p_up*100:.1f}%

<i>Walk-forward backtest:</i>
• Sharpe:        <code>{sharpe:.3f}</code>
• Annual return: <code>{annual:+.1f}%</code>
• Win rate:      <code>{win_rate:.0f}%</code>
• Trades:        <code>{n_trades}</code>
• Max DD:        <code>{max_dd:.1f}%</code>

📂 Источник: <code>{signal.get("report_dir", "—")}</code>
🤖 Модель:  <b>E3b</b> (cross-asset + adaptive barriers)

<i>Это тестовое сообщение для проверки интеграции.</i>"""


def main():
    parser = argparse.ArgumentParser(description="Test Telegram notification for E3b")
    parser.add_argument("--force", action="store_true",
                        help="Послать даже если сигнал старше 5 дней")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать сообщение, не отправлять")
    args = parser.parse_args()

    print("=" * 60)
    print(" E3b Telegram Test")
    print("=" * 60)

    # 1. Credentials
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TG_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("\n❌ TG_BOT_TOKEN или TG_CHAT_ID не заданы в .env")
        print("\nЗапусти: python scripts/telegram_setup.py")
        sys.exit(1)
    print(f"✓ Token: {token[:10]}…{token[-6:]}")
    print(f"✓ Chat ID: {chat_id}")

    # 2. Загружаем сигнал
    signal = load_latest_signal()
    if signal is None:
        print("\n❌ Нет E3b daily report. Запусти: python scripts/daily_e3b.py")
        sys.exit(1)
    print(f"✓ Сигнал найден: {signal['report_dir']}")
    print(f"   Date: {signal.get('date')}, Signal: {signal.get('signal')}, "
          f"p_up: {signal.get('p_up'):.4f}")

    # 3. Проверка свежести
    sig_date_str = signal.get("date", "").split("T")[0]
    try:
        sig_date = pd.Timestamp(sig_date_str)
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        age = (today - sig_date).days
        print(f"   Возраст: {age} дней")
        if age >= 5 and not args.force:
            print(f"\n⚠ Сигнал старше 5 дней. Используй --force чтобы отправить.")
            sys.exit(1)
    except Exception:
        pass

    # 4. Walk-forward metrics
    metrics = load_walkforward_metrics()
    if metrics:
        print(f"✓ Backtest metrics: Sharpe {metrics.get('sharpe', 0):.3f}, "
              f"trades {metrics.get('n_trades', 0)}")

    # 5. Формируем сообщение
    text = format_message(signal, metrics)
    print("\n" + "=" * 60)
    print(" Сообщение:")
    print("=" * 60)
    print(text)
    print()

    if args.dry_run:
        print("[DRY-RUN] Не отправляем.")
        return

    # 6. Отправка
    print("=" * 60)
    print(" Отправка...")
    print("=" * 60)
    ok, response = send_telegram(text, token, chat_id, parse_mode="HTML")
    if ok:
        print(f"\n✅ Telegram отправил сообщение успешно!")
        print(f"   Проверь Telegram chat {chat_id}")
    else:
        print(f"\n❌ Ошибка отправки: {response[:300]}")
        # Retry с обычным Markdown как fallback
        print("\nПробую отправить без Markdown форматирования...")
        plain = (
            f"E3b Silver Assistant — Test\n\n"
            f"Signal: {signal.get('signal')}\n"
            f"Date: {sig_date_str}\n"
            f"Close: ${signal.get('close', 0):.2f}\n"
            f"p_up: {signal.get('p_up', 0):.4f}\n\n"
            f"Walk-forward backtest:\n"
            f"  Sharpe: {metrics.get('sharpe', 0):.3f}\n"
            f"  Win rate: {metrics.get('win_rate', 0)*100:.0f}%\n"
            f"  Trades: {metrics.get('n_trades', 0)}\n"
            f"  Annual: {metrics.get('annual_return', 0)*100:+.1f}%\n\n"
            f"Test message for integration check."
        )
        ok2, resp2 = send_telegram(plain, token, chat_id, parse_mode="")
        if ok2:
            print(f"✅ Простое сообщение отправлено")
        else:
            print(f"❌ Тоже не получилось: {resp2[:300]}")
            sys.exit(1)


if __name__ == "__main__":
    main()
