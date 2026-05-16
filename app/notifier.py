"""
app/notifier.py — Telegram уведомления

Используется в:
- scripts/daily_run.py (отправка при BUY/SELL/error)
- pages/5_⚙_Настройки.py (test notification)

Конфиг через env:
  TG_BOT_TOKEN — токен бота от @BotFather
  TG_CHAT_ID   — chat_id пользователя
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests


class TelegramNotifier:
    """Минималистичный Telegram отправитель.

    Если TG_BOT_TOKEN или TG_CHAT_ID не заданы — fail silently
    (не ломаем daily run если уведомления не настроены).
    """

    def __init__(self,
                 token: Optional[str] = None,
                 chat_id: Optional[str] = None) -> None:
        self.token = token or os.getenv("TG_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.getenv("TG_CHAT_ID", "").strip()
        self.enabled = bool(self.token and self.chat_id)

    def send(self, text: str, parse_mode: str = "Markdown",
             disable_preview: bool = True) -> bool:
        """Отправляет сообщение. Возвращает True при успехе."""
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            r = requests.post(url, data={
                "chat_id":                  self.chat_id,
                "text":                     text,
                "parse_mode":               parse_mode,
                "disable_web_page_preview": disable_preview,
            }, timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────
    # Convenience methods для типичных событий
    # ──────────────────────────────────────────────────────────────────

    def notify_buy(self, ticker: str, p_up: float, price: float,
                   lots: int, order_id: str,
                   trend_5d: Optional[float] = None,
                   regime: str = "") -> bool:
        trend = f"\n📈 Тренд 5d: {trend_5d:.0%}" if trend_5d is not None else ""
        regime_line = f"\n🌐 Режим: `{regime}`" if regime else ""
        msg = (
            f"🟢 *BUY {ticker}*\n\n"
            f"💰 Куплено: {lots} лот{'а' if 1 < lots < 5 else 'ов' if lots > 4 else ''} "
            f"@ {price:.2f}\n"
            f"🎯 Уверенность модели: *{p_up:.0%}*"
            f"{trend}"
            f"{regime_line}\n"
            f"🆔 `{order_id[:8]}...`\n\n"
            f"✅ Ордер исполнен в Tinkoff sandbox"
        )
        return self.send(msg)

    def notify_sell(self, ticker: str, p_up: float, price: float,
                    lots: int, order_id: str,
                    reason: str = "") -> bool:
        reason_line = f"\n📉 Причина: {reason}" if reason else ""
        msg = (
            f"🔴 *SELL {ticker}*\n\n"
            f"💸 Продано: {lots} лот{'а' if 1 < lots < 5 else 'ов' if lots > 4 else ''} "
            f"@ {price:.2f}\n"
            f"📊 Уверенность модели: *{p_up:.0%}*"
            f"{reason_line}\n"
            f"🆔 `{order_id[:8]}...`\n\n"
            f"✅ Позиция закрыта"
        )
        return self.send(msg)

    def notify_hold(self, ticker: str, p_up: float,
                    cooldown_remaining: int = 0,
                    daily_summary: bool = True) -> bool:
        """Опциональное HOLD уведомление (обычно не нужно — спам)."""
        if not daily_summary:
            return False
        if cooldown_remaining > 0:
            extra = f"⏳ Cooldown ещё {cooldown_remaining}d"
        else:
            extra = "Ждём сигнал"
        msg = (
            f"⚪ *HOLD {ticker}*\n\n"
            f"p_up = {p_up:.0%}\n"
            f"{extra}"
        )
        return self.send(msg)

    def notify_error(self, where: str, error: str) -> bool:
        msg = (
            f"❌ *Ошибка в daily run*\n\n"
            f"📍 Где: `{where}`\n"
            f"💬 {error[:300]}"
        )
        return self.send(msg, parse_mode="Markdown")

    def notify_stale_data(self, days: int) -> bool:
        msg = (
            f"⚠ *Данные устарели*\n\n"
            f"Последние данные — {days}d назад.\n"
            f"Сигнал на сегодня **не доверять**.\n"
            f"Запустите Refresh signal в UI."
        )
        return self.send(msg)

    def notify_daily_summary(self, silver_sig: dict, gold_sig: dict,
                              portfolio: dict) -> bool:
        """Сводка раз в день (опционально)."""
        s_sig = silver_sig.get("signal", "—") if silver_sig else "—"
        s_pup = silver_sig.get("p_up", 0) if silver_sig else 0
        g_sig = gold_sig.get("signal", "—") if gold_sig else "—"
        g_pup = gold_sig.get("p_up", 0) if gold_sig else 0
        total = (portfolio.get("totalAmountPortfolio", {}).get("units", "?")
                 if portfolio else "?")

        msg = (
            f"📊 *Daily Summary*\n\n"
            f"🥈 Silver: {s_sig} (p_up = {s_pup:.0%})\n"
            f"🥇 Gold:   {g_sig} (p_up = {g_pup:.0%})\n\n"
            f"💼 Портфель: {total} RUB"
        )
        return self.send(msg)


def quick_send(text: str) -> bool:
    """One-liner для быстрой отправки из других модулей."""
    return TelegramNotifier().send(text)


if __name__ == "__main__":
    # CLI тест: python -m app.notifier "Hello from silver assistant"
    n = TelegramNotifier()
    if not n.enabled:
        print("ERROR: TG_BOT_TOKEN или TG_CHAT_ID не заданы в .env")
        sys.exit(2)
    text = sys.argv[1] if len(sys.argv) > 1 else "🥈 Test from Silver Assistant"
    ok = n.send(text)
    print(f"Отправлено: {ok}")
    sys.exit(0 if ok else 1)
