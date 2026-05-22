"""
app/simple_storage.py — локальное хранилище для облегчённого приложения.

Хранит в ~/.silver_simple/:
  config.json   — капитал пользователя, Telegram creds
  trades.json   — пользовательские отметки "открыл / закрыл сделку"
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


STORAGE_DIR = Path.home() / ".silver_simple"
CONFIG_PATH = STORAGE_DIR / "config.json"
TRADES_PATH = STORAGE_DIR / "trades.json"


def _ensure_dir() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Config (капитал, Telegram)
# =============================================================================

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    _ensure_dir()
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_capital() -> float:
    return float(load_config().get("capital_rub", 0))


def set_capital(value: float) -> None:
    cfg = load_config()
    cfg["capital_rub"] = float(value)
    save_config(cfg)


def get_telegram() -> dict:
    cfg = load_config()
    return {
        "bot_token": cfg.get("tg_bot_token", ""),
        "chat_id":   cfg.get("tg_chat_id",   ""),
    }


def set_telegram(bot_token: str, chat_id: str) -> None:
    cfg = load_config()
    cfg["tg_bot_token"] = bot_token.strip()
    cfg["tg_chat_id"]   = chat_id.strip()
    save_config(cfg)


# =============================================================================
# Trades log (пользовательские отметки)
# =============================================================================

def _load_trades_raw() -> list:
    if not TRADES_PATH.exists():
        return []
    try:
        data = json.loads(TRADES_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_trades_raw(trades: list) -> None:
    _ensure_dir()
    TRADES_PATH.write_text(
        json.dumps(trades, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_open_trade() -> Optional[dict]:
    """Возвращает открытую сделку, если она есть."""
    for t in _load_trades_raw():
        if t.get("status") == "open":
            return t
    return None


def get_all_trades() -> list[dict]:
    return _load_trades_raw()


def add_open_trade(entry_price: float, lots: int, signal_date: str,
                   trail_pct: float, max_hold_days: int) -> dict:
    """Регистрирует факт открытия позиции."""
    trades = _load_trades_raw()
    if any(t.get("status") == "open" for t in trades):
        raise ValueError("Уже есть открытая позиция — сначала закройте её")
    trade = {
        "id":              datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status":          "open",
        "entry_date":      datetime.now().strftime("%Y-%m-%d"),
        "entry_price":     float(entry_price),
        "lots":            int(lots),
        "signal_date":     signal_date,
        "trail_pct":       float(trail_pct),
        "max_hold_days":   int(max_hold_days),
        "max_price_seen":  float(entry_price),  # для trailing stop
    }
    trades.append(trade)
    _save_trades_raw(trades)
    return trade


def close_open_trade(exit_price: float, reason: str = "manual") -> Optional[dict]:
    """Закрывает текущую открытую сделку."""
    trades = _load_trades_raw()
    for t in trades:
        if t.get("status") == "open":
            t["status"]     = "closed"
            t["exit_date"]  = datetime.now().strftime("%Y-%m-%d")
            t["exit_price"] = float(exit_price)
            t["close_reason"] = reason
            # P&L расчёт (для SLVRUBF: 1 лот = 100 единиц)
            t["pnl_pct"] = (exit_price / t["entry_price"] - 1) * 100
            _save_trades_raw(trades)
            return t
    return None


def update_max_price(current_price: float) -> Optional[dict]:
    """Обновляет максимум цены для trailing stop. Возвращает обновлённую сделку."""
    trades = _load_trades_raw()
    for t in trades:
        if t.get("status") == "open":
            if current_price > t.get("max_price_seen", 0):
                t["max_price_seen"] = float(current_price)
                _save_trades_raw(trades)
            return t
    return None


def remove_trade(trade_id: str) -> bool:
    trades = _load_trades_raw()
    new_trades = [t for t in trades if t.get("id") != trade_id]
    if len(new_trades) == len(trades):
        return False
    _save_trades_raw(new_trades)
    return True
