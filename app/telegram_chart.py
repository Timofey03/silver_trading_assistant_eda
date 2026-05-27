"""
app/telegram_chart.py — генерация PNG-чарта для Telegram-уведомлений.

Темная тема в стиле Argentum:
- 90 дней цены серебра
- BUY/SELL маркеры из истории сделок
- Текущий сигнал большой плашкой
- Метрики walk-forward внизу

Использование:
    from app.telegram_chart import send_signal_with_chart
    send_signal_with_chart(signal_info, metrics)
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SILVER_PARQUET = REPO_ROOT / "data" / "multi_asset" / "metals" / "silver_daily.parquet"
TRADES_CSV     = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv"

# Цвета — Argentum dark theme
BG_BASE     = "#0a0a0b"
BG_ELEVATED = "#131316"
BORDER      = "#27272a"
TEXT_PRIM   = "#fafafa"
TEXT_MUTED  = "#71717a"
TEXT_FAINT  = "#52525b"
EMERALD     = "#10b981"
ROSE        = "#f43f5e"
AMBER       = "#f59e0b"


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _load_silver_prices(days: int = 120) -> Optional[pd.DataFrame]:
    """Последние N дней цены серебра."""
    if not SILVER_PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(SILVER_PARQUET)
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
        return df[df.index >= cutoff].copy()
    except Exception:
        return None


def _load_recent_trades(days: int = 365) -> pd.DataFrame:
    """Сделки за последний год."""
    if not TRADES_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(TRADES_CSV)
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"]  = pd.to_datetime(df["exit_date"])
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
        return df[df["exit_date"] >= cutoff].copy()
    except Exception:
        return pd.DataFrame()


def generate_signal_chart_png(signal_info: dict, metrics: dict) -> bytes:
    """
    Генерирует красивый PNG-чарт сигнала в стиле Argentum.

    Возвращает PNG bytes (можно сразу слать в Telegram).
    """
    sig = signal_info.get("signal", "HOLD")
    p_up = float(signal_info.get("p_up", 0))
    close = float(signal_info.get("close", 0))
    sig_date = str(signal_info.get("date", "")).split("T")[0]
    is_repeat = bool(signal_info.get("is_repeat", False))
    prev = signal_info.get("previous_signal", "")

    sig_color = {"BUY": EMERALD, "SELL": ROSE, "HOLD": TEXT_MUTED}.get(sig, TEXT_MUTED)
    sig_label = {"BUY": "ПОКУПАТЬ", "SELL": "ПРОДАВАТЬ", "HOLD": "ОЖИДАТЬ"}.get(sig, sig)

    # Данные
    prices = _load_silver_prices(days=120)
    trades = _load_recent_trades(days=180)

    # --- Figure ---
    fig = plt.figure(figsize=(10, 7.5), dpi=120, facecolor=BG_BASE)

    # Top section — большая плашка с сигналом
    ax_top = fig.add_axes([0.05, 0.78, 0.90, 0.18])
    ax_top.set_facecolor(BG_ELEVATED)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(1)

    # Большой текст сигнала
    ax_top.text(
        0.04, 0.55, sig_label,
        ha="left", va="center",
        fontsize=44, fontweight="bold",
        color=sig_color, family="monospace",
        transform=ax_top.transAxes,
    )
    # Confidence + p_up
    ax_top.text(
        0.04, 0.18,
        f"Уверенность {int(p_up * 100)}%   ·   цена ${close:.2f}   ·   {sig_date}",
        ha="left", va="center",
        fontsize=11, color=TEXT_MUTED, family="monospace",
        transform=ax_top.transAxes,
    )
    # Top-right tag
    tag = "ПОВТОР" if is_repeat else ("ИЗМЕНЕНИЕ " + str(prev) + " → " + sig if prev else "НОВЫЙ")
    ax_top.text(
        0.96, 0.82, tag,
        ha="right", va="top",
        fontsize=8, color=TEXT_FAINT,
        family="monospace",
        transform=ax_top.transAxes,
    )

    # === Middle — price chart ===
    ax = fig.add_axes([0.08, 0.20, 0.88, 0.52])
    ax.set_facecolor(BG_BASE)

    if prices is not None and len(prices):
        ax.plot(
            prices.index, prices["close"],
            color=TEXT_PRIM, linewidth=1.5,
            alpha=0.85,
        )
        # Заливка ниже linии
        ax.fill_between(
            prices.index, prices["close"].min() * 0.97, prices["close"],
            color=sig_color, alpha=0.06,
        )

        # Маркеры сделок (только попавшие в окно prices)
        if len(trades):
            xmin, xmax = prices.index.min(), prices.index.max()
            in_win = trades[
                (trades["entry_date"] >= xmin) & (trades["entry_date"] <= xmax)
            ]
            for _, t in in_win.iterrows():
                # BUY marker
                ax.scatter(
                    t["entry_date"], t["entry_price"],
                    s=80, marker="^", color=EMERALD,
                    edgecolors=BG_BASE, linewidths=1.5, zorder=10,
                )
                # SELL marker (если exit в окне)
                if t["exit_date"] <= xmax:
                    ret = float(t["net_return"])
                    sell_color = EMERALD if ret > 0 else ROSE
                    ax.scatter(
                        t["exit_date"], t["exit_price"],
                        s=80, marker="v", color=sell_color,
                        edgecolors=BG_BASE, linewidths=1.5, zorder=10,
                    )
                    # Текст с результатом
                    ax.annotate(
                        f"{ret * 100:+.1f}%",
                        xy=(t["exit_date"], t["exit_price"]),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=8, color=sell_color,
                        family="monospace",
                        fontweight="bold",
                    )

        # Текущая цена — пунктирная линия
        ax.axhline(y=close, color=sig_color, linestyle=":", linewidth=1, alpha=0.5)
        ax.text(
            prices.index[-1], close,
            f"  ${close:.2f}",
            color=sig_color, fontsize=9, va="center",
            family="monospace", fontweight="bold",
        )

    # Стилизация осей
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_color(BORDER)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.grid(True, color=BORDER, linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Заголовок графика
    ax.text(
        0.0, 1.02,
        "SI=F · Silver Futures · последние 120 дней",
        ha="left", va="bottom",
        fontsize=9, color=TEXT_FAINT,
        family="monospace",
        transform=ax.transAxes,
    )

    # === Bottom metrics strip ===
    ax_bot = fig.add_axes([0.05, 0.04, 0.90, 0.10])
    ax_bot.set_facecolor(BG_ELEVATED)
    ax_bot.set_xticks([])
    ax_bot.set_yticks([])
    for spine in ax_bot.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(1)

    sharpe = metrics.get("sharpe", 0)
    win = metrics.get("win_rate", 0) * 100
    n_tr = metrics.get("n_trades", 0)
    total = metrics.get("total_return", 0) * 100
    # max_dd хранится как отрицательное число (-0.2976 = -29.76%)
    max_dd_raw = metrics.get("max_dd", metrics.get("max_drawdown", 0))
    max_dd = abs(max_dd_raw) * 100

    cols = [
        ("Sharpe", f"{sharpe:.2f}", TEXT_PRIM),
        ("Win Rate", f"{win:.0f}%", TEXT_PRIM),
        ("Сделок", str(n_tr), TEXT_PRIM),
        ("Доходность", f"{total:+.1f}%", EMERALD if total > 0 else ROSE),
        ("Просадка", f"−{max_dd:.1f}%", ROSE),
    ]
    n_cols = len(cols)
    for i, (label, value, vcolor) in enumerate(cols):
        x = 0.04 + i * (0.92 / n_cols)
        ax_bot.text(
            x + 0.01, 0.72, label,
            ha="left", va="center",
            fontsize=7, color=TEXT_FAINT,
            family="monospace",
            transform=ax_bot.transAxes,
        )
        ax_bot.text(
            x + 0.01, 0.30, value,
            ha="left", va="center",
            fontsize=15, color=vcolor,
            family="monospace", fontweight="bold",
            transform=ax_bot.transAxes,
        )

    # argentum wordmark в углу
    fig.text(
        0.05, 0.965,
        "argentum",
        fontsize=10, color=TEXT_PRIM,
        family="monospace", fontweight="bold",
    )
    fig.text(
        0.118, 0.965, ".",
        fontsize=10, color=EMERALD,
        family="monospace", fontweight="bold",
    )
    fig.text(
        0.95, 0.965,
        f"Модель E3b · {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=7, color=TEXT_FAINT, family="monospace",
        ha="right",
    )

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG_BASE, dpi=120, bbox_inches=None)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Telegram sender
# ---------------------------------------------------------------------------

def send_signal_with_chart(signal_info: dict, metrics: dict) -> bool:
    """
    Отправляет PNG-чарт + краткий caption в Telegram.
    """
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TG_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("  (TG credentials отсутствуют, skip)")
        return False

    try:
        png = generate_signal_chart_png(signal_info, metrics)
    except Exception as e:
        print(f"  Chart generation failed: {e}")
        return False

    sig = signal_info.get("signal", "HOLD")
    p_up = float(signal_info.get("p_up", 0))
    close = float(signal_info.get("close", 0))
    is_repeat = bool(signal_info.get("is_repeat", False))
    prev = signal_info.get("previous_signal", "")

    sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig, "⚪")

    if is_repeat:
        caption = (
            f"{sig_emoji} <b>{sig}</b> · напоминание\n"
            f"Уверенность <b>{int(p_up * 100)}%</b> · ${close:.2f}\n"
            f"<i>Если уже отреагировал — действий не требуется</i>"
        )
    elif prev and prev != sig:
        caption = (
            f"{sig_emoji} <b>{prev} → {sig}</b>\n"
            f"Уверенность <b>{int(p_up * 100)}%</b> · ${close:.2f}\n"
            f"Sharpe {metrics.get('sharpe', 0):.2f} · "
            f"Win {metrics.get('win_rate', 0) * 100:.0f}% · "
            f"Trades {metrics.get('n_trades', 0)}"
        )
    else:
        caption = (
            f"{sig_emoji} <b>Новый сигнал: {sig}</b>\n"
            f"Уверенность <b>{int(p_up * 100)}%</b> · ${close:.2f}"
        )

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        r = requests.post(
            url,
            data={
                "chat_id":    chat_id,
                "caption":    caption,
                "parse_mode": "HTML",
            },
            files={"photo": ("signal.png", png, "image/png")},
            timeout=30,
        )
        ok = r.status_code == 200
        if not ok:
            print(f"  Telegram sendPhoto failed: {r.status_code} {r.text[:200]}")
        return ok
    except Exception as e:
        print(f"  Telegram sendPhoto exception: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI: python -m app.telegram_chart  → отправить демо-чарт
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Подгружаем последний сигнал и метрики из repo
    sig_file = REPO_ROOT / "daily_reports" / "e3b" / "trading" / "2026-05-22" / "signal.json"
    met_file = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "metrics.json"

    if not sig_file.exists():
        # Поиск любого последнего signal.json
        trading_root = REPO_ROOT / "daily_reports" / "e3b" / "trading"
        if trading_root.exists():
            dirs = sorted([d for d in trading_root.iterdir() if d.is_dir()], reverse=True)
            for d in dirs:
                if (d / "signal.json").exists():
                    sig_file = d / "signal.json"
                    break

    if not sig_file.exists():
        print("ERROR: signal.json не найден")
        sys.exit(1)

    sig = json.loads(sig_file.read_text(encoding="utf-8"))
    met = {}
    if met_file.exists():
        met = json.loads(met_file.read_text(encoding="utf-8"))

    # Если "--png-only" — сохраняем картинку в файл и не шлём
    if "--png-only" in sys.argv:
        png = generate_signal_chart_png(sig, met)
        out = REPO_ROOT / "argentum" / "telegram_preview.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png)
        print(f"PNG saved -> {out}")
    else:
        ok = send_signal_with_chart(sig, met)
        print("Sent OK" if ok else "Send FAIL")
        sys.exit(0 if ok else 1)
