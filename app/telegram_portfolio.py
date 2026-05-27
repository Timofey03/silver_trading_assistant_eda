"""
app/telegram_portfolio.py — Telegram уведомление с PNG-графиком и
портфолио в стиле мульти-position трейдинга.

Использование:
    from app.telegram_portfolio import send_portfolio_chart
    send_portfolio_chart()
"""
from __future__ import annotations

import io
import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Use DejaVu Sans (поддерживает Cyrillic + ₽ U+20BD) вместо DejaVu Sans Mono
plt.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.dates as mdates
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SILVER_PARQUET = REPO_ROOT / "data" / "multi_asset" / "metals" / "silver_daily.parquet"
POSITIONS_FILE = REPO_ROOT / "argentum" / "backend" / "data" / "positions.json"

# Argentum dark theme
BG_BASE     = "#0a0a0b"
BG_ELEVATED = "#131316"
BORDER      = "#27272a"
TEXT_PRIM   = "#fafafa"
TEXT_MUTED  = "#71717a"
TEXT_FAINT  = "#52525b"
EMERALD     = "#10b981"
ROSE        = "#f43f5e"
AMBER       = "#f59e0b"
API_BASE = os.getenv("ARGENTUM_API", "http://127.0.0.1:8000")


def _api(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{API_BASE}{path}", timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def generate_portfolio_png() -> bytes:
    """
    PNG из 3 секций:
    1. Хедер с master-сигналом
    2. График silver за 30 дней + точки входа открытых позиций
    3. Таблица позиций с P&L и советом
    """
    # Get state from API
    sig = _api("/api/signal")
    pos = _api("/api/positions")
    positions = pos.get("positions", [])

    sig_value = sig.get("signal", "HOLD")
    # p_up берём из POSITIONS endpoint (master_p_up — синхронизирован с UI)
    # Fallback: signal endpoint (тоже smoothed теперь)
    p_up = float(pos.get("master_p_up", 0)) or float(sig.get("p_up", 0))
    sig_color = {"BUY": EMERALD, "SELL": ROSE, "HOLD": AMBER}.get(sig_value, TEXT_MUTED)
    sig_label = {"BUY": "ПОКУПАТЬ", "SELL": "ПРОДАВАТЬ", "HOLD": "ОЖИДАТЬ"}.get(sig_value, sig_value)

    # Silver prices (30 дней — компактнее, понятнее)
    prices = None
    if SILVER_PARQUET.exists():
        try:
            df = pd.read_parquet(SILVER_PARQUET)
            from datetime import datetime, timedelta
            cutoff = pd.Timestamp(datetime.now() - timedelta(days=30))
            prices = df[df.index >= cutoff].copy()
        except Exception:
            pass

    # Adaptive figure size — больше позиций = выше
    n_pos = len(positions)
    table_h = max(1.5, 0.4 * (n_pos + 2))    # rows × line height
    fig_h = 7 + table_h
    fig = plt.figure(figsize=(10, fig_h), dpi=120, facecolor=BG_BASE)

    # ─── Top header ───────────────────────────────────────────────
    ax_top = fig.add_axes([0.05, 1 - 1.4/fig_h, 0.90, 1.2/fig_h])
    ax_top.set_facecolor(BG_ELEVATED)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_color(BORDER); s.set_linewidth(1)

    ax_top.text(0.04, 0.55, sig_label, ha="left", va="center",
                fontsize=36, fontweight="bold", color=sig_color,
                family="monospace", transform=ax_top.transAxes)
    ax_top.text(0.04, 0.20,
                f"уверенность {int(p_up*100)}% · ${sig.get('close',0):.2f}/oz · "
                f"{sig.get('date','')}",
                ha="left", va="center", fontsize=10,
                color=TEXT_MUTED, family="monospace",
                transform=ax_top.transAxes)
    # Open positions count badge
    ax_top.text(0.96, 0.5,
                f"{n_pos}\n{'позиция' if n_pos==1 else 'позиций'}",
                ha="right", va="center", fontsize=12,
                color=EMERALD if n_pos else TEXT_FAINT,
                family="monospace", fontweight="bold",
                transform=ax_top.transAxes)

    # ─── Price chart ──────────────────────────────────────────────
    chart_top = 1 - 1.6/fig_h
    chart_bottom = (table_h + 0.5)/fig_h
    ax = fig.add_axes([0.08, chart_bottom, 0.88, chart_top - chart_bottom])
    ax.set_facecolor(BG_BASE)

    if prices is not None and len(prices):
        ax.plot(prices.index, prices["close"], color=TEXT_PRIM,
                linewidth=1.5, alpha=0.85)
        ax.fill_between(prices.index, prices["close"].min() * 0.97,
                        prices["close"], color=sig_color, alpha=0.06)

        # Position entry markers — каждый ордер своей точкой
        for pi, p in enumerate(positions):
            entry_date = pd.to_datetime(p["opened_at"])
            if entry_date < prices.index.min():
                continue
            # Convert RUB to USD/oz approx via current rate
            usd_silver = float(sig.get("close", 0))
            ax.axvline(x=entry_date, color=EMERALD, linestyle="--",
                       linewidth=0.8, alpha=0.5)
            ax.scatter(entry_date, usd_silver, s=120, marker="^",
                       color=EMERALD, edgecolors=BG_BASE, linewidths=1.5,
                       zorder=10)
            # Position label
            ax.annotate(f"#{pi+1}",
                        xy=(entry_date, usd_silver),
                        xytext=(0, -18), textcoords="offset points",
                        fontsize=8, color=EMERALD, ha="center",
                        fontweight="bold", family="monospace")

        # Current price line
        close = float(sig.get("close", 0))
        if close > 0:
            ax.axhline(y=close, color=sig_color, linestyle=":",
                       linewidth=1, alpha=0.5)
            ax.text(prices.index[-1], close, f"  ${close:.2f}",
                    color=sig_color, fontsize=9, va="center",
                    family="monospace", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_color(BORDER)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.grid(True, color=BORDER, linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.text(0.0, 1.02, "Silver USD · последние 30 дней · ▲ = твои входы",
            ha="left", va="bottom", fontsize=8, color=TEXT_FAINT,
            family="monospace", transform=ax.transAxes)

    # ─── Positions table ──────────────────────────────────────────
    ax_t = fig.add_axes([0.05, 0.05, 0.90, table_h/fig_h - 0.04])
    ax_t.set_facecolor(BG_ELEVATED)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    for s in ax_t.spines.values():
        s.set_color(BORDER); s.set_linewidth(1)

    # Header
    ax_t.text(0.5, 0.92, "Открытые позиции (каждая трекается независимо)",
              ha="center", va="top", fontsize=11, color=TEXT_PRIM,
              family="monospace", fontweight="bold",
              transform=ax_t.transAxes)

    if not positions:
        ax_t.text(0.5, 0.45, "нет открытых позиций",
                  ha="center", va="center", fontsize=10, color=TEXT_FAINT,
                  family="monospace", transform=ax_t.transAxes)
    else:
        # Columns: # · открыто · вход · сейчас · P&L · совет
        col_x = [0.05, 0.13, 0.30, 0.46, 0.62, 0.78]
        col_labels = ["#", "ОТКРЫТО", "ВХОД", "СЕЙЧАС", "P&L", "СОВЕТ"]
        for x, lbl in zip(col_x, col_labels):
            ax_t.text(x, 0.78, lbl, ha="left", va="center", fontsize=8,
                      color=TEXT_FAINT, family="monospace",
                      transform=ax_t.transAxes)

        for i, p in enumerate(positions):
            y = 0.65 - i * 0.13
            if y < 0.05: break
            pnl = float(p.get("unrealized_pnl_pct", 0))
            advice = p.get("advice", "—")
            advice_color = EMERALD if advice == "HOLD" else ROSE
            pnl_color = EMERALD if pnl >= 0 else ROSE

            ax_t.text(col_x[0], y, f"#{i+1}", ha="left", va="center",
                      fontsize=9, color=TEXT_PRIM, family="monospace",
                      transform=ax_t.transAxes)
            ax_t.text(col_x[1], y, str(p.get("opened_at", ""))[:10],
                      ha="left", va="center", fontsize=8,
                      color=TEXT_MUTED, family="monospace",
                      transform=ax_t.transAxes)
            ax_t.text(col_x[2], y, f"₽{p.get('entry_price', 0):,.0f}",
                      ha="left", va="center", fontsize=9,
                      color=TEXT_PRIM, family="monospace",
                      transform=ax_t.transAxes)
            ax_t.text(col_x[3], y, f"₽{p.get('current_price', 0):,.0f}",
                      ha="left", va="center", fontsize=9,
                      color=TEXT_PRIM, family="monospace",
                      transform=ax_t.transAxes)
            ax_t.text(col_x[4], y,
                      f"{'+' if pnl > 0 else ''}{pnl*100:.2f}%",
                      ha="left", va="center", fontsize=9,
                      color=pnl_color, family="monospace", fontweight="bold",
                      transform=ax_t.transAxes)
            ax_t.text(col_x[5], y,
                      ("ДЕРЖАТЬ" if advice == "HOLD" else "ПРОДАТЬ"),
                      ha="left", va="center", fontsize=9,
                      color=advice_color, family="monospace", fontweight="bold",
                      transform=ax_t.transAxes)

    # Wordmark
    fig.text(0.05, 0.965, "argentum", fontsize=10, color=TEXT_PRIM,
             family="monospace", fontweight="bold")
    fig.text(0.118, 0.965, ".", fontsize=10, color=EMERALD,
             family="monospace", fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG_BASE, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def send_portfolio_chart() -> bool:
    """Отправить PNG портфолио в Telegram с master signal + positions."""
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TG_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("TG creds missing")
        return False

    png = generate_portfolio_png()

    sig = _api("/api/signal")
    pos = _api("/api/positions")
    positions = pos.get("positions", [])
    n_open = len(positions)

    # === MASTER ASSISTANT verdict ===
    master_signal = pos.get("master_signal", "WAIT")
    master_p_up = float(pos.get("master_p_up", 0))
    master_emoji = {"BUY": "🟢", "WAIT": "🟡", "AVOID": "🔴"}.get(master_signal, "⚪")
    master_label = {
        "BUY":   "ОТКРЫТЬ новую позицию",
        "WAIT":  "ОЖИДАТЬ (сейчас не входить)",
        "AVOID": "НЕ ВХОДИТЬ (рынок против)",
    }.get(master_signal, master_signal)
    master_reason = pos.get("master_reason", "")

    caption = (
        f"🤖 <b>Главный помощник:</b> {master_emoji} <b>{master_label}</b>\n"
        f"📊 Уверенность: <b>{int(master_p_up*100)}%</b> "
        f"(strong filter ≥ 85%)\n"
        f"<i>{master_reason}</i>\n"
    )

    # === Per-position section ===
    if n_open == 0:
        caption += f"\n💼 <b>Открытых позиций нет</b>\n<i>Ждём сильный сигнал для входа</i>"
    else:
        avg_pnl = sum(float(p["unrealized_pnl_pct"]) for p in positions) / n_open * 100
        n_sell_advice = sum(1 for p in positions if p.get("advice") == "SELL")
        caption += (
            f"\n💼 <b>Портфолио: {n_open}</b> "
            f"{'позиция' if n_open == 1 else 'позиций'}\n"
            f"📈 Средний P&L: <b>{avg_pnl:+.2f}%</b>\n"
        )
        if n_sell_advice:
            caption += f"⚠ <b>{n_sell_advice}</b> с советом ПРОДАТЬ\n"
        else:
            caption += "✓ Все позиции рекомендуется ДЕРЖАТЬ\n"

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        r = requests.post(url, data={
            "chat_id": chat_id, "caption": caption, "parse_mode": "HTML",
        }, files={"photo": ("portfolio.png", png, "image/png")}, timeout=30)
        ok = r.status_code == 200
        if not ok:
            print(f"TG sendPhoto: {r.status_code} {r.text[:200]}")
        return ok
    except Exception as e:
        print(f"TG: {e}")
        return False


if __name__ == "__main__":
    import sys
    if "--png-only" in sys.argv:
        png = generate_portfolio_png()
        out = REPO_ROOT / "argentum" / "portfolio_preview.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png)
        print(f"PNG -> {out}")
    else:
        ok = send_portfolio_chart()
        print("Sent OK" if ok else "FAIL")
        sys.exit(0 if ok else 1)
