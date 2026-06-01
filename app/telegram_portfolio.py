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
REPO_ROOT_LOCAL = Path(__file__).resolve().parent.parent


def _read_signal_from_files() -> dict:
    """Прямое чтение signal.json без backend — для GitHub Actions."""
    trading = REPO_ROOT_LOCAL / "daily_reports" / "e3b" / "trading"
    if not trading.exists():
        return {}
    dirs = sorted([d for d in trading.iterdir() if d.is_dir()], reverse=True)
    # Smoothed по 3 последним
    sigs = []
    for d in dirs[:3]:
        f = d / "signal.json"
        if f.exists():
            try:
                sigs.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                continue
    if not sigs:
        return {}
    latest = sigs[0]
    p_ups = [float(s.get("p_up", 0)) for s in sigs if s.get("p_up") is not None]
    smoothed = sum(p_ups) / len(p_ups) if p_ups else 0
    date_str = str(latest.get("date", ""))
    if "T" in date_str:
        date_str = date_str.split("T")[0]
    return {
        "signal":  latest.get("signal", "HOLD"),
        "date":    date_str,
        "close":   float(latest.get("close", 0)),
        "p_up":    smoothed,   # smoothed для consistency с UI
    }


def _tinkoff_money(m: Optional[dict]) -> float:
    """Tinkoff Quotation/MoneyValue → float."""
    if not m:
        return 0.0
    units = int(m.get("units", 0) or 0)
    nano = int(m.get("nano", 0) or 0)
    return units + nano / 1e9


def _tinkoff_call(service: str, method: str, body: dict, timeout: int = 15) -> dict:
    """POST к Tinkoff Invest REST API v2. Бросает RuntimeError при ошибке."""
    token = os.getenv("TINKOFF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TINKOFF_TOKEN не задан")
    url = f"https://invest-public-api.tinkoff.ru/rest/{service}/{method}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    r = requests.post(url, data=json.dumps(body), headers=headers, timeout=timeout)
    data = r.json() if r.text else {}
    if r.status_code >= 400:
        raise RuntimeError(f"Tinkoff API {r.status_code}: {data}")
    return data


def _read_positions_from_tinkoff() -> list[dict]:
    """Реальные позиции из Tinkoff sandbox портфолио.

    Возвращает список dict в формате /api/positions:
    [{"id", "ticker", "figi", "entry_price", "lots", "current_price",
      "unrealized_pnl_pct", "advice", ...}]

    При ошибке (нет токена, нет сети) возвращает [].
    """
    SANDBOX = "tinkoff.public.invest.api.contract.v1.SandboxService"
    try:
        # 1. Берём первый sandbox-аккаунт
        accs = _tinkoff_call(SANDBOX, "GetSandboxAccounts", {}).get("accounts", [])
        if not accs:
            return []
        acc_id = accs[0]["id"]

        # 2. Запрашиваем портфолио
        portfolio = _tinkoff_call(SANDBOX, "GetSandboxPortfolio", {"accountId": acc_id})

        # 3. Маппим позиции (исключая валюту)
        positions = []
        for i, p in enumerate(portfolio.get("positions", [])):
            if p.get("instrumentType") == "currency":
                continue
            qty = _tinkoff_money(p.get("quantity"))
            if qty <= 0:
                continue
            avg_price = _tinkoff_money(p.get("averagePositionPrice"))
            current_price = _tinkoff_money(p.get("currentPrice"))
            pnl_pct = ((current_price - avg_price) / avg_price) if avg_price > 0 else 0.0

            figi = p.get("figi", "")
            ticker = "SLVRUBF" if figi == "FSLVRUB00000" else figi

            positions.append({
                "id":               f"tinkoff_{i}",
                "ticker":           ticker,
                "figi":             figi,
                "entry_price":      avg_price,
                "lots":             int(qty),
                "lot_size_g":       100 if ticker == "SLVRUBF" else 1,
                "current_price":    current_price,
                "unrealized_pnl_pct": pnl_pct,
                "advice":           "HOLD",
                "advice_reason":    "из Tinkoff sandbox",
                "source":           "tinkoff_sandbox",
            })
        return positions
    except Exception as e:
        print(f"[telegram] Tinkoff fallback failed: {e}")
        return []


def _theoretical_rub_price(silver_usd_close: float, usdrub: float,
                            lot_size_g: int = 100) -> float:
    """Цена 1 лота в рублях из формулы: USD цена силвера × USDRUB × г/oz.

    Формула повторяет логику argentum/backend/routers/positions.py
    (_theoretical_rub_price). 1 oz = 31.1035 г.
    """
    if silver_usd_close <= 0 or usdrub <= 0:
        return 0.0
    return silver_usd_close * usdrub * lot_size_g / 31.1035


def _compute_position_pnl(position: dict) -> dict:
    """Рассчитать current_price, market_pnl_pct и advice для позиции
    локально, без backend. Используется в fallback-режиме (cron).

    Читает silver_daily.parquet и USDRUB.parquet, считает теоретическую
    рублёвую цену на дату входа и сегодня, выводит P&L.
    """
    out = dict(position)  # копия
    silver_path = REPO_ROOT_LOCAL / "data" / "multi_asset" / "metals" / "silver_daily.parquet"
    usdrub_path = REPO_ROOT_LOCAL / "data" / "multi_asset" / "macro" / "USDRUB.parquet"

    if not silver_path.exists() or not usdrub_path.exists():
        return out

    try:
        silver = pd.read_parquet(silver_path)
        usdrub = pd.read_parquet(usdrub_path)

        # Авто-детект колонки value у USDRUB
        usdrub_col = None
        for c in usdrub.columns:
            if pd.api.types.is_numeric_dtype(usdrub[c]):
                usdrub_col = c
                break
        if usdrub_col is None:
            return out

        # Подготовить серии по датам
        silver_ts = silver["close"].copy()
        usdrub_ts = usdrub[usdrub_col].copy()

        # Сегодняшние значения (последняя доступная свеча)
        s_today = float(silver_ts.iloc[-1])
        u_today = float(usdrub_ts.iloc[-1])

        # Цена на дату входа
        opened_at = position.get("opened_at", "")
        if "T" in str(opened_at):
            entry_date = pd.Timestamp(str(opened_at).split("T")[0])
        else:
            entry_date = pd.Timestamp(str(opened_at))

        s_entry_series = silver_ts[silver_ts.index <= entry_date]
        u_entry_series = usdrub_ts[usdrub_ts.index <= entry_date]
        s_entry = float(s_entry_series.iloc[-1]) if len(s_entry_series) else s_today
        u_entry = float(u_entry_series.iloc[-1]) if len(u_entry_series) else u_today

        lot_size_g = int(position.get("lot_size_g", 100))
        market_entry = _theoretical_rub_price(s_entry, u_entry, lot_size_g)
        market_now = _theoretical_rub_price(s_today, u_today, lot_size_g)

        # current_price — теоретическая текущая (рыночная)
        out["current_price"] = market_now
        out["market_entry_price"] = market_entry
        out["market_current_price"] = market_now

        # P&L: 2 варианта
        entry_user = float(position.get("entry_price", 0))  # что пользователь записал (Tinkoff sandbox)
        if entry_user > 0 and market_now > 0:
            out["unrealized_pnl_pct"] = (market_now - entry_user) / entry_user
        if market_entry > 0 and market_now > 0:
            out["market_pnl_pct"] = (market_now - market_entry) / market_entry

        # Простой совет на основе peak / trailing stop
        peak = float(position.get("peak_price", entry_user))
        if market_now > peak:
            peak = market_now
        out["peak_price"] = peak
        trail_pct = 0.20
        trail_stop = peak * (1 - trail_pct)
        if market_now < trail_stop:
            out["advice"] = "SELL"
            out["advice_reason"] = f"trailing stop ₽{trail_stop:.0f} пробит"
        else:
            out["advice"] = "HOLD"
            out["advice_reason"] = f"peak ₽{peak:.0f}, trail ₽{trail_stop:.0f}"

        return out
    except Exception as e:
        print(f"[telegram] _compute_position_pnl failed: {e}")
        return out


def _read_positions_from_files() -> dict:
    """Прямое чтение positions без backend — для GitHub Actions.

    Каскад:
    1. SQLite (argentum.db) — позиции пользователя локально
    2. positions.json — legacy JSON
    3. Tinkoff sandbox API — реальные позиции на бирже (для cron в облаке)

    Для всех источников P&L и current_price рассчитываются локально из
    silver_daily.parquet + USDRUB.parquet (без обращения к backend).
    """
    import sqlite3
    db = REPO_ROOT_LOCAL / "argentum" / "backend" / "data" / "argentum.db"
    js = REPO_ROOT_LOCAL / "argentum" / "backend" / "data" / "positions.json"
    positions = []
    source = "no_data"
    if db.exists():
        try:
            conn = sqlite3.connect(str(db))
            conn.row_factory = sqlite3.Row
            for r in conn.execute("SELECT * FROM positions"):
                positions.append(dict(r))
            conn.close()
            if positions:
                source = "sqlite"
        except Exception:
            pass

    if not positions and js.exists():
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                positions = data
                source = "json"
        except Exception:
            pass

    # Tinkoff fallback — если локальные DB/JSON пусты, но есть TINKOFF_TOKEN
    if not positions and os.getenv("TINKOFF_TOKEN", "").strip():
        tinkoff_positions = _read_positions_from_tinkoff()
        if tinkoff_positions:
            positions = tinkoff_positions
            source = "tinkoff_sandbox"

    # Обогащение P&L локально из parquet (для всех источников где нет)
    positions = [
        _compute_position_pnl(p) if "current_price" not in p or not p.get("current_price")
        else p
        for p in positions
    ]

    reason_map = {
        "sqlite":          "из локальной БД + P&L посчитан локально",
        "json":            "из positions.json + P&L посчитан локально",
        "tinkoff_sandbox": "из Tinkoff sandbox",
        "no_data":         "позиций нет",
    }

    # master_p_up подсчитаем потом в send_portfolio_chart, тут оставим 0
    return {
        "positions":     positions,
        "master_signal": "WAIT",
        "master_p_up":   0.0,
        "master_reason": reason_map[source],
        "n_open":        len(positions),
        "can_buy":       False,
        "_source":       source,
    }


def _api_with_fallback(path: str) -> dict:
    """Сначала backend, если не отвечает — fallback на прямое чтение файлов."""
    try:
        result = _api(path)
        # Если backend вернул valid data (не дефолтные нули) — используем
        if path == "/api/signal" and result.get("close", 0) > 0:
            return result
        if path == "/api/positions" and "positions" in result:
            return result
    except Exception:
        pass
    # Fallback
    if path == "/api/signal":
        return _read_signal_from_files()
    if path == "/api/positions":
        return _read_positions_from_files()
    return {}


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
    sig = _api_with_fallback("/api/signal")
    pos = _api_with_fallback("/api/positions")
    positions = pos.get("positions", [])

    # Применяем strong-filter (≥ 0.85) для синхронности с UI:
    # raw signal=BUY превращается в HOLD, если уверенность ниже 85%.
    raw_signal = sig.get("signal", "HOLD")
    smoothed_p_up = float(pos.get("master_p_up", 0)) or float(sig.get("p_up", 0))
    STRONG_THRESHOLD = 0.85
    if raw_signal == "BUY" and smoothed_p_up < STRONG_THRESHOLD:
        sig_value = "HOLD"
    else:
        sig_value = raw_signal
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

            entry_price = float(p.get("entry_price", 0))
            current_price = float(p.get("current_price", 0))
            # Используем market_pnl_pct (близкое к реальной бирже), fallback на unrealized_pnl_pct
            pnl_raw = p.get("market_pnl_pct")
            if pnl_raw is None or pnl_raw == 0:
                pnl_raw = p.get("unrealized_pnl_pct", 0)
            pnl = float(pnl_raw or 0)
            advice = p.get("advice", "—")
            has_data = current_price > 0

            # Цвета — если данных нет, всё серое
            if not has_data:
                pnl_color = TEXT_MUTED
                advice_label = "—"
                advice_color = TEXT_MUTED
            else:
                pnl_color = EMERALD if pnl >= 0 else ROSE
                advice_label = "ДЕРЖАТЬ" if advice == "HOLD" else (
                    "ПРОДАТЬ" if advice == "SELL" else "—"
                )
                advice_color = EMERALD if advice == "HOLD" else (
                    ROSE if advice == "SELL" else TEXT_MUTED
                )

            # Используем DejaVu Sans (а не monospace) — у Mono нет глифа ₽ (U+20BD)
            ax_t.text(col_x[0], y, f"#{i+1}", ha="left", va="center",
                      fontsize=9, color=TEXT_PRIM, family="monospace",
                      transform=ax_t.transAxes)
            ax_t.text(col_x[1], y, str(p.get("opened_at", ""))[:10],
                      ha="left", va="center", fontsize=8,
                      color=TEXT_MUTED, family="monospace",
                      transform=ax_t.transAxes)
            entry_str = f"₽{entry_price:,.0f}".replace(",", " ") if entry_price > 0 else "—"
            ax_t.text(col_x[2], y, entry_str,
                      ha="left", va="center", fontsize=9,
                      color=TEXT_PRIM, fontname="DejaVu Sans",
                      transform=ax_t.transAxes)
            current_str = f"₽{current_price:,.0f}".replace(",", " ") if has_data else "—"
            ax_t.text(col_x[3], y, current_str,
                      ha="left", va="center", fontsize=9,
                      color=TEXT_PRIM if has_data else TEXT_MUTED,
                      fontname="DejaVu Sans",
                      transform=ax_t.transAxes)
            pnl_str = f"{'+' if pnl > 0 else ''}{pnl*100:.2f}%" if has_data else "—"
            ax_t.text(col_x[4], y, pnl_str,
                      ha="left", va="center", fontsize=9,
                      color=pnl_color, family="monospace", fontweight="bold",
                      transform=ax_t.transAxes)
            ax_t.text(col_x[5], y, advice_label,
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

    sig = _api_with_fallback("/api/signal")
    pos = _api_with_fallback("/api/positions")
    positions = pos.get("positions", [])
    n_open = len(positions)

    # === MASTER ASSISTANT verdict ===
    # master_p_up: pos endpoint первичный, fallback на sig.p_up (если pos в fallback-режиме = 0)
    master_p_up = float(pos.get("master_p_up", 0)) or float(sig.get("p_up", 0))

    # master_signal: pos endpoint первичный, fallback на маппинг из sig
    raw_master = pos.get("master_signal", "")
    if not raw_master or pos.get("master_reason", "").startswith("computed locally"):
        # Fallback из sig: BUY если passed strong filter (>=0.85), иначе WAIT
        sig_value = sig.get("signal", "HOLD")
        master_signal = "BUY" if (sig_value == "BUY" and master_p_up >= 0.85) else "WAIT"
    else:
        master_signal = raw_master

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
        # P&L доступен только из backend (/api/positions с расчётами). В fallback его нет
        pnl_values = [float(p.get("unrealized_pnl_pct", 0)) for p in positions
                       if "unrealized_pnl_pct" in p]
        if pnl_values:
            avg_pnl = sum(pnl_values) / len(pnl_values) * 100
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
        else:
            # Fallback-режим: позиции есть в DB, но P&L не посчитан (нет backend)
            caption += (
                f"\n💼 <b>Портфолио: {n_open}</b> "
                f"{'позиция' if n_open == 1 else 'позиций'}\n"
                f"<i>(P&L доступен в Argentum UI — backend не подключён)</i>\n"
            )

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
