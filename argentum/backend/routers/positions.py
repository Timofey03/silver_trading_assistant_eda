"""
GET/POST/DELETE /api/positions — мульти-position management.

Архитектура:
- Каждая позиция в Tinkoff = отдельный record в positions.json
- Per-position advisor вычисляет HOLD/SELL для каждой:
    1. Trail-stop:  current < peak × (1 - trail_pct)        → SELL
    2. Max-hold:    days_held >= max_hold_days              → SELL
    3. Model-flip:  smoothed p_up < exit_threshold          → SELL
    4. Иначе                                                 → HOLD
- Master assistant решает можно ли открывать НОВУЮ позицию:
    ✓ Strong signal:    smoothed p_up ≥ entry_strong
    ✓ Max позиций:      < max_positions
    ✓ Cooldown:         с последней >= cooldown_days
    ✓ Free cash:        >= 1 lot × buffer

Хранилище: data/positions.json (массив открытых позиций).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cache import ttl_cache
import db as positions_db

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POSITIONS_FILE = DATA_DIR / "positions.json"   # legacy, для миграции

SILVER_PARQUET = REPO_ROOT / "data" / "multi_asset" / "metals" / "silver_daily.parquet"
PREDICTIONS = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "predictions.parquet"

# Default config (из оптимального grid search)
DEFAULTS = {
    "trail_pct":       0.20,
    "max_hold_days":   60,
    "exit_threshold":  0.30,
    "entry_strong":    0.85,    # smoothed p_up для master BUY signal
    "max_positions":   3,
    "buy_cooldown_d":  5,
    "lot_size_g":      100,
    "free_cash_buffer": 1.2,
}


# ─── Models ─────────────────────────────────────────────────────────────────

class Position(BaseModel):
    id: str
    ticker: str
    figi: str
    opened_at: str          # ISO datetime
    entry_price: float
    lots: int
    lot_size_g: int       # 1 lot SLVRUBF = 100 граммов серебра
    peak_price: float
    source: str = "user"    # user | model_auto
    # Computed fields (set in GET):
    current_price: float = 0.0
    days_held: int = 0
    unrealized_pnl_pct: float = 0.0           # sandbox P&L (entry vs current)
    market_pnl_pct: float = 0.0               # theoretical market P&L (без sandbox slippage)
    market_entry_price: float = 0.0           # теоретическая market цена в момент входа
    market_current_price: float = 0.0         # теоретическая market цена сейчас
    advice: str = "HOLD"
    advice_reason: str = ""


class PositionsResponse(BaseModel):
    positions: list[Position]
    master_signal: str           # BUY / WAIT / AVOID
    master_reason: str
    master_p_up: float = 0.0
    n_open: int = 0
    can_buy: bool = False


class OpenPositionRequest(BaseModel):
    ticker: str = "SLVRUBF"
    lots: int = 1


class OpenPositionResponse(BaseModel):
    success: bool
    position: Optional[Position] = None
    tinkoff_order_id: str = ""
    executed_price: float = 0.0
    error: str = ""


class ClosePositionResponse(BaseModel):
    success: bool
    closed_at: str = ""
    exit_price: float = 0.0
    realized_pnl_pct: float = 0.0
    tinkoff_order_id: str = ""
    error: str = ""


router = APIRouter()


# ─── Storage ────────────────────────────────────────────────────────────────

def _load_positions() -> list[dict]:
    """Загрузить все открытые позиции из SQLite (с auto-migrate из JSON)."""
    # One-time migration from JSON if SQLite пустой
    if POSITIONS_FILE.exists() and not positions_db.list_positions():
        positions_db.migrate_from_json(POSITIONS_FILE)
    return positions_db.list_positions()


def _save_positions(positions: list[dict]) -> None:
    """Legacy no-op — SQLite пишется по факту операций (insert/update/delete)."""
    # Сохраняем peak updates через update_peak (см. list_positions)
    for p in positions:
        try:
            positions_db.update_peak(p["id"], float(p.get("peak_price", 0)))
        except Exception:
            pass


# ─── Market state ───────────────────────────────────────────────────────────

@ttl_cache(ttl_seconds=60)
@ttl_cache(ttl_seconds=300)
def _current_silver_price_rub(figi: str = "FSLVRUB00000") -> float:
    """
    Цена FSLVRUB фьючерса в рублях (live).

    Fallback chain:
    1. Tinkoff GetLastPrices (live market quote)
    2. silver_close (USD/oz) × USDRUB × 100г / 31.1г (теоретическая)
       — используется после-часовом / выходные когда Tinkoff API возвращает 0

    Это правильная цена для сравнения с entry_price (executed_price тоже в RUB).
    """
    import os, requests
    try:
        from dotenv import load_dotenv; load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass
    token = os.getenv("TINKOFF_TOKEN", "").strip()

    # === Source 1: Tinkoff GetLastPrices ===
    if token:
        try:
            url = ("https://invest-public-api.tinkoff.ru/rest/"
                   "tinkoff.public.invest.api.contract.v1.MarketDataService/GetLastPrices")
            r = requests.post(url, json={"figi": [figi]},
                              headers={"Authorization": f"Bearer {token}",
                                       "Content-Type": "application/json"},
                              timeout=3)  # быстрый fail → fallback на theoretical
            data = r.json()
            prices = data.get("lastPrices", [])
            if prices:
                p = prices[0].get("price", {})
                units = int(p.get("units", 0) or 0)
                nano = int(p.get("nano", 0) or 0)
                quote_per_oz = units + nano / 1e9
                price = float(quote_per_oz * 100)
                if price > 0:
                    return price
        except Exception:
            pass

    # === Source 2: Fallback через USD silver × USDRUB ===
    # 1 контракт SLVRUBF = 100 г, USD silver — за oz, 1 oz = 31.1035 г
    try:
        if SILVER_PARQUET.exists():
            silver_df = pd.read_parquet(SILVER_PARQUET)
            usd_per_oz = float(silver_df["close"].iloc[-1])
            # USDRUB из macro кеша или yfinance
            usdrub = 0.0
            usdrub_file = REPO_ROOT / "data" / "multi_asset" / "macro" / "USDRUB_daily.parquet"
            if usdrub_file.exists():
                try:
                    df = pd.read_parquet(usdrub_file)
                    if len(df.columns):
                        usdrub = float(df[df.columns[0]].dropna().iloc[-1])
                except Exception:
                    pass
            if usdrub <= 0:
                try:
                    import yfinance as yf
                    h = yf.Ticker("RUB=X").history(period="5d")
                    if len(h):
                        usdrub = float(h["Close"].iloc[-1])
                except Exception:
                    pass
            if usd_per_oz > 0 and usdrub > 0:
                # 100 граммов = 100 / 31.1035 oz
                return usd_per_oz * (100 / 31.1035) * usdrub
    except Exception:
        pass

    return 0.0


def _current_silver_price() -> float:
    """Backward-compat alias — returns RUB price (per contract)."""
    return _current_silver_price_rub()


@ttl_cache(ttl_seconds=30)
def _last_n_signal_files(n: int = 3) -> list[dict]:
    """Возвращает последние N production signal.json (свежий первый)."""
    trading_dir = REPO_ROOT / "daily_reports" / "e3b" / "trading"
    if not trading_dir.exists():
        return []
    dirs = sorted([d for d in trading_dir.iterdir() if d.is_dir()], reverse=True)
    out = []
    for d in dirs:
        f = d / "signal.json"
        if f.exists():
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
                if len(out) >= n:
                    break
            except Exception:
                continue
    return out


def _current_p_up_smoothed() -> float:
    """
    Smoothed p_up — единый источник истины для UI.
    Сглаживает по последним 3 production signal.json (а не по walk-forward parquet),
    так что значение на /positions СОВПАДАЕТ со значением на /.
    """
    sigs = _last_n_signal_files(n=3)
    if not sigs:
        return 0.0
    p_ups = [float(s.get("p_up", 0)) for s in sigs if s.get("p_up") is not None]
    return sum(p_ups) / len(p_ups) if p_ups else 0.0


def _current_p_up_raw() -> float:
    """Latest signal.json p_up — raw, без сглаживания."""
    sigs = _last_n_signal_files(n=1)
    return float(sigs[0].get("p_up", 0)) if sigs else 0.0


# ─── Per-position advisor ───────────────────────────────────────────────────

def _advise_position(pos: dict, current_price: float, p_smooth: float) -> tuple[str, str]:
    """
    Возвращает (advice, reason) для одной открытой позиции.

    Логика:
    1. Trail-stop: current < peak × (1 - trail_pct) → SELL
    2. Max-hold: days_held >= max_hold_days        → SELL
    3. Model-flip: p_smoothed < exit_threshold     → SELL
    4. Иначе                                       → HOLD
    """
    trail_pct = DEFAULTS["trail_pct"]
    max_hold = DEFAULTS["max_hold_days"]
    exit_th = DEFAULTS["exit_threshold"]

    try:
        opened = datetime.fromisoformat(pos["opened_at"].replace("Z", ""))
        days_held = (date.today() - opened.date()).days
    except Exception:
        days_held = 0

    peak = max(float(pos.get("peak_price", current_price)), current_price)
    trail_level = peak * (1 - trail_pct)
    pnl = (current_price - pos["entry_price"]) / pos["entry_price"]

    # 1. Trail
    if current_price > 0 and current_price < trail_level:
        return "SELL", (
            f"trail-stop сработал · peak ₽{peak:,.0f} × 0.80 = ₽{trail_level:,.0f}, "
            f"сейчас ₽{current_price:,.0f}"
        )

    # 2. Max hold
    if days_held >= max_hold:
        return "SELL", f"max hold ({max_hold} дн) превышен · удерживается {days_held} дн"

    # 3. Model flip
    if p_smooth > 0 and p_smooth < exit_th:
        return "SELL", f"модель развернулась bearish · smoothed p_up = {p_smooth:.2f} < {exit_th}"

    # 4. Hold
    return "HOLD", (
        f"{days_held}/{max_hold} дн · P&L {pnl*100:+.2f}% · "
        f"peak ₽{peak:,.0f}, trail ₽{trail_level:,.0f} · p_up {p_smooth:.2f}"
    )


# ─── Master assistant ───────────────────────────────────────────────────────

def _master_advise(positions: list[dict], p_smooth: float) -> tuple[str, str, bool]:
    """
    Возвращает (signal, reason, can_buy).
    """
    n_open = len(positions)
    max_pos = DEFAULTS["max_positions"]
    cooldown_d = DEFAULTS["buy_cooldown_d"]
    entry_strong = DEFAULTS["entry_strong"]

    # 1. Strong signal check
    if p_smooth < 0.30:
        return "AVOID", f"p_up слишком низкий ({p_smooth:.2f}) — рынок против", False

    if p_smooth < entry_strong:
        return "WAIT", (
            f"сигнал в зоне шума (p_up = {p_smooth:.2f}, нужно ≥ {entry_strong}) — "
            f"ждём подтверждения"
        ), False

    # 2. Max positions
    if n_open >= max_pos:
        return "WAIT", f"уже {n_open} открытых позиций (макс {max_pos}) — место занято", False

    # 3. Cooldown с последней покупки
    if positions:
        last_open = max(p["opened_at"] for p in positions)
        try:
            last_d = datetime.fromisoformat(last_open.replace("Z", "")).date()
            days_since = (date.today() - last_d).days
            if days_since < cooldown_d:
                return "WAIT", (
                    f"cooldown {days_since}/{cooldown_d} дней с последней покупки "
                    f"({last_d})"
                ), False
        except Exception:
            pass

    # All checks passed
    return "BUY", (
        f"strong signal (p_up = {p_smooth:.2f} ≥ {entry_strong}) · "
        f"{n_open}/{max_pos} позиций · open для нового лота"
    ), True


# ─── Routes ─────────────────────────────────────────────────────────────────

def _theoretical_rub_price(at_date: Optional[date] = None) -> float:
    """
    Теоретическая цена 1 контракта SLVRUBF (100г) на указанную дату.
    = USD silver close × 100/31.1 oz × USDRUB

    Если at_date None — текущая (latest data).
    """
    try:
        df = pd.read_parquet(SILVER_PARQUET)
        if at_date is not None:
            sd = df.index.asof(pd.Timestamp(at_date))
            if pd.isna(sd):
                return 0.0
            usd_per_oz = float(df.loc[sd, "close"])
        else:
            usd_per_oz = float(df["close"].iloc[-1])
    except Exception:
        return 0.0

    usdrub = 0.0
    usdrub_file = REPO_ROOT / "data" / "multi_asset" / "macro" / "USDRUB_daily.parquet"
    if usdrub_file.exists():
        try:
            df = pd.read_parquet(usdrub_file)
            if at_date:
                sd = df.index.asof(pd.Timestamp(at_date))
                if not pd.isna(sd):
                    usdrub = float(df.loc[sd, df.columns[0]])
            if usdrub <= 0 and len(df.columns):
                usdrub = float(df[df.columns[0]].dropna().iloc[-1])
        except Exception:
            pass
    if usdrub <= 0:
        try:
            import yfinance as yf
            hist = yf.Ticker("RUB=X").history(period="30d")
            hist.index = hist.index.tz_localize(None).normalize()
            if at_date:
                sd = hist.index.asof(pd.Timestamp(at_date))
                if not pd.isna(sd):
                    usdrub = float(hist.loc[sd, "Close"])
            if usdrub <= 0:
                usdrub = float(hist["Close"].iloc[-1])
        except Exception:
            pass

    if usd_per_oz > 0 and usdrub > 0:
        return usd_per_oz * (100 / 31.1035) * usdrub
    return 0.0


@ttl_cache(ttl_seconds=120)
def _list_positions_cached() -> dict:
    """
    Полный вычисленный response cached на 2 минуты.
    Цены обновляются каждые 5 минут (TTL _current_silver_price_rub),
    response cache 2 мин чтобы UI auto-refresh не дёргал HEavy work каждые 30с.
    """
    raw = _load_positions()
    current_price = _current_silver_price()
    p_smooth = _current_p_up_smoothed()

    positions_out: list[dict] = []
    for p in raw:
        new_peak = max(float(p.get("peak_price", current_price)), current_price)
        if new_peak > float(p.get("peak_price", 0)):
            p["peak_price"] = new_peak

        try:
            opened = datetime.fromisoformat(p["opened_at"].replace("Z", "").split("_")[0])
            days_held = (date.today() - opened.date()).days
        except Exception:
            days_held = 0

        pnl = ((current_price - p["entry_price"]) / p["entry_price"]) if p.get("entry_price") else 0.0

        # Theoretical market prices (без sandbox slippage)
        try:
            entry_date = datetime.fromisoformat(p["opened_at"].replace("Z", "").split("_")[0]).date()
        except Exception:
            entry_date = date.today()
        market_entry = _theoretical_rub_price(entry_date)
        market_current = _theoretical_rub_price()  # today
        market_pnl = ((market_current - market_entry) / market_entry) if market_entry > 0 else 0.0

        advice, reason = _advise_position(p, current_price, p_smooth)

        positions_out.append({
            "id": p["id"], "ticker": p["ticker"], "figi": p["figi"],
            "opened_at": p["opened_at"], "entry_price": p["entry_price"],
            "lots": p["lots"], "lot_size_g": p.get("lot_size_g", 100),
            "peak_price": new_peak, "source": p.get("source", "user"),
            "current_price": current_price, "days_held": days_held,
            "unrealized_pnl_pct": pnl,
            "market_pnl_pct": market_pnl,
            "market_entry_price": market_entry,
            "market_current_price": market_current,
            "advice": advice, "advice_reason": reason,
        })

    # Persist peak updates
    _save_positions(raw)
    master, mreason, can_buy = _master_advise(raw, p_smooth)
    return {
        "positions_data": positions_out,
        "master_signal": master, "master_reason": mreason,
        "master_p_up": p_smooth, "n_open": len(positions_out),
        "can_buy": can_buy,
    }


@router.get("/positions", response_model=PositionsResponse)
def list_positions():
    """Все открытые позиции + master assistant + per-position advice."""
    cached = _list_positions_cached()
    positions_out = [Position(**p) for p in cached["positions_data"]]
    return PositionsResponse(
        positions=positions_out,
        master_signal=cached["master_signal"],
        master_reason=cached["master_reason"],
        master_p_up=cached["master_p_up"],
        n_open=cached["n_open"],
        can_buy=cached["can_buy"],
    )


def _list_positions_OLD_BACKUP():
    """Старая версия без кеша — сохранена на всякий случай."""
    raw = _load_positions()
    current_price = _current_silver_price()
    p_smooth = _current_p_up_smoothed()

    positions_out: list[Position] = []
    for p in raw:
        # Update peak in storage if current is higher
        new_peak = max(float(p.get("peak_price", current_price)), current_price)
        if new_peak > float(p.get("peak_price", 0)):
            p["peak_price"] = new_peak

        try:
            opened = datetime.fromisoformat(p["opened_at"].replace("Z", ""))
            days_held = (date.today() - opened.date()).days
        except Exception:
            days_held = 0

        pnl = ((current_price - p["entry_price"]) / p["entry_price"]) if p.get("entry_price") else 0.0
        advice, reason = _advise_position(p, current_price, p_smooth)

        positions_out.append(Position(
            id=p["id"], ticker=p["ticker"], figi=p["figi"],
            opened_at=p["opened_at"], entry_price=p["entry_price"],
            lots=p["lots"], lot_size_g=p.get("lot_size_g", 100),
            peak_price=new_peak,
            source=p.get("source", "user"),
            current_price=current_price,
            days_held=days_held,
            unrealized_pnl_pct=pnl,
            advice=advice,
            advice_reason=reason,
        ))

    # Persist updated peaks
    _save_positions(raw)

    master, mreason, can_buy = _master_advise(raw, p_smooth)
    return PositionsResponse(
        positions=positions_out,
        master_signal=master,
        master_reason=mreason,
        master_p_up=p_smooth,
        n_open=len(positions_out),
        can_buy=can_buy,
    )


@router.post("/positions", response_model=OpenPositionResponse)
def open_position(req: OpenPositionRequest):
    """Открыть новую позицию: Tinkoff order + сохранить в store."""
    # Используем существующий tinkoff.post_order через прямой импорт
    from routers.tinkoff import OrderRequest, post_order

    # Pre-check: master assistant
    existing = _load_positions()
    p_smooth = _current_p_up_smoothed()
    _, mreason, can_buy = _master_advise(existing, p_smooth)
    if not can_buy:
        return OpenPositionResponse(
            success=False,
            error=f"Master блокирует: {mreason}",
        )

    # Создаём ордер в Tinkoff
    order_req = OrderRequest(direction="BUY", lots=req.lots, ticker=req.ticker)
    order_res = post_order(order_req)
    if not order_res.success:
        return OpenPositionResponse(
            success=False,
            error=f"Tinkoff order failed: {order_res.error}",
        )

    # Сохраняем позицию в SQLite (atomic)
    pos = {
        "id":           str(uuid.uuid4()),
        "ticker":       req.ticker,
        "figi":         order_res.figi or "FSLVRUB00000",
        "opened_at":    datetime.now().isoformat(),
        "entry_price":  float(order_res.executed_price),
        "lots":         req.lots,
        "lot_size_g":   DEFAULTS["lot_size_g"],
        "peak_price":   float(order_res.executed_price),
        "source":       "user",
    }
    positions_db.insert_position(pos)

    return OpenPositionResponse(
        success=True,
        position=Position(
            **pos,
            current_price=float(order_res.executed_price),
            days_held=0,
            unrealized_pnl_pct=0.0,
            advice="HOLD",
            advice_reason="новая позиция",
        ),
        tinkoff_order_id=order_res.order_id,
        executed_price=float(order_res.executed_price),
    )


@router.delete("/positions/{position_id}", response_model=ClosePositionResponse)
def close_position(position_id: str):
    """Закрыть позицию: Tinkoff SELL order + удалить из store."""
    from routers.tinkoff import OrderRequest, post_order

    existing = _load_positions()
    target = next((p for p in existing if p["id"] == position_id), None)
    if target is None:
        raise HTTPException(404, f"Position {position_id} not found")

    order_req = OrderRequest(direction="SELL", lots=target["lots"], ticker=target["ticker"])
    order_res = post_order(order_req)
    if not order_res.success:
        return ClosePositionResponse(
            success=False,
            error=f"Tinkoff SELL failed: {order_res.error}",
        )

    exit_price = float(order_res.executed_price)
    pnl = (exit_price - target["entry_price"]) / target["entry_price"]

    # Записываем в closed_trades (история)
    try:
        opened_dt = datetime.fromisoformat(target["opened_at"].replace("Z", ""))
        days_held = (date.today() - opened_dt.date()).days
    except Exception:
        days_held = 0

    positions_db.insert_closed_trade({
        "id":               str(uuid.uuid4()),
        "original_id":      target["id"],
        "ticker":           target["ticker"],
        "figi":             target["figi"],
        "opened_at":        target["opened_at"],
        "closed_at":        datetime.now().isoformat(),
        "entry_price":      target["entry_price"],
        "exit_price":       exit_price,
        "lots":             target["lots"],
        "lot_size_g":       target.get("lot_size_g", 100),
        "realized_pnl_pct": pnl,
        "exit_reason":      "user_close",
        "days_held":        days_held,
    })

    # Удаляем из открытых
    positions_db.delete_position(position_id)

    return ClosePositionResponse(
        success=True,
        closed_at=datetime.now().isoformat(),
        exit_price=exit_price,
        realized_pnl_pct=pnl,
        tinkoff_order_id=order_res.order_id,
    )


@router.get("/positions/closed")
def list_closed(limit: int = 50):
    """История закрытых позиций."""
    return {"trades": positions_db.list_closed_trades(limit=limit)}


@router.post("/positions/sync")
def sync_from_tinkoff():
    """
    Импорт существующих позиций из Tinkoff portfolio в наш tracker.
    Создаёт записи с opened_at=now() (точная дата неизвестна) и
    entry_price = avg_price из Tinkoff.
    """
    from routers.tinkoff import get_balance
    balance = get_balance()
    if not balance.connected:
        return {"success": False, "error": balance.error, "imported": 0}

    existing_figis = {p["figi"] for p in positions_db.list_positions()}
    imported = []
    for p in balance.positions:
        if p["instrument_type"] not in ("futures", "share", "etf"):
            continue
        if int(p["qty"]) <= 0:
            continue
        figi = p["figi"]
        # Skip если такой FIGI уже отслеживается
        # (не идеально для multi-lot но защищает от дубликатов на повторный sync)
        if figi in existing_figis:
            continue
        # avg_price из Tinkoff = USD/oz или RUB/lot, normalize via lastPrice
        avg = float(p.get("avg_price", 0)) * 100  # 100g per contract
        new_pos = {
            "id":           str(uuid.uuid4()),
            "ticker":       "SLVRUBF" if figi == "FSLVRUB00000" else figi,
            "figi":         figi,
            "opened_at":    datetime.now().isoformat() + "_synced",
            "entry_price":  avg if avg > 0 else _current_silver_price_rub(figi),
            "lots":         int(p["qty"]),
            "lot_size_g":   DEFAULTS["lot_size_g"],
            "peak_price":   avg if avg > 0 else _current_silver_price_rub(figi),
            "source":       "tinkoff_sync",
        }
        positions_db.insert_position(new_pos)
        imported.append(new_pos)

    return {"success": True, "imported": len(imported), "positions": imported}
