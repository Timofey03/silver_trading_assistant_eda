"""GET /api/tinkoff/* — sandbox через raw REST API (без SDK)."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cache import ttl_cache

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ACCOUNT_FILE = REPO_ROOT / "baseline_outputs_v23" / "v23_sandbox_account.json"

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

API_BASE = "https://invest-public-api.tinkoff.ru/rest"
SANDBOX = "tinkoff.public.invest.api.contract.v1.SandboxService"
INSTRUMENTS = "tinkoff.public.invest.api.contract.v1.InstrumentsService"
MARKETDATA = "tinkoff.public.invest.api.contract.v1.MarketDataService"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TinkoffBalance(BaseModel):
    connected: bool
    total_rub: float = 0.0
    expected_yield_rub: float = 0.0
    free_cash_rub: float = 0.0
    open_positions: int = 0
    positions: list[dict] = []
    error: str = ""


class OrderRequest(BaseModel):
    direction: str = "BUY"        # BUY / SELL
    lots: int = 1
    ticker: str = "SLVRUBF"       # SLVRUBF = continuous silver futures RUB
                                  # (FIGI FSLVRUB00000, торгуется в Tinkoff sandbox)


class OrderResponse(BaseModel):
    success: bool
    order_id: str = ""
    executed_lots: int = 0
    executed_price: float = 0.0
    direction: str = ""
    figi: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# REST helper
# ---------------------------------------------------------------------------

def _call(service: str, method: str, body: dict, timeout: int = 5) -> dict:
    """POST к Tinkoff Invest REST API v2."""
    token = os.getenv("TINKOFF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TINKOFF_TOKEN не задан в .env")

    url = f"{API_BASE}/{service}/{method}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    r = requests.post(url, data=json.dumps(body), headers=headers, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}
    if r.status_code >= 400:
        raise RuntimeError(f"Tinkoff API {r.status_code}: {data}")
    return data


def _money(m: dict) -> float:
    """Quotation/MoneyValue → float."""
    if not m:
        return 0.0
    units = int(m.get("units", 0) or 0)
    nano = int(m.get("nano", 0) or 0)
    return units + nano / 1e9


def _load_account_id() -> str:
    """Берём сохранённый sandbox-аккаунт; иначе первый из API."""
    if ACCOUNT_FILE.exists():
        try:
            return json.loads(ACCOUNT_FILE.read_text(encoding="utf-8"))["account_id"]
        except Exception:
            pass
    accs = _call(SANDBOX, "GetSandboxAccounts", {}).get("accounts", [])
    if not accs:
        raise RuntimeError("Sandbox-аккаунтов нет — запусти `python silver_paper_tinkoff.py --setup`")
    return accs[0]["id"]


def _find_figi(ticker: str) -> str:
    """Найти FIGI по тикеру (например SLV)."""
    res = _call(INSTRUMENTS, "FindInstrument", {
        "query":              ticker,
        "instrumentKind":     "INSTRUMENT_TYPE_UNSPECIFIED",
        "apiTradeAvailableFlag": False,
    })
    for inst in res.get("instruments", []):
        if inst.get("ticker", "").upper() == ticker.upper():
            return inst["figi"]
    # fallback — первый из списка
    items = res.get("instruments", [])
    if items:
        return items[0]["figi"]
    raise RuntimeError(f"Инструмент {ticker} не найден")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

router = APIRouter()


@ttl_cache(ttl_seconds=120)
def _cached_balance() -> TinkoffBalance:
    """Internal — cached на 2 минуты."""
    return _compute_balance()


@router.get("/tinkoff/balance", response_model=TinkoffBalance)
def get_balance():
    """Live баланс из Tinkoff sandbox (cached 2 мин)."""
    return _cached_balance()


def _compute_balance():
    """Live баланс — внутренняя функция, без кеша."""
    try:
        acc_id = _load_account_id()
        portfolio = _call(SANDBOX, "GetSandboxPortfolio", {"accountId": acc_id})

        total_rub = _money(portfolio.get("totalAmountPortfolio"))
        yield_rub = _money(portfolio.get("expectedYield"))

        # Свободные деньги (только rub)
        free_cash = 0.0
        for p in portfolio.get("positions", []):
            if p.get("instrumentType") == "currency":
                qty = _money(p.get("quantity"))
                price = _money(p.get("currentPrice"))
                free_cash += qty * price

        # Только не-валютные позиции (акции/ETF/фьючерсы)
        non_currency = [
            {
                "figi": p.get("figi"),
                "instrument_type": p.get("instrumentType"),
                "qty": _money(p.get("quantity")),
                "avg_price": _money(p.get("averagePositionPrice")),
                "current_price": _money(p.get("currentPrice")),
                "yield_rub": _money(p.get("expectedYield")),
            }
            for p in portfolio.get("positions", [])
            if p.get("instrumentType") != "currency"
        ]

        return TinkoffBalance(
            connected=True,
            total_rub=total_rub,
            expected_yield_rub=yield_rub,
            free_cash_rub=free_cash,
            open_positions=len(non_currency),
            positions=non_currency,
        )
    except Exception as e:
        return TinkoffBalance(
            connected=False,
            error=f"{type(e).__name__}: {e}",
        )


@router.post("/tinkoff/order", response_model=OrderResponse)
def post_order(req: OrderRequest):
    """Создать рыночный ордер на sandbox-аккаунте."""
    try:
        if req.direction.upper() not in ("BUY", "SELL"):
            raise HTTPException(400, "direction должно быть BUY или SELL")
        if req.lots < 1:
            raise HTTPException(400, "lots >= 1")

        acc_id = _load_account_id()
        figi = _find_figi(req.ticker)
        direction = f"ORDER_DIRECTION_{req.direction.upper()}"
        order_id = str(uuid.uuid4())

        resp = _call(SANDBOX, "PostSandboxOrder", {
            "figi":      figi,
            "quantity":  str(req.lots),
            "direction": direction,
            "accountId": acc_id,
            "orderType": "ORDER_TYPE_MARKET",
            "orderId":   order_id,
        })

        executed_lots = int(resp.get("lotsExecuted", "0") or 0)
        executed_price = _money(resp.get("executedOrderPrice"))

        return OrderResponse(
            success=True,
            order_id=order_id,
            executed_lots=executed_lots,
            executed_price=executed_price,
            direction=req.direction.upper(),
            figi=figi,
        )
    except HTTPException:
        raise
    except Exception as e:
        return OrderResponse(
            success=False,
            error=f"{type(e).__name__}: {e}",
        )
