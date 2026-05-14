"""
silver_paper_tinkoff.py — Paper trading через Tinkoff Invest REST API v2 (sandbox)

Что делает:
1. Использует sandbox-режим Tinkoff Invest (виртуальные деньги, реальные котировки)
2. Читает токен ТОЛЬКО из env var TINKOFF_TOKEN (или .env файла)
3. Читает сигналы из baseline_outputs_v23/v23_decision_audit_log.jsonl
4. Конвертирует BUY/SHORT → market orders
5. Логирует P&L и состояние портфеля в baseline_outputs_v23/v23_paper_trading_log.csv

ТРЕБОВАНИЯ К ТОКЕНУ:
- Только sandbox-права! Никаких real-money перестановок!
- Не хранить в коде. Использовать .env (см. .env.example)

REST API docs: https://developer.tbank.ru/invest/api/

Использование:
  # Шаг 1: один раз настроить sandbox-счёт
  python silver_paper_tinkoff.py --setup

  # Шаг 2: найти инструмент (silver ETF/futures)
  python silver_paper_tinkoff.py --find SLV

  # Шаг 3: проиграть исторические сигналы из v23 audit log
  python silver_paper_tinkoff.py --replay --ticker SLV --since 2025-01-01

  # Шаг 4: статус портфеля
  python silver_paper_tinkoff.py --status

  # Шаг 5: live mode (раз в день проверять новый сигнал)
  python silver_paper_tinkoff.py --live --ticker SLV
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

V23_DIR = Path("baseline_outputs_v23")
V23_DIR.mkdir(exist_ok=True)

PAPER_LOG = V23_DIR / "v23_paper_trading_log.csv"
PORTFOLIO_SNAPSHOTS = V23_DIR / "v23_paper_portfolio_snapshots.jsonl"
ACCOUNT_FILE = V23_DIR / "v23_sandbox_account.json"

# Tinkoff REST API endpoints
API_BASE = "https://invest-public-api.tinkoff.ru/rest"
SANDBOX_SERVICE = "tinkoff.public.invest.api.contract.v1.SandboxService"
INSTRUMENTS_SERVICE = "tinkoff.public.invest.api.contract.v1.InstrumentsService"
MARKETDATA_SERVICE = "tinkoff.public.invest.api.contract.v1.MarketDataService"


# ===========================================================================
# 1. CLIENT
# ===========================================================================

@dataclass
class TinkoffClient:
    token: str
    session: requests.Session = None

    def __post_init__(self):
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}",
                "Content-Type":  "application/json",
                "Accept":        "application/json",
            })

    def call(self, service: str, method: str, body: dict, timeout: int = 30) -> dict:
        url = f"{API_BASE}/{service}/{method}"
        r = self.session.post(url, data=json.dumps(body), timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        if r.status_code >= 400:
            raise RuntimeError(
                f"Tinkoff API error {r.status_code} on {service}/{method}: {data}"
            )
        return data

    # --- sandbox account ---
    def sandbox_open_account(self) -> str:
        return self.call(SANDBOX_SERVICE, "OpenSandboxAccount", {})["accountId"]

    def sandbox_pay_in(self, account_id: str, amount_rub: float) -> dict:
        return self.call(SANDBOX_SERVICE, "SandboxPayIn", {
            "accountId": account_id,
            "amount": {"currency": "rub", "units": str(int(amount_rub)), "nano": 0},
        })

    def sandbox_close_account(self, account_id: str) -> dict:
        return self.call(SANDBOX_SERVICE, "CloseSandboxAccount", {"accountId": account_id})

    def sandbox_accounts(self) -> list:
        return self.call(SANDBOX_SERVICE, "GetSandboxAccounts", {}).get("accounts", [])

    def sandbox_portfolio(self, account_id: str) -> dict:
        return self.call(SANDBOX_SERVICE, "GetSandboxPortfolio", {"accountId": account_id})

    def sandbox_post_order(
        self,
        account_id: str,
        figi: str,
        quantity: int,
        direction: str,                  # ORDER_DIRECTION_BUY / ORDER_DIRECTION_SELL
        order_type: str = "ORDER_TYPE_MARKET",
        order_id: Optional[str] = None,
    ) -> dict:
        if order_id is None:
            order_id = str(uuid.uuid4())
        return self.call(SANDBOX_SERVICE, "PostSandboxOrder", {
            "figi":      figi,
            "quantity":  str(int(quantity)),
            "direction": direction,
            "accountId": account_id,
            "orderType": order_type,
            "orderId":   order_id,
        })

    def sandbox_operations(self, account_id: str, frm: str, to: str) -> dict:
        return self.call(SANDBOX_SERVICE, "GetSandboxOperations", {
            "accountId": account_id,
            "from": frm,
            "to":   to,
        })

    # --- instruments / market data ---
    def find_instrument(self, query: str) -> list:
        res = self.call(INSTRUMENTS_SERVICE, "FindInstrument", {
            "query":             query,
            "instrumentKind":    "INSTRUMENT_TYPE_UNSPECIFIED",
            "apiTradeAvailableFlag": False,
        })
        return res.get("instruments", [])

    def get_candles(
        self,
        figi: str,
        frm: str,
        to: str,
        interval: str = "CANDLE_INTERVAL_DAY",
    ) -> list:
        res = self.call(MARKETDATA_SERVICE, "GetCandles", {
            "figi": figi, "from": frm, "to": to, "interval": interval,
        })
        return res.get("candles", [])

    def get_last_price(self, figi: str) -> Optional[float]:
        res = self.call(MARKETDATA_SERVICE, "GetLastPrices", {"figi": [figi]})
        prices = res.get("lastPrices", [])
        if not prices:
            return None
        p = prices[0].get("price", {})
        units = int(p.get("units", 0))
        nano  = int(p.get("nano", 0))
        return units + nano / 1e9


# ===========================================================================
# 2. HELPERS
# ===========================================================================

def _get_token_or_exit() -> str:
    token = os.getenv("TINKOFF_TOKEN", "").strip()
    if not token:
        print("ERROR: переменная TINKOFF_TOKEN не задана.")
        print("Создайте файл .env (см. .env.example) с токеном sandbox.")
        sys.exit(2)
    if token.startswith("t.") and len(token) < 30:
        print("WARN: токен подозрительно короткий. Проверьте.")
    return token


def _load_account_id(client: TinkoffClient) -> str:
    """Загружает сохранённый sandbox account_id или создаёт новый."""
    if ACCOUNT_FILE.exists():
        data = json.loads(ACCOUNT_FILE.read_text(encoding="utf-8"))
        return data["account_id"]
    accs = client.sandbox_accounts()
    if accs:
        acc = accs[0]["id"]
        ACCOUNT_FILE.write_text(json.dumps({"account_id": acc}, indent=2), encoding="utf-8")
        return acc
    raise RuntimeError("Sandbox-счёт не найден. Запустите --setup")


def _signal_to_direction_and_qty(
    signal: str,
    p_signal: float,
    price: float,
    free_cash_rub: float,
    base_size_rub: float = 5000.0,
    max_size_rub:  float = 20000.0,
    lot:           int   = 1,
    instrument_type: str = "etf",
    futures_max_lots: int = 1,
) -> Tuple[str, int]:
    """
    Конвертирует сигнал v23 в (direction, lots).
    Kelly-inspired sizing для ETF/share.
    Для futures — фиксировано futures_max_lots (1 по умолчанию),
    т.к. quote price << notional × multiplier.
    """
    if price <= 0 or free_cash_rub <= 0:
        return ("", 0)

    p = max(0.0, min(1.0, p_signal if p_signal == p_signal else 0.5))

    if instrument_type == "futures":
        # 1 лот фьючерса; Kelly масштаб через "сколько контрактов"
        # Округляем по p_signal: 0.5 → 1, 0.6 → 1, 0.7+ → 2, ...
        kelly_mult = 1 + int((p - 0.5) * 4)  # 1..3
        lots = min(kelly_mult, futures_max_lots)
    else:
        size_rub = base_size_rub + (max_size_rub - base_size_rub) * max(0.0, p - 0.5) * 2.0
        size_rub = min(size_rub, free_cash_rub * 0.9)
        qty_units = max(int(size_rub / price), 0)
        lots = max(qty_units // lot, 0)

    if lots == 0:
        return ("", 0)

    if signal == "BUY":
        return ("ORDER_DIRECTION_BUY", lots)
    if signal == "SHORT":
        return ("ORDER_DIRECTION_SELL", lots)
    return ("", 0)


# ===========================================================================
# 3. COMMANDS
# ===========================================================================

def cmd_setup(initial_rub: float = 100000.0) -> None:
    """Один раз: создать sandbox-счёт и пополнить."""
    client = TinkoffClient(_get_token_or_exit())

    print("=== Tinkoff Sandbox setup ===")
    existing = client.sandbox_accounts()
    if existing:
        print(f"  Найден существующий sandbox-счёт: {existing[0]['id']}")
        print("  Используем его.")
        account_id = existing[0]["id"]
    else:
        account_id = client.sandbox_open_account()
        print(f"  Создан sandbox-счёт: {account_id}")

    ACCOUNT_FILE.write_text(json.dumps({"account_id": account_id}, indent=2), encoding="utf-8")
    print(f"  Account ID сохранён в {ACCOUNT_FILE}")

    print(f"  Пополнение: {initial_rub:,.0f} RUB")
    client.sandbox_pay_in(account_id, initial_rub)

    portfolio = client.sandbox_portfolio(account_id)
    total = portfolio.get("totalAmountPortfolio", {})
    print(f"  Текущая стоимость портфеля: {total}")


def cmd_find(query: str) -> None:
    """Найти инструмент по тикеру/имени."""
    client = TinkoffClient(_get_token_or_exit())
    print(f"=== Поиск инструмента: '{query}' ===")
    items = client.find_instrument(query)
    if not items:
        print("  Ничего не найдено.")
        return
    for it in items[:20]:
        print(f"  {it.get('ticker', '?'):10s} | "
              f"figi={it.get('figi', '?')} | "
              f"{it.get('instrumentType', '?'):8s} | "
              f"{it.get('name', '')[:50]} | "
              f"currency={it.get('currency', '?')}")


def cmd_status() -> None:
    """Текущий статус портфеля."""
    client = TinkoffClient(_get_token_or_exit())
    account_id = _load_account_id(client)
    p = client.sandbox_portfolio(account_id)

    def _money(m: dict) -> str:
        if not m:
            return "0"
        u = int(m.get("units", 0))
        n = int(m.get("nano", 0))
        return f"{u + n / 1e9:,.2f} {m.get('currency', '').upper()}"

    print("=== Sandbox портфель ===")
    print(f"  Account ID: {account_id}")
    print(f"  Total:      {_money(p.get('totalAmountPortfolio'))}")
    print(f"  Shares:     {_money(p.get('totalAmountShares'))}")
    print(f"  Bonds:      {_money(p.get('totalAmountBonds'))}")
    print(f"  ETF:        {_money(p.get('totalAmountEtf'))}")
    print(f"  Currencies: {_money(p.get('totalAmountCurrencies'))}")
    print(f"  Futures:    {_money(p.get('totalAmountFutures'))}")
    print(f"  Expected yield: {p.get('expectedYield', {})}")
    print(f"  Positions: {len(p.get('positions', []))}")
    for pos in p.get("positions", []):
        print(f"    - {pos.get('instrumentType')} figi={pos.get('figi')} "
              f"qty={pos.get('quantity', {}).get('units', '?')} "
              f"avg={pos.get('averagePositionPrice', {}).get('units', '?')}")

    snap = {
        "ts":         datetime.now(timezone.utc).isoformat(),
        "account_id": account_id,
        "portfolio":  p,
    }
    with PORTFOLIO_SNAPSHOTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap, ensure_ascii=False, default=str) + "\n")


def cmd_replay(
    ticker: str,
    since: Optional[str] = None,
    until: Optional[str] = None,
    initial_rub: float = 100000.0,
    base_size_rub: float = 5000.0,
    max_size_rub: float = 20000.0,
    dry_run: bool = False,
) -> None:
    """
    Проигрывает исторические сигналы из v23 audit log в sandbox.
    ВНИМАНИЕ: sandbox реально выставляет ордера по текущей цене, а не
    по цене сигнала. Это даёт оценку «как если бы сегодня запустили».
    """
    client = TinkoffClient(_get_token_or_exit())
    account_id = _load_account_id(client)

    audit_path = V23_DIR / "v23_decision_audit_log.jsonl"
    if not audit_path.exists():
        print(f"ERROR: нет {audit_path}. Запустите v23 --audit-log сначала.")
        return

    items = client.find_instrument(ticker)
    candidates = [
        it for it in items
        if it.get("ticker", "").upper() == ticker.upper()
        and it.get("apiTradeAvailableFlag", True)
    ]
    if not candidates:
        candidates = items
    if not candidates:
        print(f"ERROR: инструмент {ticker} не найден.")
        return
    inst = candidates[0]
    figi  = inst["figi"]
    lot   = int(inst.get("lot", 1))
    inst_type = inst.get("instrumentType", inst.get("instrument_type", "etf"))
    print(f"  Инструмент: {inst.get('ticker')} ({inst.get('name', '')}) "
          f"figi={figi}, lot={lot}, type={inst_type}, "
          f"currency={inst.get('currency')}")

    rows: List[dict] = []
    with audit_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("ts", "")[:10]
            if since and ts < since:
                continue
            if until and ts > until:
                continue
            rows.append(rec)

    print(f"  Сигналов к проигрыванию: {len(rows)} "
          f"(окно {since or 'all'} → {until or 'now'})")
    if not rows:
        return

    log_rows: List[dict] = []
    for rec in rows:
        signal_long  = rec.get("signal_long",  "HOLD")
        signal_short = rec.get("signal_short", "HOLD")
        signal = "BUY" if signal_long == "BUY" else "SHORT" if signal_short == "SHORT" else ""
        if not signal:
            continue
        p_signal = float(rec.get("p_up", 0.5) if signal == "BUY" else rec.get("p_short", 0.5) or 0.5)

        try:
            price = client.get_last_price(figi) or 0.0
        except Exception as e:
            print(f"  WARN price fetch failed: {e}")
            price = 0.0

        portfolio = client.sandbox_portfolio(account_id)
        free_rub = 0.0
        for pos in portfolio.get("positions", []):
            if pos.get("instrumentType") == "currency" and pos.get("figi", "").startswith("RUB"):
                q = pos.get("quantity", {})
                free_rub += int(q.get("units", 0)) + int(q.get("nano", 0)) / 1e9

        direction, lots = _signal_to_direction_and_qty(
            signal, p_signal, price, free_rub,
            base_size_rub=base_size_rub, max_size_rub=max_size_rub, lot=lot,
            instrument_type=inst_type, futures_max_lots=2,
        )

        log_entry = {
            "ts_signal": rec.get("ts"),
            "signal":    signal,
            "p_signal":  p_signal,
            "ticker":    ticker,
            "figi":      figi,
            "price":     price,
            "free_rub_before": round(free_rub, 2),
            "direction": direction,
            "lots":      lots,
            "executed":  False,
            "order_id":  None,
            "error":     None,
        }

        if not direction or lots <= 0:
            log_entry["error"] = "no-qty (insufficient cash or invalid signal)"
            log_rows.append(log_entry)
            continue

        if dry_run:
            log_entry["error"] = "DRY_RUN"
            log_rows.append(log_entry)
            print(f"  DRY {rec.get('ts')}: {signal} {lots}x{ticker}@{price}")
            continue

        try:
            res = client.sandbox_post_order(account_id, figi, lots, direction)
            log_entry["executed"] = True
            log_entry["order_id"] = res.get("orderId")
            print(f"  OK  {rec.get('ts')}: {signal} {lots}x{ticker}@{price} "
                  f"order={res.get('orderId')}")
        except Exception as e:
            log_entry["error"] = str(e)[:200]
            print(f"  ERR {rec.get('ts')}: {e}")
        log_rows.append(log_entry)

    import csv
    write_header = not PAPER_LOG.exists()
    with PAPER_LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in log_rows:
            w.writerow(r)
    print(f"\n  Лог записан: {PAPER_LOG}")


def cmd_live(ticker: str, base_size_rub: float = 5000.0, max_size_rub: float = 20000.0) -> None:
    """
    Live режим: читает ПОСЛЕДНЕЕ решение v23 на сегодня и исполняет в sandbox.
    Запускать раз в день после генерации сигнала (cron / scheduled task).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    audit_path = V23_DIR / "v23_decision_audit_log.jsonl"
    if not audit_path.exists():
        print("ERROR: нет audit log.")
        return

    today_rows = []
    with audit_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ts", "")[:10] == today:
                today_rows.append(rec)

    if not today_rows:
        print(f"  Нет сигналов на {today}. Пропускаем.")
        return

    rec = today_rows[-1]
    signal_long  = rec.get("signal_long",  "HOLD")
    signal_short = rec.get("signal_short", "HOLD")
    signal = "BUY" if signal_long == "BUY" else "SHORT" if signal_short == "SHORT" else ""
    if not signal:
        print(f"  HOLD на {today}. Ничего не делаем.")
        return

    print(f"  Сигнал на {today}: {signal} (p_up={rec.get('p_up')}, p_short={rec.get('p_short')})")
    cmd_replay(ticker, since=today, until=today,
               base_size_rub=base_size_rub, max_size_rub=max_size_rub)


# ===========================================================================
# 4. MAIN
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Tinkoff sandbox paper trading bridge")
    ap.add_argument("--setup", action="store_true",
                    help="Создать sandbox-счёт и пополнить")
    ap.add_argument("--find", metavar="QUERY", help="Найти инструмент")
    ap.add_argument("--status", action="store_true", help="Статус портфеля")
    ap.add_argument("--replay", action="store_true",
                    help="Проиграть исторические сигналы из v23 audit log")
    ap.add_argument("--live", action="store_true",
                    help="Прочитать сегодняшний сигнал и исполнить")
    ap.add_argument("--ticker", default="SLV", help="Тикер (default: SLV)")
    ap.add_argument("--since", help="С даты (YYYY-MM-DD)")
    ap.add_argument("--until", help="По дату (YYYY-MM-DD)")
    ap.add_argument("--initial-rub", type=float, default=100000.0)
    ap.add_argument("--base-size", type=float, default=5000.0)
    ap.add_argument("--max-size", type=float, default=20000.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Не выполнять ордера, только логировать")
    args = ap.parse_args()

    if args.setup:
        cmd_setup(initial_rub=args.initial_rub)
    elif args.find:
        cmd_find(args.find)
    elif args.status:
        cmd_status()
    elif args.replay:
        cmd_replay(args.ticker, since=args.since, until=args.until,
                   base_size_rub=args.base_size, max_size_rub=args.max_size,
                   dry_run=args.dry_run)
    elif args.live:
        cmd_live(args.ticker, base_size_rub=args.base_size, max_size_rub=args.max_size)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
