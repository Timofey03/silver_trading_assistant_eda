"""GET /api/tinkoff/* — интеграция с Tinkoff Invest sandbox."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


class TinkoffBalance(BaseModel):
    connected: bool
    total_rub: float = 0.0
    expected_yield_rub: float = 0.0
    open_positions: int = 0
    error: str = ""


router = APIRouter()


@router.get("/tinkoff/balance", response_model=TinkoffBalance)
def get_balance():
    """Live баланс из Tinkoff sandbox."""
    token = os.getenv("TINKOFF_TOKEN", "").strip()
    if not token:
        return TinkoffBalance(
            connected=False,
            error="TINKOFF_TOKEN не задан в .env",
        )

    try:
        # Используем существующий paper-trading модуль
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from tinkoff.invest import Client
        from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX

        with Client(token, target=INVEST_GRPC_API_SANDBOX) as client:
            accounts = client.users.get_accounts()
            if not accounts.accounts:
                return TinkoffBalance(connected=False, error="Нет sandbox аккаунтов")

            acc_id = accounts.accounts[0].id
            portfolio = client.operations.get_portfolio(account_id=acc_id)

            total = portfolio.total_amount_portfolio
            total_rub = total.units + total.nano / 1e9

            yield_amount = portfolio.expected_yield
            yield_rub = yield_amount.units + yield_amount.nano / 1e9

            return TinkoffBalance(
                connected=True,
                total_rub=float(total_rub),
                expected_yield_rub=float(yield_rub),
                open_positions=len(portfolio.positions),
            )
    except Exception as e:
        return TinkoffBalance(connected=False, error=f"{type(e).__name__}: {e}")
