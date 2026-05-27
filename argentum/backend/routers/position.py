"""GET /api/position — открытая позиция (если signal=BUY и нет закрытия)."""
from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
E3B_TRADING = REPO_ROOT / "daily_reports" / "e3b" / "trading"
TRADES_CSV  = REPO_ROOT / "baseline_outputs_multiasset" / "e3b_adaptive" / "trades.csv"
SILVER_PARQUET = REPO_ROOT / "data" / "multi_asset" / "metals" / "silver_daily.parquet"


class OpenPosition(BaseModel):
    is_open: bool
    entry_date: str = ""
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_return: float = 0.0
    days_held: int = 0
    max_hold_days: int = 30
    trail_pct: float = 0.12
    stop_price: float = 0.0
    target_close: float = 0.0
    signal: str = "HOLD"
    p_up: float = 0.0
    source: str = "computed"
    # Regime context
    regime_allows_trade: bool = True
    regime_reason: str = ""
    # OOD context
    is_ood: bool = False
    ood_score: float = 0.0
    ood_summary: str = ""


router = APIRouter()


def _all_signal_files() -> list[Path]:
    """Все signal*.json во всех trading-датах, отсортированные по времени."""
    if not E3B_TRADING.exists():
        return []
    out = []
    for d in sorted(E3B_TRADING.iterdir()):
        if d.is_dir():
            out += sorted(d.glob("signal*.json"))
    return out


def _find_entry_date() -> Optional[tuple[str, float]]:
    """
    Сканирует историю signal-файлов чтобы найти момент HOLD→BUY.
    Возвращает (entry_date_iso, entry_price) или None.
    """
    files = _all_signal_files()
    if not files:
        return None

    # Идём с конца назад до момента смены previous_signal != BUY
    for f in reversed(files):
        try:
            sig = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if sig.get("signal") != "BUY":
            return None  # последняя смена — не BUY, значит позиция не открыта
        if sig.get("previous_signal") != "BUY":
            # Нашли точку входа
            d = sig.get("date", "")
            if "T" in d:
                d = d.split("T")[0]
            return d, float(sig.get("close", 0))

    # Все BUY — берём самый ранний (резерв)
    try:
        first = json.loads(files[0].read_text(encoding="utf-8"))
        d = first.get("date", "")
        if "T" in d:
            d = d.split("T")[0]
        return d, float(first.get("close", 0))
    except Exception:
        return None


def _live_silver_price() -> Optional[float]:
    """Свежая цена серебра — из parquet или yfinance fallback."""
    # 1. Parquet
    if SILVER_PARQUET.exists():
        try:
            df = pd.read_parquet(SILVER_PARQUET)
            if "close" in df.columns and len(df):
                return float(df["close"].iloc[-1])
        except Exception:
            pass
    # 2. yfinance fallback
    try:
        ticker = yf.Ticker("SI=F")
        hist = ticker.history(period="5d")
        if len(hist):
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


STRONG_THRESHOLD = 0.85   # smoothed p_up для strong-signal filter


def _check_ood() -> dict:
    """Проверить current features против OOD detector."""
    ood_file = REPO_ROOT / "baseline_outputs_multiasset" / "ood_detector.json"
    if not ood_file.exists():
        return {"is_ood": False, "ood_score": 0.0, "summary": "detector not fitted"}
    try:
        import sys as _sys
        _sys.path.insert(0, str(REPO_ROOT))
        from app.multi_asset.ood_detector import OODDetector
        from app.multi_asset.features import build_feature_frame
        from app.multi_asset.metal_loader import load_metals
        det = OODDetector.load(ood_file)
        metals = load_metals()
        features = build_feature_frame(target="silver", metals=metals,
                                       ffill_limit=5, audit_ffill=False)
        last_features = features.iloc[-1]
        return det.check(last_features)
    except Exception as e:
        return {"is_ood": False, "ood_score": 0.0, "summary": f"check failed: {e}"}


def _check_regime(p_up_today: float) -> tuple[bool, str]:
    """
    Проверяет strong-signal filter: модель торгует только когда smoothed
    p_up >= STRONG_THRESHOLD (0.85).

    Grid search (clean data):
      no_filter:           Sharpe 0.515 / +91%
      strong_signal_only:  Sharpe 1.202 / +271% / DD -5.6% / Win 80.8% ← winner
    """
    if p_up_today >= STRONG_THRESHOLD:
        return True, f"strong signal: p_up={p_up_today:.2f} ≥ {STRONG_THRESHOLD}"
    else:
        return False, (
            f"weak signal: p_up={p_up_today:.2f} < {STRONG_THRESHOLD} "
            f"— модель в зоне шума (0.4-0.8), не торгует"
        )


def _was_closed_after(entry_date: str) -> bool:
    """True если в trades.csv есть закрытая сделка с exit_date >= entry_date."""
    if not TRADES_CSV.exists():
        return False
    try:
        df = pd.read_csv(TRADES_CSV)
        if "exit_date" not in df.columns:
            return False
        ed = pd.to_datetime(entry_date)
        df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
        return bool((df["exit_date"] >= ed).any())
    except Exception:
        return False


def _open_trade_from_trades_csv() -> Optional[dict]:
    """
    Самый авторитетный источник: backtest trades.csv с exit_reason='OPEN'.
    Возвращает dict с entry_date / entry_price / peak_price если есть.
    """
    if not TRADES_CSV.exists():
        return None
    try:
        df = pd.read_csv(TRADES_CSV)
        if "exit_reason" not in df.columns:
            return None
        opens = df[df["exit_reason"] == "OPEN"]
        if opens.empty:
            return None
        row = opens.iloc[-1]
        return {
            "entry_date":  str(row["entry_date"]),
            "entry_price": float(row["entry_price"]),
            "peak_price":  float(row["peak_price"]),
            "hold_days":   int(row["hold_days"]),
        }
    except Exception:
        return None


@router.get("/position", response_model=OpenPosition)
def get_position():
    """Текущая открытая позиция (или is_open=False).

    Приоритет источника:
    1. trades.csv с exit_reason='OPEN' (backtest, authoritative)
    2. signal.json (production inference, если backtest нет)
    """
    files = _all_signal_files()
    if not files:
        return OpenPosition(is_open=False)

    # Берём самый свежий сигнал для p_up / параметров
    try:
        latest = json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return OpenPosition(is_open=False)

    signal = latest.get("signal", "HOLD")
    p_up = float(latest.get("p_up", 0))
    trail_pct = float(latest.get("trail_pct", 0.20))
    max_hold = int(latest.get("max_hold_days", 60))

    # Strong-signal check (применяется к ОБОИМ источникам)
    regime_allows, regime_reason = _check_regime(p_up)

    # OOD check (доп. контекст)
    ood_result = _check_ood()
    ood_dict = {
        "is_ood":      bool(ood_result.get("is_ood", False)),
        "ood_score":   float(ood_result.get("ood_score", 0.0)),
        "ood_summary": str(ood_result.get("summary", "")),
    }

    # === Источник №1: trades.csv OPEN row ===
    backtest_open = _open_trade_from_trades_csv()
    if backtest_open:
        current = _live_silver_price() or float(latest.get("close", 0))
        entry_price = backtest_open["entry_price"]
        peak_price = max(backtest_open["peak_price"], current)
        unrealized = (current - entry_price) / entry_price if entry_price > 0 else 0.0
        try:
            entry_d = datetime.fromisoformat(backtest_open["entry_date"]).date()
            days = (date.today() - entry_d).days
        except Exception:
            days = backtest_open["hold_days"]
        return OpenPosition(
            is_open=True,
            entry_date=backtest_open["entry_date"],
            entry_price=entry_price,
            current_price=current,
            unrealized_return=unrealized,
            days_held=days,
            max_hold_days=max_hold,
            trail_pct=trail_pct,
            stop_price=peak_price * (1 - trail_pct),
            target_close=entry_price * 1.15,
            signal=signal,
            p_up=p_up,
            source="backtest_trades_csv",
            regime_allows_trade=regime_allows,
            regime_reason=regime_reason,
            **ood_dict,
        )

    # Apply regime filter to live signal
    if signal == "BUY" and not regime_allows:
        return OpenPosition(
            is_open=False,
            signal=signal,
            p_up=p_up,
            trail_pct=trail_pct,
            max_hold_days=max_hold,
            regime_allows_trade=False,
            regime_reason=regime_reason,
            **ood_dict,
        )

    if signal != "BUY":
        return OpenPosition(
            is_open=False,
            signal=signal,
            p_up=p_up,
            trail_pct=trail_pct,
            max_hold_days=max_hold,
            regime_allows_trade=regime_allows,
            regime_reason=regime_reason,
            **ood_dict,
        )

    # Ищем точку входа
    entry = _find_entry_date()
    if entry is None:
        return OpenPosition(is_open=False, signal=signal, p_up=p_up)
    entry_date, entry_price = entry

    # Если уже была закрытая сделка с этой даты — позиция не открыта
    if _was_closed_after(entry_date):
        return OpenPosition(
            is_open=False,
            signal=signal,
            p_up=p_up,
            trail_pct=trail_pct,
            max_hold_days=max_hold,
        )

    # Live price
    current = _live_silver_price() or float(latest.get("close", 0))
    unrealized = (current - entry_price) / entry_price if entry_price > 0 else 0.0

    # Дни в позиции
    try:
        entry_d = datetime.fromisoformat(entry_date).date()
        days = (date.today() - entry_d).days
    except Exception:
        days = 0

    # Trailing stop level
    stop_price = current * (1 - trail_pct)
    target_close = float(latest.get("target_close", current * 1.05))  # резерв 5%

    return OpenPosition(
        is_open=True,
        entry_date=entry_date,
        entry_price=entry_price,
        current_price=current,
        unrealized_return=unrealized,
        days_held=days,
        max_hold_days=max_hold,
        trail_pct=trail_pct,
        stop_price=stop_price,
        target_close=target_close,
        signal=signal,
        p_up=p_up,
    )
