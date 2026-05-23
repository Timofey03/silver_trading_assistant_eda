"""GET /api/signal — current BUY/HOLD/SELL signal from E3b daily reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
E3B_TRADING = REPO_ROOT / "daily_reports" / "e3b" / "trading"


class SignalResponse(BaseModel):
    signal: str           # "BUY" | "HOLD" | "SELL"
    date: str
    close: float
    p_up: float
    entry_threshold: float
    exit_threshold: float
    trail_pct: float
    max_hold_days: int
    cooldown_days: int
    alert_type: Optional[str] = None       # "action" | "info"
    is_repeat: Optional[bool] = None
    previous_signal: Optional[str] = None
    n_features_used: Optional[int] = None
    selected_features: Optional[list] = None
    source: str = "e3b_daily"
    report_dir: Optional[str] = None


router = APIRouter()


def _load_latest_signal() -> Optional[dict]:
    """Найти последний E3b сигнал."""
    if not E3B_TRADING.exists():
        return None
    dirs = sorted([d for d in E3B_TRADING.iterdir() if d.is_dir()], reverse=True)
    for d in dirs:
        sig_file = d / "signal.json"
        if sig_file.exists():
            try:
                data = json.loads(sig_file.read_text(encoding="utf-8"))
                data["report_dir"] = d.name
                return data
            except Exception:
                continue
    return None


@router.get("/signal", response_model=SignalResponse)
def get_signal():
    """Возвращает текущий сигнал E3b."""
    sig = _load_latest_signal()
    if sig is None:
        # Fallback: nothing
        return SignalResponse(
            signal="HOLD",
            date="—",
            close=0.0,
            p_up=0.0,
            entry_threshold=0.48,
            exit_threshold=0.35,
            trail_pct=0.12,
            max_hold_days=30,
            cooldown_days=25,
            source="none",
        )

    date_str = sig.get("date", "")
    if isinstance(date_str, str) and "T" in date_str:
        date_str = date_str.split("T")[0]

    return SignalResponse(
        signal=sig.get("signal", "HOLD"),
        date=date_str,
        close=float(sig.get("close", 0)),
        p_up=float(sig.get("p_up", 0)),
        entry_threshold=float(sig.get("entry_threshold", 0.48)),
        exit_threshold=float(sig.get("exit_threshold", 0.35)),
        trail_pct=float(sig.get("trail_pct", 0.12)),
        max_hold_days=int(sig.get("max_hold_days", 30)),
        cooldown_days=int(sig.get("cooldown_days", 25)),
        alert_type=sig.get("alert_type"),
        is_repeat=bool(sig.get("is_repeat", False)),
        previous_signal=sig.get("previous_signal"),
        n_features_used=int(sig.get("n_features_used", 30)),
        selected_features=sig.get("selected_features", [])[:10],  # top 10 для UI
        source="e3b_daily",
        report_dir=sig.get("report_dir"),
    )
