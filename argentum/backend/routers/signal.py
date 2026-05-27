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
    p_up: float           # Smoothed 3-day average (для UI)
    p_up_raw: float = 0.0 # Сегодняшний raw p_up (для дебага)
    entry_threshold: float
    exit_threshold: float
    trail_pct: float
    max_hold_days: int
    cooldown_days: int
    alert_type: Optional[str] = None
    is_repeat: Optional[bool] = None
    previous_signal: Optional[str] = None
    n_features_used: Optional[int] = None
    selected_features: Optional[list] = None
    source: str = "e3b_daily"
    report_dir: Optional[str] = None
    # OOD detector
    is_ood: bool = False
    ood_score: float = 0.0
    ood_summary: str = ""


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


def _load_last_n_signals(n: int = 3) -> list[dict]:
    """Последние N signal.json для smoothing."""
    if not E3B_TRADING.exists():
        return []
    dirs = sorted([d for d in E3B_TRADING.iterdir() if d.is_dir()], reverse=True)
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


STRONG_THRESHOLD = 0.85


def _ood_check() -> dict:
    """OOD detector на текущих features. Cached 5 мин."""
    from cache import ttl_cache

    @ttl_cache(ttl_seconds=300, key_args=False)
    def _inner():
        try:
            import sys
            REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
            sys.path.insert(0, str(REPO_ROOT))
            from app.multi_asset.ood_detector import OODDetector
            from app.multi_asset.features import build_feature_frame
            import warnings; warnings.filterwarnings("ignore")

            ood_path = REPO_ROOT / "baseline_outputs_multiasset" / "ood_detector.json"
            if not ood_path.exists():
                return {"is_ood": False, "ood_score": 0.0, "summary": "OOD detector not fitted"}

            det = OODDetector.load(ood_path)
            feats = build_feature_frame(target="silver", ffill_limit=5, audit_ffill=False)
            current = feats.iloc[-1]
            return det.check(current)
        except Exception as e:
            return {"is_ood": False, "ood_score": 0.0, "summary": f"OOD check failed: {e}"}

    return _inner()


# Lazy-init in get_signal to avoid module-load slowness
_OOD_CACHE = _ood_check


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

    # Smoothed p_up по последним 3 signal.json (синхронно с /positions master)
    recent = _load_last_n_signals(n=3)
    p_ups = [float(s.get("p_up", 0)) for s in recent if s.get("p_up") is not None]
    p_up_smoothed = sum(p_ups) / len(p_ups) if p_ups else float(sig.get("p_up", 0))
    p_up_raw = float(sig.get("p_up", 0))

    # Strong filter — применяется к smoothed: ≥0.85 → BUY, иначе HOLD
    raw_signal = sig.get("signal", "HOLD")
    effective_signal = raw_signal
    if raw_signal == "BUY" and p_up_smoothed < STRONG_THRESHOLD:
        effective_signal = "HOLD"

    # OOD check
    try:
        ood = _OOD_CACHE()
    except Exception:
        ood = {"is_ood": False, "ood_score": 0.0, "summary": "OOD check error"}

    return SignalResponse(
        signal=effective_signal,
        date=date_str,
        close=float(sig.get("close", 0)),
        p_up=p_up_smoothed,           # SMOOTHED — для UI и единой логики
        p_up_raw=p_up_raw,             # raw сегодняшнее значение
        entry_threshold=float(sig.get("entry_threshold", 0.48)),
        exit_threshold=float(sig.get("exit_threshold", 0.35)),
        trail_pct=float(sig.get("trail_pct", 0.20)),
        max_hold_days=int(sig.get("max_hold_days", 60)),
        cooldown_days=int(sig.get("cooldown_days", 10)),
        alert_type=sig.get("alert_type"),
        is_repeat=bool(sig.get("is_repeat", False)),
        previous_signal=sig.get("previous_signal"),
        n_features_used=int(sig.get("n_features_used", 30)),
        selected_features=sig.get("selected_features", [])[:10],
        source="e3b_daily",
        report_dir=sig.get("report_dir"),
        is_ood=bool(ood.get("is_ood", False)),
        ood_score=float(ood.get("ood_score", 0)),
        ood_summary=str(ood.get("summary", "")),
    )
