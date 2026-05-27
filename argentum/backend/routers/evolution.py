"""GET /api/evolution — evolution метрик через все эксперименты E1..Ensemble."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = REPO_ROOT / "baseline_outputs_multiasset"

# Эксперименты в хронологическом порядке + человеческие имена
EXPERIMENTS = [
    {
        "id": "e1_baseline",
        "name": "E1 Baseline",
        "description": "HistGB на 5 per-asset фичах. Минимальная модель — точка отсчёта.",
        "stage": "baseline",
    },
    {
        "id": "e2_cross_asset",
        "name": "E2 Cross-Asset",
        "description": "Добавили cross-asset ratios (gold/silver, etc.) и correlations.",
        "stage": "feature_eng",
    },
    {
        "id": "e2b_feature_selected",
        "name": "E2b Feature Selection",
        "description": "Top-30 features через SelectKBest mutual_info_classif.",
        "stage": "feature_eng",
    },
    {
        "id": "e3a_macro",
        "name": "E3a Macro",
        "description": "Добавили macro: T10YIE, USDRUB, CPI, INDPRO, VIX.",
        "stage": "macro",
    },
    {
        "id": "e3b_adaptive_ffill0_backup",
        "name": "E3b OLD (data gaps)",
        "description": "⚠ Изначальный E3b — оказался артефактом дыр в walk-forward (ffill_limit=0 на месячных макро).",
        "stage": "artifact",
    },
    {
        "id": "e3c_metalabel",
        "name": "E3c Meta-Labeling",
        "description": "2-stage classifier: primary + meta-confidence. Не зашло.",
        "stage": "advanced",
    },
    {
        "id": "e4_stacking",
        "name": "E4 Stacking",
        "description": "Стекинг 3 базовых моделей (HistGB + RF + XGB). Worse than E3b.",
        "stage": "advanced",
    },
    {
        "id": "e3b_adaptive",
        "name": "E3b Ensemble (FINAL)",
        "description": "🚀 Очищенные данные + smoothing + entry=0.85 + breakout_120 + momentum ensemble.",
        "stage": "final",
    },
]


class ExperimentMetrics(BaseModel):
    id: str
    name: str
    description: str
    stage: str
    sharpe: float = 0.0
    sortino: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    max_dd: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    profit_factor: float = 0.0
    period_years: float = 0.0
    available: bool = False


class EvolutionResponse(BaseModel):
    experiments: list[ExperimentMetrics]
    best_sharpe: str = ""
    best_return: str = ""


router = APIRouter()


@router.get("/evolution", response_model=EvolutionResponse)
def get_evolution():
    """Метрики всех экспериментов от E1 до финального ensemble."""
    result = []
    best_sharpe_val = -999
    best_return_val = -999
    best_sharpe_id = ""
    best_return_id = ""

    for exp in EXPERIMENTS:
        metrics_file = EXP_DIR / exp["id"] / "metrics.json"
        item = ExperimentMetrics(
            id=exp["id"], name=exp["name"],
            description=exp["description"], stage=exp["stage"],
        )
        if metrics_file.exists():
            try:
                m = json.loads(metrics_file.read_text(encoding="utf-8"))
                item.sharpe = float(m.get("sharpe", 0))
                item.sortino = float(m.get("sortino", 0))
                item.total_return = float(m.get("total_return", 0))
                item.annual_return = float(m.get("annual_return", 0))
                item.max_dd = float(m.get("max_dd", 0))
                item.win_rate = float(m.get("win_rate", 0))
                item.n_trades = int(m.get("n_trades", 0))
                item.profit_factor = float(m.get("profit_factor", 0))
                item.period_years = float(m.get("period_years", 0))
                item.available = True

                if item.sharpe > best_sharpe_val:
                    best_sharpe_val = item.sharpe
                    best_sharpe_id = exp["id"]
                if item.total_return > best_return_val:
                    best_return_val = item.total_return
                    best_return_id = exp["id"]
            except Exception:
                pass
        result.append(item)

    return EvolutionResponse(
        experiments=result,
        best_sharpe=best_sharpe_id,
        best_return=best_return_id,
    )
