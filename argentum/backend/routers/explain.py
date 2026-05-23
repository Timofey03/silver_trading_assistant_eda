"""GET /api/explain — "почему BUY?" — feature importance из ML модели."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
E3B_TRAINING = REPO_ROOT / "daily_reports" / "e3b" / "training"


class FeatureInsight(BaseModel):
    feature: str
    human_name: str       # "Long-term gold volatility"
    interpretation: str   # "Растёт — поддерживает BUY"


class ExplainResponse(BaseModel):
    insights: List[FeatureInsight]
    model_version: str
    last_updated: str


router = APIRouter()


# Human-readable названия фичей
FEATURE_DESCRIPTIONS = {
    "silver_rvol_60": ("60-дневная волатильность серебра",
                        "Долгосрочная волатильность — индикатор режима рынка"),
    "silver_rvol_20": ("Короткая волатильность серебра",
                        "Скорость движения цены"),
    "silver_atr_14": ("ATR серебра", "Среднее истинное движение цены"),
    "silver_dist_ma200": ("Расстояние до 200-дневной MA",
                            "Положение относительно долгосрочного тренда"),
    "silver_dist_ma50": ("Расстояние до 50-дневной MA",
                          "Положение относительно среднесрочного тренда"),
    "gold_rvol_60":   ("Волатильность золота (длинная)",
                        "Защитный спрос на драгметаллы"),
    "gold_rvol_20":   ("Волатильность золота (короткая)",
                        "Текущая активность золота"),
    "gold_vol_z":     ("Z-score объёма золота",
                        "Аномалии в объёмах торгов золотом"),
    "palladium_rvol_60": ("Долгосрочная волатильность палладия",
                          "Регим автомобильного спроса (катализаторы)"),
    "palladium_vol_z":   ("Z-score объёма палладия",
                          "Институциональный интерес к палладию"),
    "palladium_rsi_14":  ("RSI палладия", "Перекупленность/перепроданность"),
    "platinum_rvol_60":  ("Волатильность платины", "Промышленный спрос"),
    "platinum_vol_z":    ("Z-score объёма платины", "Аномалии в палладии"),
    "platinum_dist_ma200": ("Расстояние платины до 200MA",
                              "Долгосрочный тренд платины"),
    "platinum_rvol_20":  ("Короткая волатильность платины",
                            "Активность платины"),
    "copper_rvol_60":    ("Волатильность меди (длинная)",
                            "Промышленный цикл — proxy industrial demand"),
    "corr_silver_gold_90": ("Корреляция silver-gold 90д",
                              "Согласованность драгметаллов"),
    "ratio_gold_silver":   ("Отношение gold/silver",
                              "Классический индикатор силера"),
    "ratio_silver_copper": ("Отношение silver/copper",
                              "Драгметалл vs индустриальный"),
    "DGS10":   ("10-летние US Treasury", "Доходности — обратная связь с золотом"),
    "DFII10":  ("TIPS 10Y (реальные ставки)",
                  "Реальные ставки — главный драйвер драгметаллов"),
    "DTWEXBGS": ("USD Index", "Сила доллара — обратная связь с silver"),
    "VIXCLS":  ("VIX (страх рынка)", "Защитный спрос в страхе"),
    "INDPRO":  ("US Industrial Production",
                  "Промышленный спрос на silver"),
    "USDRUB":  ("USDRUB", "Для российских пользователей SLVRUBF"),
    "DCOILWTICO": ("Цена нефти WTI", "Commodity context"),
}


def _load_latest_signal():
    if not E3B_TRAINING.exists():
        return {}
    trading_dir = REPO_ROOT / "daily_reports" / "e3b" / "trading"
    if not trading_dir.exists():
        return {}
    dirs = sorted([d for d in trading_dir.iterdir() if d.is_dir()], reverse=True)
    for d in dirs:
        sig_file = d / "signal.json"
        if sig_file.exists():
            try:
                return json.loads(sig_file.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


@router.get("/explain", response_model=ExplainResponse)
def explain():
    """Объяснение текущего сигнала через top features."""
    sig = _load_latest_signal()
    features = sig.get("selected_features", [])[:10]  # top 10

    insights = []
    p_up = float(sig.get("p_up", 0))
    direction = "поддерживает BUY" if p_up >= 0.48 else (
        "указывает на SELL" if p_up < 0.35 else "режим ожидания"
    )

    for f in features:
        if f in FEATURE_DESCRIPTIONS:
            human, base_interp = FEATURE_DESCRIPTIONS[f]
            insights.append(FeatureInsight(
                feature=f,
                human_name=human,
                interpretation=f"{base_interp} · сейчас {direction}",
            ))
        else:
            insights.append(FeatureInsight(
                feature=f,
                human_name=f.replace("_", " ").title(),
                interpretation=f"Сейчас {direction}",
            ))

    return ExplainResponse(
        insights=insights,
        model_version="E3b multi-asset + adaptive barriers",
        last_updated=sig.get("date", "—").split("T")[0] if sig else "—",
    )
