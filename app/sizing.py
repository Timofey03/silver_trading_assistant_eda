"""
app/sizing.py — Position sizing calculator для всех инструментов.

Поддерживает:
  • SLV (US ETF, USD-denominated)
  • SLVRUBF (MOEX silver futures, RUB, multiplier=100)
  • PLZL, POLY (российские акции, RUB)
  • Произвольный инструмент

Sizing методы:
  1) Fixed RUB: фиксированная сумма на каждую сделку
  2) Risk-based: рискуем X% капитала на stop-loss дистанцию
  3) Kelly: размер ∝ p_up edge
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Instrument:
    ticker: str
    name: str
    type: str                   # "etf" / "futures" / "share"
    currency: str
    lot_size: int = 1           # 1 для большинства, иногда 10/100
    multiplier: int = 1         # для futures: 100 для SLVRUBF
    min_price_step: float = 0.01


INSTRUMENTS = {
    "SLV": Instrument(
        ticker="SLV", name="iShares Silver Trust",
        type="etf", currency="USD",
        lot_size=1, multiplier=1,
    ),
    "SLVRUBF": Instrument(
        ticker="SLVRUBF", name="MOEX Silver Futures",
        type="futures", currency="RUB",
        lot_size=1, multiplier=100,
        min_price_step=0.01,
    ),
    "PLZL": Instrument(
        ticker="PLZL", name="Полюс (золото, RU)",
        type="share", currency="RUB",
        lot_size=1, multiplier=1,
    ),
    "POLY": Instrument(
        ticker="POLY", name="Polymetal (RU)",
        type="share", currency="RUB",
        lot_size=1, multiplier=1,
    ),
}


@dataclass
class SizingInputs:
    account_value_rub:  float   # текущая стоимость счёта в RUB
    risk_per_trade_pct: float   # % капитала, готовый потерять на одной сделке (1-3% типично)
    stop_distance_pct:  float   # дистанция до стопа в %, e.g. 0.07 для trailing 7%
    instrument:         Instrument
    current_price:      float   # текущая цена инструмента в валюте инструмента
    usd_rub_rate:       float = 80.0   # курс для конвертации USD→RUB

    # Опциональные модификаторы
    p_up:               float = 0.55   # вероятность UP (для Kelly tilt)
    confidence_tilt:    bool = True    # применять Kelly tilt


@dataclass
class SizingResult:
    lots:               int
    notional_rub:       float        # полный notional позиции в RUB
    margin_rub:         float        # для futures: ~10% от notional
    max_loss_rub:       float        # если стоп сработает
    max_loss_pct_of_account: float
    risk_pct_used:      float        # фактический % риска
    reason:             str
    warnings:           list


def calculate_position_size(inputs: SizingInputs) -> SizingResult:
    """
    Рассчитывает размер позиции по risk-based методу с Kelly-tilt.

    Логика:
      1. Базовый risk = account × risk_per_trade_pct
      2. Stop distance в RUB на 1 лот = price × multiplier × stop_distance_pct
      3. Базовое кол-во лотов = base_risk / stop_distance_per_lot
      4. Kelly tilt: умножаем на (1 + (p_up - 0.5) × 2)
      5. Округляем вниз, проверяем что хватает капитала
    """
    inst = inputs.instrument
    warnings_list = []

    # Конвертируем USD-инструменты в RUB
    if inst.currency == "USD":
        price_rub = inputs.current_price * inputs.usd_rub_rate
    else:
        price_rub = inputs.current_price

    notional_per_lot_rub = price_rub * inst.multiplier

    if notional_per_lot_rub <= 0:
        return SizingResult(0, 0, 0, 0, 0, 0,
                            "ОШИБКА: цена ≤ 0", ["Некорректная цена"])

    # 1. Базовый risk
    base_risk_rub = inputs.account_value_rub * (inputs.risk_per_trade_pct / 100)

    # 2. Stop distance в RUB на 1 лот
    stop_per_lot_rub = notional_per_lot_rub * inputs.stop_distance_pct

    # 3. Базовое кол-во лотов
    base_lots = base_risk_rub / stop_per_lot_rub if stop_per_lot_rub > 0 else 0

    # 4. Kelly tilt — масштаб от p_up
    if inputs.confidence_tilt:
        p_excess = max(0.0, inputs.p_up - 0.5) * 2  # 0..1
        # При p_up=0.5 — 0.5x базы, при p_up=0.6 — 1.0x, при p_up=0.7 — 1.5x
        kelly_multiplier = 0.5 + p_excess
    else:
        kelly_multiplier = 1.0

    adjusted_lots = base_lots * kelly_multiplier

    # 5. Округляем вниз
    final_lots = int(adjusted_lots)
    if final_lots < 1 and adjusted_lots > 0.5:
        final_lots = 1
        warnings_list.append("Округление вверх до 1 лота")

    if final_lots < 1:
        return SizingResult(
            lots=0,
            notional_rub=0, margin_rub=0,
            max_loss_rub=0, max_loss_pct_of_account=0,
            risk_pct_used=0,
            reason="Слишком маленький счёт или слишком большой стоп — "
                   "размер позиции <1 лота",
            warnings=warnings_list,
        )

    # 6. Финальные числа
    notional_total = final_lots * notional_per_lot_rub
    margin_pct = 0.12 if inst.type == "futures" else 1.0  # 12% маржа на фьючерсы
    margin_rub = notional_total * margin_pct
    max_loss_rub = final_lots * stop_per_lot_rub
    max_loss_pct = (max_loss_rub / inputs.account_value_rub) * 100

    # Проверка: хватает ли денег
    if margin_rub > inputs.account_value_rub * 0.5:
        warnings_list.append(
            f"⚠ Маржа ({margin_rub/1000:.0f}k ₽) > 50% счёта — слишком крупная позиция"
        )
    if margin_rub > inputs.account_value_rub:
        return SizingResult(0, 0, 0, 0, 0, 0,
                            "Недостаточно капитала",
                            ["Маржа > всего счёта"])

    return SizingResult(
        lots=final_lots,
        notional_rub=notional_total,
        margin_rub=margin_rub,
        max_loss_rub=max_loss_rub,
        max_loss_pct_of_account=max_loss_pct,
        risk_pct_used=max_loss_pct,
        reason=f"Risk-based + Kelly tilt (p_up={inputs.p_up:.2f}, "
               f"multiplier={kelly_multiplier:.2f})",
        warnings=warnings_list,
    )


def explain_sizing(result: SizingResult, inputs: SizingInputs) -> str:
    """Markdown-форматированное объяснение для пользователя."""
    if result.lots == 0:
        return f"**❌ Невозможно открыть позицию**: {result.reason}"

    inst = inputs.instrument
    pct_account = (result.notional_rub / inputs.account_value_rub) * 100

    md = f"""
**✅ Рекомендация: купить {result.lots} лот{'ов' if result.lots > 1 else ''} {inst.ticker}**

📊 **Размер позиции**:
- Лотов: **{result.lots}**
- Notional: **{result.notional_rub:,.0f} ₽** ({pct_account:.1f}% от счёта)
- Маржа (для входа): **{result.margin_rub:,.0f} ₽**

⚠ **Если сработает стоп (-{inputs.stop_distance_pct*100:.0f}%)**:
- Убыток: **−{result.max_loss_rub:,.0f} ₽**
- Это **{result.max_loss_pct_of_account:.2f}% от счёта**

📈 **Если позиция вырастет на +10%**:
- Прибыль: ~{result.notional_rub * 0.10:,.0f} ₽
- Это **{result.notional_rub * 0.10 / inputs.account_value_rub * 100:.1f}% от счёта**

🧮 **Метод расчёта**: {result.reason}
"""
    if result.warnings:
        md += "\n⚠ **Предупреждения**:\n"
        for w in result.warnings:
            md += f"- {w}\n"
    return md.strip()
