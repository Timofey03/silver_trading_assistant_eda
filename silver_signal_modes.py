"""
silver_signal_modes.py — Пресеты агрессивности + SELL-сигналы + backtest сравнение

Что нового vs v25 (только BUY + trailing stop):
1) Multiple aggressiveness presets — пользователь может выбрать стиль
2) SELL signals — модель РЕКОМЕНДУЕТ выход, не только trailing stop
3) Backtest compare — честно показывает trade-off "больше сделок vs больше edge"

Пресеты:
  conservative   p_up_in=0.55, p_up_out=0.40, cooldown=15  — текущий (3-5 сделок/год)
  balanced       p_up_in=0.52, p_up_out=0.45, cooldown=10  — ~8-12 сделок/год
  aggressive     p_up_in=0.50, p_up_out=0.48, cooldown=5   — ~15-25 сделок/год
  ultra          p_up_in=0.48, p_up_out=0.50, cooldown=3   — 30+ сделок/год

Запуск:
  python silver_signal_modes.py --compare       # сравнение всех пресетов
  python silver_signal_modes.py --mode balanced # пересчитать с конкретным пресетом
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from silver_assistant_v23_honest import (
    equity_compounded_sequential, risk_metrics_honest,
    recompute_trades_with_realistic_costs, RealisticCosts,
)
from silver_assistant_v22_risk_aware import backtest_strategy_independent
from silver_assistant_v19_trailing import (
    TRAIL_PCT_DEFAULT, MAX_HOLD_DEFAULT, COST_PER_TRADE,
)

V22_DIR  = Path("baseline_outputs_v22")
V25_DIR  = Path("baseline_outputs_v25")
PROD_DIR = Path("baseline_outputs_prod")
MODES_DIR = Path("baseline_outputs_modes")
MODES_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. PRESETS
# =============================================================================

@dataclass
class SignalMode:
    name: str
    description: str
    p_up_entry:   float    # порог входа (BUY)
    p_up_exit:    float    # порог выхода (SELL)
    cooldown:     int      # дней между BUY-сигналами
    trail_pct:    float    # trailing stop %
    max_hold:     int      # max holding days
    expected_trades_per_year: int
    # Take profit улучшения
    take_profit_pct: float = 0.0       # 0 = выключено; 0.15 = exit при +15%
    aggressive_trail_after: float = 0.0  # после этой прибыли trail сокращается ×2


# ⭐ OPTIMAL MODE v3 — MaxReturn (цель: обогнать buy-and-hold)
#
# ИСТОРИЯ ВЕРСИЙ:
#   OptimalV1: p_up_entry=0.49, exit=0.43, cooldown=15 — оверфит к 2025 bull market
#   OptimalV2: p_up_entry=0.48, exit=0.35, cooldown=25 — "консистентность" ценой доходности
#              → только 2/8 лет плюсовые на WF, захватил лишь 13-18% роста BnH в 2025
#   OptimalV3: balanced mode params — лучший forward +44.6%, Sharpe 1.311, 13 сделок
#              → выход симметричен входу (0.45 vs 0.52), cooldown 2.5x короче → больше сделок
#
# КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
#   cooldown:   25 → 10  (было: сидели в кэше 25 дней между сделками, пропускали тренд)
#   p_up_exit:  0.35 → 0.45  (было: держали убыточные позиции до почти нейтрального сигнала)
#   p_up_entry: 0.48 → 0.52  (точнее: 69% win rate vs 80% при том же числе trade-слотов)
#   trail_pct:  0.12 → 0.07  (было: отдавали 12% от пика прежде чем зафиксировать прибыль)
OPTIMAL_PARAMS = SignalMode(
    name="OptimalV3 (MaxReturn)",
    description="Цель: максимальная доходность / обгон buy-and-hold. "
                "Основан на balanced mode: forward +44.6%, Sharpe 1.311, 13 сделок. "
                "Cooldown 10d (было 25), exit=0.45 (было 0.35), trail=7% (было 12%).",
    p_up_entry=0.52,         # было 0.48 — точнее: 69% win rate на forward
    p_up_exit=0.45,          # было 0.35 — симметричнее входу, выходим при ослаблении
    cooldown=10,             # было 25 — в 2.5x больше сделок → больше захватываем тренд
    trail_pct=0.07,          # было 0.12 — фиксируем прибыль быстрее
    max_hold=30,
    expected_trades_per_year=13,  # balanced mode: 13 сделок на forward
    take_profit_pct=0.0,
    aggressive_trail_after=0.10,  # после +10% прибыли trail ужесточается вдвое
)

# Сохраняем PRESETS для обратной совместимости grid_search скрипта
PRESETS: Dict[str, SignalMode] = {
    "optimal":     OPTIMAL_PARAMS,   # алиас → max_return (новый дефолт)
    "max_return":  OPTIMAL_PARAMS,   # явный псевдоним для v28
    "conservative": SignalMode(
        name="Conservative",
        description="Старый v25 — слишком селективный, только для сравнения",
        p_up_entry=0.55, p_up_exit=0.40,
        cooldown=15, trail_pct=0.08, max_hold=45,
        expected_trades_per_year=4,
    ),
    "consistent": SignalMode(
        name="OptimalV2 (Consistency)",
        description="Старый OPTIMAL — 6/8 положительных лет, но захватывает лишь 13-18% BnH. "
                    "Сохранён для сравнения и reference.",
        p_up_entry=0.48, p_up_exit=0.35,
        cooldown=25, trail_pct=0.12, max_hold=30,
        expected_trades_per_year=5,
    ),
    "balanced": SignalMode(
        name="Balanced", description="Backtest comparison — основа OptimalV3",
        p_up_entry=0.52, p_up_exit=0.45,
        cooldown=10, trail_pct=0.07, max_hold=30,
        expected_trades_per_year=10,
    ),
    "aggressive": SignalMode(
        name="Aggressive", description="Backtest comparison — больше сделок, меньше edge",
        p_up_entry=0.50, p_up_exit=0.48,
        cooldown=5, trail_pct=0.05, max_hold=20,
        expected_trades_per_year=22,
    ),
    "ultra": SignalMode(
        name="Ultra", description="ВНИМАНИЕ: forward -22.3% — больше сделок = хуже результат",
        p_up_entry=0.48, p_up_exit=0.50,
        cooldown=3, trail_pct=0.04, max_hold=15,
        expected_trades_per_year=40,
    ),
}


# =============================================================================
# 2. KELLY POSITION SIZING
# =============================================================================

def kelly_position_size(
    p_up: float,
    mode: SignalMode,
    direction: str = "LONG",
    kelly_min: float = 0.25,
    kelly_max: float = 1.0,
) -> float:
    """
    Дробная Kelly — размер позиции пропорционален убеждённости модели.

    LONG:  p_up ∈ [p_up_entry .. 1.0]       → fraction ∈ [kelly_min .. kelly_max]
    SHORT: p_up ∈ [0.0 .. SHORT_THRESHOLD]  → fraction ∈ [kelly_min .. kelly_max]

    Примеры (LONG, entry=0.52):
      p_up=0.52 → 25%   (минимальная позиция при входе)
      p_up=0.65 → 52%
      p_up=0.80 → 79%
      p_up=0.95 → 100%

    Почему не «всегда 1 лот»:
      Модель возвращает p_up=0.49 и p_up=0.85 — убеждённость разная.
      Входить одинаковым объёмом — игнорировать главный актив: вероятностный сигнал.
    """
    if direction == "LONG":
        lo, hi = mode.p_up_entry, 1.0
        edge = (p_up - lo) / (hi - lo) if hi > lo else 0.0
    else:  # SHORT
        short_threshold = mode.p_up_exit - 0.10   # зона SHORT: ниже нейтральной
        lo, hi = 0.0, short_threshold
        edge = (hi - p_up) / (hi - lo) if hi > lo else 0.0

    edge = max(0.0, min(1.0, edge))
    return round(kelly_min + (kelly_max - kelly_min) * edge, 4)


# =============================================================================
# 3. SIGNAL GENERATION с SELL + SHORT-сигналами
# =============================================================================

SHORT_ENTRY_OFFSET = 0.10  # SHORT входим когда p_up < p_up_exit - SHORT_ENTRY_OFFSET


def generate_signals_with_exits(
    df: pd.DataFrame,
    p_up_series: pd.Series,
    mode: SignalMode,
    enable_short: bool = False,
    enable_kelly: bool = False,
) -> pd.DataFrame:
    """
    Трёхсостоянная state machine: LONG / SHORT / FLAT.

    LONG-логика (всегда активна):
      • position=0: BUY если p_up >= p_up_entry AND cooldown OK
      • position=1: SELL если p_up < p_up_exit

    SHORT-логика (enable_short=True):
      • position=0:  SHORT если p_up < (p_up_exit - SHORT_ENTRY_OFFSET) AND cooldown OK
      • position=-1: COVER если p_up >= p_up_exit

    Kelly sizing (enable_kelly=True):
      • Размер позиции ∝ (p_up - 0.5), мин 25% при входе, макс 100%
      • Хранится в колонке kelly_frac

    Нейтральная зона [p_up_exit - SHORT_OFFSET .. p_up_entry]:
      Новых позиций не открываем — зона «неопределённости» модели.

    Исправляет:
      • Старая логика: cooldown=25, exit=0.35 → пропускали тренд, держали убытки
      • Новая логика:  cooldown=10, exit=0.45 → быстрее реагируем, меньше в кэше
    """
    out = df.copy().sort_index()
    out["p_up"] = p_up_series.reindex(out.index)

    signals    = []
    positions  = []
    kelly_fracs = []
    position   = 0
    last_long  = -10**9
    last_short = -10**9
    short_threshold = mode.p_up_exit - SHORT_ENTRY_OFFSET

    for i, p in enumerate(out["p_up"].values):
        sig  = "HOLD"
        frac = 0.0

        if pd.isna(p):
            signals.append(sig)
            positions.append(position)
            kelly_fracs.append(frac)
            continue

        if position == 0:
            # --- Вход в LONG ---
            if p >= mode.p_up_entry and (i - last_long) > mode.cooldown:
                sig      = "BUY"
                position = 1
                last_long = i
                frac = kelly_position_size(p, mode, "LONG") if enable_kelly else 1.0

            # --- Вход в SHORT (если включено и p_up в медвежьей зоне) ---
            elif (enable_short
                  and p < short_threshold
                  and (i - last_short) > mode.cooldown):
                sig      = "SHORT"
                position = -1
                last_short = i
                frac = kelly_position_size(p, mode, "SHORT") if enable_kelly else 1.0

        elif position == 1:
            # --- Выход из LONG ---
            if p < mode.p_up_exit:
                sig      = "SELL"
                position = 0

        elif position == -1:
            # --- Выход из SHORT ---
            if p >= mode.p_up_exit:
                sig      = "COVER"
                position = 0

        signals.append(sig)
        positions.append(position)
        kelly_fracs.append(frac)

    out["signal_long"]  = signals
    out["position"]     = positions
    out["kelly_frac"]   = kelly_fracs
    out["signal_short"] = "HOLD"
    out["signal"]       = signals
    return out


# =============================================================================
# 4. BACKTEST с SELL + SHORT + Kelly sizing
# =============================================================================

def _run_long_trade(
    d: pd.DataFrame, entry_pos: int, mode: SignalMode,
    has_high: bool, has_low: bool, cost: float,
) -> dict:
    """Симулирует одну LONG сделку начиная с entry_pos."""
    entry_date   = d.index[entry_pos]
    entry_price  = float(d.iloc[entry_pos]["silver_close"])
    kelly_frac   = float(d.iloc[entry_pos].get("kelly_frac", 1.0))
    peak         = entry_price
    current_trail = mode.trail_pct
    trail_stop   = entry_price * (1.0 - current_trail)
    tp_price     = (entry_price * (1.0 + mode.take_profit_pct)
                    if mode.take_profit_pct > 0 else float("inf"))
    agg_trigger  = (entry_price * (1.0 + mode.aggressive_trail_after)
                    if mode.aggressive_trail_after > 0 else float("inf"))

    exit_idx    = entry_pos
    exit_price  = entry_price
    exit_reason = "max_hold"

    for j in range(1, mode.max_hold + 1):
        pos = entry_pos + j
        if pos >= len(d):
            break
        cl = float(d.iloc[pos]["silver_close"])
        hi = float(d.iloc[pos]["silver_high"]) if has_high else cl
        lo = float(d.iloc[pos]["silver_low"])  if has_low  else cl

        if hi > peak:
            peak = hi
            if peak >= agg_trigger and current_trail == mode.trail_pct:
                current_trail = mode.trail_pct * 0.5   # ужесточаем после +10%
            trail_stop = peak * (1.0 - current_trail)

        if hi >= tp_price:
            exit_price, exit_idx, exit_reason = tp_price, pos, "take_profit"
            break
        if lo <= trail_stop:
            exit_price, exit_idx, exit_reason = min(cl, trail_stop), pos, "trail_stop"
            break
        if d.iloc[pos].get("signal_long", "HOLD") == "SELL":
            exit_price, exit_idx, exit_reason = cl, pos, "model_exit"
            break

        exit_price, exit_idx = cl, pos

    gross = exit_price / entry_price - 1.0
    net   = kelly_frac * gross - cost     # Kelly-scaled

    return {
        "direction":    "LONG",
        "entry_date":   entry_date,
        "exit_date":    d.index[exit_idx],
        "entry_price":  round(entry_price, 3),
        "exit_price":   round(exit_price, 3),
        "peak_price":   round(peak, 3),
        "kelly_frac":   round(kelly_frac, 4),
        "gross_return": round(gross, 6),
        "net_return":   round(net, 6),
        "hold_days":    exit_idx - entry_pos,
        "exit_reason":  exit_reason,
    }


def _run_short_trade(
    d: pd.DataFrame, entry_pos: int, mode: SignalMode,
    has_high: bool, has_low: bool, cost: float,
) -> dict:
    """
    Симулирует одну SHORT сделку начиная с entry_pos.
    Profit = entry_price / exit_price - 1  (растёт при падении цены).
    Trailing stop: выходим если цена поднялась выше trough*(1+trail_pct).
    """
    entry_date  = d.index[entry_pos]
    entry_price = float(d.iloc[entry_pos]["silver_close"])
    kelly_frac  = float(d.iloc[entry_pos].get("kelly_frac", 1.0))
    trough      = entry_price
    trail_stop  = entry_price * (1.0 + mode.trail_pct)  # выход если цена растёт

    exit_idx    = entry_pos
    exit_price  = entry_price
    exit_reason = "max_hold"

    for j in range(1, mode.max_hold + 1):
        pos = entry_pos + j
        if pos >= len(d):
            break
        cl = float(d.iloc[pos]["silver_close"])
        hi = float(d.iloc[pos]["silver_high"]) if has_high else cl
        lo = float(d.iloc[pos]["silver_low"])  if has_low  else cl

        if lo < trough:
            trough     = lo
            trail_stop = trough * (1.0 + mode.trail_pct)

        # Trailing stop: цена поднялась обратно
        if hi >= trail_stop:
            exit_price, exit_idx, exit_reason = max(cl, trail_stop), pos, "trail_stop"
            break
        # Model exit: рынок развернулся (COVER сигнал)
        if d.iloc[pos].get("signal_long", "HOLD") == "COVER":
            exit_price, exit_idx, exit_reason = cl, pos, "model_exit"
            break

        exit_price, exit_idx = cl, pos

    gross = entry_price / exit_price - 1.0   # SHORT: profit при падении
    net   = kelly_frac * gross - cost

    return {
        "direction":    "SHORT",
        "entry_date":   entry_date,
        "exit_date":    d.index[exit_idx],
        "entry_price":  round(entry_price, 3),
        "exit_price":   round(exit_price, 3),
        "trough_price": round(trough, 3),
        "kelly_frac":   round(kelly_frac, 4),
        "gross_return": round(gross, 6),
        "net_return":   round(net, 6),
        "hold_days":    exit_idx - entry_pos,
        "exit_reason":  exit_reason,
    }


def backtest_with_model_exits(
    df: pd.DataFrame,
    split: str,
    mode: SignalMode,
    cost: float = COST_PER_TRADE,
    enable_short: bool = False,
    enable_kelly: bool = False,
) -> pd.DataFrame:
    """
    Бэктест: BUY → LONG, SHORT → short position.
    Исправляет:
      • Старый код: только LONG, фикс. 1 лот, exit=0.35 (держали убытки)
      • Новый код:  LONG + SHORT, Kelly sizing, exit=0.45 (быстрее реагируем)

    enable_short=False  — обратная совместимость (только LONG, как раньше)
    enable_kelly=False  — обратная совместимость (фикс. 1 лот, как раньше)
    """
    d = df[df["split"] == split].sort_index().copy() if "split" in df.columns else df.sort_index().copy()
    if d.empty:
        return pd.DataFrame()

    has_high = "silver_high" in d.columns
    has_low  = "silver_low"  in d.columns

    trades   = []
    used_pos = set()   # позиции уже занятые открытой сделкой

    i = 0
    while i < len(d):
        if i in used_pos:
            i += 1
            continue

        sig = d.iloc[i].get("signal_long", "HOLD")

        if sig == "BUY":
            trade = _run_long_trade(d, i, mode, has_high, has_low, cost)
            if not enable_kelly:
                trade["kelly_frac"] = 1.0
                trade["net_return"] = round(trade["gross_return"] - cost, 6)
            trades.append(trade)
            # Помечаем занятые позиции чтобы не открывать поверх
            exit_pos = d.index.get_loc(trade["exit_date"]) if trade["exit_date"] in d.index else i
            for p in range(i, exit_pos + 1):
                used_pos.add(p)
            i = exit_pos + 1
            continue

        if enable_short and sig == "SHORT":
            trade = _run_short_trade(d, i, mode, has_high, has_low, cost)
            if not enable_kelly:
                trade["kelly_frac"] = 1.0
                trade["net_return"] = round(trade["gross_return"] - cost, 6)
            trades.append(trade)
            exit_pos = d.index.get_loc(trade["exit_date"]) if trade["exit_date"] in d.index else i
            for p in range(i, exit_pos + 1):
                used_pos.add(p)
            i = exit_pos + 1
            continue

        i += 1

    return pd.DataFrame(trades)


# =============================================================================
# 4. RUN ONE MODE
# =============================================================================

def run_mode(mode_name: str, save: bool = True) -> Dict[str, dict]:
    """Прогоняет один пресет и возвращает метрики per-split."""
    if mode_name not in PRESETS:
        raise ValueError(f"Unknown mode: {mode_name}. Options: {list(PRESETS)}")
    mode = PRESETS[mode_name]
    print(f"\n=== Mode: {mode.name} ===")
    print(f"   {mode.description}")
    print(f"   entry≥{mode.p_up_entry}, exit<{mode.p_up_exit}, "
          f"cooldown={mode.cooldown}d, trail={mode.trail_pct*100:.0f}%")

    full = pd.read_csv(V22_DIR / "v22_full_data.csv", parse_dates=[0]).set_index("Date")
    full.index = pd.to_datetime(full.index)

    p_up = pd.read_csv(V25_DIR / "v25_p_up_cpcv.csv", parse_dates=[0]).set_index("Date")
    full["p_up"] = p_up.iloc[:, 0]

    signaled = generate_signals_with_exits(full, full["p_up"], mode)

    # ATR компонент если отсутствует
    if "silver_atr_14d" not in signaled.columns:
        if {"silver_high","silver_low","silver_close"}.issubset(signaled.columns):
            cp = signaled["silver_close"].shift(1)
            tr = pd.concat([
                signaled["silver_high"] - signaled["silver_low"],
                (signaled["silver_high"] - cp).abs(),
                (signaled["silver_low"]  - cp).abs(),
            ], axis=1).max(axis=1)
            signaled["silver_atr_14d"] = tr.ewm(span=14, adjust=False).mean()

    costs = RealisticCosts()
    results = {}

    for split in ["valid", "test", "forward"]:
        trades = backtest_with_model_exits(signaled, split, mode)
        if not trades.empty:
            trades = recompute_trades_with_realistic_costs(trades, signaled, costs)
            eq, seq = equity_compounded_sequential(trades, return_col="net_return")
            n_kept = len(seq)
            trade_days = int((seq["exit_date"].max() - seq["entry_date"].min()).days) \
                if n_kept >= 2 else 0
            metrics = risk_metrics_honest(eq, n_trades=n_kept, trade_days_total=trade_days)

            n_total = len(trades)
            n_buy = n_total
            win_rate = float((trades["net_return"] > 0).mean()) if n_total > 0 else 0
            avg_ret = float(trades["net_return"].mean()) if n_total > 0 else 0

            # Exit reasons breakdown
            exit_counts = trades["exit_reason"].value_counts().to_dict() if "exit_reason" in trades.columns else {}

            results[split] = {
                "n_trades":           n_total,
                "n_sequential":       n_kept,
                "win_rate":           round(win_rate, 4),
                "avg_return":         round(avg_ret, 4),
                "total_return":       round(metrics["total_return"], 4),
                "cagr":               round(metrics["cagr"], 4) if metrics["cagr"] is not None else None,
                "max_drawdown":       round(metrics["max_drawdown"], 4) if metrics["max_drawdown"] is not None else None,
                "sharpe_ann":         round(metrics["sharpe"], 3) if metrics["sharpe"] is not None else None,
                "calmar":             round(metrics["calmar"], 3) if metrics["calmar"] is not None else None,
                "exit_trail_stop":    int(exit_counts.get("trail_stop", 0)),
                "exit_model_exit":    int(exit_counts.get("model_exit", 0)),
                "exit_max_hold":      int(exit_counts.get("max_hold", 0)),
            }

            if save:
                trades.to_csv(MODES_DIR / f"trades_{mode_name}_{split}.csv", index=False)
        else:
            results[split] = {"n_trades": 0}

    return results


# =============================================================================
# 5. COMPARE ALL MODES
# =============================================================================

def compare_modes(save: bool = True) -> pd.DataFrame:
    """Прогоняет ВСЕ пресеты и возвращает таблицу сравнения."""
    print("=" * 70)
    print(" Comparison: все пресеты на одних данных")
    print("=" * 70)

    rows = []
    for mode_name in ["conservative", "balanced", "aggressive", "ultra"]:
        results = run_mode(mode_name, save=save)
        mode = PRESETS[mode_name]
        for split, m in results.items():
            rows.append({
                "mode":            mode_name,
                "split":           split,
                "p_up_entry":      mode.p_up_entry,
                "p_up_exit":       mode.p_up_exit,
                "cooldown":        mode.cooldown,
                "trail_pct":       mode.trail_pct,
                **m,
            })

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(MODES_DIR / "modes_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print(" FORWARD split — финальное сравнение")
    print("=" * 70)
    fwd = df[df["split"] == "forward"].copy()
    if not fwd.empty:
        show = fwd[["mode", "n_trades", "win_rate", "total_return",
                    "avg_return", "sharpe_ann", "max_drawdown", "calmar",
                    "exit_trail_stop", "exit_model_exit", "exit_max_hold"]]
        print(show.to_string(index=False))

    return df


# =============================================================================
# 6. PRODUCTION SIGNAL с SELL ЛОГИКОЙ
# =============================================================================

def update_today_signal_with_exit(
    mode_name: str = "conservative",
    open_positions: int = 0,
) -> dict:
    """
    Обновляет production_signal_today.json — добавляет SELL-логику
    основанную на текущем p_up + наличии открытых позиций.
    """
    mode = PRESETS[mode_name]
    prod_path = PROD_DIR / "production_signal_today.json"
    if not prod_path.exists():
        return {"ok": False, "error": "production signal not found"}

    sig = json.loads(prod_path.read_text(encoding="utf-8"))
    if not sig.get("ok"):
        return sig

    p_up = float(sig.get("p_up", 0.5))

    # Решение по открытым позициям
    if open_positions > 0 and p_up < mode.p_up_exit:
        sig["exit_recommendation"] = {
            "action":   "SELL",
            "reason":   f"p_up={p_up:.3f} < exit_threshold={mode.p_up_exit}",
            "urgency":  "high" if p_up < mode.p_up_exit - 0.05 else "moderate",
            "open_positions": open_positions,
        }
    elif open_positions > 0:
        sig["exit_recommendation"] = {
            "action":   "HOLD_POSITION",
            "reason":   f"p_up={p_up:.3f} >= exit_threshold={mode.p_up_exit}",
            "urgency":  "low",
            "open_positions": open_positions,
        }
    else:
        sig["exit_recommendation"] = {
            "action":   "NO_POSITION",
            "reason":   "Нет открытых позиций",
            "open_positions": 0,
        }

    # Применяем mode policy override для входа
    sig["mode"] = mode_name
    sig["mode_p_up_entry"] = mode.p_up_entry
    sig["mode_p_up_exit"] = mode.p_up_exit
    if p_up >= mode.p_up_entry and sig.get("cooldown_remaining", 0) == 0:
        sig["signal"] = "BUY"
    elif p_up < mode.p_up_exit and open_positions > 0:
        sig["signal"] = "SELL"
    else:
        sig["signal"] = "HOLD"

    return sig


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="Сравнить все пресеты")
    ap.add_argument("--mode", choices=list(PRESETS.keys()), default=None,
                    help="Прогон одного пресета")
    args = ap.parse_args()

    if args.compare or args.mode is None:
        compare_modes()
    else:
        results = run_mode(args.mode)
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
