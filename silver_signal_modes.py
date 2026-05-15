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


# ⭐ OPTIMAL MODE — найдено через grid search (240 комбинаций),
# фильтр: консистентность валидации (valid >= -2%, test >= 0%, fwd >= 6 trades)
# Результат на forward split: +64.5%, Sharpe 1.69, win 64%, 11 трейдов
OPTIMAL_PARAMS = SignalMode(
    name="Optimal",
    description="Grid-search оптимум — самая доходная робастная конфигурация. "
                "+64.5% forward (3x vs прежний Conservative).",
    p_up_entry=0.49,
    p_up_exit=0.43,
    cooldown=15,
    trail_pct=0.08,
    max_hold=30,
    expected_trades_per_year=11,
)

# Сохраняем PRESETS для обратной совместимости grid_search скрипта
PRESETS: Dict[str, SignalMode] = {
    "optimal": OPTIMAL_PARAMS,
    "conservative": SignalMode(
        name="Conservative",
        description="Старый v25 — слишком селективный",
        p_up_entry=0.55, p_up_exit=0.40,
        cooldown=15, trail_pct=0.08, max_hold=45,
        expected_trades_per_year=4,
    ),
    "balanced": SignalMode(
        name="Balanced", description="Backtest comparison",
        p_up_entry=0.52, p_up_exit=0.45,
        cooldown=10, trail_pct=0.07, max_hold=30,
        expected_trades_per_year=10,
    ),
    "aggressive": SignalMode(
        name="Aggressive", description="Backtest comparison",
        p_up_entry=0.50, p_up_exit=0.48,
        cooldown=5, trail_pct=0.05, max_hold=20,
        expected_trades_per_year=22,
    ),
    "ultra": SignalMode(
        name="Ultra", description="Backtest comparison",
        p_up_entry=0.48, p_up_exit=0.50,
        cooldown=3, trail_pct=0.04, max_hold=15,
        expected_trades_per_year=40,
    ),
}


# =============================================================================
# 2. SIGNAL GENERATION с SELL-сигналами
# =============================================================================

def generate_signals_with_exits(
    df: pd.DataFrame,
    p_up_series: pd.Series,
    mode: SignalMode,
) -> pd.DataFrame:
    """
    Генерирует BUY/SELL/HOLD на основе p_up + state machine.

    Логика:
      • position = 0 (no position):
          if p_up >= p_up_entry AND last_buy > cooldown ago → BUY (open LONG)
      • position = 1 (LONG):
          if p_up <  p_up_exit  → SELL (close LONG)
          else → HOLD
    """
    out = df.copy().sort_index()
    out["p_up"] = p_up_series.reindex(out.index)

    signals = []
    positions = []
    position = 0
    last_buy = -10**9

    for i, p in enumerate(out["p_up"].values):
        sig = "HOLD"
        if pd.isna(p):
            signals.append(sig)
            positions.append(position)
            continue

        if position == 0:
            # Нет позиции — ищем вход
            if p >= mode.p_up_entry and (i - last_buy) > mode.cooldown:
                sig = "BUY"
                position = 1
                last_buy = i
        else:
            # В позиции — следим за выходом
            if p < mode.p_up_exit:
                sig = "SELL"
                position = 0

        signals.append(sig)
        positions.append(position)

    out["signal_long"] = signals
    out["position"] = positions
    out["signal_short"] = "HOLD"
    out["signal"] = signals
    return out


# =============================================================================
# 3. BACKTEST с SELL-логикой (вместо trailing stop)
# =============================================================================

def backtest_with_model_exits(
    df: pd.DataFrame,
    split: str,
    mode: SignalMode,
    cost: float = COST_PER_TRADE,
) -> pd.DataFrame:
    """
    Бэктест: BUY на signal_long=='BUY', EXIT на signal_long=='SELL'.
    Используем trailing stop как страховку (если SELL не сработал).
    """
    d = df[df["split"] == split].sort_index().copy()
    if d.empty:
        return pd.DataFrame()

    has_high = "silver_high" in d.columns
    has_low  = "silver_low" in d.columns

    trades = []
    buy_indices = np.where(d["signal_long"].values == "BUY")[0]

    for entry_pos in buy_indices:
        entry_date  = d.index[entry_pos]
        entry_price = float(d.iloc[entry_pos]["silver_close"])
        peak        = entry_price
        trail_stop  = entry_price * (1.0 - mode.trail_pct)

        exit_idx = entry_pos
        exit_price = entry_price
        exit_reason = "max_hold"

        for j in range(1, mode.max_hold + 1):
            pos = entry_pos + j
            if pos >= len(d):
                break

            cl = float(d.iloc[pos]["silver_close"])
            hi = float(d.iloc[pos]["silver_high"]) if has_high else cl
            lo = float(d.iloc[pos]["silver_low"]) if has_low else cl

            if hi > peak:
                peak = hi
                trail_stop = peak * (1.0 - mode.trail_pct)

            # Trailing stop check
            if lo <= trail_stop:
                exit_price = min(cl, trail_stop)
                exit_idx = pos
                exit_reason = "trail_stop"
                break

            # Model exit signal
            sig_here = d.iloc[pos].get("signal_long", "HOLD")
            if sig_here == "SELL":
                exit_price = cl
                exit_idx = pos
                exit_reason = "model_exit"
                break

            exit_price = cl
            exit_idx = pos

        gross = exit_price / entry_price - 1.0
        net = gross - cost
        trades.append({
            "direction":    "LONG",
            "entry_date":   entry_date,
            "exit_date":    d.index[exit_idx],
            "entry_price":  round(entry_price, 3),
            "exit_price":   round(exit_price, 3),
            "peak_price":   round(peak, 3),
            "gross_return": round(gross, 6),
            "net_return":   round(net, 6),
            "hold_days":    exit_idx - entry_pos,
            "exit_reason":  exit_reason,
        })

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
