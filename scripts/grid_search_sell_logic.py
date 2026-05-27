"""
scripts/grid_search_sell_logic.py — сравниваем все варианты SELL логики.

Текущая конфигурация: trail=0.20, max_hold=60, exit_thr=0.30, smoothed>=0.85.

Варианты SELL:
A. trail only      — только trailing stop
B. max_hold only   — только лимит дней
C. model_flip only — только p_up<0.30
D. profit_take only — только TP при p<0.50 и pnl>10%
AB. trail + max_hold
ABC. trail + max_hold + model_flip (TEKUSCHEE)
ABCD. trail + max_hold + model_flip + profit_take (ALL)
PT_alt. ABC + profit-take при pnl>15%, p<0.40 (более строгий)
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def simulate_with_profit_take(
    preds, prices, entry_th=0.70, exit_th=0.30, trail_pct=0.20,
    max_hold=60, cooldown=10, smooth=3,
    profit_take_pnl=0.0, profit_take_pup=1.0,   # 0/1 disabled
    require_smoothed_entry=0.85,
):
    """Кастомный симулятор с profit-take. Возвращает list of trades + metrics."""
    # Smooth
    preds = preds.copy()
    p_smooth = preds["p_1"].rolling(smooth, min_periods=1).mean()

    # Strong-signal filter
    p_smooth = p_smooth.where(p_smooth >= require_smoothed_entry, 0.0)
    preds["p_1"] = p_smooth

    common = preds.index.intersection(prices.index)
    p_up = preds["p_1"].reindex(common).values
    p_raw = p_smooth.reindex(common).values  # for exits
    cl = prices["close"].reindex(common).values
    hi = prices["high"].reindex(common).values
    lw = prices["low"].reindex(common).values
    dates = list(common)
    n = len(dates)

    trades = []
    state = "FLAT"
    entry_idx = None
    entry_price = None
    peak = None
    cooldown_until = -1

    for i in range(n):
        c, h, low, p = cl[i], hi[i], lw[i], p_up[i]
        if not np.isfinite(c):
            continue

        if state == "LONG":
            if h > peak: peak = h
            hold_days = i - entry_idx
            trail_level = peak * (1 - trail_pct)
            pnl = (c - entry_price) / entry_price
            exit_reason = None
            exit_price = None

            # 1. trail
            if low <= trail_level:
                exit_reason = "trail"; exit_price = trail_level
            # 2. max_hold
            elif hold_days >= max_hold:
                exit_reason = "max_hold"; exit_price = c
            # 3. model_flip
            elif np.isfinite(p) and p < exit_th:
                exit_reason = "model_exit"; exit_price = c
            # 4. profit_take (custom)
            elif profit_take_pnl > 0 and pnl >= profit_take_pnl and np.isfinite(p_raw[i]) and p_raw[i] < profit_take_pup:
                exit_reason = "profit_take"; exit_price = c

            if exit_reason:
                gross = exit_price / entry_price - 1
                cost = 2 * (0.0005 + 0.0005)
                net = gross - cost
                trades.append({
                    "entry_date": dates[entry_idx], "exit_date": dates[i],
                    "entry_price": entry_price, "exit_price": exit_price,
                    "peak_price": peak, "hold_days": hold_days,
                    "gross_return": gross, "net_return": net, "exit_reason": exit_reason,
                })
                state = "FLAT"
                cooldown_until = i + cooldown
                entry_idx = entry_price = peak = None

        elif state == "FLAT":
            if i < cooldown_until: continue
            if not np.isfinite(p): continue
            if p >= entry_th:
                state = "LONG"
                entry_idx = i; entry_price = c; peak = h

    return trades


def metrics_of(trades):
    if not trades:
        return {"n":0, "ret":0, "sharpe":0, "dd":0, "win":0}
    nr = np.array([t["net_return"] for t in trades])
    total = float(np.prod(1 + nr) - 1)
    first = min(t["entry_date"] for t in trades)
    last = max(t["exit_date"] for t in trades)
    yrs = max((last - first).days / 365.25, 0.01)
    tpy = len(nr) / yrs
    sr_t = nr.mean() / nr.std() if nr.std() > 0 else 0
    sr = sr_t * np.sqrt(tpy) if tpy > 0 else 0
    eq = (1 + pd.Series(nr)).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    win = float((nr > 0).sum() / len(nr))
    return {"n":len(nr), "ret":total*100, "sharpe":sr, "dd":dd*100, "win":win*100}


def main():
    from app.multi_asset.metal_loader import load_metals
    preds = pd.read_parquet("baseline_outputs_multiasset/e3b_adaptive/predictions.parquet")
    silver = load_metals()["silver"]

    HUGE = 999
    SMALL = 0.0

    variants = [
        ("A:trail",         dict(trail_pct=0.20, max_hold=HUGE, exit_th=SMALL,  profit_take_pnl=0.0)),
        ("B:max_hold",      dict(trail_pct=1.0,  max_hold=60,   exit_th=SMALL,  profit_take_pnl=0.0)),
        ("C:model_flip",    dict(trail_pct=1.0,  max_hold=HUGE, exit_th=0.30,   profit_take_pnl=0.0)),
        ("D:profit_take",   dict(trail_pct=1.0,  max_hold=HUGE, exit_th=SMALL,  profit_take_pnl=0.10, profit_take_pup=0.50)),
        ("A+B",             dict(trail_pct=0.20, max_hold=60,   exit_th=SMALL,  profit_take_pnl=0.0)),
        ("A+B+C [TEKUSCHEE]", dict(trail_pct=0.20, max_hold=60, exit_th=0.30,   profit_take_pnl=0.0)),
        ("A+B+C+D",         dict(trail_pct=0.20, max_hold=60,   exit_th=0.30,   profit_take_pnl=0.10, profit_take_pup=0.50)),
        ("ABC+strict_PT",   dict(trail_pct=0.20, max_hold=60,   exit_th=0.30,   profit_take_pnl=0.15, profit_take_pup=0.40)),
        ("ABC+lenient_PT",  dict(trail_pct=0.20, max_hold=60,   exit_th=0.30,   profit_take_pnl=0.05, profit_take_pup=0.60)),
    ]

    rows = []
    for name, params in variants:
        trades = simulate_with_profit_take(preds, silver, **params)
        m = metrics_of(trades)
        rows.append({"variant":name, **m})

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print()
    print("=" * 110)
    print(" SELL LOGIC GRID SEARCH (на финальных predictions, strong_signal>=0.85)")
    print("=" * 110)
    print(df.to_string(index=False))
    df.to_csv("baseline_outputs_multiasset/sell_logic_grid.csv", index=False)


if __name__ == "__main__":
    main()
