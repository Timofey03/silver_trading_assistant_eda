"""
silver_assistant_v27_portfolio.py — Multi-asset portfolio backtest

Объединяет silver + gold в один портфель.

Архитектура: ISOLATED BUCKETS
  • Капитал делится между активами по фиксированным весам
  • Silver bucket: weight × capital, компаундится через silver trades
  • Gold bucket:   weight × capital, компаундится через gold trades
  • Общая equity = сумма buckets

Преимущества:
  • Снижение overall volatility (диверсификация)
  • Капитал работает в одном активе даже когда другой idle
  • Простая интерпретация и risk control

Запуск:
  python silver_assistant_v27_portfolio.py                       # default 50/50
  python silver_assistant_v27_portfolio.py --silver-w 0.6        # 60/40
  python silver_assistant_v27_portfolio.py --grid                # перебор весов
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

V25_DIR = Path("baseline_outputs_v25")
V26_DIR = Path("baseline_outputs_v26")
V27_DIR = Path("baseline_outputs_v27")
V27_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. LOAD TRADES
# =============================================================================

def load_trades(asset: str, split: str) -> pd.DataFrame:
    """Загружает трейды актива и стандартизует колонки."""
    if asset == "silver":
        path = V25_DIR / f"v25_{split}_trades.csv"
    elif asset == "gold":
        path = V26_DIR / f"gold_trades_{split}.csv"
    else:
        raise ValueError(f"Unknown asset: {asset}")

    if not path.exists():
        return pd.DataFrame()

    t = pd.read_csv(path)
    if t.empty:
        return t
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"]  = pd.to_datetime(t["exit_date"])
    t["asset"] = asset
    return t


# =============================================================================
# 2. COMBINED EQUITY CURVE
# =============================================================================

def build_portfolio_equity(
    silver_trades: pd.DataFrame,
    gold_trades:   pd.DataFrame,
    silver_weight: float = 0.5,
    gold_weight:   float = 0.5,
    initial: float = 1.0,
) -> pd.DataFrame:
    """
    Строит equity curve портфеля.

    Подход: каждый bucket компаундится независимо. Общая equity = сумма buckets.

    Возвращает DataFrame с колонками:
      date, silver_equity, gold_equity, total_equity, drawdown
    """
    # Каждый bucket = independent compound chain
    s_eq = initial * silver_weight
    g_eq = initial * gold_weight

    silver_eq_path: List[Tuple[pd.Timestamp, float]] = [(pd.Timestamp("2025-01-01"), s_eq)]
    gold_eq_path:   List[Tuple[pd.Timestamp, float]] = [(pd.Timestamp("2025-01-01"), g_eq)]

    # Silver compound
    if not silver_trades.empty:
        st = silver_trades.sort_values("exit_date").reset_index(drop=True)
        for _, row in st.iterrows():
            s_eq *= (1.0 + float(row["net_return"]))
            silver_eq_path.append((row["exit_date"], s_eq))

    # Gold compound
    if not gold_trades.empty:
        gt = gold_trades.sort_values("exit_date").reset_index(drop=True)
        for _, row in gt.iterrows():
            g_eq *= (1.0 + float(row["net_return"]))
            gold_eq_path.append((row["exit_date"], g_eq))

    # Merge timelines
    all_dates = sorted(set([d for d, _ in silver_eq_path] + [d for d, _ in gold_eq_path]))

    # Forward-fill каждый bucket
    s_df = pd.Series(dict(silver_eq_path)).sort_index()
    g_df = pd.Series(dict(gold_eq_path)).sort_index()

    full_idx = pd.DatetimeIndex(all_dates)
    s_series = s_df.reindex(full_idx, method="ffill").fillna(initial * silver_weight)
    g_series = g_df.reindex(full_idx, method="ffill").fillna(initial * gold_weight)

    portfolio = pd.DataFrame({
        "silver_equity":  s_series,
        "gold_equity":    g_series,
        "total_equity":   s_series + g_series,
    })
    portfolio["drawdown"] = (
        portfolio["total_equity"] / portfolio["total_equity"].cummax() - 1
    )
    return portfolio


# =============================================================================
# 3. PORTFOLIO METRICS
# =============================================================================

def portfolio_metrics(
    eq_df: pd.DataFrame, silver_trades: pd.DataFrame, gold_trades: pd.DataFrame,
) -> dict:
    """Sharpe/CAGR/MaxDD на combined equity."""
    if eq_df.empty:
        return {}

    total = eq_df["total_equity"]
    initial = total.iloc[0]
    final = total.iloc[-1]
    total_return = float(final / initial - 1.0)

    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    years = max(n_days / 365.25, 0.1)
    cagr = float(final ** (1.0 / years) - 1.0) / initial ** (1.0 / years) if initial > 0 else 0
    # Корректно: CAGR = (final / initial)^(1/years) - 1
    cagr = float((final / initial) ** (1.0 / years) - 1.0) if initial > 0 else 0

    max_dd = float(eq_df["drawdown"].min())

    # Combined trades для Sharpe
    all_t = pd.concat([silver_trades, gold_trades], ignore_index=True)
    if not all_t.empty and len(all_t) >= 3:
        rets = all_t["net_return"].astype(float).values
        if rets.std() > 0:
            n_per_year = len(all_t) / years
            sharpe = float(rets.mean() / rets.std() * np.sqrt(n_per_year))
        else:
            sharpe = None
    else:
        sharpe = None

    calmar = (cagr / abs(max_dd)) if (max_dd < -0.001 and cagr) else None

    return {
        "total_return":   round(total_return, 4),
        "cagr":           round(cagr, 4),
        "max_drawdown":   round(max_dd, 4),
        "sharpe_ann":     round(sharpe, 3) if sharpe is not None else None,
        "calmar":         round(calmar, 3) if calmar is not None else None,
        "n_silver":       len(silver_trades),
        "n_gold":         len(gold_trades),
        "n_total":        len(all_t),
        "silver_final":   round(float(eq_df["silver_equity"].iloc[-1]), 4),
        "gold_final":     round(float(eq_df["gold_equity"].iloc[-1]), 4),
        "duration_days":  int(n_days),
    }


# =============================================================================
# 4. RUN
# =============================================================================

def run_portfolio(silver_w: float = 0.5, gold_w: float = 0.5,
                   save: bool = True) -> dict:
    """Прогоняет combined backtest на всех splits."""
    results = {}

    for split in ["valid", "test", "forward"]:
        silver_t = load_trades("silver", split)
        gold_t   = load_trades("gold", split)

        eq = build_portfolio_equity(silver_t, gold_t, silver_w, gold_w)
        m = portfolio_metrics(eq, silver_t, gold_t)
        m["silver_weight"] = silver_w
        m["gold_weight"]   = gold_w
        m["split"] = split
        results[split] = m

        if save:
            eq.to_csv(V27_DIR / f"portfolio_equity_{split}_s{int(silver_w*100)}_g{int(gold_w*100)}.csv")

    return results


def print_results(results: dict, title: str = "") -> None:
    print(f"\n=== {title} ===" if title else "\n=== RESULTS ===")
    for split, m in results.items():
        sw = m.get('silver_weight', 0.5)
        gw = m.get('gold_weight', 0.5)
        print(f"  {split:8s} (s{int(sw*100)}/g{int(gw*100)}):  "
              f"return={m.get('total_return', 0)*100:+.1f}%  "
              f"CAGR={m.get('cagr', 0)*100:+.1f}%  "
              f"DD={m.get('max_drawdown', 0)*100:.1f}%  "
              f"Sharpe={m.get('sharpe_ann')}  "
              f"n=({m.get('n_silver', 0)}+{m.get('n_gold', 0)})")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-w", type=float, default=0.5)
    ap.add_argument("--gold-w",   type=float, default=0.5)
    ap.add_argument("--grid",     action="store_true",
                    help="Перебрать разные веса")
    args = ap.parse_args()

    print("=" * 70)
    print(" Multi-asset portfolio backtest")
    print("=" * 70)

    # Stand-alone results для сравнения
    print("\n=== STANDALONE для сравнения ===")
    for asset in ["silver", "gold"]:
        for split in ["valid", "test", "forward"]:
            trades = load_trades(asset, split)
            if trades.empty:
                continue
            eq_initial = 1.0
            for _, row in trades.sort_values("exit_date").iterrows():
                eq_initial *= (1.0 + float(row["net_return"]))
            ret_pct = (eq_initial - 1.0) * 100
            print(f"  {asset:6s} {split:8s}: trades={len(trades):3d}  "
                  f"compound_return={ret_pct:+.1f}%")

    # Portfolio combined
    if args.grid:
        print("\n=== GRID SEARCH weights ===")
        weights = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                   (0.8, 0.2), (1.0, 0.0), (0.0, 1.0)]
        all_results = []
        for sw, gw in weights:
            r = run_portfolio(sw, gw, save=False)
            fwd = r["forward"]
            all_results.append({
                "silver_w": sw, "gold_w": gw,
                "fwd_return":   fwd["total_return"],
                "fwd_cagr":     fwd["cagr"],
                "fwd_dd":       fwd["max_drawdown"],
                "fwd_sharpe":   fwd["sharpe_ann"],
                "fwd_calmar":   fwd["calmar"],
            })
        df = pd.DataFrame(all_results).sort_values("fwd_return", ascending=False)
        for c in ["fwd_return", "fwd_cagr", "fwd_dd"]:
            df[c] = df[c].apply(lambda x: f"{x*100:+.1f}%")
        print(df.to_string(index=False))
        df.to_csv(V27_DIR / "weights_grid.csv", index=False)
    else:
        results = run_portfolio(args.silver_w, args.gold_w)
        print_results(results,
                       title=f"PORTFOLIO {int(args.silver_w*100)}% silver / {int(args.gold_w*100)}% gold")

        # JSON для UI
        (V27_DIR / "portfolio_summary.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # Сравнение vs silver-only
        silver_only = run_portfolio(1.0, 0.0, save=False)
        print("\n=== SILVER-ONLY vs PORTFOLIO ===")
        for split in ["valid", "test", "forward"]:
            so = silver_only[split]
            po = results[split]
            print(f"  {split:8s}: silver-only={so['total_return']*100:+.1f}% (DD {so['max_drawdown']*100:.1f}%)  "
                  f"vs portfolio={po['total_return']*100:+.1f}% (DD {po['max_drawdown']*100:.1f}%)")


if __name__ == "__main__":
    main()
