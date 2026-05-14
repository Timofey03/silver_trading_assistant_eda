"""
Silver Trading Assistant v24 — Gate Overlays + v23 Honest Math

Что нового vs v22:
1) Liquidity gate            — блок сигналов в low-volume дни
2) VIX risk-off gate          — блок LONG когда VIX > 25 и растёт
3) GSR extreme gate           — блок LONG/SHORT при экстремуме gold_silver_ratio z-score
4) Drawdown kill-switch       — блок новых сделок при equity drawdown > 20%
5) Регресс через v23 honest math (compound, single-position, realistic costs)
6) Force-include extended features в model (для будущей перетренировки CPCV)

Пайплайн:
   v22_base_decisions.csv (UP/SHORT сигналы) ↓
   Liquidity gate ↓
   VIX gate ↓
   GSR gate ↓
   v23 backtest (compound) ↓
   Сравнение vs v22

Запуск:
  python silver_assistant_v24_gates.py                    # все гейты
  python silver_assistant_v24_gates.py --gates liquidity  # отдельный гейт
  python silver_assistant_v24_gates.py --no-vix-gate      # выключить VIX
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# v23 honest math
from silver_assistant_v23_honest import (
    equity_compounded_sequential,
    risk_metrics_honest,
    true_buy_and_hold,
    bootstrap_ci_metrics,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    sharpe_stats,
    recompute_trades_with_realistic_costs,
    RealisticCosts,
)

# v22 backtest engine
from silver_assistant_v22_risk_aware import (
    backtest_strategy_independent,
)
from silver_assistant_v19_trailing import (
    TRAIL_PCT_DEFAULT, MAX_HOLD_DEFAULT, COST_PER_TRADE,
)

V22_DIR = Path("baseline_outputs_v22")
V24_DIR = Path("baseline_outputs_v24")
V24_DIR.mkdir(exist_ok=True)


# ===========================================================================
# 1. GATES
# ===========================================================================

@dataclass
class GateConfig:
    use_liquidity:     bool  = True
    use_vix:           bool  = True
    use_gsr:           bool  = True
    use_drawdown_kill: bool  = True

    # Liquidity
    liquidity_pct_of_median:    float = 0.5
    liquidity_median_window:    int   = 60

    # VIX
    vix_block_level:    float = 25.0      # VIX выше — risk-off
    vix_chg_5d_block:   float = 2.0       # И растёт на 2+ пункта за 5 дней

    # GSR
    gsr_z60_extreme_high: float = 2.0     # gold переоценен → лучше не шортить
    gsr_z60_extreme_low:  float = -2.0    # silver переоценен → лучше не лонг

    # Drawdown kill
    dd_kill_threshold:  float = -0.20     # стоп при -20% equity


def apply_liquidity_gate(
    df: pd.DataFrame, cfg: GateConfig,
) -> pd.DataFrame:
    """
    Блокирует BUY/SHORT в дни с volume < threshold от 60-day median.
    Сохраняет колонку gate_liquidity (1=blocked, 0=ok).
    """
    out = df.copy()
    if "silver_volume" not in out.columns:
        out["gate_liquidity"] = 0
        return out
    vol_med = out["silver_volume"].rolling(
        cfg.liquidity_median_window, min_periods=20,
    ).median()
    illiquid = (out["silver_volume"] < cfg.liquidity_pct_of_median * vol_med) & vol_med.notna()
    out["gate_liquidity"] = illiquid.astype(int)
    if "signal_long" in out.columns:
        out.loc[illiquid & (out["signal_long"] == "BUY"), "signal_long"] = "HOLD"
    if "signal_short" in out.columns:
        out.loc[illiquid & (out["signal_short"] == "SHORT"), "signal_short"] = "HOLD"
    return out


def apply_vix_gate(df: pd.DataFrame, cfg: GateConfig) -> pd.DataFrame:
    """
    Risk-off фильтр: блок LONG когда VIX > vix_block_level И VIX вырос
    на vix_chg_5d_block+ за последние 5 дней.

    SHORT в этих условиях НЕ блокируется (даже наоборот — risk-off часто
    сопровождается коррекцией металлов).
    """
    out = df.copy()
    if "vix_close" not in out.columns and "vix_level" not in out.columns:
        out["gate_vix"] = 0
        return out
    vix = out["vix_close"] if "vix_close" in out.columns else out["vix_level"]
    vix_chg5 = vix.diff(5)
    risk_off = (vix > cfg.vix_block_level) & (vix_chg5 > cfg.vix_chg_5d_block)
    out["gate_vix"] = risk_off.astype(int)
    if "signal_long" in out.columns:
        out.loc[risk_off & (out["signal_long"] == "BUY"), "signal_long"] = "HOLD"
    return out


def apply_gsr_gate(df: pd.DataFrame, cfg: GateConfig) -> pd.DataFrame:
    """
    Gold/Silver ratio mean-reversion logic:
      • gsr_z60 > +2σ  → gold переоценен → silver скоро догонит → УСИЛИВАЕМ LONG
                          (т.е. блокируем SHORT)
      • gsr_z60 < -2σ  → silver переоценен → может откатиться → блок LONG
    """
    out = df.copy()
    col_candidates = ["gold_silver_ratio_z60", "gsr_zscore_60d"]
    z60 = None
    for c in col_candidates:
        if c in out.columns:
            z60 = out[c]
            break
    if z60 is None:
        out["gate_gsr"] = 0
        return out

    block_long  = z60 < cfg.gsr_z60_extreme_low
    block_short = z60 > cfg.gsr_z60_extreme_high
    out["gate_gsr_block_long"]  = block_long.astype(int)
    out["gate_gsr_block_short"] = block_short.astype(int)
    if "signal_long" in out.columns:
        out.loc[block_long & (out["signal_long"] == "BUY"), "signal_long"] = "HOLD"
    if "signal_short" in out.columns:
        out.loc[block_short & (out["signal_short"] == "SHORT"), "signal_short"] = "HOLD"
    return out


def apply_drawdown_killswitch(
    trades: pd.DataFrame, cfg: GateConfig,
) -> pd.DataFrame:
    """
    Удаляет сделки, которые открылись бы во время drawdown > dd_kill_threshold.
    Работает на trades-уровне (после backtest), а не на сигналах.
    """
    if trades.empty:
        return trades.copy()
    t = trades.copy().sort_values("entry_date").reset_index(drop=True)
    rets = t["net_return"].astype(float).values
    eq = np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0
    # dd[i+1] — drawdown сразу после i-й сделки
    keep = []
    killed = False
    for i in range(len(t)):
        if killed:
            continue
        keep.append(i)
        if dd[i + 1] <= cfg.dd_kill_threshold:
            killed = True
    return t.iloc[keep].reset_index(drop=True)


# ===========================================================================
# 2. RETRAIN PIPELINE (lite)
# ===========================================================================

def _load_v22_base_decisions() -> pd.DataFrame:
    p = V22_DIR / "v22_base_decisions.csv"
    if not p.exists():
        raise FileNotFoundError(f"Нет {p}. Запустите v22 сначала.")
    df = pd.read_csv(p, parse_dates=[0])
    df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    return df


def _load_full_data() -> pd.DataFrame:
    p = V22_DIR / "v22_full_data.csv"
    df = pd.read_csv(p, parse_dates=[0])
    df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    return df


def build_gated_decisions(cfg: GateConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Берёт v22_base_decisions, накладывает гейты, возвращает (gated_df, stats_df).
    """
    base = _load_v22_base_decisions()
    full = _load_full_data()

    keep_cols = ["silver_volume", "vix_close", "vix_level",
                 "gold_silver_ratio_z60", "gsr_zscore_60d"]
    for c in keep_cols:
        if c in full.columns and c not in base.columns:
            base[c] = full[c].reindex(base.index)

    initial_long  = (base.get("signal_long",  pd.Series()) == "BUY").sum()
    initial_short = (base.get("signal_short", pd.Series()) == "SHORT").sum()

    df = base.copy()
    blocks = {}

    if cfg.use_liquidity:
        df = apply_liquidity_gate(df, cfg)
        blocks["liquidity_blocked"] = int(df["gate_liquidity"].sum())

    if cfg.use_vix:
        df = apply_vix_gate(df, cfg)
        blocks["vix_blocked"] = int(df["gate_vix"].sum())

    if cfg.use_gsr:
        df = apply_gsr_gate(df, cfg)
        blocks["gsr_block_long"]  = int(df["gate_gsr_block_long"].sum())
        blocks["gsr_block_short"] = int(df["gate_gsr_block_short"].sum())

    final_long  = (df.get("signal_long",  pd.Series()) == "BUY").sum()
    final_short = (df.get("signal_short", pd.Series()) == "SHORT").sum()

    stats = pd.DataFrame([{
        "initial_long":   initial_long,
        "initial_short":  initial_short,
        "final_long":     int(final_long),
        "final_short":    int(final_short),
        "long_blocked":   int(initial_long - final_long),
        "short_blocked":  int(initial_short - final_short),
        **blocks,
    }])
    return df, stats


# ===========================================================================
# 3. HONEST BACKTEST
# ===========================================================================

def run_v24_backtest(
    gated_df: pd.DataFrame,
    cfg: GateConfig,
    full_df: pd.DataFrame,
    apply_realistic_costs: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Бэктест v24 с гейтами + v23 honest math + realistic costs.
    Возвращает {variant: DataFrame_results}.
    """
    print("\n" + "=" * 70)
    print(" v24 BACKTEST — gated signals + v23 honest math")
    print("=" * 70)

    splits = ["valid", "test", "forward"]

    rows_pnl:  List[dict] = []
    rows_risk: List[dict] = []
    rows_dsr:  List[dict] = []
    rows_boot: List[dict] = []
    rows_attr: List[dict] = []

    costs_model = RealisticCosts() if apply_realistic_costs else None

    # Подготовка ATR
    if "silver_atr_14d" not in full_df.columns:
        if {"silver_high", "silver_low", "silver_close"}.issubset(full_df.columns):
            close_prev = full_df["silver_close"].shift(1)
            tr = pd.concat([
                full_df["silver_high"] - full_df["silver_low"],
                (full_df["silver_high"] - close_prev).abs(),
                (full_df["silver_low"]  - close_prev).abs(),
            ], axis=1).max(axis=1)
            full_df["silver_atr_14d"] = tr.ewm(span=14, adjust=False).mean()
        else:
            full_df["silver_atr_14d"] = full_df["silver_close"] * 0.015

    # bake silver_close, OHLC из full в gated_df (нужны для backtest_strategy_independent)
    needed = ["silver_open", "silver_high", "silver_low", "silver_close",
              "silver_atr_14d", "split"]
    for c in needed:
        if c not in gated_df.columns and c in full_df.columns:
            gated_df[c] = full_df[c].reindex(gated_df.index)

    for split in splits:
        # --- backtest gated ---
        trades_g = backtest_strategy_independent(
            gated_df, split,
            trail_pct_long=TRAIL_PCT_DEFAULT,  max_hold_long=MAX_HOLD_DEFAULT,
            trail_pct_short=TRAIL_PCT_DEFAULT, max_hold_short=MAX_HOLD_DEFAULT,
            cost=COST_PER_TRADE,
        )

        if costs_model is not None and not trades_g.empty:
            trades_g = recompute_trades_with_realistic_costs(trades_g, full_df, costs_model)

        # --- drawdown kill ---
        if cfg.use_drawdown_kill and not trades_g.empty:
            t_before = len(trades_g)
            trades_g = apply_drawdown_killswitch(trades_g, cfg)
            t_killed = t_before - len(trades_g)
        else:
            t_killed = 0

        trades_g.to_csv(V24_DIR / f"v24_{split}_trades.csv", index=False)

        # --- honest equity ---
        eq, seq = equity_compounded_sequential(trades_g, return_col="net_return")
        n_kept = len(seq)
        trade_days = int(
            (seq["exit_date"].max() - seq["entry_date"].min()).days,
        ) if n_kept >= 2 else 0
        rm = risk_metrics_honest(eq, n_trades=n_kept, trade_days_total=trade_days)

        # --- baseline (v22 без гейтов, тот же honest math для apples-to-apples) ---
        v22_trades_path = V22_DIR / f"v22_base_{split}_trades.csv"
        if v22_trades_path.exists():
            v22_trades = pd.read_csv(v22_trades_path)
            v22_trades["entry_date"] = pd.to_datetime(v22_trades["entry_date"])
            v22_trades["exit_date"]  = pd.to_datetime(v22_trades["exit_date"])
            if costs_model is not None and not v22_trades.empty:
                v22_trades = recompute_trades_with_realistic_costs(v22_trades, full_df, costs_model)
            eq22, seq22 = equity_compounded_sequential(v22_trades, return_col="net_return")
            v22_total = float(eq22[-1] - 1.0)
        else:
            v22_total = float("nan")

        # --- BnH ---
        d = full_df[full_df["split"] == split]
        if len(d) >= 2:
            true_bnh = float(d["silver_close"].iloc[-1] / d["silver_close"].iloc[0] - 1.0)
        else:
            true_bnh = float("nan")

        # --- DSR / PSR ---
        seq_rets = seq["net_return"].astype(float).values if not seq.empty else np.array([])
        sharpe_per, skew, kurt = sharpe_stats(seq_rets) if len(seq_rets) >= 4 else (float("nan"), 0, 3)
        psr = probabilistic_sharpe_ratio(sharpe_per, len(seq_rets), skew, kurt, 0.0) if not np.isnan(sharpe_per) else float("nan")
        dsr = deflated_sharpe_ratio(
            sharpe_per, len(seq_rets), n_trials=10,
            sharpe_variance=0.5 * (sharpe_per**2 + 1e-6),
            skew=skew, kurt=kurt,
        ) if not np.isnan(sharpe_per) else float("nan")

        # --- bootstrap CI ---
        ci = bootstrap_ci_metrics(seq_rets, n_boot=2000, block_len=5.0) if len(seq_rets) >= 3 else {}

        rows_pnl.append({
            "variant":               "v24_gated",
            "split":                 split,
            "n_trades_v22":          len(v22_trades) if v22_trades_path.exists() else 0,
            "n_trades_v24_total":    len(trades_g) + t_killed,
            "n_killed_dd":           t_killed,
            "n_trades_kept":         len(trades_g),
            "n_sequential":          n_kept,
            "v22_honest_total":      round(v22_total, 4) if not np.isnan(v22_total) else None,
            "v24_honest_total":      round(rm["total_return"], 4),
            "improvement_pp":        round(rm["total_return"] - v22_total, 4) if not np.isnan(v22_total) else None,
            "true_bnh":              round(true_bnh, 4) if not np.isnan(true_bnh) else None,
            "vs_bnh":                round(rm["total_return"] - true_bnh, 4) if not np.isnan(true_bnh) else None,
            "cagr":                  round(rm["cagr"], 4) if rm["cagr"] is not None else None,
            "max_dd":                round(rm["max_drawdown"], 4) if rm["max_drawdown"] is not None else None,
            "sharpe_ann":            round(rm["sharpe"], 3) if rm["sharpe"] is not None else None,
            "calmar":                round(rm["calmar"], 3) if rm["calmar"] is not None else None,
        })
        rows_risk.append({"variant": "v24_gated", "split": split, **rm})
        rows_dsr.append({
            "variant": "v24_gated", "split": split, "n_obs": len(seq_rets),
            "sharpe_per_trade": round(sharpe_per, 4) if not np.isnan(sharpe_per) else None,
            "skew": round(skew, 3), "kurt": round(kurt, 3),
            "psr":  round(psr, 4) if not np.isnan(psr) else None,
            "dsr":  round(dsr, 4) if not np.isnan(dsr) else None,
        })
        if ci:
            rows_boot.append({
                "variant": "v24_gated", "split": split, "n_obs": len(seq_rets),
                "tr_lower":  round(ci["total_return"]["lower"],  4),
                "tr_median": round(ci["total_return"]["median"], 4),
                "tr_upper":  round(ci["total_return"]["upper"],  4),
                "shr_lower": round(ci["sharpe"]["lower"],  3),
                "shr_median":round(ci["sharpe"]["median"], 3),
                "shr_upper": round(ci["sharpe"]["upper"],  3),
                "mdd_lower": round(ci["max_drawdown"]["lower"],  4),
                "mdd_median":round(ci["max_drawdown"]["median"], 4),
                "mdd_upper": round(ci["max_drawdown"]["upper"],  4),
            })

    pnl_df   = pd.DataFrame(rows_pnl)
    risk_df  = pd.DataFrame(rows_risk)
    dsr_df   = pd.DataFrame(rows_dsr)
    boot_df  = pd.DataFrame(rows_boot)

    pnl_df.to_csv (V24_DIR / "v24_pnl_summary.csv",  index=False)
    risk_df.to_csv(V24_DIR / "v24_risk_metrics.csv", index=False)
    dsr_df.to_csv (V24_DIR / "v24_dsr_psr.csv",       index=False)
    boot_df.to_csv(V24_DIR / "v24_bootstrap_ci.csv",  index=False)

    print("\n=== v24 (gated) vs v22_base (оба honest math + realistic costs) ===")
    print(pnl_df.to_string(index=False))
    print("\n=== v24 bootstrap 95% CI ===")
    if not boot_df.empty:
        print(boot_df.to_string(index=False))
    print("\n=== v24 DSR / PSR ===")
    print(dsr_df.to_string(index=False))

    return {
        "pnl":  pnl_df,
        "risk": risk_df,
        "dsr":  dsr_df,
        "boot": boot_df,
    }


# ===========================================================================
# 4. MAIN
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="v24 gate overlays + honest math")
    ap.add_argument("--no-liquidity", action="store_true")
    ap.add_argument("--no-vix-gate",  action="store_true")
    ap.add_argument("--no-gsr",       action="store_true")
    ap.add_argument("--no-dd-kill",   action="store_true")
    ap.add_argument("--no-realistic-costs", action="store_true")
    ap.add_argument("--vix-block",    type=float, default=25.0)
    ap.add_argument("--gsr-extreme",  type=float, default=2.0)
    ap.add_argument("--dd-kill",      type=float, default=-0.20)
    args = ap.parse_args()

    cfg = GateConfig(
        use_liquidity     = not args.no_liquidity,
        use_vix           = not args.no_vix_gate,
        use_gsr           = not args.no_gsr,
        use_drawdown_kill = not args.no_dd_kill,
        vix_block_level   = args.vix_block,
        gsr_z60_extreme_high = args.gsr_extreme,
        gsr_z60_extreme_low  = -args.gsr_extreme,
        dd_kill_threshold = args.dd_kill,
    )

    print("=" * 70)
    print(" v24: gate overlays")
    print("=" * 70)
    print(f"  liquidity gate:   {'✅' if cfg.use_liquidity else '⏭'} "
          f"(vol < {cfg.liquidity_pct_of_median*100:.0f}% of 60d median)")
    print(f"  VIX gate:         {'✅' if cfg.use_vix else '⏭'} "
          f"(VIX > {cfg.vix_block_level} and rising)")
    print(f"  GSR gate:         {'✅' if cfg.use_gsr else '⏭'} "
          f"(|z60| > {cfg.gsr_z60_extreme_high})")
    print(f"  Drawdown kill:    {'✅' if cfg.use_drawdown_kill else '⏭'} "
          f"({cfg.dd_kill_threshold:.0%})")
    print(f"  Realistic costs:  {'✅' if not args.no_realistic_costs else '⏭'}")

    gated, stats = build_gated_decisions(cfg)
    gated.to_csv(V24_DIR / "v24_gated_decisions.csv")
    stats.to_csv(V24_DIR / "v24_gate_blocks_stats.csv", index=False)

    print("\n=== Статистика блокировок ===")
    print(stats.to_string(index=False))

    full = _load_full_data()
    run_v24_backtest(
        gated, cfg, full,
        apply_realistic_costs=(not args.no_realistic_costs),
    )

    # Сохраняем конфиг
    cfg_dict = {
        "use_liquidity":          cfg.use_liquidity,
        "use_vix":                cfg.use_vix,
        "use_gsr":                cfg.use_gsr,
        "use_drawdown_kill":      cfg.use_drawdown_kill,
        "liquidity_pct_median":   cfg.liquidity_pct_of_median,
        "vix_block_level":        cfg.vix_block_level,
        "vix_chg_5d_block":       cfg.vix_chg_5d_block,
        "gsr_z60_extreme_high":   cfg.gsr_z60_extreme_high,
        "gsr_z60_extreme_low":    cfg.gsr_z60_extreme_low,
        "dd_kill_threshold":      cfg.dd_kill_threshold,
    }
    (V24_DIR / "v24_config.json").write_text(
        json.dumps(cfg_dict, indent=2), encoding="utf-8",
    )
    print(f"\n  Конфиг сохранён: {V24_DIR / 'v24_config.json'}")


if __name__ == "__main__":
    main()
