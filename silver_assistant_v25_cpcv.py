"""
Silver Trading Assistant v25 — CPCV-based retraining (полноценный walk-forward)

Что нового vs v22 expanding-window:
1) CPCV (Combinatorial Purged Cross-Validation) вместо expanding window
2) Каждая точка получает прогноз из НЕСКОЛЬКИХ моделей (а не одной)
3) Purging + embargo защищает от утечки информации через перекрытые labels
4) Aggregation: усреднение p_up по всем folds, где точка была в test
5) Backtest с v23 honest math + опционально v24 gates

Запуск:
  python silver_assistant_v25_cpcv.py                    # CPCV + UP-сигналы
  python silver_assistant_v25_cpcv.py --with-gates       # + v24 gates
  python silver_assistant_v25_cpcv.py --n-groups 6 --k-test 2
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# v23 honest math
from silver_assistant_v23_honest import (
    cpcv_splits, CPCVSplit,
    equity_compounded_sequential, risk_metrics_honest,
    recompute_trades_with_realistic_costs, RealisticCosts,
    bootstrap_ci_metrics, sharpe_stats,
    probabilistic_sharpe_ratio, deflated_sharpe_ratio,
)

# v22 backtest + v18 model
from silver_assistant_v22_risk_aware import backtest_strategy_independent
from silver_assistant_v18_adaptive import (
    RegimeEnsembleV18, compute_sample_weights, _get_regimes,
    HISTORICAL_UP_RATE,
)
from silver_assistant_v19_trailing import (
    TRAIL_PCT_DEFAULT, MAX_HOLD_DEFAULT, COST_PER_TRADE,
)
from silver_assistant_v16_binary import TOP_FEATURES_N, apply_policy_v16

# v24 gates (optional)
from silver_assistant_v24_gates import (
    GateConfig, apply_liquidity_gate, apply_vix_gate, apply_gsr_gate,
    apply_drawdown_killswitch,
)

V22_DIR = Path("baseline_outputs_v22")
V25_DIR = Path("baseline_outputs_v25")
V25_DIR.mkdir(exist_ok=True)


# ===========================================================================
# 1. CPCV TRAINING LOOP
# ===========================================================================

def train_predict_one_fold(
    df_labeled: pd.DataFrame,
    fold:       CPCVSplit,
    feature_cols: List[str],
    halflife_years: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Обучает RegimeEnsembleV18 на fold.train_idx, предсказывает p_up на fold.test_idx.
    Возвращает (test_idx_absolute, p_up_predictions).
    """
    train_df = df_labeled.iloc[fold.train_idx]
    test_df  = df_labeled.iloc[fold.test_idx]

    X_train = train_df[feature_cols]
    y_train = train_df["tb_label_bin"].astype(int).values
    regimes_train = _get_regimes(train_df)

    sw = compute_sample_weights(train_df, halflife_years=halflife_years)

    # adaptive_weight = recent_up_rate / historical_up_rate
    recent_up = float(train_df.tail(252)["tb_label_bin"].mean())
    not_up_w  = HISTORICAL_UP_RATE / max(recent_up, 0.05)

    model = RegimeEnsembleV18(not_up_weight=not_up_w)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train, regimes_train, sample_weight=sw)

    X_test = test_df[feature_cols]
    regimes_test = _get_regimes(test_df)
    with contextlib.redirect_stdout(io.StringIO()):
        p_up_test = model.p_up(X_test, regimes_test)

    return fold.test_idx, p_up_test


def cpcv_train_and_predict(
    df_labeled: pd.DataFrame,
    feature_cols: List[str],
    n_groups: int = 6,
    k_test: int = 2,
    embargo_frac: float = 0.01,
) -> pd.Series:
    """
    Запускает полный CPCV-цикл и возвращает Series p_up (по индексу df_labeled).
    Каждая точка усредняется по всем foldам, где она была в test.
    """
    n = len(df_labeled)
    end_t = df_labeled.index.values
    folds = cpcv_splits(n, end_t, n_groups=n_groups, k_test=k_test, embargo_frac=embargo_frac)
    print(f"\n  CPCV folds: {len(folds)} (n_groups={n_groups}, k_test={k_test})")
    if not folds:
        return pd.Series(np.nan, index=df_labeled.index)

    p_sum = np.zeros(n, dtype=float)
    p_cnt = np.zeros(n, dtype=int)

    for i_fold, fold in enumerate(folds):
        test_idx, p_test = train_predict_one_fold(df_labeled, fold, feature_cols)
        p_sum[test_idx] += p_test
        p_cnt[test_idx] += 1
        print(f"    fold {i_fold+1:2d}/{len(folds)}: train={len(fold.train_idx)} "
              f"test={len(fold.test_idx)} p_up_test_mean={p_test.mean():.3f}")

    p_up_aggregated = np.where(p_cnt > 0, p_sum / np.maximum(p_cnt, 1), np.nan)
    print(f"  Покрытие точек: {(p_cnt > 0).sum()}/{n} ({(p_cnt > 0).mean()*100:.1f}%)")
    print(f"  Среднее число folds на точку: {p_cnt[p_cnt > 0].mean():.1f}")

    return pd.Series(p_up_aggregated, index=df_labeled.index, name="p_up_cpcv")


# ===========================================================================
# 2. SIGNAL GENERATION (v22-style policy на CPCV-предсказаниях)
# ===========================================================================

def generate_cpcv_signals(
    df: pd.DataFrame,
    p_up_cpcv: pd.Series,
    up_threshold: float = 0.42,
    cooldown: int = 7,
) -> pd.DataFrame:
    """
    Применяет v22 policy (порог + cooldown) к CPCV p_up.
    Возвращает df с signal_long и signal_short=HOLD (CPCV здесь только UP).
    """
    out = df.copy().sort_index()
    out["p_up"] = p_up_cpcv.reindex(out.index)

    raw = (out["p_up"] >= up_threshold) & out["p_up"].notna()

    signals = []
    last_buy = -10**9
    for i, ok in enumerate(raw.values):
        if ok and (i - last_buy) > cooldown:
            signals.append("BUY")
            last_buy = i
        else:
            signals.append("HOLD")
    out["signal_long"]  = signals
    out["signal_short"] = "HOLD"
    out["signal"]       = signals
    return out


def search_best_policy_on_valid(
    df: pd.DataFrame,
    p_up_cpcv: pd.Series,
) -> dict:
    """Подбирает (up_threshold, cooldown) по precision на valid."""
    valid_df = df[df["split"] == "valid"].copy()
    if "tb_label_bin" not in valid_df.columns:
        return {"up_threshold": 0.50, "cooldown": 7}

    best_obj = -float("inf")
    best     = {"up_threshold": 0.50, "cooldown": 7}
    for thr in [0.42, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for cd in [5, 7, 10, 15]:
            sig = generate_cpcv_signals(valid_df, p_up_cpcv, up_threshold=thr, cooldown=cd)
            buys = sig[(sig["signal_long"] == "BUY") & sig["tb_label_bin"].notna()]
            if len(buys) < 3:
                continue
            prec = float((buys["tb_label_bin"] == 1).mean())
            base = float((valid_df["tb_label_bin"].dropna() == 1).mean())
            obj  = (prec - base) * len(buys)  # lift × volume
            if obj > best_obj:
                best_obj = obj
                best = {"up_threshold": thr, "cooldown": cd,
                        "valid_buys": len(buys), "valid_precision": prec}
    return best


# ===========================================================================
# 3. FEATURE SELECTION (берём v22 top-30 как было)
# ===========================================================================

def load_v22_features() -> List[str]:
    """Берёт top-30 features из v22_feature_importance.csv."""
    p = V22_DIR / "v22_feature_importance.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    fi = pd.read_csv(p)
    return fi.sort_values("importance", ascending=False).head(TOP_FEATURES_N)["feature"].tolist()


# ===========================================================================
# 4. MAIN
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-groups",    type=int, default=6)
    ap.add_argument("--k-test",      type=int, default=2)
    ap.add_argument("--embargo-frac", type=float, default=0.01)
    ap.add_argument("--halflife",    type=float, default=1.5)
    ap.add_argument("--with-gates",  action="store_true",
                    help="Применить v24 gate overlays к CPCV-сигналам")
    args = ap.parse_args()

    print("=" * 70)
    print(" v25 CPCV: Combinatorial Purged Cross-Validation training")
    print("=" * 70)

    # --- 1. Load data ---
    full_path = V22_DIR / "v22_full_data.csv"
    df = pd.read_csv(full_path, parse_dates=[0])
    df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    print(f"  Загружено: {len(df)} строк, {df.index.min().date()} → {df.index.max().date()}")

    # --- 2. Features (top-30 из v22) ---
    feature_cols = load_v22_features()
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Используемых признаков: {len(feature_cols)}")

    # --- 3. Подготовка labeled subset ---
    labeled = df[df["tb_label_bin"].notna() & df[feature_cols].notna().all(axis=1)].copy()
    print(f"  Labeled rows: {len(labeled)} / {len(df)}")

    # --- 4. CPCV training ---
    p_up_cpcv = cpcv_train_and_predict(
        labeled, feature_cols,
        n_groups=args.n_groups, k_test=args.k_test,
        embargo_frac=args.embargo_frac,
    )

    p_up_full = p_up_cpcv.reindex(df.index)
    df["p_up_cpcv"] = p_up_full
    df["p_up_cpcv"].to_csv(V25_DIR / "v25_p_up_cpcv.csv", header=True)

    # --- 5. Policy selection ---
    print("\n=== Подбор policy на valid ===")
    best_policy = search_best_policy_on_valid(df, p_up_full)
    print(f"  Best policy: {best_policy}")
    (V25_DIR / "v25_policy.json").write_text(
        json.dumps(best_policy, indent=2), encoding="utf-8",
    )

    # --- 6. Signal generation ---
    print("\n=== Генерация сигналов ===")
    signaled = generate_cpcv_signals(
        df, p_up_full,
        up_threshold=best_policy["up_threshold"],
        cooldown=best_policy["cooldown"],
    )
    n_buy = (signaled["signal_long"] == "BUY").sum()
    print(f"  Всего BUY-сигналов: {n_buy}")
    for s in ["train", "valid", "test", "forward"]:
        n = (signaled[signaled["split"] == s]["signal_long"] == "BUY").sum()
        print(f"    {s}: {n}")

    # --- 7. Optional v24 gates ---
    if args.with_gates:
        print("\n=== v24 gates overlay ===")
        cfg = GateConfig(
            use_liquidity=True, use_vix=True, use_gsr=True,
            use_drawdown_kill=True,
        )
        signaled = apply_liquidity_gate(signaled, cfg)
        signaled = apply_vix_gate(signaled, cfg)
        signaled = apply_gsr_gate(signaled, cfg)
        for s in ["train", "valid", "test", "forward"]:
            n = (signaled[signaled["split"] == s]["signal_long"] == "BUY").sum()
            print(f"    после gates: {s}: {n}")

    signaled.to_csv(V25_DIR / "v25_decisions.csv")

    # --- 8. Backtest (v23 honest math + realistic costs) ---
    print("\n" + "=" * 70)
    print(" v25 BACKTEST — CPCV signals + v23 honest math")
    print("=" * 70)
    costs = RealisticCosts()

    if "silver_atr_14d" not in signaled.columns and {"silver_high","silver_low","silver_close"}.issubset(signaled.columns):
        cp = signaled["silver_close"].shift(1)
        tr = pd.concat([
            signaled["silver_high"] - signaled["silver_low"],
            (signaled["silver_high"] - cp).abs(),
            (signaled["silver_low"]  - cp).abs(),
        ], axis=1).max(axis=1)
        signaled["silver_atr_14d"] = tr.ewm(span=14, adjust=False).mean()

    rows_pnl, rows_dsr = [], []
    rows_boot = []

    for split in ["valid", "test", "forward"]:
        trades = backtest_strategy_independent(
            signaled, split,
            trail_pct_long=TRAIL_PCT_DEFAULT,  max_hold_long=MAX_HOLD_DEFAULT,
            trail_pct_short=TRAIL_PCT_DEFAULT, max_hold_short=MAX_HOLD_DEFAULT,
            cost=COST_PER_TRADE,
        )
        if not trades.empty:
            trades = recompute_trades_with_realistic_costs(trades, signaled, costs)
        if args.with_gates and not trades.empty:
            cfg = GateConfig()
            trades = apply_drawdown_killswitch(trades, cfg)
        trades.to_csv(V25_DIR / f"v25_{split}_trades.csv", index=False)

        eq, seq = equity_compounded_sequential(trades, return_col="net_return")
        n_kept = len(seq)
        trade_days = int((seq["exit_date"].max() - seq["entry_date"].min()).days) if n_kept >= 2 else 0
        rm = risk_metrics_honest(eq, n_trades=n_kept, trade_days_total=trade_days)

        # v22 baseline для сравнения
        v22_p = V22_DIR / f"v22_base_{split}_trades.csv"
        v22_total = float("nan")
        if v22_p.exists():
            v22_t = pd.read_csv(v22_p)
            v22_t["entry_date"] = pd.to_datetime(v22_t["entry_date"])
            v22_t["exit_date"]  = pd.to_datetime(v22_t["exit_date"])
            v22_t = recompute_trades_with_realistic_costs(v22_t, signaled, costs)
            eq22, _ = equity_compounded_sequential(v22_t, return_col="net_return")
            v22_total = float(eq22[-1] - 1.0)

        d = signaled[signaled["split"] == split]
        true_bnh = float(d["silver_close"].iloc[-1] / d["silver_close"].iloc[0] - 1.0) \
                       if len(d) >= 2 else float("nan")

        seq_rets = seq["net_return"].astype(float).values if not seq.empty else np.array([])
        sharpe_per, skew, kurt = sharpe_stats(seq_rets) if len(seq_rets) >= 4 else (float("nan"), 0, 3)
        psr = probabilistic_sharpe_ratio(sharpe_per, len(seq_rets), skew, kurt, 0.0) if not np.isnan(sharpe_per) else float("nan")
        dsr = deflated_sharpe_ratio(
            sharpe_per, len(seq_rets), n_trials=11,
            sharpe_variance=0.5*(sharpe_per**2 + 1e-6), skew=skew, kurt=kurt,
        ) if not np.isnan(sharpe_per) else float("nan")

        ci = bootstrap_ci_metrics(seq_rets, n_boot=2000, block_len=5.0) if len(seq_rets) >= 3 else {}

        rows_pnl.append({
            "split":                 split,
            "n_trades_v25":          len(trades),
            "n_sequential":          n_kept,
            "v22_honest_total":      round(v22_total, 4) if not np.isnan(v22_total) else None,
            "v25_honest_total":      round(rm["total_return"], 4),
            "improvement_pp":        round(rm["total_return"] - v22_total, 4) if not np.isnan(v22_total) else None,
            "true_bnh":              round(true_bnh, 4) if not np.isnan(true_bnh) else None,
            "vs_bnh":                round(rm["total_return"] - true_bnh, 4) if not np.isnan(true_bnh) else None,
            "cagr":                  round(rm["cagr"], 4) if rm["cagr"] is not None else None,
            "max_dd":                round(rm["max_drawdown"], 4) if rm["max_drawdown"] is not None else None,
            "sharpe_ann":            round(rm["sharpe"], 3) if rm["sharpe"] is not None else None,
            "calmar":                round(rm["calmar"], 3) if rm["calmar"] is not None else None,
        })
        rows_dsr.append({
            "split": split, "n_obs": len(seq_rets),
            "sharpe_per_trade": round(sharpe_per, 4) if not np.isnan(sharpe_per) else None,
            "skew": round(skew, 3), "kurt": round(kurt, 3),
            "psr":  round(psr, 4) if not np.isnan(psr) else None,
            "dsr":  round(dsr, 4) if not np.isnan(dsr) else None,
        })
        if ci:
            rows_boot.append({
                "split": split, "n_obs": len(seq_rets),
                "tr_lower":   round(ci["total_return"]["lower"],  4),
                "tr_median":  round(ci["total_return"]["median"], 4),
                "tr_upper":   round(ci["total_return"]["upper"],  4),
                "shr_lower":  round(ci["sharpe"]["lower"],  3),
                "shr_median": round(ci["sharpe"]["median"], 3),
                "shr_upper":  round(ci["sharpe"]["upper"],  3),
                "mdd_lower":  round(ci["max_drawdown"]["lower"],  4),
                "mdd_median": round(ci["max_drawdown"]["median"], 4),
                "mdd_upper":  round(ci["max_drawdown"]["upper"],  4),
            })

    pnl_df  = pd.DataFrame(rows_pnl)
    dsr_df  = pd.DataFrame(rows_dsr)
    boot_df = pd.DataFrame(rows_boot)
    pnl_df.to_csv (V25_DIR / "v25_pnl_summary.csv", index=False)
    dsr_df.to_csv (V25_DIR / "v25_dsr_psr.csv",    index=False)
    boot_df.to_csv(V25_DIR / "v25_bootstrap_ci.csv", index=False)

    print("\n=== v25 CPCV vs v22 base (оба honest math + realistic costs) ===")
    print(pnl_df.to_string(index=False))
    print("\n=== v25 bootstrap 95% CI ===")
    if not boot_df.empty:
        print(boot_df.to_string(index=False))
    print("\n=== v25 DSR / PSR ===")
    print(dsr_df.to_string(index=False))

    print(f"\n  Все артефакты: {V25_DIR}/")


if __name__ == "__main__":
    main()
