"""
silver_assistant_v26_multiasset.py — Foundation для multi-asset торговли

ЦЕЛЬ session 1: показать что gold-only стратегия работает аналогично silver.
ЦЕЛЬ session 2: объединить signals в портфель (risk parity allocation).
ЦЕЛЬ session 3: pair trading + полноценный SHORT.

Что делает СЕЙЧАС:
1. Fetch OHLC золотых фьючерсов (GC=F) через yfinance
2. Build features (re-use v14 build_features логику)
3. Triple-barrier labels для золота
4. CPCV training на gold (повторяем v25 архитектуру)
5. Backtest gold-only стратегии с Optimal mode
6. Сравнение с silver baseline

Запуск:
  python silver_assistant_v26_multiasset.py --fetch          # только данные
  python silver_assistant_v26_multiasset.py --train          # обучение
  python silver_assistant_v26_multiasset.py --backtest       # бэктест
  python silver_assistant_v26_multiasset.py --all            # всё (default)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

V22_DIR = Path("baseline_outputs_v22")
V26_DIR = Path("baseline_outputs_v26")
V26_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. FETCH GOLD FUTURES
# =============================================================================

def fetch_gold_data(start: str = "2013-01-01", end: str | None = None) -> pd.DataFrame:
    """Скачивает OHLC GC=F (gold futures) через yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"  Fetching GC=F (Gold futures) {start} → {end}")
    data = yf.download("GC=F", start=start, end=end, progress=False,
                       auto_adjust=False, threads=False)
    if data is None or data.empty:
        raise RuntimeError("Не удалось скачать GC=F")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    df = pd.DataFrame({
        "gold_open":   data["Open"],
        "gold_high":   data["High"],
        "gold_low":    data["Low"],
        "gold_close":  data["Close"],
        "gold_volume": data["Volume"],
    })
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.dropna(subset=["gold_close"])


# =============================================================================
# 2. FEATURE ENGINEERING (gold-specific)
# =============================================================================

def build_gold_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт ~30 фичей для gold (returns, RSI, MACD, ATR, volatility regimes).
    Аналог build_features из v14, но для gold.
    """
    d = df.copy()

    # Returns
    for n in [1, 2, 3, 5, 10, 20]:
        d[f"gold_ret_{n}d"] = d["gold_close"].pct_change(n)

    # Volatility (rolling std of daily returns)
    d["gold_realized_vol_20d"] = d["gold_ret_1d"].rolling(20).std() * np.sqrt(252)
    d["gold_realized_vol_60d"] = d["gold_ret_1d"].rolling(60).std() * np.sqrt(252)

    # ATR
    close_prev = d["gold_close"].shift(1)
    tr = pd.concat([
        d["gold_high"] - d["gold_low"],
        (d["gold_high"] - close_prev).abs(),
        (d["gold_low"]  - close_prev).abs(),
    ], axis=1).max(axis=1)
    d["gold_atr_14"] = tr.ewm(span=14, adjust=False).mean()

    # RSI
    delta = d["gold_close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["gold_rsi_14"] = 100 - (100 / (1 + rs))

    # Z-scores
    for n in [20, 60]:
        ma = d["gold_close"].rolling(n).mean()
        std = d["gold_close"].rolling(n).std()
        d[f"gold_zscore_{n}d"] = (d["gold_close"] - ma) / std
        d[f"gold_dist_ma{n}"] = d["gold_close"] / ma - 1

    # MA slope
    ma20 = d["gold_close"].rolling(20).mean()
    d["gold_ma20_slope_5d"] = (ma20 - ma20.shift(5)) / ma20.shift(5)

    # MACD histogram
    ema12 = d["gold_close"].ewm(span=12, adjust=False).mean()
    ema26 = d["gold_close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["gold_macd_hist"] = macd - signal

    # Distance to recent high
    for n in [20, 60]:
        d[f"gold_dist_high_{n}d"] = d["gold_close"] / d["gold_close"].rolling(n).max() - 1

    return d


# =============================================================================
# 3. TRIPLE BARRIER LABELS (для gold)
# =============================================================================

def add_gold_labels(df: pd.DataFrame, horizon: int = 15,
                    barrier_up: float = 0.01, barrier_dn: float = 0.01) -> pd.DataFrame:
    """
    Triple-barrier labels для gold:
      UP    = цена выросла >+2% в течение horizon дней
      DOWN  = цена упала >-2% в течение horizon
      NEUTRAL = ни одно из условий
    """
    d = df.copy()
    close = d["gold_close"].values
    n = len(close)

    label = np.full(n, "NEUTRAL", dtype=object)
    label_bin = np.zeros(n, dtype=float)
    label_bin[:] = np.nan

    for i in range(n - horizon):
        entry = close[i]
        upper = entry * (1 + barrier_up)
        lower = entry * (1 - barrier_dn)
        for j in range(1, horizon + 1):
            if close[i + j] >= upper:
                label[i] = "UP"
                label_bin[i] = 1
                break
            elif close[i + j] <= lower:
                label[i] = "DOWN"
                label_bin[i] = 0
                break
        else:
            label_bin[i] = 1 if close[i + horizon] > entry else 0

    d["gold_tb_label"] = label
    d["gold_tb_label_bin"] = label_bin
    return d


# =============================================================================
# 4. ADD SPLITS (consistent with silver)
# =============================================================================

def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Train/valid/test/forward — синхронизировано с silver."""
    d = df.copy()
    d["split"] = "train"
    d.loc[d.index >= "2023-01-01", "split"] = "valid"
    d.loc[d.index >= "2024-01-01", "split"] = "test"
    d.loc[d.index >= "2025-01-01", "split"] = "forward"
    return d


# =============================================================================
# 5. CPCV training на gold
# =============================================================================

def train_gold_cpcv(
    df: pd.DataFrame, feature_cols: List[str],
    n_groups: int = 6, k_test: int = 2,
) -> pd.Series:
    """CPCV на gold — повторяем silver v25 pipeline."""
    from silver_assistant_v23_honest import cpcv_splits
    from silver_assistant_v18_adaptive import (
        RegimeEnsembleV18, compute_sample_weights, _get_regimes, HISTORICAL_UP_RATE,
    )

    labeled = df[df["gold_tb_label_bin"].notna() & df[feature_cols].notna().all(axis=1)].copy()
    print(f"  Labeled rows: {len(labeled)}")

    # Простой regime для gold (нет полноценной режимной модели)
    if "regime" not in labeled.columns:
        ma60 = labeled["gold_close"].rolling(60).mean()
        labeled["regime"] = np.where(
            labeled["gold_close"] > ma60 * 1.02, "uptrend_medium",
            np.where(labeled["gold_close"] < ma60 * 0.98, "downtrend_medium",
                     "sideways_medium"),
        )

    n = len(labeled)
    end_t = labeled.index.values
    folds = cpcv_splits(n, end_t, n_groups=n_groups, k_test=k_test, embargo_frac=0.01)
    print(f"  CPCV folds: {len(folds)}")

    p_sum = np.zeros(n, dtype=float)
    p_cnt = np.zeros(n, dtype=int)

    for i_fold, fold in enumerate(folds):
        train_df = labeled.iloc[fold.train_idx]
        test_df  = labeled.iloc[fold.test_idx]

        X_train = train_df[feature_cols]
        y_train = train_df["gold_tb_label_bin"].astype(int).values
        regimes_train = _get_regimes(train_df)

        sw = compute_sample_weights(train_df, halflife_years=1.5)
        recent_up = float(train_df.tail(252)["gold_tb_label_bin"].mean())
        not_up_w = HISTORICAL_UP_RATE / max(recent_up, 0.05)

        model = RegimeEnsembleV18(not_up_weight=not_up_w)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X_train, y_train, regimes_train, sample_weight=sw)

        X_test = test_df[feature_cols]
        regimes_test = _get_regimes(test_df)
        with contextlib.redirect_stdout(io.StringIO()):
            p_up_test = model.p_up(X_test, regimes_test)

        p_sum[fold.test_idx] += p_up_test
        p_cnt[fold.test_idx] += 1
        if (i_fold + 1) % 5 == 0:
            print(f"    fold {i_fold+1}/{len(folds)} done")

    p_up_agg = np.where(p_cnt > 0, p_sum / np.maximum(p_cnt, 1), np.nan)
    return pd.Series(p_up_agg, index=labeled.index, name="p_up_gold")


# =============================================================================
# 6. BACKTEST gold-only
# =============================================================================

def backtest_gold(df: pd.DataFrame, p_up: pd.Series) -> dict:
    """Бэктест gold с Optimal mode параметрами."""
    from silver_signal_modes import OPTIMAL_PARAMS

    d = df.copy()
    d["p_up"] = p_up.reindex(d.index)

    # Renaming: алгоритм бэктеста ожидает silver_* колонки. Сделаем proxy.
    d_renamed = d.copy()
    d_renamed["silver_open"]   = d["gold_open"]
    d_renamed["silver_high"]   = d["gold_high"]
    d_renamed["silver_low"]    = d["gold_low"]
    d_renamed["silver_close"]  = d["gold_close"]
    d_renamed["silver_volume"] = d["gold_volume"]

    from silver_signal_modes import generate_signals_with_exits, backtest_with_model_exits
    from silver_assistant_v23_honest import (
        equity_compounded_sequential, risk_metrics_honest,
        recompute_trades_with_realistic_costs, RealisticCosts,
    )

    signaled = generate_signals_with_exits(d_renamed, d_renamed["p_up"], OPTIMAL_PARAMS)

    if "silver_atr_14d" not in signaled.columns:
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
        trades = backtest_with_model_exits(signaled, split, OPTIMAL_PARAMS)
        if trades.empty:
            results[split] = {"n_trades": 0, "total_return": 0}
            continue
        trades = recompute_trades_with_realistic_costs(trades, signaled, costs)
        eq, seq = equity_compounded_sequential(trades, return_col="net_return")
        n_kept = len(seq)
        trade_days = int((seq["exit_date"].max() - seq["entry_date"].min()).days) \
            if n_kept >= 2 else 0
        m = risk_metrics_honest(eq, n_trades=n_kept, trade_days_total=trade_days)
        results[split] = {
            "n_trades":     len(trades),
            "win_rate":     round(float((trades["net_return"] > 0).mean()), 4),
            "total_return": round(m["total_return"], 4),
            "max_drawdown": round(m["max_drawdown"], 4) if m["max_drawdown"] else None,
            "sharpe_ann":   round(m["sharpe"], 3) if m["sharpe"] else None,
        }
        trades.to_csv(V26_DIR / f"gold_trades_{split}.csv", index=False)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch",   action="store_true")
    ap.add_argument("--train",   action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--all",     action="store_true")
    args = ap.parse_args()

    run_all = args.all or not any([args.fetch, args.train, args.backtest])

    # === FETCH ===
    gold_data_path = V26_DIR / "gold_full_data.csv"
    if args.fetch or run_all or not gold_data_path.exists():
        print("=== STEP 1: Fetch gold data ===")
        gold = fetch_gold_data()
        gold = build_gold_features(gold)
        gold = add_gold_labels(gold)
        gold = add_splits(gold)
        gold.to_csv(gold_data_path)
        print(f"  Saved: {gold_data_path} ({len(gold)} rows)")
    else:
        gold = pd.read_csv(gold_data_path, parse_dates=["Date"]).set_index("Date")
        print(f"  Loaded: {gold_data_path} ({len(gold)} rows)")

    # === TRAIN ===
    p_up_path = V26_DIR / "gold_p_up_cpcv.csv"
    feature_cols = [c for c in gold.columns if c.startswith("gold_ret_")
                    or c.startswith("gold_zscore_")
                    or c.startswith("gold_dist_")
                    or c.startswith("gold_ma")
                    or c.startswith("gold_atr")
                    or c.startswith("gold_rsi")
                    or c.startswith("gold_macd")
                    or c.startswith("gold_realized_vol")]
    print(f"\n  Features ({len(feature_cols)}): {feature_cols[:5]}...")

    if args.train or run_all or not p_up_path.exists():
        print("\n=== STEP 2: CPCV training on gold ===")
        p_up_gold = train_gold_cpcv(gold, feature_cols)
        p_up_gold.to_csv(p_up_path, header=True)
        print(f"  Saved: {p_up_path}")
    else:
        p_up_gold = pd.read_csv(p_up_path, parse_dates=["Date"]).set_index("Date").iloc[:, 0]

    # === BACKTEST ===
    if args.backtest or run_all:
        print("\n=== STEP 3: Backtest gold-only with Optimal mode ===")
        gold = gold.reindex(gold.index)  # already aligned
        # Important: p_up_gold has labeled index, need full reindex
        p_up_full = p_up_gold.reindex(gold.index)
        results = backtest_gold(gold, p_up_full)

        print("\n=== GOLD STANDALONE RESULTS ===")
        for split, r in results.items():
            print(f"  {split:8s}: trades={r.get('n_trades', 0):3d}  "
                  f"win={r.get('win_rate', 0):.2%}  "
                  f"total_return={r.get('total_return', 0):+.2%}  "
                  f"Sharpe={r.get('sharpe_ann', 'n/a')}")

        # Save summary
        summary = pd.DataFrame([
            {"split": split, **{k: v for k, v in r.items()}}
            for split, r in results.items()
        ])
        summary.to_csv(V26_DIR / "gold_backtest_summary.csv", index=False)
        print(f"\n  Saved: {V26_DIR / 'gold_backtest_summary.csv'}")

        # Compare with silver
        silver_pnl = pd.read_csv(Path("baseline_outputs_v25") / "v25_pnl_summary.csv")
        print("\n=== SILVER vs GOLD comparison (forward) ===")
        for asset, df_pnl in [("Silver", silver_pnl), ("Gold", summary)]:
            if asset == "Silver":
                fwd = df_pnl[df_pnl["split"] == "forward"].iloc[0]
                ret = fwd.get("v25_honest_total", 0)
                sharpe = fwd.get("sharpe_ann", 0)
            else:
                fwd = df_pnl[df_pnl["split"] == "forward"].iloc[0]
                ret = fwd.get("total_return", 0)
                sharpe = fwd.get("sharpe_ann", 0)
            print(f"  {asset:6s}: total_return={ret*100:+.1f}%  Sharpe={sharpe}")


if __name__ == "__main__":
    main()
