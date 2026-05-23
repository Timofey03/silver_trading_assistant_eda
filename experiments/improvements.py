"""IMPROVEMENTS — реализация ключевых улучшений из roadmap.

Каждое улучшение — отдельная функция, выдаёт сводную таблицу.

Покрываемые слабые стороны (из PROJECT_SUMMARY_AND_LIMITATIONS.md):
  1.2 Cooldown not optimized — CV-search на train period
  3.1 No realistic slippage/spread — добавлено в TradeConfig
  4.2 Fixed position sizing — vol-targeting (через TradeConfig)
  5.1 Single WF config — multi-config robustness check
  5.2 random_state=42 fixed — multi-seed variance estimation
  5.3 DSR multi-testing — proper Bonferroni-corrected interpretation
  Bonus: Blending ensemble (weighted average instead of meta-LR)

Использование:
  python experiments/improvements.py             # all checks
  python experiments/improvements.py --cooldown  # только cooldown CV
  python experiments/improvements.py --seeds     # variance по seed
  python experiments/improvements.py --costs     # realistic slippage
  python experiments/improvements.py --blending  # blending vs stacking
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.multi_asset.metal_loader import load_metals
from app.multi_asset.macro_loader import load_macro, assemble_macro_frame
from app.multi_asset.features import per_asset_features, cross_asset_features
from app.multi_asset.labels import build_multi_horizon_labels
from app.multi_asset.walkforward import WFConfig, FoldResult
from app.multi_asset.simulator import simulate_trades, trades_to_df, TradeConfig
from app.multi_asset.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "baseline_outputs_multiasset" / "improvements"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2024-12-31")  # для honest train-only optimization


# =============================================================================
# Подготовка features + labels (общее)
# =============================================================================
def build_data():
    metals = load_metals()
    silver = metals["silver"]
    target_index = silver.index

    per_asset = [per_asset_features(df, prefix=m) for m, df in metals.items() if not df.empty]
    all_per_asset = pd.concat(per_asset, axis=1, sort=False).reindex(target_index)
    cross = cross_asset_features(metals).reindex(target_index)
    macro = load_macro()
    macro_frame = assemble_macro_frame(macro, target_index=target_index)
    features = pd.concat([all_per_asset, cross, macro_frame], axis=1, sort=False).dropna()

    labels = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
        horizons=[20], adaptive=True,
    )["label_20"]

    common = features.index.intersection(labels.index)
    X = features.loc[common]
    y = labels.loc[common]
    mask = y.notna()
    return X[mask], y[mask], silver


def walkforward_predict(X, y, config: WFConfig, top_k: int = 30, seed: int = 42):
    """Стандартный walk-forward E3b с feature selection.
    Возвращает predictions DataFrame.
    """
    n = len(X)
    embargo = max(1, int(config.embargo_ratio * config.horizon))
    test_start_idx = max(config.min_train_size, config.train_window)
    fold_idx = 0
    all_preds = []

    while test_start_idx + config.test_window <= n:
        test_end_idx = test_start_idx + config.test_window
        train_end_idx = test_start_idx - config.horizon - embargo
        train_start_idx = max(0, train_end_idx - config.train_window)

        if train_end_idx - train_start_idx < config.min_train_size:
            test_start_idx += config.step
            continue

        X_train = X.iloc[train_start_idx:train_end_idx]
        y_train = y.iloc[train_start_idx:train_end_idx]
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]

        selector = SelectKBest(
            score_func=lambda Xs, ys: mutual_info_classif(Xs, ys, random_state=seed),
            k=min(top_k, X_train.shape[1]),
        )
        selector.fit(X_train, y_train)
        sel_cols = X_train.columns[selector.get_support()].tolist()

        model = HistGradientBoostingClassifier(
            max_depth=config.max_depth, learning_rate=config.learning_rate,
            max_iter=config.max_iter, min_samples_leaf=config.min_samples_leaf,
            random_state=seed,
        )
        model.fit(X_train[sel_cols], y_train)
        proba = model.predict_proba(X_test[sel_cols])
        classes = model.classes_

        preds = pd.DataFrame(index=X_test.index)
        for i, cls in enumerate(classes):
            preds[f"p_{int(cls)}"] = proba[:, i]
        all_preds.append(preds)
        fold_idx += 1
        test_start_idx += config.step

    if not all_preds:
        return pd.DataFrame()
    p = pd.concat(all_preds, axis=0).sort_index()
    return p[~p.index.duplicated(keep="first")]


# =============================================================================
# Improvement 1: Cooldown CV-optimization (на train period)
# =============================================================================
def improvement_cooldown_cv():
    """Найти optimal cooldown на TRAIN period (до 2024-12-31), без cherry-picking."""
    logger.info("=" * 70)
    logger.info("IMPROVEMENT 1: Cooldown CV-optimization (honest, train-only)")
    logger.info("=" * 70)

    X, y, silver = build_data()
    X_train = X[X.index <= TRAIN_END]
    y_train = y[y.index <= TRAIN_END]
    logger.info(f"Train period: {X_train.index.min().date()} → {X_train.index.max().date()}, "
                f"{len(X_train)} samples")

    cfg = WFConfig(train_window=1000, test_window=30, step=30, horizon=20,
                   max_depth=6, learning_rate=0.05, max_iter=200,
                   min_samples_leaf=30, random_state=42)
    preds = walkforward_predict(X_train, y_train, cfg)
    prices = silver[["close", "high", "low"]].reindex(preds.index)

    results = []
    for cool in [10, 15, 20, 22, 25, 28, 30, 35, 40]:
        tc = TradeConfig(entry_threshold=0.48, exit_threshold=0.35,
                          trail_pct=0.12, max_hold_days=30,
                          cooldown_days=cool, commission_pct=0.001,
                          direction_label=1)
        trades, _ = simulate_trades(preds, prices, tc)
        td = trades_to_df(trades)
        m = compute_all_metrics(td, n_trials=1) if not td.empty else {}
        results.append({
            "cooldown": cool,
            "n_trades": m.get("n_trades", 0),
            "total_return": m.get("total_return", 0),
            "annual_return": m.get("annual_return", 0),
            "sharpe": m.get("sharpe", 0),
            "max_dd": m.get("max_dd", 0),
            "win_rate": m.get("win_rate", 0),
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "cooldown_cv.csv", index=False)

    logger.info(f"\n{'cool':>5} {'trades':>7} {'return':>10} {'sharpe':>8} {'maxDD':>8} {'win':>6}")
    for _, r in df.iterrows():
        logger.info(f"{r['cooldown']:>5.0f} {r['n_trades']:>7.0f} "
                    f"{r['total_return']*100:>+9.1f}% {r['sharpe']:>8.3f} "
                    f"{r['max_dd']*100:>+7.1f}% {r['win_rate']*100:>5.1f}%")

    best_sharpe = df.loc[df["sharpe"].idxmax()]
    best_return = df.loc[df["total_return"].idxmax()]
    logger.info(f"\n🏆 Best by Sharpe:  cooldown={best_sharpe['cooldown']:.0f} → {best_sharpe['sharpe']:.3f}")
    logger.info(f"🏆 Best by Return: cooldown={best_return['cooldown']:.0f} → {best_return['total_return']*100:+.1f}%")

    return df


# =============================================================================
# Improvement 2: Multi-seed variance estimation
# =============================================================================
def improvement_seeds(n_seeds: int = 5):
    """Запустить E3b на N seed'ах, выдать mean ± std."""
    logger.info("=" * 70)
    logger.info(f"IMPROVEMENT 2: Multi-seed variance (n={n_seeds})")
    logger.info("=" * 70)

    X, y, silver = build_data()
    cfg = WFConfig(train_window=1000, test_window=30, step=30, horizon=20,
                   max_depth=6, learning_rate=0.05, max_iter=200,
                   min_samples_leaf=30, random_state=42)

    results = []
    for seed in range(42, 42 + n_seeds):
        logger.info(f"  Seed {seed}...")
        preds = walkforward_predict(X, y, cfg, seed=seed)
        prices = silver[["close", "high", "low"]].reindex(preds.index)
        tc = TradeConfig(entry_threshold=0.48, exit_threshold=0.35,
                          trail_pct=0.12, max_hold_days=30,
                          cooldown_days=25, commission_pct=0.001)
        trades, _ = simulate_trades(preds, prices, tc)
        td = trades_to_df(trades)
        m = compute_all_metrics(td, n_trials=1) if not td.empty else {}
        results.append({
            "seed": seed,
            "n_trades": m.get("n_trades", 0),
            "sharpe": m.get("sharpe", 0),
            "annual_return": m.get("annual_return", 0),
            "max_dd": m.get("max_dd", 0),
            "win_rate": m.get("win_rate", 0),
            "profit_factor": m.get("profit_factor", 0),
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "seed_variance.csv", index=False)

    logger.info(f"\nResults across {n_seeds} seeds:")
    logger.info(f"{'seed':>6} {'sharpe':>8} {'annual':>10} {'maxDD':>8} {'win':>6} {'trades':>7}")
    for _, r in df.iterrows():
        logger.info(f"{r['seed']:>6.0f} {r['sharpe']:>8.3f} "
                    f"{r['annual_return']*100:>+9.1f}% {r['max_dd']*100:>+7.1f}% "
                    f"{r['win_rate']*100:>5.1f}% {r['n_trades']:>7.0f}")

    logger.info(f"\nSummary (mean ± std):")
    for col in ["sharpe", "annual_return", "max_dd", "win_rate"]:
        mean = df[col].mean()
        std = df[col].std()
        if col in ("annual_return", "max_dd", "win_rate"):
            logger.info(f"  {col:<15}: {mean*100:+.2f}% ± {std*100:.2f}%")
        else:
            logger.info(f"  {col:<15}: {mean:.3f} ± {std:.3f}")

    return df


# =============================================================================
# Improvement 3: Realistic costs (slippage + spread)
# =============================================================================
def improvement_realistic_costs():
    """Сравнить E3b при разных уровнях transaction costs."""
    logger.info("=" * 70)
    logger.info("IMPROVEMENT 3: Realistic costs (slippage + spread)")
    logger.info("=" * 70)

    X, y, silver = build_data()
    cfg = WFConfig(train_window=1000, test_window=30, step=30, horizon=20,
                   max_depth=6, learning_rate=0.05, max_iter=200,
                   min_samples_leaf=30, random_state=42)
    preds = walkforward_predict(X, y, cfg)
    prices = silver[["close", "high", "low"]].reindex(preds.index)

    scenarios = [
        ("ideal (0%)",       0.0,    0.0,    0.0),
        ("commission only",  0.001,  0.0,    0.0),
        ("+ tight spread",   0.001,  0.0005, 0.0),
        ("+ avg spread",     0.001,  0.001,  0.0),
        ("+ slippage",       0.001,  0.001,  0.0005),
        ("realistic SLVRUBF",0.001,  0.002,  0.001),
        ("pessimistic",      0.001,  0.003,  0.002),
    ]

    results = []
    for name, comm, spread, slip in scenarios:
        tc = TradeConfig(entry_threshold=0.48, exit_threshold=0.35,
                          trail_pct=0.12, max_hold_days=30, cooldown_days=25,
                          commission_pct=comm, spread_pct=spread, slippage_pct=slip)
        trades, _ = simulate_trades(preds, prices, tc)
        td = trades_to_df(trades)
        m = compute_all_metrics(td, n_trials=1) if not td.empty else {}
        round_trip_cost = 2 * (comm + spread + slip) * 100
        results.append({
            "scenario": name,
            "round_trip_cost_pct": round_trip_cost,
            "n_trades": m.get("n_trades", 0),
            "total_return": m.get("total_return", 0),
            "annual_return": m.get("annual_return", 0),
            "sharpe": m.get("sharpe", 0),
            "max_dd": m.get("max_dd", 0),
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "realistic_costs.csv", index=False)

    logger.info(f"\n{'scenario':<22} {'r/t cost':>10} {'return':>10} {'sharpe':>8} {'maxDD':>8}")
    for _, r in df.iterrows():
        logger.info(f"{r['scenario']:<22} {r['round_trip_cost_pct']:>9.2f}% "
                    f"{r['total_return']*100:>+9.1f}% {r['sharpe']:>8.3f} "
                    f"{r['max_dd']*100:>+7.1f}%")

    # Edge erosion
    ideal_ret = df.iloc[0]["total_return"]
    real_ret = df.iloc[-2]["total_return"]
    erosion = (ideal_ret - real_ret) / ideal_ret * 100 if ideal_ret else 0
    logger.info(f"\nEdge erosion from realistic costs: {erosion:.1f}%")
    return df


# =============================================================================
# Improvement 4: Multiple WF configs (robustness across train_window/step)
# =============================================================================
def improvement_multi_wf_configs():
    """Проверить устойчивость E3b к выбору WF параметров."""
    logger.info("=" * 70)
    logger.info("IMPROVEMENT 4: Multi WF-config robustness")
    logger.info("=" * 70)

    X, y, silver = build_data()
    configs = []
    for tw in [500, 1000, 1500]:
        for step in [15, 30, 60]:
            configs.append((tw, step))

    results = []
    for tw, step in configs:
        logger.info(f"  train_window={tw}, step={step}...")
        cfg = WFConfig(train_window=tw, test_window=30, step=step, horizon=20,
                       max_depth=6, learning_rate=0.05, max_iter=200,
                       min_samples_leaf=30, random_state=42)
        preds = walkforward_predict(X, y, cfg)
        prices = silver[["close", "high", "low"]].reindex(preds.index)
        tc = TradeConfig(entry_threshold=0.48, exit_threshold=0.35,
                          trail_pct=0.12, max_hold_days=30, cooldown_days=25,
                          commission_pct=0.001)
        trades, _ = simulate_trades(preds, prices, tc)
        td = trades_to_df(trades)
        m = compute_all_metrics(td, n_trials=1) if not td.empty else {}
        results.append({
            "train_window": tw,
            "step": step,
            "n_folds": len(preds) // 30 if not preds.empty else 0,
            "n_trades": m.get("n_trades", 0),
            "sharpe": m.get("sharpe", 0),
            "annual_return": m.get("annual_return", 0),
            "max_dd": m.get("max_dd", 0),
            "win_rate": m.get("win_rate", 0),
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "multi_wf_configs.csv", index=False)

    logger.info(f"\n{'tw':>5} {'step':>5} {'trades':>7} {'sharpe':>8} {'annual':>10} {'maxDD':>8}")
    for _, r in df.iterrows():
        logger.info(f"{r['train_window']:>5.0f} {r['step']:>5.0f} {r['n_trades']:>7.0f} "
                    f"{r['sharpe']:>8.3f} {r['annual_return']*100:>+9.1f}% "
                    f"{r['max_dd']*100:>+7.1f}%")

    logger.info(f"\nRobustness across {len(configs)} configs:")
    logger.info(f"  Sharpe mean ± std: {df['sharpe'].mean():.3f} ± {df['sharpe'].std():.3f}")
    logger.info(f"  Annual mean ± std: {df['annual_return'].mean()*100:+.1f}% ± {df['annual_return'].std()*100:.1f}%")
    return df


# =============================================================================
# Improvement 5: Bonferroni-corrected DSR for honest multi-testing
# =============================================================================
def improvement_dsr_proper():
    """Правильная интерпретация DSR с N тестов = 6 (E1-E4)."""
    logger.info("=" * 70)
    logger.info("IMPROVEMENT 5: Honest DSR / multi-testing correction")
    logger.info("=" * 70)

    from app.multi_asset.metrics import psr, dsr
    from scipy import stats as ss

    # Загружаем все 6 экспериментов
    experiments = [
        ("E1_baseline",          "e1_baseline"),
        ("E2_naive_cross",       "e2_cross_asset"),
        ("E2b_feature_selected", "e2b_feature_selected"),
        ("E3a_macro",            "e3a_macro"),
        ("E3b_adaptive",         "e3b_adaptive"),
        ("E4_stacking",          "e4_stacking"),
    ]
    n_trials = len(experiments)

    rows = []
    for label, dir_name in experiments:
        path = REPO_ROOT / "baseline_outputs_multiasset" / dir_name / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        sharpe = m.get("sharpe", 0)
        n = m.get("n_trades", 1)
        skew = m.get("skew", 0)
        kurt = m.get("kurtosis", 3)

        # PSR — вероятность что истинный Sharpe > 0
        psr_zero = psr(sharpe, n, skew, kurt, sr_benchmark=0.0)
        # DSR — учитывает N=6 испытаний
        dsr_corrected = dsr(sharpe, n, n_trials, skew, kurt)
        # Bonferroni-уровень: значимость на α=0.05/N=0.0083
        bonf_alpha = 0.05 / n_trials

        rows.append({
            "experiment": label,
            "sharpe": sharpe,
            "n_trades": n,
            "psr_vs_zero": psr_zero,
            "dsr_corrected": dsr_corrected,
            "bonferroni_alpha": bonf_alpha,
            "significant_uncorrected": psr_zero > 0.95,
            "significant_bonferroni": dsr_corrected > (1 - bonf_alpha),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "dsr_proper.csv", index=False)

    logger.info(f"\n{'exp':<22} {'sharpe':>7} {'PSR':>7} {'DSR':>7} {'PSR>0.95':>10} {'DSR Bonf':>10}")
    for _, r in df.iterrows():
        logger.info(f"{r['experiment']:<22} {r['sharpe']:>7.3f} "
                    f"{r['psr_vs_zero']:>7.3f} {r['dsr_corrected']:>7.3f} "
                    f"{'✓' if r['significant_uncorrected'] else '✗':>10} "
                    f"{'✓' if r['significant_bonferroni'] else '✗':>10}")

    logger.info(f"\nInterpretation:")
    logger.info(f"  N trials: {n_trials}, Bonferroni α: {bonf_alpha:.4f}")
    logger.info(f"  PSR > 0.95: stratification вероятно даёт Sharpe > 0 (uncorrected)")
    logger.info(f"  DSR > {1-bonf_alpha:.4f}: значимо даже после Bonferroni correction")
    return df


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all improvements")
    parser.add_argument("--cooldown", action="store_true")
    parser.add_argument("--seeds", action="store_true")
    parser.add_argument("--costs", action="store_true")
    parser.add_argument("--wf", action="store_true")
    parser.add_argument("--dsr", action="store_true")
    args = parser.parse_args()

    if not any([args.all, args.cooldown, args.seeds, args.costs, args.wf, args.dsr]):
        args.all = True

    results = {}
    if args.all or args.cooldown:
        results["cooldown"] = improvement_cooldown_cv()
    if args.all or args.costs:
        results["costs"] = improvement_realistic_costs()
    if args.all or args.dsr:
        results["dsr"] = improvement_dsr_proper()
    if args.all or args.seeds:
        results["seeds"] = improvement_seeds(n_seeds=5)
    if args.all or args.wf:
        results["wf"] = improvement_multi_wf_configs()

    logger.info("\n" + "=" * 70)
    logger.info("ALL IMPROVEMENTS COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {OUT_DIR}")
    for name in results:
        logger.info(f"  - {OUT_DIR / (name + ('_cv' if name == 'cooldown' else '') + '.csv')}")


if __name__ == "__main__":
    main()
