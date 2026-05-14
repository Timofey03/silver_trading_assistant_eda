# Архитектура pipeline

## Общая схема

```
┌─────────────────────────────────────────────────────────────────────┐
│ DATA INGESTION                                                       │
│   yfinance (silver, gold, copper, oil, sp500, eurusd, dxy, vix)     │
│     +                                                                │
│   nasdaqdatalink (COT report, CFTC)                                  │
│     +                                                                │
│   yfinance ETF proxies (TIP, RINF, HYG, ^TNX, ^IRX) ← FRED suprogate│
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ FEATURE ENGINEERING (v14 + v15 + v17)                                │
│   - returns, RSI, MACD, ATR, BB, MA distances                        │
│   - volatility regimes (low/medium/high)                             │
│   - trend regimes (up/sideways/down)                                 │
│   - macro features (real rates, inflation expectations, credit)      │
│   - COT (commercials/specs/open interest)                            │
│   - gold/silver ratio + z-scores                                     │
│   Итого: ~130 фичей                                                  │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LABELS (v14 — Triple Barrier, López de Prado)                        │
│   Горизонт: 15 торговых дней                                         │
│   Barriers: TP (+0.6%), SL (−0.6%), time (15d)                       │
│   Output: tb_label ∈ {UP, NEUTRAL, DOWN}                             │
│   Binary: tb_label_bin = 1 if UP else 0  (v16+)                      │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ TRAINING                                                             │
│                                                                      │
│   v22 (expanding window):                                            │
│     for cutoff in [2020-01, 2021-01, 2022-01, 2023-01, 2024-01]:    │
│         train on all data <= cutoff                                  │
│         predict on (cutoff, next_cutoff)                             │
│                                                                      │
│   v25 (CPCV) ← ТЕКУЩАЯ:                                              │
│     groups = split data into 6 chunks                                │
│     for combo in choose(6, 2):  # 15 folds                           │
│         test = combo (with purging+embargo)                          │
│         train = remaining                                            │
│         predict on test                                              │
│     aggregate: avg(p_up) per row across folds covering it            │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODEL: RegimeEnsembleV18                                             │
│   - HistGradientBoosting per regime (uptrend/sideways/downtrend)     │
│   - Calibrated (isotonic)                                            │
│   - Sample weights: exponential time decay (half-life 1.5 years)     │
│   - Adaptive class weight (compensates UP/NOT_UP imbalance)          │
│   - Top-30 features (permutation importance)                         │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ POLICY APPLICATION                                                   │
│   if p_up >= up_threshold and (today - last_buy) > cooldown:         │
│       signal = "BUY"                                                 │
│   else:                                                              │
│       signal = "HOLD"                                                │
│                                                                      │
│   v22:  up_threshold=0.42, cooldown=7                                │
│   v25:  up_threshold=0.55, cooldown=15  ← honest threshold            │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ GATES (v24, опционально)                                             │
│   - liquidity_gate: блок при volume < 50% median                     │
│   - vix_gate: блок LONG при VIX > 25 & растёт                        │
│   - gsr_gate: блок при экстремуме gold/silver ratio z-score         │
│   - drawdown_killswitch: остановка при equity DD > 20%               │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ EXECUTION (v19 trailing stop)                                        │
│   For each BUY signal:                                               │
│     entry = next day open                                            │
│     peak = entry                                                     │
│     trail_stop = peak × (1 − trail_pct)                              │
│     while not stopped and hold < max_hold:                           │
│         peak = max(peak, today_high)                                 │
│         trail_stop = peak × (1 − trail_pct)                          │
│         if today_low < trail_stop:                                   │
│             exit = trail_stop                                        │
│             break                                                    │
│     net_return = (exit/entry - 1) - realistic_cost                   │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ HONEST EVALUATION (v23)                                              │
│   equity_compounded_sequential(trades):                              │
│     for trade in sorted_by_entry:                                    │
│         if entry >= prev_exit:                                       │
│             equity *= (1 + net_return)                               │
│   → real money equity curve                                          │
│                                                                      │
│   Risk metrics: total_return, CAGR, Sharpe, Sortino, Calmar,         │
│                 Ulcer, time_underwater, max_drawdown                 │
│   Stat tests: bootstrap CI, DSR, PSR                                 │
│   Performance attribution: model / sizing / execution / costs        │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PAPER TRADING (silver_paper_tinkoff.py)                              │
│   1. Audit log → JSONL декларация каждого сигнала                    │
│   2. Tinkoff REST API → sandbox order placement                      │
│   3. Daily --live mode → cron / Windows Task Scheduler               │
│   4. Status check → portfolio snapshot                               │
│   Инструмент: SLVRUBF (MOEX silver futures, RUB)                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Версионная цепочка импортов

`silver_assistant_v25_cpcv.py` импортирует:
- `silver_assistant_v23_honest`: CPCV splits, equity math, DSR/PSR, bootstrap
- `silver_assistant_v22_risk_aware`: backtest_strategy_independent
- `silver_assistant_v19_trailing`: TRAIL_PCT_DEFAULT, MAX_HOLD_DEFAULT, COST_PER_TRADE
- `silver_assistant_v18_adaptive`: RegimeEnsembleV18, compute_sample_weights, _get_regimes
- `silver_assistant_v16_binary`: TOP_FEATURES_N, apply_policy_v16
- `silver_assistant_v24_gates`: GateConfig, apply_*_gate

`silver_assistant_v22_risk_aware.py` импортирует:
- v21 → v20 → v19 → v18 → v17 → v16 → v15 → v14

То есть **ВСЕ v14-v22 файлы нужны** для работы v25.

## Данные

### v22_full_data.csv (канонический датасет)
- 3481 строка (2013-01-02 → 2026-05-12)
- ~130 колонок: OHLC + features + labels + split + regime
- Splits: train (до 2022-12), valid (2023), test (2024), forward (2025+)
- Метки: `tb_label_bin` (UP=1, иначе=0)

### Splits и cutoffs
```python
EXPAND_CUTOFFS = {
    "train":   "2013-01-01",
    "valid":   "2023-01-01",
    "test":    "2024-01-01",
    "forward": "2025-01-01",
}
EMBARGO_DAYS = 15  # = HORIZON
```

## Output структура

```
baseline_outputs_v22/      ← Канонические данные + v22 trades
  v22_full_data.csv         (главный input для v23, v24, v25)
  v22_*_trades.csv          (per-variant trades)
  v22_feature_importance.csv (permutation importance)
  v22_policy.json

baseline_outputs_v23/      ← Honest math
  v23_honest_pnl_summary.csv
  v23_bootstrap_ci.csv
  v23_dsr_psr.csv
  v23_realistic_costs_impact.csv
  v23_apples_to_apples_bnh.csv
  v23_performance_attribution.csv
  v23_feature_drift_train_vs_forward.csv
  v23_decision_audit_log.jsonl
  v23_cpcv_folds.json
  v23_paper_trading_log.csv     ← реальные исполненные ордера в Tinkoff
  v23_sandbox_account.json      ← sandbox account ID

baseline_outputs_v24/      ← Gate overlay results
  v24_pnl_summary.csv
  v24_risk_metrics.csv
  v24_bootstrap_ci.csv
  v24_gate_blocks_stats.csv
  v24_*_trades.csv
  v24_gated_decisions.csv
  v24_config.json

baseline_outputs_v25/      ← CPCV results (CURRENT)
  v25_pnl_summary.csv       ← ⭐ главные цифры
  v25_dsr_psr.csv
  v25_bootstrap_ci.csv
  v25_*_trades.csv
  v25_p_up_cpcv.csv
  v25_decisions.csv
  v25_policy.json
```
