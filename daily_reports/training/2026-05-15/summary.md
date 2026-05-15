# Training report — 2026-05-15

**Run time (UTC)**: 2026-05-15T17:40:50.285233+00:00
**Model**: v25_cpcv

## Health check

| Метрика | Значение | Норма |
|---|---|---|
| Forward total return | 53.34% | > 0 |
| Forward Sharpe (ann.) | 1.71 | > 1.0 |
| Forward Max DD | -8.54% | > -25% |
| Forward DSR | 0.400 | > 0.7 |
| Forward PSR | 0.958 | > 0.95 |
| Bootstrap 95% lower | 19.15% | > 0 |
| Beats BnH | ❌ | True |
| N sequential trades | 7 | > 30 |

## Policy
```json
{
  "up_threshold": 0.55,
  "cooldown": 15,
  "valid_buys": 16,
  "valid_precision": 0.5
}
```

## PnL Summary (compound equity, realistic costs)

| Split | v22 honest | v25 CPCV | Δ | BnH | vs BnH | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|---|---|---|---|
| valid | -27.74% | 8.09% | 35.83% | -0.04% | 8.13% | 8.64% | -15.04% | 0.376 |
| test | -3.59% | 20.86% | 24.45% | 21.33% | -0.47% | 21.07% | -4.26% | 1.347 |
| forward | 175.29% | 53.34% | -121.95% | 187.39% | -134.05% | 39.11% | -8.54% | 1.712 |

## Statistical robustness

- **valid**: Sharpe=0.1215, PSR=0.6439, DSR=0.4784 (n=9)
- **test**: Sharpe=0.5474, PSR=0.9192, DSR=0.4185 (n=6)
- **forward**: Sharpe=0.7365, PSR=0.9583, DSR=0.3995 (n=7)

## Feature drift (train vs последние 60 дней)

- Проверено фичей: **50**
- С drift (p<0.01): **46** (92%)
- Топ дрейфующих: silver_open, silver_high, silver_low, silver_close, gold_open, gold_high, gold_low, gold_close, vix_open, dxy_open