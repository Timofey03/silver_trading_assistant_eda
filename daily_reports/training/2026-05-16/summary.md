# Training report — 2026-05-16

**Run time (UTC)**: 2026-05-16T13:02:23.093064+00:00
**Model**: v25_cpcv

## Health check

| Метрика | Значение | Норма |
|---|---|---|
| Forward total return | 111.95% | > 0 |
| Forward Sharpe (ann.) | 1.85 | > 1.0 |
| Forward Max DD | -8.54% | > -25% |
| Forward DSR | 0.337 | > 0.7 |
| Forward PSR | 0.998 | > 0.95 |
| Bootstrap 95% lower | 35.28% | > 0 |
| Beats BnH | ❌ | True |
| N sequential trades | 10 | > 30 |

## Policy
```json
{
  "up_threshold": 0.55,
  "cooldown": 5,
  "valid_buys": 38,
  "valid_precision": 0.47368421052631576
}
```

## PnL Summary (compound equity, realistic costs)

| Split | v22 honest | v25 CPCV | Δ | BnH | vs BnH | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|---|---|---|---|
| valid | -9.39% | 7.49% | 16.88% | -0.04% | 7.53% | 7.60% | -15.69% | 0.382 |
| test | 7.64% | 8.62% | 0.98% | 21.33% | -12.70% | 10.24% | -7.63% | 0.625 |
| forward | 79.72% | 111.95% | 32.23% | 186.65% | -74.70% | 77.54% | -8.54% | 1.854 |

## Statistical robustness

- **valid**: Sharpe=0.1263, PSR=0.6445, DSR=0.4783 (n=9)
- **test**: Sharpe=nan, PSR=nan, DSR=nan (n=3)
- **forward**: Sharpe=0.6708, PSR=0.9979, DSR=0.3369 (n=10)

## Feature drift (train vs последние 60 дней)

- Проверено фичей: **50**
- С drift (p<0.01): **45** (90%)
- Топ дрейфующих: silver_open, silver_high, silver_low, silver_close, gold_open, gold_high, gold_low, gold_close, vix_open, dxy_open