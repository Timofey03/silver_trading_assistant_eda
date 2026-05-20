# Training report — 2026-05-20

**Run time (UTC)**: 2026-05-20T21:00:39.035675+00:00
**Model**: v25_cpcv

## Health check

| Метрика | Значение | Норма |
|---|---|---|
| Forward total return | 94.55% | > 0 |
| Forward Sharpe (ann.) | 1.42 | > 1.0 |
| Forward Max DD | -21.41% | > -25% |
| Forward DSR | 0.392 | > 0.7 |
| Forward PSR | 0.969 | > 0.95 |
| Bootstrap 95% lower | 2.08% | > 0 |
| Beats BnH | ❌ | True |
| N sequential trades | 11 | > 30 |

## Policy
```json
{
  "up_threshold": 0.5,
  "cooldown": 7,
  "valid_buys": 32,
  "valid_precision": 0.46875
}
```

## PnL Summary (compound equity, realistic costs)

| Split | v22 honest | v25 CPCV | Δ | BnH | vs BnH | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|---|---|---|---|
| valid | -40.62% | -3.49% | 37.12% | -0.04% | -3.46% | -3.53% | -20.32% | 0.025 |
| test | 6.08% | 32.77% | 26.69% | 21.33% | 11.44% | 39.80% | -1.76% | 2.072 |
| forward | 50.40% | 94.55% | 44.15% | 161.79% | -67.24% | 67.18% | -21.41% | 1.417 |

## Statistical robustness

- **valid**: Sharpe=0.0078, PSR=0.5094, DSR=0.4985 (n=10)
- **test**: Sharpe=0.7781, PSR=0.9884, DSR=0.3692 (n=6)
- **forward**: Sharpe=0.4862, PSR=0.9692, DSR=0.3917 (n=11)

## Feature drift (train vs последние 60 дней)

- Проверено фичей: **50**
- С drift (p<0.01): **46** (92%)
- Топ дрейфующих: silver_open, silver_high, silver_low, silver_close, gold_open, gold_high, gold_low, gold_close, vix_open, dxy_open