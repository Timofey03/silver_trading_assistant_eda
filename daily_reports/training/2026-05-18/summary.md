# Training report — 2026-05-18

**Run time (UTC)**: 2026-05-18T17:52:21.343889+00:00
**Model**: v25_cpcv

## Health check

| Метрика | Значение | Норма |
|---|---|---|
| Forward total return | 53.38% | > 0 |
| Forward Sharpe (ann.) | 1.03 | > 1.0 |
| Forward Max DD | -30.15% | > -25% |
| Forward DSR | 0.424 | > 0.7 |
| Forward PSR | 0.904 | > 0.95 |
| Bootstrap 95% lower | -33.04% | > 0 |
| Beats BnH | ❌ | True |
| N sequential trades | 15 | > 30 |

## Policy
```json
{
  "up_threshold": 0.45,
  "cooldown": 7,
  "valid_buys": 32,
  "valid_precision": 0.46875
}
```

## PnL Summary (compound equity, realistic costs)

| Split | v22 honest | v25 CPCV | Δ | BnH | vs BnH | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|---|---|---|---|
| valid | -20.74% | -3.26% | 17.48% | -0.04% | -3.22% | -3.33% | -18.35% | -0.022 |
| test | -24.37% | 22.25% | 46.62% | 21.33% | 0.92% | 22.33% | -3.44% | 1.333 |
| forward | 52.45% | 53.38% | 0.93% | 160.49% | -107.10% | 37.83% | -30.15% | 1.034 |

## Statistical robustness

- **valid**: Sharpe=-0.0071, PSR=0.492, DSR=0.4827 (n=9)
- **test**: Sharpe=0.4435, PSR=0.941, DSR=0.4091 (n=9)
- **forward**: Sharpe=0.3081, PSR=0.9037, DSR=0.424 (n=15)

## Feature drift (train vs последние 60 дней)

- Проверено фичей: **50**
- С drift (p<0.01): **45** (90%)
- Топ дрейфующих: silver_open, silver_high, silver_low, silver_close, gold_open, gold_high, gold_low, gold_close, vix_open, dxy_open