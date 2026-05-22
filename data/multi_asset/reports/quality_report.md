# Multi-Asset Pipeline Quality Report

*Generated: 2026-05-21 21:31:26*


## 1. Metal data (yfinance)


| Metal | Ticker | Rows | First date | Last date | Years |
|---|---|---:|---|---|---:|
| silver | SI=F | 4,118 | 2010-01-04 | 2026-05-19 | 16.4 |
| gold | GC=F | 4,118 | 2010-01-04 | 2026-05-19 | 16.4 |
| platinum | PL=F | 4,117 | 2010-01-04 | 2026-05-19 | 16.4 |
| palladium | PA=F | 4,100 | 2010-01-04 | 2026-05-19 | 16.4 |
| copper | HG=F | 4,119 | 2010-01-04 | 2026-05-19 | 16.4 |

## 2. Macro data (FRED + yfinance)


| Series | Source | Rows | Frequency | Description |
|---|---|---:|---|---|
| DGS10 | fred | 4,097 | D | Nominal 10Y yield |
| DFII10 | fred | 4,097 | D | Real 10Y rate |
| T10YIE | fred | 4,098 | D | Inflation expectations |
| DTWEXBGS | fred | 4,076 | D | Trade-weighted dollar |
| USDRUB | yf | 4,263 | D | USD to Ruble |
| INDPRO | fred | 196 | M | US industrial production |
| VIXCLS | fred | 4,149 | D | Implied volatility S&P 500 |
| CPIAUCSL | fred | 195 | M | Consumer price index |
| DCOILWTICO | fred | 4,107 | D | Oil price spot |

## 3. Feature engineering


- **Total rows**: 4,118
- **Total columns (features)**: 105
- **Rows after dropna**: 3,110 (75.5% retention)
- **Period**: 2010-01-04 → 2026-05-19

### Feature breakdown
- Per-asset technical: 70
- Cross-asset ratios: 10
- Cross-asset correlations: 2
- Macro raw: 9
- Macro age_days: 9
- Composite/other: 2

## 4. Multi-horizon labels


- **Horizons**: [5, 10, 20, 60]
- **Total label rows**: 4,118

### Label distribution per horizon
| Horizon | TP (+1) | Timeout (0) | SL (−1) | NaN |
|---|---:|---:|---:|---:|
| 5 days | 1,724 (42%) | 779 (19%) | 1,595 (39%) | 20 |
| 10 days | 2,018 (49%) | 208 (5%) | 1,872 (45%) | 20 |
| 20 days | 2,110 (51%) | 29 (1%) | 1,959 (48%) | 20 |
| 60 days | 2,126 (52%) | 3 (0%) | 1,969 (48%) | 20 |

### Regime distribution
- **sideways**: 1,485 (36%)
- **uptrend**: 1,462 (36%)
- **downtrend**: 1,171 (28%)

## 5. Effective training size


- **Clean silver days**: 3,110
- **Metals available for backbone**: 5
- **Horizons**: 4
- **Effective supervision pairs (silver only)**: 12,440
- **Effective with cross-asset backbone (~×5)**: ~62,200

## 6. Summary


- Old setup: ~3 000 supervision pairs (silver daily, 2018-2025, 1 horizon)
- New setup: ~12,440 pairs (silver), ~62,200 with backbone
- **Effective data growth: ×21**