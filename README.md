# Silver Trading Assistant

ML-помощник для торговли серебром (SI=F / SLVRUBF) с полным циклом: feature engineering → ML signals → risk-aware execution → paper trading через Tinkoff Invest API.

**Финальная модель — E3b** (multi-asset cross-asset + adaptive barriers + feature selection через mutual information). Sharpe **0.530**, Win Rate **69%**, накопленная доходность **+114.7%** на walk-forward за 10.3 года (2015–2025). Интегрирована в production-инфраструктуру с автоматическим переобучением через GitHub Actions 3 раза в день и Telegram-уведомлениями с дедупликацией action/info.

> ⚠️ Это исследовательский проект. Не финансовый совет. Все цифры backtest в реальных деньгах могут быть значительно ниже из-за market impact, slippage и режимных сдвигов. Перед реальными вложениями — минимум 6 месяцев paper trading.

---

## 🚀 Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt
pip install lightgbm catboost yfinance pyarrow seaborn  # для E3b multi-asset pipeline

# 2. Создать .env (см. .env.example):
#    TINKOFF_TOKEN=...     — sandbox-only (https://www.tinkoff.ru/invest/settings)
#    TG_BOT_TOKEN=...      — для Telegram-уведомлений (опционально)
#    TG_CHAT_ID=...

# 3. Запустить ОДНО из двух Streamlit-приложений (см. ниже)
```

---

## 📱 Два Streamlit-приложения

В проекте **два независимых веб-интерфейса** для разных пользователей:

### A. `dashboard_app.py` — полнофункциональная панель (профессиональная)

Для торгующих через Tinkoff с подробной аналитикой и paper trading.

```bash
streamlit run dashboard_app.py
# → http://localhost:8501
```

| Экран | Что показывает |
|---|---|
| 🏠 **Главная** | Карточка сигнала E3b с метой источника, KPI, equity curve |
| 💰 **Портфель** | Live баланс Tinkoff, donut, открытые позиции, история ордеров |
| 📊 **Сигналы** | Win rate, P&L каждой сделки, фильтры |
| 📈 **Графики** | Candlestick + signals overlay + drawdown |
| 🤖 **Модель** | DSR/PSR/Sharpe, bootstrap CI, drift detection |
| ⚙ **Настройки** | Tinkoff conn, расписание, гейты, Telegram alerts |
| 🧮 **Калькулятор** | Расчёт лотов от капитала |

### B. `simple_app.py` — облегчённая версия (для конечного пользователя и защиты диплома)

Минимум технических терминов, акцент на действиях. Подходит для непрофессионалов и презентации проекта.

```bash
streamlit run simple_app.py --server.port 8502
# → http://localhost:8502
```

| Экран | Что показывает |
|---|---|
| 📍 **Сейчас** | Что делать сейчас (BUY/HOLD/SELL) с метками E3b и action/info |
| 💼 **Мои сделки** | Локальная история + бэктест (E3b / V25 / базовая WF) |
| 🧮 **Калькулятор** | Доходность по моделям с реальной статистикой |
| 📊 **Как работал** | 3 блока: E3b (winner) + V25 + базовая walk-forward |
| 🔬 **Эволюция модели** | Все 6 экспериментов E1-E4 с графиками |
| ⚙ **Настройки** | Локальный капитал + Telegram credentials |

### Запуск обоих параллельно

```bash
# Терминал 1:
streamlit run dashboard_app.py                    # → :8501

# Терминал 2:
streamlit run simple_app.py --server.port 8502    # → :8502
```

**Деплой в облако**: см. [docs/HOSTING.md](docs/HOSTING.md) → Streamlit Cloud (бесплатно, 5 минут).

---

## 🛠 CLI инструменты

### Production daily run (E3b — финальная модель)

```bash
# Полный цикл: refresh data → retrain → signal → Telegram уведомление
python scripts/daily_e3b.py

# Только инференс без переобучения (быстрый)
python scripts/daily_e3b.py --skip-training

# Тест Telegram-нотификации (использует latest signal)
python scripts/test_telegram_e3b.py
```

### Multi-asset pipeline (база E3b)

```bash
# Refresh всех данных (5 металлов + 9 макроиндикаторов FRED)
python multiasset_pipeline.py --refresh

# Запустить все эксперименты диплома
python experiments/e1_baseline.py            # E1: silver-only baseline
python experiments/e2b_feature_selected.py   # E2b: + cross-asset + feature selection
python experiments/e3_macro_adaptive.py      # E3a/b/c: + macro + adaptive barriers
python experiments/e4_stacking.py            # E4: stacking ensemble
python experiments/forward_test_2025.py      # OOS forward test 2025-2026
python experiments/compare_v25.py            # E3b vs V25 сравнение
python experiments/visualize.py              # 7 PNG для диплома (главные)
python experiments/visualize_competitors.py  # 3 PNG vs конкуренты
python experiments/visualize_metrics_honest.py # 5 PNG расширенных метрик
```

### Legacy V25 (для совместимости)

```bash
python silver_assistant_v25_cpcv.py          # переобучение V25
python scripts/daily_run.py                  # daily run V25
python silver_paper_tinkoff.py --live --ticker SLVRUBF  # Paper trading
```

### Автоматизация в облаке

Два workflow в `.github/workflows/`:
- `daily_e3b.yml` — **E3b** (3× в день: 08:00, 14:00, 22:00 МСК) с дедупликацией Telegram (action vs info)
- `daily.yml` — V25 legacy (тот же график)

---

## 📊 Текущие результаты — финальная модель E3b

### Walk-forward валидация 2015–2025 (10.3 года, 48 сделок)

| Метрика | Значение | Комментарий |
|---|---:|---|
| **Total return** | **+114.7%** | За 10.3 года |
| CAGR | +7.7% годовых | Compound annual growth rate |
| **Sharpe Ratio** | **0.530** | Annualized |
| Sortino | 1.002 | Downside-adjusted |
| Max Drawdown | −17.9% | Максимальная просадка |
| Profit Factor | 2.12 | $2.12 заработано на каждый $1 потерянный |
| **Win Rate** | **68.8%** | 33 прибыльных из 48 |
| OOS Accuracy | 55.5% | На out-of-sample предсказаниях |
| PSR | 1.000 | Вероятность что истинный Sharpe > 0 |

### Forward test 2025–2026 (out-of-sample, 16 месяцев)

Модель обучена **только на данных до конца 2024**, протестирована на 2025–2026:

| Метрика | Значение |
|---|---:|
| Sharpe Ratio | **2.173** |
| Win Rate | **83.3%** (5 из 6) |
| Max Drawdown | **−0.15%** |
| Profit Factor | 116.3 |
| Сделок за 16 мес | 6 (селективно) |
| Total return | +18.4% |

### Прогрессия экспериментов (E1 → E4)

| Эксп. | Что добавлено | Sharpe | Annual | Max DD |
|---|---|---:|---:|---:|
| E1 | Silver-only baseline (14 фичей) | 0.459 | +4.6% | −24.5% |
| E2 | Naive cross-asset (84 фичи, без отбора) | −0.248 ❌ | −3.9% | −51.1% |
| E2b | + Feature selection (mutual info top-25) | 0.580 | +5.9% | −18.3% |
| E3a | + 9 макрофичей (TIPS, DXY, VIX, …) | 0.424 ❌ | +5.5% | −31.3% |
| **E3b** | **+ Adaptive volatility-scaled barriers** ★ | **0.530** | **+7.7%** | **−17.9%** |
| E4 | Stacking ensemble (HistGB+LGBM+CatBoost) | 0.194 ❌ | +1.9% | −36.5% |

**5 научных находок документированы**: curse of dimensionality, feature selection как необходимое условие, ограничения macro в forward-fill, эффективность adaptive barriers, провал stacking на 1k samples per fold.

Подробное сравнение с конкурентами (AQR Managed Futures, SLV ETF, Renaissance Medallion) — [`docs/COMPETITORS_METRICS_HONEST.md`](docs/COMPETITORS_METRICS_HONEST.md).

---

## 🗂 Структура проекта

```
silver_trading_assistant_eda/
├── README.md                              ← вы здесь
├── requirements.txt
├── .env.example                           ← шаблон credentials
│
├── 🏆 ФИНАЛЬНАЯ МОДЕЛЬ E3b ───────────────────────────────────────────────
│
├── app/multi_asset/                       ← Multi-asset pipeline (Phase 1)
│   ├── config.py                          ← METALS, MACRO, paths
│   ├── metal_loader.py                    ← yfinance × 5 металлов
│   ├── macro_loader.py                    ← FRED × 9 + yfinance USDRUB
│   ├── features.py                        ← 105 фичей (5 металлов + cross-asset + macro)
│   ├── labels.py                          ← Triple-barrier + adaptive barriers
│   ├── walkforward.py                     ← WF engine + purging + embargo
│   ├── simulator.py                       ← Trade execution
│   ├── metrics.py                         ← Sharpe, DSR, PSR, Sortino, Calmar
│   └── quality_report.py
│
├── experiments/                            ← Дипломные эксперименты E1-E4
│   ├── e1_baseline.py                     ← E1: Sharpe 0.459
│   ├── e2b_feature_selected.py            ← E2b: Sharpe 0.580
│   ├── e3_macro_adaptive.py               ← E3a/b/c: WINNER E3b Sharpe 0.530
│   ├── e4_stacking.py                     ← E4: Sharpe 0.194 (overfit)
│   ├── forward_test_2025.py               ← OOS forward 2025-26 (Sharpe 2.17)
│   ├── compare_v25.py                     ← E3b vs V25 comparison
│   ├── visualize.py                       ← Главные 7 PNG для диплома
│   ├── visualize_competitors.py           ← vs AQR / Medallion / SLV B&H
│   └── visualize_metrics_honest.py        ← Calmar, Sortino, skewness
│
├── data/multi_asset/                       ← Кеш данных + графики
│   ├── metals/                            ← 5 parquet (silver, gold, …)
│   ├── macro/                             ← 9 parquet (FRED + USDRUB)
│   ├── features/silver_features.parquet   ← 105 cols × 4118 rows
│   ├── labels/silver_labels.parquet       ← 4 horizons (5/10/20/60 дней)
│   ├── reports/quality_report.md
│   └── figures/                           ← 17 PNG графиков для диплома
│
├── baseline_outputs_multiasset/            ← Результаты всех 6 экспериментов
│   ├── e1_baseline/                       ← trades.csv, predictions.parquet, metrics.json
│   ├── e2_cross_asset/                    ← E2 naive (negative result)
│   ├── e2b_feature_selected/              ← E2b + FS
│   ├── e3a_macro/                         ← E3a + macro (negative result)
│   ├── e3b_adaptive/                      ← 🏆 WINNER
│   ├── e3c_metalabel/                     ← E3c + meta-labeling
│   ├── e4_stacking/                       ← E4 stacking (negative result)
│   ├── forward_test_2025/                 ← OOS 2025-26 with 6 trades
│   ├── forward_grid_search/               ← Cooldown/threshold sensitivity
│   ├── e3b_cool20_honest/                 ← Honest cooldown check
│   ├── comparison_v25/                    ← E3b vs V25 detailed
│   └── competitors_metrics.json           ← Метрики AQR/SLV/Medallion
│
├── scripts/
│   ├── daily_e3b.py                       ← 🏆 Production daily E3b run
│   ├── test_telegram_e3b.py               ← Test Telegram notification
│   ├── daily_run.py                       ← Legacy V25 daily
│   └── telegram_setup.py                  ← Bot setup
│
├── .github/workflows/
│   ├── daily_e3b.yml                      ← 🏆 E3b automation (3× в день)
│   └── daily.yml                          ← Legacy V25 automation
│
├── daily_reports/e3b/                      ← E3b daily training + trading reports
│
├── 📱 STREAMLIT ПРИЛОЖЕНИЯ ────────────────────────────────────────────────
│
├── dashboard_app.py                       ← 🏠 Professional dashboard (:8501)
├── pages/                                  ← Страницы dashboard_app
│   ├── 1_💰_Портфель.py
│   ├── 2_📊_Сигналы.py
│   ├── 3_📈_Графики.py
│   ├── 4_🤖_Модель.py
│   ├── 5_⚙_Настройки.py
│   └── 6_🧮_Калькулятор.py
│
├── simple_app.py                          ← Simple version для пользователя (:8502)
├── simple_pages/
│   ├── 1_now.py                           ← 📍 Сейчас (E3b сигнал + action/info)
│   ├── 2_trades.py                        ← 💼 Мои сделки + 3 модели
│   ├── 3_calculator.py                    ← 🧮 Калькулятор по 3 моделям
│   ├── 4_stats.py                         ← 📊 Как работал (3 блока)
│   ├── 5_settings.py                      ← ⚙ Настройки + Telegram
│   └── 6_evolution.py                     ← 🔬 Эволюция модели E1-E4
│
├── app/                                    ← Общие utils для обоих apps
│   ├── utils.py                           ← Cached loaders, signal sources priority
│   ├── charts.py                          ← Plotly helpers
│   ├── simple_storage.py                  ← Local storage (~/.silver_simple/)
│   └── notifier.py                        ← Telegram notifier
│
├── 🔧 LEGACY V25 (для совместимости) ─────────────────────────────────────
│
├── silver_assistant_v25_cpcv.py           ← V25 CPCV модель
├── silver_assistant_v14_main.py → v24     ← История версий v14-v24
├── silver_paper_tinkoff.py                ← Tinkoff REST API paper trading
├── silver_signal_modes.py                 ← Signal generation modes
├── silver_walkforward_backtest.py         ← Legacy walk-forward
├── silver_production_inference.py         ← V25 production inference
├── silver_features.py, silver_data_loader.py  ← Legacy data layer
│
├── baseline_outputs_v22/, v23/, v24/, v25/ ← Legacy V25 results
├── baseline_outputs_walkforward/          ← Legacy walk-forward (74 trades, -37%)
│
├── docs/                                   ← 📚 Документация
│   ├── ВКР_Silver_Trading_Assistant.docx  ← Дипломная работа
│   ├── THESIS_FULL.md                     ← Markdown исходник диплома
│   ├── DEFENSE_QA.md                      ← Q&A для защиты
│   ├── COMPETITORS_METRICS_HONEST.md      ← 🎯 Анализ метрик vs конкуренты
│   ├── COMPETITORS_DEEP_ANALYSIS.md       ← Сравнение с рынком
│   ├── PROJECT_SUMMARY_AND_LIMITATIONS.md ← Итог + слабые места
│   ├── ML_ATTRIBUTION.md                  ← ML attribution analysis
│   ├── PRESENTATION_SUMMARY.md            ← Облегчённая для защиты
│   ├── Приложение_Г_Презентация.docx      ← Презентационный appendix
│   ├── ARCHITECTURE.md, RESULTS.md, HOSTING.md, PAPER_TRADING.md
│
└── _archive/                              ← Старые ноутбуки и outputs v1-v13
```

---

## 🔬 Архитектура финальной модели E3b

**Финальный pipeline** (за весь walk-forward fold):

```
[1] Multi-asset data
    yfinance × 5 металлов (silver / gold / platinum / palladium / copper)
    FRED × 9 macro (TIPS, DXY, breakeven, VIX, INDPRO, CPI, oil, USDRUB)
    Период 2010–2026 (16.4 года, 4118 торговых дней)
       ↓
[2] Feature engineering
    Per-asset (RSI/ADX/ATR/MA/momentum) × 5 = 70 фичей
    Cross-asset ratios + correlations = 12 фичей
    Macro + age_days = 18 фичей
    Composite metals index = 5 фичей
    ИТОГО: 105 фичей × 3110 чистых наблюдений
       ↓
[3] Multi-horizon labels (triple-barrier López de Prado)
    Adaptive volatility-scaled barriers (vol-scaled TP/SL)
    Regime-aware asymmetric (uptrend: TP>SL, downtrend: TP<SL)
    Horizons: 5, 10, 20, 60 дней (financial supervision ×4)
       ↓
[4] Walk-forward с purging
    Train window: 1000 дней (sliding)
    Test window: 30 дней
    Step: 30 дней
    Purge: 20 дней + embargo 1 день
       ↓
[5] Feature selection (на каждом fold)
    SelectKBest with mutual_info_classif
    Top-30 из 105 фичей
       ↓
[6] Model training
    HistGradientBoostingClassifier
    max_depth=6, learning_rate=0.05, max_iter=200
       ↓
[7] Signal generation
    BUY:  p_up ≥ 0.48
    SELL: p_up < 0.35
    HOLD: иначе
       ↓
[8] Execution mechanics
    Trailing stop 12%
    Max hold: 30 торговых дней
    Cooldown: 25 дней между сделками
       ↓
[9] Telegram + Streamlit
    Дедупликация action vs info
    Бейдж source в UI
```

---

## ⚙ Daily production workflow

```
[Каждый рабочий день 08:00 / 14:00 / 22:00 МСК через GitHub Actions]

  1. Refresh data
     ├─ yfinance: 5 металлов
     └─ FRED: 9 macro indicators
       ↓
  2. Walk-forward retraining E3b
     ├─ Обновляет baseline_outputs_multiasset/e3b_adaptive/
     ├─ Sharpe / Win / Max DD пересчитываются
       ↓
  3. Production inference
     ├─ features с ffill_limit=5 (заполняет gaps в палладии)
     ├─ Train на ВСЕХ данных до вчерашнего дня (purge 21 день)
     ├─ Predict p_up на сегодня
       ↓
  4. Signal classification
     ├─ p_up ≥ 0.48 → BUY
     ├─ p_up < 0.35 → SELL
     ├─ иначе → HOLD
       ↓
  5. Дедупликация
     ├─ Сравнение с предыдущим signal
     ├─ alert_type: action | info
       ↓
  6. Save reports
     ├─ daily_reports/e3b/trading/YYYY-MM-DD/signal.json (latest)
     └─ daily_reports/e3b/trading/YYYY-MM-DD/signal_HHMMSS.json (история)
       ↓
  7. Telegram уведомление
     ├─ Action: "📢 НОВЫЙ СИГНАЛ: HOLD → BUY"
     └─ Info:   "ℹ Сигнал не изменился"
       ↓
  8. Git commit обратно в репо
     └─ daily_reports/ + обновлённые parquet файлы
```

---

## 🎯 Roadmap

- [x] Phase 1: Multi-asset data pipeline (5 металлов + 9 macro, 16 лет)
- [x] Phase 2: Walk-forward engine + Trade simulator + Metrics module
- [x] Phase 3: E1 baseline reproduction
- [x] Phase 4: E2/E2b cross-asset + feature selection
- [x] Phase 5: E3a/b/c macro + adaptive barriers + meta-labeling
- [x] Phase 6: E4 stacking ensemble (negative result)
- [x] Phase 7: Forward test 2025-2026 (Sharpe 2.17 OOS)
- [x] Phase 8: Production integration (Telegram + GitHub Actions + Streamlit)
- [x] Phase 9: Дедупликация intraday сигналов (action vs info)
- [x] Дипломная работа (60-80 страниц + 17 PNG графиков)
- [ ] **Live forward validation** (6 месяцев daily live mode с real-money tracking)
- [ ] **Volatility targeting position sizing** (для улучшения Calmar Ratio с 0.25 до 1.5+)
- [ ] **Drawdown circuit-breaker** (заморозить новые сделки если DD > 8%)
- [ ] **Adaptive holding period** (убрать max_hold для let-winners-run, fix отрицательный skewness)
- [ ] **Online learning** через River library для OOD-адаптации (E3b даёт mean p_up 0.28 в bull market 2025)
- [ ] **Multi-horizon multi-task** обучение на 4 horizons одновременно
- [ ] **NLP sentiment features** через FinBERT на финансовых новостях
- [ ] **Hidden Markov Models** для regime detection (макро-режимы 4 состояний)

---

## 📚 Документация

### Дипломная работа

- [docs/ВКР_Silver_Trading_Assistant.docx](docs/ВКР_Silver_Trading_Assistant.docx) — финальная дипломная работа (60-80 страниц, 6 встроенных графиков)
- [docs/THESIS_FULL.md](docs/THESIS_FULL.md) — markdown исходник для редактирования
- [docs/DEFENSE_QA.md](docs/DEFENSE_QA.md) — Q&A для защиты (14 типовых вопросов + 5 каверзных)

### Анализы

- [docs/COMPETITORS_METRICS_HONEST.md](docs/COMPETITORS_METRICS_HONEST.md) — **главный** анализ vs AQR/SLV/Medallion с честными цифрами (Calmar, Sortino, skewness)
- [docs/COMPETITORS_DEEP_ANALYSIS.md](docs/COMPETITORS_DEEP_ANALYSIS.md) — обзор рынка конкурентов
- [docs/PROJECT_SUMMARY_AND_LIMITATIONS.md](docs/PROJECT_SUMMARY_AND_LIMITATIONS.md) — итог + 25 слабых мест проекта
- [docs/ML_ATTRIBUTION.md](docs/ML_ATTRIBUTION.md) — ML attribution analysis vs random baseline
- [docs/PRESENTATION_SUMMARY.md](docs/PRESENTATION_SUMMARY.md) — облегчённая презентация для непрофессионалов

### Технические

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — детали pipeline + где какой код
- [docs/RESULTS.md](docs/RESULTS.md) — все результаты экспериментов в табличном виде
- [docs/HOSTING.md](docs/HOSTING.md) — деплой в Streamlit Cloud
- [docs/PAPER_TRADING.md](docs/PAPER_TRADING.md) — гайд по Tinkoff bridge

### Презентационные

- [docs/Приложение_Г_Презентация.docx](docs/Приложение_Г_Презентация.docx) — appendix к диплому для защиты

---

## 🏆 Конкурентное позиционирование

| Стратегия (период 2018-2025) | CAGR | Sharpe | Calmar | Max DD | Hit rate |
|---|---:|---:|---:|---:|---:|
| **E3b (наша модель)** | **+3.5%** | **0.429** | 0.25 | −14.1% | **75%** |
| SLV B&H ETF | +19.0% | 0.536 | 1.53 | −12.5% | 62% |
| AQR Managed Futures | +4.6% | 0.404 | **3.12** | **−1.5%** | 57% |
| Per-commodity trend (industry avg) | ~+5% | 0.15-0.20 | — | — | — |
| SG Trend Index (2000-2023) | — | 0.42 | — | — | — |
| Renaissance Medallion (недосягаемый) | +37.9% | **2.192** | ∞ | 0.0% | **100%** |

**Где E3b выигрывает**: Sharpe выше AQR Managed Futures ($3B AUM), hit rate 75% выше всех, защита в боковых периодах (2018, 2023).

**Где E3b проигрывает**: Calmar Ratio в 12× хуже AQR — серьёзный недостаток риск-менеджмента. CAGR ниже AQR на 1.1 п.п. Отрицательный skewness (−0.53) — следствие архитектуры stop-loss.

Полное сравнение с цифрами — [`docs/COMPETITORS_METRICS_HONEST.md`](docs/COMPETITORS_METRICS_HONEST.md).
