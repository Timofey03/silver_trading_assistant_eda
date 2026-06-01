# Silver Trading Assistant

ML-помощник для торговли серебром (SI=F / SLVRUBF) с полным циклом: feature engineering → ML signals → risk-aware execution → paper trading через Tinkoff Invest API.

**Финальная модель — E3b** (multi-asset cross-asset + adaptive volatility-scaled barriers + ансамбль из 3 моделей градиентного бустинга + smoothed strong-signal filter). На walk-forward за **11,2 года** (2015 — Q2 2026, 90 сделок): Sharpe **0,994**, накопленная доходность **+343,5 %**, годовая **+14,3 %**, Win Rate **64,4 %**, Max DD **−25,7 %**, Profit Factor **3,31**. Полная production-инфраструктура: автоматическое переобучение E3b 3 раза в день через GitHub Actions, веб-интерфейс Argentum (FastAPI + Next.js 16) с интеграцией Tinkoff sandbox, Telegram-уведомления с дедупликацией action/info и графиком накопленной P&L.

> ⚠️ Это исследовательский проект и материал для дипломной работы. Не финансовый совет. Backtest-цифры в реальных деньгах могут быть значительно ниже из-за market impact, slippage и режимных сдвигов. Перед реальными вложениями — минимум 6 месяцев paper trading через Tinkoff sandbox.

---

## 🚀 Быстрый старт

```bash
# 1. Установить Python-зависимости
pip install -r requirements.txt
pip install lightgbm catboost yfinance pyarrow seaborn

# 2. Создать .env в корне (см. .env.example):
#    TINKOFF_TOKEN=...   — sandbox-токен (https://www.tbank.ru/invest/settings/api/)
#    TG_BOT_TOKEN=...    — для Telegram-уведомлений (опционально)
#    TG_CHAT_ID=...

# 3. Запустить веб-интерфейс Argentum (см. ниже)
#    или Streamlit-приложения (legacy, для разработки)
```

---

## 🖥 Веб-интерфейс Argentum (production UI)

**FastAPI backend + Next.js 16 frontend** поверх E3b модели. Заменяет старый Streamlit-интерфейс для конечного пользователя.

### Запуск одной командой (Windows)

```bash
cd argentum
start_all.bat       # поднимает backend на :8000 и frontend на :3000
```

### Запуск вручную (двумя терминалами)

```bash
# Терминал 1 — Backend (FastAPI на :8000)
cd argentum/backend
uvicorn main:app --reload --port 8000
# → Swagger docs: http://127.0.0.1:8000/docs

# Терминал 2 — Frontend (Next.js на :3000)
cd argentum/frontend
npm install
npm run dev
# → http://127.0.0.1:3000
```

### Страницы UI

| URL | Назначение |
|---|---|
| `/` | **Сейчас**: hero-блок BUY/HOLD/SELL с p_up, цена лота SLVRUBF в ₽, мини-график 30 дней с сделками, кнопка «Купить через Tinkoff» |
| `/positions` | **Позиции**: открытые сделки с master signal (BUY/WAIT/AVOID), per-position SELL advisor, нереализованная P&L в ₽ |
| `/history` | **История**: свечной график (TradingView LWC) со всеми сделками, equity curve vs Buy-and-Hold, monthly heatmap, period-aware метрики |
| `/methodology` | **Методология**: описание E3b архитектуры, текущие параметры стратегии, эволюция моделей E1→E3b |
| `/settings` | **Настройки**: подключение Tinkoff, тема (light/dark), TTL кэша |

### REST API (16 endpoints)

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/signal` | Текущий E3b сигнал + p_up + meta |
| GET | `/api/price` | Текущая цена серебра (SI=F) + USDRUB + sparkline |
| GET | `/api/positions` | Открытые позиции + master signal + per-position advisor |
| GET | `/api/positions-closed` | История закрытых сделок |
| GET | `/api/positions-sync` | Синхронизация с Tinkoff sandbox portfolio |
| GET | `/api/history` | Equity curve + список 90 сделок E3b |
| GET | `/api/metrics?period=...` | Period-aware метрики (1Y/3Y/5Y/ALL) |
| GET | `/api/candles` | OHLC свечи + BUY/SELL/OPEN markers |
| GET | `/api/explain` | Топ-признаки «почему BUY?» |
| GET | `/api/evolution` | Прогрессия E1→E3b для страницы методологии |
| GET | `/api/fx` | Курс USDRUB с fallback |
| GET | `/api/tinkoff/balance` | Баланс Tinkoff sandbox |
| GET | `/api/tinkoff/order` | Подготовка ордера |
| GET | `/api/monthly` | Помесячная доходность (heatmap) |
| GET | `/api/equity` | Equity curve E3b vs Buy-and-Hold |
| GET | `/api/health` | Liveness check |

Persistent cache + TTL для тяжёлых эндпоинтов (30–300 с). SQLite-хранение позиций. OOD-детектор интегрирован в `/api/signal` (warning при extreme features).

---

## 📱 Streamlit-приложения (legacy / для разработки)

Два независимых веб-интерфейса остались для разработки и для демонстрации:

### `dashboard_app.py` — профессиональная панель (порт 8501)

```bash
streamlit run dashboard_app.py
```

| Экран | Что показывает |
|---|---|
| 🏠 Главная | Карточка сигнала E3b, KPI, equity curve |
| 💰 Портфель | Live баланс Tinkoff, donut, открытые позиции |
| 📊 Сигналы | Win rate, P&L каждой сделки |
| 📈 Графики | Candlestick + signals overlay + drawdown |
| 🤖 Модель | DSR / PSR / Sharpe, bootstrap CI, drift detection |
| ⚙ Настройки | Tinkoff, Telegram alerts |
| 🧮 Калькулятор | Расчёт лотов от капитала |

### `simple_app.py` — облегчённая версия (порт 8502)

```bash
streamlit run simple_app.py --server.port 8502
```

Минимум технических терминов, акцент на действиях. 6 страниц: Сейчас / Мои сделки / Калькулятор / Как работал / Эволюция модели / Настройки.

---

## 🛠 CLI инструменты

### Production daily run (E3b)

```bash
# Полный цикл: refresh data → retrain walk-forward → signal → Telegram
python scripts/daily_e3b.py

# Только инференс без переобучения (быстрый)
python scripts/daily_e3b.py --skip-training

# Тест Telegram-нотификации (использует latest signal)
python scripts/test_telegram_e3b.py
```

### Multi-asset pipeline (база E3b)

```bash
# Refresh всех данных (5 металлов через yfinance + 9 макроиндикаторов FRED)
python multiasset_pipeline.py --refresh

# Запустить все эксперименты диплома
python experiments/e1_baseline.py            # E1: silver-only baseline
python experiments/e2b_feature_selected.py   # E2b: + cross-asset + feature selection
python experiments/e3_macro_adaptive.py      # E3a/b/c: + macro + adaptive barriers (E3b — winner)
python experiments/e4_stacking.py            # E4: stacking ensemble (negative result)
python experiments/compare_v25.py            # E3b vs V25 сравнение
```

### Аудит согласованности UI ↔ Telegram

```bash
# Сверяет данные между Argentum API и Telegram-нотификацией
python scripts/audit_assistant.py
```

---

## 🤖 Автоматизация (GitHub Actions)

Три активных workflow в `.github/workflows/`:

| Файл | Расписание | Что делает |
|---|---|---|
| `daily_e3b.yml` | 08:00 / 14:00 / 22:00 МСК (пн–пт) | Refresh data → retrain walk-forward E3b → production inference → Telegram уведомление с дедупликацией action/info |
| `weekly_backfill.yml` | Воскресенье 22:00 UTC | Полный backfill walk-forward на свежих данных + apply optimal config |
| `test.yml` | На каждый PR и push в main | pytest (17 тестов: look-ahead bias, simulator, FX, etc.) |

**Важно**: для работы Tinkoff sandbox fallback в Telegram-уведомлениях в env workflow должна быть переменная `TINKOFF_TOKEN: ${{ secrets.TINKOFF_TOKEN }}` помимо TG_*. Без неё помощник в cron-режиме покажет нулевые позиции.

---

## 📊 Финальная модель E3b — метрики

### Walk-forward валидация 2015–2026 (11,2 года, 90 сделок)

| Метрика | Значение | Комментарий |
|---|---:|---|
| **Total return** | **+343,5 %** | За 11,2 года реальных торгов |
| **CAGR (Annual return)** | **+14,3 %** | Compound annual growth rate |
| **Sharpe Ratio** | **0,994** | Annualized, граница «промышленного» уровня |
| Sortino | 1,633 | Downside-adjusted |
| **Max Drawdown** | **−25,7 %** | Максимальная просадка |
| **Profit Factor** | **3,31** | $3,31 заработано на каждый $1 потерянный |
| **Win Rate** | **64,4 %** | 58 прибыльных из 90 |
| Best trade | +28,7 % | Лучшая сделка |
| Worst trade | −17,0 % | Худшая сделка |
| Trades per year | 8,06 | Селективность |
| PSR / DSR | 1,000 / 0,002 | Probabilistic / Deflated Sharpe |

### Распределение по годам

| Год | Сделок | Доходность за год | Накопленная |
|---|---:|---:|---:|
| 2015 | 6 | +18,8 % | +18,8 % |
| 2016 | 7 | +22,5 % | +45,5 % |
| 2017 | 6 | +3,2 % | +50,1 % |
| 2018 | 9 | +10,6 % | +66,0 % |
| 2019 | 11 | +0,4 % | +66,6 % |
| 2020 | 8 | +50,7 % | +151,0 % |
| 2021 | 9 | +11,4 % | +179,5 % |
| 2022 | 6 | −22,9 % | +115,4 % |
| 2023 | 8 | +14,8 % | +147,3 % |
| 2024 | 6 | +33,8 % | +231,0 % |
| 2025 | 10 | +3,0 % | +240,8 % |
| 2026 (Q2) | 4 | +30,1 % | **+343,5 %** |

### Прогрессия экспериментов E1 → E3c

| Эксп. | Что добавлено | Sharpe | Annual | Max DD | Win |
|---|---|---:|---:|---:|---:|
| E1 | Silver-only baseline (14 фичей) | 0,459 | +4,6 % | −24,5 % | 60,0 % |
| E2 | Naive cross-asset (84 фичи, без отбора) | −0,248 ❌ | −3,9 % | −51,1 % | 57,7 % |
| E2b | + Feature selection (mutual_info top-25) | 0,580 | +5,9 % | −18,3 % | 67,9 % |
| E3a | + 9 макрофичей (TIPS, DXY, VIX, INDPRO, CPI…) | 0,424 ❌ | +5,5 % | −31,3 % | 60,0 % |
| **E3b** | **+ adaptive volatility-scaled barriers + ансамбль 3 моделей** ★ | **0,994** | **+14,3 %** | **−25,7 %** | **64,4 %** |
| E3c | + meta-labeling LogisticRegression | 0,530 | +7,7 % | −17,9 % | 68,8 % |
| E4 | Stacking ensemble (HistGB + LGBM + CatBoost) | 0,194 ❌ | +1,9 % | −36,5 % | 50,0 % |

**Победитель — E3b** благодаря trifecta: ансамблю + adaptive barriers + smoothed strong-signal filter (порог сильного сигнала 0,85, smoothing окно 3 дня). Stacking (E4) и meta-labeling (E3c) дали худшие результаты из-за overfit на малых fold-выборках.

**Топ-10 признаков финальной модели** (частота отбора в 94 фолдах walk-forward):

| # | Признак | Частота | Интерпретация |
|---|---|---:|---|
| 1 | `silver_rvol_60` | 100 % | Долгосрочная волатильность серебра |
| 2 | `gold_rvol_60` | 100 % | Долгосрочная волатильность золота |
| 3 | `platinum_rvol_60` | 100 % | Долгосрочная волатильность платины |
| 4 | `palladium_rvol_60` | 100 % | Долгосрочная волатильность палладия |
| 5 | `palladium_vol_z` | 100 % | Z-score объёма палладия |
| 6 | `target_close` | 100 % | Дневная цена закрытия серебра |
| 7 | `platinum_vol_z` | 98,9 % | Z-score объёма платины |
| 8 | `target_high` | 97,9 % | Внутридневной максимум серебра |
| 9 | `silver_rvol_20` | 96,8 % | Среднесрочная волатильность серебра |
| 10 | `CPIAUCSL` | 95,7 % | Индекс потребительских цен США |

---

## 🔬 Архитектура финальной модели E3b

```
[1] Multi-asset data
    yfinance × 5 металлов (silver / gold / platinum / palladium / copper)
    FRED × 9 macro (TIPS, DXY, breakeven, VIX, INDPRO, CPI, oil) + USDRUB
    Период 2010–2026 (16,4 года, 4118 торговых дней)
       ↓
[2] Feature engineering — 105 признаков
    Per-asset (RSI / ADX / ATR / MA / momentum / rvol) × 5 = 70
    Cross-asset ratios + correlations                     = 12
    Macro + age_days                                      = 18
    Composite metals index                                = 5
       ↓
[3] Triple-barrier labels (López de Prado)
    Adaptive volatility-scaled barriers (vol-scaled TP/SL)
    Regime-aware asymmetric (uptrend: TP>SL, downtrend: TP<SL)
    Horizons: 5 / 10 / 20 / 60 дней (multi-supervision ×4)
       ↓
[4] Walk-forward с purging
    Train window: 1000 дней (sliding)
    Test window:  30 дней
    Step:         30 дней
    Purge:        20 дней + embargo 1 день
       ↓
[5] Feature selection (на каждом fold)
    SelectKBest с mutual_info_classif → top-30 из 105
       ↓
[6] Ensemble training
    3 регим-зависимые модели HistGradientBoostingClassifier
    каждая обучается на подмножестве данных своего рыночного режима
    (восходящий / боковой / нисходящий) через SMA200 + ATR-фильтр
       ↓
[7] Signal generation
    Smoothing: 3-дневное скользящее среднее p_up
    Strong-signal filter: p_up_smoothed ≥ 0,85
    BUY:  p_up_smoothed ≥ 0,48 + strong filter
    SELL: p_up_smoothed < 0,35
    HOLD: иначе
       ↓
[8] Execution mechanics
    Trailing stop: 12 %
    Max hold:      30 торговых дней
    Cooldown:      25 дней между сделками
    OOD detector:  warning при extreme features
       ↓
[9] UI + Уведомления
    Argentum: hero / positions / history / methodology / settings
    Telegram: дедупликация action vs info, график P&L в ₽
    Tinkoff sandbox fallback для cron-режима
```

---

## 🗂 Структура проекта

```
silver_trading_assistant_eda/
├── README.md
├── requirements.txt
├── .env.example
│
├── 🏆 ФИНАЛЬНАЯ МОДЕЛЬ E3b ───────────────────────────────────────────────
│
├── app/multi_asset/                       ← Multi-asset pipeline
│   ├── metal_loader.py                    ← yfinance × 5 металлов + data quality
│   ├── macro_loader.py                    ← FRED × 9 + USDRUB
│   ├── features.py                        ← 105 признаков
│   ├── labels.py                          ← Triple-barrier + adaptive barriers
│   ├── walkforward.py                     ← WF engine + purging + embargo
│   ├── simulator.py                       ← Trade execution
│   └── metrics.py                         ← Sharpe / DSR / PSR / Sortino / Calmar
│
├── experiments/                            ← Дипломные эксперименты E1-E4
│   ├── e1_baseline.py                     ← E1: Sharpe 0,459
│   ├── e2b_feature_selected.py            ← E2b: Sharpe 0,580
│   ├── e3_macro_adaptive.py               ← E3a/b/c
│   ├── e4_stacking.py                     ← E4: Sharpe 0,194 (overfit)
│   ├── compare_v25.py                     ← E3b vs V25
│   └── visualize.py                       ← Графики для диплома
│
├── data/multi_asset/
│   ├── metals/                            ← 5 parquet
│   ├── macro/                             ← 9 parquet + USDRUB
│   ├── features/silver_features.parquet   ← 105 cols × 3110 чистых строк
│   └── labels/silver_labels.parquet
│
├── baseline_outputs_multiasset/           ← Результаты экспериментов
│   ├── e1_baseline/                       ← trades.csv + metrics.json
│   ├── e2_cross_asset/                    ← E2 naive (negative)
│   ├── e2b_feature_selected/              ← E2b + FS
│   ├── e3a_macro/                         ← E3a + macro (negative)
│   ├── e3b_adaptive/                      ← 🏆 90 сделок, +343,5 %
│   ├── e3c_metalabel/                     ← E3c
│   ├── e4_stacking/                       ← E4 (negative)
│   └── competitors_metrics.json           ← AQR / SLV / Medallion
│
├── 🖥 ARGENTUM WEB UI ─────────────────────────────────────────────────────
│
├── argentum/
│   ├── README.md
│   ├── start_all.bat                      ← Запуск backend + frontend
│   ├── start_backend.bat / start_frontend.bat
│   ├── install_desktop_shortcut.bat
│   ├── backend/                           ← FastAPI :8000
│   │   ├── main.py
│   │   └── routers/                       ← 14 роутеров (см. таблицу выше)
│   └── frontend/                          ← Next.js 16 :3000
│       └── app/
│           ├── page.tsx                   ← / Сейчас
│           ├── positions/page.tsx         ← /positions
│           ├── history/page.tsx           ← /history
│           ├── methodology/page.tsx       ← /methodology
│           └── settings/page.tsx          ← /settings
│
├── 📱 STREAMLIT (legacy) ─────────────────────────────────────────────────
│
├── dashboard_app.py                       ← Professional dashboard :8501
├── pages/                                  ← 7 страниц
├── simple_app.py                          ← Облегчённая :8502
├── simple_pages/                          ← 6 страниц
│
├── 🤖 PRODUCTION ──────────────────────────────────────────────────────────
│
├── scripts/
│   ├── daily_e3b.py                       ← 🏆 Daily production run
│   ├── backfill_walkforward_ffill5.py     ← Weekly backfill
│   ├── apply_optimal_exits.py             ← Apply optimal config
│   ├── audit_assistant.py                 ← Аудит UI ↔ Telegram
│   ├── test_telegram_e3b.py               ← Telegram test
│   └── thesis/                            ← Скрипты для ВКР (генерация PNG, .docx)
│
├── .github/workflows/
│   ├── daily_e3b.yml                      ← 🏆 3× в день
│   ├── weekly_backfill.yml                ← Воскресенье
│   └── test.yml                           ← pytest CI
│
├── app/
│   ├── telegram_portfolio.py              ← Tinkoff fallback + ₽ rendering
│   ├── notifier.py                        ← Telegram отправитель
│   ├── utils.py                           ← Cached loaders
│   ├── charts.py                          ← Plotly helpers
│   └── simple_storage.py                  ← Local storage
│
├── tests/                                  ← pytest (17 тестов)
│   ├── test_no_lookahead.py
│   ├── test_simulator.py
│   ├── test_fx.py
│   └── ...
│
├── daily_reports/e3b/                     ← E3b daily training + trading reports
│
├── 📚 ДОКУМЕНТАЦИЯ ────────────────────────────────────────────────────────
│
└── docs/
    ├── ВКР_Silver_Trading_Assistant_v2.docx  ← 🎓 Дипломная работа
    ├── Презентация_защита_ВКР.pptx           ← Презентация на 5 мин
    ├── Речь_защита_5мин.docx                  ← Текст защиты
    ├── ОТВЕТЫ_НА_ВОПРОСЫ.md                   ← Q&A для защиты
    ├── НАУЧНАЯ_ЧЕСТНОСТЬ.md                   ← Отдельный файл (не в ВКР)
    └── images/                                ← PNG для thesis + скриншоты Argentum
```

---

## ⚙ Daily production workflow

```
[Каждый рабочий день 08:00 / 14:00 / 22:00 МСК через GitHub Actions]

  1. Refresh data
     ├─ yfinance: 5 металлов + USDRUB
     └─ FRED: 9 macro indicators
        ↓
  2. Walk-forward retraining E3b
     ├─ Обновляет baseline_outputs_multiasset/e3b_adaptive/
     └─ Sharpe / Win / Max DD пересчитываются
        ↓
  3. Production inference
     ├─ features с ffill_limit=5 (заполняет gaps в палладии)
     ├─ Train на ВСЕХ данных до вчера (purge 21 день)
     └─ Predict p_up_smoothed на сегодня
        ↓
  4. Signal classification
     ├─ p_up_smoothed ≥ 0,48 + strong-filter ≥ 0,85 → BUY
     ├─ p_up_smoothed < 0,35                        → SELL
     └─ иначе                                       → HOLD
        ↓
  5. Дедупликация
     ├─ Сравнение с предыдущим сигналом
     └─ alert_type: action | info
        ↓
  6. Save reports
     ├─ daily_reports/e3b/trading/YYYY-MM-DD/signal.json (latest)
     └─ daily_reports/e3b/trading/YYYY-MM-DD/signal_HHMMSS.json (history)
        ↓
  7. Telegram уведомление
     ├─ PNG-график с master signal + 30-day history + позиции
     ├─ Цены и P&L полностью в ₽ (через Tinkoff GetLastPrices + USDRUB)
     ├─ Action: "📢 НОВЫЙ СИГНАЛ: HOLD → BUY"
     └─ Info:   "ℹ Сигнал не изменился"
        ↓
  8. Git commit обратно в репо
     └─ daily_reports/ + обновлённые parquet
```

---

## 🏆 Конкурентное позиционирование

Сравнение с эталонами на одном периоде (2015–2026, 11,2 года):

| Стратегия | CAGR | Sharpe | Max DD | Hit rate |
|---|---:|---:|---:|---:|
| **E3b (наша модель)** | **+14,3 %** | **0,994** | **−25,7 %** | **64,4 %** |
| Buy-and-Hold SLVRUBF | +6,7 % | 0,310 | −47,1 % | — |
| AQR Managed Futures (industry) | ~+5 % | ~0,40 | ~−15 % | ~57 % |
| SG Trend Index (2000–2023) | — | ~0,42 | — | — |
| Renaissance Medallion (недосягаемый) | ~+37 % | ~2,19 | ~0 % | ~100 % |

**Где E3b выигрывает**: Total return +343,5 % против +106 % у Buy-and-Hold (+237 п.п. альфа); Sharpe **+0,68** к B&H; защита просадки −25,7 % vs −47,1 % у пассивной стратегии; работает в боковых годах (2018, 2023) и в кризис (2022: −22,9 % vs −47 % у B&H в моменте).

**Где E3b проигрывает**: CAGR ниже Renaissance Medallion (это уже золотой стандарт квантовых фондов с inside-доступом к биржам). Calmar Ratio 0,56 — ниже промышленных трендфолловеров. Отрицательная асимметрия (skewness обратная) — следствие архитектуры stop-loss.

Полное сравнение с цифрами и обсуждением методологии — [`docs/НАУЧНАЯ_ЧЕСТНОСТЬ.md`](docs/НАУЧНАЯ_ЧЕСТНОСТЬ.md).

---

## 🎯 Roadmap

- [x] Phase 1: Multi-asset data pipeline (5 металлов + 9 macro, 16 лет)
- [x] Phase 2: Walk-forward engine + Trade simulator + Metrics
- [x] Phase 3: E1 silver-only baseline (Sharpe 0,459)
- [x] Phase 4: E2 / E2b cross-asset + feature selection
- [x] Phase 5: E3a / b / c — macro + adaptive barriers + meta-labeling
- [x] Phase 6: E4 stacking ensemble (negative result, документировано)
- [x] Phase 7: Production integration — Telegram + GitHub Actions
- [x] Phase 8: Argentum UI (FastAPI + Next.js 16)
- [x] Phase 9: Multi-position dashboard + per-position SELL advisor
- [x] Phase 10: Smoothed strong-signal filter + ансамбль регим-зависимых моделей (Sharpe вырос с 0,53 до 0,99)
- [x] Phase 11: Tinkoff sandbox fallback для cron-Telegram
- [x] Phase 12: Полная локализация в ₽ (UI + Telegram + графики)
- [x] Phase 13: ВКР 50–60 страниц + защитные материалы
- [ ] **Live forward validation** — 6 месяцев daily live mode на реальной торговой статистике
- [ ] **Volatility targeting position sizing** для улучшения Calmar Ratio
- [ ] **Drawdown circuit-breaker** (заморозить новые сделки при DD > 8 %)
- [ ] **Online learning** через River library для OOD-адаптации в bull market
- [ ] **NLP sentiment features** через FinBERT на финансовых новостях
- [ ] **HMM regime detection** (расширение SMA200 + ATR до 3-состояния)

---

## 📚 Документация

### Дипломная работа

- [`docs/ВКР_Silver_Trading_Assistant_v2.docx`](docs/ВКР_Silver_Trading_Assistant_v2.docx) — основной документ ВКР (50–60 страниц, 9 таблиц, 12 рисунков Главы 3 + Приложения А–Е)
- [`docs/Речь_защита_5мин.docx`](docs/Речь_защита_5мин.docx) — текст защиты на 5 минут
- [`docs/Презентация_защита_ВКР.pptx`](docs/Презентация_защита_ВКР.pptx) — слайды по шаблону ФТК
- [`docs/ОТВЕТЫ_НА_ВОПРОСЫ.md`](docs/ОТВЕТЫ_НА_ВОПРОСЫ.md) — Q&A: типовые + каверзные вопросы комиссии
- [`docs/НАУЧНАЯ_ЧЕСТНОСТЬ.md`](docs/НАУЧНАЯ_ЧЕСТНОСТЬ.md) — отдельный документ о допущениях и ограничениях исследования (не входит в основной текст ВКР по требованиям ГОСТ)

### Технические

- [`argentum/README.md`](argentum/README.md) — гайд по запуску backend + frontend
- [`docs/PAPER_TRADING.md`](docs/PAPER_TRADING.md) — гайд по Tinkoff sandbox bridge (если есть)

---

## 🛡 Безопасность

- `.env` в `.gitignore` — токены никогда не попадают в репо
- Secrets в GitHub Actions хранятся в `Settings → Secrets and variables → Actions`
- Тинькофф-токен — **только sandbox** (никаких реальных ордеров через API)
- OOD-детектор предупреждает о distribution shift на странице `/api/signal`

---

## 📄 Лицензия

Исследовательский проект для учебных целей. Перед коммерческим использованием — согласовать с автором.
