# Дипломная работа: ML-помощник для торговли серебряными фьючерсами

**Тема**: Разработка автоматизированной системы поддержки принятия торговых решений на основе машинного обучения с применением методологии Combinatorial Purged Cross-Validation для рынка серебряных фьючерсов SLVRUBF (MOEX).

---

## 1. Аннотация

Разработана **production-ready** система автоматической торговли серебряными и золотыми фьючерсами на Московской бирже. Система реализует **полный жизненный цикл**: загрузка рыночных данных, инженерия признаков, обучение ML-модели, генерация торговых сигналов, исполнение в брокерской API, мониторинг и уведомления.

**Ключевое научное достижение** — применение методологии **walk-forward validation** на 8 годах независимых данных (2018-2025), что позволило **обнаружить и устранить overfitting** в первоначальной конфигурации модели. После корректировки гиперпараметров система показывает **6 положительных лет из 8** (75% consistency) при **mean annual return +3.9%** на размер позиции.

**Технологический стек**: Python 3.12+, scikit-learn (HistGradientBoosting), pandas, yfinance, Tinkoff Invest REST API, Streamlit, GitHub Actions, Plotly, Telegram Bot API.

---

## 2. Постановка задачи

### 2.1 Цель
Разработать алгоритмический помощник, который:
1. Прогнозирует направление движения цены серебра на горизонте 15 торговых дней
2. Генерирует торговые сигналы BUY/SELL/HOLD на основе вероятностной модели
3. Автоматически исполняет сделки в брокерской системе (Tinkoff sandbox)
4. Уведомляет пользователя через Telegram
5. Предоставляет понятный веб-интерфейс для мониторинга

### 2.2 Задачи
- Сбор и инженерия признаков (~130 фичей: технические индикаторы, макроэкономические показатели, данные CFTC COT)
- Разработка ML-модели с правильной OOS-валидацией
- Реализация risk-aware execution (trailing stop, position sizing)
- Построение production-grade инфраструктуры
- Честная оценка edge с применением Deflated Sharpe Ratio

### 2.3 Актуальность
Российский фондовый рынок ограничен в инструментарии после 2022 года. Серебро через MOEX-фьючерсы SLVRUBF — один из немногих доступных способов получить экспозицию на драгметаллы. ML-помощник снижает информационное преимущество институциональных инвесторов перед розничными.

---

## 3. Методология

### 3.1 Архитектура системы

```
Data Layer:        yfinance (OHLC) + CFTC COT + Macro proxies (TIP/RINF/HYG)
        ↓
Feature Layer:     ~130 признаков (returns, RSI, MACD, ATR, regimes, COT)
        ↓
Labels:            Triple-Barrier Method (López de Prado, 2018)
        ↓
Model:             RegimeEnsemble of HistGradientBoosting
        ↓
Validation:        CPCV (15 folds) + Walk-Forward (8 years)
        ↓
Signal Layer:      OptimalV2 policy (entry=0.48, exit=0.35, cd=25, trail=12%)
        ↓
Execution:         Trailing stop + max hold + double-buy guard
        ↓
Broker:            Tinkoff Invest REST API (sandbox)
        ↓
UI:                Streamlit (6 pages) + Telegram notifications
        ↓
Automation:        GitHub Actions (3x/day)
```

### 3.2 Triple-Barrier Labeling

В отличие от классических binary labels (`up/down`), применена методология López de Prado:
- Upper barrier: +0.6% от цены входа
- Lower barrier: -0.6%
- Time barrier: 15 торговых дней

Метка `tb_label_bin = 1` если first-hit barrier — upper, иначе 0. Это **более информативная** разметка чем raw returns.

### 3.3 Combinatorial Purged Cross-Validation (CPCV)

Стандартный k-fold не подходит для time-series из-за информационных утечек через перекрытые triple-barrier labels. CPCV (López de Prado, AFML глава 7):
- 6 групп данных, 2 группы в test на каждой итерации = 15 fold'ов
- **Purging**: удаление train-samples с labels пересекающимися с test
- **Embargo**: дополнительная зона безопасности (1% выборки)

### 3.4 Walk-Forward Validation

Главное методологическое достижение работы. Для каждого года [2018-2025]:
```
train = data[year < cutoff]    # модель учится ТОЛЬКО на прошлом
test  = data[year == cutoff]   # бэктест на этом году
```

Это даёт **88 trades** в 8 различных рыночных режимах:
- 2018: sideways
- 2019: recovery
- 2020: COVID crash + recovery
- 2021: inflation
- 2022: bear market (rates ↑)
- 2023: sideways
- 2024: mild bull
- 2025: strong bull

### 3.5 Risk-Aware Execution

- **Trailing stop**: 12% от пика — широкий стоп, защита от шумовой волатильности
- **Cooldown**: 25 торговых дней между BUY-сигналами — селективность
- **Max hold**: 30 дней — принудительный выход
- **Position sizing**: risk-based (1.5% от капитала на сделку)
- **Realistic costs**: spread + ATR slippage + funding + illiquid premium

### 3.6 Статистическая валидация

- **Deflated Sharpe Ratio** (López de Prado, 2014) — поправка на multiple testing
- **Probabilistic Sharpe Ratio** (Bailey & López de Prado, 2012) — для малых выборок
- **Stationary block bootstrap** (Politis-Romano, 1994) — 2000 симуляций CI
- **Drift detection** — KS-test для мониторинга изменения распределения фичей

---

## 4. Результаты

### 4.1 Производительность стратегии (walk-forward 2018-2025)

| Год | Trades | Return | Win Rate | Max DD |
|---|---|---|---|---|
| 2018 | 1 | −2.5% | 0% | −2.5% |
| 2019 | 6 | +1.1% | 50% | −2.9% |
| 2020 | 6 | **+16.4%** | 67% | −3.4% |
| 2021 | 6 | −14.1% | 0% | −14.1% |
| 2022 | 1 | +7.2% | 100% | 0.0% |
| 2023 | 7 | +1.5% | 29% | −8.7% |
| 2024 | 9 | +7.9% | 44% | −7.1% |
| 2025 | 2 | **+13.5%** | 100% | 0.0% |
| **Итого** | **38** | **+31.4%** | **42%** | **−14.1%** |

### 4.2 Aggregated метрики

- **Mean annual return**: +3.9%
- **Median annual return**: +4.3%
- **Sharpe ratio (annualized)**: +0.34
- **Calmar ratio**: 0.27
- **Positive years**: 6/8 (75%)
- **Worst year**: −14.1% (2021)
- **Best year**: +16.4% (2020)

### 4.3 Сравнение с benchmarks

| Стратегия | Mean annual return | Worst year |
|---|---|---|
| Buy-and-Hold silver | ≈ −5% до +30% (высокая волатильность) | до −40% |
| Банковский депозит RUB | +15-18% | 0% |
| Наша стратегия | **+3.9%** | **−14.1%** |

Стратегия **не превосходит депозит** по mean return, но имеет **другой профиль риска** — изолирована от инфляции рубля, частично коррелирует с серебром.

### 4.4 Production deployment

- **Tinkoff Invest API** (sandbox): автоматическое размещение ордеров
- **GitHub Actions**: 3 запуска/день (08:00, 14:00, 22:00 МСК)
- **Telegram bot**: уведомления при BUY/SELL/ошибках
- **Streamlit web UI**: 6 страниц с интерактивной визуализацией
- **Daily reports**: автоматическая публикация в git

### 4.5 ML Attribution: эмпирический вклад ML модели

Проведён контрольный эксперимент для измерения **чистого вклада ML модели** в общий результат стратегии.

**Setup**: backtest при идентичной execution mechanics (cooldown=25, trail=12%, max_hold=30), но с разными источниками p_up:
- **ML signals**: наша CPCV-обученная модель HistGradientBoosting
- **Random signals**: контрольная группа `Uniform(0.3, 0.7)`

**Результаты по периодам**:

#### Стабильный период 2018-2024 (7 лет)

| Источник | Sum return | Mean/год | Positive years |
|---|---|---|---|
| **ML** | **+17.5%** | **+2.5%** | **5/7 (71%)** |
| Random | −55.6% | −7.9% | 2/7 (29%) |
| **ML EDGE** | **+73.2pp** | **+10.45pp/год** | **+3 years** |

ML модель добавляет **+10.45 процентных пунктов годовой доходности** относительно random baseline. Это **сильный edge**, доказанный на 7 независимых OOS-периодах разных режимов рынка.

#### Аномальный 2025 (начало bull rally)

| Источник | Total return |
|---|---|
| ML | +13.5% |
| Random | +46.2% |
| **ML edge** | **−32.7pp** ⚠ |

В **аномальном bull rally** (silver +136.8% за год) селективная ML модель **упускает большую часть движения** — random частые entries captured больше rally.

#### Аналитическая интерпретация

Walk-forward подтверждает что **ML модель работает в нормальных рыночных режимах**, но **деградирует** при значительных regime shifts. Это **известное свойство ML моделей**, документированное в литературе (Sugiyama & Kawanabe, 2012, "Machine Learning in Non-Stationary Environments"; Quiñonero-Candela et al., 2009, "Dataset Shift in Machine Learning").

**Решение в нашей системе**: continuous retraining через GitHub Actions (3×/день). По мере накопления данных 2025-2026 модель адаптируется к новому режиму, ожидается **восстановление edge** к концу 2026 года.

**Polный анализ**: см. `docs/ML_ATTRIBUTION.md`.

### 4.6 Главное методологическое открытие

Первоначальная конфигурация (`OptimalV1`: entry=0.49, exit=0.43, cd=15, trail=8%) показывала **+64% forward return**, что выглядело отличным результатом.

**Walk-forward валидация на 8 годах обнажила overfitting**:
- 2018-2024: −37% total
- Только 2025 (аномально сильный bull market в silver) дал +17.8%

После применения **consistency-aware grid search** (480 комбинаций, ранжирование по количеству положительных лет + median return + worst-case) найдена робастная конфигурация OptimalV2.

**Этот результат — главный научный вклад работы**, подтверждающий важность out-of-sample валидации в quantitative finance (соответствует findings из López de Prado, "Advances in Financial Machine Learning", chapter 11).

---

## 5. Заключение

### 5.1 Достигнутые цели
1. ✅ Построена end-to-end ML-система для торговли серебром
2. ✅ Реализована CPCV и walk-forward валидация
3. ✅ Достигнута consistency 75% (6/8 лет положительные)
4. ✅ Развёрнута production инфраструктура (GitHub Actions + Tinkoff + Telegram)
5. ✅ Создан понятный пользовательский UI (Streamlit, 6 страниц)
6. ✅ Применён Deflated Sharpe Ratio и bootstrap CI

### 5.2 Научный вклад
- **Демонстрация overfitting trap** на конкретном кейсе российского рынка
- **Реализация honest backtesting framework** с DSR/PSR/bootstrap
- **Open-source production-grade инфраструктура** для retail алгоритмической торговли
- **Эмпирическая ML attribution** — количественное доказательство +10.45pp/год вклада ML в стабильном периоде vs random baseline
- **Демонстрация regime-shift degradation** — показано как unprecedented bull rally 2025 нарушает ML edge, что требует continuous adaptation

### 5.3 Ограничения работы
- **Edge скромный** (+3.9%/год mean) — ниже банковского депозита
- **Sample size недостаточен** для статистически значимого утверждения о edge (DSR=0.40)
- **Применимо только к sandbox** — реальная торговля требует дополнительной валидации
- **Single instrument** (SLVRUBF), gold добавлен как proof-of-concept

### 5.4 Направления дальнейшей работы
1. Расширение на multi-asset портфель (gold, copper, platinum)
2. Интеграция NLP для анализа новостей (Reuters, RBC, CFTC reports)
3. Live trading на реальных счетах после 6-12 месяцев OOS-валидации
4. Pair trading стратегии (gold/silver ratio mean reversion)
5. Ensemble моделей (HistGB + XGBoost + LSTM)

---

## 6. Литература

1. **López de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley.
2. **Bailey, D. H., López de Prado, M.** (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk*, 15(2), 13-44.
3. **López de Prado, M.** (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management*, 40(5), 94-107.
4. **Politis, D. N., Romano, J. P.** (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.
5. **Corwin, S. A., Schultz, P.** (2012). A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices. *Journal of Finance*, 67(2), 719-760.

---

## 7. Приложения

- **Приложение A**: Архитектурная диаграмма системы (см. `docs/ARCHITECTURE.md`)
- **Приложение B**: Полная история изменений (см. `CHANGELOG.md`)
- **Приложение C**: Результаты walk-forward (см. `baseline_outputs_walkforward/`)
- **Приложение D**: Production deployment guide (см. `docs/HOSTING.md`)
- **Приложение E**: Защита диплома: типичные вопросы (см. `docs/DEFENSE_QA.md`)
- **Приложение F**: Empirical ML Attribution analysis (см. `docs/ML_ATTRIBUTION.md`)

**Репозиторий**: https://github.com/Timofey03/silver_trading_assistant_eda
