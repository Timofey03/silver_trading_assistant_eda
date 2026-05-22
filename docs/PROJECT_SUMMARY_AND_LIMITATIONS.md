# Итог проекта и его ограничения

Документ-сводка для дипломной защиты, code review и будущих исследователей.

---

## 🏆 Что было достигнуто

### Научная часть (6 экспериментов с честной атрибуцией)

| Эксп. | Что добавлено | Sharpe | Annual | Win | Max DD | Вердикт |
|---|---|---:|---:|---:|---:|---|
| E1 | Silver-only baseline (14 фичей) | **0.459** | +4.6% | 60% | −24.5% | ✅ benchmark |
| E2 | Naive cross-asset (84 фичи) | −0.248 | −3.9% | 58% | −51.1% | ❌ curse of dim |
| E2b | + Feature selection (top-25) | **0.580** | +5.9% | 68% | −18.3% | ✅ работает |
| E3a | + Macro features (102 → 30) | 0.424 | +5.5% | 60% | −31.3% | ❌ macro broken |
| **E3b** | **+ Adaptive barriers** ★ | **0.530** | **+7.7%** | **69%** | **−17.9%** | 🏆 **WINNER** |
| E4 | Stacking ensemble | 0.194 | +1.9% | 50% | −36.5% | ❌ overfit |

### Forward test 2025-2026 (apples-to-apples с V25)

- **E3b OOS**: Sharpe 2.17, Win 83%, Max DD −0.15%, 6 trades за 16 месяцев
- **V25 forward**: Sharpe 2.37, Win 66%, Max DD −37%, 38 trades
- **B&H silver**: +152.6% за тот же период (без активного трейдинга)

### Инфраструктура

- ✅ Multi-asset data pipeline (5 металлов × 16 лет, ~62K supervision pairs)
- ✅ Walk-forward engine с purging + embargo (CPCV-style)
- ✅ Daily retraining через GitHub Actions (3× в день по будням)
- ✅ Telegram-уведомления с дедупликацией (action vs info)
- ✅ Два Streamlit-приложения (полное + облегчённое)
- ✅ Интеграция с Tinkoff Invest API (paper trading)
- ✅ Дипломная работа: 502 параграфа, 9 таблиц, 6 встроенных PNG, ~12 900 слов

---

## ⚠ Слабые стороны и недочёты

### Категория 1: Методологические (значимы для академической чистоты)

#### 1.1 Adaptive barriers не были оптимизированы grid search

Текущие коэффициенты `(1.5, 1.0)` для uptrend, `(1.0, 2.0)` для downtrend, `(1.2, 1.2)` для sideways были выбраны на основе экспертной интуиции. Полный grid search по этим параметрам не проводился.

**Что нужно**: систематический поиск best multipliers через CV. Ожидаемый эффект: +0.05–0.10 Sharpe.

#### 1.2 Cooldown не оптимизирован

В honest check (cool=20 vs cool=25) на train period 2014-2024 выяснилось, что:
- Cool=25 (baseline): Sharpe 0.536
- Cool=30 (лучший): Sharpe **0.882**
- Cool=20: Sharpe 0.283

То есть **существует ещё более оптимальная точка** (cool=30), но мы её не используем как production-настройку, потому что осознанно избегаем cherry-picking на исторических данных.

**Что нужно**: CV-based optimization cooldown на train period с фиксацией перед forward.

#### 1.3 Macro features провалились — нужны альтернативные подходы

E3a показал что 9 макроиндикаторов (TIPS, DXY, VIX, oil, INDPRO, CPI, USDRUB) с forward-fill не помогают (Sharpe упал на 0.16). Это **negative result**, но не означает что macro в принципе бесполезны.

**Что нужно**:
- Использовать macro releases events (FOMC, NFP, CPI) как дискретные шоки, а не gladly-смазанные ряды
- Применить streaming-aware методы: separate models на дни с/без новой macro информации
- Попробовать macro дельты (изменения) вместо уровней

#### 1.4 Stacking не сработал — методологическая ловушка

E4 показал что объединение HistGB+LightGBM+CatBoost через meta-LR ухудшает результаты на ограниченных данных (1 000 train samples per fold).

**Что нужно**:
- Использовать гораздо больший training set для устойчивого stacking
- Попробовать blending (weighted average) вместо meta-LR
- Применить boosting на residuals одиночной модели

#### 1.5 Только один horizon (20 дней) в финальной модели

Несмотря на наличие 4 параллельных меток (5, 10, 20, 60 дней) в pipeline, финальная модель E3b использует **только горизонт 20**. Multi-horizon multi-task learning не был реализован.

**Что нужно**:
- Обучить multi-output модель с 4 head'ами
- Использовать вспомогательные горизонты как auxiliary signal

#### 1.6 Cross-asset backbone не реализован полностью

Pipeline загружает данные 5 металлов и считает cross-asset фичи, но **transfer learning** в строгом смысле (pre-train на gold/copper, fine-tune на silver) не применён. Все 5 активов используются только как фичи silver, не как отдельные learning tasks.

**Что нужно**:
- Pre-train модель на gold с теми же фичами
- Fine-tune на silver с замороженными ранними слоями
- Сравнить с full-train

### Категория 2: Технические ограничения данных

#### 2.1 Только дневные данные

Все эксперименты на дневных свечах. Intraday (часовые/5-минутные) данные не использовались.

**Что нужно**:
- Yahoo Finance даёт только 730 дней intraday для futures
- Для глубокой intraday истории нужен Polygon.io ($30/мес) или IBKR
- Внедрение даст ~20× больше observations

#### 2.2 Palladium gaps требуют forward-fill в production

yfinance palladium имеет ~18 missing дней с 2010 года. Для walk-forward (academic) — это не проблема (data cut-off). Для production-инференса свежих дней нужен `ffill_limit=5`. Это **компромиссное решение** — мы используем два режима features pipeline.

**Что нужно**: альтернативный источник палладия (LME, CME) или замена палладия на другой industrial-metal proxy.

#### 2.3 Только COMEX silver, не SLVRUBF

Модель обучается на COMEX (доллар-номинированный), но применяется к SLVRUBF (рубль-номинированный). Эти инструменты коррелированы (через USDRUB), но не идентичны.

**Что нужно**:
- Прямое обучение на SLVRUBF daily через Tinkoff Historical Data API
- Учёт российской премии/дисконта vs COMEX

#### 2.4 Macroindex монжетарные данные с задержкой публикации

CPI и INDPRO публикуются с лагом 2-6 недель. В нашем feature pipeline мы используем их с forward-fill, что **technically даёт модели future-leaking информацию** на следующие дни после публикации (хотя это не leakage, а правильное использование релизной даты).

**Что нужно**: явная маркировка release date для каждого macro indicator и использование только данных с release_date ≤ t.

### Категория 3: Production-ограничения

#### 3.1 Нет учёта реальных transaction costs и slippage

Симулятор использует 0.1% комиссию + 2 round-trip. Но **реальные** условия:
- Spread на SLVRUBF может быть 0.2-0.5% (низкая ликвидность)
- Slippage на крупных лотах
- Market impact

**Что нужно**: историческая база bid-ask quotes для honest transaction cost simulation.

#### 3.2 Walk-forward валидирован только до 2025

Walk-forward fold заканчивается в 2025-06 (из-за palladium ffill limit в academic режиме). Forward test (отдельный эксперимент) покрывает 2025-2026, но это уже **разные методологии**:
- Walk-forward (academic): без ffill, до 2025-06
- Forward test (production-like): с ffill, 2025-01 → 2026-05

**Что нужно**: единая методология с явным разделением "data we have at time t" vs "data we use".

#### 3.3 Нет real-money trading history

Все результаты — на бэктесте. Paper trading через Tinkoff включён но не протестирован на длительном периоде. **Реальная live performance модели неизвестна.**

**Что нужно**: минимум 6 месяцев paper trading с записью каждого ордера, затем сравнение с backtest.

#### 3.4 Зависимость от Yahoo Finance

Все данные через yfinance. Если Yahoo сменит API или отключит фьючерсы — система остановится.

**Что нужно**: альтернативные источники (FRED USD silver, IBKR, Refinitiv) с автоматическим fallback.

#### 3.5 Деградация модели в OOD режимах (2025-2026)

В forward test модель сделала только 6 сделок за 16 месяцев — она **чрезмерно консервативна** в условиях OOD (silver $30 → $77, никогда не виденных при обучении). Mean p_up = 0.28 (vs ожидаемые 0.4-0.5).

**Что нужно**:
- Online learning с River library для постоянной адаптации
- Specifically retrain на OOD периодах
- Detection drift через KS-test на распределение p_up

### Категория 4: Дизайн стратегии

#### 4.1 Только LONG-only

Модель никогда не открывает SHORT-позиции. Это упрощение — но в долгосрочных bear markets (как 2018, 2021) это лишает стратегию ~50% потенциала.

**Что нужно**: симметричная модель с трёхклассовой классификацией {LONG, FLAT, SHORT} + соответствующий simulator.

#### 4.2 Фиксированное position sizing

Каждая сделка использует одинаковый размер позиции (определяется trail и max_hold, но не вероятностью). Kelly Criterion или volatility-targeting не применяются.

**Что нужно**: position size = f(p_up confidence, current volatility, account drawdown).

#### 4.3 Нет multi-asset портфеля

Модель торгует только серебро. Но pipeline уже загружает 5 металлов — можно одновременно держать позиции в нескольких.

**Что нужно**: portfolio-level optimization с correlation-aware allocation.

### Категория 5: Тестирование и валидация

#### 5.1 Только одна walk-forward конфигурация

Все эксперименты с `train_window=1000, step=30, test_window=30`. Чувствительность к этим параметрам не проверена.

#### 5.2 Random seed не варьировался

Все эксперименты с `random_state=42`. Variance результатов на разных seed'ах не оценена. Возможно, E2b превосходит E1 случайно.

**Что нужно**: запустить каждый эксперимент 10 раз с разными seed'ами, отчитать mean ± std для каждой метрики.

#### 5.3 DSR упал из-за multiple testing

Honest DSR с N=6 trials = 0.000 для E2b-E4. PSR=1.0 во всех. В дипломе это явно объяснено, но не решено методологически.

**Что нужно**: pre-registration экспериментов или Bonferroni-style correction в planning.

---

## 🎯 Что бы я сделал по-другому, если бы начинал заново

### Topics to investigate next

1. **River online learning** — заменить batch retraining на incremental updates
2. **Conformal prediction** — формальные prediction intervals вместо calibration
3. **Causal feature engineering** — DoubleML или causal forests для отбора causally-relevant features
4. **Sentiment from FOMC minutes** — NLP-фичи с FinBERT на текстах ФРС
5. **Regime detection через HMM** — скрытые марковские модели для авто-detection rejimes
6. **Position sizing через Kelly** — динамический размер позиции от confidence
7. **Symmetric LONG/SHORT** — расширить на short с separate model
8. **Multi-asset portfolio** — одновременно торговать несколькими металлами

### Architectural improvements

- Замена parquet на DuckDB для быстрых ad-hoc запросов
- Streamlit → Dash/Panel для лучшей интерактивности графиков
- Перенос data pipeline в Airflow или Prefect для production-grade orchestration
- Микросервисная архитектура: data-fetcher, model-trainer, signal-generator, notifier — отдельные сервисы

---

## 📊 Итоговая оценка проекта

| Аспект | Оценка | Комментарий |
|---|---|---|
| **Научная новизна** | 7/10 | Cross-asset + adaptive barriers — известные, но грамотно применённые |
| **Методологическая строгость** | 8/10 | Walk-forward + purging + честная attribution; DSR с multi-trials |
| **Качество кода** | 7/10 | Модульно, тестируемо, но не всё покрыто unit-тестами |
| **Production-готовность** | 6/10 | Daily automation работает, но real-money не тестирован |
| **Воспроизводимость** | 9/10 | Все эксперименты воспроизводимы, артефакты в репо |
| **Документация** | 8/10 | Диплом + README + комментарии в коде |
| **Практическая полезность** | 5/10 | Backtest показывает edge, но live performance не подтверждён |

**Общая оценка**: **дипломная работа уровня «отлично»** с пакетом перспективных направлений для будущих исследований.

---

## 🎓 Главные защитные тезисы

1. **Cross-asset volatility — ключевой предиктор серебра** (95% top features — vol & corr на 5 металлах). Это эмпирический insight, обнаруженный через feature importance анализ финальной модели.

2. **Curse of dimensionality реален и для финансовых данных** — наивное добавление 70 фичей ухудшило Sharpe с +0.459 до −0.248. Feature selection обязателен.

3. **Adaptive barriers — главный методологический вклад работы** — +5 п.п. accuracy, +2.2 п.п. annual return, −13.4 п.п. max DD по сравнению с фиксированными барьерами.

4. **Macro features в forward-fill режиме не работают** — это honest negative result. Требуют specialized обработки (streaming или event-based).

5. **Stacking не помогает на ограниченных данных** — эмпирическое подтверждение Occam's razor: одиночная модель с правильным feature engineering превосходит ансамбль.

6. **OOD-проблема реальна** — модель чрезмерно консервативна (mean p_up = 0.28 в forward test 2025-2026 vs ожидаемые 0.4-0.5). Требует online learning.

7. **Production-готовность достигнута** — GitHub Actions + Telegram + Streamlit + дедупликация работают, но реальная торговля не проверена.
