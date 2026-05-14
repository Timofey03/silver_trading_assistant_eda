# Changelog

Хронология версий помощника. Каждая версия добавляла одно конкретное улучшение и оценивалась по test/forward.

> **Числа в этой таблице — v22-стиль (sum-of-returns, не compound).** Реальные compound числа см. в [docs/RESULTS.md](docs/RESULTS.md).

## v25 — CPCV training (текущая) — май 2026

**Главное изменение**: Combinatorial Purged Cross-Validation вместо expanding window.

- Каждая точка получает p_up через усреднение из 15 моделей (а не одной)
- Purging + embargo предотвращают утечку через перекрытые triple-barrier labels
- Purified policy: `up_threshold=0.55, cooldown=15` (вместо v22 `0.42, 7`)
- Меньше сигналов, выше точность, более честная OOS-валидация

**Результаты (honest math, см. RESULTS.md)**:
- valid 2023: **+8.1%** (vs v22: −27.7%) — **впервые положителен**
- test 2024: **+20.9%** (vs v22: −3.6%) — почти равен BnH
- forward 2025+: **+53.3%** (vs v22: +175%; v22 был inflated overlapping trades)
- Forward bootstrap 95% CI: **[+19%, +53%, +97%]** — нижняя положительна
- DSR ≈ 0.40 — всё ещё не статистически значимо, но уверенность выше

**Файлы**: [`silver_assistant_v25_cpcv.py`](silver_assistant_v25_cpcv.py), [`baseline_outputs_v25/`](baseline_outputs_v25/)

---

## v24 — Gate Overlays + Honest Math — май 2026

**Изменение**: Применение overlay-фильтров поверх v22 сигналов:
- **Liquidity gate**: блок сигналов в low-volume дни (< 50% от 60d медианы)
- **VIX gate**: блок LONG когда VIX > 25 и растёт > 2 пунктов за 5 дней
- **GSR gate**: блок при экстремуме gold/silver ratio z-score (|z60| > 2)
- **Drawdown kill-switch**: остановка торговли при equity drawdown > 20%

**Результаты (honest math)**:
- valid 2023: −15.0% (vs v22: −27.7%) — гейты помогли
- test 2024: +12.5% (vs v22: −3.6%) — гейты помогли
- forward 2025+: +75.6% (vs v22: +175%) — гейты повредили в бычьем тренде

**Урок**: гейты — двухсторонний меч. Помогают в боковике/коррекции, режут прибыль в трендах.

**Файлы**: [`silver_assistant_v24_gates.py`](silver_assistant_v24_gates.py), [`baseline_outputs_v24/`](baseline_outputs_v24/)

---

## v23 — Honest Math + Statistical Robustness — май 2026

**Главный CRITICAL FIX**: математические корректировки v14-v22.

Обнаружено что v22 "+279% forward" — артефакт методологии:
1. `equity = 1 + cumsum(rets)` суммирует доходности, не компаундирует
2. Позиции сильно перекрываются (28 из 45 forward-трейдов overlap)
3. При single-account compound: v22 даёт +175% vs BnH +187% → **проигрыш на 12pp**

**Добавлено**:
- ✅ Compounded equity (cumprod) + single-position-at-a-time
- ✅ Apples-to-apples buy-and-hold (одна позиция, close-to-close)
- ✅ Deflated Sharpe Ratio (López de Prado 2014) — поправка на multiple testing
- ✅ Probabilistic Sharpe Ratio (Bailey & López de Prado 2012)
- ✅ Stationary block bootstrap CI (Politis-Romano)
- ✅ CPCV скелет (использован в v25)
- ✅ Realistic costs: spread + ATR slippage + funding + illiquid premium
- ✅ Liquidity gate
- ✅ Performance attribution (model / execution / sizing / costs)
- ✅ Audit log JSONL (для регуляторного аудита)
- ✅ Drift detection (KS-test 130 фичей → 112 имеют drift!)

**Файлы**: [`silver_assistant_v23_honest.py`](silver_assistant_v23_honest.py), [`baseline_outputs_v23/`](baseline_outputs_v23/)

---

## v22 — Risk-Aware (5 модулей) — апрель 2026

**Изменение**: Добавлены 5 модулей риск-управления.

- **A**: ATR-based trailing stops (вместо фиксированных %)
- **B**: Kelly position sizing (size ∝ p_signal / mean_p)
- **C**: Risk metrics — Sharpe, Calmar, Ulcer, time-underwater
- **D**: Multi-horizon ensemble (5d + 15d + 30d прогнозы)
- **E**: Walk-forward retraining (каждые 60 рабочих дней)

**Результаты (по v22-математике)**:
- forward base: +279.85% (vs BnH +187%) — **inflated**
- forward wf: +189.55%
- forward all: +258.45%

**Файлы**: [`silver_assistant_v22_risk_aware.py`](silver_assistant_v22_risk_aware.py), [`baseline_outputs_v22/`](baseline_outputs_v22/) (содержит канонический v22_full_data.csv)

---

## v21 — Regime Short + Independent Streams — март 2026

**Изменения**:
- LONG и SHORT потоки идут независимо (без state machine, как было в v20)
- SHORT блокируется в uptrend режиме
- Cooldown grid для SHORT

**Результаты**: forward 2025 = +178% (vs v22-style)

**Файлы**: [`silver_assistant_v21_regime_short.py`](silver_assistant_v21_regime_short.py)

---

## v20 — Directional (UP + DOWN classifiers) — март 2026

**Изменения**:
- Отдельный DOWN-классификатор для SHORT-сигналов
- State machine: позиция = либо LONG, либо SHORT (не оба)

**Проблема**: state machine блокировала много сигналов → суммарный P&L просел до +66%. Исправлено в v21.

**Файлы**: [`silver_assistant_v20_directional.py`](silver_assistant_v20_directional.py)

---

## v19 — Trailing Stop — март 2026

**Главное изменение**: Trailing stop (7% от пика) вместо фиксированного 15-day exit.

**Результат**: +123pp прирост к forward P&L. Самое большое единичное улучшение в проекте.

**Файлы**: [`silver_assistant_v19_trailing.py`](silver_assistant_v19_trailing.py)

---

## v18 — Adaptive Weight + Expanding Window — февраль 2026

**Изменения**:
- Expanding window training (каждые 6 месяцев — новая модель)
- Adaptive sample weight: больший вес recent rows
- Half-life decay (1.5 года)

**Файлы**: [`silver_assistant_v18_adaptive.py`](silver_assistant_v18_adaptive.py)

---

## v17 — FRED Macro via yfinance — февраль 2026

**Изменение**: Макро-фичи (FRED не работал из РФ → через yfinance ETF-прокси):
- `^TNX, ^IRX` — yields
- `TIP, RINF, HYG` — ETF proxies для real rates, inflation, credit risk

**TIP zscore стабильно в топ-5 по importance.**

**Файлы**: [`silver_assistant_v17_fred.py`](silver_assistant_v17_fred.py)

---

## v16 — Binary Classification + Top-30 Features — январь 2026

**Изменения**:
- 3-классовая задача → бинарная (UP vs not-UP)
- Регуляризация HistGB (`l2_regularization=3, max_depth=3`)
- Permutation importance → top-30 фичей

**Файлы**: [`silver_assistant_v16_binary.py`](silver_assistant_v16_binary.py)

---

## v15 — Regime Ensemble + COT — декабрь 2025

**Изменения**:
- COT (CFTC) данные через `nasdaqdatalink`
- Режимная сегментация: uptrend / sideways / downtrend × low_vol / med / high
- Отдельные модели для каждого режима
- `cot_index_52w` стабильно в топе

**Файлы**: [`silver_assistant_v15_regime_cot.py`](silver_assistant_v15_regime_cot.py)

---

## v14 — Triple Barrier Labels — декабрь 2025

**Главный фундамент проекта**.

- Скачивает OHLC (silver, gold, copper, oil, S&P, EUR/USD)
- Создаёт ~80 фичей (моментум, волатильность, режимы)
- Триплет-барьерные labels (López de Prado): UP/DOWN/NEUTRAL на горизонте 15d с TP/SL барьерами
- Базовый бэктестер

**Файлы**: [`silver_assistant_v14_main.py`](silver_assistant_v14_main.py)

---

## v1 — v13 (архивные)

Ранние эксперименты:
- v1-v9: Различные базовые архитектуры (logistic, RF, GBT)
- v10: Ordinal regression
- v11: Meta-modeling
- v12: Sentiment + custom indicators
- v13: Triple barrier prototype

Все в [`_archive/`](_archive/) для исторической справки.

---

## Главный урок

> **ML — не главный инструмент.** Из +279% на forward, ML добавлял максимум 15-30pp. Остальное было от:
> - Trailing stop вместо fixed exit (v19: +123pp)
> - Risk management (Kelly, ATR stops)
> - Regime filters
>
> **Но в HONEST MATH (v23+)** — даже эти числа сжимаются:
> - Compound equity vs sum-of-returns: −104pp
> - Apples-to-apples BnH: стратегия проигрывает простому buy-and-hold
> - CPCV vs expanding window: real edge оказался +8% в valid (vs −27% inflated), +21% в test
>
> **Bottom line**: после всех очисток в v25 у стратегии есть **скромный положительный edge в боковике/коррекциях**, но она **не побеждает простой BnH в сильных бычьих трендах**. Это базовое свойство любой trend-following стратегии — нормально, но нужно знать.
