# Silver Trading Assistant

ML-помощник для торговли серебром (SI=F / SLVRUBF) с полным циклом: feature engineering → ML signals → risk-aware execution → paper trading через Tinkoff Invest API.

> ⚠️ **Это исследовательский проект.** Не финансовый совет. Все цифры backtest в реальных деньгах могут быть значительно ниже из-за market impact, slippage и режимных сдвигов. Перед любыми реальными вложениями — минимум 6 месяцев paper trading.

---

## 🚀 Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Создать .env с Tinkoff sandbox токеном (см. .env.example)
#    Токен получить: https://www.tinkoff.ru/invest/settings (sandbox-only!)

# 3. Полный pipeline: v25 CPCV — главная актуальная модель
python silver_assistant_v25_cpcv.py

# 4. Paper trading в Tinkoff sandbox
python silver_paper_tinkoff.py --setup --initial-rub 1000000
python silver_paper_tinkoff.py --find SLVRUBF
python silver_paper_tinkoff.py --live --ticker SLVRUBF

# 5. Дашборд (Streamlit)
streamlit run dashboard_app.py
```

---

## 📊 Текущие результаты (honest math + realistic costs)

| Версия / Split | valid 2023 | test 2024 | forward 2025+ | vs BnH forward |
|---|---|---|---|---|
| v22 expanding-window | −27.7% | −3.6% | +175.3% | −12pp |
| v24 с gates | −15.0% | +12.5% | +75.6% | −111pp |
| **v25 CPCV** | **+8.1%** ✅ | **+20.9%** ✅ | **+53.3%** | −134pp |

**Главная честная картина (v25 CPCV)**:
- В **valid и test** (правильный OOS) стратегия впервые показывает положительный edge выше BnH
- Forward 2025+ всё ещё проигрывает простому buy-and-hold (силовой бычий тренд серебра)
- Forward bootstrap 95% CI: **[+19%, +53%, +97%]** — нижняя граница положительна
- **DSR ≈ 0.40** — статистически не отличимо от удачи; нужен живой OOS

Полная сводка результатов: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## 🗂️ Структура проекта

```
silver_trading_assistant_eda/
├── README.md                              ← вы здесь
├── CHANGELOG.md                           ← хронология v14 → v25
├── requirements.txt
├── .env.example                           ← шаблон креденшалов (НЕ настоящие токены!)
├── .gitignore
│
├── silver_assistant_v14_main.py           ← OHLC, базовые фичи, triple-barrier labels
├── silver_assistant_v15_regime_cot.py     ← COT данные, режимная сегментация
├── silver_assistant_v16_binary.py         ← Бинарная классификация, RegimeEnsembleBinary
├── silver_assistant_v17_fred.py           ← Макро фичи (TIP, RINF, HYG)
├── silver_assistant_v18_adaptive.py       ← Expanding window + adaptive weight
├── silver_assistant_v19_trailing.py       ← Trailing stop backtester
├── silver_assistant_v20_directional.py    ← DOWN-classifier (SHORT)
├── silver_assistant_v21_regime_short.py   ← Режимный фильтр SHORT
├── silver_assistant_v22_risk_aware.py     ← ATR + Kelly + Multi-horizon + WF (главный v22)
├── silver_assistant_v23_honest.py         ← ⭐ Honest math: compound equity, DSR, PSR, bootstrap, CPCV
├── silver_assistant_v24_gates.py          ← Gate overlays: liquidity / VIX / GSR / DD-kill
├── silver_assistant_v25_cpcv.py           ← ⭐ CPCV retraining — текущая лучшая модель
│
├── silver_paper_tinkoff.py                ← Tinkoff REST API paper trading bridge
├── silver_spread_estimator.py             ← Бесплатный proxy spread (Corwin-Schultz)
├── dashboard_app.py                       ← Streamlit dashboard
│
├── baseline_outputs_v22/                  ← Канонические данные + v22 trades (32 файла)
│   └── v22_full_data.csv                  ← 🔑 Главный датасет (3481 строка, 130+ фичей)
├── baseline_outputs_v23/                  ← Honest math + paper trading логи (15 файлов)
├── baseline_outputs_v24/                  ← Gate overlay результаты (10 файлов)
├── baseline_outputs_v25/                  ← ⭐ CPCV результаты (9 файлов)
│
├── docs/
│   ├── ARCHITECTURE.md                    ← Архитектура pipeline
│   ├── RESULTS.md                         ← Все результаты в одном месте
│   └── PAPER_TRADING.md                   ← Гайд по Tinkoff bridge
│
└── _archive/                              ← Старые версии (v1-v13 + старые outputs)
    ├── notebooks/                         ← 19 экспериментальных ноутбуков
    ├── patches/                           ← 8 старых .py патчей (v10-v13)
    ├── outputs/                           ← baseline_outputs/v2/.../v21/ (22 директории)
    └── misc/                              ← логи и тестовые скрипты
```

---

## 🔗 Граф зависимостей кода

```
v14_main ──┬─→ v15_cot ──┬─→ v16_binary ──→ v17_fred ──→ v18_adaptive
           │             │                                      │
           └─→ (yfinance)│                                      ↓
                         └─→ v19_trailing ──→ v20_directional ──┤
                                                                 │
                                              v21_regime_short ←─┘
                                                       │
                                                       ↓
                                              v22_risk_aware ←──→ v23_honest
                                                       │            │
                                              v24_gates ◀──────────┤
                                                                    │
                                                       ←──── v25_cpcv ─→ paper_tinkoff
```

`v25_cpcv` — текущая production-модель. Зависит от всех v14-v22 для feature engineering + от v23 для honest math + опционально от v24 для gate overlays.

---

## 📈 Workflow

```
[Daily схема production]

  06:00  Yfinance fetch OHLC + macro                       (v14, v17)
   ↓
  06:05  COT/CFTC weekly (если новый release)              (v15)
   ↓
  06:10  Feature engineering + triple-barrier              (v14, v15)
   ↓
  06:15  v25 CPCV predict (latest p_up)                    (v25)
   ↓
  06:20  Apply v22 policy + v24 gates                      (v24)
   ↓
  06:25  Audit log записывается в v23_decision_audit_log   (v23)
   ↓
  19:30  python silver_paper_tinkoff.py --live --ticker SLVRUBF
   ↓
  19:35  Order поставлен в Tinkoff sandbox
   ↓
  Раз в неделю:
   ├─ python silver_paper_tinkoff.py --status
   ├─ python silver_assistant_v25_cpcv.py    (переобучить)
   └─ python silver_spread_estimator.py       (обновить spread оценку)
```

---

## 🎯 Roadmap

- [x] v22 → v25: replaced expanding window with CPCV for honest OOS
- [x] Tinkoff Invest paper trading (sandbox)
- [x] Realistic cost model (Corwin-Schultz spread + ATR slippage)
- [x] DSR / PSR / bootstrap CI для всех метрик
- [ ] **Live forward validation** (3-6 месяцев daily live mode)
- [ ] Multi-asset: gold/silver pair, copper, platinum
- [ ] LSTM stack on top of CPCV предсказаний
- [ ] Streamlit dashboard для paper trading мониторинга
- [ ] Telegram алерты на новые сигналы

---

## 📚 Документация

- [CHANGELOG.md](CHANGELOG.md) — Хронология версий v14 → v25 с инсайтами
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Детали pipeline + где какой код
- [docs/RESULTS.md](docs/RESULTS.md) — Все результаты в табличном виде
- [docs/PAPER_TRADING.md](docs/PAPER_TRADING.md) — Гайд по Tinkoff bridge

---

## ⚠️ Безопасность

1. **Никогда** не коммитить `.env` (он в `.gitignore`)
2. `.env.example` — публичный шаблон, **только placeholder'ы**, не настоящие токены
3. Tinkoff токен — только с правами **sandbox**, не full-access
4. Регулярно ротировать токены (раз в месяц)

---

## 📝 Лицензия

Исследовательский проект. Использование на свой страх и риск.
