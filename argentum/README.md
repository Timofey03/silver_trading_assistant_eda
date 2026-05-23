# argentum.

**AI-помощник для рынка серебра.** Новая версия пользовательского интерфейса — FastAPI backend + Next.js 16 frontend поверх существующей E3b модели.

## Запуск

### 1. Backend (FastAPI на порту 8000)

```bash
cd argentum/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

→ Swagger docs: http://127.0.0.1:8000/docs

### 2. Frontend (Next.js на порту 3000)

```bash
cd argentum/frontend
npm install
npm run dev
```

→ http://127.0.0.1:3000

Запускать в двух отдельных терминалах.

## Структура

```
argentum/
├── backend/
│   ├── main.py                  ← FastAPI entry
│   ├── routers/
│   │   ├── signal.py            ← /api/signal (читает daily_reports/e3b/)
│   │   ├── price.py             ← /api/price (yfinance silver)
│   │   ├── history.py           ← /api/history (equity + trades)
│   │   ├── metrics.py           ← /api/metrics (Sharpe, Win, etc.)
│   │   ├── candles.py           ← /api/candles (OHLC + markers)
│   │   ├── tinkoff.py           ← /api/tinkoff/balance
│   │   └── explain.py           ← /api/explain (feature importance)
│   └── requirements.txt
│
└── frontend/
    ├── app/
    │   ├── layout.tsx           ← Header + footer
    │   ├── page.tsx             ← / — Сейчас (hero + price + explain)
    │   ├── history/page.tsx     ← /history — метрики + сделки
    │   └── settings/page.tsx    ← /settings — Tinkoff + модель
    ├── lib/
    │   ├── api.ts               ← API client (typed)
    │   └── utils.ts             ← cn(), formatRub, formatPct
    └── package.json
```

## Стек

- **Backend**: FastAPI + uvicorn + pandas/yfinance
- **Frontend**: Next.js 16 + React 19 + Tailwind v4 + TypeScript
- **Шрифты**: Inter (UI) + JetBrains Mono (цифры)
- **Иконки**: Lucide React
- **Стиль**: dark mode, Linear/Vercel минимализм

## Что уже работает

- ✅ Главная страница: Hero BUY/HOLD/SELL + цена с sparkline + feature importance
- ✅ История: 3 главных метрики + таблица последних 20 сделок
- ✅ Настройки: Tinkoff баланс + информация о модели
- ✅ Real signal из `daily_reports/e3b/trading/<latest>/signal.json`
- ✅ Real metrics из `baseline_outputs_multiasset/e3b_adaptive/metrics.json`

## Что планируется на след итерациях

- ⏳ TradingView Lightweight Charts свечной график на /history
- ⏳ WebSocket для real-time цены силера (yfinance polling 60с)
- ⏳ Кнопка «Купить через Tinkoff sandbox» (создаёт реальный ордер)
- ⏳ Light/dark theme toggle
- ⏳ Tooltips на всех метриках (что такое Sharpe и т.д.)
- ⏳ shadcn/ui компоненты для полировки
- ⏳ Эволюция модели E1→E4 + сравнение с конкурентами (expanders)
- ⏳ Animations через Framer Motion

## Не путать со старыми приложениями

- `simple_app.py` — старое облегчённое Streamlit
- `dashboard_app.py` — старое полное Streamlit
- `argentum/` — новое production-grade приложение

Все три могут работать параллельно (разные порты).
