# Paper Trading через Tinkoff Invest API

Руководство по запуску paper trading стратегии в sandbox Tinkoff.

## Подготовка

### 1. Создание sandbox токена

1. Зайти на [tinkoff.ru/invest/settings](https://www.tinkoff.ru/invest/settings)
2. Создать новый токен с правами **только sandbox** (НЕ full-access!)
3. Скопировать токен

### 2. Настройка .env

```bash
cp .env.example .env
# Откройте .env и вставьте свой токен
```

Содержимое `.env`:
```ini
TINKOFF_TOKEN=t.YOUR_NEW_SANDBOX_TOKEN_HERE
TINKOFF_MODE=sandbox
TINKOFF_SILVER_TICKER=SLVRUBF
```

> ⚠️ **Никогда** не коммитьте `.env`. Файл уже в `.gitignore`.

## Команды

### Поиск инструментов
```bash
python silver_paper_tinkoff.py --find SLV
python silver_paper_tinkoff.py --find silver
python silver_paper_tinkoff.py --find серебро
```

Доступные серебряные инструменты в Tinkoff:

| Ticker | FIGI | Тип | Особенности |
|---|---|---|---|
| **SLV** | `BBG000NDCRW7` | US ETF | ❌ Недоступен для РФ (санкции) |
| **SLVRUBF** | `FSLVRUB00000` | MOEX futures | ✅ Основной выбор, ~20k RUB notional/лот |
| **SLVRUB_TOM** | `BBG000VHQTD1` | currency | Спот-курс серебра, обычно не торгуется |

### Setup (один раз)

```bash
python silver_paper_tinkoff.py --setup --initial-rub 1000000
```

Создаёт sandbox-счёт, пополняет на 1M RUB. Account ID сохраняется в [baseline_outputs_v23/v23_sandbox_account.json](../baseline_outputs_v23/v23_sandbox_account.json).

**Почему 1M, а не 100k**: SLVRUBF — фьючерс. 1 лот = ~20,000 RUB notional. На 100k RUB поместится только 5 лотов = 5 одновременных позиций.

### Статус портфеля

```bash
python silver_paper_tinkoff.py --status
```

Печатает:
- Total / cash / futures positions
- Список открытых позиций с avg price
- Сохраняет snapshot в `v23_paper_portfolio_snapshots.jsonl`

### Replay (исторические сигналы)

```bash
# Dry-run — только лог, без реальных ордеров
python silver_paper_tinkoff.py --replay --ticker SLVRUBF \
    --since 2025-01-01 --dry-run

# Реальный replay (исполняет в sandbox)
python silver_paper_tinkoff.py --replay --ticker SLVRUBF \
    --since 2025-01-01 \
    --base-size 1000 --max-size 3000
```

**Параметры sizing для фьючерсов**:
- `--base-size` и `--max-size` для ETF/shares (USD/RUB)
- Для futures используется `futures_max_lots` (1-2 лота)
- Kelly масштаб: 1 лот при p_up~0.5, 2 лота при p_up~0.7+

⚠️ **Важно**: sandbox исполняет ордера по **текущей цене**, не по исторической. Replay не симулирует реальный бэктест, только проверяет mechanics.

### Live mode (production)

```bash
python silver_paper_tinkoff.py --live --ticker SLVRUBF
```

Читает **сегодняшний** сигнал из `v23_decision_audit_log.jsonl` и исполняет 1 ордер. Если HOLD — ничего не делает.

## Автоматизация — Windows Task Scheduler

Создать задачу, которая запускает live mode раз в день после 19:00 MSK (когда MOEX закрылся и сигналы готовы):

1. Открыть **Планировщик заданий** (Task Scheduler)
2. **Создать задачу...**
3. **Триггер**: ежедневно в 19:30
4. **Действие**: 
   ```
   Program: D:\silver_trading_assistant_eda\.venv\Scripts\python.exe
   Arguments: silver_paper_tinkoff.py --live --ticker SLVRUBF
   Start in: D:\silver_trading_assistant_eda
   ```
5. Сохранить

### Альтернативно — через PowerShell

Создать скрипт `run_daily.ps1`:
```powershell
cd D:\silver_trading_assistant_eda
& .\.venv\Scripts\python.exe silver_paper_tinkoff.py --live --ticker SLVRUBF 2>&1 |
    Tee-Object -FilePath "baseline_outputs_v23\v23_live_log.txt" -Append
```

И запускать через `schtasks.exe`:
```bash
schtasks /Create /TN "SilverPaperTrading" /TR "powershell.exe -File D:\silver_trading_assistant_eda\run_daily.ps1" /SC DAILY /ST 19:30
```

## Мониторинг

### Daily check
```bash
python silver_paper_tinkoff.py --status
```

### Weekly review
```bash
# Переобучить v25 с обновлёнными данными
python silver_assistant_v25_cpcv.py

# Сравнить paper P&L vs backtest expectation
# (открыть baseline_outputs_v23/v23_paper_trading_log.csv)
```

### Monthly drift check
```bash
# Перегенерировать audit log с новыми сигналами
python silver_assistant_v23_honest.py --drift-demo

# Проверить v23_feature_drift_*.csv: если новые фичи дрейфуют —
# модель устарела
```

## Troubleshooting

### "Instrument is not available for trading"
- SLV (US ETF) недоступен из РФ. Используйте SLVRUBF.

### "Not enough balance"
- Для фьючерсов 1 лот SLVRUBF = ~20k RUB notional, плюс margin
- Решение: пополнить sandbox или использовать `futures_max_lots=1`
- Pop-up: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); from silver_paper_tinkoff import TinkoffClient, _load_account_id; c = TinkoffClient(os.getenv('TINKOFF_TOKEN')); a = _load_account_id(c); c.sandbox_pay_in(a, 1_000_000)"`

### "Invalid UUID format"
- Уже исправлено — используется `uuid.uuid4()`. Если ошибка — проверьте, что версия `silver_paper_tinkoff.py` свежая.

### Лог пустой / нет сигналов на сегодня
- v23 audit log имеет данные только до даты последнего refresh
- Перегенерировать: `python silver_assistant_v23_honest.py --audit-log`

## Best Practices

1. **Запускайте live только после закрытия рынка** (19:00 MSK для MOEX). Дневные сигналы используют OHLC закрытия.
2. **Не перезапускайте --replay несколько раз** — каждый раз создаёт новые ордера.
3. **Проверяйте --status еженедельно** — следите за drawdown и distribution позиций.
4. **Ротируйте токен раз в месяц** — даже для sandbox.
5. **Не используйте сandbox-токен в production** — это разные права в API.

## Дальнейшие шаги

- [ ] Streamlit дашборд: визуализация paper trading vs backtest expectations
- [ ] Telegram алерты на новые сигналы
- [ ] Автоматическая сверка realized vs expected P&L (drift > 30% → alert)
- [ ] Exit логика: trailing stop в paper trading (сейчас только entry, ручной exit)
