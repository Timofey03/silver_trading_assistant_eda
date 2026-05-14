# Hosting Guide — Daily Auto-Training & Signals

Помощник может крутиться **бесплатно на GitHub Actions** — каждый день в 19:30 MSK переобучается, генерирует сигнал, исполняет в Tinkoff sandbox, и коммитит отчёты обратно в репо. Не нужен свой сервер.

---

## 🎯 Что вы получите

После настройки каждый рабочий день в репо появится:

```
daily_reports/
├── INDEX.md                          ← оглавление со ссылками
├── training/2026-05-15/
│   ├── summary.md                    ← человеко-читаемая сводка обучения
│   ├── summary.json                  ← машинная версия (для парсинга)
│   ├── v25_pnl_summary.csv
│   ├── v25_dsr_psr.csv
│   ├── v25_bootstrap_ci.csv
│   ├── v25_p_up_cpcv.csv
│   ├── v25_decisions.csv
│   ├── v25_policy.json
│   └── feature_drift_train_vs_recent.csv
└── trading/2026-05-15/
    ├── action.md                     ← 🟢/🔴/⚪ что делать с серебром
    ├── action.json                   ← машинный формат
    └── ALERT.json                    ← создаётся только если signal=BUY/SHORT
```

**Training report** содержит:
- Health-check: forward total return, Sharpe, MaxDD, DSR/PSR, bootstrap lower bound
- Policy (up_threshold, cooldown)
- PnL summary по всем splitам
- Statistical robustness (DSR/PSR)
- Feature drift (KS-test на 50 ключевых фичах: train vs последние 60 дней)

**Trading action** содержит:
- BUY / SHORT / HOLD с обоснованием
- Текущая цена SLVRUBF
- Статус paper trading в Tinkoff sandbox
- Полный портфель

---

## 🚀 Шаги настройки GitHub Actions

### Шаг 1: Заливка проекта на GitHub

```bash
# В корне проекта
git status   # проверьте, что .env НЕ в индексе

# Создайте репозиторий на GitHub (например: silver-assistant)
# Это можно через gh CLI:
gh repo create silver-assistant --private --source=. --remote=origin

# Или вручную:
# 1. https://github.com/new — создать репозиторий (Private рекомендуется)
# 2. Скопировать URL: git@github.com:USER/silver-assistant.git
git remote add origin git@github.com:USER/silver-assistant.git
git branch -M main
git push -u origin main
```

> ⚠️ **Critical**: убедитесь что `.env` НЕ ушёл в репо. Проверить: `git ls-files | grep .env` — должен показать ТОЛЬКО `.env.example`.

### Шаг 2: Добавить TINKOFF_TOKEN в GitHub Secrets

1. На странице репо: **Settings** → **Secrets and variables** → **Actions**
2. **New repository secret**
3. Name: `TINKOFF_TOKEN`
4. Value: ваш sandbox токен (можно тот же, что и в локальном .env)
5. Save

### Шаг 3: Включить GitHub Actions

Уже сделано — `.github/workflows/daily.yml` в репо. После первого push GitHub Actions автоматически активируется.

Проверить:
1. Перейти на **Actions** tab репозитория
2. Должна появиться workflow "Daily Silver Assistant"

### Шаг 4: Тестовый запуск

В **Actions** → **Daily Silver Assistant** → справа кнопка **Run workflow** → Run.

Через ~10 минут смотрите:
- В новом коммите появится `daily_reports/training/YYYY-MM-DD/` и `daily_reports/trading/YYYY-MM-DD/`
- В **Actions** → run → **Artifacts** можно скачать zip с отчётами

### Шаг 5 (опционально): GitHub Pages для красивого просмотра

1. **Settings** → **Pages**
2. Source: **Deploy from a branch**, Branch: `main`, Folder: `/daily_reports`
3. Save

После этого отчёты будут доступны по `https://USER.github.io/silver-assistant/INDEX.md`.

---

## 📅 Расписание

По умолчанию **Пн-Пт в 16:30 UTC = 19:30 MSK**. После закрытия MOEX.

Изменить — в `.github/workflows/daily.yml`:
```yaml
schedule:
  - cron: '30 16 * * 1-5'   # Пн-Пт
  # - cron: '0 9 * * 1-5'   # 9:00 UTC = 12:00 MSK (утренняя проверка)
```

> ⚠️ GitHub Actions расписание имеет задержку до 30 минут — не критично для дневной стратегии.

---

## 💰 Стоимость

| Сценарий | Цена |
|---|---|
| Private repo, ≤ 2000 минут/мес | **БЕСПЛАТНО** (включено в free plan) |
| Public repo | **БЕСПЛАТНО неограниченно** |

Один daily run = ~10 минут × 22 рабочих дня = **220 минут/мес**. Влезаете в free даже на private.

---

## 🔧 Структура daily_run.py

Скрипт делает 5 шагов:

1. **Refresh данных**: если v22_full_data.csv старше 20 часов — перезапускает `silver_assistant_v22_risk_aware.py --no-wf --no-mh` (быстрая версия без walk-forward)

2. **CPCV retrain**: `silver_assistant_v25_cpcv.py` — 15 folds, ~3-5 минут на GitHub runners

3. **Training report** (`build_training_report`):
   - Копирует v25 csv'ы
   - Вычисляет health-check метрики
   - KS-test drift на 50 фичах
   - Генерирует summary.md и summary.json

4. **Trading action** (`build_trading_report`):
   - Читает последний сигнал из v25_decisions.csv
   - Опционально исполняет в Tinkoff sandbox через `silver_paper_tinkoff.py --live`
   - Получает статус портфеля
   - Генерирует action.md, action.json, ALERT.json (если есть сигнал)

5. **Update INDEX.md**: оглавление со ссылками на все daily-отчёты

---

## 🔔 Уведомления

### Вариант A: GitHub Email Notifications
По умолчанию GitHub шлёт email при падении workflow. Достаточно для базового мониторинга.

### Вариант B: Telegram бот (рекомендую)
Добавьте в `.github/workflows/daily.yml` step после "Run daily pipeline":

```yaml
- name: Telegram alert on BUY/SHORT
  if: success()
  env:
    TG_TOKEN: ${{ secrets.TG_BOT_TOKEN }}
    TG_CHAT:  ${{ secrets.TG_CHAT_ID }}
  run: |
    if [ -f "daily_reports/trading/$(date -u +%Y-%m-%d)/ALERT.json" ]; then
      MSG=$(cat daily_reports/trading/$(date -u +%Y-%m-%d)/ALERT.json | python -c "
        import json, sys; a = json.load(sys.stdin)
        print(f\"🚨 Silver signal: {a['signal']} {a['ticker']} @ {a['price']} (p={a['p']:.2f})\")
      ")
      curl -s "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
        -d chat_id=$TG_CHAT -d text="$MSG"
    fi
```

Добавьте секреты `TG_BOT_TOKEN` и `TG_CHAT_ID` в Settings → Secrets.

### Вариант C: Email через GitHub Issues
В `daily_reports/trading/YYYY-MM-DD/ALERT.json` появилось — можно скриптом открывать GitHub Issue с лейблом "alert". Issues по умолчанию шлют email.

---

## 🔍 Локальный запуск (для отладки)

```bash
# Полный цикл (как на GitHub Actions)
python scripts/daily_run.py

# Без переобучения (быстро, использует существующие v25 результаты)
python scripts/daily_run.py --skip-training

# Без отправки в Tinkoff (только отчёты)
python scripts/daily_run.py --no-paper-trade

# Без обновления данных (offline-режим)
python scripts/daily_run.py --no-data-refresh --skip-training
```

---

## 📊 Просмотр результатов

### Через GitHub UI
- **Code** tab → `daily_reports/INDEX.md` — оглавление
- Каждый daily report — обычный markdown в репо, читается прямо в GitHub

### Через клон локально
```bash
git pull
ls daily_reports/training/
cat daily_reports/trading/$(date +%Y-%m-%d)/action.md
```

### Через JSON API (для интеграции)
```bash
# Последний сигнал
TODAY=$(date +%Y-%m-%d)
curl -H "Authorization: token $GH_PAT" \
  "https://raw.githubusercontent.com/USER/silver-assistant/main/daily_reports/trading/$TODAY/action.json"
```

---

## ⚠️ Что может пойти не так

| Проблема | Решение |
|---|---|
| Workflow упал на pip install | Закрепить версии в requirements.txt |
| Timeout 30 минут превышен | Закомментировать walk-forward в v22 (`--no-wf`) |
| TINKOFF_TOKEN не работает на GH | Создать новый sandbox-токен (через VPN если из ограниченной зоны) |
| Daily reports не коммитятся | Проверить `permissions: contents: write` в workflow |
| GitHub Actions не запускается ночью | Это known issue — actions при низкой нагрузке откладываются до 30 мин |
| `.env` случайно ушёл в репо | `git rm --cached .env; git commit; git push` + **немедленно отозвать токен** |

---

## 🎓 Альтернативы GitHub Actions

| Платформа | Плюсы | Минусы |
|---|---|---|
| **GitHub Actions** ⭐ | Бесплатно, прямо в репо, secrets management | 2000 мин/мес free для private |
| Render Cron | Простой setup | Free tier 90 дней неактивности → выключение |
| Railway | $5/мес постоянно | Платно |
| PythonAnywhere | Free scheduled tasks | Только 1 task/day на free |
| Yandex Cloud Functions | Российский, недорого | Сложнее setup, нужна карта |
| Свой VPS (Beget/Selectel) | Полный контроль | От 300₽/мес, надо администрировать |

Для этого проекта GitHub Actions — **оптимально**.
