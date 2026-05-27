# Argentum в GitHub Codespaces

One-click демо для защиты диплома.

## Запуск

1. На странице репо нажмите **Code → Codespaces → Create codespace on main**
2. Подождите ~2 минуты (postCreateCommand: ставит deps, downloads cached data)
3. Откроется VSCode в браузере + автоматически запустятся:
   - Backend на порту 8000 (FastAPI)
   - Frontend на порту 3000 (Next.js)
4. Codespace откроет порт 3000 в браузере — это и есть UI

## Что находится внутри

- `argentum/backend/` — FastAPI (10+ endpoints)
- `argentum/frontend/` — Next.js 16 (App Router)
- `app/multi_asset/` — ML model (features, simulator, regime filters, OOD)
- `experiments/` — walk-forward experiments E1..E3b
- `tests/test_no_lookahead.py` — pytest гарантии отсутствия leakage

## Минимальные требования

- Codespace: 2-core, 4 GB RAM (default Basic) — достаточно
- Python 3.12 + Node 20 (ставятся feature'ами)

## Если порт 3000 не открылся

```bash
bash .devcontainer/start.sh
```

## Логи

```bash
tail -f /tmp/argentum-backend.log
tail -f /tmp/argentum-frontend.log
```
