"""Страница: настройки приложения и интеграций."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils import (
    get_current_signal, get_tinkoff_status, load_policy,
    inject_styles, top_signal_badge, rub,
)

st.set_page_config(page_title="Настройки", page_icon="⚙", layout="wide")
inject_styles()

st.markdown("# ⚙ Настройки")
top_signal_badge(get_current_signal())


# =============================================================================
# 1. Tinkoff connection
# =============================================================================

st.markdown("## 📍 Tinkoff подключение")

tinkoff = get_tinkoff_status()

col1, col2 = st.columns([2, 1])
with col1:
    if tinkoff.get("ok"):
        st.success(f"✅ Подключено · аккаунт `{tinkoff['account_id']}`")
        st.caption(f"Баланс: {rub(tinkoff['total']['value'])}")
    else:
        st.error(f"❌ {tinkoff.get('error', 'Неизвестная ошибка')}")
        st.caption("Проверьте .env файл и наличие токена")
with col2:
    if st.button("🔄 Проверить подключение", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("### Изменить токен")
with st.expander("Открыть форму обновления токена"):
    st.warning("⚠ Токен будет сохранён в `.env` файле локально. **НИКОГДА** не коммитьте `.env` в git!")
    new_token = st.text_input("Новый Tinkoff TOKEN", type="password",
                               placeholder="t.XXXXXXXX...")
    new_ticker = st.selectbox("Тикер серебра",
                               ["SLVRUBF", "SLV", "SLVRUB_TOM"], index=0,
                               help="SLVRUBF — MOEX futures (рекомендуется для РФ)")
    mode = st.radio("Режим", ["sandbox", "production"], index=0, horizontal=True,
                    help="⚠ production = реальные деньги! Используйте только после месяцев успешного sandbox.")
    if st.button("💾 Сохранить в .env"):
        if not new_token:
            st.error("Токен не может быть пустым")
        elif not new_token.startswith("t."):
            st.error("Tinkoff токены начинаются с 't.'")
        elif mode == "production":
            st.error("⛔ Production режим заблокирован. Используйте sandbox.")
        else:
            env_file = ROOT / ".env"
            try:
                env_file.write_text(
                    f"TINKOFF_TOKEN={new_token}\n"
                    f"TINKOFF_MODE={mode}\n"
                    f"TINKOFF_SILVER_TICKER={new_ticker}\n",
                    encoding="utf-8",
                )
                st.success(f"✅ Сохранено в `.env`. Перезагрузите приложение.")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Ошибка: {e}")


# =============================================================================
# 2. Расписание daily run
# =============================================================================

st.markdown("---")
st.markdown("## 📅 Расписание автообучения")

st.info("""
**Сейчас**: автоматический daily run настроен через GitHub Actions —
каждый рабочий день в 19:30 МСК (16:30 UTC).

[Открыть workflow на GitHub →](https://github.com/Timofey03/silver_trading_assistant_eda/actions)
""")

with st.expander("Как изменить расписание"):
    st.markdown("""
1. Открыть [.github/workflows/daily.yml](https://github.com/Timofey03/silver_trading_assistant_eda/blob/main/.github/workflows/daily.yml)
2. Найти строку с `cron`:
   ```yaml
   schedule:
     - cron: '30 16 * * 1-5'   # Пн-Пт в 16:30 UTC
   ```
3. Изменить cron-выражение:
   - `'30 16 * * 1-5'` → Пн-Пт 16:30 UTC = 19:30 MSK *(сейчас)*
   - `'0 9 * * 1-5'`   → Пн-Пт 9:00 UTC  = 12:00 MSK
   - `'30 7 * * *'`    → Каждый день в 7:30 UTC = 10:30 MSK
4. Commit + push → следующий запуск по новому расписанию
""")


# =============================================================================
# 3. Размеры позиций
# =============================================================================

st.markdown("---")
st.markdown("## 🎚 Режим агрессивности")

modes_csv = ROOT / "baseline_outputs_modes" / "modes_comparison.csv"
if modes_csv.exists():
    modes_df = pd.read_csv(modes_csv)
    fwd_modes = modes_df[modes_df["split"] == "forward"][
        ["mode", "n_trades", "win_rate", "total_return", "sharpe_ann"]
    ].copy()
    fwd_modes["win_rate"] = fwd_modes["win_rate"].apply(lambda x: f"{x*100:.0f}%")
    fwd_modes["total_return"] = fwd_modes["total_return"].apply(lambda x: f"{x*100:+.1f}%")
    fwd_modes.columns = ["Режим", "Сделок/год", "Win rate", "Total return", "Sharpe"]

    st.markdown("**Бэктест на forward split (2025+):**")
    st.dataframe(fwd_modes, hide_index=True, use_container_width=True)

    st.info("""
**🏆 Рекомендуется: Balanced** — лучший Sharpe (1.31) + win rate 69% + total return +44%.

**Conservative (текущий)** — слишком селективный (5 сделок/год, +21% return).
**Aggressive** — больше шума, win rate падает до 45%.
**Ultra** — теряет деньги (-22%).
""")

selected_mode = st.radio(
    "Выберите режим помощника",
    ["conservative", "balanced", "aggressive", "ultra"],
    index=1,  # balanced рекомендован
    help="Меняет пороги входа/выхода и cooldown. После смены — следующий daily run "
         "будет использовать новые параметры.",
    horizontal=True,
)

mode_params = {
    "conservative": {"p_up_in": 0.55, "p_up_out": 0.40, "cooldown": 15, "trail": "8%"},
    "balanced":     {"p_up_in": 0.52, "p_up_out": 0.45, "cooldown": 10, "trail": "7%"},
    "aggressive":   {"p_up_in": 0.50, "p_up_out": 0.48, "cooldown":  5, "trail": "5%"},
    "ultra":        {"p_up_in": 0.48, "p_up_out": 0.50, "cooldown":  3, "trail": "4%"},
}
mp = mode_params[selected_mode]
st.caption(f"Параметры режима **{selected_mode}**: "
            f"вход p_up≥{mp['p_up_in']}, выход p_up<{mp['p_up_out']}, "
            f"cooldown={mp['cooldown']}d, trailing stop {mp['trail']}")

if st.button("💾 Применить режим"):
    cfg_path = ROOT / "baseline_outputs_modes" / "active_mode.json"
    cfg_path.parent.mkdir(exist_ok=True)
    cfg_path.write_text(json.dumps({"mode": selected_mode}, indent=2), encoding="utf-8")
    st.success(f"✅ Режим '{selected_mode}' сохранён. Применится при следующем daily run.")


st.markdown("---")
st.markdown("## 💰 Размеры позиций")

policy = load_policy()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("up_threshold (текущий)", policy.get("up_threshold", "—"),
              help="Порог вероятности для генерации BUY-сигнала")
with col2:
    st.metric("cooldown (дней)", policy.get("cooldown", "—"),
              help="Минимум торговых дней между сигналами")
with col3:
    st.metric("Sizing", "Kelly",
              help="Размер позиции: 1 лот при p~0.55, 2 лота при p≥0.7")

st.info("""
**Текущие параметры sizing для SLVRUBF**:
- `base_size_rub = 2000` — минимум RUB на сделку
- `max_size_rub = 5000`  — максимум RUB
- `futures_max_lots = 2` — макс. лотов фьючерса за раз

Для изменения отредактируйте [silver_paper_tinkoff.py](https://github.com/Timofey03/silver_trading_assistant_eda/blob/main/silver_paper_tinkoff.py) и `silver_assistant_v25_cpcv.py`.

Policy `up_threshold` и `cooldown` подбирается **автоматически** при CPCV training на valid split.
Не меняйте вручную — это нарушит OOS-валидацию.
""")


# =============================================================================
# 4. Risk management gates
# =============================================================================

st.markdown("---")
st.markdown("## 🛡 Risk management (v24 gates)")

st.markdown("Гейты — пост-фильтры, отключающие сигналы в неблагоприятных условиях.")

g1, g2, g3, g4 = st.columns(4)
with g1:
    use_liquidity = st.checkbox("Liquidity gate", value=True,
                                 help="Блок сигналов в low-volume дни (< 50% от 60d median)")
with g2:
    use_vix = st.checkbox("VIX risk-off", value=True,
                          help="Блок LONG когда VIX > 25 и растёт")
with g3:
    use_gsr = st.checkbox("GSR extreme", value=False,
                          help="Блок при экстремуме gold/silver ratio z-score (|z60| > 2)")
with g4:
    use_dd = st.checkbox("Drawdown kill-switch", value=True,
                          help="Остановка при equity drawdown > 20%")

st.caption("⚠ Изменения применятся только при следующем daily run и могут потребовать переобучения.")

if st.button("💾 Сохранить настройки гейтов"):
    cfg_path = ROOT / "baseline_outputs_v24" / "v24_config.json"
    cfg = {
        "use_liquidity":      use_liquidity,
        "use_vix":            use_vix,
        "use_gsr":            use_gsr,
        "use_drawdown_kill":  use_dd,
        "saved_at":           datetime.now().isoformat(),
    }
    try:
        cfg_path.parent.mkdir(exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        st.success(f"✅ Сохранено в {cfg_path.name}")
    except Exception as e:
        st.error(f"Ошибка: {e}")


# =============================================================================
# 5. GitHub Actions secrets
# =============================================================================

st.markdown("---")
st.markdown("## ☁ GitHub Secrets (для облачного daily run)")

st.info("""
Для **автоматического daily run на GitHub Actions** нужны секреты в репозитории:

1. [Открыть Secrets settings](https://github.com/Timofey03/silver_trading_assistant_eda/settings/secrets/actions)
2. **New repository secret**
3. Добавить:
   - `TINKOFF_TOKEN` — ваш sandbox-токен (тот же что в локальном .env)
   - `TINKOFF_SILVER_TICKER` — `SLVRUBF` (опционально)

После этого GitHub Actions сможет торговать в Tinkoff sandbox автоматически каждый день.
""")


# =============================================================================
# 6. Telegram уведомления
# =============================================================================

st.markdown("---")
st.markdown("## 🔔 Telegram уведомления")

st.caption("Получать алерт при появлении BUY/SHORT сигнала")

with st.expander("Как настроить (5 минут)"):
    st.markdown("""
**Шаг 1: создать бота**
1. В Telegram открыть [@BotFather](https://t.me/BotFather)
2. `/newbot` → выбрать имя → получить **Bot Token**

**Шаг 2: узнать ваш chat_id**
1. Написать боту любое сообщение
2. Открыть: `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Найти `chat.id` в JSON ответе

**Шаг 3: добавить в GitHub Secrets**
- `TG_BOT_TOKEN` — токен бота
- `TG_CHAT_ID` — ваш chat ID

**Шаг 4: раскомментировать блок в [.github/workflows/daily.yml](https://github.com/Timofey03/silver_trading_assistant_eda/blob/main/.github/workflows/daily.yml)**:
```yaml
- name: Telegram alert
  if: success()
  env:
    TG_TOKEN: ${{ secrets.TG_BOT_TOKEN }}
    TG_CHAT:  ${{ secrets.TG_CHAT_ID }}
  run: |
    TODAY=$(date -u +%Y-%m-%d)
    if [ -f "daily_reports/trading/$TODAY/ALERT.json" ]; then
      MSG=$(python -c "
        import json
        a = json.load(open('daily_reports/trading/$TODAY/ALERT.json'))
        print(f'🚨 Silver: {a[\\\"signal\\\"]} {a[\\\"ticker\\\"]} @ {a[\\\"price\\\"]}')
      ")
      curl -s 'https://api.telegram.org/bot'$TG_TOKEN'/sendMessage' \\
        -d chat_id=$TG_CHAT -d text="$MSG"
    fi
```
""")


# =============================================================================
# 7. About
# =============================================================================

st.markdown("---")
st.markdown("## ℹ О приложении")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Модель**: v25 CPCV (Combinatorial Purged Cross-Validation)
**Стек**: Python 3.12+ · Streamlit · Plotly · HistGradientBoosting

**Текущий статус (forward 2025+)**:
- Total return: +53.34%
- Sharpe: 1.71
- Bootstrap 95% lower: +19%
- DSR: 0.40 (edge не статистически значим)
""")
with col2:
    st.markdown("""
**Источники данных**:
- yfinance (OHLC silver/gold/copper/oil/sp500/eurusd)
- yfinance ETF (TIP, RINF, HYG ← FRED proxy)
- CFTC/COT report
- Tinkoff Invest API (sandbox)

**Документация**:
- [README.md](https://github.com/Timofey03/silver_trading_assistant_eda/blob/main/README.md)
- [docs/ARCHITECTURE.md](https://github.com/Timofey03/silver_trading_assistant_eda/blob/main/docs/ARCHITECTURE.md)
- [docs/RESULTS.md](https://github.com/Timofey03/silver_trading_assistant_eda/blob/main/docs/RESULTS.md)
""")

st.caption("⚠ Не финансовый совет. Исследовательский проект. "
           "Используйте только sandbox-режим до 6+ месяцев живого live tracking.")
