"""Финальный аудит помощника Argentum.

Проверяет:
1. Свежесть данных (parquet, signal.json)
2. Синхронность UI ↔ Telegram (signal, P&L, current_price)
3. Корректность вычислений (formula sanity)
4. Cron-задачи (расписание, дедупликация)
5. Логика fallback (без backend)
6. Backend health (endpoints возвращают разумное)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parent.parent

OK = "✅"
WARN = "⚠"
FAIL = "❌"


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def check(label, condition, fail_msg="", warn_msg=""):
    if condition:
        print(f"  {OK} {label}")
        return True
    elif warn_msg:
        print(f"  {WARN} {label}: {warn_msg}")
        return None
    else:
        print(f"  {FAIL} {label}: {fail_msg}")
        return False


# =============================================================================
# 1. СВЕЖЕСТЬ ДАННЫХ
# =============================================================================
section("1. СВЕЖЕСТЬ ДАННЫХ")

silver_path = REPO / "data" / "multi_asset" / "metals" / "silver_daily.parquet"
silver = pd.read_parquet(silver_path)
silver_last = silver.index[-1].date()
today = datetime.now().date()
days_old = (today - silver_last).days
print(f"  Последняя свеча silver: {silver_last} ({days_old} дн. назад, today={today})")
check("silver.parquet свежий (< 4 дней)", days_old < 4,
      f"данные устарели на {days_old} дней")

usdrub_path = REPO / "data" / "multi_asset" / "macro" / "USDRUB.parquet"
usdrub = pd.read_parquet(usdrub_path)
print(f"  Последний USDRUB: {usdrub.index[-1].date()} = {usdrub.iloc[-1].iloc[0]:.2f}")

signal_dirs = sorted((REPO / "daily_reports" / "e3b" / "trading").iterdir(),
                     key=lambda p: p.name, reverse=True)
if signal_dirs:
    latest_signal = signal_dirs[0] / "signal.json"
    sig = json.loads(latest_signal.read_text(encoding="utf-8"))
    print(f"  Последний signal.json: {sig.get('date','')[:10]} signal={sig.get('signal')} p_up={sig.get('p_up',0):.3f}")
    sig_date = pd.Timestamp(sig["date"][:10]).date()
    check("signal.json не старше parquet", sig_date >= silver_last,
          f"signal {sig_date} устарел против данных {silver_last}")

# =============================================================================
# 2. BACKEND ENDPOINTS
# =============================================================================
section("2. BACKEND ENDPOINTS")

def api(path):
    try:
        r = requests.get(f"http://127.0.0.1:8000{path}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

health = api("/api/health")
check("/api/health отвечает", health is not None,
      "backend не запущен на 127.0.0.1:8000")

if health:
    signal_api = api("/api/signal")
    if signal_api:
        print(f"  /api/signal: date={signal_api.get('date')} signal={signal_api.get('signal')} "
              f"p_up={signal_api.get('p_up',0):.3f}")
        check("/api/signal возвращает свежий p_up",
              signal_api.get("p_up", 0) > 0,
              "p_up = 0 (backend cache не обновлён?)")

    pos_api = api("/api/positions")
    if pos_api and pos_api.get("positions"):
        p = pos_api["positions"][0]
        print(f"  /api/positions[0]: entry=₽{p.get('entry_price',0):.0f} "
              f"current=₽{p.get('current_price',0):.0f}")
        print(f"     market_entry=₽{p.get('market_entry_price',0):.0f} "
              f"market_current=₽{p.get('market_current_price',0):.0f}")
        print(f"     unrealized_pnl={p.get('unrealized_pnl_pct',0)*100:.2f}% "
              f"market_pnl={p.get('market_pnl_pct',0)*100:.2f}%")
        check("entry_price > 0", p.get("entry_price", 0) > 0)
        check("current_price > 0", p.get("current_price", 0) > 0)
        if p.get("market_entry_price", 0) > 0:
            calc = (p["market_current_price"] - p["market_entry_price"]) / p["market_entry_price"]
            check("market_pnl_pct согласован с market_entry/current",
                  abs(calc - p["market_pnl_pct"]) < 1e-6,
                  f"формула рассогласована (calc={calc:.4f} vs api={p['market_pnl_pct']:.4f})")
        else:
            check("market_entry_price > 0", False,
                  "market_entry_price = 0 — backend cache не вычислил (RESTART BACKEND)")

# =============================================================================
# 3. СИНХРОННОСТЬ UI ↔ TELEGRAM
# =============================================================================
section("3. СИНХРОННОСТЬ UI ↔ TELEGRAM")

sys.path.insert(0, str(REPO))
from app.telegram_portfolio import _read_signal_from_files, _read_positions_from_files

sig_fallback = _read_signal_from_files()
print(f"  TG signal fallback:    p_up={sig_fallback.get('p_up',0):.3f} signal={sig_fallback.get('signal')}")
if signal_api:
    print(f"  UI signal endpoint:    p_up={signal_api.get('p_up',0):.3f} signal={signal_api.get('signal')}")
    diff = abs(sig_fallback.get("p_up", 0) - signal_api.get("p_up", 0))
    check(f"p_up между UI и TG fallback расходится не больше 0.02",
          diff < 0.02, f"расхождение {diff:.3f}",
          warn_msg=f"небольшое расхождение {diff:.3f} (TG smoothing берёт другой набор файлов)" if diff < 0.1 else "")

pos_fallback = _read_positions_from_files()
if pos_fallback["positions"]:
    p_fb = pos_fallback["positions"][0]
    print(f"  TG fallback позиция:   entry=₽{p_fb.get('entry_price',0):.0f} "
          f"current=₽{p_fb.get('current_price',0):.0f} "
          f"market_pnl={p_fb.get('market_pnl_pct',0)*100:.2f}%")
    if pos_api:
        p_ui = pos_api["positions"][0]
        market_diff = abs(p_fb.get("market_pnl_pct", 0) - p_ui.get("market_pnl_pct", 0)) * 100
        check(f"market_pnl_pct между UI и TG расходится не больше 1.0 п.п.",
              market_diff < 1.0, f"расхождение {market_diff:.2f} п.п.",
              warn_msg=f"расхождение {market_diff:.2f} п.п. — возможно кэш USDRUB" if market_diff < 5 else "")

# =============================================================================
# 4. ФОРМУЛА: theoretical_rub_price
# =============================================================================
section("4. ФОРМУЛА theoretical RUB price")

silver_usd = float(silver["close"].iloc[-1])
usdrub_today = float(usdrub.iloc[-1].iloc[0])
theoretical = silver_usd * usdrub_today * 100 / 31.1035
print(f"  silver USD: ${silver_usd:.2f}")
print(f"  USDRUB: {usdrub_today:.2f}")
print(f"  Theoretical RUB/lot (100г): ₽{theoretical:.2f}")
print(f"  Формула: $ × USDRUB × 100/31.1035 = ${silver_usd:.2f} × {usdrub_today:.2f} × {100/31.1035:.4f} = ₽{theoretical:.2f}")

if pos_api and pos_api.get("positions"):
    backend_market = pos_api["positions"][0]["market_current_price"]
    diff_rub = abs(theoretical - backend_market)
    check(f"Backend market_current = моя theoretical (diff < ₽300)",
          diff_rub < 300, f"diff = ₽{diff_rub:.0f}",
          warn_msg=f"backend cache отстал на ₽{diff_rub:.0f} — нужен restart" if diff_rub < 500 else "")

# =============================================================================
# 5. ЦЕНА В UI: совпадает с актуальным yfinance + USDRUB?
# =============================================================================
section("5. ЦЕНЫ В UI (живые)")

price_api = api("/api/price")
fx_api = api("/api/fx")
if price_api:
    print(f"  /api/price.current = ${price_api.get('current',0):.2f}")
    check("UI цена силвера ≈ parquet close (diff < $0.5)",
          abs(price_api.get("current", 0) - silver_usd) < 0.5,
          f"UI ${price_api.get('current',0):.2f} vs parquet ${silver_usd:.2f}")
if fx_api:
    print(f"  /api/fx.usdrub = {fx_api.get('usdrub',0):.2f}")
    check("UI USDRUB ≈ parquet USDRUB (diff < 0.5)",
          abs(fx_api.get("usdrub", 0) - usdrub_today) < 0.5,
          f"UI {fx_api.get('usdrub',0):.2f} vs parquet {usdrub_today:.2f}")

# =============================================================================
# 6. CRON / TG DEDUP
# =============================================================================
section("6. CRON-РАСПИСАНИЕ И TG ДЕДУПЛИКАЦИЯ")

cron_yml = REPO / ".github" / "workflows" / "daily_e3b.yml"
if cron_yml.exists():
    content = cron_yml.read_text(encoding="utf-8")
    has_cron = "cron:" in content.lower()
    check("daily_e3b.yml существует с cron", has_cron)
    if "0 5,11,19" in content or "0 5 11 19" in content:
        print(f"  Cron: 08:00, 14:00, 22:00 МСК пн-пт")

# Проверим что send_telegram() пропускает info-сообщения
daily_py = REPO / "scripts" / "daily_e3b.py"
if daily_py.exists():
    content = daily_py.read_text(encoding="utf-8")
    check("send_telegram() пропускает info/is_repeat",
          'alert_type == "info"' in content and "is_repeat" in content,
          "дедупликация info-сообщений не реализована")

# =============================================================================
# 7. ИТОГИ
# =============================================================================
section("7. ИТОГИ АУДИТА")
print("""
Если все проверки выше зелёные (✅) — система работает корректно.
Жёлтые (⚠) — мелкие расхождения, не критично (обычно кэш backend).
Красные (❌) — нужно чинить.

Типовые причины расхождений UI ↔ Telegram:
  • USDRUB кэш backend отстаёт от parquet → перезапустить backend
  • signal.json smoothing берёт разные 3 дня → нормально, если разница < 0.05
  • current_price из Tinkoff sandbox имеет spread vs theoretical → это by design

Cron-расписание:
  • 3 раза в день для мониторинга данных (yfinance/FRED обновления)
  • Telegram отправляется только при СМЕНЕ сигнала (action), info-spam отключён
""")
