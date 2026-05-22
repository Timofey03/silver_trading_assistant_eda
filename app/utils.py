"""
app/utils.py — общие утилиты: загрузка данных, форматирование, кэширование.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent.parent
V22_DIR = REPO_ROOT / "baseline_outputs_v22"
V23_DIR = REPO_ROOT / "baseline_outputs_v23"
V25_DIR = REPO_ROOT / "baseline_outputs_v25"
PROD_DIR = REPO_ROOT / "baseline_outputs_prod"
REPORTS_DIR = REPO_ROOT / "daily_reports"


# =============================================================================
# Кэшируемые загрузчики
# =============================================================================

@st.cache_data(ttl=60)
def load_decisions() -> pd.DataFrame:
    """Решения v25 (signal_long, p_up, etc) с индексом по дате."""
    p = V25_DIR / "v25_decisions.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=[0]).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=60)
def load_full_data() -> pd.DataFrame:
    """v22_full_data: OHLC + все признаки + split."""
    p = V22_DIR / "v22_full_data.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=[0]).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=60)
def load_pnl_summary() -> pd.DataFrame:
    p = V25_DIR / "v25_pnl_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(ttl=60)
def load_dsr_psr() -> pd.DataFrame:
    p = V25_DIR / "v25_dsr_psr.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(ttl=60)
def load_bootstrap() -> pd.DataFrame:
    p = V25_DIR / "v25_bootstrap_ci.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(ttl=60)
def load_policy() -> dict:
    p = V25_DIR / "v25_policy.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def load_trades(split: str = "forward") -> pd.DataFrame:
    p = V25_DIR / f"v25_{split}_trades.csv"
    if not p.exists():
        return pd.DataFrame()
    t = pd.read_csv(p)
    if not t.empty:
        t["entry_date"] = pd.to_datetime(t["entry_date"])
        t["exit_date"] = pd.to_datetime(t["exit_date"])
    return t


@st.cache_data(ttl=60)
def load_paper_trading_log() -> pd.DataFrame:
    p = V23_DIR / "v23_paper_trading_log.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "ts_signal" in df.columns:
        # Robust parse: берём только дату (первые 10 символов YYYY-MM-DD)
        # Работает для ISO ("2026-05-16T13:17...") и простых дат ("2026-05-08")
        date_str = df["ts_signal"].astype(str).str.slice(0, 10)
        df["ts_signal"] = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
    return df


@st.cache_data(ttl=60)
def load_drift_report() -> pd.DataFrame:
    p = REPORTS_DIR / "training"
    if not p.exists():
        return pd.DataFrame()
    dates = sorted([d for d in p.iterdir() if d.is_dir()], reverse=True)
    if not dates:
        return pd.DataFrame()
    drift_file = dates[0] / "feature_drift_train_vs_recent.csv"
    if not drift_file.exists():
        return pd.DataFrame()
    return pd.read_csv(drift_file)


# =============================================================================
# Форматирование
# =============================================================================

def pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:+.{digits}f}%"


def rub(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M ₽"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}k ₽"
    return f"{x:,.2f} ₽".replace(",", " ")


def usd(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.2f}".replace(",", " ")


def signal_emoji(signal: str) -> str:
    return {"BUY": "🟢", "SHORT": "🔴", "SELL": "🔴", "HOLD": "⚪"}.get(signal, "❔")


def signal_color(signal: str) -> str:
    return {
        "BUY": "#00C853",     # green
        "SHORT": "#D32F2F",   # red
        "SELL": "#D32F2F",
        "HOLD": "#9E9E9E",    # gray
    }.get(signal, "#9E9E9E")


# =============================================================================
# Текущий сигнал (используется на всех экранах)
# =============================================================================

@st.cache_data(ttl=60)
def load_production_signal() -> dict:
    """Свежий silver production-сигнал."""
    p = PROD_DIR / "production_signal_today.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_gold_signal() -> dict:
    """Свежий gold production-сигнал с динамическим cooldown."""
    from datetime import datetime as _dt
    p = PROD_DIR / "gold_signal_today.json"
    if not p.exists():
        return {}
    try:
        sig = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not sig.get("ok"):
        return sig

    # Динамический пересчёт cooldown + signal
    today = pd.Timestamp(_dt.now().date())
    sig_date = pd.Timestamp(sig["date"])
    days_passed = max(0, (today - sig_date).days)
    trading_days_passed = int(days_passed * 5 / 7)
    original_cooldown = int(sig.get("cooldown_remaining", 0))
    cooldown_fresh = max(0, original_cooldown - trading_days_passed)

    p_up = float(sig["p_up"])
    threshold = float(sig.get("threshold", 0.49))
    exit_threshold = float(sig.get("exit_threshold", 0.43))

    if cooldown_fresh == 0 and p_up >= threshold:
        signal_fresh = "BUY"
    elif p_up < exit_threshold:
        signal_fresh = "SELL"
    else:
        signal_fresh = "HOLD"

    sig["signal_original"]    = sig["signal"]
    sig["signal"]             = signal_fresh
    sig["cooldown_remaining"] = cooldown_fresh
    sig["data_age_days"]      = days_passed
    sig["today_date"]         = today.strftime("%Y-%m-%d")
    return sig


@st.cache_data(ttl=60)
def load_production_predictions() -> pd.DataFrame:
    """История production-предсказаний за последние ~30 дней."""
    p = PROD_DIR / "production_predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=[0]).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=60)
def load_e3b_signal() -> dict | None:
    """Загрузить последний доступный E3b daily signal.

    Источник: `daily_reports/e3b/trading/YYYY-MM-DD/signal.json`
    Берём последний по дате каталог.
    """
    e3b_root = REPO_ROOT / "daily_reports" / "e3b" / "trading"
    if not e3b_root.exists():
        return None
    dirs = sorted([d for d in e3b_root.iterdir() if d.is_dir()],
                  reverse=True)
    for d in dirs:
        sig_file = d / "signal.json"
        if sig_file.exists():
            try:
                data = json.loads(sig_file.read_text(encoding="utf-8"))
                data["report_dir"] = d.name
                return data
            except Exception:
                continue
    return None


@st.cache_data(ttl=60)
def get_current_signal() -> dict:
    """
    Возвращает свежий сигнал.

    Приоритет: E3b daily → V25 production_inference → CPCV fallback.
    Cooldown пересчитывается ДИНАМИЧЕСКИ на основе сегодняшней даты.
    """
    from datetime import datetime as _dt

    # ===== E3b (новая модель, приоритет) =====
    e3b = load_e3b_signal()
    if e3b is not None:
        sig_date = pd.Timestamp(e3b.get("date"))
        today = pd.Timestamp(_dt.now().date())
        days_passed = max(0, (today - sig_date).days)

        signal = e3b.get("signal", "HOLD")
        p_up = float(e3b.get("p_up", 0))
        threshold = float(e3b.get("entry_threshold", 0.48))
        exit_threshold = float(e3b.get("exit_threshold", 0.35))

        # Stale-проверка: если данные старше 5 дней — HOLD + warning
        is_stale = days_passed >= 5

        return {
            "source":             "e3b_daily",
            "signal":             signal if not is_stale else "HOLD",
            "signal_original":    signal,
            "signal_short":       "HOLD",
            "p_up":               p_up,
            "above_threshold":    p_up >= threshold,
            "threshold":          threshold,
            "exit_threshold":     exit_threshold,
            "cooldown_remaining": int(e3b.get("cooldown_days", 25)),
            "signal_date":        sig_date,
            "today_date":         today,
            "data_age_days":      days_passed,
            "is_stale":           is_stale,
            "stale_reason":       (f"⚠ Сигнал старше {days_passed} дней. "
                                   f"Запустите retraining (daily_e3b.py) или дождитесь "
                                   f"автоматического обновления.")
                                  if is_stale else None,
            "current_date":       sig_date,
            "current_price":      float(e3b.get("close", 0)),
            "regime":             "—",
            "report_dir":         e3b.get("report_dir"),
            "model_features":     int(e3b.get("n_features_used", 30)),
            # Дедупликация (заполнено если signal.json содержит alert_type)
            "alert_type":         e3b.get("alert_type"),  # "action" | "info" | None
            "is_repeat":          bool(e3b.get("is_repeat", False)),
            "previous_signal":    e3b.get("previous_signal"),
            "alert_headline":     e3b.get("headline"),
            "alert_explanation":  e3b.get("explanation"),
            "run_time_utc":       e3b.get("run_time_utc"),
        }

    # ===== V25 (legacy fallback) =====
    prod = load_production_signal()
    d = load_decisions()

    last_price = None
    last_date = None
    if not d.empty:
        last_row = d.iloc[-1]
        last_date = d.index[-1]
        last_price = float(last_row.get("silver_close", 0)) if pd.notna(last_row.get("silver_close", float("nan"))) else None

    today = pd.Timestamp(_dt.now().date())

    if prod and prod.get("ok"):
        sig_date = pd.Timestamp(prod["date"])
        days_passed = max(0, (today - sig_date).days)
        original_cooldown = int(prod.get("cooldown_remaining", 0))
        trading_days_passed = int(days_passed * 5 / 7)
        cooldown_fresh = max(0, original_cooldown - trading_days_passed)

        p_up = float(prod["p_up"])
        threshold = float(prod.get("threshold", 0.49))
        exit_threshold = float(prod.get("exit_threshold", 0.43))

        # КОНСЕРВАТИВНАЯ ЛОГИКА:
        # Если данные свежие (< 2 дней) — пересчитываем сигнал нормально
        # Если данные устарели (>= 2 дней) — НЕ доверяем (p_up может уже не быть актуальным)
        STALE_THRESHOLD_DAYS = 2
        is_stale = days_passed >= STALE_THRESHOLD_DAYS

        if is_stale:
            # Не доверяем сигналу — показываем HOLD + предупреждение
            signal_fresh = "HOLD"
            stale_reason = (
                f"⚠ Данные устарели ({days_passed}d). "
                f"p_up={p_up:.0%} могло сильно измениться. "
                f"Нажмите 'Refresh signal' для свежего расчёта."
            )
        else:
            if cooldown_fresh == 0 and p_up >= threshold:
                signal_fresh = "BUY"
            elif p_up < exit_threshold:
                signal_fresh = "SELL"
            else:
                signal_fresh = "HOLD"
            stale_reason = None

        return {
            "source":            "production",
            "signal":            signal_fresh,
            "signal_original":   prod["signal"],
            "signal_short":      "HOLD",
            "p_up":              p_up,
            "p_up_trend_5d":     prod.get("p_up_trend_5d"),
            "p_up_trend_10d":    prod.get("p_up_trend_10d"),
            "p_up_trend_20d":    prod.get("p_up_trend_20d"),
            "above_threshold":   p_up >= threshold,
            "threshold":         threshold,
            "cooldown_remaining": cooldown_fresh,
            "signal_date":       sig_date,
            "today_date":        today,
            "data_age_days":     days_passed,
            "is_stale":          is_stale,
            "stale_reason":      stale_reason,
            "current_date":      last_date if last_date is not None else sig_date,
            "current_price":     last_price if last_price is not None else prod.get("silver_close"),
            "regime":            prod.get("regime", "—"),
        }

    # Fallback: CPCV
    if d.empty:
        return {"signal": "—", "p_up": None, "signal_date": None,
                "current_price": None, "regime": "—", "source": "none"}

    valid = d[d["p_up"].notna()]
    if not valid.empty:
        sig_row = valid.iloc[-1]
        sig_date = valid.index[-1]
    else:
        sig_row = d.iloc[-1]
        sig_date = d.index[-1]

    return {
        "source":         "cpcv_fallback",
        "signal":         str(sig_row.get("signal_long", "HOLD")),
        "signal_short":   str(sig_row.get("signal_short", "HOLD")),
        "p_up":           float(sig_row.get("p_up", 0)) if pd.notna(sig_row.get("p_up", float("nan"))) else None,
        "signal_date":    sig_date,
        "current_date":   last_date,
        "current_price":  last_price,
        "regime":         str(d.iloc[-1].get("regime", "—")),
    }


def get_kpis() -> dict:
    """Главные KPI для шапки + dashboard."""
    d = load_decisions()
    if d.empty:
        return {}

    last_price = float(d["silver_close"].iloc[-1])
    price_7d_ago = float(d["silver_close"].iloc[-min(7, len(d))]) if len(d) >= 7 else last_price
    price_30d_ago = float(d["silver_close"].iloc[-min(30, len(d))]) if len(d) >= 30 else last_price

    return {
        "last_price":         last_price,
        "ret_7d":             (last_price / price_7d_ago - 1.0) if price_7d_ago > 0 else 0,
        "ret_30d":            (last_price / price_30d_ago - 1.0) if price_30d_ago > 0 else 0,
        "last_update":        d.index[-1],
    }


# =============================================================================
# Tinkoff lazy-loader (избегаем падения если токена нет)
# =============================================================================

def get_tinkoff_status() -> dict:
    """Возвращает портфель Tinkoff или ошибку. Кэширует на 30 сек."""
    return _tinkoff_status_cached()


@st.cache_data(ttl=30, show_spinner=False)
def _tinkoff_status_cached() -> dict:
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    token = os.getenv("TINKOFF_TOKEN", "").strip()
    if not token:
        return {"ok": False, "error": "TINKOFF_TOKEN не задан в .env"}

    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from silver_paper_tinkoff import TinkoffClient, _load_account_id

        client = TinkoffClient(token)
        try:
            account_id = _load_account_id(client)
        except Exception:
            return {"ok": False, "error": "Sandbox-счёт не настроен. Запустите --setup"}

        portfolio = client.sandbox_portfolio(account_id)

        def _q(d: dict, currency_default: str = "rub") -> dict:
            if not d:
                return {"value": 0.0, "currency": currency_default}
            u = int(d.get("units", 0))
            n = int(d.get("nano", 0))
            return {"value": u + n / 1e9, "currency": d.get("currency", currency_default)}

        positions = []
        for pos in portfolio.get("positions", []):
            qty = pos.get("quantity", {})
            avg = pos.get("averagePositionPrice", {})
            cur = pos.get("currentPrice", {})
            positions.append({
                "instrument_type": pos.get("instrumentType", ""),
                "figi":            pos.get("figi", ""),
                "qty":             int(qty.get("units", 0)) + int(qty.get("nano", 0)) / 1e9,
                "avg_price":       _q(avg)["value"],
                "current_price":   _q(cur)["value"],
            })

        return {
            "ok":              True,
            "account_id":      account_id,
            "total":           _q(portfolio.get("totalAmountPortfolio")),
            "cash":            _q(portfolio.get("totalAmountCurrencies")),
            "futures":         _q(portfolio.get("totalAmountFutures")),
            "shares":          _q(portfolio.get("totalAmountShares")),
            "etf":             _q(portfolio.get("totalAmountEtf")),
            "expected_yield":  _q(portfolio.get("expectedYield")),
            "positions":       positions,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# =============================================================================
# Стиль / тема
# =============================================================================

def inject_styles() -> None:
    """Кастомные CSS-стили для всех страниц."""
    st.markdown("""
    <style>
    .stMetric {background: rgba(255,255,255,0.04); padding: 12px; border-radius: 8px;}
    .signal-card {
        text-align: center; padding: 30px; border-radius: 12px;
        margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .signal-buy   {background: linear-gradient(135deg, #00C853, #00897B); color: white;}
    .signal-sell  {background: linear-gradient(135deg, #D32F2F, #B71C1C); color: white;}
    .signal-hold  {background: linear-gradient(135deg, #424242, #616161); color: white;}
    .signal-title {font-size: 48px; font-weight: 700; margin: 0;}
    .signal-sub   {font-size: 18px; opacity: 0.95; margin-top: 8px;}
    div[data-testid="stMetric"] > label {opacity: 0.7;}
    </style>
    """, unsafe_allow_html=True)


def top_signal_badge(sig: dict) -> None:
    """Большая карточка сигнала наверху страницы.

    Показывает СЕГОДНЯШНЮЮ дату как актуальный момент рекомендации.
    Дата данных (на которых сделан прогноз) — в подзаголовке.
    """
    from datetime import datetime as _dt

    s = sig.get("signal", "HOLD")
    css_class = {
        "BUY": "signal-buy",
        "SHORT": "signal-sell",
        "SELL": "signal-sell",
        "HOLD": "signal-hold",
    }.get(s, "signal-hold")
    label = {
        "BUY":   "🟢 ПОКУПАТЬ",
        "SHORT": "🔴 ПРОДАВАТЬ",
        "SELL":  "🔴 ПРОДАВАТЬ",
        "HOLD":  "⚪ ДЕРЖАТЬ",
    }.get(s, f"❔ {s}")

    p_up_txt = f"{sig['p_up']:.0%}" if sig.get("p_up") is not None else "—"
    price = sig.get("current_price") or 0
    source = sig.get("source", "unknown")

    cooldown = sig.get("cooldown_remaining", 0)
    above = sig.get("above_threshold", False)
    extra = ""
    if s == "HOLD" and above and cooldown > 0:
        extra = f" &nbsp;·&nbsp; ⏳ cooldown ещё {cooldown}d → ожидается BUY"
    elif s == "HOLD" and above and cooldown == 0:
        extra = " &nbsp;·&nbsp; 🔔 cooldown истёк — BUY может сработать в следующий запуск"

    # СЕГОДНЯШНЯЯ дата + дата данных в подзаголовке
    today_str = _dt.now().strftime("%d %b %Y")
    data_age = sig.get("data_age_days", 0)

    if source == "production":
        if data_age == 0:
            data_note = ""
        elif data_age == 1:
            data_note = " · 📅 по данным за вчера"
        else:
            data_note = f" · 📅 по данным {data_age}d назад"
        source_lbl = "🟢 production-модель" + data_note
        date_txt = f"Рекомендация на {today_str}"
    elif source == "cpcv_fallback":
        source_lbl = "🟡 CPCV fallback (устаревший)"
        date_txt = f"Рекомендация на {today_str}"
    else:
        source_lbl = ""
        date_txt = today_str

    st.markdown(f"""
    <div class="signal-card {css_class}">
        <p class="signal-title">{label}</p>
        <p class="signal-sub">
            Серебро: ${price:.2f} &nbsp;·&nbsp;
            Уверенность модели: {p_up_txt} &nbsp;·&nbsp;
            {date_txt}{extra}
        </p>
        <p style="font-size: 13px; opacity: 0.75; margin: 4px 0 0;">{source_lbl}</p>
    </div>
    """, unsafe_allow_html=True)
