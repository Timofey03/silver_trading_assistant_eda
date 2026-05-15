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
        df["ts_signal"] = pd.to_datetime(df["ts_signal"], errors="coerce")
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
def get_current_signal() -> dict:
    """
    Возвращает свежий сигнал: ищет последний день с не-NaN p_up.
    """
    d = load_decisions()
    if d.empty:
        return {"signal": "—", "p_up": None, "date": None, "price": None, "regime": "—"}

    # Берём последний день вообще (для текущей цены)
    last_row = d.iloc[-1]
    last_date = d.index[-1]

    # Последний день с валидным p_up (для собственно сигнала)
    valid = d[d["p_up"].notna()]
    if not valid.empty:
        sig_row = valid.iloc[-1]
        sig_date = valid.index[-1]
    else:
        sig_row = last_row
        sig_date = last_date

    return {
        "signal":      str(sig_row.get("signal_long", "HOLD")),
        "signal_short": str(sig_row.get("signal_short", "HOLD")),
        "p_up":        float(sig_row.get("p_up", 0)) if pd.notna(sig_row.get("p_up", float("nan"))) else None,
        "signal_date": sig_date,
        "current_date": last_date,
        "current_price": float(last_row.get("silver_close", 0)) if pd.notna(last_row.get("silver_close", float("nan"))) else None,
        "regime":      str(last_row.get("regime", "—")),
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
    """Большая карточка сигнала наверху страницы."""
    s = sig["signal"]
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
    date_txt = sig["signal_date"].strftime("%d %b %Y") if sig.get("signal_date") is not None else "—"

    st.markdown(f"""
    <div class="signal-card {css_class}">
        <p class="signal-title">{label}</p>
        <p class="signal-sub">
            Серебро: ${price:.2f} &nbsp;·&nbsp;
            Уверенность модели: {p_up_txt} &nbsp;·&nbsp;
            Сигнал от {date_txt}
        </p>
    </div>
    """, unsafe_allow_html=True)
