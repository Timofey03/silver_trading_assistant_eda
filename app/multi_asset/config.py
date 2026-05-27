"""Конфигурация multi-asset pipeline: тикеры, пути, периоды."""
from __future__ import annotations

from pathlib import Path

# =============================================================================
# Пути
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "multi_asset"
DATA_DIR.mkdir(parents=True, exist_ok=True)

METALS_DIR = DATA_DIR / "metals"
MACRO_DIR = DATA_DIR / "macro"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"
REPORTS_DIR = DATA_DIR / "reports"
for d in (METALS_DIR, MACRO_DIR, FEATURES_DIR, LABELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Период обучения
# =============================================================================
# Серебро на COMEX уже хорошо ликвидно с 2010, но FRED COT/TIPS дают
# полный набор с 2010-01-01. Берём с 2010 для широты регимов.
START_DATE = "2010-01-01"
# END_DATE — динамически = завтра (yfinance включает данные до end exclusive)
from datetime import date, timedelta as _td
END_DATE = (date.today() + _td(days=1)).isoformat()


# =============================================================================
# Тикеры металлов (yfinance continuous futures)
# =============================================================================
METALS = {
    "silver":    {"ticker": "SI=F", "name": "Silver",    "unit": "$/oz",  "is_target": True},
    "gold":      {"ticker": "GC=F", "name": "Gold",      "unit": "$/oz",  "is_target": False},
    "platinum":  {"ticker": "PL=F", "name": "Platinum",  "unit": "$/oz",  "is_target": False},
    "palladium": {"ticker": "PA=F", "name": "Palladium", "unit": "$/oz",  "is_target": False},
    "copper":    {"ticker": "HG=F", "name": "Copper",    "unit": "$/lb",  "is_target": False},
}


# =============================================================================
# Макроэкономические индикаторы
# =============================================================================
# FRED-серии (через pandas_datareader или yfinance proxy)
# COT — через CFTC API (отдельный модуль)
MACRO = {
    # Real interest rates (один из главных драйверов драгметаллов)
    "DGS10":   {"source": "fred", "name": "10-Year Treasury Yield",
                "freq": "D", "description": "Nominal 10Y yield"},
    "DFII10":  {"source": "fred", "name": "10-Year TIPS Yield",
                "freq": "D", "description": "Real 10Y rate"},
    "T10YIE":  {"source": "fred", "name": "10Y Breakeven Inflation",
                "freq": "D", "description": "Inflation expectations"},

    # Dollar strength
    "DTWEXBGS": {"source": "fred", "name": "USD Broad Index",
                 "freq": "D", "description": "Trade-weighted dollar"},

    # USDRUB для российской адаптации
    "USDRUB":  {"source": "yf", "name": "USDRUB",
                "ticker": "RUB=X", "freq": "D", "description": "USD to Ruble"},

    # Industrial demand (особенно важно для серебра)
    "INDPRO":  {"source": "fred", "name": "Industrial Production Index",
                "freq": "M", "description": "US industrial production"},

    # Volatility / fear
    "VIXCLS":  {"source": "fred", "name": "VIX Index",
                "freq": "D", "description": "Implied volatility S&P 500"},

    # Inflation
    "CPIAUCSL": {"source": "fred", "name": "CPI All Items",
                 "freq": "M", "description": "Consumer price index"},

    # Energy (commodity context)
    "DCOILWTICO": {"source": "fred", "name": "WTI Crude Oil",
                   "freq": "D", "description": "Oil price spot"},
}


# =============================================================================
# Multi-horizon label config
# =============================================================================
HORIZONS = [5, 10, 20, 60]  # торговых дней

# Adaptive barriers config
# TP/SL = volatility_multiplier × realized_vol_20d
ADAPTIVE_BARRIERS = {
    "base_multiplier": 1.5,   # базовый множитель ATR/vol
    "vol_window": 20,         # окно для realized vol
    "asymmetric_uptrend": (2.0, 1.0),  # (TP_mult, SL_mult) для uptrend
    "asymmetric_downtrend": (1.0, 2.0),  # для downtrend (если short разрешён)
    "range_symmetric": (1.2, 1.2),  # для боковика
}
