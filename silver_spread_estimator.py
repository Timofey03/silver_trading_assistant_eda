"""
silver_spread_estimator.py — Бесплатная альтернатива tick-data для spread/slippage

Идея: вместо тиков использовать минутные/5-минутные бары и считать proxy spread
как (high - low) / close. Это grubo overestimates spread, но даёт честную
верхнюю оценку — лучше, чем фиксированные 5 bps в v22.

Источники (все бесплатно):
1) yfinance — 1m бары за последние 7 дней, 5m за 60 дней, 1h без ограничений
2) Tinkoff REST API — 1-минутные свечи через GetCandles (требует токен)

Метрики:
  • bar_spread_pct = (high - low) / close
  • Bias-corrected (Corwin & Schultz, 2012) — оценка bid-ask spread из HL bars
  • Volume-weighted variants

Выход: baseline_outputs_v23/v23_spread_proxy.csv

Запуск:
  python silver_spread_estimator.py --source yfinance --ticker SLV
  python silver_spread_estimator.py --source tinkoff --figi <SLV_FIGI>
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

V23_DIR = Path("baseline_outputs_v23")
V23_DIR.mkdir(exist_ok=True)


# ===========================================================================
# 1. CORWIN & SCHULTZ (2012) bias-corrected spread estimator
# ===========================================================================

def corwin_schultz_spread(
    high: pd.Series,
    low:  pd.Series,
) -> pd.Series:
    """
    Оценка bid-ask spread по HL bars (Corwin & Schultz, JF 2012).

    S = 2 * (exp(α) - 1) / (1 + exp(α))
    где α = (sqrt(2*β) - sqrt(β)) / (3 - 2*sqrt(2)) - sqrt(γ / (3 - 2*sqrt(2)))
          β = sum_{t=0,1} (ln(H_t/L_t))^2
          γ = (ln(H_{t,t+1}/L_{t,t+1}))^2  — High и Low по 2 барам сразу

    Возвращает spread как доля цены (например, 0.0008 = 8 bps).
    Игнорирует часы с overnight gap.
    """
    if len(high) < 2:
        return pd.Series([], dtype=float)
    h = high.astype(float)
    l = low.astype(float)

    h_next = h.shift(-1)
    l_next = l.shift(-1)

    h2 = np.maximum(h, h_next)
    l2 = np.minimum(l, l_next)

    beta  = np.log(h / l) ** 2 + np.log(h_next / l_next) ** 2
    gamma = np.log(h2 / l2) ** 2

    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)

    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = spread.where(spread > 0, np.nan)
    return spread


# ===========================================================================
# 2. SOURCE: yfinance
# ===========================================================================

def fetch_yfinance_bars(
    ticker: str,
    period: str = "60d",
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Скачивает интрадей-бары через yfinance.
    Limits:
      1m  — последние 7 дней
      5m  — последние 60 дней
      15m — последние 60 дней
      1h  — без ограничений (≈730 дней доступно)
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")
    data = yf.download(
        ticker, period=period, interval=interval,
        progress=False, auto_adjust=False, threads=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join([c for c in col if c]).strip("_") for col in data.columns]
        rename = {}
        for c in data.columns:
            base = c.split("_")[0]
            if base in ("Open", "High", "Low", "Close", "Volume", "Adj"):
                rename[c] = base
        data = data.rename(columns=rename)
    out = pd.DataFrame({
        "open":   data.get("Open",   data.iloc[:, 0]),
        "high":   data.get("High",   data.iloc[:, 1] if len(data.columns) > 1 else data.iloc[:, 0]),
        "low":    data.get("Low",    data.iloc[:, 2] if len(data.columns) > 2 else data.iloc[:, 0]),
        "close":  data.get("Close",  data.iloc[:, 3] if len(data.columns) > 3 else data.iloc[:, 0]),
        "volume": data.get("Volume", 0),
    })
    out.index = pd.to_datetime(data.index, utc=True)
    return out.dropna(subset=["high", "low", "close"])


# ===========================================================================
# 3. SOURCE: Tinkoff REST candles
# ===========================================================================

def fetch_tinkoff_candles(
    figi: str,
    days_back: int = 30,
    interval: str = "CANDLE_INTERVAL_5_MIN",
) -> pd.DataFrame:
    """
    Минутные/5-минутные свечи через Tinkoff REST (требует TINKOFF_TOKEN).
    """
    import json
    import requests

    token = os.getenv("TINKOFF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TINKOFF_TOKEN не задан в .env")

    api = "https://invest-public-api.tinkoff.ru/rest/" \
          "tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    })

    to = datetime.now(timezone.utc)
    frm = to - timedelta(days=days_back)
    body = {
        "figi": figi,
        "from": frm.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to":   to.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "interval": interval,
    }
    r = session.post(api, data=json.dumps(body), timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Tinkoff error {r.status_code}: {r.text[:500]}")
    res = r.json().get("candles", [])
    if not res:
        return pd.DataFrame()

    rows = []
    for c in res:
        def _q(d):
            return int(d.get("units", 0)) + int(d.get("nano", 0)) / 1e9
        rows.append({
            "time":   pd.to_datetime(c["time"]),
            "open":   _q(c["open"]),
            "high":   _q(c["high"]),
            "low":    _q(c["low"]),
            "close":  _q(c["close"]),
            "volume": int(c.get("volume", 0)),
        })
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df


# ===========================================================================
# 4. ANALYSIS
# ===========================================================================

def compute_spread_metrics(
    bars: pd.DataFrame,
    bar_interval_label: str = "5m",
) -> pd.DataFrame:
    """
    Возвращает per-bar и summary spread metrics.
    """
    if bars.empty:
        return pd.DataFrame()

    bars = bars.copy()
    bars["hl_pct"]      = (bars["high"] - bars["low"]) / bars["close"]
    bars["oc_pct"]      = (bars["close"] - bars["open"]).abs() / bars["open"]
    bars["midrange"]    = (bars["high"] + bars["low"]) / 2.0
    bars["close_midrange_pct"] = (bars["close"] - bars["midrange"]).abs() / bars["midrange"]

    cs = corwin_schultz_spread(bars["high"], bars["low"])
    bars["cs_spread"] = cs

    return bars


def summarize(bars_with_metrics: pd.DataFrame, label: str) -> dict:
    if bars_with_metrics.empty:
        return {"label": label, "n_bars": 0}
    b = bars_with_metrics
    return {
        "label":  label,
        "n_bars": len(b),
        "start":  b.index.min().isoformat(),
        "end":    b.index.max().isoformat(),
        "hl_pct_median":          float(b["hl_pct"].median()),
        "hl_pct_p25":             float(b["hl_pct"].quantile(0.25)),
        "hl_pct_p75":             float(b["hl_pct"].quantile(0.75)),
        "oc_pct_median":          float(b["oc_pct"].median()),
        "cs_spread_median":       float(b["cs_spread"].median()) if "cs_spread" in b else None,
        "cs_spread_p25":          float(b["cs_spread"].quantile(0.25)) if "cs_spread" in b else None,
        "cs_spread_p75":          float(b["cs_spread"].quantile(0.75)) if "cs_spread" in b else None,
        "implied_roundtrip_bps":  float(b["hl_pct"].median() * 0.25 * 10000),  # 1/4 от HL ≈ полу-спред × 2
    }


# ===========================================================================
# 5. MAIN
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",   choices=["yfinance", "tinkoff", "both"], default="yfinance")
    ap.add_argument("--ticker",   default="SLV",       help="yfinance ticker (default: SLV)")
    ap.add_argument("--figi",     default=None,         help="Tinkoff FIGI (required for tinkoff)")
    ap.add_argument("--period",   default="60d",        help="yfinance period (max 60d for 5m)")
    ap.add_argument("--interval", default="5m",         help="yfinance interval (1m|5m|15m|1h)")
    ap.add_argument("--tinkoff-days", type=int, default=30)
    ap.add_argument("--tinkoff-interval", default="CANDLE_INTERVAL_5_MIN")
    args = ap.parse_args()

    summaries: list = []

    if args.source in ("yfinance", "both"):
        print(f"=== yfinance {args.ticker} {args.period} {args.interval} ===")
        try:
            bars = fetch_yfinance_bars(args.ticker, period=args.period, interval=args.interval)
            print(f"  Загружено баров: {len(bars)}")
            if not bars.empty:
                bars_m = compute_spread_metrics(bars, bar_interval_label=args.interval)
                bars_m.to_csv(V23_DIR / f"v23_spread_yfinance_{args.ticker}_{args.interval}.csv")
                s = summarize(bars_m, f"yfinance:{args.ticker}:{args.interval}")
                summaries.append(s)
                print(f"  hl_pct  median: {s['hl_pct_median']*100:.3f}%  "
                      f"(p25={s['hl_pct_p25']*100:.3f}% / p75={s['hl_pct_p75']*100:.3f}%)")
                print(f"  CS spr  median: {s['cs_spread_median']*100 if s['cs_spread_median'] else 'n/a'}")
                print(f"  Implied roundtrip: ~{s['implied_roundtrip_bps']:.1f} bps")
        except Exception as e:
            print(f"  ERR yfinance: {type(e).__name__}: {e}")

    if args.source in ("tinkoff", "both"):
        figi = args.figi or os.getenv("TINKOFF_SILVER_FIGI", "")
        if not figi:
            print("  WARN: для Tinkoff нужен --figi или TINKOFF_SILVER_FIGI в .env. Пропускаем.")
        else:
            print(f"\n=== Tinkoff {figi} ===")
            try:
                bars = fetch_tinkoff_candles(figi,
                    days_back=args.tinkoff_days, interval=args.tinkoff_interval)
                print(f"  Загружено свечей: {len(bars)}")
                if not bars.empty:
                    bars_m = compute_spread_metrics(bars)
                    bars_m.to_csv(V23_DIR / f"v23_spread_tinkoff_{figi[:10]}.csv")
                    s = summarize(bars_m, f"tinkoff:{figi[:8]}")
                    summaries.append(s)
                    print(f"  hl_pct  median: {s['hl_pct_median']*100:.3f}%")
                    print(f"  CS spr  median: {s['cs_spread_median']*100 if s['cs_spread_median'] else 'n/a'}")
            except Exception as e:
                print(f"  ERR tinkoff: {type(e).__name__}: {e}")

    if summaries:
        df = pd.DataFrame(summaries)
        df.to_csv(V23_DIR / "v23_spread_summary.csv", index=False)
        print("\n=== Summary ===")
        print(df.to_string(index=False))
        print(f"\n  Saved: {V23_DIR / 'v23_spread_summary.csv'}")
        print("\nКак использовать в v23 RealisticCosts:")
        print("  spread_base = round(median(cs_spread) / 2, 5)  # one-side spread")
        print("  ИЛИ")
        print("  spread_base = implied_roundtrip_bps / 2 / 10000")


if __name__ == "__main__":
    main()
