"""tests/test_simulator.py — поведение simulator + regime filters."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ohlcv(n=200, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.2, n)
    vol = rng.integers(1000, 10000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=dates)


class TestSimulatorBasic:
    def test_entry_threshold_respected(self):
        from app.multi_asset.simulator import simulate_trades, TradeConfig
        df = _ohlcv()
        p = pd.DataFrame({"p_1": pd.Series(0.4, index=df.index)})  # ниже 0.48
        cfg = TradeConfig(entry_threshold=0.48, direction_label=1)
        trades, _ = simulate_trades(p, df, cfg)
        assert len(trades) == 0, "No trades should fire with p_up<threshold"

    def test_trail_stop_fires(self):
        from app.multi_asset.simulator import simulate_trades, TradeConfig
        df = _ohlcv()
        # Strong signal then crash
        p_vals = [0.9] * 100 + [0.5] * 100
        df["close"].iloc[50:] *= 0.5  # 50% crash mid-way
        df["low"].iloc[50:] *= 0.5
        p = pd.DataFrame({"p_1": pd.Series(p_vals, index=df.index)})
        cfg = TradeConfig(entry_threshold=0.8, exit_threshold=0.0,
                          trail_pct=0.10, max_hold_days=999,
                          direction_label=1)
        trades, _ = simulate_trades(p, df, cfg)
        # At least 1 trail exit should fire
        assert any(t.exit_reason == "trail" for t in trades)

    def test_max_hold_fires(self):
        from app.multi_asset.simulator import simulate_trades, TradeConfig
        df = _ohlcv(n=100)
        p = pd.DataFrame({"p_1": pd.Series(0.9, index=df.index)})
        cfg = TradeConfig(entry_threshold=0.8, exit_threshold=0.0,
                          trail_pct=1.0, max_hold_days=10,
                          cooldown_days=0, direction_label=1)
        trades, _ = simulate_trades(p, df, cfg)
        assert any(t.exit_reason == "max_hold" for t in trades)
        assert all(t.hold_days <= 10 for t in trades if t.exit_reason == "max_hold")


class TestMetrics:
    def test_sharpe_positive_for_winning_trades(self):
        from app.multi_asset.metrics import compute_all_metrics
        tdf = pd.DataFrame([
            {"entry_date": pd.Timestamp("2020-01-01"),
             "exit_date": pd.Timestamp("2020-02-01"),
             "net_return": 0.05, "gross_return": 0.05,
             "hold_days": 22, "exit_reason": "model_exit"},
            {"entry_date": pd.Timestamp("2020-03-01"),
             "exit_date": pd.Timestamp("2020-04-01"),
             "net_return": 0.03, "gross_return": 0.03,
             "hold_days": 22, "exit_reason": "model_exit"},
        ])
        m = compute_all_metrics(tdf)
        assert m["sharpe"] > 0, "Sharpe should be positive on winners"
        assert m["win_rate"] == 1.0

    def test_max_dd_negative_on_loss(self):
        from app.multi_asset.metrics import compute_all_metrics
        tdf = pd.DataFrame([
            {"entry_date": pd.Timestamp("2020-01-01"),
             "exit_date": pd.Timestamp("2020-02-01"),
             "net_return": -0.10, "gross_return": -0.10,
             "hold_days": 22, "exit_reason": "trail"},
        ])
        m = compute_all_metrics(tdf)
        assert m["max_dd"] <= -0.10


class TestDataQuality:
    def test_fix_ohlc_clamps(self):
        from app.multi_asset.data_quality import fix_ohlc_ordering
        df = pd.DataFrame({
            "open": [10.0], "high": [9.0], "low": [11.0], "close": [10.5],
        }, index=pd.bdate_range("2020-01-01", periods=1))
        fixed, n = fix_ohlc_ordering(df)
        assert n == 1
        assert fixed["high"].iloc[0] == 11.0
        assert fixed["low"].iloc[0] == 9.0

    def test_outlier_detection(self):
        from app.multi_asset.data_quality import detect_outliers
        # 200% spike day-over-day
        s = pd.DataFrame({"close": [100, 100, 100, 250, 100]},
                         index=pd.bdate_range("2020-01-01", periods=5))
        out = detect_outliers(s, threshold=0.5)
        assert int(out.sum()) >= 1


class TestRegimeFilters:
    def test_trend_filter_above_sma200(self):
        from app.multi_asset.regime_filters import trend_filter
        # Rising prices → all above SMA200
        prices = pd.Series(range(1, 251), index=pd.bdate_range("2020-01-01", periods=250))
        mask = trend_filter(prices, sma_period=200)
        # Recent values must be above SMA
        assert mask.iloc[-1] == True

    def test_volatility_filter_excludes_extreme(self):
        from app.multi_asset.regime_filters import volatility_filter
        n = 300
        close = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, n)),
                          index=pd.bdate_range("2020-01-01", periods=n))
        high = close + 0.5
        low = close - 0.5
        mask = volatility_filter(high, low, close, pctile=0.90)
        # Most days should be allowed (vol_low = True)
        assert mask.mean() > 0.5
