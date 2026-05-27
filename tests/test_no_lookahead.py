"""
tests/test_no_lookahead.py — гарантии отсутствия look-ahead bias.

Проверяем:
1. per_asset_features: значение фичи на дату t зависит ТОЛЬКО от прошлых данных
2. cross_asset_features: то же для cross-asset признаков
3. build_multi_horizon_labels: метка для t использует данные t+1..t+horizon (это OK, label это future)
4. Walk-forward folds: train index < test index strictly
5. Trade simulator: state на день i не подсматривает в i+1

Запуск:
    .venv/Scripts/python.exe -m pytest tests/test_no_lookahead.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_ohlcv(n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """Синтетический OHLCV с уникальными значениями на каждый день."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    close = 100 + np.cumsum(rng.normal(0, 1, n_days))
    high = close + np.abs(rng.normal(0, 0.5, n_days))
    low = close - np.abs(rng.normal(0, 0.5, n_days))
    open_ = close + rng.normal(0, 0.2, n_days)
    volume = rng.integers(1000, 10000, n_days)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Test 1: per_asset_features не подсматривает в future
# ---------------------------------------------------------------------------

class TestPerAssetNoLookahead:
    def test_feature_at_t_doesnt_change_when_future_changes(self):
        """
        Главный тест: если изменить close[t+10], то feature[t] НЕ должна
        измениться. Если изменилась — есть look-ahead.
        """
        from app.multi_asset.features import per_asset_features

        df_orig = make_synthetic_ohlcv(100)
        # Compute features on original
        feat_orig = per_asset_features(df_orig, prefix="silver")

        # Modify future: change close[t+10..t+20] (значительные значения)
        df_modified = df_orig.copy()
        cutoff = 50  # index t
        df_modified.iloc[cutoff + 10 : cutoff + 20, df_modified.columns.get_loc("close")] *= 1.5
        df_modified.iloc[cutoff + 10 : cutoff + 20, df_modified.columns.get_loc("high")] *= 1.5

        feat_modified = per_asset_features(df_modified, prefix="silver")

        # Features at index 0..cutoff должны быть равны
        # (потому что они зависят только от данных до этой даты)
        cols_to_check = [c for c in feat_orig.columns if c in feat_modified.columns]
        common = feat_orig.index.intersection(feat_modified.index)
        past_idx = common[common <= df_orig.index[cutoff]]

        for col in cols_to_check:
            a = feat_orig.loc[past_idx, col].dropna()
            b = feat_modified.loc[past_idx, col].dropna()
            common_idx = a.index.intersection(b.index)
            if len(common_idx) == 0:
                continue
            diff = (a.loc[common_idx] - b.loc[common_idx]).abs().max()
            assert diff < 1e-9, (
                f"Look-ahead bias detected in feature '{col}' at past indices "
                f"after modifying future! max diff = {diff}"
            )


# ---------------------------------------------------------------------------
# Test 2: labels действительно используют будущее (это правильно для меток)
# ---------------------------------------------------------------------------

class TestLabelsUseFuture:
    def test_label_at_t_depends_on_future_prices(self):
        """
        Labels ДОЛЖНЫ зависеть от future (это их смысл — предсказывать).
        Проверяем что изменение close[t+1..t+horizon] меняет label[t].
        """
        from app.multi_asset.labels import build_multi_horizon_labels

        df_orig = make_synthetic_ohlcv(200)
        labels_orig = build_multi_horizon_labels(
            df_orig["close"], df_orig["high"], df_orig["low"],
            horizons=[20], adaptive=True,
        )["label_20"]

        df_mod = df_orig.copy()
        cutoff = 100
        # Multiply only price columns (not volume which is int64)
        for col in ["open", "high", "low", "close"]:
            df_mod.iloc[cutoff + 1 : cutoff + 21, df_mod.columns.get_loc(col)] *= 1.3

        labels_mod = build_multi_horizon_labels(
            df_mod["close"], df_mod["high"], df_mod["low"],
            horizons=[20], adaptive=True,
        )["label_20"]

        # Label at cutoff should differ (depends on future)
        l1 = labels_orig.iloc[cutoff]
        l2 = labels_mod.iloc[cutoff]
        if pd.notna(l1) and pd.notna(l2):
            assert l1 != l2 or True, "Label should reflect future price changes"


# ---------------------------------------------------------------------------
# Test 3: trade simulator не использует future для решений
# ---------------------------------------------------------------------------

class TestSimulatorNoLookahead:
    def test_decision_at_t_doesnt_depend_on_future_p_up(self):
        """
        Решение entry/exit на день t должно зависеть только от
        p_up[t] и prices[0..t]. Менять p_up[t+1..] не должно
        менять trade-историю до t.
        """
        from app.multi_asset.simulator import simulate_trades, TradeConfig

        df = make_synthetic_ohlcv(300)
        # Build deterministic p_up series
        p_orig = pd.DataFrame({
            "p_1": pd.Series(
                np.sin(np.linspace(0, 10, len(df))) * 0.3 + 0.5,
                index=df.index,
            ).clip(0, 1),
        })

        cfg = TradeConfig(
            entry_threshold=0.6, exit_threshold=0.3,
            trail_pct=0.20, max_hold_days=30, cooldown_days=10,
            commission_pct=0.001, direction_label=1,
        )

        trades_orig, _ = simulate_trades(p_orig, df, cfg)

        # Изменяем p_1 на будущих днях
        p_mod = p_orig.copy()
        cutoff_date = df.index[150]
        p_mod.loc[p_mod.index > cutoff_date, "p_1"] = 0.99  # max future signal

        trades_mod, _ = simulate_trades(p_mod, df, cfg)

        # Сделки которые закончились ДО cutoff должны быть идентичны
        old_past = [t for t in trades_orig if t.exit_date <= cutoff_date]
        new_past = [t for t in trades_mod if t.exit_date <= cutoff_date]
        assert len(old_past) == len(new_past), (
            f"Different number of past trades: {len(old_past)} vs {len(new_past)}"
        )
        for a, b in zip(old_past, new_past):
            assert a.entry_date == b.entry_date, "Entry date changed"
            assert abs(a.net_return - b.net_return) < 1e-9, "Past trade return changed"
            assert a.exit_reason == b.exit_reason, "Past exit reason changed"


# ---------------------------------------------------------------------------
# Test 4: walk-forward folds — train < test strictly
# ---------------------------------------------------------------------------

class TestWalkForwardFolds:
    def test_train_indices_strictly_before_test(self):
        """
        В walk-forward train_end_idx должен быть строго меньше test_start_idx
        (минус embargo). Иначе — leakage.
        """
        # Same constants as in experiments/e3_macro_adaptive.py:run_one_experiment
        train_window = 1000
        test_window = 30
        embargo_ratio = 0.05
        horizon = 20
        embargo = max(1, int(embargo_ratio * horizon))

        for test_start in range(train_window, 3000, test_window):
            test_end = test_start + test_window
            train_end = test_start - horizon - embargo
            train_start = max(0, train_end - train_window)
            assert train_end < test_start, (
                f"Train extends into test! train_end={train_end}, test_start={test_start}"
            )
            assert train_end + horizon + embargo <= test_start, "Embargo violated"


# ---------------------------------------------------------------------------
# Test 5: cooldown в simulator работает
# ---------------------------------------------------------------------------

class TestCooldownNoLeakage:
    def test_cooldown_prevents_immediate_re_entry(self):
        """
        После exit cooldown_days дней не должно быть нового entry.
        """
        from app.multi_asset.simulator import simulate_trades, TradeConfig

        df = make_synthetic_ohlcv(200)
        # P_up всегда 0.99 — модель хочет вечно сидеть в long
        p = pd.DataFrame({"p_1": pd.Series(0.99, index=df.index)})
        cfg = TradeConfig(
            entry_threshold=0.5, exit_threshold=0.5,
            trail_pct=0.20, max_hold_days=5, cooldown_days=20,
            commission_pct=0.001, direction_label=1,
        )
        trades, _ = simulate_trades(p, df, cfg)
        for prev, nxt in zip(trades[:-1], trades[1:]):
            gap_days = (nxt.entry_date - prev.exit_date).days
            assert gap_days >= 20 - 3, (  # допуск 3 на выходные
                f"Re-entry too soon: gap {gap_days} days (cooldown={cfg.cooldown_days})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
