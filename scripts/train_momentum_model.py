"""
scripts/train_momentum_model.py — обучить отдельный momentum-классификатор.

Цель: дополнить E3b (mean-reversion) моделью которая хорошо предсказывает
ПРОДОЛЖЕНИЕ трендов. E3b не справляется когда силе ребро пробивает ATH.

Features для momentum (без cross-asset макро — фокус на trend dynamics):
- close / SMA_20, close / SMA_60, close / SMA_200 (где относительно МА)
- ROC: 5d, 20d, 60d returns
- ATR_14 normalized by close
- Breakout flags: > 60d high, > 120d high
- ATH distance: 1 - close / rolling_max(252d)
- Consecutive up days
- Volume surge: volume / 20d MA
- Bollinger position: (close - mid) / std

Label: тот же label_20 что у E3b (adaptive triple barrier).

Архитектура: HistGradientBoostingClassifier, walk-forward 1000/30/30,
predictions сохраняются в baseline_outputs_multiasset/momentum/predictions.parquet.

Ensemble (после): p_combined = max(p_e3b, p_momentum) — наиболее уверенный
голос побеждает.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


def make_momentum_features(silver: pd.DataFrame) -> pd.DataFrame:
    """Trend-focused features."""
    close = silver["close"]
    high = silver["high"]
    low = silver["low"]
    volume = silver["volume"]

    feat = pd.DataFrame(index=silver.index)
    # Position vs moving averages
    for w in [10, 20, 50, 100, 200]:
        sma = close.rolling(w, min_periods=w // 2).mean()
        feat[f"close_over_sma{w}"] = close / sma - 1
        feat[f"sma{w}_slope_5"] = sma.pct_change(5)

    # Return rates
    for w in [5, 10, 20, 60, 120]:
        feat[f"ret_{w}d"] = close.pct_change(w)

    # ATR normalized
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    feat["atr14_pct"] = atr14 / close

    # Breakouts
    for w in [20, 60, 120, 252]:
        h = close.rolling(w, min_periods=w // 2).max().shift(1)
        feat[f"breakout_{w}"] = (close > h).astype(float)
        feat[f"dist_to_high{w}"] = close / h - 1

    # Consecutive up days
    up = (close.diff() > 0).astype(int)
    feat["consec_up_5"] = up.rolling(5).sum()
    feat["consec_up_10"] = up.rolling(10).sum()

    # Volume surge
    feat["vol_surge_20"] = volume / volume.rolling(20).mean()

    # Bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_position"] = (close - sma20) / (2 * std20 + 1e-9)

    # Distance from ATH
    ath = close.rolling(252, min_periods=100).max()
    feat["dist_from_ath"] = close / ath - 1

    return feat


def walk_forward_train(features: pd.DataFrame, labels: pd.Series,
                        train_window: int = 1000, test_window: int = 30,
                        step: int = 30, horizon: int = 20) -> pd.DataFrame:
    """Walk-forward classifier — возвращает DataFrame с p_-1, p_0, p_1, pred."""
    embargo = max(1, int(0.05 * horizon))
    common = features.index.intersection(labels.index)
    X = features.loc[common]
    y = labels.loc[common]
    mask = y.notna() & X.notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]
    dates = X.index.tolist()
    n = len(X)

    print(f"  Training samples: {n}, features: {X.shape[1]}")
    if n < train_window:
        return pd.DataFrame()

    rows = []
    test_start = train_window
    while test_start + test_window <= n:
        test_end = test_start + test_window
        train_end = test_start - horizon - embargo
        train_start = max(0, train_end - train_window)
        X_tr = X.iloc[train_start:train_end].values
        y_tr = y.iloc[train_start:train_end].values
        X_te = X.iloc[test_start:test_end].values
        idx_te = X.index[test_start:test_end]

        if len(np.unique(y_tr)) < 2:
            test_start += step
            continue

        model = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=100,
            min_samples_leaf=30, random_state=42,
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        classes = list(model.classes_)
        pred = model.predict(X_te)

        for i, d in enumerate(idx_te):
            row = {"date": d}
            for c in [-1, 0, 1]:
                if c in classes:
                    row[f"p_{c}"] = float(proba[i, classes.index(c)])
                else:
                    row[f"p_{c}"] = 0.0
            row["pred"] = int(pred[i])
            row["y_true"] = float(y.iloc[test_start + i])
            rows.append(row)

        test_start += step

    return pd.DataFrame(rows).set_index("date")


def main() -> int:
    from app.multi_asset.metal_loader import load_metals
    from app.multi_asset.labels import build_multi_horizon_labels

    print("=" * 70)
    print(" Training MOMENTUM model (HistGB на trend features)")
    print("=" * 70)
    silver = load_metals()["silver"]
    print(f"  Silver: {len(silver)} rows, до {silver.index[-1].date()}")

    mom_features = make_momentum_features(silver).dropna()
    print(f"  Momentum features: {mom_features.shape[1]} cols × {len(mom_features)} rows")

    # Same labels как у E3b
    labels = build_multi_horizon_labels(
        silver["close"], silver["high"], silver["low"],
        horizons=[20], adaptive=True,
    )["label_20"]

    preds = walk_forward_train(mom_features, labels, train_window=1000,
                                test_window=30, step=30, horizon=20)
    print(f"\n  Generated {len(preds)} predictions")
    print(f"  Period: {preds.index.min().date()} -> {preds.index.max().date()}")

    # Accuracy
    if len(preds):
        acc = accuracy_score(preds["y_true"], preds["pred"])
        print(f"  OOS accuracy: {acc:.3f}")
        # p_1 distribution
        print(f"  p_1 stats: min={preds['p_1'].min():.2f}, max={preds['p_1'].max():.2f}, "
              f"mean={preds['p_1'].mean():.2f}")
        print(f"  Days p_1>=0.85: {(preds['p_1']>=0.85).sum()}")

    out_dir = ROOT / "baseline_outputs_multiasset" / "momentum"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / "predictions.parquet", compression="snappy")
    print(f"\n  Saved -> {out_dir / 'predictions.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
