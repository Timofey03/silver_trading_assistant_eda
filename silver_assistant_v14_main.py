"""
Silver Trading Assistant v14 — production-grade overhaul

Fixes vs v13:
1. Самостоятельная загрузка OHLC (yfinance) — не зависит от артефактов v9.
2. Реальный triple-barrier через High/Low вместо close-only аппроксимации.
3. Обучение на 10 годах (2013-2022) вместо 1 года (2023).
4. Purged walk-forward CV с embargo — нет утечки между фолдами.
5. HistGradientBoostingClassifier вместо LogisticRegression.
6. Данные FRED сдвинуты на 1 день (.shift(1)) — нет lookahead.
7. Честный бенчмарк buy-and-hold на каждой выборке.
8. Bootstrap доверительные интервалы для precision.
9. Калибровка вероятностей (Isotonic).

Run:
  python silver_assistant_v14_main.py
  python silver_assistant_v14_main.py --out-dir baseline_outputs_v14 --start 2013-01-01
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import yfinance as yf
except ImportError:
    raise ImportError("pip install yfinance>=0.2.40")

try:
    import pandas_datareader.data as web
    _HAS_DATAREADER = True
except Exception:
    _HAS_DATAREADER = False

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        brier_score_loss, log_loss, roc_auc_score,
    )
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    raise ImportError("pip install scikit-learn>=1.3.0")

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

TICKERS = {
    "silver": "SI=F",
    "gold":   "GC=F",
    "dxy":    "DX-Y.NYB",
    "vix":    "^VIX",
    "copper": "HG=F",
    "oil":    "CL=F",
    "sp500":  "^GSPC",
    "eurusd": "EURUSD=X",
}

FRED_SERIES = {
    "us10y":         "DGS10",
    "us2y":          "DGS2",
    "tips10y":       "DFII10",
    "breakeven_10y": "T10YIE",
    "fed_funds":     "FEDFUNDS",
}

SPLITS = {
    "train":   ("2013-01-01", "2022-12-31"),
    "valid":   ("2023-01-01", "2023-12-31"),
    "test":    ("2024-01-01", "2024-12-31"),
    "forward": ("2025-01-01", "2099-12-31"),
}

HORIZON   = 5       # дней до выхода
EMBARGO   = 5       # дней embargo после обучающего окна
MIN_TRAIN = 500     # минимум строк в окне обучения
COST      = 0.0005  # round-trip транзакционные издержки

TB_MULT    = 0.75   # барьер = mult * realized_vol_daily * sqrt(horizon)
TB_MIN     = 0.006  # минимум 0.6%
TB_MAX     = 0.04   # максимум 4%


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def fetch_ohlc(start: str, end: str) -> pd.DataFrame:
    """Скачивает OHLC для всех тикеров, возвращает единый DataFrame по торговым дням."""
    frames: Dict[str, pd.DataFrame] = {}
    for name, ticker in TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  ПРЕДУПРЕЖДЕНИЕ: нет данных для {ticker}")
                continue
            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            for col in ["open", "high", "low", "close", "volume"]:
                if col in raw.columns:
                    frames[f"{name}_{col}"] = raw[col]
        except Exception as e:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: ошибка загрузки {ticker}: {e}")

    if not frames:
        raise RuntimeError("Не удалось загрузить ни один тикер.")

    df = pd.DataFrame(frames)
    df.index.name = "Date"
    df = df.sort_index()
    # Заполнить пропуски методом forward fill (выходные/праздники)
    df = df.ffill().dropna(subset=["silver_close"])
    return df


def fetch_fred(start: str, end: str) -> pd.DataFrame:
    """Скачивает макро-ряды из FRED и СДВИГАЕТ на 1 день, чтобы избежать lookahead."""
    if not _HAS_DATAREADER:
        print("  pandas-datareader не установлен, FRED-данные пропущены.")
        return pd.DataFrame()

    frames: Dict[str, pd.Series] = {}
    for name, series_id in FRED_SERIES.items():
        try:
            s = web.DataReader(series_id, "fred", start, end)[series_id]
            s.index = pd.to_datetime(s.index)
            # Сдвиг на 1 день — FRED публикует с задержкой
            frames[name] = s.shift(1).ffill()
        except Exception as e:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: ошибка загрузки FRED {series_id}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Инженерия признаков
# ---------------------------------------------------------------------------

def _ret(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n)


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _zscore(s: pd.Series, n: int) -> pd.Series:
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


def _realized_vol(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change().rolling(n).std() * math.sqrt(252)


def build_features(df: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sl = out["silver_close"]
    gl = out["gold_close"]
    dx = out["dxy_close"]
    vx = out["vix_close"]

    # --- Silver returns & momentum ---
    for n in [1, 2, 3, 5, 10, 20]:
        out[f"silver_ret_{n}d"] = _ret(sl, n)
    out["silver_rsi_14"]         = _rsi(sl, 14)
    out["silver_zscore_20d"]     = _zscore(sl, 20)
    out["silver_zscore_60d"]     = _zscore(sl, 60)
    ma20 = sl.rolling(20).mean()
    ma60 = sl.rolling(60).mean()
    out["silver_dist_ma20"]      = sl / ma20 - 1
    out["silver_dist_ma60"]      = sl / ma60 - 1
    out["silver_ma20_slope_5d"]  = ma20.pct_change(5)
    out["silver_dist_high_20d"]  = sl / sl.rolling(20).max() - 1
    out["silver_dist_high_60d"]  = sl / sl.rolling(60).max() - 1

    # MACD
    ema12 = sl.ewm(span=12, adjust=False).mean()
    ema26 = sl.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    out["silver_macd_hist"] = macd - sig
    out["silver_macd_cross"] = (macd > sig).astype(float) - (macd < sig).astype(float)

    # --- Volatility ---
    rv20  = _realized_vol(sl, 20)
    rv60  = _realized_vol(sl, 60)
    out["silver_realized_vol_20d"]    = rv20
    out["silver_realized_vol_60d"]    = rv60
    out["silver_vol_ratio_5_20"]      = _realized_vol(sl, 5) / rv20
    out["silver_vol_ratio_20_60"]     = rv20 / rv60
    out["silver_vol_percentile_252d"] = rv20.rolling(252).rank(pct=True)

    # ATR (если есть H/L)
    if "silver_high" in out.columns and "silver_low" in out.columns:
        sh, slo = out["silver_high"], out["silver_low"]
        tr = pd.concat([sh - slo, (sh - sl.shift()).abs(), (slo - sl.shift()).abs()], axis=1).max(axis=1)
        out["silver_atr_14"] = tr.rolling(14).mean() / sl

    # --- Gold ---
    for n in [1, 2, 3, 5, 20]:
        out[f"gold_ret_{n}d"] = _ret(gl, n)
    out["gold_rsi_14"]        = _rsi(gl, 14)
    out["gold_zscore_20d"]    = _zscore(gl, 20)

    # Gold/Silver ratio
    out["gold_silver_ratio"]      = gl / sl
    out["gold_silver_ratio_z20"]  = _zscore(gl / sl, 20)
    out["gold_silver_ratio_z60"]  = _zscore(gl / sl, 60)

    # --- Dollar ---
    for n in [1, 5, 10, 20]:
        out[f"dxy_ret_{n}d"] = _ret(dx, n)
    out["dxy_zscore_20d"]    = _zscore(dx, 20)

    # --- VIX ---
    for n in [1, 5, 10]:
        out[f"vix_ret_{n}d"] = _ret(vx, n)
    out["vix_zscore_20d"]    = _zscore(vx, 20)
    out["vix_level"]         = vx

    # --- Cross-asset: Copper, Oil, S&P ---
    for asset in ["copper", "oil", "sp500", "eurusd"]:
        col = f"{asset}_close"
        if col in out.columns:
            s = out[col]
            for n in [1, 5, 10, 20]:
                out[f"{asset}_ret_{n}d"] = _ret(s, n)

    # --- Macro score (без FRED) ---
    dxy_z = out["dxy_zscore_20d"].fillna(0)
    vix_z = out["vix_zscore_20d"].fillna(0)
    out["macro_score"] = -dxy_z - 0.5 * vix_z

    # --- FRED: реальные доходности ---
    if not fred.empty:
        fred_aligned = fred.reindex(out.index, method="ffill")
        if "us10y" in fred_aligned.columns:
            out["us10y"]           = fred_aligned["us10y"]
            out["us10y_chg_20d"]   = fred_aligned["us10y"].diff(20)
        if "us2y" in fred_aligned.columns:
            out["us2y"]            = fred_aligned["us2y"]
            out["yield_curve_2_10"] = fred_aligned.get("us10y", pd.Series(dtype=float)) - fred_aligned.get("us2y", pd.Series(dtype=float))
        if "tips10y" in fred_aligned.columns:
            out["real_yield_10y"]      = fred_aligned["tips10y"]
            out["real_yield_chg_20d"]  = fred_aligned["tips10y"].diff(20)
        if "breakeven_10y" in fred_aligned.columns:
            out["breakeven_10y"]       = fred_aligned["breakeven_10y"]
            out["breakeven_chg_20d"]   = fred_aligned["breakeven_10y"].diff(20)

    # --- Trend regime (правило, без ML) ---
    out["trend_regime"] = "sideways"
    out.loc[sl > sl.rolling(60).mean(), "trend_regime"] = "uptrend"
    out.loc[sl < sl.rolling(60).mean(), "trend_regime"] = "downtrend"

    # --- Vol regime ---
    med_vol = rv20.rolling(252).median()
    out["vol_regime"] = "medium"
    out.loc[rv20 > med_vol * 1.4, "vol_regime"] = "high"
    out.loc[rv20 < med_vol * 0.7, "vol_regime"] = "low"

    return out


# ---------------------------------------------------------------------------
# Triple-barrier с OHLC High/Low
# ---------------------------------------------------------------------------

def add_triple_barrier_labels(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Использует реальные High/Low для обнаружения барьеров.
    Если данных нет — fallback на close-only (как v13, но честно помечен).
    """
    out = df.sort_index().copy()
    close  = out["silver_close"].values
    rv20   = out.get("silver_realized_vol_20d", pd.Series(np.nan, index=out.index)).values
    daily_vol = rv20 / math.sqrt(252)
    barrier = np.clip(TB_MULT * daily_vol * math.sqrt(horizon), TB_MIN, TB_MAX)
    barrier = np.where(np.isfinite(barrier), barrier, TB_MIN)

    has_hl = ("silver_high" in out.columns and "silver_low" in out.columns)
    if has_hl:
        highs = out["silver_high"].values
        lows  = out["silver_low"].values
        mode  = "ohlc"
    else:
        mode  = "close_only"

    labels, hit_days, hit_rets = [], [], []
    n = len(out)
    for i in range(n):
        entry = close[i]
        b     = barrier[i]
        if not np.isfinite(entry) or i + 1 >= n:
            labels.append(np.nan); hit_days.append(np.nan); hit_rets.append(np.nan)
            continue
        tp, sl_ = b, -b
        label = "NEUTRAL"; hd = horizon; hr = np.nan
        for j in range(1, horizon + 1):
            if i + j >= n:
                label = np.nan; hd = np.nan; hr = np.nan; break
            if has_hl:
                high_ret = highs[i + j] / entry - 1
                low_ret  = lows[i + j]  / entry - 1
                if high_ret >= tp and low_ret <= sl_:
                    # Оба барьера в один день — определяем по закрытию
                    cl_ret = close[i + j] / entry - 1
                    label  = "UP" if cl_ret > 0 else "DOWN"
                    hd = j; hr = cl_ret; break
                elif high_ret >= tp:
                    label = "UP";   hd = j; hr = high_ret; break
                elif low_ret <= sl_:
                    label = "DOWN"; hd = j; hr = low_ret;  break
                hr = close[i + j] / entry - 1
            else:
                r = close[i + j] / entry - 1
                if r >= tp:   label = "UP";   hd = j; hr = r; break
                if r <= sl_:  label = "DOWN"; hd = j; hr = r; break
                hr = r
        labels.append(label); hit_days.append(hd); hit_rets.append(hr)

    out["tb_label"]      = labels
    out["tb_hit_day"]    = hit_days
    out["tb_hit_return"] = hit_rets
    out["tb_barrier"]    = barrier
    out["tb_mode"]       = mode
    return out


# ---------------------------------------------------------------------------
# Purged walk-forward CV с embargo
# ---------------------------------------------------------------------------

def purged_walk_forward_splits(
    index: pd.DatetimeIndex,
    n_train_years: int = 3,
    n_test_months: int = 6,
    embargo_days: int = EMBARGO,
    horizon: int = HORIZON,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Генерирует (train_idx, test_idx) пары.
    Purge: из обучения удаляются строки, перекрывающиеся с тестом (horizon дней).
    Embargo: дополнительно удаляются embargo_days после конца обучения.
    """
    splits = []
    dates  = sorted(index)
    start  = dates[0]
    end    = dates[-1]

    test_start = start + pd.DateOffset(years=n_train_years)
    while test_start < end:
        test_end = test_start + pd.DateOffset(months=n_test_months)
        train_cutoff = test_start - pd.Timedelta(days=horizon + embargo_days)

        train_idx = index[(index >= start) & (index <= train_cutoff)]
        test_idx  = index[(index >= test_start) & (index < test_end)]

        if len(train_idx) >= MIN_TRAIN and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        test_start = test_end

    return splits


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    "silver_ret_1d", "silver_ret_2d", "silver_ret_3d", "silver_ret_5d",
    "silver_ret_10d", "silver_ret_20d",
    "silver_rsi_14", "silver_zscore_20d", "silver_zscore_60d",
    "silver_dist_ma20", "silver_dist_ma60",
    "silver_ma20_slope_5d", "silver_dist_high_20d", "silver_dist_high_60d",
    "silver_macd_hist", "silver_macd_cross",
    "silver_realized_vol_20d", "silver_realized_vol_60d",
    "silver_vol_ratio_5_20", "silver_vol_ratio_20_60",
    "silver_vol_percentile_252d", "silver_atr_14",
    "gold_ret_1d", "gold_ret_2d", "gold_ret_3d", "gold_ret_5d", "gold_ret_20d",
    "gold_rsi_14", "gold_zscore_20d",
    "gold_silver_ratio", "gold_silver_ratio_z20", "gold_silver_ratio_z60",
    "dxy_ret_1d", "dxy_ret_5d", "dxy_ret_10d", "dxy_ret_20d", "dxy_zscore_20d",
    "vix_ret_1d", "vix_ret_5d", "vix_ret_10d", "vix_zscore_20d", "vix_level",
    "copper_ret_1d", "copper_ret_5d", "copper_ret_10d", "copper_ret_20d",
    "oil_ret_1d", "oil_ret_5d", "oil_ret_10d", "oil_ret_20d",
    "sp500_ret_1d", "sp500_ret_5d", "sp500_ret_20d",
    "eurusd_ret_1d", "eurusd_ret_5d",
    "macro_score",
    "us10y", "us10y_chg_20d",
    "real_yield_10y", "real_yield_chg_20d",
    "breakeven_10y", "breakeven_chg_20d",
    "yield_curve_2_10",
]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=1.0,
        learning_rate=0.05,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        class_weight="balanced",
        random_state=42,
    )


def train_and_calibrate(X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    base = build_model()
    # CalibratedClassifierCV с cv="prefit" требует отдельного cal-набора;
    # используем cv=3 для простоты
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_precision(y_true: np.ndarray, n: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """Bootstrap 95% CI для доли UP среди сигналов."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(y_true, size=len(y_true), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_split(
    df: pd.DataFrame, split_name: str,
    model: CalibratedClassifierCV, feature_cols: List[str],
) -> Dict[str, object]:
    d = df[df["split"] == split_name].dropna(subset=["tb_label"]).copy()
    if d.empty:
        return {"split": split_name, "n": 0}

    X = d[feature_cols]
    y = d["tb_label"].astype(str)

    classes = list(model.classes_)
    proba   = model.predict_proba(X)
    pred    = model.predict(X)

    result: Dict[str, object] = {
        "split":             split_name,
        "n":                 len(d),
        "label_up_rate":     (y == "UP").mean(),
        "label_neutral_rate": (y == "NEUTRAL").mean(),
        "label_down_rate":   (y == "DOWN").mean(),
        "accuracy":          accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "tb_mode":           d["tb_mode"].iloc[0] if "tb_mode" in d.columns else "unknown",
    }

    if "UP" in classes and "DOWN" in classes:
        up_idx  = classes.index("UP")
        dn_idx  = classes.index("DOWN")
        result["auc_up_vs_rest"] = roc_auc_score(
            (y == "UP").astype(int), proba[:, up_idx]
        )
        result["brier_up"] = brier_score_loss(
            (y == "UP").astype(int), proba[:, up_idx]
        )

    try:
        result["log_loss"] = log_loss(y, proba, labels=classes)
    except Exception:
        result["log_loss"] = np.nan

    return result


# ---------------------------------------------------------------------------
# Политика сигналов
# ---------------------------------------------------------------------------

def apply_policy(
    df: pd.DataFrame, model: CalibratedClassifierCV,
    feature_cols: List[str],
    up_threshold: float = 0.40,
    margin_threshold: float = 0.10,
    down_cap: float = 0.30,
    cooldown: int = 5,
) -> pd.DataFrame:
    out = df.copy().sort_index()
    X   = out[feature_cols]
    classes = list(model.classes_)
    proba   = model.predict_proba(X)

    up_i  = classes.index("UP")   if "UP"      in classes else None
    dn_i  = classes.index("DOWN") if "DOWN"     in classes else None
    ne_i  = classes.index("NEUTRAL") if "NEUTRAL" in classes else None

    out["p_up"]      = proba[:, up_i]  if up_i  is not None else 0.0
    out["p_down"]    = proba[:, dn_i]  if dn_i  is not None else 0.0
    out["p_neutral"] = proba[:, ne_i]  if ne_i  is not None else 0.0
    out["up_margin"] = out["p_up"] - out[["p_down", "p_neutral"]].max(axis=1)

    raw = (
        (out["p_up"] >= up_threshold) &
        (out["up_margin"] >= margin_threshold) &
        (out["p_down"] <= down_cap)
    )
    out["raw_buy"] = raw

    signals, reasons = [], []
    last_buy = -9999
    for i, (idx, ok) in enumerate(zip(out.index, raw)):
        if ok and i - last_buy > cooldown:
            signals.append("BUY"); reasons.append("triple_barrier_edge"); last_buy = i
        elif ok:
            signals.append("HOLD"); reasons.append("cooldown")
        else:
            signals.append("HOLD"); reasons.append("no_edge")

    out["signal"] = signals
    out["reason"] = reasons
    return out


def select_policy(valid_df: pd.DataFrame, model: CalibratedClassifierCV, feature_cols: List[str]) -> dict:
    """Grid search на valid — выбираем параметры политики."""
    best_obj = -np.inf
    best_params: dict = {}

    for up_thr in [0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55]:
        for margin in [0.0, 0.05, 0.10, 0.15]:
            for down_cap in [0.25, 0.30, 0.35, 0.40]:
                for cooldown in [3, 5, 7]:
                    tmp = apply_policy(valid_df, model, feature_cols, up_thr, margin, down_cap, cooldown)
                    sigs = tmp[tmp["signal"] == "BUY"]
                    n = len(sigs)
                    if n < 5:
                        continue
                    correct = int((sigs["tb_label"] == "UP").sum()) if "tb_label" in sigs.columns else 0
                    base = float((tmp["tb_label"] == "UP").mean()) if "tb_label" in tmp.columns else 0.5
                    prec = correct / n
                    lift = prec - base
                    lo, _ = wilson_ci(correct, n)
                    obj = (lo - base) + 0.003 * min(n, 30) if lift > 0 else -999
                    if obj > best_obj:
                        best_obj = obj
                        best_params = {
                            "up_threshold": up_thr,
                            "margin_threshold": margin,
                            "down_cap": down_cap,
                            "cooldown": cooldown,
                        }

    return best_params if best_params else {
        "up_threshold": 0.46, "margin_threshold": 0.10,
        "down_cap": 0.30, "cooldown": 5,
    }


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def compute_guardrails(df: pd.DataFrame, split: str) -> dict:
    d = df[(df["split"] == split)].copy()
    labeled = d.dropna(subset=["tb_label"])
    sigs    = d[d["signal"] == "BUY"]
    n       = len(sigs)

    if n == 0:
        return {"split": split, "n_signals": 0, "warning": "no_signals"}

    correct = int((sigs["tb_label"] == "UP").sum()) if "tb_label" in sigs.columns else 0
    base    = float((labeled["tb_label"] == "UP").mean()) if len(labeled) else np.nan
    prec    = correct / n
    lo, hi  = wilson_ci(correct, n)
    lift    = prec - base

    warnings_list = []
    if n < 30:               warnings_list.append("small_sample")
    if np.isfinite(lo) and np.isfinite(base) and lo <= base:
        warnings_list.append("ci_lower_not_above_base")
    if lift < 0:             warnings_list.append("negative_lift")

    return {
        "split":             split,
        "n_signals":         n,
        "correct_over_n":    f"{correct}/{n}",
        "precision":         prec,
        "wilson_95_low":     lo,
        "wilson_95_high":    hi,
        "base_up_rate":      base,
        "lift_vs_base":      lift,
        "warning":           ";".join(warnings_list) if warnings_list else "OK",
    }


# ---------------------------------------------------------------------------
# Бэктест с бенчмарками
# ---------------------------------------------------------------------------

def backtest_strategy(df: pd.DataFrame, split: str, horizon: int = HORIZON) -> pd.DataFrame:
    d = df[df["split"] == split].sort_index().reset_index()
    trades = []
    i = 0
    while i < len(d) - 1:
        if d.loc[i, "signal"] != "BUY":
            i += 1; continue
        entry_i = i + 1
        if entry_i >= len(d): break
        entry = float(d.loc[entry_i, "silver_close"])
        b     = float(d.loc[i, "tb_barrier"]) if "tb_barrier" in d.columns else TB_MIN

        exit_i = min(entry_i + horizon, len(d) - 1)
        reason = "time_exit"
        for j in range(entry_i + 1, min(entry_i + horizon, len(d) - 1) + 1):
            # Используем High/Low для реалистичного выхода
            if "silver_high" in d.columns:
                if d.loc[j, "silver_high"] / entry - 1 >= b:
                    exit_i = j; reason = "take_profit"; break
                if d.loc[j, "silver_low"]  / entry - 1 <= -b:
                    exit_i = j; reason = "stop_loss";  break
            else:
                r = float(d.loc[j, "silver_close"]) / entry - 1
                if r >= b:   exit_i = j; reason = "take_profit"; break
                if r <= -b:  exit_i = j; reason = "stop_loss";   break

        exit_p = float(d.loc[exit_i, "silver_close"])
        gross  = exit_p / entry - 1
        net    = gross - COST
        trades.append({
            "split":       split,
            "signal_date": d.loc[i, "Date"],
            "entry_date":  d.loc[entry_i, "Date"],
            "exit_date":   d.loc[exit_i, "Date"],
            "entry_price": entry,
            "exit_price":  exit_p,
            "gross_return": gross,
            "net_return":  net,
            "exit_reason": reason,
            "p_up":        d.loc[i, "p_up"],
            "barrier":     b,
        })
        i = exit_i + 1

    t = pd.DataFrame(trades)
    if not t.empty:
        t["cum_net_return"] = t["net_return"].cumsum()
    return t


def buy_and_hold_return(df: pd.DataFrame, split: str) -> float:
    d = df[df["split"] == split].sort_index()
    if len(d) < 2:
        return np.nan
    return float(d["silver_close"].iloc[-1] / d["silver_close"].iloc[0] - 1)


def backtest_summary(trades: pd.DataFrame, split: str, bnh: float) -> dict:
    if trades.empty:
        return {"split": split, "n_trades": 0, "buy_and_hold": bnh}
    wins  = trades.loc[trades.net_return > 0, "net_return"].sum()
    loss_ = -trades.loc[trades.net_return < 0, "net_return"].sum()
    return {
        "split":            split,
        "n_trades":         len(trades),
        "sum_net_return":   trades.net_return.sum(),
        "avg_net_return":   trades.net_return.mean(),
        "win_rate":         (trades.net_return > 0).mean(),
        "profit_factor":    wins / loss_ if loss_ > 0 else np.inf,
        "take_profit_pct":  (trades.exit_reason == "take_profit").mean(),
        "stop_loss_pct":    (trades.exit_reason == "stop_loss").mean(),
        "buy_and_hold":     bnh,
        "vs_bnh":           trades.net_return.sum() - bnh,
    }


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def build_equity_curve(df: pd.DataFrame, split: str, trades: pd.DataFrame) -> pd.DataFrame:
    d = df[df["split"] == split].sort_index()[["silver_close"]].copy()
    if d.empty:
        return d
    bnh_start = d["silver_close"].iloc[0]
    d["bnh_equity"] = d["silver_close"] / bnh_start

    d["strategy_equity"] = 1.0
    if not trades.empty:
        for _, tr in trades.iterrows():
            mask = d.index >= tr["exit_date"]
            if mask.any():
                d.loc[mask, "strategy_equity"] = (
                    d.loc[d.index < tr["exit_date"], "strategy_equity"].iloc[-1]
                    * (1 + tr["net_return"])
                    if len(d.loc[d.index < tr["exit_date"]]) else 1 + tr["net_return"]
                )
    return d


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def feature_importance_report(
    model: CalibratedClassifierCV, X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    try:
        # Пробуем permutation importance на val-наборе
        base_model = model.calibrated_classifiers_[0].estimator
        imp = permutation_importance(base_model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        return pd.DataFrame({
            "feature":    X.columns,
            "importance": imp.importances_mean,
            "std":        imp.importances_std,
        }).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def md_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if df is None or df.empty:
        return "_Нет данных._"
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    out = out.fillna("")
    cols = [str(c) for c in out.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in out.iterrows():
        vals = [str(r[c]).replace("\n", " ").replace("|", "\\|") for c in out.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def pct(x: float, d: int = 2) -> str:
    if not np.isfinite(x):
        return ""
    return f"{100*x:.{d}f}%"


# ---------------------------------------------------------------------------
# Основной pipeline
# ---------------------------------------------------------------------------

def split_name(date: pd.Timestamp) -> str:
    if date.year <= 2022: return "train"
    if date.year == 2023: return "valid"
    if date.year == 2024: return "test"
    return "forward"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",   default="2013-01-01", help="Начало истории")
    ap.add_argument("--end",     default="2099-12-31", help="Конец истории (или today)")
    ap.add_argument("--out-dir", default="baseline_outputs_v14")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    args = ap.parse_args(argv)

    end = min(args.end, pd.Timestamp.today().strftime("%Y-%m-%d"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка данных
    print("=== v14: загрузка OHLC ===")
    df = fetch_ohlc(args.start, end)
    print(f"  OHLC: {len(df)} строк, {df.index[0].date()} — {df.index[-1].date()}")

    print("=== v14: загрузка FRED ===")
    fred = fetch_fred(args.start, end)
    print(f"  FRED: {len(fred)} строк" if not fred.empty else "  FRED: пропущены")

    # 2. Признаки
    print("=== v14: инженерия признаков ===")
    df = build_features(df, fred)

    # 3. Метки triple-barrier
    print("=== v14: triple-barrier labeling ===")
    df = add_triple_barrier_labels(df, horizon=args.horizon)
    tb_mode = df["tb_mode"].iloc[-1] if "tb_mode" in df.columns else "unknown"
    print(f"  Режим triple-barrier: {tb_mode}")

    # 4. Разбивка по выборкам
    df["split"] = df.index.map(split_name)
    df.to_csv(out / "v14_full_data.csv")

    # Распределение меток
    lab_rows = []
    for s in ["train", "valid", "test", "forward"]:
        d = df[(df["split"] == s) & df["tb_label"].notna()]
        vc = d["tb_label"].value_counts().to_dict()
        n  = len(d)
        lab_rows.append({
            "split": s, "n": n,
            "UP":      vc.get("UP", 0),
            "NEUTRAL": vc.get("NEUTRAL", 0),
            "DOWN":    vc.get("DOWN", 0),
            "UP_rate": f"{vc.get('UP', 0)/n:.3f}" if n else "?",
            "NEUTRAL_rate": f"{vc.get('NEUTRAL', 0)/n:.3f}" if n else "?",
        })
    label_report = pd.DataFrame(lab_rows)
    label_report.to_csv(out / "v14_label_distribution.csv", index=False)
    print("  Распределение меток:")
    print(label_report.to_string(index=False))

    # 5. Обучение
    feature_cols = get_feature_cols(df)
    print(f"\n=== v14: обучение (признаков: {len(feature_cols)}) ===")

    train_df = df[(df["split"] == "train") & df["tb_label"].notna()].copy()
    valid_df = df[(df["split"] == "valid") & df["tb_label"].notna()].copy()

    if len(train_df) < 100:
        raise RuntimeError(f"Недостаточно обучающих данных: {len(train_df)} строк. Проверьте подключение к интернету.")

    X_train = train_df[feature_cols]
    y_train = train_df["tb_label"].astype(str)

    print(f"  Обучение: {len(X_train)} строк ({X_train.index[0].date()} — {X_train.index[-1].date()})")
    model = train_and_calibrate(X_train, y_train)
    print(f"  Классы: {model.classes_}")

    # Feature importance на valid
    if len(valid_df) > 0:
        X_val = valid_df[feature_cols]
        y_val = valid_df["tb_label"].astype(str)
        fi    = feature_importance_report(model, X_val, y_val)
        if not fi.empty:
            fi.to_csv(out / "v14_feature_importance.csv", index=False)
            print("  Топ-10 признаков по важности:")
            print(fi.head(10).to_string(index=False))

    # 6. Метрики классификатора
    print("\n=== v14: метрики ===")
    cls_metrics = []
    for s in ["train", "valid", "test", "forward"]:
        cls_metrics.append(evaluate_split(df, s, model, feature_cols))
    cls_df = pd.DataFrame(cls_metrics)
    cls_df.to_csv(out / "v14_classifier_metrics.csv", index=False)
    print(cls_df[["split","n","balanced_accuracy","log_loss","auc_up_vs_rest"]].to_string(index=False))

    # 7. Выбор политики на valid
    print("\n=== v14: выбор политики на valid ===")
    valid_full = df[df["split"] == "valid"].copy()
    policy_params = select_policy(valid_full, model, feature_cols)
    print(f"  Параметры: {policy_params}")
    policy_params["horizon_days"] = args.horizon
    policy_params["tb_mode"]      = tb_mode
    policy_params["train_window"] = "2013-2022"
    with open(out / "v14_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_params, f, indent=2, ensure_ascii=False)

    # 8. Применение политики ко всем выборкам
    all_df = df.copy()
    all_df = apply_policy(
        all_df, model, feature_cols,
        policy_params["up_threshold"],
        policy_params["margin_threshold"],
        policy_params["down_cap"],
        policy_params["cooldown"],
    )
    all_df.to_csv(out / "v14_decisions_all.csv")

    # 9. Guardrails
    print("\n=== v14: guardrails ===")
    guardrails = pd.DataFrame([compute_guardrails(all_df, s) for s in ["valid", "test", "forward"]])
    guardrails.to_csv(out / "v14_guardrails.csv", index=False)
    print(guardrails[["split","n_signals","precision","wilson_95_low","base_up_rate","lift_vs_base","warning"]].to_string(index=False))

    # 10. Бэктест + бенчмарки
    print("\n=== v14: бэктест + buy-and-hold бенчмарк ===")
    bt_rows = []
    for s in ["valid", "test", "forward"]:
        trades = backtest_strategy(all_df, s, args.horizon)
        trades.to_csv(out / f"{s}_trades_v14.csv", index=False)
        bnh = buy_and_hold_return(all_df, s)
        summary = backtest_summary(trades, s, bnh)
        bt_rows.append(summary)
        sig_out = all_df[all_df["split"] == s]
        sig_out.to_csv(out / f"{s}_decisions_v14.csv")
    bt_df = pd.DataFrame(bt_rows)
    bt_df.to_csv(out / "v14_backtest_report.csv", index=False)
    print(bt_df[["split","n_trades","sum_net_return","win_rate","profit_factor","buy_and_hold","vs_bnh"]].to_string(index=False))

    # 11. Сигнальные карточки (последние строки)
    cards = []
    for s in ["valid", "test", "forward"]:
        d = all_df[all_df["split"] == s].sort_index()
        if d.empty: continue
        r = d.iloc[-1]
        cards.append({
            "split":         s,
            "date":          r.name.date(),
            "silver_close":  r.get("silver_close", np.nan),
            "signal":        r.get("signal", "HOLD"),
            "reason":        r.get("reason", ""),
            "p_up":          round(r.get("p_up", np.nan), 4),
            "p_down":        round(r.get("p_down", np.nan), 4),
            "trend_regime":  r.get("trend_regime", ""),
            "vol_regime":    r.get("vol_regime", ""),
        })
    cards_df = pd.DataFrame(cards)
    cards_df.to_csv(out / "v14_latest_signal_cards.csv", index=False)

    # 12. Purged CV статистика
    print("\n=== v14: purged walk-forward CV ===")
    labeled = df[df["tb_label"].notna()].copy()
    wf_splits = purged_walk_forward_splits(labeled.index, n_train_years=3, n_test_months=6)
    wf_scores = []
    for i, (tr_idx, te_idx) in enumerate(wf_splits):
        Xtr = labeled.loc[tr_idx, feature_cols]
        ytr = labeled.loc[tr_idx, "tb_label"].astype(str)
        Xte = labeled.loc[te_idx, feature_cols]
        yte = labeled.loc[te_idx, "tb_label"].astype(str)
        if Xtr.empty or Xte.empty: continue
        m = build_model()
        try:
            m.fit(Xtr, ytr)
            ba = balanced_accuracy_score(yte, m.predict(Xte))
            wf_scores.append({"fold": i, "train_end": tr_idx[-1].date(), "test_start": te_idx[0].date(), "balanced_acc": ba})
        except Exception as e:
            print(f"  fold {i} ошибка: {e}")
    wf_df = pd.DataFrame(wf_scores)
    wf_df.to_csv(out / "v14_purged_wf_cv.csv", index=False)
    if not wf_df.empty:
        print(f"  Фолдов: {len(wf_df)}, средний balanced_acc: {wf_df['balanced_acc'].mean():.3f} ± {wf_df['balanced_acc'].std():.3f}")
        print(wf_df.to_string(index=False))

    # 13. Итоговый отчёт
    vs_v13_note = (
        "OHLC triple-barrier вместо close-only; "
        "10 лет обучения вместо 1; "
        "HistGradientBoosting вместо LogisticRegression; "
        "FRED сдвинуты на 1 день; "
        "buy-and-hold бенчмарк добавлен; "
        "purged CV с embargo."
    )

    report_md = f"""# Silver Trading Assistant v14 — итоговый отчёт

## Изменения vs v13
{vs_v13_note}

## Режим triple-barrier: `{tb_mode}`

## Распределение меток
{md_table(label_report)}

## Метрики классификатора
{md_table(cls_df[['split','n','label_up_rate','balanced_accuracy','log_loss','auc_up_vs_rest']])}

> **Интерпретация**: balanced_accuracy > 0.50 = лучше случайного выбора.
> auc_up_vs_rest > 0.55 = модель различает UP от остальных.

## Guardrails (политика сигналов)
{md_table(guardrails)}

> **Интерпретация**: `wilson_95_low > base_up_rate` означает статистически доказанный edge.
> Без этого условия — сигналы неотличимы от «покупать всегда».

## Бэктест vs buy-and-hold
{md_table(bt_df)}

> **Примечание**: `vs_bnh` — разница между суммарной стратегией и passive buy-and-hold за период.
> Отрицательное значение = стратегия проиграла пассивному удержанию.

## Последние сигнальные карточки
{md_table(cards_df)}

## Purged walk-forward CV (balanced_accuracy по фолдам)
{md_table(wf_df)}

## Честный вывод
v14 исправляет ключевые методологические ошибки v13:
triple-barrier теперь использует реальные H/L, обучение расширено в 10 раз,
модель мощнее, утечки данных FRED устранены, добавлен buy-and-hold бенчмарк.

**Если guardrails показывают `ci_lower_not_above_base` на test — edge не доказан,
торговать реальными деньгами нельзя.** Продолжать итерации по модели.
"""
    (out / "v14_integrated_report.md").write_text(report_md, encoding="utf-8")

    print(f"\n=== v14 завершён. Результаты: {out.resolve()} ===")
    print(f"  Запустите дашборд: streamlit run dashboard_app.py")


if __name__ == "__main__":
    main()
