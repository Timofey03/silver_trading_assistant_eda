"""
Silver Trading Assistant v15 — COT + горизонт 15 дней + режимная ансамблевая модель

Изменения vs v14:
1. COT (Commitments of Traders) от CFTC — бесплатные альтернативные данные.
   Признаки: net_spec, cot_index_52w, cot_change_4w (с 4-дневным сдвигом публикации).
2. Горизонт 15 дней (больше NEUTRAL, меньше шума) + embargo 15 дней.
3. Режимная ансамблевая модель (Mixture of Experts):
   - Три отдельных HistGradientBoostingClassifier: uptrend / sideways / downtrend.
   - Мета-выбор: на каждом баре используется классификатор текущего тренд-режима.
   - Если режим не определён или мало данных — fallback на глобальную модель.

Запуск:
  python silver_assistant_v15_regime_cot.py
  python silver_assistant_v15_regime_cot.py --horizon 15 --out-dir baseline_outputs_v15
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

# Принудительный UTF-8 stdout для Windows cp1251 (sklearn/numpy используют → и другие символы)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import json
import math
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
except ImportError:
    raise ImportError("pip install scikit-learn>=1.3.0")

# Переиспользуем функции из v14
try:
    from silver_assistant_v14_main import (
        fetch_ohlc, fetch_fred, build_features,
        add_triple_barrier_labels, get_feature_cols,
        compute_guardrails, backtest_strategy, buy_and_hold_return,
        backtest_summary, purged_walk_forward_splits,
        wilson_ci, md_table, pct,
        SPLITS, EMBARGO,
    )
    print("  v14 функции загружены.")
except ImportError as e:
    raise ImportError(
        "Запустите сначала: python silver_assistant_v14_main.py\n"
        f"Детали: {e}"
    )

HORIZON_V15 = 15        # 15-дневный горизонт
EMBARGO_V15 = 15        # embargo = horizon, чтобы не было overlap
COT_RELEASE_LAG = 4    # CFTC публикует во вторник отчёт, выходит в пятницу → lag ~4 дня

# ---------------------------------------------------------------------------
# 1. COT данные из CFTC (бесплатно)
# ---------------------------------------------------------------------------

SILVER_COT_NAMES = [
    "SILVER - COMMODITY EXCHANGE INC.",
    "SILVER",
]

COT_LEGACY_COLS = [
    "Market_and_Exchange_Names",
    "As_of_Date_In_Form_YYMMDD",
    "Open_Interest_All",
    "Noncommercial_Positions_Long_All",
    "Noncommercial_Positions_Short_All",
    "Noncommercial_Positions_Spreading_All",
    "Commercial_Positions_Long_All",
    "Commercial_Positions_Short_All",
    "Total_Reportable_Positions_Long_All",
    "Total_Reportable_Positions_Short_All",
    "Nonreportable_Positions_Long_All",
    "Nonreportable_Positions_Short_All",
    "Report_Date_as_MM_DD_YYYY",
]


def _try_download(url: str, timeout: int = 30) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.content) > 1000:
            return r.content
    except Exception:
        pass
    return None


def _parse_cot_zip(content: bytes) -> pd.DataFrame:
    """Разбирает ZIP с COT legacy-файлом, возвращает строки по серебру."""
    frames = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith((".txt", ".csv")):
                continue
            raw = zf.read(name).decode("latin-1", errors="replace")
            try:
                df = pd.read_csv(io.StringIO(raw), low_memory=False)
            except Exception:
                continue
            if "Market_and_Exchange_Names" not in df.columns:
                continue
            mask = df["Market_and_Exchange_Names"].str.upper().str.contains("SILVER", na=False)
            if mask.any():
                frames.append(df[mask].copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _parse_cot_txt(content: bytes) -> pd.DataFrame:
    """Разбирает текстовый/CSV COT-файл напрямую."""
    raw = content.decode("latin-1", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(raw), low_memory=False)
    except Exception:
        return pd.DataFrame()
    if "Market_and_Exchange_Names" not in df.columns:
        return pd.DataFrame()
    mask = df["Market_and_Exchange_Names"].str.upper().str.contains("SILVER", na=False)
    return df[mask].copy() if mask.any() else pd.DataFrame()


def fetch_cot_silver(start_year: int = 2013, end_year: int = 2026) -> pd.DataFrame:
    """
    Скачивает COT Legacy Futures-Only отчёты с CFTC.
    Публикуется еженедельно. Сдвиг публикации ~4 дня уже учтён при merge.
    Возвращает еженедельный DataFrame с датой в индексе.
    """
    all_frames = []

    # Паттерны URL CFTC для разных лет
    url_patterns = [
        "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip",
        "https://www.cftc.gov/files/dea/history/annual_{year}.zip",
        "https://www.cftc.gov/files/dea/history/com_disagg_txt_{year}.zip",
    ]

    for year in range(start_year, end_year + 1):
        found = False
        for pat in url_patterns:
            url  = pat.format(year=year)
            data = _try_download(url, timeout=25)
            if data is None:
                continue
            try:
                df = _parse_cot_zip(data)
            except Exception:
                # Может быть не ZIP, а обычный CSV
                df = _parse_cot_txt(data)

            if not df.empty:
                all_frames.append(df)
                print(f"  COT {year}: {len(df)} строк ({url.split('/')[-1]})")
                found = True
                break

        if not found:
            print(f"  COT {year}: данные не найдены, пропускаем")

    if not all_frames:
        print("  ПРЕДУПРЕЖДЕНИЕ: COT данные недоступны — работаем без них.")
        return pd.DataFrame()

    raw = pd.concat(all_frames, ignore_index=True)
    return _process_cot(raw)


def _process_cot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует сырой COT DataFrame в чистые еженедельные признаки.
    Поддерживает оба формата CFTC:
      - Legacy:        Noncommercial_Positions_Long/Short_All, Commercial_Positions_Long/Short_All
      - Disaggregated: M_Money_Positions_Long/Short_All, Prod_Merc_Positions_Long/Short_All
    """
    out = df.copy()

    # --- Дата ---
    date_col = None
    for c in ["Report_Date_as_MM_DD_YYYY", "As_of_Date_In_Form_YYMMDD",
              "As_of_Date_in_Form_YYYY-MM-DD", "As_of_Date_In_Form_YYYY-MM-DD"]:
        if c in out.columns:
            date_col = c
            break

    if date_col is None:
        # Последняя попытка: любая колонка содержащая "date"
        for c in out.columns:
            if "date" in c.lower():
                date_col = c
                break

    if date_col is None:
        print("  COT: колонка с датой не найдена. Колонки:", list(out.columns)[:10])
        return pd.DataFrame()

    try:
        if "YYYY" in date_col.upper() or "-" in str(out[date_col].iloc[0]):
            out["cot_date"] = pd.to_datetime(out[date_col], errors="coerce")
        else:
            out["cot_date"] = pd.to_datetime(out[date_col], format="%y%m%d", errors="coerce")
            if out["cot_date"].isna().mean() > 0.5:
                out["cot_date"] = pd.to_datetime(out[date_col], errors="coerce")
    except Exception:
        return pd.DataFrame()

    out = out.dropna(subset=["cot_date"]).sort_values("cot_date")

    # --- Нормализация чисел ---
    def _to_num(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(np.nan, index=out.index)
        return pd.to_numeric(
            out[col].astype(str).str.replace(",", "").str.strip(), errors="coerce"
        )

    # --- Определяем формат: legacy или disaggregated ---
    is_disagg = "M_Money_Positions_Long_All" in out.columns

    if is_disagg:
        # Disaggregated (com_disagg): Managed Money = спекулянты (хедж-фонды/CTA)
        spec_long  = _to_num("M_Money_Positions_Long_All")
        spec_short = _to_num("M_Money_Positions_Short_All")
        # Producer/Merchant = коммерческие хеджеры
        comm_long  = _to_num("Prod_Merc_Positions_Long_All")
        comm_short = _to_num("Prod_Merc_Positions_Short_All")
        print(f"  COT формат: disaggregated ({len(out)} строк)")
    else:
        # Legacy: Non-Commercial = спекулянты
        spec_long  = _to_num("Noncommercial_Positions_Long_All")
        spec_short = _to_num("Noncommercial_Positions_Short_All")
        comm_long  = _to_num("Commercial_Positions_Long_All")
        comm_short = _to_num("Commercial_Positions_Short_All")
        print(f"  COT формат: legacy ({len(out)} строк)")

    oi = _to_num("Open_Interest_All")

    result = pd.DataFrame({
        "cot_date":    out["cot_date"].values,
        "net_spec":    spec_long - spec_short,
        "net_comm":    comm_long - comm_short,
        "open_interest": oi,
    })
    result = (
        result
        .dropna(subset=["net_spec"])
        .sort_values("cot_date")
        .drop_duplicates("cot_date")
        .set_index("cot_date")
    )

    if result.empty:
        print("  COT: после очистки нет строк. Проверьте имена колонок.")
        print("  Доступные колонки:", [c for c in out.columns if "position" in c.lower() or "money" in c.lower()][:10])
        return pd.DataFrame()

    # --- COT-признаки ---
    ws = 52
    roll_min = result["net_spec"].rolling(ws, min_periods=ws // 2).min()
    roll_max = result["net_spec"].rolling(ws, min_periods=ws // 2).max()
    denom    = (roll_max - roll_min).replace(0, np.nan)

    result["cot_index_52w"]   = (result["net_spec"] - roll_min) / denom * 100
    result["net_spec_norm"]   = result["net_spec"] / result["open_interest"].replace(0, np.nan) * 100
    result["cot_change_4w"]   = result["net_spec"].diff(4)
    result["cot_change_norm"] = result["cot_change_4w"] / result["open_interest"].replace(0, np.nan)
    # Экстремальные позиции — контрарный сигнал
    result["cot_extreme_bull"] = (result["cot_index_52w"] > 75).astype(float)
    result["cot_extreme_bear"] = (result["cot_index_52w"] < 25).astype(float)

    return result


def merge_cot_to_daily(daily: pd.DataFrame, cot: pd.DataFrame, lag_days: int = COT_RELEASE_LAG) -> pd.DataFrame:
    """
    Объединяет еженедельный COT с ежедневным DataFrame.
    Сдвигает на lag_days для устранения lookahead (CFTC публикует через ~4 дня после даты позиции).
    """
    if cot.empty:
        return daily

    # Сдвиг: позиция от вторника появляется в пятницу → сдвигаем ещё
    cot_shifted = cot.copy()
    cot_shifted.index = cot_shifted.index + pd.Timedelta(days=lag_days)

    # Reindex на ежедневный, forward fill
    merged = daily.join(cot_shifted.reindex(daily.index, method="ffill"), how="left")
    return merged


# ---------------------------------------------------------------------------
# 2. Дополнительные признаки COT
# ---------------------------------------------------------------------------

COT_FEATURES = [
    "net_spec", "net_comm", "open_interest",
    "cot_index_52w", "net_spec_norm",
    "cot_change_4w", "cot_change_norm",
    "cot_extreme_bull", "cot_extreme_bear",
]


def add_vol_trend_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Трёхзонный режим — комбинация тренда и волатильности.

    trend_regime (v15, исправленный):
      ±2% полоса вокруг 60-дневной MA → sideways не пустой.
      Дополнительно: 20-дневный возврат подтверждает направление.
        uptrend   = dist_ma60 > +2%  ИЛИ  ret20 > +5%
        downtrend = dist_ma60 < -2%  ИЛИ  ret20 < -5%
        sideways  = иначе

    regime (v15):
      Комбинирует trend_regime × vol_regime → 6 зон:
        uptrend_highvol, uptrend_lowvol, downtrend_highvol, downtrend_lowvol,
        sideways_highvol, sideways_lowvol
      Каждая зона обучается отдельно (если ≥ MIN_REGIME_SAMPLES строк).
    """
    out = df.copy()
    sl   = out["silver_close"]
    ma60 = sl.rolling(60, min_periods=30).mean()
    dist = sl / ma60 - 1
    ret20 = sl.pct_change(20)

    trend = pd.Series("sideways", index=out.index)
    trend[(dist > 0.02) | (ret20 > 0.05)]  = "uptrend"
    trend[(dist < -0.02) | (ret20 < -0.05)] = "downtrend"
    out["trend_regime"] = trend

    # vol_regime уже строится в build_features через rv20/med_vol
    # но пересчитаем здесь для надёжности
    if "silver_realized_vol_20d" in out.columns:
        rv20    = out["silver_realized_vol_20d"]
        med_vol = rv20.rolling(252, min_periods=60).median()
        vol_r   = pd.Series("medium", index=out.index)
        vol_r[rv20 > med_vol * 1.4] = "high"
        vol_r[rv20 < med_vol * 0.7] = "low"
        out["vol_regime"] = vol_r
        out["regime"] = out["trend_regime"] + "_" + out["vol_regime"]
    else:
        out["regime"] = out["trend_regime"]

    return out


# ---------------------------------------------------------------------------
# 3. Режимная ансамблевая модель (Mixture of Experts)
# ---------------------------------------------------------------------------

MIN_REGIME_SAMPLES = 150


class RegimeEnsemble:
    """
    Mixture-of-Experts: отдельный классификатор для каждого режима.
    Предсказание на баре i → модель текущего режима.
    Fallback = глобальная модель, если режим неизвестен или мало данных.
    """

    def __init__(self):
        self.models: Dict[str, CalibratedClassifierCV] = {}
        self.fallback: Optional[CalibratedClassifierCV] = None
        self.classes_: Optional[np.ndarray] = None
        self.regime_col: str = "regime"   # колонка используемого режима

    def _build_base(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=1.5,
            learning_rate=0.04,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            class_weight="balanced",
            random_state=42,
        )

    def _calibrate(self, X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
        cal = CalibratedClassifierCV(self._build_base(), method="isotonic", cv=3)
        cal.fit(X, y)
        return cal

    def fit(self, X: pd.DataFrame, y: pd.Series, regimes: pd.Series) -> "RegimeEnsemble":
        print("  Обучение: глобальная fallback-модель...")
        self.fallback = self._calibrate(X, y)
        self.classes_ = self.fallback.classes_

        unique_regimes = regimes.value_counts()
        print(f"  Режимы в train: {dict(unique_regimes)}")
        for regime, n in unique_regimes.items():
            if n >= MIN_REGIME_SAMPLES:
                print(f"  Обучение: режим '{regime}' ({n} строк)...")
                mask = regimes == regime
                self.models[regime] = self._calibrate(X[mask], y[mask])
            else:
                print(f"  Режим '{regime}': {n} строк < {MIN_REGIME_SAMPLES} → fallback")
        return self

    def predict_proba(self, X: pd.DataFrame, regimes: pd.Series) -> np.ndarray:
        classes = list(self.classes_)
        n_cls   = len(classes)
        result  = np.zeros((len(X), n_cls))
        all_regimes = list(self.models.keys()) + ["__fallback__"]

        for regime in set(regimes.unique()) | {"__fallback__"}:
            mask = (regimes == regime).values if regime != "__fallback__" \
                   else ~regimes.isin(self.models).values
            if not mask.any():
                continue
            model = self.models.get(regime, self.fallback)
            proba = model.predict_proba(X[mask])
            model_cls = list(model.classes_)
            aligned   = np.zeros((proba.shape[0], n_cls))
            for i, cls in enumerate(model_cls):
                if cls in classes:
                    aligned[:, classes.index(cls)] = proba[:, i]
            result[mask] = aligned

        return result

    def predict(self, X: pd.DataFrame, regimes: pd.Series) -> np.ndarray:
        proba = self.predict_proba(X, regimes)
        return self.classes_[np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# 4. Метрики и гарды (адаптированы для RegimeEnsemble)
# ---------------------------------------------------------------------------

def evaluate_split_v15(
    df: pd.DataFrame, split_name: str,
    model: RegimeEnsemble, feature_cols: List[str],
) -> Dict[str, object]:
    d = df[df["split"] == split_name].dropna(subset=["tb_label"]).copy()
    if d.empty:
        return {"split": split_name, "n": 0}

    X       = d[feature_cols]
    regimes = d["regime"].fillna("sideways") if "regime" in d.columns else d["trend_regime"].fillna("sideways")
    y       = d["tb_label"].astype(str)
    classes = list(model.classes_)

    proba = model.predict_proba(X, regimes)
    pred  = model.predict(X, regimes)

    result: Dict[str, object] = {
        "split":              split_name,
        "n":                  len(d),
        "label_up_rate":      (y == "UP").mean(),
        "label_neutral_rate": (y == "NEUTRAL").mean(),
        "label_down_rate":    (y == "DOWN").mean(),
        "accuracy":           float((pred == y.values).mean()),
        "balanced_accuracy":  balanced_accuracy_score(y, pred),
    }

    if "UP" in classes:
        up_i = classes.index("UP")
        result["auc_up_vs_rest"] = roc_auc_score(
            (y == "UP").astype(int), proba[:, up_i]
        )
    try:
        result["log_loss"] = log_loss(y, proba, labels=classes)
    except Exception:
        result["log_loss"] = np.nan

    return result


def apply_policy_v15(
    df: pd.DataFrame, model: RegimeEnsemble,
    feature_cols: List[str],
    up_threshold: float = 0.40,
    margin_threshold: float = 0.10,
    down_cap: float = 0.30,
    cooldown: int = 10,
) -> pd.DataFrame:
    out     = df.copy().sort_index()
    X       = out[feature_cols]
    regimes = out["regime"].fillna("sideways") if "regime" in out.columns else out["trend_regime"].fillna("sideways")
    classes = list(model.classes_)
    proba   = model.predict_proba(X, regimes)

    up_i  = classes.index("UP")      if "UP"      in classes else None
    dn_i  = classes.index("DOWN")    if "DOWN"     in classes else None
    ne_i  = classes.index("NEUTRAL") if "NEUTRAL"  in classes else None

    out["p_up"]      = proba[:, up_i]  if up_i  is not None else 0.0
    out["p_down"]    = proba[:, dn_i]  if dn_i  is not None else 0.0
    out["p_neutral"] = proba[:, ne_i]  if ne_i  is not None else 0.0
    out["up_margin"] = out["p_up"] - out[["p_down", "p_neutral"]].max(axis=1)
    out["regime_model"] = regimes.values

    raw = (
        (out["p_up"] >= up_threshold) &
        (out["up_margin"] >= margin_threshold) &
        (out["p_down"] <= down_cap)
    )
    out["raw_buy"] = raw

    signals, reasons = [], []
    last_buy = -9999
    for i, ok in enumerate(raw):
        if ok and i - last_buy > cooldown:
            signals.append("BUY"); reasons.append("regime_tb_edge"); last_buy = i
        elif ok:
            signals.append("HOLD"); reasons.append("cooldown")
        else:
            signals.append("HOLD"); reasons.append("no_edge")

    out["signal"] = signals
    out["reason"] = reasons
    return out


def select_policy_v15(
    valid_df: pd.DataFrame, model: RegimeEnsemble, feature_cols: List[str]
) -> dict:
    best_obj    = -np.inf
    best_params = {}

    for up_thr in [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]:
        for margin in [0.0, 0.05, 0.10, 0.15]:
            for down_cap in [0.25, 0.30, 0.35, 0.40]:
                for cooldown in [5, 10, 15]:
                    tmp  = apply_policy_v15(valid_df, model, feature_cols, up_thr, margin, down_cap, cooldown)
                    sigs = tmp[(tmp["signal"] == "BUY") & tmp["tb_label"].notna()]
                    n    = len(sigs)
                    if n < 4:
                        continue
                    labeled  = tmp[tmp["tb_label"].notna()]
                    correct  = int((sigs["tb_label"] == "UP").sum())
                    base     = float((labeled["tb_label"] == "UP").mean())
                    prec     = correct / n
                    lo, _    = wilson_ci(correct, n)
                    lift     = prec - base
                    obj      = (lo - base) + 0.003 * min(n, 20) if lift > 0 else -999
                    if obj > best_obj:
                        best_obj    = obj
                        best_params = {
                            "up_threshold": up_thr, "margin_threshold": margin,
                            "down_cap": down_cap, "cooldown": cooldown,
                        }

    return best_params if best_params else {
        "up_threshold": 0.44, "margin_threshold": 0.10,
        "down_cap": 0.30, "cooldown": 10,
    }


# ---------------------------------------------------------------------------
# 5. Purged CV для RegimeEnsemble
# ---------------------------------------------------------------------------

def purged_cv_regime(
    df: pd.DataFrame, feature_cols: List[str],
    n_train_years: int = 3, n_test_months: int = 6,
) -> pd.DataFrame:
    labeled  = df[df["tb_label"].notna()].copy()
    wf_splits = purged_walk_forward_splits(
        labeled.index, n_train_years=n_train_years,
        n_test_months=n_test_months, embargo_days=EMBARGO_V15, horizon=HORIZON_V15,
    )
    rows = []
    for i, (tr_idx, te_idx) in enumerate(wf_splits):
        Xtr  = labeled.loc[tr_idx, feature_cols]
        ytr  = labeled.loc[tr_idx, "tb_label"].astype(str)
        Xte  = labeled.loc[te_idx, feature_cols]
        yte  = labeled.loc[te_idx, "tb_label"].astype(str)
        rcol = "regime" if "regime" in labeled.columns else "trend_regime"
        rgtr = labeled.loc[tr_idx, rcol].fillna("sideways")
        rgte = labeled.loc[te_idx, rcol].fillna("sideways")

        if len(Xtr) < MIN_REGIME_SAMPLES or Xte.empty:
            continue
        try:
            m = RegimeEnsemble()
            # io.StringIO() всегда принимает любой юникод — нет проблем с cp1251
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xtr, ytr, rgtr)
                pred = m.predict(Xte, rgte)
            ba = balanced_accuracy_score(yte, pred)
            rows.append({
                "fold": i,
                "train_end":    tr_idx[-1].date(),
                "test_start":   te_idx[0].date(),
                "n_train":      len(Xtr),
                "n_test":       len(Xte),
                "balanced_acc": ba,
            })
        except Exception as e:
            print(f"  fold {i} skipped: {type(e).__name__}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Вспомогательные функции
# ---------------------------------------------------------------------------

def split_name(date: pd.Timestamp) -> str:
    if date.year <= 2022: return "train"
    if date.year == 2023: return "valid"
    if date.year == 2024: return "test"
    return "forward"


def label_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in ["train", "valid", "test", "forward"]:
        d  = df[(df["split"] == s) & df["tb_label"].notna()]
        vc = d["tb_label"].value_counts().to_dict()
        n  = len(d)
        rows.append({
            "split": s, "n": n,
            "UP":      vc.get("UP", 0),
            "NEUTRAL": vc.get("NEUTRAL", 0),
            "DOWN":    vc.get("DOWN", 0),
            "UP_rate":      f"{vc.get('UP',0)/n:.3f}" if n else "?",
            "NEUTRAL_rate": f"{vc.get('NEUTRAL',0)/n:.3f}" if n else "?",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Основной pipeline
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",   default="2013-01-01")
    ap.add_argument("--end",     default="2099-12-31")
    ap.add_argument("--out-dir", default="baseline_outputs_v15")
    ap.add_argument("--horizon", type=int, default=HORIZON_V15)
    args = ap.parse_args(argv)

    end = min(args.end, pd.Timestamp.today().strftime("%Y-%m-%d"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== v15: горизонт {args.horizon} дней, embargo {EMBARGO_V15} дней ===")

    # ---- Данные ----
    print("\n=== v15: загрузка OHLC ===")
    df = fetch_ohlc(args.start, end)
    print(f"  OHLC: {len(df)} строк, {df.index[0].date()} — {df.index[-1].date()}")

    print("\n=== v15: загрузка FRED ===")
    try:
        from silver_assistant_v14_main import fetch_fred
        fred = fetch_fred(args.start, end)
        print(f"  FRED: {len(fred)} строк" if not fred.empty else "  FRED: пропущены")
    except Exception:
        fred = pd.DataFrame()

    print("\n=== v15: загрузка COT (CFTC) ===")
    start_year = pd.Timestamp(args.start).year
    end_year   = pd.Timestamp(end).year
    cot = fetch_cot_silver(start_year, end_year)
    if cot.empty:
        print("  COT: данные недоступны — продолжаем без них")
    else:
        print(f"  COT: {len(cot)} недельных записей, {cot.index[0].date()} — {cot.index[-1].date()}")
        cot.to_csv(out / "v15_cot_raw.csv")

    # ---- Признаки ----
    print("\n=== v15: инженерия признаков ===")
    df = build_features(df, fred)

    # Пересчитываем режимы с широкой полосой (было бинарно MA60, теперь ±2% + ret20)
    print("  Пересчёт режимов (±2% от MA60 + ret20)...")
    df = add_vol_trend_regime(df)
    regime_counts = df["trend_regime"].value_counts().to_dict()
    print(f"  trend_regime: {regime_counts}")
    if "regime" in df.columns:
        rc = df["regime"].value_counts().to_dict()
        print(f"  regime (trend x vol): {rc}")

    # Добавляем COT
    if not cot.empty:
        df = merge_cot_to_daily(df, cot, lag_days=COT_RELEASE_LAG)
        present_cot = [c for c in COT_FEATURES if c in df.columns]
        print(f"  COT признаки добавлены: {present_cot}")
    else:
        present_cot = []

    # ---- Triple-barrier (горизонт 15) ----
    print("\n=== v15: triple-barrier (horizon=15) ===")
    df = add_triple_barrier_labels(df, horizon=args.horizon)
    tb_mode = df["tb_mode"].iloc[-1] if "tb_mode" in df.columns else "unknown"
    print(f"  Режим: {tb_mode}")

    # Разбивка
    df["split"] = df.index.map(split_name)
    df.to_csv(out / "v15_full_data.csv")

    # Распределение меток
    lb = label_report(df)
    lb.to_csv(out / "v15_label_distribution.csv", index=False)
    print("  Распределение меток:")
    print(lb.to_string(index=False))

    # ---- Признаки ----
    base_features = get_feature_cols(df)
    all_features  = base_features + [c for c in present_cot if c not in base_features]
    print(f"\n  Всего признаков: {len(all_features)} (COT: {len(present_cot)})")

    # ---- Обучение ----
    print("\n=== v15: обучение режимной ансамблевой модели ===")
    train_df = df[(df["split"] == "train") & df["tb_label"].notna()].copy()

    if len(train_df) < 200:
        raise RuntimeError(f"Недостаточно обучающих данных: {len(train_df)}")

    X_train = train_df[all_features]
    y_train = train_df["tb_label"].astype(str)
    r_train = train_df["trend_regime"].fillna("sideways")

    print(f"  Обучение: {len(X_train)} строк ({X_train.index[0].date()} — {X_train.index[-1].date()})")
    model = RegimeEnsemble()
    model.fit(X_train, y_train, r_train)
    print(f"  Классы: {model.classes_}")
    print(f"  Режимные модели: {list(model.models.keys())}")

    # ---- Метрики ----
    print("\n=== v15: метрики ===")
    cls_metrics = [evaluate_split_v15(df, s, model, all_features)
                   for s in ["train", "valid", "test", "forward"]]
    cls_df = pd.DataFrame(cls_metrics)
    cls_df.to_csv(out / "v15_classifier_metrics.csv", index=False)
    cols = [c for c in ["split","n","balanced_accuracy","log_loss","auc_up_vs_rest"] if c in cls_df.columns]
    print(cls_df[cols].to_string(index=False))

    # ---- Политика ----
    print("\n=== v15: выбор политики (valid 2023) ===")
    valid_full   = df[df["split"] == "valid"].copy()
    policy_params = select_policy_v15(valid_full, model, all_features)
    print(f"  Параметры: {policy_params}")
    policy_params.update({
        "horizon_days":  args.horizon,
        "tb_mode":       tb_mode,
        "train_window":  "2013-2022",
        "cot_features":  present_cot,
        "regime_models": list(model.models.keys()),
    })
    with open(out / "v15_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_params, f, indent=2, ensure_ascii=False)

    # ---- Применение политики ----
    all_df = apply_policy_v15(
        df, model, all_features,
        policy_params["up_threshold"],
        policy_params["margin_threshold"],
        policy_params["down_cap"],
        policy_params["cooldown"],
    )
    all_df.to_csv(out / "v15_decisions_all.csv")

    # ---- Guardrails ----
    print("\n=== v15: guardrails ===")
    grd_rows = []
    for s in ["valid", "test", "forward"]:
        grd_rows.append(compute_guardrails(all_df, s))
    guardrails = pd.DataFrame(grd_rows)
    guardrails.to_csv(out / "v15_guardrails.csv", index=False)
    cols = [c for c in ["split","n_signals","precision","wilson_95_low","base_up_rate","lift_vs_base","warning"] if c in guardrails.columns]
    print(guardrails[cols].to_string(index=False))

    # ---- Бэктест ----
    print("\n=== v15: бэктест + buy-and-hold ===")
    bt_rows = []
    for s in ["valid", "test", "forward"]:
        trades = backtest_strategy(all_df, s, args.horizon)
        trades.to_csv(out / f"{s}_trades_v15.csv", index=False)
        all_df[all_df["split"] == s].to_csv(out / f"{s}_decisions_v15.csv")
        bnh     = buy_and_hold_return(all_df, s)
        summary = backtest_summary(trades, s, bnh)
        bt_rows.append(summary)
    bt_df = pd.DataFrame(bt_rows)
    bt_df.to_csv(out / "v15_backtest_report.csv", index=False)
    cols = [c for c in ["split","n_trades","sum_net_return","win_rate","profit_factor","buy_and_hold","vs_bnh"] if c in bt_df.columns]
    print(bt_df[cols].to_string(index=False))

    # ---- Последние карточки ----
    cards = []
    for s in ["valid", "test", "forward"]:
        d = all_df[all_df["split"] == s].sort_index()
        if d.empty: continue
        r = d.iloc[-1]
        cards.append({
            "split":        s,
            "date":         r.name.date(),
            "silver_close": round(r.get("silver_close", np.nan), 2),
            "signal":       r.get("signal", "HOLD"),
            "reason":       r.get("reason", ""),
            "p_up":         round(r.get("p_up", np.nan), 4),
            "p_down":       round(r.get("p_down", np.nan), 4),
            "regime":       r.get("regime", r.get("trend_regime", "")),
            "regime_model": r.get("regime_model", ""),
            "vol_regime":   r.get("vol_regime", ""),
        })
    cards_df = pd.DataFrame(cards)
    cards_df.to_csv(out / "v15_latest_signal_cards.csv", index=False)

    # ---- Purged CV ----
    print("\n=== v15: purged CV (режимная модель) ===")
    wf_df = purged_cv_regime(df, all_features)
    wf_df.to_csv(out / "v15_purged_wf_cv.csv", index=False)
    if not wf_df.empty:
        mean_ba = wf_df["balanced_acc"].mean()
        std_ba  = wf_df["balanced_acc"].std()
        n_above = (wf_df["balanced_acc"] > 0.50).sum()
        print(f"  Фолдов: {len(wf_df)}, mean balanced_acc: {mean_ba:.3f} ± {std_ba:.3f}")
        print(f"  Фолдов выше 0.50 (случайного): {n_above}/{len(wf_df)}")
        print(wf_df.to_string(index=False))

    # ---- Сравнение v14 vs v15 ----
    v14_gr_path = Path("baseline_outputs_v14/v14_guardrails.csv")
    comparison_md = ""
    if v14_gr_path.exists():
        v14_gr = pd.read_csv(v14_gr_path)
        comp_rows = []
        split_map = {"valid": "valid", "test": "test", "forward": "forward"}
        for s in ["valid", "test", "forward"]:
            r14 = v14_gr[v14_gr["split"] == s]
            r15 = guardrails[guardrails["split"] == s]
            comp_rows.append({
                "split":         s,
                "v14_precision": pct(r14["precision"].values[0]) if not r14.empty else "-",
                "v15_precision": pct(r15["precision"].values[0]) if not r15.empty else "-",
                "v14_lift":      pct(r14["lift_vs_base"].values[0]) if not r14.empty else "-",
                "v15_lift":      pct(r15["lift_vs_base"].values[0]) if not r15.empty else "-",
                "v14_warning":   r14["warning"].values[0] if not r14.empty else "-",
                "v15_warning":   r15["warning"].values[0] if not r15.empty else "-",
            })
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(out / "v15_vs_v14_comparison.csv", index=False)
        comparison_md = f"\n## Сравнение v14 vs v15\n{md_table(comp_df)}\n"

    # ---- Отчёт ----
    cv_summary = ""
    if not wf_df.empty:
        cv_summary = f"Фолдов: {len(wf_df)}, mean balanced_acc: {wf_df['balanced_acc'].mean():.3f} ± {wf_df['balanced_acc'].std():.3f}, выше 0.50: {(wf_df['balanced_acc'] > 0.50).sum()}/{len(wf_df)}"

    report = f"""# Silver Trading Assistant v15 — итоговый отчёт

## Изменения vs v14
- **COT**: {len(present_cot)} признаков от CFTC (net_spec, cot_index_52w, cot_change_4w) с 4-дневным сдвигом публикации.
- **Горизонт**: {args.horizon} дней (было 5) → больше NEUTRAL-меток, меньше шума.
- **Режимная модель**: отдельные классификаторы для {list(model.models.keys())} + fallback.

## Режим triple-barrier: `{tb_mode}`

## Распределение меток (horizon={args.horizon})
{md_table(lb)}

## Метрики классификатора
{md_table(cls_df[[c for c in ["split","n","label_up_rate","balanced_accuracy","log_loss","auc_up_vs_rest"] if c in cls_df.columns]])}

## Guardrails
{md_table(guardrails)}

## Бэктест vs Buy-and-Hold
{md_table(bt_df)}
{comparison_md}
## Purged Walk-Forward CV
{cv_summary}
{md_table(wf_df) if not wf_df.empty else '_нет данных_'}

## Честный вывод
v15 добавляет реальные альтернативные данные (COT) и режимную структуру модели.
**Если balanced_accuracy в purged CV всё ещё < 0.50 и guardrails показывают ci_lower_not_above_base —
edge не доказан на дневных данных.** Следующий шаг: внутридневные данные (15-мин/часовые)
или другие альтернативные источники (опционный скью, SLV AUM-потоки).
"""
    (out / "v15_integrated_report.md").write_text(report, encoding="utf-8")
    print(f"\n=== v15 завершён. Результаты: {out.resolve()} ===")
    print("  Дашборд: streamlit run dashboard_app.py")


if __name__ == "__main__":
    main()
