"""
app/multi_asset/ood_detector.py — Out-Of-Distribution detector.

Если ключевые features (silver_dist_ma200, atr, gold_silver_ratio) выходят
за пределы 99-percentile training distribution → модель в untrained territory,
её предсказаниям нельзя доверять.

Это объясняет почему E3b пропустила силе ребро ралли 2025-2026: features были
extremal (silver_dist_ma200 >> historical max).

Использование:
    detector = OODDetector.fit(train_features)
    is_ood, score, reasons = detector.check(current_features)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Ключевые features для OOD контроля (наиболее volatility-sensitive)
KEY_FEATURES = [
    "silver_dist_ma200",  # дистанция от 200-MA — нет в training если ATH
    "silver_atr_14",      # ATR — экстремальная волатильность
    "silver_rvol_60",     # rolling vol — режим
    "ratio_gold_silver",  # ratio — относительная сила
]


@dataclass
class OODDetector:
    """Per-feature percentile bounds для OOD detection."""
    feature_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    pctile_low: float = 0.005     # 0.5th
    pctile_high: float = 0.995    # 99.5th
    n_train: int = 0

    @classmethod
    def fit(cls, train_features: pd.DataFrame,
            features: list[str] = None,
            pctile_low: float = 0.005,
            pctile_high: float = 0.995) -> "OODDetector":
        """Fit percentile bounds на training data."""
        features = features or KEY_FEATURES
        bounds = {}
        for f in features:
            if f not in train_features.columns:
                continue
            s = train_features[f].dropna()
            if len(s) < 100:
                continue
            lo = float(s.quantile(pctile_low))
            hi = float(s.quantile(pctile_high))
            bounds[f] = (lo, hi)
        return cls(
            feature_bounds=bounds,
            pctile_low=pctile_low,
            pctile_high=pctile_high,
            n_train=len(train_features),
        )

    def check(self, current_features: pd.Series | pd.DataFrame) -> dict:
        """
        Проверить текущие features. Возвращает:
          {
            'is_ood':       bool,
            'ood_score':    float,  # 0=ID, 1=полностью OOD
            'reasons':      list of {feature, value, bound, severity}
          }
        """
        if isinstance(current_features, pd.DataFrame):
            current_features = current_features.iloc[-1]

        reasons = []
        n_out = 0
        for f, (lo, hi) in self.feature_bounds.items():
            if f not in current_features.index:
                continue
            v = float(current_features[f])
            if not np.isfinite(v):
                continue
            if v < lo:
                severity = (lo - v) / (abs(hi - lo) + 1e-9)
                reasons.append({
                    "feature": f, "value": round(v, 4),
                    "bound_low": round(lo, 4), "bound_high": round(hi, 4),
                    "direction": "BELOW", "severity": round(severity, 2),
                })
                n_out += 1
            elif v > hi:
                severity = (v - hi) / (abs(hi - lo) + 1e-9)
                reasons.append({
                    "feature": f, "value": round(v, 4),
                    "bound_low": round(lo, 4), "bound_high": round(hi, 4),
                    "direction": "ABOVE", "severity": round(severity, 2),
                })
                n_out += 1

        total = len(self.feature_bounds)
        score = n_out / max(1, total)
        return {
            "is_ood":    n_out > 0,
            "ood_score": round(score, 2),
            "n_features_out": n_out,
            "n_features_checked": total,
            "reasons":   reasons,
            "summary":   _summarize(reasons) if reasons else "all features within training distribution",
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OODDetector":
        d = json.loads(path.read_text(encoding="utf-8"))
        # tuple восстанавливаем
        d["feature_bounds"] = {k: tuple(v) for k, v in d["feature_bounds"].items()}
        return cls(**d)


def _summarize(reasons: list[dict]) -> str:
    parts = []
    for r in reasons:
        parts.append(
            f"{r['feature']} = {r['value']} ({r['direction']} {r['bound_high'] if r['direction']=='ABOVE' else r['bound_low']}, severity {r['severity']})"
        )
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# CLI: fit на cached features и сохранить
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))
    from app.multi_asset.features import build_feature_frame
    from app.multi_asset.metal_loader import load_metals

    print("=" * 70)
    print(" Fit OOD detector on training distribution")
    print("=" * 70)
    metals = load_metals()
    features = build_feature_frame(target="silver", metals=metals, ffill_limit=5).dropna()

    # Use first 80% as training, last 20% as test
    n = len(features)
    train_end = int(n * 0.80)
    train_features = features.iloc[:train_end]
    test_features = features.iloc[train_end:]

    print(f"\nTrain: {len(train_features)} rows ({train_features.index.min().date()} -> {train_features.index.max().date()})")
    print(f"Test:  {len(test_features)} rows ({test_features.index.min().date()} -> {test_features.index.max().date()})")

    detector = OODDetector.fit(train_features)
    print(f"\nFitted detector on {len(detector.feature_bounds)} key features:")
    for f, (lo, hi) in detector.feature_bounds.items():
        print(f"  {f:30s} bounds [{lo:.4f}, {hi:.4f}]")

    # Save
    out = ROOT / "baseline_outputs_multiasset" / "ood_detector.json"
    detector.save(out)
    print(f"\nSaved -> {out}")

    # Test on test_features: in каких датах OOD?
    print(f"\nChecking last 30 days of data for OOD:")
    for d in features.index[-30:]:
        result = detector.check(features.loc[d])
        if result["is_ood"]:
            print(f"  ⚠ {d.date()}: OOD score {result['ood_score']:.2f} — {result['summary']}")

    # Сколько в test period было OOD?
    ood_days = sum(1 for d in test_features.index if detector.check(test_features.loc[d])["is_ood"])
    print(f"\nTest period OOD coverage: {ood_days}/{len(test_features)} days "
          f"({ood_days/len(test_features)*100:.1f}%)")
