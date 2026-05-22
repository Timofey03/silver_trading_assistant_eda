"""Quality report для multi-asset dataset.

Запускается после полного pipeline. Создаёт markdown-отчёт в data/multi_asset/reports/
с диагностикой каждого этапа.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.multi_asset.config import METALS, MACRO, HORIZONS, REPORTS_DIR
from app.multi_asset.metal_loader import load_metals
from app.multi_asset.macro_loader import load_macro
from app.multi_asset.features import build_feature_frame
from app.multi_asset.labels import build_multi_horizon_labels

logger = logging.getLogger(__name__)


def section(title: str) -> str:
    return f"\n## {title}\n\n"


def build_report() -> str:
    """Собрать полный markdown-отчёт по data pipeline."""
    lines = []
    lines.append(f"# Multi-Asset Pipeline Quality Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # ===== Метал-данные =====
    lines.append(section("1. Metal data (yfinance)"))
    metals = load_metals()
    lines.append("| Metal | Ticker | Rows | First date | Last date | Years |")
    lines.append("|---|---|---:|---|---|---:|")
    for m, df in metals.items():
        ticker = METALS[m]["ticker"]
        if df.empty:
            lines.append(f"| {m} | {ticker} | FAILED | — | — | — |")
            continue
        years = round((df.index.max() - df.index.min()).days / 365.25, 1)
        lines.append(f"| {m} | {ticker} | {len(df):,} | "
                     f"{df.index.min().date()} | {df.index.max().date()} | {years} |")

    # ===== Макроданные =====
    lines.append(section("2. Macro data (FRED + yfinance)"))
    macro = load_macro()
    lines.append("| Series | Source | Rows | Frequency | Description |")
    lines.append("|---|---|---:|---|---|")
    for name, df in macro.items():
        cfg = MACRO[name]
        rows = len(df) if not df.empty else 0
        lines.append(f"| {name} | {cfg['source']} | {rows:,} | "
                     f"{cfg['freq']} | {cfg['description']} |")

    # ===== Feature frame =====
    lines.append(section("3. Feature engineering"))
    frame = build_feature_frame(target="silver", metals=metals)
    n_clean = len(frame.dropna())
    n_total = len(frame)
    lines.append(f"- **Total rows**: {n_total:,}")
    lines.append(f"- **Total columns (features)**: {len(frame.columns)}")
    lines.append(f"- **Rows after dropna**: {n_clean:,} "
                 f"({n_clean / n_total * 100:.1f}% retention)")
    lines.append(f"- **Period**: {frame.index.min().date()} → {frame.index.max().date()}")

    # Feature categories
    per_asset_cols = [c for c in frame.columns
                      if any(c.startswith(f"{m}_") for m in METALS)]
    ratio_cols = [c for c in frame.columns if c.startswith("ratio_")]
    corr_cols = [c for c in frame.columns if c.startswith("corr_")]
    macro_cols = [c for c in frame.columns if c in MACRO]
    age_cols = [c for c in frame.columns if c.endswith("_age_days")]
    other = [c for c in frame.columns
             if c not in per_asset_cols + ratio_cols + corr_cols + macro_cols + age_cols
             and not c.startswith("target_")]

    lines.append(f"\n### Feature breakdown")
    lines.append(f"- Per-asset technical: {len(per_asset_cols)}")
    lines.append(f"- Cross-asset ratios: {len(ratio_cols)}")
    lines.append(f"- Cross-asset correlations: {len(corr_cols)}")
    lines.append(f"- Macro raw: {len(macro_cols)}")
    lines.append(f"- Macro age_days: {len(age_cols)}")
    lines.append(f"- Composite/other: {len(other)}")

    # ===== Labels =====
    lines.append(section("4. Multi-horizon labels"))
    silver = metals["silver"]
    labels = build_multi_horizon_labels(silver["close"], silver["high"], silver["low"])

    lines.append(f"- **Horizons**: {HORIZONS}")
    lines.append(f"- **Total label rows**: {len(labels):,}")
    lines.append(f"\n### Label distribution per horizon")
    lines.append("| Horizon | TP (+1) | Timeout (0) | SL (−1) | NaN |")
    lines.append("|---|---:|---:|---:|---:|")
    for h in HORIZONS:
        col = f"label_{h}"
        vc = labels[col].value_counts(dropna=False)
        tp = vc.get(1.0, 0)
        zero = vc.get(0.0, 0)
        sl = vc.get(-1.0, 0)
        nan = vc.get(np.nan, 0) if labels[col].isna().any() else 0
        total = len(labels)
        lines.append(f"| {h} days | {tp:,} ({tp / total * 100:.0f}%) | "
                     f"{zero:,} ({zero / total * 100:.0f}%) | "
                     f"{sl:,} ({sl / total * 100:.0f}%) | {nan} |")

    lines.append(f"\n### Regime distribution")
    vc = labels["regime"].value_counts(dropna=False)
    for reg, cnt in vc.items():
        lines.append(f"- **{reg}**: {cnt:,} ({cnt / len(labels) * 100:.0f}%)")

    # ===== Supervision pairs =====
    lines.append(section("5. Effective training size"))
    clean = frame.dropna().index.intersection(labels.dropna(subset=[f"label_{HORIZONS[0]}"]).index)
    n_clean_silver = len(clean)
    n_metals = sum(1 for d in metals.values() if not d.empty)
    n_horizons = len(HORIZONS)
    supervision = n_clean_silver * n_horizons

    lines.append(f"- **Clean silver days**: {n_clean_silver:,}")
    lines.append(f"- **Metals available for backbone**: {n_metals}")
    lines.append(f"- **Horizons**: {n_horizons}")
    lines.append(f"- **Effective supervision pairs (silver only)**: "
                 f"{supervision:,}")
    lines.append(f"- **Effective with cross-asset backbone (~×{n_metals})**: "
                 f"~{supervision * n_metals:,}")

    # ===== Сводка =====
    lines.append(section("6. Summary"))
    lines.append(f"- Old setup: ~3 000 supervision pairs (silver daily, 2018-2025, "
                 f"1 horizon)")
    lines.append(f"- New setup: ~{supervision:,} pairs (silver), "
                 f"~{supervision * n_metals:,} with backbone")
    lines.append(f"- **Effective data growth: ×{supervision * n_metals / 3000:.0f}**")

    return "\n".join(lines)


def save_report(report: str) -> Path:
    path = REPORTS_DIR / "quality_report.md"
    path.write_text(report, encoding="utf-8")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Building quality report...")
    report = build_report()
    path = save_report(report)
    print(f"\nReport saved: {path}\n")
    print("=" * 70)
    print(report[-2000:])  # print last 2k chars
