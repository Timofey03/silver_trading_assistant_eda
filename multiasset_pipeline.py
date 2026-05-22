"""CLI для запуска полного multi-asset data pipeline.

Использование:
    # Полный pipeline (refresh данные, сохранить features и labels, отчёт)
    python multiasset_pipeline.py --refresh

    # Только отчёт по существующим данным
    python multiasset_pipeline.py --report-only

    # Только обновить metals (не трогая macro)
    python multiasset_pipeline.py --metals-only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.multi_asset.metal_loader import refresh_metals_cache, load_metals
from app.multi_asset.macro_loader import load_macro, assemble_macro_frame
from app.multi_asset.features import build_feature_frame
from app.multi_asset.labels import build_multi_horizon_labels
from app.multi_asset.quality_report import build_report, save_report
from app.multi_asset.config import FEATURES_DIR, LABELS_DIR, HORIZONS


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_full_pipeline(force_refresh: bool = False) -> None:
    logger.info("=" * 70)
    logger.info("MULTI-ASSET PIPELINE — start")
    logger.info("=" * 70)

    # Step 1: Metals
    logger.info("\n[1/5] Loading 5 metals...")
    if force_refresh:
        refresh_metals_cache(verbose=True)
    metals = load_metals(force_refresh=False)
    logger.info(f"  → {len([m for m, df in metals.items() if not df.empty])}/5 metals loaded")

    # Step 2: Macro
    logger.info("\n[2/5] Loading macro indicators...")
    macro = load_macro(force_refresh=force_refresh)
    logger.info(f"  → {len([m for m, df in macro.items() if not df.empty])} macro series loaded")

    # Step 3: Features
    logger.info("\n[3/5] Building feature frame...")
    frame = build_feature_frame(target="silver", metals=metals)
    features_path = FEATURES_DIR / "silver_features.parquet"
    frame.to_parquet(features_path, compression="snappy")
    logger.info(f"  → {len(frame.columns)} cols × {len(frame):,} rows → {features_path.name}")

    # Step 4: Labels
    logger.info("\n[4/5] Building multi-horizon labels...")
    silver = metals["silver"]
    labels = build_multi_horizon_labels(silver["close"], silver["high"], silver["low"])
    labels_path = LABELS_DIR / "silver_labels.parquet"
    labels.to_parquet(labels_path, compression="snappy")
    logger.info(f"  → {len(labels.columns)} cols × {len(labels):,} rows → {labels_path.name}")

    # Step 5: Report
    logger.info("\n[5/5] Building quality report...")
    report = build_report()
    report_path = save_report(report)
    logger.info(f"  → {report_path}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE ✓")
    logger.info("=" * 70)
    logger.info(f"\nЧтобы посмотреть отчёт: cat {report_path}")


def report_only() -> None:
    logger.info("Building report from existing cache...")
    report = build_report()
    path = save_report(report)
    print(f"\nReport: {path}")


def metals_only(force_refresh: bool = True) -> None:
    logger.info("Refreshing metals only...")
    refresh_metals_cache(verbose=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-asset data pipeline")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh всех данных")
    parser.add_argument("--report-only", action="store_true",
                        help="Только отчёт по существующим данным")
    parser.add_argument("--metals-only", action="store_true",
                        help="Только обновить metals")
    args = parser.parse_args()

    if args.report_only:
        report_only()
    elif args.metals_only:
        metals_only()
    else:
        run_full_pipeline(force_refresh=args.refresh)


if __name__ == "__main__":
    main()
