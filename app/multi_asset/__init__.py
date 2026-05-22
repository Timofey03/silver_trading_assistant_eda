"""Multi-asset data pipeline для дипломных экспериментов E2–E5.

Цель: построить обучающую базу из 5 металлов + макрофакторы для
cross-asset transfer learning. Эффективный размер выборки ~65 000
supervision pairs (vs ~3 000 текущих).
"""
from app.multi_asset.config import METALS, MACRO, DATA_DIR, START_DATE, END_DATE
from app.multi_asset.metal_loader import (
    load_metals,
    load_single_metal,
    refresh_metals_cache,
)

__all__ = [
    "METALS", "MACRO", "DATA_DIR", "START_DATE", "END_DATE",
    "load_metals", "load_single_metal", "refresh_metals_cache",
]
