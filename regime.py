"""
Piaci rezsim detektor.

A rezsim alapjan valt sulyokat az ugynok:
  - TREND  (ADX magas, +DI vs -DI tisztan elkulonul) -> trendkoveto sulyok
  - RANGE  (ADX alacsony, mean-reversion mukodik)    -> oscilattor sulyok
  - NEUTRAL (atmenet)                                -> default sulyok
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from config import (
    DEFAULT_WEIGHTS,
    RANGE_WEIGHTS,
    RegimeConfig,
    TREND_WEIGHTS,
)


@dataclass
class RegimeReading:
    label: str          # "trend", "range", "neutral"
    adx: float
    weights: Dict[str, float]


def detect_regime(row: pd.Series, config: RegimeConfig) -> RegimeReading:
    """
    Rezsim detekcio az aktualis sor ADX-e alapjan.
    A row-ban legyen: adx, plus_di, minus_di (ezeket az indicators.compute_all adja).
    """
    if not config.enabled or pd.isna(row.get("adx", float("nan"))):
        return RegimeReading("neutral", float("nan"), dict(DEFAULT_WEIGHTS))

    adx = float(row["adx"])

    if adx >= config.adx_trend_threshold:
        return RegimeReading("trend", adx, dict(TREND_WEIGHTS))
    if adx <= config.adx_range_threshold:
        return RegimeReading("range", adx, dict(RANGE_WEIGHTS))

    # atmenet -> default
    return RegimeReading("neutral", adx, dict(DEFAULT_WEIGHTS))
