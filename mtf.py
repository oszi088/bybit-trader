"""
Multi-TimeFrame (MTF) elemzes.

Az alacsonyabb timeframe-en (pl. 1h) hozott dontest megerositi (vagy
akadalyozza) a magasabb timeframe-ek trendje. Ket modban hasznalhato:

  * "weighted" - a magasabb tf-ek osszesitett trend score-ja egy plusz
                 szavazatkent szamit a fo dontesi score-ban.
  * "gate"     - ha a magasabb tf-ek erosen ellentmondanak a foi
                 jelnek, a Decision-t HOLD-ra korlatozza.

Tamogatott tf-ek: 6h, 8h, 12h, 1d, 1w, 1M (Bybit native), valamint a
'yearly' nezet (12 honap havi gyertyabol).

Az MTFAnalyzer kicsit kulonbozoen mukodik backteszt vs eles modban:
  * Backtestben pre-loadoljuk a resample-elt OHLCV-ket es slice-eljuk
    az aktualis idobelyegig (no look-ahead bias).
  * Elesben a Trader periodikusan letolti CCXT-vel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("mtf")


@dataclass
class MTFReading:
    timeframe_signals: Dict[str, int] = field(default_factory=dict)  # tf -> {-1, 0, +1}
    composite_score: float = 0.0     # -1..+1 sulyozott osszeg

    @property
    def label(self) -> str:
        if self.composite_score >= 0.4:
            return "bullish"
        if self.composite_score <= -0.4:
            return "bearish"
        return "mixed"

    def explain(self) -> str:
        parts = []
        for tf, s in self.timeframe_signals.items():
            arrow = "+" if s > 0 else ("-" if s < 0 else ".")
            parts.append(f"{tf}={arrow}")
        return f"MTF[{','.join(parts)}] score={self.composite_score:+.2f} ({self.label})"


class MTFAnalyzer:
    """
    Tobb timeframe-en futtatja az SMA-cross trendjelzot, sulyozottan
    aggregalja az eredmenyt.
    """

    def __init__(self, timeframes: List[str], weights: Dict[str, float],
                 fast: int = 20, slow: int = 50):
        self.timeframes = list(timeframes)
        self.weights = dict(weights)
        self.fast = fast
        self.slow = slow
        # tf -> teljes OHLCV DataFrame (idoindex)
        self._data: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    # Adatfeltoltes (a backtest vagy a Trader hivja)
    # ------------------------------------------------------------------ #

    def set_data(self, tf: str, ohlcv: pd.DataFrame) -> None:
        """Egy timeframe-hez tartozo teljes OHLCV beallitasa."""
        if not ohlcv.empty:
            self._data[tf] = ohlcv.sort_index()

    def has_data(self, tf: str) -> bool:
        return tf in self._data and not self._data[tf].empty

    # ------------------------------------------------------------------ #
    # Trend szignal egy timeframe-en
    # ------------------------------------------------------------------ #

    def _trend_signal(self, df: pd.DataFrame) -> int:
        """Egyszeru SMA-cross trend: gyors > lassu => +1, alatta => -1."""
        if df is None or df.empty:
            return 0
        # Megfelelo periodusok: ha keves az adat, alkalmazkodunk
        n = len(df)
        if n < 5:
            return 0
        fast_p = min(self.fast, max(2, n // 3))
        slow_p = min(self.slow, max(fast_p + 1, n - 1))

        close = df["close"]
        f = close.rolling(fast_p).mean().iloc[-1]
        s = close.rolling(slow_p).mean().iloc[-1]
        if pd.isna(f) or pd.isna(s):
            return 0
        if f > s:
            return 1
        if f < s:
            return -1
        return 0

    # ------------------------------------------------------------------ #
    # Aggregalt elemzes
    # ------------------------------------------------------------------ #

    def analyze(self, as_of: Optional[pd.Timestamp] = None) -> MTFReading:
        """
        Kiszamolja minden timeframe-re a trendjelzot, es sulyozottan aggregalja.
        Ha `as_of` meg van adva, a slice-eles oda van limitalva (no look-ahead).
        """
        signals: Dict[str, int] = {}
        for tf in self.timeframes:
            df = self._data.get(tf)
            if df is None:
                signals[tf] = 0
                continue
            if as_of is not None:
                df = df[df.index <= as_of]
            signals[tf] = self._trend_signal(df)

        # Sulyozott osszeg, normalizalva
        total_w = sum(abs(self.weights.get(tf, 0.0)) for tf in self.timeframes) or 1.0
        score = sum(self.weights.get(tf, 0.0) * signals[tf] for tf in self.timeframes) / total_w
        return MTFReading(timeframe_signals=signals, composite_score=score)


# ============================================================================
# Resample helper - backtesthez
# ============================================================================

# pandas resample alias-ek a TF stringekhez
# Sub-5min frame-ekhez is hasznalhato (scalping mod): 1m-es CSV-bol fel
# tudunk resample-elni 3m/5m/15m/30m/1h/2h/4h-ra is.
RESAMPLE_RULES: Dict[str, str] = {
    "1m":  "1min",
    "3m":  "3min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "6h":  "6h",
    "8h":  "8h",
    "12h": "12h",
    "1d":  "1D",
    "1w":  "1W",
    "1M":  "1ME",   # honap-ev (pandas 2.x)
}


def resample_ohlcv(ohlcv: pd.DataFrame, tf: str) -> pd.DataFrame:
    """1h (vagy alacsonyabb) OHLCV-t aggregalja egy magasabb tf-re."""
    if tf not in RESAMPLE_RULES:
        raise ValueError(f"Ismeretlen MTF timeframe: {tf}")
    rule = RESAMPLE_RULES[tf]
    agg = {
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }
    return ohlcv.resample(rule).agg(agg).dropna()
