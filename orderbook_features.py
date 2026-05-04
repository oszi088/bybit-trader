"""
Orderbook mikrostruktúra feature-ök.

Az árfolyam-adatból (OHLCV) nem látszik az, amit az orderbook megmutat:
  * Ki nyomja az árat — vevők vagy eladók?
  * Mekkora a bid/ask spread (likviditás)?
  * Hol vannak a nagy limit megbízások (support/resistance szintek)?

Ezeket a nagy alapok mikrostruktúra-elemzésnek hívják.
Mi ingyenesen elérhetjük a Bybit L2 orderbook API-ból.

Főbb metrikák:

  Order Book Imbalance (OBI):
    OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    OBI ∈ [-1, +1]
    +1 = csak vételek vannak → áremelkedés várható
    -1 = csak eladások vannak → áresés várható

  Bid-Ask Spread:
    (ask_price - bid_price) / mid_price
    Nagy spread = alacsony likviditás = volatilitás várható

  Depth Ratio:
    bid_depth_N / ask_depth_N (N szinten belül)
    >1 = több vételi megbízás → bullish nyomás

  Large Order Detection:
    Vannak-e kirívóan nagy megbízások a könyv tetején?
    (intézményi "jéghegy" megbízások jele)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("orderbook")


@dataclass
class OrderBookSnapshot:
    bids: List[Tuple[float, float]]   # [(ár, méret), ...]  csökkenő
    asks: List[Tuple[float, float]]   # [(ár, méret), ...]  növekvő
    timestamp: Optional[pd.Timestamp] = None

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else float("inf")

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        if mid <= 0:
            return 0.0
        return (self.best_ask - self.best_bid) / mid

    def imbalance(self, levels: int = 10) -> float:
        """
        Order Book Imbalance az első N szinten.
        +1 = teljes vételi nyomás, -1 = teljes eladási nyomás.
        """
        bid_vol = sum(sz for _, sz in self.bids[:levels])
        ask_vol = sum(sz for _, sz in self.asks[:levels])
        total   = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def depth_ratio(self, levels: int = 20) -> float:
        """Bid depth / Ask depth az első N szinten."""
        bid_vol = sum(sz for _, sz in self.bids[:levels])
        ask_vol = sum(sz for _, sz in self.asks[:levels])
        if ask_vol <= 0:
            return 1.0
        return bid_vol / ask_vol

    def large_order_signal(self, top_levels: int = 5,
                           size_multiplier: float = 5.0) -> int:
        """
        Detektál kirívóan nagy megbízásokat a könyv tetején.
        Ha a legjobb bid sokkal nagyobb mint az átlag → vételi fal (+1).
        Ha a legjobb ask sokkal nagyobb mint az átlag → eladási fal (-1).
        """
        if len(self.bids) < top_levels or len(self.asks) < top_levels:
            return 0

        avg_bid = np.mean([sz for _, sz in self.bids[:top_levels]])
        avg_ask = np.mean([sz for _, sz in self.asks[:top_levels]])

        top_bid = self.bids[0][1]
        top_ask = self.asks[0][1]

        if top_bid > avg_bid * size_multiplier:
            return  1   # vételi fal — support szint
        if top_ask > avg_ask * size_multiplier:
            return -1   # eladási fal — resistance szint
        return 0


class OrderBookFetcher:
    """Bybit L2 orderbook lekérdezése CCXT-n keresztül."""

    def __init__(self, symbol: str, endpoint: str = "testnet",
                 market_type: str = "spot"):
        self.symbol      = symbol
        self.endpoint    = endpoint
        self.market_type = market_type
        self._exchange   = None

    def _get_exchange(self):
        if self._exchange is None:
            try:
                import ccxt
                self._exchange = ccxt.bybit({
                    "enableRateLimit": True,
                    "options": {"defaultType": self.market_type},
                })
                if self.endpoint == "testnet":
                    self._exchange.set_sandbox_mode(True)
            except ImportError:
                raise ImportError("pip install ccxt")
        return self._exchange

    def fetch(self, depth: int = 50) -> Optional[OrderBookSnapshot]:
        """L2 orderbook lekérdezése."""
        try:
            ex = self._get_exchange()
            ob = ex.fetch_order_book(self.symbol, limit=depth)
            return OrderBookSnapshot(
                bids      = [(float(p), float(s)) for p, s in ob["bids"]],
                asks      = [(float(p), float(s)) for p, s in ob["asks"]],
                timestamp = pd.Timestamp.now(tz="UTC"),
            )
        except Exception as e:
            logger.warning("Orderbook lekérés sikertelen (%s): %s", self.symbol, e)
            return None

    def feature_dict(self, depth: int = 50) -> dict:
        """
        Egyetlen lekérdezésből az összes mikrostruktúra feature dict-ként.
        Fallback: semleges értékek ha az API nem elérhető.
        """
        ob = self.fetch(depth)
        if ob is None:
            return {
                "ob_imbalance_10":    0.0,
                "ob_imbalance_20":    0.0,
                "ob_depth_ratio_20":  1.0,
                "ob_spread_pct":      0.0,
                "ob_large_order":     0,
            }
        return {
            "ob_imbalance_10":   ob.imbalance(levels=10),
            "ob_imbalance_20":   ob.imbalance(levels=20),
            "ob_depth_ratio_20": ob.depth_ratio(levels=20),
            "ob_spread_pct":     ob.spread_pct,
            "ob_large_order":    ob.large_order_signal(),
        }


# ============================================================================
# Historikus OB imbalance proxy (backtesthez — valódi OB nincs)
# ============================================================================

def estimate_ob_imbalance_from_ohlcv(ohlcv: pd.DataFrame,
                                      window: int = 14) -> pd.Series:
    """
    Orderbook imbalance közelítése OHLCV-ből backtesthez.

    A valódi OB nem áll rendelkezésre historikusan, de az ár pozíciója
    a gyertya high-low tartományán belül és a volumen együtt ad egy
    durva proxyt:
      ha close közel a high-hoz és nagy volumen → vételi nyomás
      ha close közel a low-hoz és nagy volumen  → eladási nyomás
    """
    hl_range = (ohlcv["high"] - ohlcv["low"]).replace(0, np.nan)
    position = (ohlcv["close"] - ohlcv["low"]) / hl_range  # 0..1
    # Centrálás: 0.5 = semleges
    centered = (position - 0.5) * 2   # -1..+1

    vol_norm = ohlcv["volume"] / ohlcv["volume"].rolling(window).mean()
    vol_norm = vol_norm.fillna(1.0).clip(0.1, 5.0)

    return (centered * vol_norm).rolling(window).mean()
