"""
Crypto Fear & Greed Index (alternative.me).

A F&G napi szintu hangulati mutato (0-100):
   0-24   Extreme Fear   -> kontrarian: vasarlasi lehetoseg
  25-44   Fear
  45-55   Neutral
  56-74   Greed
  75-100  Extreme Greed  -> kontrarian: elado oldali nyomas

API: https://api.alternative.me/fng/
- Nyilvanos, kulcs nem szukseges
- Naponta 1x frissul (UTC 00:00), tehat agresszivan cache-eljuk

A backteszt es a felhasznalo nem szakad meg, ha az API nem elerheto:
ilyenkor 'neutral' (50) erteket adunk vissza.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("fear_greed")

API_URL = "https://api.alternative.me/fng/?limit=1&format=json"
CACHE_TTL_SEC = 60 * 60   # 1 ora; a F&G napi, igy boven eleg


@dataclass
class FearGreedReading:
    value: int               # 0..100
    classification: str      # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: datetime

    @property
    def is_extreme_fear(self) -> bool:
        return self.value <= 24

    @property
    def is_extreme_greed(self) -> bool:
        return self.value >= 75


class FearGreedSource:
    """Cache-elt F&G lekero, halozati hibara graceful."""

    def __init__(self, ttl_sec: int = CACHE_TTL_SEC):
        self.ttl_sec = ttl_sec
        self._cached: Optional[FearGreedReading] = None
        self._fetched_at: float = 0.0

    def _fetch(self) -> Optional[FearGreedReading]:
        try:
            with urllib.request.urlopen(API_URL, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as e:
            logger.warning("Fear & Greed API hiba: %s", e)
            return None

        try:
            entry = payload["data"][0]
            return FearGreedReading(
                value=int(entry["value"]),
                classification=str(entry.get("value_classification", "")),
                timestamp=datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("F&G valasz feldolgozasa sikertelen: %s", e)
            return None

    def get(self) -> FearGreedReading:
        """Frissit, ha lejart a cache. Ha az API nem elerheto, neutral 50-et ad."""
        now = time.time()
        if self._cached and (now - self._fetched_at) < self.ttl_sec:
            return self._cached

        reading = self._fetch()
        if reading is None:
            # Fallback: neutral, hogy a strategia ne essen ossze
            reading = FearGreedReading(
                value=50,
                classification="Neutral (fallback)",
                timestamp=datetime.now(timezone.utc),
            )
        self._cached = reading
        self._fetched_at = now
        return reading
