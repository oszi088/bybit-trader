"""
fear_greed_history.py — Historikus Fear & Greed adatok

Az alternative.me API-tól letölti az összes historikus F&G adatot
(2018 február óta), és dátum → érték szótárként tárolja.

Fő felhasználás: lookahead-mentes ML tanítás.
  - Élő kereskedésnél: fear_greed.FearGreedSource (live API)
  - ML train loopban: FearGreedHistory.get_value(date) → int

URL: https://api.alternative.me/fng/?limit=0
  → visszaadja az összes historikus bejegyzést; API kulcs nem szükséges.

CSV cache: data/fear_greed_history.csv (ha a data/ könyvtár létezik)
  Automatikusan újratölti, ha a fájl több mint 24 óra régi.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("fear_greed_history")

_API_URL = "https://api.alternative.me/fng/?limit=0&format=json"
_DEFAULT_CACHE = Path("data/fear_greed_history.csv")
_CACHE_MAX_AGE_HOURS = 24


class FearGreedHistory:
    """
    Dátum-alapú historikus Fear & Greed lookup.

    Példa:
        fgh = FearGreedHistory.load()
        value = fgh.get_value(date(2022, 11, 8))  # FTX összeomlás napja
        # → 24 (Extreme Fear)
    """

    def __init__(self, data: Dict[date, int]) -> None:
        self._data: Dict[date, int] = data

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        cache_path: Optional[Path] = _DEFAULT_CACHE,
        force_refresh: bool = False,
    ) -> "FearGreedHistory":
        """
        Betölti a historikus adatokat.

        Sorrend:
          1. Ha a CSV cache friss (< 24h) és nem kérünk frissítést → onnan tölt.
          2. Ellenkező esetben letölti az API-tól, elmenti a cache-be.
          3. Ha az API nem elérhető és van cache (bármilyen régi) → azt használja.
          4. Ha semmi sincs → üres historikummal indul (fallback: 50).
        """
        cache_path = Path(cache_path) if cache_path else None
        cache_ok = (
            cache_path is not None
            and cache_path.exists()
            and not force_refresh
            and cls._cache_fresh(cache_path)
        )

        if cache_ok:
            logger.info("F&G history betöltés cache-ből: %s", cache_path)
            return cls._load_csv(cache_path)

        logger.info("F&G history letöltés API-ról (%s)...", _API_URL)
        data = cls._fetch_from_api()

        if data:
            logger.info("  %d nap F&G adat letöltve.", len(data))
            if cache_path is not None:
                cls._save_csv(data, cache_path)
        else:
            logger.warning("API letöltés sikertelen. Fallback: cache vagy üres.")
            if cache_path is not None and cache_path.exists():
                return cls._load_csv(cache_path)
            return cls({})

        return cls(data)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_value(self, query_date: date) -> int:
        """
        Visszaadja a Fear & Greed értéket az adott napra.

        Ha a pontos dátum nem található, az előző (legközelebbi korábbi)
        napot keresi, maximum 7 napra visszamenőleg.
        Fallback: 50 (Neutral).
        """
        for delta in range(8):
            d = query_date - timedelta(days=delta)
            if d in self._data:
                return self._data[d]
        logger.debug("F&G adat nem található: %s (±7 nap). Fallback: 50.", query_date)
        return 50

    def get_value_for_ts(self, timestamp: datetime) -> int:
        """Timestamp (UTC-aware vagy naiv) alapján keres."""
        if timestamp.tzinfo is not None:
            d = timestamp.astimezone(timezone.utc).date()
        else:
            d = timestamp.date()
        return self.get_value(d)

    def __len__(self) -> int:
        return len(self._data)

    def date_range(self) -> tuple[Optional[date], Optional[date]]:
        """(legkorábbi, legkésőbbi) dátum a historikumban."""
        if not self._data:
            return None, None
        return min(self._data), max(self._data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_fresh(path: Path) -> bool:
        import os
        age_seconds = datetime.now().timestamp() - os.path.getmtime(path)
        return age_seconds < _CACHE_MAX_AGE_HOURS * 3600

    @staticmethod
    def _fetch_from_api() -> Dict[date, int]:
        try:
            with urllib.request.urlopen(_API_URL, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as e:
            logger.warning("F&G API hiba: %s", e)
            return {}

        data: Dict[date, int] = {}
        try:
            for entry in payload.get("data", []):
                ts_unix = int(entry["timestamp"])
                dt = datetime.fromtimestamp(ts_unix, tz=timezone.utc).date()
                data[dt] = int(entry["value"])
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("F&G JSON feldolgozási hiba: %s", e)
            return {}

        return data

    @staticmethod
    def _save_csv(data: Dict[date, int], path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = ["date,value"]
            for d, v in sorted(data.items()):
                lines.append(f"{d},{v}")
            path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("F&G history cache elmentve: %s (%d sor)", path, len(data))
        except OSError as e:
            logger.warning("F&G cache mentés sikertelen: %s", e)

    @staticmethod
    def _load_csv(path: Path) -> "FearGreedHistory":
        data: Dict[date, int] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    data[date.fromisoformat(parts[0])] = int(parts[1])
            logger.info("F&G history cache betöltve: %d nap (%s)", len(data), path)
        except (OSError, ValueError) as e:
            logger.warning("F&G cache betöltés sikertelen: %s", e)
        return FearGreedHistory(data)
