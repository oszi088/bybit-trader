"""
Alternatív adat forrás — ingyenes Bybit + CoinGecko endpoint-ok.

Amit itt gyűjtünk:
  * Funding rate history   — perpetual futures finanszírozási díj
  * Open interest history  — nyitott pozíciók összértéke
  * Long/short ratio       — retail kereskedők pozíció-aránya
  * BTC dominance          — BTC piaci részesedés (makro sentiment)
  * ETH/BTC arány          — kockázatvállalási hajlandóság proxy

Miért fontos ezek az OHLCV-nél jobban?

  Funding rate: ha extrém pozitív (+0.1%/8h felett), a long pozíciók
  túlzsúfoltak — a piac egy "long squeeze"-re érzékeny. Kontrarian jel.

  Open interest + ár együtt:
    OI nő + ár nő  = erős trend (új pénz áramlik be)
    OI nő + ár esik = veszélyes (shortok nyílnak, potenciális cascade)
    OI csökken     = pozíciók zárása, trend gyengülhet

  Long/short ratio: ha 70%+ long → crowded trade → potenciális dump.

Ezek a retail kereskedők viselkedését mutatják — a nagyok ezt fade-elik.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("alt_data")

# Cache TTL másodpercben (ezek az adatok lassan változnak)
_FUNDING_TTL  = 3600   # 1 óra
_OI_TTL       = 1800   # 30 perc
_DOM_TTL      = 3600   # 1 óra
_LS_TTL       = 900    # 15 perc


@dataclass
class FundingSnapshot:
    symbol: str
    rate: float          # aktuális funding rate (pl. 0.0001 = 0.01%)
    annualized: float    # éves szintre vetítve (3 fizetés/nap × 365)
    timestamp: datetime

    @property
    def is_extreme_long(self) -> bool:
        return self.rate >= 0.001   # 0.1%/8h felett = túlzsúfolt long

    @property
    def is_extreme_short(self) -> bool:
        return self.rate <= -0.001  # -0.1%/8h alatt = túlzsúfolt short

    @property
    def signal(self) -> int:
        """Kontrarian: extrém long → -1 (sell), extrém short → +1 (buy)."""
        if self.is_extreme_long:  return -1
        if self.is_extreme_short: return  1
        return 0


@dataclass
class OpenInterestSnapshot:
    symbol: str
    oi_usd: float        # USD értékben
    oi_change_pct: float # változás az előző periódushoz képest
    timestamp: datetime

    @property
    def signal(self) -> int:
        """OI növekedés = erősödő meggyőződés (trend irányában)."""
        if self.oi_change_pct >= 5.0:   return  1
        if self.oi_change_pct <= -5.0:  return -1
        return 0


@dataclass
class LongShortSnapshot:
    symbol: str
    long_pct: float    # pl. 0.65 = 65% long
    short_pct: float
    timestamp: datetime

    @property
    def signal(self) -> int:
        """Kontrarian: ha >65% long → potenciális squeeze → -1."""
        if self.long_pct >= 0.65:  return -1
        if self.long_pct <= 0.35:  return  1
        return 0


@dataclass
class DominanceSnapshot:
    btc_dominance: float   # 0..100
    timestamp: datetime

    @property
    def signal(self) -> int:
        """
        BTC dominancia emelkedés = risk-off (altok gyengülnek).
        Csökkenés = altseason / risk-on.
        Csak extrém értékeken adunk jelet.
        """
        if self.btc_dominance >= 60: return  1   # BTC-be menekülés
        if self.btc_dominance <= 40: return -1   # altcoin eufória → óvatosság
        return 0


class AltDataSource:
    """
    Ingyenes alternatív adatok Bybit + CoinGecko API-ból.

    Minden lekérés cache-elt — a TTL-en belüli hívások nem mennek ki
    az API-ra (rate limit védelem).
    """

    def __init__(self, symbol: str = "BTC/USDT", endpoint: str = "testnet"):
        self.symbol   = symbol
        self.endpoint = endpoint
        self._cache: Dict[str, tuple] = {}   # key -> (timestamp, value)

    # ------------------------------------------------------------------ #
    # Cache helper
    # ------------------------------------------------------------------ #

    def _cached(self, key: str, ttl: int, fetcher):
        now = time.monotonic()
        if key in self._cache:
            ts, val = self._cache[key]
            if now - ts < ttl:
                return val
        val = fetcher()
        self._cache[key] = (now, val)
        return val

    # ------------------------------------------------------------------ #
    # Funding Rate (Bybit perpetual)
    # ------------------------------------------------------------------ #

    def funding_rate(self) -> Optional[FundingSnapshot]:
        """Aktuális funding rate a perphez (BTC/USDT:USDT linear)."""
        def _fetch():
            try:
                import ccxt
                ex = ccxt.bybit({"enableRateLimit": True,
                                 "options": {"defaultType": "linear"}})
                if self.endpoint == "testnet":
                    ex.set_sandbox_mode(True)
                perp_symbol = self.symbol.split("/")[0] + "/USDT:USDT"
                data = ex.fetch_funding_rate(perp_symbol)
                rate = float(data.get("fundingRate", 0.0))
                return FundingSnapshot(
                    symbol     = self.symbol,
                    rate       = rate,
                    annualized = rate * 3 * 365,
                    timestamp  = datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.warning("Funding rate lekérés sikertelen: %s", e)
                return None

        return self._cached(f"funding_{self.symbol}", _FUNDING_TTL, _fetch)

    def funding_rate_history(self, limit: int = 100) -> pd.DataFrame:
        """
        Historikus funding rate — feature engineering-hez.
        Visszatér: DataFrame [timestamp, rate]
        """
        try:
            import ccxt
            ex = ccxt.bybit({"enableRateLimit": True,
                             "options": {"defaultType": "linear"}})
            if self.endpoint == "testnet":
                ex.set_sandbox_mode(True)
            perp_symbol = self.symbol.split("/")[0] + "/USDT:USDT"
            rows = ex.fetch_funding_rate_history(perp_symbol, limit=limit)
            df = pd.DataFrame([{
                "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                "rate":      float(r["fundingRate"]),
            } for r in rows])
            return df.set_index("timestamp").sort_index()
        except Exception as e:
            logger.warning("Funding history sikertelen: %s", e)
            return pd.DataFrame(columns=["rate"])

    # ------------------------------------------------------------------ #
    # Open Interest (Bybit)
    # ------------------------------------------------------------------ #

    def open_interest(self) -> Optional[OpenInterestSnapshot]:
        """Aktuális open interest USD-ben."""
        def _fetch():
            try:
                import ccxt
                ex = ccxt.bybit({"enableRateLimit": True,
                                 "options": {"defaultType": "linear"}})
                if self.endpoint == "testnet":
                    ex.set_sandbox_mode(True)
                perp = self.symbol.split("/")[0] + "/USDT:USDT"
                data = ex.fetch_open_interest(perp)
                oi   = float(data.get("openInterestAmount", 0.0))
                return OpenInterestSnapshot(
                    symbol        = self.symbol,
                    oi_usd        = oi,
                    oi_change_pct = 0.0,   # history nélkül nem számítható
                    timestamp     = datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.warning("Open interest lekérés sikertelen: %s", e)
                return None

        return self._cached(f"oi_{self.symbol}", _OI_TTL, _fetch)

    def open_interest_history(self, limit: int = 100) -> pd.DataFrame:
        """Historikus OI — feature-ök számításához."""
        try:
            import ccxt
            ex = ccxt.bybit({"enableRateLimit": True,
                             "options": {"defaultType": "linear"}})
            if self.endpoint == "testnet":
                ex.set_sandbox_mode(True)
            perp = self.symbol.split("/")[0] + "/USDT:USDT"
            rows = ex.fetch_open_interest_history(perp, limit=limit)
            df = pd.DataFrame([{
                "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                "oi":        float(r.get("openInterestAmount", 0.0)),
            } for r in rows])
            return df.set_index("timestamp").sort_index()
        except Exception as e:
            logger.warning("OI history sikertelen: %s", e)
            return pd.DataFrame(columns=["oi"])

    # ------------------------------------------------------------------ #
    # Long/Short Ratio (Bybit)
    # ------------------------------------------------------------------ #

    def long_short_ratio(self) -> Optional[LongShortSnapshot]:
        """Retail kereskedők long/short aránya."""
        def _fetch():
            try:
                import ccxt
                ex = ccxt.bybit({"enableRateLimit": True,
                                 "options": {"defaultType": "linear"}})
                if self.endpoint == "testnet":
                    ex.set_sandbox_mode(True)
                # Bybit specifikus endpoint
                base = self.symbol.split("/")[0]
                resp = ex.publicGetV5MarketAccountRatio({
                    "category": "linear",
                    "symbol":   f"{base}USDT",
                    "period":   "1h",
                    "limit":    1,
                })
                row      = resp["result"]["list"][0]
                long_pct = float(row["buyRatio"])
                return LongShortSnapshot(
                    symbol    = self.symbol,
                    long_pct  = long_pct,
                    short_pct = 1.0 - long_pct,
                    timestamp = datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.warning("Long/short ratio sikertelen: %s", e)
                return None

        return self._cached(f"ls_{self.symbol}", _LS_TTL, _fetch)

    # ------------------------------------------------------------------ #
    # BTC Dominance (CoinGecko free)
    # ------------------------------------------------------------------ #

    def btc_dominance(self) -> Optional[DominanceSnapshot]:
        """BTC piaci dominancia — CoinGecko publikus API, kulcs nélkül."""
        def _fetch():
            try:
                import urllib.request, json
                url = "https://api.coingecko.com/api/v3/global"
                req = urllib.request.Request(url,
                      headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    data = json.loads(r.read())
                dom = float(data["data"]["market_cap_percentage"].get("btc", 50.0))
                return DominanceSnapshot(
                    btc_dominance = dom,
                    timestamp     = datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.warning("BTC dominance lekérés sikertelen: %s", e)
                return DominanceSnapshot(btc_dominance=50.0,
                                         timestamp=datetime.now(timezone.utc))

        return self._cached("btc_dom", _DOM_TTL, _fetch)

    # ------------------------------------------------------------------ #
    # Összesített alt-data jel
    # ------------------------------------------------------------------ #

    def composite_signal(self) -> Dict[str, int]:
        """
        Minden alt-data forrás jelét visszaadja egy dict-ben.
        Hiányzó forrás esetén 0 (semleges) a fallback.
        """
        signals: Dict[str, int] = {
            "funding":    0,
            "oi":         0,
            "long_short": 0,
            "btc_dom":    0,
        }
        if (f := self.funding_rate())    is not None: signals["funding"]    = f.signal
        if (o := self.open_interest())   is not None: signals["oi"]         = o.signal
        if (ls := self.long_short_ratio()) is not None: signals["long_short"] = ls.signal
        if (d := self.btc_dominance())   is not None: signals["btc_dom"]    = d.signal
        return signals
