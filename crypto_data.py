"""
Crypto-specifikus ingyenes adat forrás aggregátor.

Források (mind ingyenes, API kulcs nélkül vagy free tier-rel):

  yfinance        — SP500, DXY, Gold, VIX, arany
  Blockchain.com  — hash rate, mempool, tx volume, miner revenue
  Coinglass       — aggregált likvidációk, long/short ratio (cross-exchange)
  Deribit         — BTC/ETH options: put/call ratio, implied volatility
  CryptoPanic     — hír sentiment (opcionális free API key)

Miért ezek a legrelevásabbak crypto tradinghez:

  SP500 / VIX:
    A BTC korrelációja az SP500-zal 2020 óta magas (0.6-0.8 risk-off
    eseményeknél). VIX > 30 = tőzsdei pánik = crypto dump is jön.

  DXY (dollár index):
    Inverz korreláció: erős dollár → gyenge kockázati eszközök → BTC esik.
    DXY trendfordulók jól megjósolják BTC irányváltásait.

  Hash rate:
    A bányászok a legtájékozottabb hosszú-távú szereplők. Csökkenő hash
    rate = bányászok lekapcsolnak = bearish. Emelkedő = invesztálnak.

  Exchange netflow:
    BTC tőzsdére áramlik → eladni akarják → bearish.
    BTC tőzsdéről kimegy → cold storage → bullish (akkumuláció).

  Likvidációk (Coinglass):
    Nagy long-likvidáció hullám után általában reversal következik
    (a "cascades" végén kimerül az eladási nyomás).

  Deribit put/call ratio:
    Opciós piac = "okos pénz". Magas put/call = félelem → kontrarian buy.
    IV skew = mennyivel drágábbak a put-ok a call-oknál.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("crypto_data")

_CACHE: Dict[str, tuple] = {}   # key -> (monotonic_ts, value)


def _cached(key: str, ttl: int, fetcher):
    now = time.monotonic()
    if key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < ttl:
            return val
    val = fetcher()
    _CACHE[key] = (now, val)
    return val


def _get_json(url: str, timeout: int = 8) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ============================================================================
# 1. MAKRO — SP500, DXY, VIX, Gold (yfinance)
# ============================================================================

@dataclass
class MacroSnapshot:
    sp500_ret_1d:   float = 0.0   # SP500 1 napos hozam
    sp500_ret_7d:   float = 0.0   # SP500 7 napos hozam
    dxy_level:      float = 100.0 # Dollár index szintje
    dxy_ret_5d:     float = 0.0   # DXY 5 napos hozam (trend)
    vix_level:      float = 20.0  # Fear index (>30 = pánik)
    gold_ret_5d:    float = 0.0   # Arany 5 napos hozam (safe haven flow)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def vix_signal(self) -> int:
        """VIX > 30 = tőzsdei pánik → crypto dump is valószínű."""
        if self.vix_level >= 35: return -1
        if self.vix_level <= 15: return  1   # nagyon alacsony félelem = kockázatvállalás
        return 0

    @property
    def dxy_signal(self) -> int:
        """Erősödő dollár = risk-off = BTC bearish."""
        if self.dxy_ret_5d >= 0.01:  return -1
        if self.dxy_ret_5d <= -0.01: return  1
        return 0


def fetch_macro(lookback_days: int = 30) -> Optional[MacroSnapshot]:
    """SP500, DXY, VIX, Gold adatok yfinance-ből."""
    def _fetch():
        try:
            import yfinance as yf
            import numpy as np

            tickers = yf.download(
                ["^GSPC", "DX-Y.NYB", "^VIX", "GC=F"],
                period=f"{lookback_days}d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            close = tickers["Close"]

            def ret(col, periods):
                if col not in close.columns or len(close) < periods + 1:
                    return 0.0
                return float(close[col].pct_change(periods).iloc[-1])

            vix = float(close["^VIX"].iloc[-1]) if "^VIX" in close.columns else 20.0
            dxy = float(close["DX-Y.NYB"].iloc[-1]) if "DX-Y.NYB" in close.columns else 100.0

            return MacroSnapshot(
                sp500_ret_1d = ret("^GSPC", 1),
                sp500_ret_7d = ret("^GSPC", 7),
                dxy_level    = dxy,
                dxy_ret_5d   = ret("DX-Y.NYB", 5),
                vix_level    = vix,
                gold_ret_5d  = ret("GC=F", 5),
            )
        except Exception as e:
            logger.warning("Makro adat sikertelen: %s", e)
            return MacroSnapshot()

    return _cached("macro", 3600, _fetch)


def fetch_macro_history(symbol_map: dict = None, years: float = 1.0) -> pd.DataFrame:
    """
    Historikus makro adat DataFrame-ként — ML tanításhoz.
    symbol_map: {"sp500": "^GSPC", "dxy": "DX-Y.NYB", "vix": "^VIX", "gold": "GC=F"}
    """
    if symbol_map is None:
        symbol_map = {
            "sp500": "^GSPC",
            "dxy":   "DX-Y.NYB",
            "vix":   "^VIX",
            "gold":  "GC=F",
        }
    try:
        import yfinance as yf
        period = f"{int(years * 365)}d"
        raw = yf.download(
            list(symbol_map.values()),
            period=period, interval="1d",
            progress=False, auto_adjust=True,
        )["Close"]
        raw.columns = list(symbol_map.keys())
        return raw.sort_index()
    except Exception as e:
        logger.warning("Makro history sikertelen: %s", e)
        return pd.DataFrame()


# ============================================================================
# 2. ON-CHAIN — Hash rate, mempool, tx volume (Blockchain.com)
# ============================================================================

@dataclass
class OnChainSnapshot:
    hash_rate_ehs:    float = 0.0   # EH/s
    hash_rate_ret_7d: float = 0.0   # 7 napos változás
    mempool_size_mb:  float = 0.0   # mempool mérete MB-ban
    tx_count_1d:      int   = 0     # napi tranzakció szám
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def hash_rate_signal(self) -> int:
        """Bányász konfidencia proxy."""
        if self.hash_rate_ret_7d >= 0.05:  return  1
        if self.hash_rate_ret_7d <= -0.05: return -1
        return 0

    @property
    def mempool_signal(self) -> int:
        """Nagy mempool = hálózat terhelés = kereslet."""
        if self.mempool_size_mb >= 200: return  1
        if self.mempool_size_mb <= 10:  return -1
        return 0


def fetch_onchain() -> Optional[OnChainSnapshot]:
    """Hash rate és mempool adatok blockchain.com API-ból (ingyenes, kulcs nélkül)."""
    def _fetch():
        try:
            hr_data = _get_json(
                "https://api.blockchain.info/charts/hash-rate?timespan=30days"
                "&format=json&sampled=true&cors=true"
            )
            hr_vals = [v["y"] for v in hr_data.get("values", [])]
            hr_now  = hr_vals[-1] if hr_vals else 0.0
            hr_7d   = hr_vals[-7] if len(hr_vals) >= 7 else hr_now
            hr_ret  = (hr_now - hr_7d) / hr_7d if hr_7d > 0 else 0.0

            mp_data = _get_json(
                "https://api.blockchain.info/charts/mempool-size?timespan=7days"
                "&format=json&sampled=true&cors=true"
            )
            mp_mb = mp_data.get("values", [{}])[-1].get("y", 0.0) / 1e6

            tx_data = _get_json(
                "https://api.blockchain.info/charts/n-transactions?timespan=7days"
                "&format=json&sampled=true&cors=true"
            )
            tx_count = int(tx_data.get("values", [{}])[-1].get("y", 0))

            return OnChainSnapshot(
                hash_rate_ehs    = hr_now,
                hash_rate_ret_7d = hr_ret,
                mempool_size_mb  = mp_mb,
                tx_count_1d      = tx_count,
            )
        except Exception as e:
            logger.warning("On-chain adat sikertelen: %s", e)
            return OnChainSnapshot()

    return _cached("onchain", 3600, _fetch)


def fetch_onchain_history(metric: str = "hash-rate",
                          timespan: str = "1year") -> pd.DataFrame:
    """
    Historikus on-chain metrika — blockchain.com charts API.
    metric: "hash-rate", "mempool-size", "n-transactions", "miners-revenue"
    """
    try:
        data = _get_json(
            f"https://api.blockchain.info/charts/{metric}?timespan={timespan}"
            f"&format=json&sampled=false&cors=true"
        )
        rows = [{"timestamp": pd.to_datetime(v["x"], unit="s", utc=True),
                 "value":     float(v["y"])}
                for v in data.get("values", [])]
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df.columns = [metric.replace("-", "_")]
        return df
    except Exception as e:
        logger.warning("On-chain history (%s) sikertelen: %s", metric, e)
        return pd.DataFrame()


# ============================================================================
# 3. LIKVIDÁCIÓK — Coinglass (cross-exchange aggregált)
# ============================================================================

@dataclass
class LiquidationSnapshot:
    long_liq_usd_1h:  float = 0.0   # Long likvidációk USD-ben az elmúlt 1h
    short_liq_usd_1h: float = 0.0
    total_liq_usd_1h: float = 0.0
    liq_ratio:        float = 0.5   # long / total (>0.7 = long squeeze)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def signal(self) -> int:
        """
        Nagy long-likvidáció hullám után reversal várható (kimerül az eladói nyomás).
        Nagy short-likvidáció = felfelé squeeze → rövid távon bullish tovább.
        """
        if self.total_liq_usd_1h <= 0:
            return 0
        if self.liq_ratio >= 0.75 and self.total_liq_usd_1h >= 50e6:
            return 1    # nagy long-squeeze → reversal buy
        if self.liq_ratio <= 0.25 and self.total_liq_usd_1h >= 50e6:
            return -1   # nagy short-squeeze → reversal sell
        return 0


def fetch_liquidations(symbol: str = "BTC") -> Optional[LiquidationSnapshot]:
    """Coinglass aggregált likvidáció adatok."""
    def _fetch():
        try:
            url  = f"https://open-api.coinglass.com/public/v2/liquidation_history?symbol={symbol}&interval=1h"
            data = _get_json(url)
            rows = data.get("data", [])
            if not rows:
                return LiquidationSnapshot()
            latest     = rows[-1]
            long_liq   = float(latest.get("longLiquidationUsd", 0.0))
            short_liq  = float(latest.get("shortLiquidationUsd", 0.0))
            total      = long_liq + short_liq
            return LiquidationSnapshot(
                long_liq_usd_1h  = long_liq,
                short_liq_usd_1h = short_liq,
                total_liq_usd_1h = total,
                liq_ratio        = long_liq / total if total > 0 else 0.5,
            )
        except Exception as e:
            logger.warning("Likvidáció adat sikertelen: %s", e)
            return LiquidationSnapshot()

    return _cached(f"liq_{symbol}", 900, _fetch)


# ============================================================================
# 4. OPTIONS — Put/Call ratio, Implied Volatility (Deribit)
# ============================================================================

@dataclass
class OptionsSnapshot:
    put_call_ratio: float = 1.0    # >1.5 = félelem, <0.7 = eufória
    iv_atm_25d:     float = 0.0    # At-the-money 25 delta IV (éves %)
    iv_skew:        float = 0.0    # 25d put IV - 25d call IV (pozitív = félelem)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def put_call_signal(self) -> int:
        """Kontrarian: magas put/call = félelem = potenciális buy."""
        if self.put_call_ratio >= 1.5: return  1
        if self.put_call_ratio <= 0.6: return -1
        return 0

    @property
    def skew_signal(self) -> int:
        """Nagy pozitív skew = piac downside-ot áraz = bearish."""
        if self.iv_skew >= 10: return -1
        if self.iv_skew <= -5: return  1
        return 0


def fetch_options(currency: str = "BTC") -> Optional[OptionsSnapshot]:
    """Deribit opciós adatok — put/call ratio és IV."""
    def _fetch():
        try:
            # Opciók listája
            inst = _get_json(
                f"https://www.deribit.com/api/v2/public/get_instruments"
                f"?currency={currency}&kind=option&expired=false"
            )
            instruments = inst.get("result", [])

            # Összegzés: hány put vs call van nyitva
            total_put_oi = 0.0
            total_call_oi = 0.0
            iv_samples: list = []

            for inst_data in instruments[:200]:   # top 200 sztrájk
                name = inst_data.get("instrument_name", "")
                try:
                    ticker = _get_json(
                        f"https://www.deribit.com/api/v2/public/ticker"
                        f"?instrument_name={name}"
                    ).get("result", {})
                    oi  = float(ticker.get("open_interest", 0.0))
                    iv  = float(ticker.get("mark_iv", 0.0))
                    if "-P" in name:
                        total_put_oi += oi
                    else:
                        total_call_oi += oi
                    if iv > 0:
                        iv_samples.append(iv)
                except Exception:
                    continue

            pcr    = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
            iv_atm = float(sum(iv_samples) / len(iv_samples)) if iv_samples else 0.0

            return OptionsSnapshot(
                put_call_ratio = pcr,
                iv_atm_25d     = iv_atm,
                iv_skew        = 0.0,   # részletes skew számításhoz több adat kell
            )
        except Exception as e:
            logger.warning("Options adat sikertelen: %s", e)
            return OptionsSnapshot()

    return _cached(f"options_{currency}", 3600, _fetch)


# ============================================================================
# 5. HÍREK — CryptoPanic sentiment (opcionális API kulcs)
# ============================================================================

@dataclass
class NewsSentiment:
    bullish_count:  int   = 0
    bearish_count:  int   = 0
    sentiment_score: float = 0.0   # -1..+1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def signal(self) -> int:
        if self.sentiment_score >= 0.3:  return  1
        if self.sentiment_score <= -0.3: return -1
        return 0


def fetch_news_sentiment(currencies: str = "BTC",
                         api_key: Optional[str] = None) -> NewsSentiment:
    """CryptoPanic hír sentiment — API kulcs nélkül is működik (limit: 50 req/nap)."""
    def _fetch():
        try:
            base = "https://cryptopanic.com/api/v1/posts/"
            params = f"?currencies={currencies}&kind=news"
            if api_key:
                params += f"&auth_token={api_key}"
            data = _get_json(base + params)

            bullish = bearish = 0
            for post in data.get("results", []):
                votes = post.get("votes", {})
                bullish += int(votes.get("positive", 0))
                bearish += int(votes.get("negative", 0))

            total = bullish + bearish
            score = (bullish - bearish) / total if total > 0 else 0.0
            return NewsSentiment(
                bullish_count   = bullish,
                bearish_count   = bearish,
                sentiment_score = score,
            )
        except Exception as e:
            logger.warning("CryptoPanic sentiment sikertelen: %s", e)
            return NewsSentiment()

    return _cached(f"news_{currencies}", 1800, _fetch)


# ============================================================================
# Összesített snapshot — egy hívás az összes forráshoz
# ============================================================================

@dataclass
class CryptoDataSnapshot:
    macro:      Optional[MacroSnapshot]        = None
    onchain:    Optional[OnChainSnapshot]      = None
    liquidation: Optional[LiquidationSnapshot] = None
    options:    Optional[OptionsSnapshot]      = None
    news:       Optional[NewsSentiment]        = None

    def all_signals(self) -> Dict[str, int]:
        """Minden forrás jele egy dict-ben. Hiányzó forrás = 0."""
        s: Dict[str, int] = {}
        if self.macro:
            s["macro_vix"]  = self.macro.vix_signal
            s["macro_dxy"]  = self.macro.dxy_signal
        if self.onchain:
            s["onchain_hashrate"] = self.onchain.hash_rate_signal
            s["onchain_mempool"]  = self.onchain.mempool_signal
        if self.liquidation:
            s["liquidation"] = self.liquidation.signal
        if self.options:
            s["options_pcr"]  = self.options.put_call_signal
            s["options_skew"] = self.options.skew_signal
        if self.news:
            s["news_sentiment"] = self.news.signal
        return s

    def to_feature_dict(self) -> Dict[str, float]:
        """Nyers float értékek az ML feature matrix-hoz."""
        f: Dict[str, float] = {}
        if self.macro:
            f["sp500_ret_1d"]  = self.macro.sp500_ret_1d
            f["sp500_ret_7d"]  = self.macro.sp500_ret_7d
            f["dxy_level"]     = self.macro.dxy_level
            f["dxy_ret_5d"]    = self.macro.dxy_ret_5d
            f["vix_level"]     = self.macro.vix_level
            f["gold_ret_5d"]   = self.macro.gold_ret_5d
        if self.onchain:
            f["hash_rate_ehs"]    = self.onchain.hash_rate_ehs
            f["hash_rate_ret_7d"] = self.onchain.hash_rate_ret_7d
            f["mempool_size_mb"]  = self.onchain.mempool_size_mb
            f["tx_count_1d"]      = float(self.onchain.tx_count_1d)
        if self.liquidation:
            f["liq_total_usd_1h"] = self.liquidation.total_liq_usd_1h
            f["liq_ratio"]        = self.liquidation.liq_ratio
        if self.options:
            f["put_call_ratio"] = self.options.put_call_ratio
            f["iv_atm"]         = self.options.iv_atm_25d
            f["iv_skew"]        = self.options.iv_skew
        if self.news:
            f["news_sentiment"] = self.news.sentiment_score
        return f


def fetch_all(symbol: str = "BTC",
              news_api_key: Optional[str] = None) -> CryptoDataSnapshot:
    """Minden forrás párhuzamos lekérdezése — egyfajta 'adat aggregátor'."""
    import threading

    snap = CryptoDataSnapshot()
    errors: list = []

    def _run(name, fn):
        try:
            return fn()
        except Exception as e:
            errors.append(f"{name}: {e}")
            return None

    results = {}
    threads = {
        "macro":   threading.Thread(target=lambda: results.__setitem__("macro",   _run("macro",   fetch_macro))),
        "onchain": threading.Thread(target=lambda: results.__setitem__("onchain", _run("onchain", fetch_onchain))),
        "liq":     threading.Thread(target=lambda: results.__setitem__("liq",     _run("liq",     lambda: fetch_liquidations(symbol)))),
        "options": threading.Thread(target=lambda: results.__setitem__("options", _run("options", lambda: fetch_options(symbol)))),
        "news":    threading.Thread(target=lambda: results.__setitem__("news",    _run("news",    lambda: fetch_news_sentiment(symbol, news_api_key)))),
    }

    for t in threads.values():
        t.start()
    for t in threads.values():
        t.join(timeout=10)

    snap.macro       = results.get("macro")
    snap.onchain     = results.get("onchain")
    snap.liquidation = results.get("liq")
    snap.options     = results.get("options")
    snap.news        = results.get("news")

    if errors:
        logger.warning("Részleges hibák: %s", "; ".join(errors))

    return snap
