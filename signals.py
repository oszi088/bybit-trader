"""
Az indikátor értékeket {-1, 0, +1} szignálra fordítjuk:
    +1 = vételi nyomás (bullish)
     0 = semleges
    -1 = eladási nyomás (bearish)

Minden függvény az indikátorokkal kibővített DataFrame UTOLSÓ sorát kapja
(egy pandas.Series), és így jelzi a 'most' aktuális véleményét.
"""

from __future__ import annotations

import pandas as pd

from config import IndicatorParams


# --------------------------------------------------------------------------- #
# Trend szignálok
# --------------------------------------------------------------------------- #

def signal_sma_cross(row: pd.Series) -> int:
    """Gyors SMA a lassú felett -> bullish, alatta -> bearish."""
    if pd.isna(row["sma_fast"]) or pd.isna(row["sma_slow"]):
        return 0
    if row["sma_fast"] > row["sma_slow"]:
        return 1
    if row["sma_fast"] < row["sma_slow"]:
        return -1
    return 0


def signal_ema_cross(row: pd.Series) -> int:
    """Ugyanaz, mint az SMA cross, csak EMA-ra (gyorsabban reagál)."""
    if pd.isna(row["ema_fast"]) or pd.isna(row["ema_slow"]):
        return 0
    if row["ema_fast"] > row["ema_slow"]:
        return 1
    if row["ema_fast"] < row["ema_slow"]:
        return -1
    return 0


def signal_macd(row: pd.Series) -> int:
    """A hisztogram előjele alapján döntünk: pozitív -> buy, negatív -> sell."""
    h = row["macd_hist"]
    if pd.isna(h):
        return 0
    if h > 0:
        return 1
    if h < 0:
        return -1
    return 0


def signal_adx(row: pd.Series) -> int:
    """
    Az ADX maga a trend erőssége (irány nélkül). Csak akkor adunk irányt,
    ha az ADX > 20 (van trend), és a +DI / -DI viszonya egyértelmű.
    """
    if pd.isna(row["adx"]) or pd.isna(row["plus_di"]) or pd.isna(row["minus_di"]):
        return 0
    if row["adx"] < 20:
        return 0  # nincs erős trend
    if row["plus_di"] > row["minus_di"]:
        return 1
    if row["minus_di"] > row["plus_di"]:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# Momentum szignálok
# --------------------------------------------------------------------------- #

def signal_rsi(row: pd.Series, params: IndicatorParams) -> int:
    """RSI < oversold -> buy, RSI > overbought -> sell."""
    r = row["rsi"]
    if pd.isna(r):
        return 0
    if r < params.rsi_oversold:
        return 1
    if r > params.rsi_overbought:
        return -1
    return 0


def signal_stochastic(row: pd.Series, params: IndicatorParams) -> int:
    """
    %K < oversold és %K > %D -> buy (oversold + felfelé fordul);
    %K > overbought és %K < %D -> sell (overbought + lefelé fordul).
    """
    k, d = row["stoch_k"], row["stoch_d"]
    if pd.isna(k) or pd.isna(d):
        return 0
    if k < params.stoch_oversold and k > d:
        return 1
    if k > params.stoch_overbought and k < d:
        return -1
    return 0


def signal_cci(row: pd.Series) -> int:
    """CCI < -100 -> buy (oversold), CCI > +100 -> sell (overbought)."""
    c = row["cci"]
    if pd.isna(c):
        return 0
    if c < -100:
        return 1
    if c > 100:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# Volatilitás szignálok
# --------------------------------------------------------------------------- #

def signal_bollinger(row: pd.Series) -> int:
    """
    Mean reversion: az ár az alsó sáv alatt -> buy, a felső sáv felett -> sell.
    """
    price = row["close"]
    if pd.isna(row["bb_upper"]) or pd.isna(row["bb_lower"]):
        return 0
    if price < row["bb_lower"]:
        return 1
    if price > row["bb_upper"]:
        return -1
    return 0


def signal_atr(row: pd.Series) -> int:
    """
    Az ATR-t nem irány-szignálnak, hanem volatilitás-szűrőnek használjuk.
    Itt 0-t adunk vissza; az ATR értékét a TradingAgent külön olvassa ki
    a pozícióméretezéshez.
    """
    return 0


# --------------------------------------------------------------------------- #
# Volumen szignálok
# --------------------------------------------------------------------------- #

def signal_obv(row: pd.Series, prev_obv: float | None) -> int:
    """
    Az OBV rövid távú változásának előjele alapján: emelkedik -> buy,
    csökken -> sell.
    """
    if prev_obv is None or pd.isna(row["obv"]):
        return 0
    if row["obv"] > prev_obv:
        return 1
    if row["obv"] < prev_obv:
        return -1
    return 0


def signal_vwap(row: pd.Series) -> int:
    """Ár a VWAP felett -> buy bias; alatta -> sell bias."""
    if pd.isna(row["vwap"]):
        return 0
    if row["close"] > row["vwap"]:
        return 1
    if row["close"] < row["vwap"]:
        return -1
    return 0


def signal_mfi(row: pd.Series) -> int:
    """MFI < 20 -> buy (oversold + alacsony pénzbeáramlás), > 80 -> sell."""
    m = row["mfi"]
    if pd.isna(m):
        return 0
    if m < 20:
        return 1
    if m > 80:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# Makro hangulat (Fear & Greed Index)
# --------------------------------------------------------------------------- #

def signal_fear_greed(value: int) -> int:
    """
    Crypto Fear & Greed kontrarian szignal:
      <= 24 (Extreme Fear)   -> +1 (vasarlasi lehetoseg)
      >= 75 (Extreme Greed)  -> -1 (eladasra utal)
      koztes ertek           -> 0
    """
    if value <= 24:
        return 1
    if value >= 75:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# Hosszu tavu trend (50/200 SMA cross)
# --------------------------------------------------------------------------- #

def signal_golden_death(row: pd.Series) -> int:
    """
    Esemeny-alapu szignal: az utolso `cross_lookback` gyertyan
      * golden cross (SMA50 atkereszti felfele SMA200-at)  -> +1
      * death cross  (SMA50 atkereszti lefele SMA200-at)   -> -1
      * egyebkent                                          ->  0
    A "recent_golden" / "recent_death" oszlopokat az indicators.compute_all
    elore kiszamolja egy lookback-rolling max-szal.
    """
    g = int(row.get("recent_golden", 0) or 0)
    d = int(row.get("recent_death", 0) or 0)
    if g and not d:
        return 1
    if d and not g:
        return -1
    return 0


def signal_long_trend(row: pd.Series) -> int:
    """
    Allapot-alapu szignal: az SMA50 az SMA200 felett (bullish hosszu trend)
    vagy alatta (bearish). NaN eseten 0.
    """
    s = row.get("sma_slow")
    l = row.get("sma_long")
    if s is None or l is None or pd.isna(s) or pd.isna(l):
        return 0
    if s > l:
        return 1
    if s < l:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# Orderflow / mikrostruktura szignalok
# --------------------------------------------------------------------------- #

def signal_ob_imbalance(row: pd.Series) -> int:
    """
    Order Book Imbalance (OBI) alapu szignal.
    Backtestben OHLCV-proxyt hasznal (estimate_ob_imbalance_from_ohlcv),
    elo kereskedésben a valodi L2 orderbook erteket.

    Kuszob: |OBI| > 0.20 → iranyt ad; kisebbnel semleges.
    Indoklas: az OBI zajos proxy, csak egyertelmu imbalancet ertekeljuk.
    """
    v = row.get("ob_imbalance", 0.0)
    if v is None or pd.isna(v):
        return 0
    if v > 0.20:
        return 1
    if v < -0.20:
        return -1
    return 0


def signal_ob_large_order(row: pd.Series) -> int:
    """
    Nagy limit megbizas (intezmenyi fal) szignalra forditva.
    Ertekek: +1 (veteli fal), 0 (semleges), -1 (eladasi fal).
    Backtestben mindig 0 (nincs historikus OB), elo kereskedésben
    az OrderBookFetcher.feature_dict() toltí ki.
    """
    v = row.get("ob_large_order", 0)
    if v is None or pd.isna(v):
        return 0
    return int(v)
