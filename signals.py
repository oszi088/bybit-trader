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


# --------------------------------------------------------------------------- #
# Vektorizált batch szignál-számítás (VectorizedBacktester / nagy adathalmaz)
# --------------------------------------------------------------------------- #

import numpy as np  # noqa: E402  (a fájl tetején is be van importálva a pd, de np nem)


def compute_signal_matrix(
    enriched: pd.DataFrame,
    params: "IndicatorParams",
    fg_value: int = 50,
) -> pd.DataFrame:
    """
    Teljes szignálmátrix egy numpy pass-ban: ~1000× gyorsabb mint
    soronkénti signal_xxx() hívás.

    Paraméterek:
        enriched : compute_all() által kiegészített OHLCV DataFrame
        params   : IndicatorParams (rsi_oversold, stoch_oversold, stb.)
        fg_value : Fear & Greed skalar (egész futásra egy érték)

    Visszatér:
        DataFrame, shape=(len(enriched), 17), dtype=int8, értékek {-1, 0, 1}
        Oszlopok: sma_cross, ema_cross, macd, adx, rsi, stochastic, cci,
                  bollinger, atr, obv, vwap, mfi, fear_greed,
                  golden_death, long_trend, ob_imbalance, ob_large_order
    """
    idx = enriched.index
    n = len(enriched)

    def _sign3(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        """Bool → int8 {+1, 0, -1}"""
        return np.where(pos, np.int8(1), np.where(neg, np.int8(-1), np.int8(0))).astype(np.int8)

    def _col(name, default=np.nan):
        return enriched[name].values if name in enriched.columns else np.full(n, default)

    close = _col("close")

    # ── Trend ────────────────────────────────────────────────────────────── #
    sma_f = _col("sma_fast"); sma_s = _col("sma_slow")
    ok = ~np.isnan(sma_f) & ~np.isnan(sma_s)
    sma_cross = np.where(ok, _sign3(sma_f > sma_s, sma_f < sma_s), np.int8(0)).astype(np.int8)

    ema_f = _col("ema_fast"); ema_s = _col("ema_slow")
    ok = ~np.isnan(ema_f) & ~np.isnan(ema_s)
    ema_cross = np.where(ok, _sign3(ema_f > ema_s, ema_f < ema_s), np.int8(0)).astype(np.int8)

    hist = _col("macd_hist")
    macd_sig = np.where(~np.isnan(hist), _sign3(hist > 0, hist < 0), np.int8(0)).astype(np.int8)

    adx_v = _col("adx"); plus_di = _col("plus_di"); minus_di = _col("minus_di")
    ok = ~np.isnan(adx_v) & ~np.isnan(plus_di) & ~np.isnan(minus_di)
    strong = adx_v >= 20
    adx_sig = np.where(ok, _sign3(strong & (plus_di > minus_di),
                                   strong & (minus_di > plus_di)), np.int8(0)).astype(np.int8)

    # ── Momentum ──────────────────────────────────────────────────────────── #
    rsi_v = _col("rsi")
    rsi_sig = np.where(~np.isnan(rsi_v),
                       _sign3(rsi_v < params.rsi_oversold, rsi_v > params.rsi_overbought),
                       np.int8(0)).astype(np.int8)

    k = _col("stoch_k"); d = _col("stoch_d")
    ok = ~np.isnan(k) & ~np.isnan(d)
    stoch_sig = np.where(ok,
        _sign3((k < params.stoch_oversold) & (k > d),
               (k > params.stoch_overbought) & (k < d)),
        np.int8(0)).astype(np.int8)

    cci_v = _col("cci")
    cci_sig = np.where(~np.isnan(cci_v),
                       _sign3(cci_v < -100, cci_v > 100),
                       np.int8(0)).astype(np.int8)

    # ── Volatilitás ───────────────────────────────────────────────────────── #
    bb_u = _col("bb_upper"); bb_l = _col("bb_lower")
    ok = ~np.isnan(bb_u) & ~np.isnan(bb_l)
    boll_sig = np.where(ok, _sign3(close < bb_l, close > bb_u), np.int8(0)).astype(np.int8)

    atr_sig = np.zeros(n, dtype=np.int8)   # ATR mindig 0 (volatilitás-szűrő, nem irány)

    # ── Volumen ───────────────────────────────────────────────────────────── #
    obv_v = _col("obv", 0.0)
    prev_obv = np.empty(n); prev_obv[0] = np.nan; prev_obv[1:] = obv_v[:-1]
    ok = ~np.isnan(obv_v) & ~np.isnan(prev_obv)
    obv_sig = np.where(ok, _sign3(obv_v > prev_obv, obv_v < prev_obv), np.int8(0)).astype(np.int8)

    vwap_v = _col("vwap")
    vwap_sig = np.where(~np.isnan(vwap_v),
                        _sign3(close > vwap_v, close < vwap_v),
                        np.int8(0)).astype(np.int8)

    mfi_v = _col("mfi")
    mfi_sig = np.where(~np.isnan(mfi_v),
                       _sign3(mfi_v < 20, mfi_v > 80),
                       np.int8(0)).astype(np.int8)

    # ── Makro / hosszú táv ────────────────────────────────────────────────── #
    fg_scalar = signal_fear_greed(fg_value)
    fg_col = np.full(n, fg_scalar, dtype=np.int8)

    g_raw = _col("recent_golden", 0.0); d_raw = _col("recent_death", 0.0)
    g = np.nan_to_num(g_raw).astype(int); d_c = np.nan_to_num(d_raw).astype(int)
    gd_sig = _sign3((g > 0) & (d_c == 0), (d_c > 0) & (g == 0))

    sma_s2 = _col("sma_slow"); sma_l = _col("sma_long")
    ok = ~np.isnan(sma_s2) & ~np.isnan(sma_l)
    lt_sig = np.where(ok, _sign3(sma_s2 > sma_l, sma_s2 < sma_l), np.int8(0)).astype(np.int8)

    # ── Orderflow ─────────────────────────────────────────────────────────── #
    obi = np.nan_to_num(_col("ob_imbalance", 0.0))
    obi_sig = _sign3(obi > 0.20, obi < -0.20)

    olo = np.nan_to_num(_col("ob_large_order", 0.0)).astype(int)
    olo_sig = np.clip(olo, -1, 1).astype(np.int8)

    return pd.DataFrame({
        "sma_cross":    sma_cross,  "ema_cross":  ema_cross,
        "macd":         macd_sig,   "adx":        adx_sig,
        "rsi":          rsi_sig,    "stochastic": stoch_sig,
        "cci":          cci_sig,    "bollinger":  boll_sig,
        "atr":          atr_sig,    "obv":        obv_sig,
        "vwap":         vwap_sig,   "mfi":        mfi_sig,
        "fear_greed":   fg_col,     "golden_death": gd_sig,
        "long_trend":   lt_sig,     "ob_imbalance": obi_sig,
        "ob_large_order": olo_sig,
    }, index=idx, dtype=np.int8)


def compute_scores_with_regime(
    signal_matrix: pd.DataFrame,
    enriched: pd.DataFrame,
    regime_cfg: "RegimeConfig",
    weights_default: dict,
    weights_trend: dict,
    weights_range: dict,
) -> np.ndarray:
    """
    Per-bar súlyozott score vektorizáltan, regime-adaptív súlyokkal.

    Az ADX értéke alapján minden bárhoz a megfelelő súlyvektort választja:
      ADX ≥ adx_trend_threshold → TREND_WEIGHTS
      ADX ≤ adx_range_threshold → RANGE_WEIGHTS
      köztes / NaN             → DEFAULT_WEIGHTS

    Returns: float64 numpy array, len=len(signal_matrix), értékek ∈ [-1, 1]
    """
    cols = list(signal_matrix.columns)
    sig_arr = signal_matrix.values.astype(np.float64)   # (n, 17)

    def _wvec(w: dict) -> np.ndarray:
        return np.array([w.get(c, 0.0) for c in cols], dtype=np.float64)

    w_def   = _wvec(weights_default)
    w_trend = _wvec(weights_trend)
    w_range = _wvec(weights_range)

    norm_def   = max(float(np.sum(np.abs(w_def))),   1e-9)
    norm_trend = max(float(np.sum(np.abs(w_trend))), 1e-9)
    norm_range = max(float(np.sum(np.abs(w_range))), 1e-9)

    s_def   = sig_arr @ w_def   / norm_def    # (n,)
    s_trend = sig_arr @ w_trend / norm_trend
    s_range = sig_arr @ w_range / norm_range

    if not regime_cfg.enabled:
        return s_def

    adx_v    = enriched["adx"].values if "adx" in enriched.columns else np.full(len(signal_matrix), np.nan)
    is_na    = np.isnan(adx_v)
    is_trend = (~is_na) & (adx_v >= regime_cfg.adx_trend_threshold)
    is_range = (~is_na) & (adx_v <= regime_cfg.adx_range_threshold)

    return np.where(is_na | (~is_trend & ~is_range), s_def,
           np.where(is_trend, s_trend, s_range))
