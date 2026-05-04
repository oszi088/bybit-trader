"""
Feature matrix az ML modellhez.

A meglévő indicators.compute_all() kimenetéből indul ki:
  * nyers indikátor értékek (nem ±1 jelek — az ML maga tanulja a küszöböt)
  * normalizált pozíció-metrikák (pl. BB-n belüli helyezet)
  * rolling stat feature-ök (mean/std több ablakon)
  * lag feature-ök (visszatekintő hozamok)
  * cross-feature-ök (RSI × volume ratio, stb.)

Az összes feature-t walk-forward safe módon számítjuk:
  csak múltbeli adatot használunk, nincs lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import IndicatorParams
from indicators import compute_all


# Nyers indikátor oszlopok (ha léteznek az enriched DataFrame-ben)
_RAW_COLS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "cci",
    "atr", "adx", "plus_di", "minus_di",
    "obv", "mfi",
    "sma_fast", "sma_slow", "sma_long",
    "ema_fast", "ema_slow",
    "bb_upper", "bb_mid", "bb_lower",
    "vwap",
]

_ROLLING_WINDOWS = [5, 10, 20]
_LAG_PERIODS     = [1, 3, 5]


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    params: IndicatorParams,
) -> pd.DataFrame:
    """
    Teljes feature matrix — egy sor = egy gyertya.

    NaN-t forward-fill + 0 zárja le (hogy az XGBoost ne kapjon NaN-t).
    """
    enriched = compute_all(ohlcv, params)
    feats: dict[str, pd.Series] = {}

    # --- Nyers indikátor értékek ---
    for col in _RAW_COLS:
        if col in enriched.columns:
            feats[col] = enriched[col]

    # --- Normalizált ár-pozíció a Bollinger-sávon belül (0..1) ---
    if {"bb_upper", "bb_lower", "bb_mid"}.issubset(enriched.columns):
        bb_range = (enriched["bb_upper"] - enriched["bb_lower"]).replace(0, np.nan)
        feats["bb_pct"] = (ohlcv["close"] - enriched["bb_lower"]) / bb_range

    # --- SMA- és EMA-távolság az ártól (relatív) ---
    for ma in ("sma_fast", "sma_slow", "ema_fast", "ema_slow", "vwap"):
        if ma in enriched.columns:
            feats[f"{ma}_dist"] = (ohlcv["close"] - enriched[ma]) / ohlcv["close"]

    # --- ATR / ár (relatív volatilitás) ---
    if "atr" in enriched.columns:
        feats["atr_pct"] = enriched["atr"] / ohlcv["close"]

    # --- Volume ratio (aktuális / rolling 20 átlag) ---
    vol_mean = ohlcv["volume"].rolling(20).mean().replace(0, np.nan)
    feats["volume_ratio"] = ohlcv["volume"] / vol_mean

    # --- Log-hozam lag-ok ---
    log_ret = np.log(ohlcv["close"]).diff()
    for lag in _LAG_PERIODS:
        feats[f"ret_lag_{lag}"] = log_ret.shift(lag)

    # --- RSI delta (momentum változás üteme) ---
    if "rsi" in enriched.columns:
        for w in (3, 5, 10):
            feats[f"rsi_delta_{w}"] = enriched["rsi"].diff(w)

    # --- Rolling mean + std az RSI-re és MACD hist-re ---
    for col in ("rsi", "macd_hist"):
        if col in enriched.columns:
            for w in _ROLLING_WINDOWS:
                feats[f"{col}_rmean_{w}"] = enriched[col].rolling(w).mean()
                feats[f"{col}_rstd_{w}"]  = enriched[col].rolling(w).std()

    # --- Cross-feature: RSI × volume_ratio (oversold + erős volumen) ---
    if "rsi" in enriched.columns:
        feats["rsi_x_vol"] = enriched["rsi"] * feats.get("volume_ratio", pd.Series(1.0, index=ohlcv.index))

    # --- ADX irányosság (+DI - -DI normalizálva) ---
    if {"plus_di", "minus_di", "adx"}.issubset(enriched.columns):
        feats["di_diff_norm"] = (enriched["plus_di"] - enriched["minus_di"]) / (enriched["adx"] + 1e-9)

    df = pd.DataFrame(feats, index=ohlcv.index)
    df = df.ffill().fillna(0.0)
    return df
