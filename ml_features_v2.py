"""
130+ feature ML matrix — intézményi szintű feature engineering.

Rétegek:
  1.  Nyers indikátor értékek           (~20 feature)
  2.  Normalizált pozíció metrikák      (~10 feature)
  3.  Rolling stat-ok                   (~20 feature)
  4.  Lag + momentum feature-ök         (~15 feature)
  5.  Cross-feature interakciók         (~10 feature)
  6.  Volatilitás rezsim feature-ök     (~10 feature)
  7.  Orderbook proxy (OHLCV-ből)       (~5  feature)
  8.  Funding rate feature-ök           (~5  feature)
  9.  Open interest feature-ök          (~5  feature)
  10. Cross-timeframe feature-ök        (~10 feature)
  11. Makro feature-ök (SP500/DXY/VIX)  (~6  feature)
  12. On-chain feature-ök               (~6  feature)
  13. Likvidáció feature-ök             (~4  feature)
  14. Options feature-ök                (~4  feature)
  15. Hír sentiment feature-ök          (~2  feature)

Összesen: ~132 feature
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import IndicatorParams
from crypto_data import CryptoDataSnapshot
from indicators import compute_all
from orderbook_features import estimate_ob_imbalance_from_ohlcv

logger = logging.getLogger("ml_features_v2")

_ROLLING_WINDOWS = [5, 10, 20, 50]
_LAG_PERIODS     = [1, 2, 3, 5, 8, 13]   # Fibonacci lag-ok


def build_feature_matrix_v2(
    ohlcv: pd.DataFrame,
    params: IndicatorParams,
    funding_df:      Optional[pd.DataFrame]      = None,   # index=timestamp, col="rate"
    oi_df:           Optional[pd.DataFrame]      = None,   # index=timestamp, col="oi"
    higher_tf_ohlcv: Optional[dict]              = None,   # {"4h": df, "1d": df}
    crypto_snap:     Optional[CryptoDataSnapshot] = None,  # élő crypto adat
    macro_history:   Optional[pd.DataFrame]      = None,   # historikus makro (tanításhoz)
    onchain_history: Optional[Dict[str, pd.DataFrame]] = None,  # historikus on-chain
) -> pd.DataFrame:
    """
    Teljes feature matrix — 100+ feature.

    Paraméterek:
        ohlcv          -- alap OHLCV (pl. 1h)
        params         -- IndicatorParams
        funding_df     -- historikus funding rate (opcionális, backtesthez)
        oi_df          -- historikus open interest (opcionális)
        higher_tf_ohlcv -- magasabb timeframe OHLCV dict-ek (opcionális)
    """
    enriched = compute_all(ohlcv, params)
    feats    = pd.DataFrame(index=ohlcv.index)

    # ================================================================== #
    # 1. NYERS INDIKÁTOR ÉRTÉKEK
    # ================================================================== #
    raw_cols = [
        "rsi", "macd", "macd_signal", "macd_hist",
        "stoch_k", "stoch_d", "cci",
        "atr", "adx", "plus_di", "minus_di",
        "obv", "mfi",
        "sma_fast", "sma_slow", "sma_long",
        "ema_fast", "ema_slow",
        "bb_upper", "bb_middle", "bb_lower",
        "vwap",
    ]
    for col in raw_cols:
        if col in enriched.columns:
            feats[col] = enriched[col]

    # ================================================================== #
    # 2. NORMALIZÁLT POZÍCIÓ METRIKÁK
    # ================================================================== #

    # Bollinger %B: hol van az ár a sávon belül (0..1, >1 kitörés fölé, <0 alá)
    if {"bb_upper", "bb_lower"}.issubset(enriched.columns):
        bb_range = (enriched["bb_upper"] - enriched["bb_lower"]).replace(0, np.nan)
        feats["bb_pct_b"]    = (ohlcv["close"] - enriched["bb_lower"]) / bb_range
        feats["bb_width_pct"] = bb_range / ohlcv["close"]   # sáv szélessége relatívan

    # Ár távolsága minden MA-tól (relatív %)
    for ma in ("sma_fast", "sma_slow", "sma_long", "ema_fast", "ema_slow", "vwap"):
        if ma in enriched.columns:
            feats[f"{ma}_dist"] = (ohlcv["close"] - enriched[ma]) / ohlcv["close"]

    # ATR relatív (volatilitás szint)
    if "atr" in enriched.columns:
        feats["atr_pct"] = enriched["atr"] / ohlcv["close"]

    # Stochastic spread (%K - %D)
    if {"stoch_k", "stoch_d"}.issubset(enriched.columns):
        feats["stoch_kd_diff"] = enriched["stoch_k"] - enriched["stoch_d"]

    # MACD histogram momentum (változás üteme)
    if "macd_hist" in enriched.columns:
        feats["macd_hist_delta"] = enriched["macd_hist"].diff(1)
        feats["macd_hist_accel"] = feats["macd_hist_delta"].diff(1)

    # ADX irányossági index (normalizált)
    if {"plus_di", "minus_di", "adx"}.issubset(enriched.columns):
        feats["di_diff_norm"] = (
            (enriched["plus_di"] - enriched["minus_di"])
            / (enriched["adx"] + 1e-9)
        )

    # ================================================================== #
    # 3. ROLLING STAT-OK
    # ================================================================== #

    for col in ("rsi", "macd_hist", "cci", "mfi"):
        if col in enriched.columns:
            for w in _ROLLING_WINDOWS:
                feats[f"{col}_rmean_{w}"] = enriched[col].rolling(w).mean()
                feats[f"{col}_rstd_{w}"]  = enriched[col].rolling(w).std()
                feats[f"{col}_rmin_{w}"]  = enriched[col].rolling(w).min()
                feats[f"{col}_rmax_{w}"]  = enriched[col].rolling(w).max()

    # Volatilitás rolling stat (ATR-ból)
    if "atr" in enriched.columns:
        for w in (10, 20, 50):
            feats[f"atr_pct_rmean_{w}"] = feats.get(
                "atr_pct", enriched["atr"] / ohlcv["close"]
            ).rolling(w).mean()

    # ================================================================== #
    # 4. LAG + MOMENTUM FEATURE-ÖK
    # ================================================================== #

    log_ret = np.log(ohlcv["close"]).diff()
    for lag in _LAG_PERIODS:
        feats[f"ret_lag_{lag}"] = log_ret.shift(lag)

    # Kumulált hozam különböző ablakokban
    for w in (3, 5, 10, 20):
        feats[f"cum_ret_{w}"] = np.exp(log_ret.rolling(w).sum()) - 1

    # RSI momentum (változás üteme)
    if "rsi" in enriched.columns:
        for w in (3, 5, 10):
            feats[f"rsi_delta_{w}"] = enriched["rsi"].diff(w)

    # OBV momentum
    if "obv" in enriched.columns:
        feats["obv_delta_5"]  = enriched["obv"].diff(5)
        feats["obv_delta_20"] = enriched["obv"].diff(20)
        obv_std = enriched["obv"].rolling(20).std().replace(0, np.nan)
        feats["obv_zscore"]   = (
            (enriched["obv"] - enriched["obv"].rolling(20).mean()) / obv_std
        )

    # ================================================================== #
    # 5. CROSS-FEATURE INTERAKCIÓK
    # ================================================================== #

    # Volume ratio (aktuális / 20 gyertyás átlag)
    vol_mean = ohlcv["volume"].rolling(20).mean().replace(0, np.nan)
    feats["volume_ratio"] = ohlcv["volume"] / vol_mean

    # Volume surge: nagy volumen + irányos mozgás
    feats["volume_x_ret"] = feats["volume_ratio"] * log_ret

    # RSI × volume (oversold + erős volumen = erős reversal jel)
    if "rsi" in enriched.columns:
        feats["rsi_x_vol"] = enriched["rsi"] * feats["volume_ratio"]

    # ADX × MACD hist (trend erőssége × irány)
    if {"adx", "macd_hist"}.issubset(enriched.columns):
        feats["adx_x_macd"] = enriched["adx"] * np.sign(enriched["macd_hist"])

    # BB szélességének változása (volatilitás felépülés/lecsengés)
    if "bb_width_pct" in feats.columns:
        feats["bb_width_delta"] = feats["bb_width_pct"].diff(5)

    # Gyertya test mérete (body / range): 1 = teljes mértékű irányos mozgás
    hl_range = (ohlcv["high"] - ohlcv["low"]).replace(0, np.nan)
    feats["candle_body_pct"] = abs(ohlcv["close"] - ohlcv["open"]) / hl_range
    feats["candle_direction"] = np.sign(ohlcv["close"] - ohlcv["open"])

    # Upper / lower wick arány
    feats["upper_wick_pct"] = (
        (ohlcv["high"] - ohlcv[["close", "open"]].max(axis=1)) / hl_range
    )
    feats["lower_wick_pct"] = (
        (ohlcv[["close", "open"]].min(axis=1) - ohlcv["low"]) / hl_range
    )

    # ================================================================== #
    # 6. VOLATILITÁS REZSIM FEATURE-ÖK
    # ================================================================== #

    # Realized volatility (log-hozamok szórása)
    for w in (5, 10, 20):
        feats[f"realized_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252 * 24)

    # Volatilitás arány: rövid / hosszú (volatilitás rezsim jelzője)
    if "realized_vol_5" in feats.columns and "realized_vol_20" in feats.columns:
        feats["vol_ratio_5_20"] = (
            feats["realized_vol_5"] / feats["realized_vol_20"].replace(0, np.nan)
        )

    # Parkinson volatilitás (high-low alapú, pontosabb mint close-close)
    feats["parkinson_vol"] = (
        np.sqrt(1 / (4 * np.log(2)))
        * np.log(ohlcv["high"] / ohlcv["low"].replace(0, np.nan))
    )

    # ================================================================== #
    # 7. ORDERBOOK PROXY (OHLCV-ALAPÚ)
    # ================================================================== #

    feats["ob_imbalance_proxy"] = estimate_ob_imbalance_from_ohlcv(ohlcv)
    for w in (5, 10):
        feats[f"ob_imbalance_proxy_ma_{w}"] = (
            feats["ob_imbalance_proxy"].rolling(w).mean()
        )

    # ================================================================== #
    # 8. FUNDING RATE FEATURE-ÖK (ha van historikus adat)
    # ================================================================== #

    if funding_df is not None and not funding_df.empty and "rate" in funding_df.columns:
        # Reindex az OHLCV index-re (forward-fill — a funding 8 óránként változik)
        fr = funding_df["rate"].reindex(ohlcv.index, method="ffill")
        feats["funding_rate"]         = fr
        feats["funding_rate_ma_3"]    = fr.rolling(3).mean()
        feats["funding_rate_extreme"] = (fr.abs() >= 0.001).astype(int)
        feats["funding_rate_delta"]   = fr.diff(1)
        # Annualizált funding: extrém értéknél kontrarian jel
        feats["funding_annualized"]   = fr * 3 * 365

    # ================================================================== #
    # 9. OPEN INTEREST FEATURE-ÖK (ha van historikus adat)
    # ================================================================== #

    if oi_df is not None and not oi_df.empty and "oi" in oi_df.columns:
        oi = oi_df["oi"].reindex(ohlcv.index, method="ffill")
        oi_norm = oi / oi.rolling(20).mean().replace(0, np.nan)
        feats["oi_norm"]       = oi_norm        # aktuális OI / 20-gyertyás átlag
        feats["oi_delta_pct"]  = oi.pct_change(3)
        # OI + ár irány kombinációja
        feats["oi_price_conf"] = feats["oi_delta_pct"] * np.sign(log_ret)

    # ================================================================== #
    # 10. CROSS-TIMEFRAME FEATURE-ÖK (ha van magasabb TF adat)
    # ================================================================== #

    if higher_tf_ohlcv:
        for tf_name, tf_df in higher_tf_ohlcv.items():
            if tf_df is None or tf_df.empty:
                continue
            try:
                tf_enr  = compute_all(tf_df, params)
                tf_rsi  = tf_enr["rsi"].reindex(ohlcv.index, method="ffill")
                tf_macd = tf_enr["macd_hist"].reindex(ohlcv.index, method="ffill")
                tf_atr  = tf_enr["atr"].reindex(ohlcv.index, method="ffill")
                feats[f"rsi_{tf_name}"]      = tf_rsi
                feats[f"macd_hist_{tf_name}"] = tf_macd
                feats[f"atr_pct_{tf_name}"]  = tf_atr / tf_df["close"].reindex(
                    ohlcv.index, method="ffill"
                )
                # Cross-TF RSI divergencia (alap vs magasabb TF)
                if "rsi" in enriched.columns:
                    feats[f"rsi_div_{tf_name}"] = enriched["rsi"] - tf_rsi
            except Exception as e:
                logger.warning("Cross-TF feature hiba (%s): %s", tf_name, e)

    # ================================================================== #
    # 11. MAKRO FEATURE-ÖK
    # ================================================================== #

    if macro_history is not None and not macro_history.empty:
        # Historikus adat (tanításhoz): daily → reindex OHLCV-re forward-fill-lel
        mh = macro_history.reindex(ohlcv.index, method="ffill")
        for col in mh.columns:
            feats[f"macro_{col}"] = mh[col]
        # Levezetett feature-ök
        if "sp500" in mh.columns:
            feats["macro_sp500_ret_5d"] = mh["sp500"].pct_change(5)
            feats["macro_sp500_ret_20d"] = mh["sp500"].pct_change(20)
        if "dxy" in mh.columns:
            feats["macro_dxy_ret_5d"] = mh["dxy"].pct_change(5)
        if "vix" in mh.columns:
            feats["macro_vix_high"]   = (mh["vix"] >= 30).astype(float)
            feats["macro_vix_extreme"] = (mh["vix"] >= 40).astype(float)
    elif crypto_snap is not None and crypto_snap.macro is not None:
        # Élő adat (paper/live módban): egyetlen érték broadcast az egész indexre
        for k, v in crypto_snap.macro.__dict__.items():
            if isinstance(v, float):
                feats[f"macro_{k}"] = float(v)

    # ================================================================== #
    # 12. ON-CHAIN FEATURE-ÖK
    # ================================================================== #

    if onchain_history is not None:
        for metric_name, metric_df in onchain_history.items():
            if metric_df is None or metric_df.empty:
                continue
            col = metric_df.columns[0]
            s   = metric_df[col].reindex(ohlcv.index, method="ffill")
            feats[f"onchain_{metric_name}"]          = s
            feats[f"onchain_{metric_name}_ret_7d"]   = s.pct_change(7)
            feats[f"onchain_{metric_name}_zscore_30"] = (
                (s - s.rolling(30).mean()) / s.rolling(30).std().replace(0, 1)
            )
    elif crypto_snap is not None and crypto_snap.onchain is not None:
        oc = crypto_snap.onchain
        feats["onchain_hash_rate"]     = oc.hash_rate_ehs
        feats["onchain_hash_ret_7d"]   = oc.hash_rate_ret_7d
        feats["onchain_mempool_mb"]    = oc.mempool_size_mb
        feats["onchain_tx_count"]      = float(oc.tx_count_1d)

    # ================================================================== #
    # 13. LIKVIDÁCIÓ FEATURE-ÖK
    # ================================================================== #

    if crypto_snap is not None and crypto_snap.liquidation is not None:
        liq = crypto_snap.liquidation
        feats["liq_total_usd"]   = liq.total_liq_usd_1h
        feats["liq_ratio"]       = liq.liq_ratio
        feats["liq_long_dom"]    = float(liq.liq_ratio >= 0.75)   # long squeeze flag
        feats["liq_short_dom"]   = float(liq.liq_ratio <= 0.25)   # short squeeze flag

    # ================================================================== #
    # 14. OPTIONS FEATURE-ÖK
    # ================================================================== #

    if crypto_snap is not None and crypto_snap.options is not None:
        opt = crypto_snap.options
        feats["opt_put_call_ratio"] = opt.put_call_ratio
        feats["opt_iv_atm"]         = opt.iv_atm_25d
        feats["opt_iv_skew"]        = opt.iv_skew
        feats["opt_fear_flag"]      = float(opt.put_call_ratio >= 1.5)

    # ================================================================== #
    # 15. HÍR SENTIMENT
    # ================================================================== #

    if crypto_snap is not None and crypto_snap.news is not None:
        feats["news_sentiment"]    = crypto_snap.news.sentiment_score
        feats["news_extreme_bull"] = float(crypto_snap.news.sentiment_score >= 0.5)

    # ================================================================== #
    # TISZTÍTÁS
    # ================================================================== #

    feats = feats.ffill().fillna(0.0)

    logger.info("Feature matrix: %d sor × %d feature", len(feats), feats.shape[1])

    return feats
