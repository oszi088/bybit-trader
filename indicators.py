"""
Technikai indikátorok tiszta pandas/numpy implementációi.

Minden függvény egy OHLCV pandas.DataFrame-et vár (oszlopok: open, high,
low, close, volume), és pandas.Series vagy DataFrame eredményt ad vissza.

A cél az olvashatóság: nem használunk TA-Lib-et vagy más natív könyvtárat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from orderbook_features import estimate_ob_imbalance_from_ohlcv


# --------------------------------------------------------------------------- #
# Trend indikátorok
# --------------------------------------------------------------------------- #

def sma(series: pd.Series, period: int) -> pd.Series:
    """Egyszerű mozgóátlag (Simple Moving Average)."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponenciális mozgóátlag (Exponential Moving Average)."""
    return series.ewm(span=period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD: a fast és slow EMA különbsége, plusz a különbség EMA-ja (signal vonal).

    Visszatér: macd, signal, hist (mind oszlop).
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": histogram})


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index — a trend erősségét méri (0-100).

    Visszatér: +DI, -DI, ADX értékek.
    """
    high, low, close = df["high"], df["low"], df["close"]

    # Igazi tartomány (True Range)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Iránymutatások
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder-féle simítás (EMA period=period, alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_value = dx.ewm(alpha=1 / period, adjust=False).mean()

    return pd.DataFrame({"+DI": plus_di, "-DI": minus_di, "ADX": adx_value})


# --------------------------------------------------------------------------- #
# Momentum indikátorok
# --------------------------------------------------------------------------- #

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (0-100). >70 túlvett, <30 túladott.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator. %K a záró ár pozíciója a tartományban; %D a %K SMA-ja.
    """
    high, low, close = df["high"], df["low"], df["close"]
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()

    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"%K": k, "%D": d})


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index. Tipikusan ±100 körüli tartomány;
    az ezen kívüli értékek erős momentumot jeleznek.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical_price.rolling(period).mean()
    mean_dev = (typical_price - sma_tp).abs().rolling(period).mean()
    return (typical_price - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


# --------------------------------------------------------------------------- #
# Volatilitás indikátorok
# --------------------------------------------------------------------------- #

def bollinger_bands(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands: középvonal SMA ± std_mult * szórás.
    """
    middle = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — abszolút volatilitás mérőszám.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# --------------------------------------------------------------------------- #
# Volumen indikátorok
# --------------------------------------------------------------------------- #

def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume. A volumen kumulatív, a záró ár irányával előjelezve.
    """
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()


def vwap(df: pd.DataFrame, period: int = 24) -> pd.Series:
    """
    Rolling Volume Weighted Average Price.

    A kumulatív cumsum() az egész dataseten előre tekint (lookahead),
    ezért rolling ablakot használunk. Az alapértelmezett 24 gyertya
    1h TF-en egyenlő egy nap VWAP-jával; 4h-on állítsd 6-ra, 1d-n 1-re.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    roll_vol_price = (typical_price * df["volume"]).rolling(period, min_periods=1).sum()
    roll_vol = df["volume"].rolling(period, min_periods=1).sum().replace(0, np.nan)
    return roll_vol_price / roll_vol


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index — a volumennel súlyozott RSI-szerű mutató (0-100).
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]

    direction = np.sign(typical_price.diff().fillna(0))
    positive_flow = pd.Series(np.where(direction > 0, money_flow, 0.0), index=df.index)
    negative_flow = pd.Series(np.where(direction < 0, money_flow, 0.0), index=df.index)

    pos_sum = positive_flow.rolling(period).sum()
    neg_sum = negative_flow.rolling(period).sum().replace(0, np.nan)

    mf_ratio = pos_sum / neg_sum
    return 100 - (100 / (1 + mf_ratio))


# --------------------------------------------------------------------------- #
# Egybe gyűjtő segédfüggvény
# --------------------------------------------------------------------------- #

def compute_all(df: pd.DataFrame, params) -> pd.DataFrame:
    """
    Egy DataFrame-ben adja vissza az összes indikátor értékét, az eredeti
    OHLCV oszlopok mellett. Ezt használja az ügynök a döntéshozáshoz.
    """
    out = df.copy()

    # Trend
    out["sma_fast"] = sma(df["close"], params.sma_fast)
    out["sma_slow"] = sma(df["close"], params.sma_slow)
    out["sma_long"] = sma(df["close"], params.sma_long)

    # Golden / Death cross detekcio: SMA50 (sma_slow) vs SMA200 (sma_long)
    # Most-tortent esemeny az utolso `cross_lookback` gyertyan
    diff = out["sma_slow"] - out["sma_long"]
    diff_sign = np.sign(diff.fillna(0))
    diff_prev = diff_sign.shift(1).fillna(0)
    golden_event = ((diff_sign > 0) & (diff_prev <= 0)).astype(int)
    death_event = ((diff_sign < 0) & (diff_prev >= 0)).astype(int)
    out["recent_golden"] = golden_event.rolling(params.cross_lookback, min_periods=1).max().fillna(0).astype(int)
    out["recent_death"] = death_event.rolling(params.cross_lookback, min_periods=1).max().fillna(0).astype(int)
    out["ema_fast"] = ema(df["close"], params.ema_fast)
    out["ema_slow"] = ema(df["close"], params.ema_slow)

    macd_df = macd(df["close"], params.ema_fast, params.ema_slow, params.macd_signal)
    out[["macd", "macd_signal", "macd_hist"]] = macd_df.values

    adx_df = adx(df, params.adx_period)
    out[["plus_di", "minus_di", "adx"]] = adx_df.values

    # Momentum
    out["rsi"] = rsi(df["close"], params.rsi_period)
    stoch_df = stochastic(df, params.stoch_k, params.stoch_d)
    out[["stoch_k", "stoch_d"]] = stoch_df.values
    out["cci"] = cci(df, params.cci_period)

    # Volatilitás
    bb = bollinger_bands(df["close"], params.bb_period, params.bb_std)
    out[["bb_upper", "bb_middle", "bb_lower"]] = bb.values
    # Megjegyzés: az `atr` függvénynév nem lehet lokális változóban elrejtve
    # ebben a scope-ban — véletlenül ne írjuk felül pl. `atr = float(row["atr"])`.
    out["atr"] = atr(df, params.atr_period)  # atr() = indicators.atr függvény

    # Volumen
    out["obv"] = obv(df)
    out["vwap"] = vwap(df)
    out["mfi"] = mfi(df, params.mfi_period)
    out["vol_ma20"] = df["volume"].rolling(20, min_periods=1).mean()

    # Orderflow proxy (backtesthez; élő kereskedésnél felülírja az agent)
    out["ob_imbalance"]  = estimate_ob_imbalance_from_ohlcv(df)
    out["ob_large_order"] = 0  # nincs historikus OB → semleges

    return out
