"""
Adatforrasok: CSV (backteszthez) es CCXT (elo piaci adatokhoz).

A ket backend ugyanazt az interfeszt adja: egy pandas.DataFrame-et a
[open, high, low, close, volume] oszlopokkal es idozonas indexszel.

A CCXT backend Bybit-specifikus endpointokat is tamogat (testnet, bybit.eu,
bybit.com) - a publikus piaci adatokhoz NEM szukseges API kulcs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import BYBIT_HOSTS


logger = logging.getLogger("data")
REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def load_csv(path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Betolt egy szabvanyos OHLCV CSV-t backteszthez."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if timestamp_col not in df.columns:
        raise ValueError(f"Hianyzo timestamp oszlop: {timestamp_col}")

    ts = df[timestamp_col]
    if pd.api.types.is_numeric_dtype(ts):
        df.index = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        df.index = pd.to_datetime(ts, utc=True)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Hianyzo OHLCV oszlopok a CSV-ben: {missing}")

    return df[REQUIRED_COLS].sort_index()


class CcxtDataSource:
    """
    Vekony CCXT wrapper. CSAK olvas (publikus vegpont): nem szukseges API kulcs.

    Bybit endpointok:
        - "testnet"  -> https://api-testnet.bybit.com
        - "eu"       -> https://api.bybit.eu  (EU mainnet)
        - "global"   -> https://api.bybit.com (globalis mainnet)
    """

    def __init__(
        self,
        exchange_id: str = "bybit",
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        endpoint: Optional[str] = None,
        market_type: str = "spot",
    ):
        try:
            import ccxt  # type: ignore
        except ImportError as e:
            raise ImportError("Telepitsd a ccxt csomagot: pip install ccxt") from e

        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Ismeretlen exchange: {exchange_id}")

        self.exchange = exchange_cls({
            "enableRateLimit": True,
            "options": {"defaultType": market_type},
        })
        self.symbol = symbol
        self.timeframe = timeframe

        if exchange_id == "bybit" and endpoint:
            self._configure_bybit_endpoint(endpoint)

    def _configure_bybit_endpoint(self, endpoint: str) -> None:
        """Bybit endpoint felulirasa (testnet, bybit.eu vagy bybit.com)."""
        if endpoint == "testnet":
            self.exchange.set_sandbox_mode(True)
            return

        host = BYBIT_HOSTS[endpoint]
        try:
            api_urls = self.exchange.urls.get("api", {})
            if isinstance(api_urls, dict):
                for key in list(api_urls.keys()):
                    api_urls[key] = host
                self.exchange.urls["api"] = api_urls
            else:
                self.exchange.urls["api"] = host
        except Exception as e:
            logger.warning("Bybit endpoint feluliras meghiusult: %s", e)

    def fetch_ohlcv(self, limit: int = 200, since: Optional[datetime] = None) -> pd.DataFrame:
        """Lekeri az utolso N gyertya OHLCV-jet."""
        if since is not None:
            # Ha már timezone-aware: .astimezone() konvertál UTC-re.
            # Ha naive: feltételezzük UTC-t (replace nem konvertál, csak jelöl).
            _since_utc = since.astimezone(timezone.utc) if since.tzinfo else since.replace(tzinfo=timezone.utc)
            since_ms = int(_since_utc.timestamp() * 1000)
        else:
            since_ms = None
        raw = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since_ms, limit=limit)
        if not raw:
            return pd.DataFrame(columns=REQUIRED_COLS)
        df = pd.DataFrame(raw, columns=["timestamp"] + REQUIRED_COLS)
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df[REQUIRED_COLS].sort_index()

    def fetch_ticker_price(self) -> float:
        """A legutolso ismert ar a piacon."""
        ticker = self.exchange.fetch_ticker(self.symbol)
        return float(ticker["last"])
