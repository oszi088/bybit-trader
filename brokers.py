"""
Broker absztrakció.

Két konkrét megvalósítás osztozik ugyanazon az interfészen:
  * PaperBroker — memóriában szimulál (jelenlegi paper trader logikája).
  * BybitBroker — valódi Bybit spot megbízásokat küld CCXT-n keresztül,
                  testnet vagy mainnet (bybit.eu / bybit.com) ellen.

A Trader broker-agnosztikus: bármelyiket elfogadja, ezért könnyen
átkapcsolható paper -> testnet -> éles üzemmód között.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from config import TradingConfig, BYBIT_HOSTS, load_api_credentials


logger = logging.getLogger("broker")


# --------------------------------------------------------------------------- #
# Közös interfész
# --------------------------------------------------------------------------- #

@dataclass
class FillReport:
    """Egy végrehajtott művelet összegzése — log/audit célra."""

    side: str            # "BUY" vagy "SELL"
    size: float
    price: float
    fee: float
    timestamp: pd.Timestamp
    pnl: float = 0.0
    note: str = ""

    def __str__(self) -> str:
        return (
            f"{self.timestamp} {self.side:<4} {self.size:.6f} @ {self.price:.2f} "
            f"fee={self.fee:.4f} pnl={self.pnl:+.2f} {self.note}"
        )


class Broker(ABC):
    """Közös interfész paper és valódi brokerhez."""

    @property
    @abstractmethod
    def in_position(self) -> bool: ...

    @property
    @abstractmethod
    def entry_price(self) -> Optional[float]: ...

    @abstractmethod
    def buy(self, price: float, timestamp: pd.Timestamp,
            size: Optional[float] = None) -> Optional[FillReport]: ...

    @abstractmethod
    def sell(self, price: float, timestamp: pd.Timestamp,
             note: str = "", fraction: float = 1.0) -> Optional[FillReport]: ...

    @abstractmethod
    def equity(self, mark_price: float) -> float: ...


# --------------------------------------------------------------------------- #
# PaperBroker — memória, hálózat nélkül
# --------------------------------------------------------------------------- #

class PaperBroker(Broker):
    """Egyszerű memória-broker: cash + 1 long pozíció, díjjal."""

    def __init__(self, cash: float, fee_rate: float, position_size: float):
        self.cash = cash
        self.fee_rate = fee_rate
        self.position_size = position_size
        self.coin_balance: float = 0.0
        self._entry_price: Optional[float] = None
        self.history: List[FillReport] = []

    @property
    def in_position(self) -> bool:
        return self.coin_balance > 0

    @property
    def entry_price(self) -> Optional[float]:
        return self._entry_price

    def buy(self, price: float, timestamp: pd.Timestamp,
            size: Optional[float] = None) -> Optional[FillReport]:
        if self.in_position:
            return None
        if size is None:
            notional = self.cash * self.position_size
            size = notional / (price * (1 + self.fee_rate))
        fee = size * price * self.fee_rate
        self.cash -= size * price + fee
        self.coin_balance = size
        self._entry_price = price
        report = FillReport("BUY", size, price, fee, timestamp)
        self.history.append(report)
        return report

    def sell(self, price: float, timestamp: pd.Timestamp,
             note: str = "", fraction: float = 1.0) -> Optional[FillReport]:
        if not self.in_position:
            return None
        fraction = max(0.0, min(1.0, fraction))
        size = self.coin_balance * fraction
        gross = size * price
        fee = gross * self.fee_rate
        proceeds = gross - fee
        entry = self._entry_price or price
        pnl = proceeds - size * entry * (1 + self.fee_rate)
        self.cash += proceeds
        self.coin_balance -= size
        if self.coin_balance <= 0 or fraction >= 1.0:
            self.coin_balance = 0.0
            self._entry_price = None
        report = FillReport("SELL", size, price, fee, timestamp, pnl=pnl, note=note)
        self.history.append(report)
        return report

    def equity(self, mark_price: float) -> float:
        return self.cash + self.coin_balance * mark_price


# --------------------------------------------------------------------------- #
# BybitBroker — valódi spot megbízások CCXT-n keresztül
# --------------------------------------------------------------------------- #

class BybitBroker(Broker):
    """
    Bybit spot broker.

    Az `endpoint` kapcsolóval váltunk testnet / bybit.eu / bybit.com között.
    API kulcsokat KIZÁRÓLAG környezeti változókból olvas (BYBIT_API_KEY,
    BYBIT_API_SECRET) — soha nem fogad őket paraméterként a forráskódban.

    Ha `dry_run=True`, NEM küld valódi megbízást, csak logol és vezet egy
    belső állapotot — így azonos megbízási útvonalon tesztelhető.
    """

    def __init__(self, config: TradingConfig, dry_run: bool = False):
        try:
            import ccxt  # type: ignore
        except ImportError as e:
            raise ImportError("Telepítsd a ccxt csomagot: `pip install ccxt`") from e

        api_key, api_secret = load_api_credentials()
        if not api_key or not api_secret:
            raise RuntimeError(
                "BYBIT_API_KEY / BYBIT_API_SECRET környezeti változók nincsenek beállítva."
            )

        self.config = config
        self.dry_run = dry_run

        # CCXT exchange példány Bybit-hez, spot piactípussal
        self.exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": config.market_type},  # 'spot'
        })

        # Endpoint felülírása (testnet vagy bybit.eu vagy bybit.com)
        self._configure_endpoint(config.bybit_endpoint)

        # Helyi nyilvántartás a pozícióról és belépő árról
        self._coin_balance: float = 0.0
        self._entry_price: Optional[float] = None
        self.history: List[FillReport] = []

    # ------------------------------------------------------------------ #
    # Endpoint felülírás
    # ------------------------------------------------------------------ #

    def _configure_endpoint(self, endpoint: str) -> None:
        """Beállítja a Bybit hosztot (testnet / bybit.eu / bybit.com)."""
        host = BYBIT_HOSTS[endpoint]

        if endpoint == "testnet":
            # CCXT beépített testnet támogatás
            self.exchange.set_sandbox_mode(True)
            logger.warning("Bybit TESTNET aktív — nem valódi pénz.")
            return

        # Mainnet endpoint kézi felülírása. Bybit.eu az európai entitás,
        # bybit.com a globális. A CCXT exchange.urls['api'] dict-jét írjuk át,
        # így minden REST hívás az új hosztra megy.
        try:
            api_urls = self.exchange.urls.get("api", {})
            if isinstance(api_urls, dict):
                for key in list(api_urls.keys()):
                    api_urls[key] = host
                self.exchange.urls["api"] = api_urls
            else:
                self.exchange.urls["api"] = host
        except Exception as e:  # ne álljunk meg ha a CCXT belső struktúra változik
            logger.warning("Endpoint felülírás meghiúsult: %s — alapértelmezett host marad.", e)

        if endpoint == "eu":
            logger.warning("Bybit.eu MAINNET aktív — VALÓDI pénzes módban vagyunk.")
        else:
            logger.warning("Bybit.com MAINNET aktív — VALÓDI pénzes módban vagyunk.")

    # ------------------------------------------------------------------ #
    # Számlainformációk
    # ------------------------------------------------------------------ #

    @property
    def in_position(self) -> bool:
        return self._coin_balance > 0

    @property
    def entry_price(self) -> Optional[float]:
        return self._entry_price

    def fetch_quote_balance(self) -> float:
        """A USDT (vagy más quote) szabad egyenlege a Bybit számlán."""
        quote = self.config.symbol.split("/")[1]
        if self.dry_run:
            return self.config.initial_balance
        bal = self.exchange.fetch_balance()
        return float(bal.get(quote, {}).get("free", 0.0))

    def fetch_base_balance(self) -> float:
        """A bázis coin (pl. BTC) szabad egyenlege."""
        base = self.config.symbol.split("/")[0]
        if self.dry_run:
            return self._coin_balance
        bal = self.exchange.fetch_balance()
        return float(bal.get(base, {}).get("free", 0.0))

    def equity(self, mark_price: float) -> float:
        """Teljes vagyon piaci értéken (USDT + base * ár)."""
        if self.dry_run:
            return self.config.initial_balance + self._coin_balance * mark_price
        try:
            return self.fetch_quote_balance() + self.fetch_base_balance() * mark_price
        except Exception as e:
            logger.warning("Equity lekérés sikertelen: %s", e)
            return float("nan")

    # ------------------------------------------------------------------ #
    # Megbízás küldés
    # ------------------------------------------------------------------ #

    def _round_size(self, size: float) -> float:
        """Bybit szerinti kerekítés, ha a piac metaadat elérhető."""
        try:
            return float(self.exchange.amount_to_precision(self.config.symbol, size))
        except Exception:
            return round(size, 6)

    def buy(self, price: float, timestamp: pd.Timestamp,
            size: Optional[float] = None) -> Optional[FillReport]:
        if self.in_position:
            return None

        # A megrendelést a SZABAD quote egyenleg position_size hányadából méretezzük
        if size is None:
            free_quote = self.fetch_quote_balance()
            notional = free_quote * self.config.position_size
            size = notional / price
        size = self._round_size(size)
        if size <= 0:
            logger.warning("Méret 0-ra kerekedett, kihagyva a vételt.")
            return None

        fill_price = price  # fallback: kért ár
        if self.dry_run:
            logger.info("[DRY-RUN] BUY %s %.6f @ %.2f", self.config.symbol, size, price)
        else:
            order = self.exchange.create_market_buy_order(self.config.symbol, size)
            order_id = order.get("id")
            status = order.get("status", "unknown")
            # Bybit valós API-n "Filled" (nagybetűs), CCXT normalizálja → "closed"/"filled"
            # Case-insensitive check a biztonság kedvéért
            if status.lower() not in ("closed", "filled"):
                logger.error("BUY order %s nem toltodott ki (status=%s) - pozicio nem nyilik",
                             order_id, status)
                return None
            # Tényleges kitöltési ár (average fill) — NEM a request ár
            # `or` helyett explicit None/0 ellenőrzés: average=0.0 helytelen fallbacket okozna
            _avg = order.get("average")
            fill_price = float(_avg) if _avg else (float(order.get("price") or price))
            logger.info("Bybit BUY order: %s (status=%s, fill=%.4f)", order_id, status, fill_price)

        # Helyi pozíció vezetése
        self._coin_balance = size
        self._entry_price = fill_price
        fee = size * fill_price * self.config.fee_rate
        report = FillReport("BUY", size, fill_price, fee, timestamp,
                            note="dry-run" if self.dry_run else "")
        self.history.append(report)
        return report

    def sell(self, price: float, timestamp: pd.Timestamp,
             note: str = "", fraction: float = 1.0) -> Optional[FillReport]:
        if not self.in_position:
            return None

        fraction = max(0.0, min(1.0, fraction))
        size = self._round_size(self._coin_balance * fraction)
        if size <= 0:
            return None

        if self.dry_run:
            logger.info("[DRY-RUN] SELL %s %.6f @ %.2f frac=%.0f%% (%s)",
                        self.config.symbol, size, price, fraction * 100, note)
        else:
            order = self.exchange.create_market_sell_order(self.config.symbol, size)
            order_id = order.get("id")
            status = order.get("status", "unknown")
            # Bybit valós API-n "Filled" (nagybetűs), CCXT normalizálja → "closed"/"filled"
            # Case-insensitive check a biztonság kedvéért
            if status.lower() not in ("closed", "filled"):
                logger.error("SELL order %s nem toltodott ki (status=%s) - pozicio nyitva marad",
                             order_id, status)
                return None
            logger.info("Bybit SELL order: %s (status=%s, frac=%.0f%%)", order_id, status, fraction * 100)

        gross = size * price
        fee = gross * self.config.fee_rate
        proceeds = gross - fee
        entry = self._entry_price or price
        # Helyes PnL: nettó bevétel mínusz belépési cost basis (belépési díjjal)
        pnl = proceeds - size * entry * (1 + self.config.fee_rate)

        # Helyi pozíció frissítése — részleges zárás esetén a maradék megmarad
        self._coin_balance -= size
        if self._coin_balance <= 0 or fraction >= 1.0:
            self._coin_balance = 0.0
            self._entry_price = None
        report = FillReport("SELL", size, price, fee, timestamp, pnl=pnl,
                            note=note or ("dry-run" if self.dry_run else ""))
        self.history.append(report)
        return report
