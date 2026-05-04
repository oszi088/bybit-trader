"""
Multi-symbol portfolio manager.

A `PortfolioTrader` egyszerre kezeli a top 20 (vagy tetszoleges) USDT
spot par koreskedeset:
  * minden coinhoz egy-egy `Trader` peldany
  * KOZOS RiskManager (portfolio-szintu napi/drawdown limit)
  * KOZOS broker es DB (a Bybiten csak egy szamla van)
  * korrelacio-vedelem: max N nyitott pozicio egyszerre

Az ugynok a coineken vegigvonul minden iteracio elott, es a nyitott
poziciok szamatol fuggoen dont uj belepesrol.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from agent import TradingAgent
from brokers import BybitBroker, PaperBroker
from config import TradingConfig
from data_source import CcxtDataSource
from db import TradeDb
from fear_greed import FearGreedSource
from notify import Notifier
from risk_manager import RiskManager
from trader import Trader

logger = logging.getLogger("portfolio")


class PortfolioTrader:
    """
    Tobb symbolot kezel parhuzamosan, kozos cash poolban es kozos
    kockazat-keretben. Iteracionkent vegigmegy a symbolokon, mindegyikre
    egy lepest tesz; a max_open_positions limit elerese utan mar nem
    nyit ujakat.
    """

    def __init__(
        self,
        symbols: List[str],
        base_config: TradingConfig,
        live_broker: bool = False,
        max_open_positions: int = 5,
    ):
        self.symbols = list(symbols)
        self.base_config = base_config
        self.max_open_positions = max_open_positions

        # KOZOS komponensek (portfolio szintu)
        self.risk = RiskManager(base_config.risk)
        self.db = TradeDb(base_config.db.db_path, enabled=base_config.db.enabled)
        self.notifier = Notifier(enabled=base_config.notify.enabled)
        self.fg = (FearGreedSource(ttl_sec=base_config.fear_greed.cache_ttl_sec)
                   if base_config.fear_greed.enabled else None)

        # Egyetlen Bybit broker (egy szamla); paperben symbolonkent kulon
        # broker, mert a PaperBroker a sajat cash poolat vezeti
        self._live = live_broker
        if live_broker:
            self._shared_broker = BybitBroker(base_config, dry_run=base_config.risk.dry_run)
        else:
            self._shared_broker = None

        # Symbolonkent egy-egy Trader
        self.traders: List[Trader] = []
        for sym in self.symbols:
            cfg = self._symbol_config(sym)
            agent = TradingAgent(cfg, fear_greed_source=self.fg)
            broker = (self._shared_broker
                      if live_broker
                      else PaperBroker(
                          cash=cfg.initial_balance / max(1, len(self.symbols)),
                          fee_rate=cfg.fee_rate,
                          position_size=cfg.position_size,
                      ))
            data = CcxtDataSource(
                exchange_id=cfg.exchange_id,
                symbol=cfg.symbol,
                timeframe=cfg.timeframe,
                endpoint=cfg.bybit_endpoint,
                market_type=cfg.market_type,
            )
            t = Trader(
                agent=agent, broker=broker, risk=self.risk, data_source=data,
                config=cfg, db=self.db, notifier=self.notifier,
            )
            self.traders.append(t)

    def _symbol_config(self, symbol: str) -> TradingConfig:
        """A base config klonja, a symbol felulirva."""
        # dataclasses.replace-szel mely-klont csinalunk a kozos almodulokrol is,
        # de itt a Risk + Notify + DB-t megosztjuk -> egyszeruen masoljuk a refet.
        from copy import copy
        cfg = copy(self.base_config)
        cfg.symbol = symbol
        return cfg

    @property
    def open_positions(self) -> int:
        return sum(1 for t in self.traders if t.broker.in_position)

    def step_all(self) -> None:
        """Egy korben mindegyik symbolra elvegez egy step-et."""
        for t in self.traders:
            # Ha a portfolio mar tele van, ne nyithasson uj poziciot;
            # de a meglevoket le tudja zarni (SL/TP/SELL signal)
            if (not t.broker.in_position
                    and self.open_positions >= self.max_open_positions):
                # 'Lite' lepes: csak SL/TP ellenorzes, uj entry blokkolva
                continue
            try:
                t.step()
            except Exception as e:
                logger.exception("[%s] step hiba: %s", t.config.symbol, e)
                if self.base_config.notify.notify_on_error:
                    self.notifier.error(f"{t.config.symbol}: {e}")

    def run_forever(self) -> None:
        logger.info("Portfolio elindult: %d symbol, max %d nyitott",
                    len(self.symbols), self.max_open_positions)
        try:
            while True:
                if self.risk.state.halted:
                    msg = f"Kill switch aktiv ({self.risk.state.halt_reason})"
                    logger.error(msg)
                    if self.base_config.notify.notify_on_kill_switch:
                        self.notifier.kill_switch(self.risk.state.halt_reason or "halted")
                    break
                self.step_all()
                logger.info("Portfolio status: %d/%d nyitott | %s",
                            self.open_positions, self.max_open_positions,
                            self.risk.status())
                time.sleep(self.base_config.poll_interval_sec)
        except KeyboardInterrupt:
            logger.info("Leallitas (Ctrl-C). %s", self.risk.status())
