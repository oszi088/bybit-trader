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
from drift_detector import DriftDetector, DriftConfig
from fear_greed import FearGreedSource
from notify import Notifier
from performance_attribution import PerformanceAttributor
from portfolio_risk import PortfolioRiskManager, PortfolioRiskConfig
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

        # --- Portfolio-szintu kockazat es teljesitmeny modulok ---
        self._portfolio_risk = PortfolioRiskManager(PortfolioRiskConfig())
        self._drift = (
            DriftDetector(DriftConfig(
                window            = base_config.spot_drift.window,
                min_trades        = base_config.spot_drift.min_trades,
                min_sharpe        = base_config.spot_drift.min_sharpe,
                min_win_rate      = base_config.spot_drift.min_win_rate,
                min_profit_factor = base_config.spot_drift.min_profit_factor,
                min_edge_ratio    = base_config.spot_drift.min_edge_ratio,
            ))
            if base_config.spot_drift.enabled else None
        )
        self._attributor = PerformanceAttributor()

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
        """A base config mély klonja, a symbol felülírva.

        deepcopy nélkül az összes Trader ugyanazt a Risk/Stop/Notify
        sub-objektumot osztaná meg — az egyik Trader stop-loss
        állapota kisziváragna a másikba.
        """
        from copy import deepcopy
        cfg = deepcopy(self.base_config)
        cfg.symbol = symbol
        return cfg

    @property
    def open_positions(self) -> int:
        return sum(1 for t in self.traders if t.broker.in_position)

    def step_all(self) -> None:
        """Egy korben mindegyik symbolra elvegez egy step-et."""
        # Nyitott poziciok listaja portfolio kockazat ellenorzeshez
        open_symbols = [t.config.symbol for t in self.traders if t.broker.in_position]

        # FIX #7: valódi hozamadatok összegyűjtése az előző iteráció OHLCV-jéből.
        # Az _last_returns az előző step() hívásban töltődik fel → 1 iteráció késés,
        # ami portfolio döntésnél teljesen elfogadható.
        returns_dict = {
            t.config.symbol: t._last_returns
            for t in self.traders
            if getattr(t, "_last_returns", [])
        }

        for t in self.traders:
            # Ha a portfolio mar tele van, ne nyithasson uj poziciot;
            # de a meglevoket le tudja zarni (SL/TP/SELL signal)
            if (not t.broker.in_position
                    and self.open_positions >= self.max_open_positions):
                continue

            # Portfolio-szintu kockazat ellenorzes uj belepes elott
            if not t.broker.in_position and self._portfolio_risk:
                risk_result = self._portfolio_risk.check_new_position(
                    symbol              = t.config.symbol,
                    open_symbols        = open_symbols,
                    returns_dict        = returns_dict,   # FIX #7: valós hozamok
                    total_portfolio_usd = self.base_config.initial_balance,
                    new_position_usd    = self.base_config.risk.max_order_value_usd,
                )
                if not risk_result.can_open:
                    logger.info("[%s] Portfolio kockazat blokk: %s",
                                t.config.symbol, risk_result.reason)
                    continue

            try:
                t.step()
            except Exception as e:
                logger.exception("[%s] step hiba: %s", t.config.symbol, e)
                if self.base_config.notify.notify_on_error:
                    self.notifier.error(f"{t.config.symbol}: {e}")

    def performance_report(self) -> str:
        """Teljesitmeny attribus report generalasa a trade logbol."""
        trades = self.db.list_trades(limit=500)
        if not trades:
            return "Nincs elegendo trade adat a riporthoz."
        report = self._attributor.generate_report(trades)
        return self._attributor.format_report(report)

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
