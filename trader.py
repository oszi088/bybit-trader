"""
Trader - broker-agnosztikus elo/paper kereskedesi loop.

Bovitesek:
  * ATR-alapu dinamikus stop-loss / take-profit
  * Trailing stop (a max-arhoz huzva)
  * Score-aranyos pozicio meretezes (Kelly-szeruen)
  * SQLite trade log perzisztencia
  * Telegram ertesites (vagy log fallback)
  * Watchdog/heartbeat
  * Drawdown vedelem (RiskManager)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from agent import Decision, TradingAgent
from brokers import Broker, BybitBroker, FillReport, PaperBroker
from config import TradingConfig
from data_source import CcxtDataSource
from db import TradeDb
from notify import Notifier
from risk_manager import RiskManager

logger = logging.getLogger("trader")


class Trader:
    """Broker-fuggetlen kereskedesi loop ugyanazzal a logikaval."""

    def __init__(
        self,
        agent: TradingAgent,
        broker: Broker,
        risk: RiskManager,
        data_source: CcxtDataSource,
        config: Optional[TradingConfig] = None,
        history_limit: int = 200,
        db: Optional[TradeDb] = None,
        notifier: Optional[Notifier] = None,
    ):
        self.agent = agent
        self.broker = broker
        self.risk = risk
        self.data = data_source
        self.config = config or agent.config
        self.history_limit = history_limit

        self.db = db or TradeDb(self.config.db.db_path, enabled=self.config.db.enabled)
        self.notifier = notifier or Notifier(enabled=self.config.notify.enabled)

        # Trailing stop allapot - a poziciohoz tarolt legnagyobb ar
        self._highest_price: Optional[float] = None
        self._stop_price: Optional[float] = None     # aktualis stop (atr / fix)
        self._tp_price: Optional[float] = None       # aktualis take-profit
        self._last_step_ts: float = time.time()

    # ------------------------------------------------------------------ #
    # Egy iteracio
    # ------------------------------------------------------------------ #

    def step(self) -> Decision:
        ohlcv = self.data.fetch_ohlcv(limit=self.history_limit)
        if len(ohlcv) < 50:
            logger.warning("Keves gyertya erkezett (%d), kihagyjuk az iteraciot.", len(ohlcv))
            return Decision(action="HOLD", score=0.0, price=0.0, atr=0.0)

        decision = self.agent.decide(ohlcv)
        timestamp = ohlcv.index[-1]

        # Trailing es SL/TP ellenorzes elsokent
        self._update_trailing(decision.price, decision.atr)
        self._check_exits(decision.price, timestamp)

        # Az ugynok dontese (csak ha nem haltunk meg)
        if decision.action == "BUY" and not self.broker.in_position:
            self._try_open(decision, timestamp)
        elif decision.action == "SELL" and self.broker.in_position:
            self._close_position(decision.price, timestamp, note="signal")

        equity = self.broker.equity(decision.price)
        self.risk.update_equity(equity)
        self._last_step_ts = time.time()

        logger.info(
            "%s | equity=%.2f | %s | %s",
            timestamp, equity, decision.explain(), self.risk.status(),
        )
        return decision

    # ------------------------------------------------------------------ #
    # Belepesi logika
    # ------------------------------------------------------------------ #

    def _try_open(self, decision: Decision, timestamp: pd.Timestamp) -> None:
        ok, reason = self.risk.should_open(decision.price, size=1.0, atr=decision.atr)
        if not ok:
            logger.info("BUY visszautasitva: %s", reason)
            if "halted" in reason and self.config.notify.notify_on_kill_switch:
                self.notifier.kill_switch(reason)
            return

        report = self.broker.buy(decision.price, timestamp)
        if report is None:
            return

        # Stopok beallitasa belepeskor
        self._set_stops_on_entry(decision.price, decision.atr)

        # Persistencia + ertesites
        self.db.log_fill(self.config.symbol, "BUY", report.size, report.price,
                         report.fee, pnl=0.0, note=report.note, timestamp=timestamp)
        if self.config.notify.notify_on_fill:
            self.notifier.fill("BUY", self.config.symbol, report.size, report.price)
        logger.info("FILL: %s", report)

    def _close_position(self, price: float, timestamp: pd.Timestamp, note: str) -> None:
        report = self.broker.sell(price, timestamp, note=note)
        if report is None:
            return

        self.risk.register_trade_pnl(report.pnl)
        self.db.log_fill(self.config.symbol, "SELL", report.size, report.price,
                         report.fee, pnl=report.pnl, note=note, timestamp=timestamp)
        if self.config.notify.notify_on_fill:
            self.notifier.fill("SELL", self.config.symbol, report.size, report.price,
                               pnl=report.pnl)
        logger.info("FILL: %s", report)

        # Trailing reset
        self._highest_price = None
        self._stop_price = None
        self._tp_price = None

        # Kill switch ertesites, ha most kapcsolt be
        if self.risk.state.halted and self.config.notify.notify_on_kill_switch:
            self.notifier.kill_switch(self.risk.state.halt_reason or "halted")

    # ------------------------------------------------------------------ #
    # Stop / TP / trailing
    # ------------------------------------------------------------------ #

    def _set_stops_on_entry(self, entry_price: float, atr: float) -> None:
        """Belepeskor allitja be a stop-loss es take-profit szinteket."""
        cfg = self.config.stops
        if cfg.use_atr_stops and atr > 0:
            self._stop_price = entry_price - cfg.atr_stop_mult * atr
            self._tp_price = entry_price + cfg.atr_tp_mult * atr
        else:
            self._stop_price = entry_price * (1 - cfg.stop_loss_pct)
            self._tp_price = entry_price * (1 + cfg.take_profit_pct)
        self._highest_price = entry_price
        logger.info(
            "Stopok beallitva: entry=%.2f, SL=%.2f, TP=%.2f",
            entry_price, self._stop_price, self._tp_price,
        )

    def _update_trailing(self, price: float, atr: float) -> None:
        """A trailing stop kovet a maximum-arhoz."""
        if not self.broker.in_position or not self.config.stops.use_trailing_stop:
            return
        if self._highest_price is None or price > self._highest_price:
            self._highest_price = price
        if atr <= 0 or self._highest_price is None:
            return
        new_stop = self._highest_price - self.config.stops.trailing_atr_mult * atr
        # Csak felfele huzhato (soha nem lazitunk a stopon)
        if self._stop_price is None or new_stop > self._stop_price:
            self._stop_price = new_stop

    def _check_exits(self, price: float, timestamp: pd.Timestamp) -> None:
        """Stop-loss vagy take-profit eseten zarunk."""
        if not self.broker.in_position:
            return
        if self._stop_price is not None and price <= self._stop_price:
            logger.info("STOP @ %.2f (stop=%.2f)", price, self._stop_price)
            self._close_position(price, timestamp, note="stop_loss")
        elif self._tp_price is not None and price >= self._tp_price:
            logger.info("TAKE-PROFIT @ %.2f (tp=%.2f)", price, self._tp_price)
            self._close_position(price, timestamp, note="take_profit")

    # ------------------------------------------------------------------ #
    # Folyamatos futas + watchdog
    # ------------------------------------------------------------------ #

    def run_forever(self) -> None:
        logger.info(
            "Trader elindult. broker=%s symbol=%s timeframe=%s",
            type(self.broker).__name__, self.config.symbol, self.config.timeframe,
        )
        try:
            while True:
                if self.risk.state.halted:
                    msg = f"Kill switch aktiv ({self.risk.state.halt_reason}) - leallas."
                    logger.error(msg)
                    if self.config.notify.notify_on_kill_switch:
                        self.notifier.kill_switch(self.risk.state.halt_reason or "halted")
                    break
                try:
                    self.step()
                except Exception as e:
                    logger.exception("Hiba az iteracioban: %s", e)
                    if self.config.notify.notify_on_error:
                        self.notifier.error(str(e))
                self._watchdog_check()
                time.sleep(self.config.poll_interval_sec)
        except KeyboardInterrupt:
            logger.info("Leallitas (Ctrl-C). %s", self.risk.status())

    def _watchdog_check(self) -> None:
        """Ha tul reg nem volt sikeres step, riasztunk."""
        if not self.config.watchdog.enabled:
            return
        silent_for = time.time() - self._last_step_ts
        if silent_for > self.config.watchdog.max_silent_seconds:
            msg = f"Watchdog: {silent_for:.0f}s nincs sikeres step"
            logger.warning(msg)
            if self.config.notify.notify_on_error:
                self.notifier.error(msg)
            self._last_step_ts = time.time()  # ne riasszunk minden iteracioban


# ============================================================================
# Factory fuggvenyek
# ============================================================================

def build_paper_trader(config: TradingConfig) -> Trader:
    agent = TradingAgent(config)
    broker = PaperBroker(
        cash=config.initial_balance,
        fee_rate=config.fee_rate,
        position_size=config.position_size,
    )
    risk = RiskManager(config.risk)
    data = CcxtDataSource(
        exchange_id=config.exchange_id,
        symbol=config.symbol,
        timeframe=config.timeframe,
        endpoint=config.bybit_endpoint,
        market_type=config.market_type,
    )
    return Trader(agent, broker, risk, data, config)


def build_bybit_trader(config: TradingConfig) -> Trader:
    agent = TradingAgent(config)
    broker = BybitBroker(config, dry_run=config.risk.dry_run)
    risk = RiskManager(config.risk)
    data = CcxtDataSource(
        exchange_id=config.exchange_id,
        symbol=config.symbol,
        timeframe=config.timeframe,
        endpoint=config.bybit_endpoint,
        market_type=config.market_type,
    )
    return Trader(agent, broker, risk, data, config)
