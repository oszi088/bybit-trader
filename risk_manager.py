"""
RiskManager - vedelmek eles kereskedeshez.

Felelosseg:
  * Megrendelesertek plafon
  * Napi vesztesegminitlimit kill switch
  * Drawdown kill switch (peak equity-bol X%)
  * Volatilitas-szuro (max ATR/ar arany)
  * Score-aranyos pozicio meretezes (Kelly-szeruen)

Az allapot reset-elheto napvaltaskor; perzisztalhato a TradeDb-be.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from config import RiskConfig

logger = logging.getLogger("risk")


@dataclass
class RiskState:
    current_day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    realized_pnl_today: float = 0.0
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: Optional[str] = None


class RiskManager:
    """Megrendelesszintu szuro, napi kill switch, drawdown vedo."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.state = RiskState()

    # ------------------------------------------------------------------ #
    # Napvaltas kezelese
    # ------------------------------------------------------------------ #

    def _maybe_reset_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.state.current_day:
            logger.info("Uj nap (%s): napi PnL nullazva, kill switch feloldva.", today)
            self.state.current_day = today
            self.state.realized_pnl_today = 0.0
            # A drawdown halt a peak-bol indul, ezt nem oldjuk fel automatikusan
            if self.state.halted and self.state.halt_reason and "daily" in self.state.halt_reason:
                self.state.halted = False
                self.state.halt_reason = None

    # ------------------------------------------------------------------ #
    # Pre-trade ellenorzes
    # ------------------------------------------------------------------ #

    def should_open(self, price: float, size: float, atr: float = 0.0) -> tuple[bool, str]:
        """Engedelyezheto-e most uj belepes? Visszater (ok, indok)-kal."""
        self._maybe_reset_day()

        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}"

        order_value = price * size
        if order_value > self.config.max_order_value_usd:
            return False, (
                f"order value ${order_value:.2f} > limit ${self.config.max_order_value_usd:.2f}"
            )

        # Volatilitas szuro
        if atr > 0 and price > 0:
            atr_pct = atr / price
            if atr_pct >= self.config.max_atr_pct:
                return False, (
                    f"volatility too high: ATR/price={atr_pct:.1%} >= "
                    f"{self.config.max_atr_pct:.1%}"
                )

        return True, "ok"

    def cap_size_to_limit(self, price: float, requested_size: float,
                          score: float = 1.0) -> float:
        """
        Vagja le a meretet a plafonra. Ha score_proportional_size be van
        kapcsolva, akkor a |score|-val is szorzunk (Kelly-szeruen).
        """
        if price <= 0:
            return 0.0
        max_size = self.config.max_order_value_usd / price
        capped = min(requested_size, max_size)
        if self.config.score_proportional_size:
            capped *= max(0.0, min(1.0, abs(score)))
        return capped

    # ------------------------------------------------------------------ #
    # Post-trade frissites
    # ------------------------------------------------------------------ #

    def register_trade_pnl(self, pnl: float) -> None:
        self._maybe_reset_day()
        self.state.realized_pnl_today += pnl

        if self.state.realized_pnl_today <= -abs(self.config.daily_loss_limit_usd):
            self.state.halted = True
            self.state.halt_reason = (
                f"daily loss limit hit: {self.state.realized_pnl_today:+.2f} USD"
            )
            logger.error("KILL SWITCH: %s", self.state.halt_reason)

    def update_equity(self, equity: float) -> None:
        """
        Folyamatos equity-frissites a drawdown vedelemhez. Ha az equity
        a peak X%-a ala esik, halt allapotot kapcsolunk.
        """
        if equity <= 0:
            return
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        if self.state.peak_equity > 0:
            drawdown = 1 - (equity / self.state.peak_equity)
            if drawdown >= self.config.max_drawdown_pct:
                self.state.halted = True
                self.state.halt_reason = (
                    f"max drawdown hit: -{drawdown:.1%} (peak ${self.state.peak_equity:.2f})"
                )
                logger.error("KILL SWITCH: %s", self.state.halt_reason)

    # ------------------------------------------------------------------ #
    # Egyeb
    # ------------------------------------------------------------------ #

    @property
    def is_dry_run(self) -> bool:
        return self.config.dry_run

    def status(self) -> str:
        return (
            f"day={self.state.current_day} "
            f"pnl_today={self.state.realized_pnl_today:+.2f} "
            f"peak=${self.state.peak_equity:.2f} "
            f"halted={self.state.halted}"
        )
