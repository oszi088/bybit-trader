"""
RiskManager - védelmek éles kereskedéshez.

Felelősség:
  * Megrendelésérték plafon
  * Napi veszteséglimit kill switch
  * Drawdown kill switch (peak equity-ből X%)
  * Volatilitás-szűrő (max ATR/ár arány)
  * Score-arányos pozíció méretezés (Kelly-szerűen)
  * Egymást követő veszteség circuit breaker (spot trading)

Egymást követő veszteség szabályok (consecutive loss circuit breaker):
  3 veszteség → méret felezhető (size_mult = 0.5)
  5 veszteség → 4 órás szünet (nem nyit új pozíciót)
  7 veszteség → teljes leállás (halt)

Az állapot reset-elhető napváltáskor; perzisztálható a TradeDb-be.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
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

    # Egymást követő veszteség számlálója
    consecutive_losses: int = 0
    # Aktuális méretszorzó (1.0 = teljes méret, 0.5 = fél méret)
    consecutive_loss_size_mult: float = 1.0
    # Ha van aktív szünet: eddig nem nyitunk új pozíciót
    pause_until: Optional[datetime] = None


class RiskManager:
    """Megrendelésszintű szűrő, napi kill switch, drawdown védő,
    és egymást követő veszteség circuit breaker."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.state = RiskState()

    # ------------------------------------------------------------------ #
    # Napváltás kezelése
    # ------------------------------------------------------------------ #

    def _maybe_reset_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.state.current_day:
            logger.info("Új nap (%s): napi PnL nullázva, kill switch feloldva.", today)
            self.state.current_day = today
            self.state.realized_pnl_today = 0.0
            # A drawdown halt a peak-ből indul, ezt nem oldjuk fel automatikusan
            if self.state.halted and self.state.halt_reason and "daily" in self.state.halt_reason:
                self.state.halted = False
                self.state.halt_reason = None
            # Napi reset: egymást követő veszteség számláló is visszaáll
            # (profi szabály: minden nap új lap)
            if self.state.consecutive_losses > 0:
                logger.info(
                    "Napi reset: consecutive_losses=%d → 0, méretszorzó 1.0-ra visszaállítva.",
                    self.state.consecutive_losses,
                )
            self.state.consecutive_losses = 0
            self.state.consecutive_loss_size_mult = 1.0
            self.state.pause_until = None

    # ------------------------------------------------------------------ #
    # Pre-trade ellenőrzés
    # ------------------------------------------------------------------ #

    def should_open(self, price: float, size: float, atr: float = 0.0) -> tuple[bool, str]:
        """Engedélyezhető-e most új belépés? Visszatér (ok, indok)-kal."""
        self._maybe_reset_day()

        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}"

        # Szünet ellenőrzés (consecutive loss pause)
        now = datetime.now(timezone.utc)
        if self.state.pause_until is not None and now < self.state.pause_until:
            remaining_min = (self.state.pause_until - now).total_seconds() / 60
            return False, (
                f"consecutive loss pause: {self.state.consecutive_losses} veszteség, "
                f"szünet még {remaining_min:.0f} percig"
            )

        order_value = price * size
        if order_value > self.config.max_order_value_usd:
            return False, (
                f"order value ${order_value:.2f} > limit ${self.config.max_order_value_usd:.2f}"
            )

        # Volatilitás szűrő
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
        Vágja le a méretet a plafonra. Ha score_proportional_size be van
        kapcsolva, akkor a |score|-val is szorzunk (Kelly-szerűen).
        A consecutive loss szorzót is alkalmazza.
        """
        if price <= 0:
            return 0.0
        max_size = self.config.max_order_value_usd / price
        capped = min(requested_size, max_size)
        if self.config.score_proportional_size:
            capped *= max(0.0, min(1.0, abs(score)))
        # Egymást követő veszteség szorzó alkalmazása
        capped *= self.state.consecutive_loss_size_mult
        return capped

    def get_size_multiplier(self) -> float:
        """Aktuális méretszorzó (1.0 = teljes, 0.5 = fél méret a veszteség circuit breakertől)."""
        return self.state.consecutive_loss_size_mult

    # ------------------------------------------------------------------ #
    # Post-trade frissítés
    # ------------------------------------------------------------------ #

    def register_trade_pnl(self, pnl: float) -> None:
        self._maybe_reset_day()
        self.state.realized_pnl_today += pnl

        # --- Egymást követő veszteség circuit breaker ---
        if pnl < 0:
            self.state.consecutive_losses += 1
            self._apply_consecutive_loss_rules()
        elif pnl > 0:
            # Nyerő trade: visszaállítjuk a számlálót és a szorzót
            if self.state.consecutive_losses > 0:
                logger.info(
                    "Nyerő trade (pnl=%.2f): consecutive_losses %d → 0, méretszorzó visszaállítva.",
                    pnl, self.state.consecutive_losses,
                )
            self.state.consecutive_losses = 0
            self.state.consecutive_loss_size_mult = 1.0
            self.state.pause_until = None

        # --- Napi veszteséglimit kill switch ---
        if self.state.realized_pnl_today <= -abs(self.config.daily_loss_limit_usd):
            self.state.halted = True
            self.state.halt_reason = (
                f"daily loss limit hit: {self.state.realized_pnl_today:+.2f} USD"
            )
            logger.error("KILL SWITCH: %s", self.state.halt_reason)

    def _apply_consecutive_loss_rules(self) -> None:
        """Egymást követő veszteség szabályok alkalmazása a konfigurált küszöbök alapján."""
        n = self.state.consecutive_losses
        stop_at  = getattr(self.config, "consecutive_loss_stop_at",  7)
        pause_at = getattr(self.config, "consecutive_loss_pause_at", 5)
        half_at  = getattr(self.config, "consecutive_loss_half_at",  3)
        pause_hours = getattr(self.config, "consecutive_loss_pause_hours", 4.0)

        if n >= stop_at:
            # 7+ veszteség: teljes leállás
            self.state.halted = True
            self.state.halt_reason = (
                f"consecutive loss circuit breaker: {n} egymást követő veszteség → STOP"
            )
            logger.error("KILL SWITCH (consecutive loss): %s", self.state.halt_reason)

        elif n >= pause_at:
            # 5-6 veszteség: időszakos szünet
            self.state.pause_until = datetime.now(timezone.utc) + timedelta(hours=pause_hours)
            self.state.consecutive_loss_size_mult = 0.5
            logger.warning(
                "Consecutive loss PAUSE: %d veszteség → szünet %.1f óráig (%s), méret: 50%%",
                n, pause_hours, self.state.pause_until.strftime("%H:%M UTC"),
            )

        elif n >= half_at:
            # 3-4 veszteség: méret felezése
            self.state.consecutive_loss_size_mult = 0.5
            logger.warning(
                "Consecutive loss SIZE REDUCE: %d veszteség → méret 50%%", n,
            )
        else:
            logger.info("Consecutive loss: %d/%d (még nincs akció)", n, half_at)

    def update_equity(self, equity: float) -> None:
        """
        Folyamatos equity-frissítés a drawdown védelemhez. Ha az equity
        a peak X%-a alá esik, halt állapotot kapcsolunk.
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
    # Egyéb
    # ------------------------------------------------------------------ #

    @property
    def is_dry_run(self) -> bool:
        return self.config.dry_run

    def status(self) -> str:
        pause_str = ""
        if self.state.pause_until:
            now = datetime.now(timezone.utc)
            if now < self.state.pause_until:
                remaining = (self.state.pause_until - now).total_seconds() / 60
                pause_str = f" pause={remaining:.0f}m"
        return (
            f"day={self.state.current_day} "
            f"pnl_today={self.state.realized_pnl_today:+.2f} "
            f"peak=${self.state.peak_equity:.2f} "
            f"consec_loss={self.state.consecutive_losses} "
            f"size_mult={self.state.consecutive_loss_size_mult:.1f}"
            f"{pause_str} "
            f"halted={self.state.halted}"
        )
