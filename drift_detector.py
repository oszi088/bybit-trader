"""
drift_detector.py — Modell drift detektálás

Ha a modell teljesítménye romlik, automatikusan jelezzük és
(opcionálisan) csökkentjük a méretet vagy megállítjuk a kereskedést.

Detektálási metrikák (csúszóablak-alapú):
  - Rolling Sharpe ratio (utolsó window trade P&L-ből)
  - Win rate (nyerő trade-ek aránya)
  - Profit factor (bruttó nyerők / bruttó veszítők)
  - Edge ratio (átlagos nyerő / átlagos veszítő abszolút értéke)

Action-küszöbök:
  "ok"           → minden rendben
  "warn"         → egy metrika romlott, de még ok
  "reduce_size"  → két metrika romlott → fél méretben kereskedünk
  "pause"        → három metrika romlott → szünet, várjuk a retrain-t
  "stop"         → retrain szükséges, kereskedés megáll
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger("drift_detector")

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

DriftAction = Literal["ok", "warn", "reduce_size", "pause", "stop"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DriftStatus:
    """A drift detektor aktuális állapotának leírója."""

    action: str                   # DriftAction értékek egyike
    rolling_sharpe: float
    win_rate: float
    profit_factor: float
    edge_ratio: float
    poor_metrics_count: int
    size_multiplier: float        # 1.0, 0.5, or 0.0 – action alapján
    is_drifting: bool
    note: str


@dataclass
class DriftConfig:
    """Drift detektálás konfigurációja."""

    window: int = 20             # utolsó N trade vizsgálata
    min_trades: int = 10         # ennél kevesebb trade esetén nincs detektálás
    min_sharpe: float = 0.3      # alatta: "warn"
    min_win_rate: float = 0.40
    min_profit_factor: float = 1.0   # alatta: veszteséges
    min_edge_ratio: float = 0.8
    warn_threshold: int = 1      # ennyi "poor" metrikánál warn
    reduce_threshold: int = 2
    pause_threshold: int = 3
    stop_threshold: int = 4
    # FIX #5: Sharpe annualizáció: évi kereskedések száma a használt timeframe-en.
    # Alapértelmezett 252 = napi. 1h TF-en: 365*24=8760, 4h: 365*6=2190 stb.
    # Használd a config.TIMEFRAME_PERIODS_PER_YEAR segédszótárat.
    annualization_factor: float = 252.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Csúszóablak-alapú drift detektor.

    Nyomon követi az utolsó N trade P&L-jét, és négy metrika alapján
    határozza meg, hogy a modell teljesítménye elfogadható-e:
      - rolling Sharpe ratio
      - win rate
      - profit factor
      - edge ratio (avg_winner / avg_loser)

    Példa:
        detector = DriftDetector()
        status = detector.update(pnl=12.5)
        if status.action == "stop":
            bot.halt("drift detected")
    """

    def __init__(self, config: Optional[DriftConfig] = None) -> None:
        self.config: DriftConfig = config if config is not None else DriftConfig()
        self._pnl_history: deque[float] = deque(maxlen=self.config.window)
        self._action_history: deque[str] = deque(maxlen=self.config.window)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, pnl: float) -> DriftStatus:
        """Új P&L értéket ad a historikus ablakhoz, majd kiszámolja az állapotot."""
        self._pnl_history.append(pnl)
        status = self.check()
        self._action_history.append(status.action)
        logger.debug(
            "drift update: pnl=%.4f | action=%s | sharpe=%.3f | wr=%.2f | pf=%.2f | er=%.2f",
            pnl,
            status.action,
            status.rolling_sharpe,
            status.win_rate,
            status.profit_factor,
            status.edge_ratio,
        )
        return status

    def check(self) -> DriftStatus:
        """Számítja az aktuális drift állapotot, nem módosítja a historikus ablakot."""
        pnls = list(self._pnl_history)
        cfg = self.config

        if len(pnls) < cfg.min_trades:
            return DriftStatus(
                action="ok",
                rolling_sharpe=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                edge_ratio=0.0,
                poor_metrics_count=0,
                size_multiplier=1.0,
                is_drifting=False,
                note=(
                    f"Not enough trades ({len(pnls)}/{cfg.min_trades}) "
                    "to evaluate drift — no action."
                ),
            )

        sharpe, win_rate, profit_factor, edge_ratio = self._compute_metrics(pnls)

        poor_count = 0
        poor_names: list[str] = []

        if sharpe < cfg.min_sharpe:
            poor_count += 1
            poor_names.append(f"sharpe={sharpe:.3f}<{cfg.min_sharpe}")
        if win_rate < cfg.min_win_rate:
            poor_count += 1
            poor_names.append(f"win_rate={win_rate:.2%}<{cfg.min_win_rate:.2%}")
        if profit_factor < cfg.min_profit_factor:
            poor_count += 1
            poor_names.append(f"profit_factor={profit_factor:.3f}<{cfg.min_profit_factor}")
        if edge_ratio < cfg.min_edge_ratio:
            poor_count += 1
            poor_names.append(f"edge_ratio={edge_ratio:.3f}<{cfg.min_edge_ratio}")

        action, size_multiplier = self._determine_action(poor_count)
        is_drifting = action not in ("ok", "warn")
        note = (
            f"Poor metrics ({poor_count}): {', '.join(poor_names)}"
            if poor_names
            else "All metrics within acceptable range."
        )

        return DriftStatus(
            action=action,
            rolling_sharpe=sharpe,
            win_rate=win_rate,
            profit_factor=profit_factor,
            edge_ratio=edge_ratio,
            poor_metrics_count=poor_count,
            size_multiplier=size_multiplier,
            is_drifting=is_drifting,
            note=note,
        )

    def reset(self) -> None:
        """Törli a P&L és action historikus ablakot."""
        self._pnl_history.clear()
        self._action_history.clear()
        logger.info("DriftDetector history cleared.")

    def describe(self) -> str:
        """Emberi olvashatóságú összefoglaló az aktuális állapotról."""
        status = self.check()
        trades_in_window = len(self._pnl_history)
        lines = [
            f"=== DriftDetector status ===",
            f"  Trades in window : {trades_in_window}/{self.config.window}",
            f"  Action           : {status.action}",
            f"  Size multiplier  : {status.size_multiplier:.1f}",
            f"  Rolling Sharpe   : {status.rolling_sharpe:.4f}",
            f"  Win rate         : {status.win_rate:.2%}",
            f"  Profit factor    : {status.profit_factor:.4f}",
            f"  Edge ratio       : {status.edge_ratio:.4f}",
            f"  Poor metric cnt  : {status.poor_metrics_count}/4",
            f"  Is drifting      : {status.is_drifting}",
            f"  Note             : {status.note}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_sharpe(self, pnls: list[float]) -> float:
        """
        Annualizált Sharpe ratio a P&L sorozatból.

        Az annualizálás sqrt(252) faktorral történik, trade-ek
        napi frekvenciáját feltételezve.
        Returns 0.0 ha a szórás nulla.
        """
        if len(pnls) < 2:
            return 0.0
        n = len(pnls)
        mean = sum(pnls) / n
        # ddof=1 (minta-szórás), a pénzügyi Sharpe-számítás sztenderdje
        variance = sum((x - mean) ** 2 for x in pnls) / (n - 1)
        std = math.sqrt(variance)
        if std == 0.0:
            return 0.0
        # FIX #5: a config.annualization_factor adja a helyes évi skálázást
        return (mean / std) * math.sqrt(self.config.annualization_factor)

    def _compute_metrics(
        self, pnls: list[float]
    ) -> tuple[float, float, float, float]:
        """
        Kiszámolja a négy metrikát a P&L listából.

        Returns:
            (sharpe, win_rate, profit_factor, edge_ratio)
        """
        sharpe = self._compute_sharpe(pnls)

        winners = [p for p in pnls if p > 0.0]
        losers = [p for p in pnls if p < 0.0]

        n = len(pnls)
        win_rate = len(winners) / n if n > 0 else 0.0

        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (
            float("inf") if gross_profit > 0.0 else 0.0
        )

        avg_winner = sum(winners) / len(winners) if winners else 0.0
        avg_loser_abs = abs(sum(losers) / len(losers)) if losers else 0.0
        edge_ratio = avg_winner / avg_loser_abs if avg_loser_abs > 0.0 else (
            float("inf") if avg_winner > 0.0 else 0.0
        )

        return sharpe, win_rate, profit_factor, edge_ratio

    def _determine_action(self, poor_count: int) -> tuple[str, float]:
        """
        A poor metrikák száma alapján meghatározza az action-t
        és a méretszorzót.

        Returns:
            (action_str, size_multiplier)
        """
        cfg = self.config
        if poor_count >= cfg.stop_threshold:
            return "stop", 0.0
        if poor_count >= cfg.pause_threshold:
            return "pause", 0.0
        if poor_count >= cfg.reduce_threshold:
            return "reduce_size", 0.5
        if poor_count >= cfg.warn_threshold:
            return "warn", 1.0
        return "ok", 1.0
