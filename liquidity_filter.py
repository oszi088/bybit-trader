"""
liquidity_filter.py — Likviditás szűrő spot tradinghez

Három ellenőrzés:
  1. Volume: 24h forgalom >= min_volume_usd (pl. 500 000 USD)
  2. Impact: pozíció mérete <= max_volume_impact_pct × 24h volume (pl. 1%)
  3. Spread: (ask - bid) / mid_price <= max_spread_pct (pl. 0.20%)

Ha bármely ellenőrzés elbukik → a trade nem ajánlott.
Megj.: Spot piacon nincs short, ezért csak long belépéshez ellenőrzünk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("liquidity_filter")


# ============================================================================
# Adatstruktúrák
# ============================================================================

@dataclass
class LiquidityConfig:
    """Likviditási szűrő paraméterei.

    min_volume_usd: Minimális 24 órás forgalom USD-ben. Ez alatti symbol
                    nem likvid elegendően spot tradinghez.
    max_volume_impact_pct: A pozíció mérete maximum ennyi % lehet a 24 órás
                           forgalomhoz képest (pl. 0.01 = 1%).
    max_spread_pct: Maximálisan elfogadható spread (ask-bid)/mid, pl.
                    0.002 = 0.20%. Ennél tágabb spread esetén a market
                    orderek implicit slippage-je elfogadhatatlan.
    """
    min_volume_usd: float = 500_000.0        # minimális napi forgalom: 500 k USD
    max_volume_impact_pct: float = 0.01      # max pozícióméret: napi vol 1%-a
    max_spread_pct: float = 0.002            # max spread: 0.20%


@dataclass
class LiquidityCheckResult:
    """Egy likviditási ellenőrzés eredménye.

    passed: True, ha mindhárom feltétel teljesül.
    volume_ok: 24 órás forgalom elegendő-e.
    impact_ok: A pozícióméret nem mozgatja-e meg a piacot.
    spread_ok: A spread a megengedett határon belül van-e.
    volume_24h_usd: A tényleges 24 órás forgalom USD-ben.
    spread_pct: A tényleges spread % értéke ((ask-bid)/mid).
    max_allowed_size_usd: Maximálisan engedélyezett pozícióméret USD-ben.
    reason: Szöveges összefoglaló (ok vagy elutasítás oka).
    """
    passed: bool
    volume_ok: bool
    impact_ok: bool
    spread_ok: bool
    volume_24h_usd: float
    spread_pct: float
    max_allowed_size_usd: float
    reason: str


# ============================================================================
# Fő osztály
# ============================================================================

class LiquidityFilter:
    """Spot trading likviditási szűrő.

    Ellenőrzi, hogy egy adott symbol elegendően likvid-e egy adott méretű
    long belépéshez. Három feltétel mindegyikének teljesülnie kell.

    Használat::

        cfg = LiquidityConfig(min_volume_usd=1_000_000)
        lf = LiquidityFilter(cfg)

        # Gyors előzetes ellenőrzés (orderbook nélkül)
        ok, reason = lf.check_volume_only("BTCUSDT", volume_24h_usd=50_000_000)

        # Teljes ellenőrzés orderbook adatokkal
        result = lf.check(
            symbol="BTCUSDT",
            volume_24h_usd=50_000_000,
            bid=59_980.0,
            ask=60_020.0,
            order_size_usd=500.0,
        )
        if not result.passed:
            logger.warning("Likviditás nem megfelelő: %s", result.reason)
    """

    def __init__(self, config: LiquidityConfig | None = None) -> None:
        self.config = config or LiquidityConfig()

    # ------------------------------------------------------------------ #
    # Publikus API
    # ------------------------------------------------------------------ #

    def check(
        self,
        symbol: str,
        volume_24h_usd: float,
        bid: float,
        ask: float,
        order_size_usd: float,
    ) -> LiquidityCheckResult:
        """Teljes likviditási ellenőrzés orderbook adatokkal.

        Args:
            symbol: Trading szimbólum (pl. "BTCUSDT").
            volume_24h_usd: 24 órás forgalom USD-ben.
            bid: Legjobb vételi ajánlat (USD).
            ask: Legjobb eladási ajánlat (USD).
            order_size_usd: Tervezett pozícióméret USD-ben.

        Returns:
            LiquidityCheckResult a részletes eredménnyel.
        """
        cfg = self.config
        reasons: list[str] = []

        # --- 1. Volume ellenőrzés ---
        volume_ok = volume_24h_usd >= cfg.min_volume_usd
        if not volume_ok:
            reasons.append(
                f"volume_24h={volume_24h_usd:,.0f} USD < "
                f"min={cfg.min_volume_usd:,.0f} USD"
            )
            logger.debug("[%s] Volume nem elegendő: %.0f < %.0f USD",
                         symbol, volume_24h_usd, cfg.min_volume_usd)

        # --- 2. Volume-impact ellenőrzés ---
        max_allowed_size_usd = self.max_position_size_usd(volume_24h_usd)
        impact_ok = order_size_usd <= max_allowed_size_usd
        if not impact_ok:
            reasons.append(
                f"order_size={order_size_usd:,.2f} USD > "
                f"max_allowed={max_allowed_size_usd:,.2f} USD "
                f"({cfg.max_volume_impact_pct:.1%} × vol24h)"
            )
            logger.debug("[%s] Volume-impact túl nagy: %.2f > %.2f USD",
                         symbol, order_size_usd, max_allowed_size_usd)

        # --- 3. Spread ellenőrzés ---
        # mid_price = (bid + ask) / 2; spread = (ask - bid) / mid
        if bid > 0 and ask > 0 and ask >= bid:
            mid_price = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid_price if mid_price > 0 else 0.0
        else:
            # Érvénytelen orderbook adat → a legrosszabb esetet feltételezzük
            logger.warning("[%s] Érvénytelen bid/ask adat: bid=%.4f ask=%.4f",
                           symbol, bid, ask)
            spread_pct = cfg.max_spread_pct + 1.0   # biztosan elbukik

        spread_ok = spread_pct <= cfg.max_spread_pct
        if not spread_ok:
            reasons.append(
                f"spread={spread_pct:.4%} > max={cfg.max_spread_pct:.4%} "
                f"(bid={bid:.4f}, ask={ask:.4f})"
            )
            logger.debug("[%s] Spread túl tág: %.4f%% > %.4f%%",
                         symbol, spread_pct * 100, cfg.max_spread_pct * 100)

        # --- Összesítés ---
        passed = volume_ok and impact_ok and spread_ok

        if passed:
            reason = (
                f"ok: vol24h={volume_24h_usd:,.0f} USD, "
                f"spread={spread_pct:.4%}, "
                f"order={order_size_usd:,.2f}/{max_allowed_size_usd:,.2f} USD"
            )
        else:
            reason = "; ".join(reasons)

        logger.info("[%s] Likviditás ellenőrzés: passed=%s — %s", symbol, passed, reason)

        return LiquidityCheckResult(
            passed=passed,
            volume_ok=volume_ok,
            impact_ok=impact_ok,
            spread_ok=spread_ok,
            volume_24h_usd=volume_24h_usd,
            spread_pct=spread_pct,
            max_allowed_size_usd=max_allowed_size_usd,
            reason=reason,
        )

    def check_volume_only(
        self,
        symbol: str,
        volume_24h_usd: float,
    ) -> tuple[bool, str]:
        """Gyors előzetes ellenőrzés orderbook nélkül (csak volume).

        Hasznos, ha az orderbook lekérdezése drága (rate-limit), és
        először szűrni akarunk forgalom alapján.

        Args:
            symbol: Trading szimbólum.
            volume_24h_usd: 24 órás forgalom USD-ben.

        Returns:
            (True, "ok") ha a forgalom elegendő,
            (False, <indok>) ha nem.
        """
        cfg = self.config
        if volume_24h_usd >= cfg.min_volume_usd:
            logger.debug("[%s] Volume OK: %.0f USD >= %.0f USD",
                         symbol, volume_24h_usd, cfg.min_volume_usd)
            return True, "ok"

        reason = (
            f"volume_24h={volume_24h_usd:,.0f} USD < "
            f"min={cfg.min_volume_usd:,.0f} USD — symbol kihagyva"
        )
        logger.info("[%s] Volume előszűrő elutasítva: %s", symbol, reason)
        return False, reason

    def max_position_size_usd(self, volume_24h_usd: float) -> float:
        """Maximálisan engedélyezett pozícióméret USD-ben.

        A 24 órás forgalom adott hányadáig szabad belépni, hogy ne
        mozgassuk meg szignifikánsan a piacot.

        Args:
            volume_24h_usd: 24 órás forgalom USD-ben.

        Returns:
            Maximális pozícióméret USD-ben.
        """
        return volume_24h_usd * self.config.max_volume_impact_pct
