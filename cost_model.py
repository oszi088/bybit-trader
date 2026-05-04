"""
cost_model.py — Spot kereskedési költség modell

Spot-only trading: vételi + eladási díj + slippage = összköltség.
Csak akkor érdemes belépni, ha a várható hozam > összköltség × szorzó.

Bybit spot díjak (2024):
  - VIP0 maker: 0.10%,  taker: 0.10%
  - VIP1 maker: 0.08%,  taker: 0.10%
  → konzervatív becslés: 0.10% mindkét irányban = 0.20% körforgás

Slippage becslés:
  - kis megbízás (< 1000 USD):  ~0.03%
  - közepes (1000–5000 USD):    ~0.05% + order_size/volume × 0.5
  - nagy (> 5000 USD):          ~0.10% + order_size/volume × 1.0
  ahol a volume_impact az ATR%-hoz is igazodik
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("cost_model")


# ============================================================================
# Adatstruktúrák
# ============================================================================

@dataclass
class CostConfig:
    """Költségmodell paraméterei.

    fee_rate: Bybit taker díj (0.10% = 0.001)
    slippage_small_usd: Ez alatt a megbízás mérete "kis" kategória
    slippage_base_pct: Alap slippage (kis megbízásoknál)
    slippage_impact_pct: Volume-impact szorzó a slippage kiszámításakor
    min_return_multiplier: A várható hozamnak legalább ennyiszer kell
                           felülmúlnia az összköltséget
    """
    fee_rate: float = 0.001                 # 0.10% taker
    slippage_small_usd: float = 1_000.0     # küszöb: kis vs. közepes megbízás
    slippage_base_pct: float = 0.0003       # alap slippage: 0.03%
    slippage_impact_pct: float = 0.005      # volume-impact szorzó
    min_return_multiplier: float = 1.5      # min_return = total_cost × 1.5


@dataclass
class CostBreakdown:
    """Egyetlen trade teljes költséglebontása.

    Minden érték százalékban (0.01 = 1%).
    is_worth_entering: True, ha expected_return_pct >= min_return_pct.
    """
    entry_fee_pct: float          # vételi díj %
    exit_fee_pct: float           # eladási díj %
    slippage_entry_pct: float     # belépési slippage %
    slippage_exit_pct: float      # kilépési slippage %
    total_cost_pct: float         # összköltség % (mindkét irány)
    break_even_pct: float         # fedezeti pont: ennyi hozam kell hogy nullán legyünk
    min_return_pct: float         # ajánlott minimális hozam (total_cost × szorzó)
    is_worth_entering: bool       # érdemes-e belépni?
    note: str = ""                # szöveges megjegyzés, ok vagy elutasítás oka


# ============================================================================
# Fő osztály
# ============================================================================

class CostModel:
    """Spot trading tranzakciós költség becslő.

    Kiszámítja a belépési/kilépési díjakat és a slippage-t, majd eldönti,
    hogy a várható hozam elegendő-e a belépéshez.

    Használat::

        cfg = CostConfig()
        model = CostModel(cfg)
        breakdown = model.calculate(
            price=60_000,
            atr=1_200,
            order_size_usd=500,
            volume_24h_usd=50_000_000,
            expected_return_pct=0.005,   # 0.5% várható hozam
        )
        ok, reason = model.filter_trade(...)
    """

    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or CostConfig()

    # ------------------------------------------------------------------ #
    # Slippage becslés
    # ------------------------------------------------------------------ #

    def estimate_slippage(
        self,
        order_size_usd: float,
        volume_24h_usd: float,
        atr_pct: float,
    ) -> float:
        """Egy irányú slippage becslése (belépés VAGY kilépés).

        Formulá:
            slippage = base_slippage + (order_size / volume) × impact_pct × (1 + atr_pct×10)

        A volume_impact az ATR%-hoz igazodik: nagyobb volatilitásnál
        az orderbook mélyebb, de a spread is tágabb → a szorzó nő.

        Args:
            order_size_usd: Megbízás mérete USD-ben.
            volume_24h_usd: 24 órás forgalom USD-ben (nullánál biztonságosan kezeljük).
            atr_pct: ATR / ár arány (pl. 0.02 = 2%).

        Returns:
            Slippage % (0.0001 – 0.01 közé clampolva).
        """
        cfg = self.config

        # Nullás forgalom esetén a legrosszabb esetet feltételezzük
        if volume_24h_usd <= 0:
            logger.warning("volume_24h_usd <= 0, maximális slippage-t alkalmazunk.")
            return 0.01

        # Volume-impact: mekkora a megbízás a napi forgalom arányában
        volume_impact = (order_size_usd / volume_24h_usd) * cfg.slippage_impact_pct

        # ATR-korrekció: volatilisabb piacon a slippage nagyobb
        atr_factor = 1.0 + atr_pct * 10.0

        slippage = cfg.slippage_base_pct + volume_impact * atr_factor

        # Értelmesre clampolás: 0.01% – 1.00%
        clamped = max(0.0001, min(0.01, slippage))

        logger.debug(
            "Slippage becslés: order=%.2f USD, vol24h=%.0f USD, atr_pct=%.4f "
            "→ raw=%.6f → clamped=%.6f",
            order_size_usd, volume_24h_usd, atr_pct, slippage, clamped,
        )
        return clamped

    # ------------------------------------------------------------------ #
    # Teljes költség kalkuláció
    # ------------------------------------------------------------------ #

    def calculate(
        self,
        price: float,
        atr: float,
        order_size_usd: float,
        volume_24h_usd: float = 1_000_000.0,
        expected_return_pct: float = 0.0,
    ) -> CostBreakdown:
        """Teljes körű költséglebontás egy spot trade-hez.

        Args:
            price: Aktuális ár (pl. BTCUSDT = 60 000).
            atr: Átlagos valódi tartomány (ugyanolyan egységben mint price).
            order_size_usd: Tervezett megbízás mérete USD-ben.
            volume_24h_usd: 24 órás forgalom USD-ben (default: 1 M USD).
            expected_return_pct: Várt hozam %-ban (pl. 0.005 = 0.5%).

        Returns:
            CostBreakdown dataclass az összes részlettel.
        """
        cfg = self.config

        # ATR relatív értéke az árhoz képest
        atr_pct = atr / price if price > 0 else 0.0

        # Díjak (mindkét irány)
        entry_fee_pct = cfg.fee_rate       # belépési taker díj
        exit_fee_pct = cfg.fee_rate        # kilépési taker díj

        # Slippage mindkét oldalra
        slippage_entry = self.estimate_slippage(order_size_usd, volume_24h_usd, atr_pct)
        slippage_exit = self.estimate_slippage(order_size_usd, volume_24h_usd, atr_pct)

        # Összköltség: belépési díj + kilépési díj + mindkét oldali slippage
        total_cost_pct = entry_fee_pct + exit_fee_pct + slippage_entry + slippage_exit

        # Fedezeti pont (break-even): ennyi hozam kell hogy ne veszítsünk
        break_even_pct = total_cost_pct

        # Ajánlott minimum hozam (total_cost × szorzó)
        min_return_pct = total_cost_pct * cfg.min_return_multiplier

        # Döntés: érdemes-e belépni?
        is_worth = expected_return_pct >= min_return_pct

        if expected_return_pct <= 0.0:
            note = (
                f"Nincs várható hozam megadva. Összköltség={total_cost_pct:.4%}, "
                f"min_return={min_return_pct:.4%}"
            )
        elif is_worth:
            note = (
                f"OK. Várható hozam {expected_return_pct:.4%} >= "
                f"min_return {min_return_pct:.4%} "
                f"(összköltség {total_cost_pct:.4%} × {cfg.min_return_multiplier})"
            )
        else:
            note = (
                f"Nem éri meg belépni: várható hozam {expected_return_pct:.4%} < "
                f"min_return {min_return_pct:.4%} "
                f"(összköltség {total_cost_pct:.4%} × {cfg.min_return_multiplier})"
            )

        logger.debug(
            "CostBreakdown: entry_fee=%.4f%% exit_fee=%.4f%% "
            "slip_in=%.4f%% slip_out=%.4f%% total=%.4f%% "
            "break_even=%.4f%% min_return=%.4f%% worth=%s",
            entry_fee_pct * 100, exit_fee_pct * 100,
            slippage_entry * 100, slippage_exit * 100,
            total_cost_pct * 100, break_even_pct * 100,
            min_return_pct * 100, is_worth,
        )

        return CostBreakdown(
            entry_fee_pct=entry_fee_pct,
            exit_fee_pct=exit_fee_pct,
            slippage_entry_pct=slippage_entry,
            slippage_exit_pct=slippage_exit,
            total_cost_pct=total_cost_pct,
            break_even_pct=break_even_pct,
            min_return_pct=min_return_pct,
            is_worth_entering=is_worth,
            note=note,
        )

    # ------------------------------------------------------------------ #
    # Gyors szűrő
    # ------------------------------------------------------------------ #

    def filter_trade(
        self,
        price: float,
        atr: float,
        order_size_usd: float,
        volume_24h_usd: float = 1_000_000.0,
        expected_return_pct: float = 0.0,
    ) -> tuple[bool, str]:
        """Gyors igen/nem döntés a várható hozam és az összköltség alapján.

        Args:
            price: Aktuális ár.
            atr: Átlagos valódi tartomány (ár egységben).
            order_size_usd: Megbízás mérete USD-ben.
            volume_24h_usd: 24 órás forgalom USD-ben.
            expected_return_pct: Várható hozam %-ban.

        Returns:
            (True, "ok") ha érdemes belépni,
            (False, <indok szöveg számokkal>) ha nem.
        """
        bd = self.calculate(
            price=price,
            atr=atr,
            order_size_usd=order_size_usd,
            volume_24h_usd=volume_24h_usd,
            expected_return_pct=expected_return_pct,
        )

        if bd.is_worth_entering:
            return True, "ok"

        # Részletes elutasítási indok
        reason = (
            f"cost_filter: várható={expected_return_pct:.4%} < "
            f"min_return={bd.min_return_pct:.4%} "
            f"[összköltség={bd.total_cost_pct:.4%}: "
            f"díjak={bd.entry_fee_pct + bd.exit_fee_pct:.4%}, "
            f"slippage={bd.slippage_entry_pct + bd.slippage_exit_pct:.4%}]"
        )
        logger.info("Trade kiszűrve: %s", reason)
        return False, reason
