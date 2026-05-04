"""
altcoin_filter.py — Alt szezon validátor + halving-túlélő coinok szűrője

PROBLÉMA AMIT MEGOLD:
  A BTC dominancia csökkenése nem mindig jelent valódi alt szezont.
  „False altseason" okai:
    - BTC flash crash (dom esik, de alts is esnek)
    - Egyetlen szektor pump (pl. memecoins, DeFi micro-cap)
    - Makro risk-off (VIX spájk, tőzsde esés → crypto kiáramolás)
    - Rövid, 2-3 napos rotáció, ami visszafordul

MEGOLDÁS — 6 validációs kritérium:
  1. BTC > 200MA               (bull market kontextus kötelező)
  2. Dom < 48% ÉS csökkenő     (tényleges rotáció, nem outlier)
  3. Dom 14+ napja csökken      (sustained, nem flash)
  4. Dom napi esése < 3%        (nem BTC crash okozza)
  5. ETH/BTC arány emelkedő    (ETH vezet → broad rotation, nem 1 coin)
  6. VIX < 35                  (nincs makro pánik)

HA VALÓDI ALT SZEZON:
  Csak azok az altcoinok vásárolhatók, amelyek legalább egy Bitcoin
  halving-ot túléltek (launch date < legutóbbi halving napja).
  Ez kizárja az újabb, nem tesztelt tokeneket.

MARKETCAP TIER SZERINTI ELOSZTÁS:
  LARGE (~top 15):  a legbiztonságosabbak, legelőbb engedélyezve
  MID   (~rank 15–60): közepes kockázat, 14+ nap megerősítés után
  SMALL (~rank 60–200): kis cap, csak érett altseason + magas ML konfidencia

Használat:
  validator = AltseasonValidator(state_path="data/altseason_state.json")
  confirmed, reasons = validator.validate(ohlcv_btc, btc_dominance=44.5,
                                          eth_btc_ratio=0.065, vix=18.0)
  if confirmed:
      syms = validator.eligible_symbols(min_halvings=1, cap_tiers=["large","mid"])
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from market_cycle import HALVING_DATES

logger = logging.getLogger("altcoin_filter")


# =============================================================================
# Marketcap tier
# =============================================================================

class CapTier(str, Enum):
    LARGE = "large"   # ~top 15 by market cap, nagyon likvid
    MID   = "mid"     # ~rank 15–60
    SMALL = "small"   # ~rank 60–200


# =============================================================================
# Coin adatbázis
# =============================================================================
# Formátum: "SYM/USDT" → {"launch": date, "tier": CapTier}
#
# launch: a coin mainnet / kereskedési launch dátuma (CoinGecko alapján)
# tier:   jelenlegi marketcap besorolás (periodikusan frissítendő)
#
# Halving dátumok:
#   2012-11-28, 2016-07-09, 2020-05-11, 2024-04-20 (legutóbbi)
#
# „1 halving" = launch < 2024-04-20 (legutóbbi halving előtt volt)
# „2 halving" = launch < 2020-05-11
# „3 halving" = launch < 2016-07-09

COIN_DB: Dict[str, Dict] = {

    # ── LARGE CAP ────────────────────────────────────────────────────────────
    # Top ~15 / legmagasabb likviditás / legjobb kockázat/hozam ratio altseasonban

    "ETH/USDT":   {"launch": date(2015, 7,  30), "tier": CapTier.LARGE},
    "BNB/USDT":   {"launch": date(2017, 7,  1),  "tier": CapTier.LARGE},
    "SOL/USDT":   {"launch": date(2020, 3,  16), "tier": CapTier.LARGE},
    "XRP/USDT":   {"launch": date(2013, 1,  1),  "tier": CapTier.LARGE},
    "ADA/USDT":   {"launch": date(2017, 9,  29), "tier": CapTier.LARGE},
    "TRX/USDT":   {"launch": date(2017, 9,  1),  "tier": CapTier.LARGE},
    "DOGE/USDT":  {"launch": date(2013, 12, 8),  "tier": CapTier.LARGE},
    "AVAX/USDT":  {"launch": date(2020, 9,  21), "tier": CapTier.LARGE},
    "DOT/USDT":   {"launch": date(2020, 8,  19), "tier": CapTier.LARGE},
    "LINK/USDT":  {"launch": date(2017, 9,  16), "tier": CapTier.LARGE},
    "TON/USDT":   {"launch": date(2021, 11, 15), "tier": CapTier.LARGE},  # 2024 előtt
    "NEAR/USDT":  {"launch": date(2020, 10, 13), "tier": CapTier.LARGE},
    "POL/USDT":   {"launch": date(2019, 4,  26), "tier": CapTier.LARGE},  # MATIC→POL
    "MATIC/USDT": {"launch": date(2019, 4,  26), "tier": CapTier.LARGE},
    "UNI/USDT":   {"launch": date(2020, 9,  17), "tier": CapTier.LARGE},

    # ── MID CAP ──────────────────────────────────────────────────────────────
    # Rank ~15–60 / jó mozgás altseasonban, de kisebb likviditás

    "ATOM/USDT":  {"launch": date(2019, 3,  14), "tier": CapTier.MID},
    "LTC/USDT":   {"launch": date(2011, 10, 7),  "tier": CapTier.MID},
    "APT/USDT":   {"launch": date(2022, 10, 17), "tier": CapTier.MID},  # pre-2024 halving
    "ARB/USDT":   {"launch": date(2023, 3,  23), "tier": CapTier.MID},  # pre-2024 halving
    "OP/USDT":    {"launch": date(2022, 5,  31), "tier": CapTier.MID},  # pre-2024 halving
    "INJ/USDT":   {"launch": date(2020, 10, 26), "tier": CapTier.MID},
    "STX/USDT":   {"launch": date(2019, 10, 30), "tier": CapTier.MID},
    "HBAR/USDT":  {"launch": date(2019, 9,  16), "tier": CapTier.MID},
    "FIL/USDT":   {"launch": date(2020, 10, 15), "tier": CapTier.MID},
    "IMX/USDT":   {"launch": date(2021, 9,  9),  "tier": CapTier.MID},  # pre-2024
    "THETA/USDT": {"launch": date(2019, 3,  23), "tier": CapTier.MID},
    "VET/USDT":   {"launch": date(2018, 7,  30), "tier": CapTier.MID},
    "ALGO/USDT":  {"launch": date(2019, 6,  19), "tier": CapTier.MID},
    "EGLD/USDT":  {"launch": date(2020, 9,  4),  "tier": CapTier.MID},
    "XTZ/USDT":   {"launch": date(2018, 6,  30), "tier": CapTier.MID},
    "AAVE/USDT":  {"launch": date(2020, 10, 1),  "tier": CapTier.MID},
    "SAND/USDT":  {"launch": date(2020, 8,  14), "tier": CapTier.MID},
    "MANA/USDT":  {"launch": date(2017, 9,  15), "tier": CapTier.MID},
    "AXS/USDT":   {"launch": date(2020, 11, 4),  "tier": CapTier.MID},
    "MKR/USDT":   {"launch": date(2017, 12, 19), "tier": CapTier.MID},
    "GRT/USDT":   {"launch": date(2020, 12, 17), "tier": CapTier.MID},
    "FLOW/USDT":  {"launch": date(2021, 4,  6),  "tier": CapTier.MID},  # pre-2024
    "LDO/USDT":   {"launch": date(2021, 12, 20), "tier": CapTier.MID},  # pre-2024
    "SNX/USDT":   {"launch": date(2018, 3,  21), "tier": CapTier.MID},
    "CRV/USDT":   {"launch": date(2020, 8,  13), "tier": CapTier.MID},
    "KAVA/USDT":  {"launch": date(2019, 10, 24), "tier": CapTier.MID},
    "ENJ/USDT":   {"launch": date(2017, 11, 1),  "tier": CapTier.MID},

    # ── SMALL CAP ────────────────────────────────────────────────────────────
    # Rank ~60–200 / magas hozam potenciál, magas kockázat
    # Csak érett altseason + szigorú ML szűrő mellett

    "RUNE/USDT":  {"launch": date(2019, 7,  23), "tier": CapTier.SMALL},
    "ZIL/USDT":   {"launch": date(2018, 1,  25), "tier": CapTier.SMALL},
    "ONE/USDT":   {"launch": date(2019, 5,  30), "tier": CapTier.SMALL},
    "FTM/USDT":   {"launch": date(2018, 6,  15), "tier": CapTier.SMALL},
    "ROSE/USDT":  {"launch": date(2020, 11, 18), "tier": CapTier.SMALL},
    "BAT/USDT":   {"launch": date(2017, 5,  31), "tier": CapTier.SMALL},
    "LRC/USDT":   {"launch": date(2017, 8,  1),  "tier": CapTier.SMALL},
    "1INCH/USDT": {"launch": date(2020, 12, 25), "tier": CapTier.SMALL},
    "COMP/USDT":  {"launch": date(2020, 6,  15), "tier": CapTier.SMALL},
    "YFI/USDT":   {"launch": date(2020, 7,  17), "tier": CapTier.SMALL},
    "ONT/USDT":   {"launch": date(2018, 3,  8),  "tier": CapTier.SMALL},
    "ETC/USDT":   {"launch": date(2016, 7,  20), "tier": CapTier.SMALL},
    "XMR/USDT":   {"launch": date(2014, 4,  18), "tier": CapTier.SMALL},
    "DASH/USDT":  {"launch": date(2014, 1,  18), "tier": CapTier.SMALL},
    "NEO/USDT":   {"launch": date(2016, 9,  9),  "tier": CapTier.SMALL},
    "CFX/USDT":   {"launch": date(2021, 10, 15), "tier": CapTier.SMALL},  # pre-2024
}


# =============================================================================
# Előre számított listák halvingok szerint
# =============================================================================

def _halvings_survived(launch: date) -> int:
    """Hány halvingot élt túl a coin a launch dátum alapján."""
    return sum(1 for h in HALVING_DATES if launch < h <= date.today())


# Statikusan előre számolt (induláskor egyszeri)
HALVING_SURVIVORS: Dict[str, int] = {
    sym: _halvings_survived(info["launch"])
    for sym, info in COIN_DB.items()
}


def get_eligible_symbols(
    min_halvings: int = 1,
    cap_tiers: Optional[List[CapTier]] = None,
    exclude_btc: bool = True,
) -> List[str]:
    """
    Altseason-ra jogosult coinok listája.

    Paraméterek:
        min_halvings: legalább ennyi halvingot kell túlélni (1 = pre-2024)
        cap_tiers:    melyik tier-ek megengedettek (None = mind)
        exclude_btc:  BTC-t kizárjuk (ő nem altcoin)
    """
    allowed_tiers: Set[CapTier] = (
        set(cap_tiers) if cap_tiers else {CapTier.LARGE, CapTier.MID, CapTier.SMALL}
    )
    result = []
    for sym, info in COIN_DB.items():
        if exclude_btc and sym == "BTC/USDT":
            continue
        if HALVING_SURVIVORS.get(sym, 0) < min_halvings:
            continue
        if info["tier"] not in allowed_tiers:
            continue
        result.append(sym)
    return sorted(result)


# =============================================================================
# AltseasonValidator
# =============================================================================

@dataclass
class ValidationResult:
    confirmed: bool
    score: int                    # hány kritériumot teljesített (max 6)
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    eligible_large: List[str] = field(default_factory=list)
    eligible_mid: List[str] = field(default_factory=list)
    eligible_small: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "✅ VALÓDI" if self.confirmed else "❌ FALSE"
        passed_str = ", ".join(self.passed) if self.passed else "—"
        failed_str = ", ".join(self.failed) if self.failed else "—"
        return (
            f"Alt Szezon: {status} ({self.score}/6 kritérium)\n"
            f"  ✓ {passed_str}\n"
            f"  ✗ {failed_str}"
        )


class AltseasonValidator:
    """
    Valódi vs. hamis altseason megkülönböztető.

    6 kritérium MIND teljesítése esetén valódi az altseason.
    Bármelyik hiánya → false altseason → BTC/USDT-ban maradunk.

    Állapot perzisztálás: JSON fájlban tárolja a dominancia historikumot,
    hogy az újraindítás után is számolható legyen a 14 napos trend.
    """

    REQUIRED_SCORE = 5          # legalább 5/6 kritérium kell
    MIN_DOM_FALLING_DAYS = 14   # legalább 14 napja csökkenő dominancia
    MAX_DOM_1D_DROP = 3.0       # > 3% egynapi esés → BTC crash gyanú
    DOM_THRESHOLD = 48.0        # dominancia <= ez alatt számít rotációnak
    VIX_PANIC_THRESHOLD = 35.0

    def __init__(self, state_path: Optional[str] = None):
        self._state_path = state_path
        # [(date, dominance)] — max 60 napos rolling buffer
        self._dom_history: List[Tuple[date, float]] = []
        # [(date, eth_btc_ratio)] — ETH/BTC historikum a trend számításhoz
        self._eth_btc_history: List[Tuple[date, float]] = []
        self._load_state()

    # ------------------------------------------------------------------ #
    # Fő validáció
    # ------------------------------------------------------------------ #

    def validate(
        self,
        ohlcv_btc: pd.DataFrame,
        btc_dominance: float,
        eth_btc_ratio: Optional[float] = None,
        vix: float = 20.0,
        today: Optional[date] = None,
    ) -> ValidationResult:
        """
        Megvizsgálja, hogy valódi alt szezon van-e.

        Paraméterek:
            ohlcv_btc:     BTC OHLCV (close kell a 200MA-hoz)
            btc_dominance: aktuális BTC dominancia % (pl. 45.3)
            eth_btc_ratio: ETH/BTC árfolyam (pl. 0.065); None = skip
            vix:           VIX index értéke; None = skip
            today:         aktuális dátum (None = date.today())
        """
        today = today or date.today()
        self._update_dom_history(today, btc_dominance)

        passed: List[str] = []
        failed: List[str] = []

        # ── 1. BTC > 200MA (bull market kontextus) ───────────────────────
        close = ohlcv_btc["close"].astype(float)
        ma200 = close.rolling(200, min_periods=50).mean().iloc[-1]
        if close.iloc[-1] > ma200:
            passed.append("BTC>200MA")
        else:
            failed.append("BTC<200MA (bear market)")

        # ── 2. Dominancia <= küszöb ───────────────────────────────────────
        if btc_dominance <= self.DOM_THRESHOLD:
            passed.append(f"Dom={btc_dominance:.1f}%<={self.DOM_THRESHOLD}%")
        else:
            failed.append(f"Dom={btc_dominance:.1f}%>{self.DOM_THRESHOLD}% (BTC még domináns)")

        # ── 3. Dom 14+ napja csökken (sustained) ─────────────────────────
        falling_days = self._dom_falling_days(today)
        if falling_days >= self.MIN_DOM_FALLING_DAYS:
            passed.append(f"Dom {falling_days}d csökken (≥{self.MIN_DOM_FALLING_DAYS})")
        else:
            failed.append(f"Dom csak {falling_days}d csökken (kell ≥{self.MIN_DOM_FALLING_DAYS})")

        # ── 4. Nincs egynapi nagy dom-esés (BTC crash gyanú) ─────────────
        daily_drop = self._max_daily_dom_drop(today, lookback=3)
        if daily_drop < self.MAX_DOM_1D_DROP:
            passed.append(f"Dom napi max esés {daily_drop:.1f}%<{self.MAX_DOM_1D_DROP}%")
        else:
            failed.append(f"Dom {daily_drop:.1f}% esett 1 nap alatt → BTC crash gyanú")

        # ── 5. ETH/BTC emelkedő (broad rotation, nem 1 coin) ─────────────
        if eth_btc_ratio is not None:
            self._update_eth_btc_history(today, eth_btc_ratio)
            eth_trend = self._eth_btc_trend()
            if eth_trend >= 0:
                passed.append("ETH/BTC↑ (vezet az altseason)")
            else:
                failed.append("ETH/BTC↓ (nem broad rotation)")
        else:
            # Ha nincs adat, nem büntetjük, de nem is számítjuk
            passed.append("ETH/BTC: n/a (kihagyva)")

        # ── 6. Nincs makro pánik (VIX) ───────────────────────────────────
        if vix < self.VIX_PANIC_THRESHOLD:
            passed.append(f"VIX={vix:.0f}<{self.VIX_PANIC_THRESHOLD}")
        else:
            failed.append(f"VIX={vix:.0f}≥{self.VIX_PANIC_THRESHOLD} (pánik)")

        score = len(passed)
        # REQUIRED_SCORE = 5 → legalább 5/6 kritérium teljesül
        # Az `and len(failed) == 0` feltétel REQUIRED_SCORE-t hatástalanná tette
        # (failed=0 → score=6 mindig), ezért csak score >= REQUIRED_SCORE kell
        confirmed = score >= self.REQUIRED_SCORE

        # Ha megerősített, számítsuk ki az elérhető coinokat
        large, mid, small = [], [], []
        if confirmed:
            days_in = falling_days  # proxy az altseason korára
            large = get_eligible_symbols(min_halvings=1, cap_tiers=[CapTier.LARGE])
            if days_in >= 14:
                mid = get_eligible_symbols(min_halvings=1, cap_tiers=[CapTier.MID])
            if days_in >= 30:
                small = get_eligible_symbols(min_halvings=1, cap_tiers=[CapTier.SMALL])

        result = ValidationResult(
            confirmed=confirmed,
            score=score,
            passed=passed,
            failed=failed,
            eligible_large=large,
            eligible_mid=mid,
            eligible_small=small,
        )

        if confirmed:
            logger.info("Valódi altseason megerősítve: %s", result)
        else:
            logger.info("False altseason kiszűrve: %s", result)

        return result

    # ------------------------------------------------------------------ #
    # Dominancia historikum kezelés
    # ------------------------------------------------------------------ #

    def _update_dom_history(self, today: date, dom: float) -> None:
        """Frissíti a rolling buffert és elmenti a state-t."""
        # Ha van már bejegyzés mára, frissítjük
        self._dom_history = [(d, v) for d, v in self._dom_history if d != today]
        self._dom_history.append((today, dom))
        # Max 60 napos ablak
        cutoff = today - timedelta(days=60)
        self._dom_history = [(d, v) for d, v in self._dom_history if d >= cutoff]
        self._dom_history.sort(key=lambda x: x[0])
        self._save_state()

    def _dom_falling_days(self, today: date) -> int:
        """
        Hány egymást követő napon csökkent a dominancia?
        Visszaszámol a mai naptól.
        """
        if len(self._dom_history) < 2:
            return 0
        sorted_hist = sorted(self._dom_history, key=lambda x: x[0], reverse=True)
        count = 0
        for i in range(len(sorted_hist) - 1):
            curr_dom = sorted_hist[i][1]
            prev_dom = sorted_hist[i + 1][1]
            if curr_dom < prev_dom:
                count += 1
            else:
                break
        return count

    def _max_daily_dom_drop(self, today: date, lookback: int = 3) -> float:
        """Max egynapi dominancia esés az utóbbi N napban."""
        cutoff = today - timedelta(days=lookback)
        recent = [(d, v) for d, v in self._dom_history if d >= cutoff]
        if len(recent) < 2:
            return 0.0
        recent.sort(key=lambda x: x[0])
        drops = [max(0.0, recent[i][1] - recent[i + 1][1])
                 for i in range(len(recent) - 1)]
        return max(drops) if drops else 0.0

    def _update_eth_btc_history(self, today: date, ratio: float) -> None:
        """Frissíti az ETH/BTC rolling buffert (max 30 nap)."""
        self._eth_btc_history = [(d, v) for d, v in self._eth_btc_history if d != today]
        self._eth_btc_history.append((today, ratio))
        cutoff = today - timedelta(days=30)
        self._eth_btc_history = [(d, v) for d, v in self._eth_btc_history if d >= cutoff]
        self._eth_btc_history.sort(key=lambda x: x[0])
        self._save_state()

    def _eth_btc_trend(self) -> float:
        """
        ETH/BTC 7 napos trend: (aktuális - 7 nappal ezelőtti) / régebbi.
        Pozitív = emelkedő, negatív = csökkenő, 0.0 = nincs elég adat.

        Adat nélkül (első néhány hívás) semleges 0.0-t ad (nem büntet,
        de nem is igazol) — a validate() 'n/a' ágába kerül ilyenkor.
        """
        if len(self._eth_btc_history) < 2:
            return 0.0   # nincs elég historikum → semleges
        sorted_hist = sorted(self._eth_btc_history, key=lambda x: x[0])
        current = sorted_hist[-1][1]
        # Kb. 7 napos visszatekintés
        ref_date = sorted_hist[-1][0] - timedelta(days=7)
        older = next(
            (v for d, v in reversed(sorted_hist[:-1]) if d <= ref_date),
            sorted_hist[0][1],   # ha nincs 7 nap adat, a legrégebbit vesszük
        )
        if older <= 0:
            return 0.0
        return (current - older) / older   # pozitív = ETH/BTC emelkedett

    # ------------------------------------------------------------------ #
    # JSON perzisztálás
    # ------------------------------------------------------------------ #

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            data = {
                "dom_history": [
                    {"date": d.isoformat(), "dom": v}
                    for d, v in self._dom_history
                ],
                "eth_btc_history": [
                    {"date": d.isoformat(), "ratio": v}
                    for d, v in self._eth_btc_history
                ],
            }
            Path(self._state_path).write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as e:
            logger.warning("Altseason állapot mentése sikertelen: %s", e)

    def _load_state(self) -> None:
        if not self._state_path:
            return
        try:
            raw = json.loads(
                Path(self._state_path).read_text(encoding="utf-8")
            )
            self._dom_history = [
                (date.fromisoformat(e["date"]), float(e["dom"]))
                for e in raw.get("dom_history", [])
            ]
            self._eth_btc_history = [
                (date.fromisoformat(e["date"]), float(e["ratio"]))
                for e in raw.get("eth_btc_history", [])
            ]
            logger.info("Altseason állapot betöltve: %d nap dom, %d nap ETH/BTC",
                        len(self._dom_history), len(self._eth_btc_history))
        except (OSError, KeyError, ValueError):
            pass


# =============================================================================
# Kényelmi függvények
# =============================================================================

def describe_coin(symbol: str) -> str:
    """Coin rövid leírása (tier + halvings)."""
    info = COIN_DB.get(symbol)
    if not info:
        return f"{symbol}: ismeretlen"
    halvings = HALVING_SURVIVORS.get(symbol, 0)
    return (
        f"{symbol}: {info['tier'].value.upper()} cap | "
        f"launch={info['launch']} | {halvings} halving túlélve"
    )


def altseason_summary(result: ValidationResult) -> str:
    """Telegram-barát összefoglaló."""
    status = "✅ VALÓDI ALT SZEZON" if result.confirmed else "⚠️ HAMIS ALT SZEZON"
    lines = [f"📊 {status} ({result.score}/6 kritérium)"]
    for p in result.passed:
        lines.append(f"   ✅ {p}")
    for f in result.failed:
        lines.append(f"   ❌ {f}")
    if result.confirmed:
        lines.append(f"\n🟢 Large cap ({len(result.eligible_large)}): "
                     + ", ".join(s.split("/")[0] for s in result.eligible_large[:8]) + "…")
        if result.eligible_mid:
            lines.append(f"🟡 Mid cap ({len(result.eligible_mid)}): "
                         + ", ".join(s.split("/")[0] for s in result.eligible_mid[:6]) + "…")
        if result.eligible_small:
            lines.append(f"🔴 Small cap ({len(result.eligible_small)}): "
                         + ", ".join(s.split("/")[0] for s in result.eligible_small[:4]) + "…")
    return "\n".join(lines)
