"""
market_cycle.py — Bitcoin/Crypto piaci ciklus detekció

A kripto piac jól ismert, visszatérő fázisokban mozog, amelyeket
elsősorban a Bitcoin halving ciklusa (~4 év), a makro likviditás,
és a retail sentiment befolyásol.

Ez a modul:
  1. 7 normalizált mutatót számít (0.0–1.0 skálán)
  2. Minden ismert ciklushoz illeszkedési pontszámot számol
  3. Visszaadja a legjobban illeszkedő ciklust + konfidenciát
  4. Megbecsüli a ciklus hátralévő hosszát historikus átlagok alapján
  5. JSON-ban perzisztálja az állapotot session-ok között

Detektált ciklusok:
  ACCUMULATION  — alap formáció, alacsony vol, okos pénz vásárol
  BULL_EARLY    — kitörés a 200MA fölé, BTC vezet, alts alszanak
  BULL_MID      — erős uptrend, altcoin szezon indul
  BULL_LATE     — parabolikus, extrém greed, funding rate csúcson
  DISTRIBUTION  — topp formáció, choppy, nagy kezek adnak el
  BEAR_EARLY    — gyors esés, pánik, high vol capitulation
  BEAR_MID      — lassú grinding, alacsony vol, érdektelenség
  ALTSEASON     — BTC dom. zuhan, alts felülteljesítenek
  RISK_OFF      — makro sokk (VIX spájk, exchange hack, szabályozói hír)

Használat:
  detector = MarketCycleDetector()
  state = detector.detect(ohlcv_btc, fg_value=62, funding_rate=0.0003, btc_dominance=52.0)
  print(state.cycle, state.days_remaining_est)
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("market_cycle")

# =============================================================================
# Bitcoin halving dátumok (UTC)
# =============================================================================
HALVING_DATES: List[date] = [
    date(2012, 11, 28),
    date(2016, 7,  9),
    date(2020, 5,  11),
    date(2024, 4,  20),
    date(2028, 4,  17),   # becsült (~210 000 blokk ~4 év)
]
HALVING_CYCLE_DAYS = 1461   # ~4 év


# =============================================================================
# Ciklus definíciók
# =============================================================================

class MarketCycle(str, Enum):
    ACCUMULATION = "accumulation"   # alap formáció
    BULL_EARLY   = "bull_early"     # kitörés
    BULL_MID     = "bull_mid"       # erős uptrend
    BULL_LATE    = "bull_late"      # parabolikus
    DISTRIBUTION = "distribution"   # topp formáció
    BEAR_EARLY   = "bear_early"     # gyors esés
    BEAR_MID     = "bear_mid"       # lassú grinding
    ALTSEASON    = "altseason"      # alt eufória
    RISK_OFF     = "risk_off"       # makro sokk


# Historikus ciklus időtartamok (min_nap, max_nap, átlag_nap)
CYCLE_DURATIONS: Dict[MarketCycle, Tuple[int, int, int]] = {
    MarketCycle.ACCUMULATION: (90,  365, 180),
    MarketCycle.BULL_EARLY:   (30,  120,  75),
    MarketCycle.BULL_MID:     (90,  240, 150),
    MarketCycle.BULL_LATE:    (14,   90,  45),
    MarketCycle.DISTRIBUTION: (30,  120,  75),
    MarketCycle.BEAR_EARLY:   (30,  120,  75),
    MarketCycle.BEAR_MID:     (90,  420, 210),
    MarketCycle.ALTSEASON:    (21,   90,  45),
    MarketCycle.RISK_OFF:     ( 3,   30,  14),
}

# Tipikus következő ciklus
NEXT_CYCLE: Dict[MarketCycle, MarketCycle] = {
    MarketCycle.ACCUMULATION: MarketCycle.BULL_EARLY,
    MarketCycle.BULL_EARLY:   MarketCycle.BULL_MID,
    MarketCycle.BULL_MID:     MarketCycle.BULL_LATE,
    MarketCycle.BULL_LATE:    MarketCycle.DISTRIBUTION,
    MarketCycle.DISTRIBUTION: MarketCycle.BEAR_EARLY,
    MarketCycle.BEAR_EARLY:   MarketCycle.BEAR_MID,
    MarketCycle.BEAR_MID:     MarketCycle.ACCUMULATION,
    MarketCycle.ALTSEASON:    MarketCycle.DISTRIBUTION,
    MarketCycle.RISK_OFF:     MarketCycle.BEAR_EARLY,
}


# =============================================================================
# Ciklus profilok (várt normalizált értékek, 0.0–1.0 skálán)
# Formátum: indicator -> (expected_value, weight)
# =============================================================================
# Mutatók leírása:
#   btc_200ma  — price/MA200 normalizálva: 0.7x = 0.0, 1.3x = 1.0 (0.5 = at MA)
#   fg_30d     — F&G 30 napos átlag / 100
#   vol_pct    — 14 napos realized vol percentilis az utóbbi 1 évben
#   halving    — hol tartunk a 4 éves ciklusban (0=halving után azonnal, 1=következő halving)
#   funding    — funding rate normalizálva: -0.2%/8h=0.0, 0%=0.5, +0.2%/8h=1.0
#   btc_dom    — BTC dominancia / 100
#   mom_90d    — 90 napos BTC price change, tanh normalizálva (−50%=0, +100%=1)

CYCLE_PROFILES: Dict[MarketCycle, Dict[str, Tuple[float, float]]] = {
    MarketCycle.ACCUMULATION: {
        "btc_200ma": (0.38, 2.5),   # kissé a 200MA alatt
        "fg_30d":    (0.25, 2.0),   # fear zone
        "vol_pct":   (0.25, 1.5),   # alacsony vol
        "halving":   (0.55, 1.0),   # ~2-3 évvel halving után (bear vége)
        "funding":   (0.45, 1.5),   # enyhén negatív
        "btc_dom":   (0.58, 1.0),   # BTC dom. magas
        "mom_90d":   (0.35, 1.5),   # negatív / lapos momentum
    },
    MarketCycle.BULL_EARLY: {
        "btc_200ma": (0.58, 2.5),   # épp a 200MA fölé tört
        "fg_30d":    (0.50, 2.0),   # neutral→greed
        "vol_pct":   (0.55, 1.5),   # növekvő vol
        "halving":   (0.10, 1.5),   # 3-12 hónappal halving után
        "funding":   (0.54, 1.5),   # enyhe positive
        "btc_dom":   (0.60, 1.5),   # BTC dom. magas (alts még nem mozognak)
        "mom_90d":   (0.60, 1.5),   # erős pozitív momentum
    },
    MarketCycle.BULL_MID: {
        "btc_200ma": (0.67, 2.0),   # jól a 200MA felett
        "fg_30d":    (0.65, 2.0),   # greed
        "vol_pct":   (0.60, 1.5),   # közepes-magas vol
        "halving":   (0.25, 1.5),   # 12-18 hónappal halving után
        "funding":   (0.60, 1.5),   # mérsékelt pozitív
        "btc_dom":   (0.52, 1.5),   # dom. csökkeni kezd (alts indulnak)
        "mom_90d":   (0.72, 2.0),   # nagyon erős momentum
    },
    MarketCycle.BULL_LATE: {
        "btc_200ma": (0.83, 2.0),   # jóval a 200MA felett (parabolikus)
        "fg_30d":    (0.85, 2.5),   # extreme greed
        "vol_pct":   (0.70, 1.5),   # magas vol
        "halving":   (0.35, 1.0),   # ~18 hónappal halving után
        "funding":   (0.78, 2.0),   # magas funding (crowded longs)
        "btc_dom":   (0.43, 1.5),   # dom. alacsony (alts eufória)
        "mom_90d":   (0.88, 2.0),   # extrém momentum
    },
    MarketCycle.DISTRIBUTION: {
        "btc_200ma": (0.70, 2.0),   # még a 200MA felett, de csökken
        "fg_30d":    (0.58, 2.0),   # greed→neutral (esik)
        "vol_pct":   (0.75, 2.0),   # magas és variable vol
        "halving":   (0.42, 1.0),
        "funding":   (0.62, 1.5),   # emelkedett de csökkenő
        "btc_dom":   (0.43, 1.0),   # alt eufória csúcs
        "mom_90d":   (0.42, 2.0),   # csökkenő momentum
    },
    MarketCycle.BEAR_EARLY: {
        "btc_200ma": (0.35, 2.5),   # 200MA alá törés
        "fg_30d":    (0.28, 2.5),   # fear
        "vol_pct":   (0.88, 2.0),   # extrém magas vol (pánik)
        "halving":   (0.55, 0.5),   # ~2 évvel halving után
        "funding":   (0.35, 1.5),   # negatív (short nyomás)
        "btc_dom":   (0.50, 1.0),   # dom. visszajön
        "mom_90d":   (0.20, 2.0),   # erős negatív momentum
    },
    MarketCycle.BEAR_MID: {
        "btc_200ma": (0.18, 2.5),   # jóval a 200MA alatt
        "fg_30d":    (0.18, 2.5),   # extreme fear
        "vol_pct":   (0.20, 2.0),   # alacsony vol (unalom)
        "halving":   (0.68, 1.5),   # ~2.5-3 évvel halving után
        "funding":   (0.42, 1.5),   # enyhén negatív
        "btc_dom":   (0.58, 1.5),   # dom. magas
        "mom_90d":   (0.15, 2.0),   # lapos/negatív
    },
    MarketCycle.ALTSEASON: {
        "btc_200ma": (0.72, 1.5),   # bull market közepén
        "fg_30d":    (0.82, 2.0),   # extreme greed
        "vol_pct":   (0.65, 1.5),
        "halving":   (0.38, 0.5),
        "funding":   (0.72, 2.0),   # magas (alts túlzsúfolt)
        "btc_dom":   (0.36, 3.0),   # NAGYON alacsony BTC dom — fő jel!
        "mom_90d":   (0.78, 1.5),
    },
    MarketCycle.RISK_OFF: {
        "btc_200ma": (0.33, 1.5),   # hirtelen esés
        "fg_30d":    (0.12, 3.0),   # extreme fear — fő jel!
        "vol_pct":   (0.95, 3.0),   # extrém vol spájk — fő jel!
        "halving":   (0.50, 0.0),   # nem releváns
        "funding":   (0.28, 2.0),   # negatív (shorting)
        "btc_dom":   (0.55, 1.0),
        "mom_90d":   (0.10, 2.0),   # crash
    },
}


# =============================================================================
# CycleState
# =============================================================================

@dataclass
class CycleState:
    cycle: MarketCycle
    confidence: float           # 0.0–1.0 (illeszkedési bizonyosság)
    days_in_cycle: int          # hány napja vagyunk ebben a ciklusban
    days_remaining_est: int     # becsült hátralévő napok (historikus átlag alapján)
    cycle_completion: float     # 0.0–1.0 (hol tartunk a cikluson belül)
    next_cycle: MarketCycle     # várható következő fázis
    halving_phase: float        # 0.0–1.0 (hol tartunk a 4 éves ciklusban)
    indicators: Dict[str, float]  # nyers normalizált mutatók (debughoz)

    @property
    def label(self) -> str:
        return self.cycle.value

    @property
    def is_bull(self) -> bool:
        return self.cycle in (MarketCycle.BULL_EARLY, MarketCycle.BULL_MID,
                              MarketCycle.BULL_LATE, MarketCycle.ALTSEASON)

    @property
    def is_bear(self) -> bool:
        return self.cycle in (MarketCycle.BEAR_EARLY, MarketCycle.BEAR_MID,
                              MarketCycle.RISK_OFF)

    @property
    def is_uncertain(self) -> bool:
        return self.cycle in (MarketCycle.DISTRIBUTION, MarketCycle.RISK_OFF)

    def __str__(self) -> str:
        return (
            f"[{self.cycle.value.upper()}] conf={self.confidence:.2f} "
            f"in_cycle={self.days_in_cycle}d  remaining~{self.days_remaining_est}d "
            f"({self.cycle_completion*100:.0f}% done) "
            f"→next:{self.next_cycle.value}  halving_phase={self.halving_phase:.2f}"
        )


# =============================================================================
# MarketCycleDetector
# =============================================================================

class MarketCycleDetector:
    """
    Kripto piaci ciklus detektor.

    Minden `detect()` hívásnál:
    1. Normalizálja a bemeneti adatokat 0–1 skálára
    2. Minden ciklus-profilhoz illeszkedési pontszámot számol
    3. A legjobban illeszkedő ciklust választja
    4. Frissíti a perzisztált állapotot (JSON)

    Paraméterek:
        state_path: JSON állapotfájl helye (None = nem perzisztál)
        smoothing:  hány napos mozgóátlaggal simítja a ciklus-átmeneteket
    """

    def __init__(self, state_path: Optional[str] = None, smoothing: int = 3):
        self.state_path = state_path
        self.smoothing  = smoothing
        self._history: List[MarketCycle] = []   # utolsó N detektált ciklus
        self._current_since: Optional[date] = None
        self._current: Optional[MarketCycle] = None
        self._load_state()

    # ------------------------------------------------------------------ #
    # Fő detektáló metódus
    # ------------------------------------------------------------------ #

    def detect(
        self,
        ohlcv: pd.DataFrame,
        fg_value: int = 50,
        funding_rate: float = 0.0,
        btc_dominance: float = 50.0,
        vix: float = 20.0,
        as_of: Optional[date] = None,
    ) -> CycleState:
        """
        Meghatározza az aktuális piaci ciklust.

        Paraméterek:
            ohlcv:         BTC OHLCV DataFrame (legalább 200 sor, 1d vagy 1h)
            fg_value:      Fear & Greed Index (0–100)
            funding_rate:  BTC perp funding rate (pl. 0.0003 = 0.03%/8h)
            btc_dominance: BTC piaci dominancia (0–100)
            vix:           VIX index (opcionális makro jel)
            as_of:         dátum (None = mai nap)
        """
        as_of = as_of or date.today()

        ind = self._compute_indicators(
            ohlcv, fg_value, funding_rate, btc_dominance, vix, as_of
        )

        scores = self._score_all_cycles(ind)
        detected = max(scores, key=scores.get)

        # Simítás: ne ugráljon ciklust minden nap
        detected = self._smooth(detected)

        # Napok száma az aktuális ciklusban
        if detected != self._current:
            self._current = detected
            self._current_since = as_of

        days_in = max(0, (as_of - self._current_since).days) if self._current_since else 0

        _, _, avg_dur = CYCLE_DURATIONS[detected]
        remaining = max(0, avg_dur - days_in)
        completion = min(1.0, days_in / avg_dur) if avg_dur > 0 else 0.0

        # Konfidencia = legjobb / (legjobb + 2. legjobb)
        sorted_scores = sorted(scores.values(), reverse=True)
        _denom = sorted_scores[0] + sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        confidence = (sorted_scores[0] / _denom
                      if _denom > 0 else 1.0)

        state = CycleState(
            cycle=detected,
            confidence=round(confidence, 3),
            days_in_cycle=days_in,
            days_remaining_est=remaining,
            cycle_completion=round(completion, 3),
            next_cycle=NEXT_CYCLE[detected],
            halving_phase=ind["halving"],
            indicators=ind,
        )

        self._save_state(as_of, detected)
        logger.info("Ciklus: %s", state)
        return state

    # ------------------------------------------------------------------ #
    # Normalizált mutatók számítása
    # ------------------------------------------------------------------ #

    def _compute_indicators(
        self,
        ohlcv: pd.DataFrame,
        fg_value: int,
        funding_rate: float,
        btc_dominance: float,
        vix: float,
        as_of: date,
    ) -> Dict[str, float]:
        close = ohlcv["close"].astype(float)

        # --- btc_200ma: price vs 200MA, normalizálva 0–1 ---
        ma200 = close.rolling(200, min_periods=50).mean().iloc[-1]
        ratio = close.iloc[-1] / ma200 if ma200 > 0 else 1.0
        # 0.7 = -30% MA alatt → 0.0; 1.3 = +30% MA felett → 1.0
        btc_200ma = float(np.clip((ratio - 0.70) / 0.60, 0.0, 1.0))

        # --- fg_30d: F&G 30 napos simítás ---
        fg_30d = float(fg_value / 100.0)

        # --- vol_pct: 14 napos realized vol percentilis ---
        log_ret = np.log(close / close.shift(1)).dropna()
        rv14 = log_ret.rolling(14).std() * math.sqrt(365)
        if len(rv14.dropna()) >= 30:
            current_rv = rv14.iloc[-1]
            vol_pct = float(np.mean(rv14.dropna() <= current_rv))
        else:
            vol_pct = 0.5

        # --- halving: hol tartunk a 4 éves ciklusban ---
        halving_phase = _halving_phase(as_of)

        # --- funding: -0.2%/8h → 0.0; 0% → 0.5; +0.2%/8h → 1.0 ---
        funding_norm = float(np.clip((funding_rate + 0.002) / 0.004, 0.0, 1.0))

        # --- btc_dom: dominancia normalizálva ---
        btc_dom = float(np.clip(btc_dominance / 100.0, 0.0, 1.0))

        # --- mom_90d: 90 napos árváltozás, tanh alapú normalizálás ---
        if len(close) >= 90:
            change = (close.iloc[-1] - close.iloc[-90]) / close.iloc[-90]
        else:
            change = 0.0
        # tanh: -100% → ~0.12, 0% → 0.5, +100% → ~0.88
        mom_90d = float((math.tanh(change * 1.5) + 1.0) / 2.0)

        # --- risk_off bónusz: VIX spájk ---
        # Ha VIX > 40, és vol_pct > 0.85, erős risk_off jel
        vix_factor = float(np.clip((vix - 20) / 40, 0.0, 1.0))

        return {
            "btc_200ma": btc_200ma,
            "fg_30d":    fg_30d,
            "vol_pct":   vol_pct,
            "halving":   halving_phase,
            "funding":   funding_norm,
            "btc_dom":   btc_dom,
            "mom_90d":   mom_90d,
            "vix":       vix_factor,   # extra (risk_off)
        }

    # ------------------------------------------------------------------ #
    # Ciklus illeszkedési pontszámok
    # ------------------------------------------------------------------ #

    def _score_all_cycles(self, ind: Dict[str, float]) -> Dict[MarketCycle, float]:
        scores = {}
        for cycle, profile in CYCLE_PROFILES.items():
            total_weight = 0.0
            weighted_match = 0.0
            for key, (expected, weight) in profile.items():
                actual = ind.get(key, 0.5)
                # Lineáris illeszkedés: tökéletes = 1.0, max eltérés (1.0) = 0.0
                match = 1.0 - abs(actual - expected)
                weighted_match += weight * match
                total_weight   += weight
            scores[cycle] = weighted_match / total_weight if total_weight > 0 else 0.0

        # Risk_off bónusz: ha VIX nagyon magas és vol extrém
        if ind.get("vix", 0.0) > 0.5 and ind.get("vol_pct", 0.0) > 0.85:
            scores[MarketCycle.RISK_OFF] = min(
                1.0, scores[MarketCycle.RISK_OFF] * 1.3
            )

        return scores

    # ------------------------------------------------------------------ #
    # Ciklus simítás (ne ugráljon minden nap)
    # ------------------------------------------------------------------ #

    def _smooth(self, detected: MarketCycle) -> MarketCycle:
        """N napos többségi szavazás — kis zajra ne váltson ciklust."""
        self._history.append(detected)
        if len(self._history) > self.smoothing:
            self._history = self._history[-self.smoothing:]
        # Többség
        from collections import Counter
        return Counter(self._history).most_common(1)[0][0]

    # ------------------------------------------------------------------ #
    # JSON állapot perzisztálás
    # ------------------------------------------------------------------ #

    def _save_state(self, today: date, cycle: MarketCycle) -> None:
        if not self.state_path:
            return
        try:
            state = {
                "cycle":         cycle.value,
                "since":         today.isoformat(),
                "saved_at":      datetime.now(timezone.utc).isoformat(),
            }
            Path(self.state_path).write_text(json.dumps(state, indent=2),
                                              encoding="utf-8")
        except OSError as e:
            logger.warning("Ciklus állapot mentése sikertelen: %s", e)

    def _load_state(self) -> None:
        if not self.state_path:
            return
        try:
            raw = json.loads(Path(self.state_path).read_text(encoding="utf-8"))
            self._current = MarketCycle(raw["cycle"])
            self._current_since = date.fromisoformat(raw["since"])
            logger.info("Ciklus állapot betöltve: %s (óta: %s)",
                        self._current.value, self._current_since)
        except (OSError, KeyError, ValueError):
            pass   # első futás, nincs state


# =============================================================================
# Segédfüggvények
# =============================================================================

def _halving_phase(today: date) -> float:
    """
    Hol tartunk a 4 éves Bitcoin halving ciklusban.
    Visszaad: 0.0 = közvetlenül halving után, 1.0 = következő halvingig.
    """
    past = [h for h in HALVING_DATES if h <= today]
    if not past:
        return 0.5
    last_halving = max(past)
    days_since = (today - last_halving).days
    return float(np.clip(days_since / HALVING_CYCLE_DAYS, 0.0, 1.0))


def halving_days_remaining(today: date = None) -> int:
    """Hány nap van a következő halving-ig (becsült)."""
    today = today or date.today()
    future = [h for h in HALVING_DATES if h > today]
    if not future:
        return -1
    return (min(future) - today).days


def cycle_summary(state: CycleState) -> str:
    """Emberi olvasható összefoglaló (logger / Telegram üzenetnek)."""
    _, max_dur, avg_dur = CYCLE_DURATIONS[state.cycle]
    return (
        f"📊 Piaci ciklus: *{state.cycle.value.upper()}*\n"
        f"   Konfidencia: {state.confidence*100:.0f}%\n"
        f"   Ciklusban: {state.days_in_cycle} nap\n"
        f"   Becsült hátralévő: ~{state.days_remaining_est} nap\n"
        f"   Következő várható: {state.next_cycle.value}\n"
        f"   Halving fázis: {state.halving_phase*100:.0f}% (4 éves ciklus)\n"
        f"   Napok a következő halving-ig: {halving_days_remaining()}"
    )
