"""
override_engine.py — Szabály-felülíró rendszer rendkívüli piaci feltételek esetén

ALAPELV:
  Az override NEM csökkenti a standardokat — éppen ellenkezőleg.
  Egy override-olt kereskedés MAGASABB bizonyítékot igényel, mint egy normál.
  Ha az összes rendszerünk egyszerre, egybehangzóan jelzi, hogy valami
  rendkívüli történik, akkor a konzervatív korlátokat fel kell oldani —
  de mindig kisebb pozíciómérettel és explicit naplózással.

  Példa: Sunday 03:00 UTC (timing hard block) + BTC -25% 4 óra alatt +
         F&G=6 + funding=-0.5% (extreme short squeeze setup) + ML p=0.84
         → OVERRIDE: BUY, de csak 40% normál mérettel

AMIT SOHA NEM LEHET FELÜLÍRNI (abszolút tilalom):
  - Kill switch / napi veszteség limit              (risk_manager.py)
  - Stop loss kötelezettség                         (minden pozícióhoz)
  - Abszolút max pozícióméret                       (ciklus max × 1.5)
  - API / environment hibák

OVERRIDE HIERARCHIA (könnyűtől a legszigorúbbig):
  Szint 1 — Timing blokk         (conviction ≥ 0.72):
    Hétvége/funding ablak → override ha az összes többi jel egybeesik
  Szint 2 — ML küszöb            (conviction ≥ 0.78):
    Ha az ML kevésbé biztos, de minden más tökéletes → engedmény
  Szint 3 — Ciklus irány-tilalom (conviction ≥ 0.85):
    Bear piacban long, distribution alatt short → csak extrém bizonyítékra
  Szint 4 — RISK_OFF long        (conviction ≥ 0.92):
    Pánik-vásárlás RISK_OFF ciklusban — szinte soha, de capitulation esetén

CONVICTION SCORE KOMPONENSEI (0.0–1.0):
  1. Jelzés-unanimitás   (30%): hány signal mutat egy irányba
  2. ML konfidencia      (25%): XGBoost p (ha van modell)
  3. F&G extrém          (15%): < 10 vételre, > 90 eladásra
  4. Funding kontrarian  (15%): extrém funding az ellentétes oldalon
  5. Volumen anomália    (15%): szokatlan volumen = fontos esemény
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("override_engine")


# =============================================================================
# Blokkolás okok
# =============================================================================

class BlockReason(str, Enum):
    TIMING_HARD        = "timing_hard"        # timing.hard_block
    TIMING_THRESHOLD   = "timing_threshold"   # timing score_threshold_delta
    CYCLE_DIRECTION    = "cycle_direction"     # allow_long/allow_short tiltás
    CYCLE_THRESHOLD    = "cycle_threshold"     # ciklus score_threshold_delta
    ML_PROBABILITY     = "ml_probability"      # ml_score.probability < min
    ALTSEASON_FALSE    = "altseason_false"     # hamis altseason
    ALTSEASON_TIER     = "altseason_tier"      # coin tier még nem nyílt meg
    ALTSEASON_HALVING  = "altseason_halving"   # nem elég halvingot élt túl


# Override küszöbök blokkolás-típusonként
_OVERRIDE_THRESHOLDS: Dict[BlockReason, float] = {
    BlockReason.TIMING_THRESHOLD:   0.68,   # könnyebb: csak enyhe timing ok
    BlockReason.TIMING_HARD:        0.78,   # nehezebb: valóban hard block
    BlockReason.ML_PROBABILITY:     0.80,   # ML-t csak ritkán írjuk felül
    BlockReason.ALTSEASON_TIER:     0.82,   # tier nyitás hamarabb ha jó jel
    BlockReason.ALTSEASON_FALSE:    0.88,   # hamis altseason override — ritka
    BlockReason.CYCLE_THRESHOLD:    0.82,   # ciklus threshold átlépés
    BlockReason.CYCLE_DIRECTION:    0.88,   # irány-tilalom — szigorú
    BlockReason.ALTSEASON_HALVING:  0.95,   # halvingot nem élte túl — szinte soha
}

# Pozícióméret szorzó override esetén (az alap ciklus szorzóhoz képest)
_OVERRIDE_SIZE_MULT: Dict[BlockReason, float] = {
    BlockReason.TIMING_THRESHOLD:   0.65,
    BlockReason.TIMING_HARD:        0.45,
    BlockReason.ML_PROBABILITY:     0.55,
    BlockReason.ALTSEASON_TIER:     0.60,
    BlockReason.ALTSEASON_FALSE:    0.50,
    BlockReason.CYCLE_THRESHOLD:    0.60,
    BlockReason.CYCLE_DIRECTION:    0.40,
    BlockReason.ALTSEASON_HALVING:  0.30,
}


# =============================================================================
# ConvictionScore
# =============================================================================

@dataclass
class ConvictionScore:
    """Az override-döntés részletes indoklása."""

    total: float                    # 0.0–1.0 összesített meggyőzés
    signal_unanimity: float         # hány signal mutat egy irányba
    ml_confidence: float            # ML p (0 ha nincs modell)
    fg_extreme: float               # F&G extrém-ség (0–1)
    funding_contrarian: float       # funding kontrarian erő (0–1)
    volume_anomaly: float           # volumen anomália (0–1)
    components: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"conviction={self.total:.3f} "
            f"[signals={self.signal_unanimity:.2f} ml={self.ml_confidence:.2f} "
            f"fg={self.fg_extreme:.2f} funding={self.funding_contrarian:.2f} "
            f"vol={self.volume_anomaly:.2f}]"
        )


# =============================================================================
# OverrideDecision
# =============================================================================

@dataclass
class OverrideDecision:
    """Az override motor döntése."""

    triggered: bool                  # van-e override
    action: str                      # az (esetleg felülírt) action
    conviction: ConvictionScore
    overridden_rules: List[BlockReason] = field(default_factory=list)
    refused_rules: List[BlockReason]   = field(default_factory=list)  # nem engedte
    position_size_mult: float = 1.0   # alap ciklus szorzóhoz képest
    note: str = ""
    justification: str = ""          # részletes emberi olvasható indoklás

    def __str__(self) -> str:
        if not self.triggered:
            return f"override=NO | {self.conviction}"
        rules = [r.value for r in self.overridden_rules]
        return (
            f"override=YES ({', '.join(rules)}) "
            f"size×{self.position_size_mult:.2f} | {self.conviction} | {self.note}"
        )


# =============================================================================
# OverrideEngine
# =============================================================================

class OverrideEngine:
    """
    Szabály-felülíró motor.

    Működés:
      1. A TradingAgent decide_at() végig futtatja a normál logikát
      2. Ahol egy szabály HOLD-ra állítja az action-t, rögzíti a
         BlockReason-t (és a szándékolt eredeti action-t)
      3. Az OverrideEngine.evaluate() megkapja a blokkolt action-t
         és a blokk okait
      4. Kiszámítja a conviction score-t az összes elérhető adatból
      5. Ha conviction ≥ threshold minden blokkolt szabályra → override
      6. Ha bármelyik szabályt nem lehet felülírni → megmarad HOLD

    Abszolút tilalom (nem vizsgálható):
      A hívó kódnak (agent.py) SOHA nem szabad ide eljuttatni a hívást
      ha kill switch vagy daily loss limit triggerelt. Ezeket az
      override motor nem látja és nem is kell hogy lássa.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def evaluate(
        self,
        intended_action: str,           # "BUY" / "SELL" — mi volt az eredeti szándék
        blocked_reasons: List[BlockReason],  # miért lett HOLD
        signals: Dict[str, int],         # technikai jelzések
        score: float,                    # aggregált score
        ml_prob: Optional[float],        # ML konfidencia (None ha nincs modell)
        fg_value: int,                   # Fear & Greed (0–100)
        row: pd.Series,                  # aktuális OHLCV + indikátor sor
        funding_rate: float = 0.0,       # perpektuális funding rate
    ) -> OverrideDecision:
        """
        Értékeli, hogy az override indokolt-e.

        Ha `enabled=False` vagy `intended_action` nem BUY/SELL
        → mindig visszaad egy nem-triggerelt döntést.
        """
        if not self.enabled or intended_action == "HOLD" or not blocked_reasons:
            return OverrideDecision(
                triggered=False,
                action="HOLD",
                conviction=self._zero_conviction(),
            )

        conviction = self._compute_conviction(
            intended_action, signals, score, ml_prob, fg_value, row, funding_rate
        )

        overridden: List[BlockReason] = []
        refused:    List[BlockReason] = []

        for reason in blocked_reasons:
            threshold = _OVERRIDE_THRESHOLDS.get(reason, 1.0)
            if conviction.total >= threshold:
                overridden.append(reason)
                logger.warning(
                    "OVERRIDE: %s felülírva — %s (threshold=%.2f)",
                    reason.value, conviction, threshold
                )
            else:
                refused.append(reason)
                logger.debug(
                    "Override elutasítva: %s — conviction=%.3f < threshold=%.2f",
                    reason.value, conviction.total, threshold
                )

        # Ha bármelyik ok nem override-olható → marad HOLD
        if refused:
            return OverrideDecision(
                triggered=False,
                action="HOLD",
                conviction=conviction,
                overridden_rules=[],
                refused_rules=refused + overridden,
                justification=self._refused_justification(
                    conviction, refused, blocked_reasons
                ),
            )

        # Minden blokk override-olható → felülírjuk
        # Pozícióméret: a legszigorúbb (legkisebb) szorzó érvényes
        size_mult = min(
            _OVERRIDE_SIZE_MULT.get(r, 0.5) for r in overridden
        )

        note = (
            f"OVERRIDE aktív: {[r.value for r in overridden]} "
            f"| conviction={conviction.total:.3f} → {intended_action} ×{size_mult:.2f}"
        )

        return OverrideDecision(
            triggered=True,
            action=intended_action,
            conviction=conviction,
            overridden_rules=overridden,
            refused_rules=[],
            position_size_mult=size_mult,
            note=note,
            justification=self._triggered_justification(
                intended_action, conviction, overridden, size_mult,
                signals, fg_value, funding_rate,
            ),
        )

    # ------------------------------------------------------------------ #
    # Conviction számítás
    # ------------------------------------------------------------------ #

    def _compute_conviction(
        self,
        direction: str,
        signals: Dict[str, int],
        score: float,
        ml_prob: Optional[float],
        fg_value: int,
        row: pd.Series,
        funding_rate: float,
    ) -> ConvictionScore:
        """
        Kiszámítja az override szintű meggyőzési pontszámot.

        Minden komponens 0.0–1.0 között van, a súlyok összeadnak 1.0-ra.
        """
        sign = 1 if direction == "BUY" else -1

        # ── 1. Jelzés-unanimitás (30%) ────────────────────────────────────
        # Hány signal mutat a kívánt irányba? Extrém esetben mind 15.
        if signals:
            agree = sum(1 for v in signals.values() if v * sign > 0)
            signal_unanimity = agree / len(signals)
        else:
            signal_unanimity = 0.0

        # ── 2. ML konfidencia (25%) ───────────────────────────────────────
        ml_confidence = float(ml_prob) if ml_prob is not None else 0.50

        # ── 3. Fear & Greed extrém (15%) ─────────────────────────────────
        # BUY: extrém fear (< 15) = jó, 0 = legjobb
        # SELL: extrém greed (> 85) = jó, 100 = legjobb
        if direction == "BUY":
            fg_extreme = max(0.0, (20 - fg_value) / 20)  # 0→1.0, 20→0.0
        else:
            fg_extreme = max(0.0, (fg_value - 80) / 20)  # 80→0.0, 100→1.0

        # ── 4. Funding rate kontrarian (15%) ──────────────────────────────
        # BUY: extrém negatív funding (short squeeze setup) = jó
        # SELL: extrém pozitív funding (long squeeze setup) = jó
        if direction == "BUY":
            # -0.2%/8h = maximálisan bullish kontrarian
            funding_contrarian = min(1.0, max(0.0, -funding_rate / 0.002))
        else:
            # +0.2%/8h = maximálisan bearish kontrarian
            funding_contrarian = min(1.0, max(0.0, funding_rate / 0.002))

        # ── 5. Volumen anomália (15%) ─────────────────────────────────────
        # Szokatlan volumen = fontos esemény; 2× átlag = 0.5, 5× = 1.0
        volume_anomaly = 0.5  # fallback ha nincs adat
        try:
            if "volume" in row.index and "vol_ma20" in row.index:
                vol_ratio = float(row["volume"]) / (float(row["vol_ma20"]) + 1e-9)
                volume_anomaly = min(1.0, (vol_ratio - 1.0) / 4.0)
                volume_anomaly = max(0.0, volume_anomaly)
            elif "volume" in row.index:
                # Közelítés: ha van volume de nincs MA, neutral 0.5
                volume_anomaly = 0.5
        except (TypeError, ZeroDivisionError):
            volume_anomaly = 0.5

        # ── Összesítés (súlyozott átlag) ──────────────────────────────────
        W = {"signal": 0.30, "ml": 0.25, "fg": 0.15, "funding": 0.15, "vol": 0.15}
        total = (
            W["signal"]  * signal_unanimity +
            W["ml"]      * ml_confidence    +
            W["fg"]      * fg_extreme        +
            W["funding"] * funding_contrarian +
            W["vol"]     * volume_anomaly
        )

        # Büntetés ha a score gyenge (a signal score maga is alacsony)
        # Ha a score alig érte el a küszöböt, az nem "meggyőző"
        score_strength = min(1.0, abs(score) * 2)   # 0.5 score → 1.0
        total = total * (0.7 + 0.3 * score_strength)

        return ConvictionScore(
            total              = round(min(1.0, max(0.0, total)), 4),
            signal_unanimity   = round(signal_unanimity, 3),
            ml_confidence      = round(ml_confidence, 3),
            fg_extreme         = round(fg_extreme, 3),
            funding_contrarian = round(funding_contrarian, 3),
            volume_anomaly     = round(volume_anomaly, 3),
            components         = {k: round(v, 3) for k, v in {
                "signal": signal_unanimity * W["signal"],
                "ml":     ml_confidence * W["ml"],
                "fg":     fg_extreme * W["fg"],
                "funding":funding_contrarian * W["funding"],
                "vol":    volume_anomaly * W["vol"],
            }.items()},
        )

    # ------------------------------------------------------------------ #
    # Indoklás generálás
    # ------------------------------------------------------------------ #

    _RULE_NAMES: dict = {
        BlockReason.TIMING_HARD:       "Timing hard block (hétvége / funding ablak)",
        BlockReason.TIMING_THRESHOLD:  "Timing score küszöb",
        BlockReason.CYCLE_DIRECTION:   "Ciklus irány-tilalom",
        BlockReason.CYCLE_THRESHOLD:   "Ciklus score küszöb",
        BlockReason.ML_PROBABILITY:    "ML konfidencia küszöb",
        BlockReason.ALTSEASON_FALSE:   "Hamis altseason blokk",
        BlockReason.ALTSEASON_TIER:    "Altseason tier zár",
        BlockReason.ALTSEASON_HALVING: "Halving-túlélő szűrő",
    }

    @staticmethod
    def _triggered_justification(
        direction: str,
        conviction: ConvictionScore,
        overridden: List[BlockReason],
        size_mult: float,
        signals: Dict[str, int],
        fg_value: int,
        funding_rate: float,
    ) -> str:
        """Részletes indoklás sikeres override esetén (log + Telegram)."""
        sign = 1 if direction == "BUY" else -1
        arrow = "⬆️ VÁSÁRLÁS" if direction == "BUY" else "⬇️ ELADÁS"
        lines = [f"⚠️  OVERRIDE — {arrow} szabály-felülírással"]
        lines.append("")

        lines.append("Felülírt szabályok:")
        for r in overridden:
            thresh = _OVERRIDE_THRESHOLDS.get(r, 1.0)
            name = OverrideEngine._RULE_NAMES.get(r, r.value)
            lines.append(f"  • {name}  (küszöb: {thresh:.2f} ✓ elért: {conviction.total:.3f})")

        lines.append("")
        lines.append(f"Meggyőzési pontszám: {conviction.total:.3f} / 1.000")
        lines.append("")
        lines.append("Bizonyítékok részletesen:")

        # Jelzések
        n_total = len(signals)
        n_agree = sum(1 for v in signals.values() if v * sign > 0) if signals else 0
        bullish_names = [k for k, v in signals.items() if v * sign > 0]
        lines.append(
            f"  • Jelzések: {n_agree}/{n_total} egybehangzó"
            f"  ({conviction.signal_unanimity*100:.0f}%)"
            f"  [{', '.join(bullish_names[:5])}{'…' if len(bullish_names)>5 else ''}]"
            f"  → súlyozott hozzájárulás: {conviction.components.get('signal', 0):.3f}"
        )

        # ML
        if conviction.ml_confidence != 0.50:   # 0.50 = nincs modell (fallback)
            ml_label = (
                "erős" if conviction.ml_confidence > 0.72
                else "mérsékelt" if conviction.ml_confidence > 0.60
                else "gyenge"
            )
            lines.append(
                f"  • ML modell: p = {conviction.ml_confidence:.3f}  ({ml_label})"
                f"  → {conviction.components.get('ml', 0):.3f}"
            )
        else:
            lines.append("  • ML modell: nincs betanított modell (neutral 0.50)")

        # F&G
        if conviction.fg_extreme > 0:
            fg_label = "Extreme Fear" if fg_value <= 20 else "Fear"
            lines.append(
                f"  • Fear & Greed: {fg_value}  ({fg_label})"
                f"  → kontrarian {direction} jel"
                f"  → {conviction.components.get('fg', 0):.3f}"
            )

        # Funding
        if conviction.funding_contrarian > 0.2:
            fr_pct = funding_rate * 100
            squeeze = "short squeeze setup" if direction == "BUY" else "long squeeze setup"
            lines.append(
                f"  • Funding rate: {fr_pct:+.3f}%/8h  ({squeeze})"
                f"  → {conviction.components.get('funding', 0):.3f}"
            )

        # Volumen
        if conviction.volume_anomaly > 0.5:
            lines.append(
                f"  • Volumen anomália: {conviction.volume_anomaly:.2f}  (szokatlanul magas)"
                f"  → fontos piaci esemény"
                f"  → {conviction.components.get('vol', 0):.3f}"
            )

        lines.append("")
        lines.append(
            f"Pozícióméret: normál ×{size_mult:.2f} ({size_mult*100:.0f}%)"
            f"  — csökkentett az override kockázata miatt"
        )
        lines.append("Stop loss: kötelező, nem override-olható.")
        return "\n".join(lines)

    @staticmethod
    def _refused_justification(
        conviction: ConvictionScore,
        refused: List[BlockReason],
        all_blocked: List[BlockReason],
    ) -> str:
        """Indoklás ha az override NEM triggerelt — miért maradt HOLD."""
        lines = [
            f"Override elutasítva  (conviction = {conviction.total:.3f})"
        ]
        lines.append("A következő szabályokhoz nem volt elegendő bizonyíték:")
        for r in refused:
            thresh = _OVERRIDE_THRESHOLDS.get(r, 1.0)
            gap = thresh - conviction.total
            name = OverrideEngine._RULE_NAMES.get(r, r.value)
            lines.append(
                f"  • {name}: kell {thresh:.2f},"
                f" van {conviction.total:.3f}  (hiány: {gap:.3f})"
            )
        lines.append("→ Döntés: HOLD marad.")
        return "\n".join(lines)

    @staticmethod
    def _zero_conviction() -> ConvictionScore:
        return ConvictionScore(
            total=0.0, signal_unanimity=0.0, ml_confidence=0.0,
            fg_extreme=0.0, funding_contrarian=0.0, volume_anomaly=0.0,
        )


# =============================================================================
# Globális példány (az agent.py ezt importálja)
# =============================================================================

_ENGINE = OverrideEngine(enabled=True)


def evaluate_override(
    intended_action: str,
    blocked_reasons: List[BlockReason],
    signals: Dict[str, int],
    score: float,
    ml_prob: Optional[float],
    fg_value: int,
    row: pd.Series,
    funding_rate: float = 0.0,
) -> OverrideDecision:
    """Kényelmi függvény az override kiértékelésére."""
    return _ENGINE.evaluate(
        intended_action=intended_action,
        blocked_reasons=blocked_reasons,
        signals=signals,
        score=score,
        ml_prob=ml_prob,
        fg_value=fg_value,
        row=row,
        funding_rate=funding_rate,
    )
