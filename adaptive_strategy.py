"""
adaptive_strategy.py — Ciklus-adaptív stratégia paraméterek

Minden piaci ciklushoz más kockázatkezelési és kereskedési
szabályok érvényesek. Ez a modul tartalmazza azokat a paramétereket,
amelyeket a TradingAgent a market_cycle.py által detektált ciklus
alapján alkalmaz.

LOGIKA:
  Bull_Early/Mid: momentum követés, nagyobb pozíció, tágabb TP
  Bull_Late:      kisebb pozíció, szoros stop, gyors profit-vétel
  Distribution:   csak nagyon magas konfidenciájú jelzésre lép be
  Bear:           csak short vagy cash (ha nincs short lehetőség)
  Accumulation:   lassú, türelmes felhalmozás, dip-vásárlás
  Risk_Off:       azonnal cash, ha lehet hedge
  Altseason:      alts preferálása BTC felett

Adaptált paraméterek kategóriái:
  1. Pozícióméretezés  (max_position_pct, kelly_cap)
  2. Kockázatkezelés   (atr_stop_mult, atr_tp_mult, max_holding_bars)
  3. ML küszöb         (min_ml_prob)
  4. Megengedett irányok (allow_long, allow_short)
  5. Signal súly szorzók (momentum_mult, mean_reversion_mult, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from market_cycle import MarketCycle


# =============================================================================
# CycleRegimeParams dataclass
# =============================================================================

@dataclass
class CycleRegimeParams:
    """
    Egy piaci ciklushoz tartozó adaptív stratégia paraméterek.

    Az agent.py ezeket alkalmazza a TradingConfig base-értékei helyett.
    Ha egy ciklus alatt nincs short lehetőség (spot piac), az allow_short
    figyelmen kívül marad.
    """

    # --- Pozícióméretezés ---
    max_position_pct: float     # portfólió max %-a egyetlen pozícióra
    kelly_cap: float            # Kelly frakció felső korlátja

    # --- Stop / TP (ATR szorzók) ---
    atr_stop_mult: float        # pl. 2.0 → stop = entry ± 2×ATR
    atr_tp_mult: float          # pl. 4.0 → TP   = entry ± 4×ATR
    max_holding_bars: int       # max. tartási idő (gyertyák száma)

    # --- ML meta-label küszöb ---
    min_ml_prob: float          # alatt: HOLD kényszer (0.5–0.75)

    # --- Megengedett kereskedési irányok ---
    allow_long: bool
    allow_short: bool           # csak perp trading esetén

    # --- Signal súly szorzók (base config × szorzó) ---
    momentum_mult: float        # trend/momentum jelzések súlya
    mean_reversion_mult: float  # kontrarian/MR jelzések súlya
    volume_mult: float          # volume-alapú jelzések súlya
    funding_contrarian_mult: float  # funding rate kontrarian súlya

    # --- Buy threshold módosító ---
    score_threshold_delta: float  # +x → nehezebben vesz, −x → könnyebben

    # --- Altseason-specifikus coin szűrő ---
    # Ha nem None: csak ezeket a symbolokat engedélyezi (altseason whitelist)
    # None = nincs szűrés (bármely coin mehet)
    altseason_validate: bool = False   # kötelező-e AltseasonValidator átmenni
    altseason_min_halvings: int = 1    # min halvings a coin-hoz
    altseason_cap_tiers: List[str] = field(default_factory=list)  # ["large","mid","small"]

    # --- Leírás (logging) ---
    note: str = ""


# =============================================================================
# Ciklus-specifikus paraméterek
# =============================================================================

CYCLE_PARAMS: Dict[MarketCycle, CycleRegimeParams] = {

    # ─── ACCUMULATION ────────────────────────────────────────────────────────
    # Lassú, csendes felhalmozás. Türelem, kis pozíciók, kontrarian dip-buy.
    MarketCycle.ACCUMULATION: CycleRegimeParams(
        max_position_pct       = 0.12,
        kelly_cap              = 0.30,
        atr_stop_mult          = 2.0,
        atr_tp_mult            = 6.0,   # hosszú TP: lassú felfelé mozgás várható
        max_holding_bars       = 40,
        min_ml_prob            = 0.60,
        allow_long             = True,
        allow_short            = False,
        momentum_mult          = 0.6,   # momentum gyenge
        mean_reversion_mult    = 1.8,   # dip-vásárlás hangsúly
        volume_mult            = 1.3,
        funding_contrarian_mult= 1.0,
        score_threshold_delta  = +0.05, # konzervatív
        note="Türelmes felhalmozás — csak magas meggyőződésű long belépések",
    ),

    # ─── BULL_EARLY ──────────────────────────────────────────────────────────
    # Kitörés a 200MA fölé. BTC vezet, alts még alszanak.
    # Érdemes pozíciót építeni, de nem overcommitolni.
    MarketCycle.BULL_EARLY: CycleRegimeParams(
        max_position_pct       = 0.20,
        kelly_cap              = 0.40,
        atr_stop_mult          = 2.0,
        atr_tp_mult            = 4.5,
        max_holding_bars       = 25,
        min_ml_prob            = 0.56,
        allow_long             = True,
        allow_short            = False,
        momentum_mult          = 1.4,
        mean_reversion_mult    = 0.8,
        volume_mult            = 1.3,
        funding_contrarian_mult= 0.8,
        score_threshold_delta  = 0.0,
        note="Kitörés fázis — momentum követés, BTC és nagy altok",
    ),

    # ─── BULL_MID ────────────────────────────────────────────────────────────
    # Erős, folyamatos uptrend. Az egész piac megy. Largest pozíciók.
    MarketCycle.BULL_MID: CycleRegimeParams(
        max_position_pct       = 0.25,
        kelly_cap              = 0.50,
        atr_stop_mult          = 2.5,   # tágabb stop: nem shake out korai
        atr_tp_mult            = 5.5,
        max_holding_bars       = 35,
        min_ml_prob            = 0.54,
        allow_long             = True,
        allow_short            = False,
        momentum_mult          = 1.7,
        mean_reversion_mult    = 0.5,   # ne vásárolj ellene
        volume_mult            = 1.5,
        funding_contrarian_mult= 1.2,
        score_threshold_delta  = -0.03, # valamivel könnyebben vesz
        note="Erős bull mid — momentum követés, max. pozíciók",
    ),

    # ─── BULL_LATE ───────────────────────────────────────────────────────────
    # Parabolikus mozgás. Extrém greed. Funding ég. Bármikor fordulhat.
    # Kisebb pozíciók, szoros stop, gyors profit-vétel.
    MarketCycle.BULL_LATE: CycleRegimeParams(
        max_position_pct       = 0.10,  # felekkora pozíció
        kelly_cap              = 0.25,
        atr_stop_mult          = 1.5,   # szoros stop
        atr_tp_mult            = 2.5,   # gyors profit-vétel
        max_holding_bars       = 12,
        min_ml_prob            = 0.65,  # magas bar
        allow_long             = True,
        allow_short            = True,  # már érdemes short jeleket figyelni
        momentum_mult          = 0.8,
        mean_reversion_mult    = 1.5,   # kontrarian kap szerepet
        volume_mult            = 1.0,
        funding_contrarian_mult= 2.5,   # funding kontrarian NAGYON fontos!
        score_threshold_delta  = +0.08, # nehezebben lép be
        note="Bull csúcs — kis pozíciók, szoros stop, funding kontrarian",
    ),

    # ─── DISTRIBUTION ────────────────────────────────────────────────────────
    # Topp formáció. Choppy, megtévesztő. Nehéz kereskedni.
    # Csak nagyon erős jelzésre szabad belépni.
    MarketCycle.DISTRIBUTION: CycleRegimeParams(
        max_position_pct       = 0.08,
        kelly_cap              = 0.20,
        atr_stop_mult          = 1.5,
        atr_tp_mult            = 2.0,
        max_holding_bars       = 10,
        min_ml_prob            = 0.70,  # nagyon magas bar!
        allow_long             = False,
        allow_short            = True,
        momentum_mult          = 0.4,
        mean_reversion_mult    = 1.0,
        volume_mult            = 1.5,
        funding_contrarian_mult= 2.0,
        score_threshold_delta  = +0.12, # nagyon nehezen lép be
        note="Distribution — alapvetően cash, csak szélsőséges jelzésre",
    ),

    # ─── BEAR_EARLY ──────────────────────────────────────────────────────────
    # Gyors esés, pánik, high vol. Ne kés a shortba.
    MarketCycle.BEAR_EARLY: CycleRegimeParams(
        max_position_pct       = 0.10,
        kelly_cap              = 0.25,
        atr_stop_mult          = 2.0,
        atr_tp_mult            = 3.5,
        max_holding_bars       = 15,
        min_ml_prob            = 0.60,
        allow_long             = False,
        allow_short            = True,
        momentum_mult          = 1.3,   # bear momentum követés
        mean_reversion_mult    = 0.3,   # ne vásárolj dip-et!
        volume_mult            = 1.3,
        funding_contrarian_mult= 1.5,
        score_threshold_delta  = +0.05,
        note="Bear eleje — short bias, ne vásárolj dip-et",
    ),

    # ─── BEAR_MID ────────────────────────────────────────────────────────────
    # Lassú, unalmas grinding. Alacsony vol. Cash a legjobb pozíció.
    MarketCycle.BEAR_MID: CycleRegimeParams(
        max_position_pct       = 0.08,
        kelly_cap              = 0.20,
        atr_stop_mult          = 1.8,
        atr_tp_mult            = 3.0,
        max_holding_bars       = 20,
        min_ml_prob            = 0.65,
        allow_long             = False,
        allow_short            = True,
        momentum_mult          = 0.8,
        mean_reversion_mult    = 0.5,
        volume_mult            = 0.8,
        funding_contrarian_mult= 1.0,
        score_threshold_delta  = +0.10,
        note="Bear közepe — alapvetően cash, short lehetőség alacsony vol mellett",
    ),

    # ─── ALTSEASON ───────────────────────────────────────────────────────────
    # BTC dominancia zuhan, alts felülteljesítenek. Momentum extrém.
    # KÖTELEZŐ: AltseasonValidator megerősítése (false altseason kiszűrés).
    # Csak halving-túlélő coinok engedélyezve, cap tier szerint fokozatosan:
    #   Első 14 nap:  csak LARGE cap
    #   14-30 nap:    LARGE + MID cap
    #   30+ nap:      LARGE + MID + SMALL (magasabb ML küszöb mellett)
    MarketCycle.ALTSEASON: CycleRegimeParams(
        max_position_pct           = 0.20,
        kelly_cap                  = 0.40,
        atr_stop_mult              = 2.0,
        atr_tp_mult                = 4.0,
        max_holding_bars           = 20,
        min_ml_prob                = 0.57,
        allow_long                 = True,
        allow_short                = False,
        momentum_mult              = 1.8,   # extrém momentum
        mean_reversion_mult        = 0.4,
        volume_mult                = 1.6,
        funding_contrarian_mult    = 2.0,   # alt funding nagyon magas lehet
        score_threshold_delta      = -0.03,
        altseason_validate         = True,  # KÖTELEZŐ AltseasonValidator
        altseason_min_halvings     = 1,     # legalább 1 halving túlélve
        altseason_cap_tiers        = ["large"],  # default: csak large (bővül idővel)
        note="Altseason — CSAK validált, halving-túlélő altok | cap tier fokozatos",
    ),

    # ─── RISK_OFF ────────────────────────────────────────────────────────────
    # Makro sokk, piaci pánik. Az egyetlen helyes stratégia: cash.
    MarketCycle.RISK_OFF: CycleRegimeParams(
        max_position_pct       = 0.05,  # minimális
        kelly_cap              = 0.10,
        atr_stop_mult          = 1.5,
        atr_tp_mult            = 2.0,
        max_holding_bars       = 8,
        min_ml_prob            = 0.75,  # extrém magas bar
        allow_long             = False,
        allow_short            = True,
        momentum_mult          = 0.5,
        mean_reversion_mult    = 0.3,
        volume_mult            = 0.5,
        funding_contrarian_mult= 1.0,
        score_threshold_delta  = +0.20, # szinte soha nem lép be
        note="RISK_OFF — cash/hedge, stop loss-ok szükségesek",
    ),
}


# =============================================================================
# Publikus API
# =============================================================================

def get_params(cycle: MarketCycle) -> CycleRegimeParams:
    """Visszaadja az adott ciklushoz tartozó paramétereket."""
    return CYCLE_PARAMS[cycle]


def apply_to_config(config, params: CycleRegimeParams) -> None:
    """
    Helyben módosítja a TradingConfig-ot a ciklus paraméterek alapján.

    Azokat a mezőket írja felül, amelyek adaptálhatók.
    A base config értékei elvesznek — mindig az eredeti config-ból
    kell meghívni (ne lancolja a hívásokat).
    """
    # Stop/TP szorzók — config.stops-ba kell írni, NEM config.risk-be!
    config.stops.atr_stop_mult = params.atr_stop_mult
    config.stops.atr_tp_mult   = params.atr_tp_mult

    # Buy/Sell küszöb módosítása
    config.buy_threshold  = config.buy_threshold  + params.score_threshold_delta
    config.sell_threshold = config.sell_threshold + params.score_threshold_delta

    # Max holding
    if hasattr(config, "max_holding_bars"):
        config.max_holding_bars = params.max_holding_bars


def describe(cycle: MarketCycle) -> str:
    """Ciklus leírás és aktuális paraméterek (loghoz / Telegramhoz)."""
    p = CYCLE_PARAMS[cycle]
    dirs = []
    if p.allow_long:  dirs.append("LONG")
    if p.allow_short: dirs.append("SHORT")
    if not dirs:      dirs.append("CASH")

    return (
        f"⚙️ Adaptív paraméterek [{cycle.value.upper()}]\n"
        f"   Irányok: {'/'.join(dirs)}\n"
        f"   Max pozíció: {p.max_position_pct*100:.0f}%\n"
        f"   Stop: {p.atr_stop_mult}×ATR  TP: {p.atr_tp_mult}×ATR\n"
        f"   Max tartás: {p.max_holding_bars} gyertya\n"
        f"   ML min prob: {p.min_ml_prob:.2f}\n"
        f"   Momentum szorzó: {p.momentum_mult}×\n"
        f"   Megjegyzés: {p.note}"
    )
