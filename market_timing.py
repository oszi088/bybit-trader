"""
market_timing.py — Időalapú piaci aktivitás elemző

MIÉRT SZÁMÍT AZ IDŐ?

  Intraday:  A kripto piac forgalma nem egyenletes. Az US+EU átfedés
             (13:00–16:00 UTC) adja a napi forgalom ~40%-át. Ázsiai
             éjszaka (01:00–06:00 UTC) vékony piac, könnyű manipuláció,
             nagyobb spread, gyengébb jelzések.

  Heti:      Hétvégén az intézményi asztalok offline vannak. A forgalom
             30–40%-kal csökken. Az ML modellek hétvégén szignifikánsan
             rosszabbul teljesítenek (alacsony S/N arány).

  Havi:      Opciók/futures lejárat (minden péntek, hó vége kiemelten)
             kiszámítható volatilitás-spájkot okoz. Hó eleje intézményi
             újrasúlyozásból forrásbeáramlást hoz.

  Szezonális (éves):
             Q4 (okt–dec) historikusan legerősebb — intézményi
             teljesítményhajhászás + retail bónusz szezon.
             Szeptember historikusan a leggyengébb hónap BTC-nek.
             "Uptober" (október) és a Q1 january-effect is jól mért.

  Funding window:
             Bybit perpetual funding 8 óránként: 00:00, 08:00, 16:00 UTC.
             Az átállás előtt 15–30 perccel a smart money manipulálja
             az árat a funding fizetés maximalizálásához. NE lépj be
             ebben az ablakban — hamis mozgás, gyorsan visszafordul.

A timing score (0.0–1.0) három helyen módosít:
  1. Pozícióméretezés: alacsony score → kisebb pozíció
  2. Belépési küszöb:  alacsony score → nehezebb belépni
  3. Hard block:       0.50 alatt és hétvégén HOLD kényszer
"""

from __future__ import annotations

import calendar
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

logger = logging.getLogger("market_timing")


# =============================================================================
# Intraday volumen profil (UTC óra → score)
# =============================================================================
# Alapja: Bybit/Binance aggregált forgalom 2021–2024, kripto twitter aktivitás,
#         CoinGlass liquidation heatmap, CME open interest

# Funding reset órák: 00, 08, 16 UTC → kerülendő belépés ±30 perccel
_FUNDING_HOURS = {0, 8, 16}

_HOUR_SCORE: dict[int, float] = {
    0:  0.50,   # funding reset + nadir (éjfél UTC = US éjfél / ázsiai reggel 9)
    1:  0.48,   # legalacsonyabb volumen
    2:  0.48,
    3:  0.50,
    4:  0.55,   # Ázsia ébredezik (Tokyo/Seoul piac)
    5:  0.60,
    6:  0.65,   # HK/SG nyit, forgalom nő
    7:  0.72,
    8:  0.58,   # EU nyit DE funding reset → vegyes
    9:  0.85,   # EU aktív, London teljes gőzzel
    10: 0.88,
    11: 0.85,
    12: 0.80,   # EU ebéd, enyhe visszaesés
    13: 0.88,   # US pre-market, forgalom épül
    14: 0.95,   # US nyit — legmagasabb likviditás!
    15: 0.95,
    16: 0.60,   # funding reset! — manipuláció csúcs
    17: 0.85,   # US delután, stabil
    18: 0.88,
    19: 0.82,
    20: 0.75,   # US este, EU zár
    21: 0.68,
    22: 0.60,   # holtidő (US / Ázsia gap)
    23: 0.55,
}

# =============================================================================
# Hét napja (hétfő=0 … vasárnap=6)
# =============================================================================
_DOW_SCORE: dict[int, float] = {
    0: 0.80,   # hétfő — CME gap fill kockázat, de javuló volumen
    1: 0.92,   # kedd — erős
    2: 0.95,   # szerda — legjobb nap (csúcs forgalom)
    3: 0.92,   # csütörtök — erős
    4: 0.78,   # péntek — opció lejárat volatilitás, délután csökken
    5: 0.52,   # szombat — vékony piac, manipuláció kockázat
    6: 0.52,   # vasárnap — legkisebb forgalom
}

# =============================================================================
# Hónap a évben (1–12)
# =============================================================================
# Forrás: BTC historikus havi hozamok 2013–2024 medián
_MONTH_SCORE: dict[int, float] = {
    1:  0.88,   # január — "new year" optimizmus, január effect
    2:  0.82,   # február — változó
    3:  0.78,   # március — Q1 vége, intézményi adóelőkészítés
    4:  0.87,   # április — Q2 start, pre-/post-halving erő
    5:  0.68,   # május — "Sell in May" minta historikusan erős
    6:  0.68,   # június — nyári pangás kezdete
    7:  0.72,   # július — nyár, alacsony forgalom
    8:  0.70,   # augusztus — nyári mélyidény
    9:  0.62,   # szeptember — historikusan a LEGGYENGÉBB hónap BTC-nek
    10: 0.92,   # október — "Uptober", Q4 start, historikusan bullish
    11: 0.90,   # november — Q4 bull folytatása
    12: 0.82,   # december — karácsonyi rali vs. év végi nyereség-realizálás
}

# =============================================================================
# Hónap napja (1–31)
# =============================================================================
def _day_of_month_score(day: int) -> float:
    """
    Hónap elejei intézményi beáramlás és hónap végi lejárat/rebalansz
    hatásait modellezi.
    """
    if 1 <= day <= 3:
        return 0.92    # intézményi DCA beáramlás, hó eleje
    elif 4 <= day <= 7:
        return 0.85
    elif 8 <= day <= 21:
        return 0.82    # "mid-month plateau" — kevesebb torzítás
    elif 22 <= day <= 25:
        return 0.78    # közeledik a hó végi expiry / rebalansz
    else:
        return 0.72    # hó vége — lejáratok, intézményi eladás

# =============================================================================
# Speciális nap detektálás
# =============================================================================

def _last_friday(year: int, month: int) -> date:
    """Adott hónap utolsó péntekje."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    offset = (d.weekday() - 4) % 7   # 4 = Friday
    return d - timedelta(days=offset)


def _is_deribit_monthly_expiry(d: date) -> bool:
    """
    Deribit / Bybit havi opció lejárat: minden hónap utolsó péntekje.
    Ez a nap kiszámítható volatilitás-spájkot okoz (08:00 UTC körül).
    """
    return d == _last_friday(d.year, d.month)


def _is_quarterly_expiry(d: date) -> bool:
    """
    Negyedéves futures lejárat: március, június, szeptember, december
    utolsó péntekje. A legnagyobb lejárat (CME, Deribit, Bybit mind).
    """
    return d.month in (3, 6, 9, 12) and _is_deribit_monthly_expiry(d)


def _is_cme_open(dt: datetime) -> bool:
    """
    CME Bitcoin futures nyitva van-e? (hétfő–péntek 22:00–21:00 UTC, zárt
    szombat 21:00 – vasárnap 22:00 UTC). A gap-ek hétfő reggelre hatnak.
    """
    dow = dt.weekday()
    hour = dt.hour
    if dow == 5:   # szombat
        return hour < 21
    if dow == 6:   # vasárnap
        return hour >= 22
    return True


def _is_funding_window(dt: datetime, margin_minutes: int = 25) -> bool:
    """
    Igaz, ha a dt ±margin_minutes percen belül van egy funding reset-től.
    Funding reset: 00:00, 08:00, 16:00 UTC minden nap.
    """
    minutes_in_day = dt.hour * 60 + dt.minute
    for fh in _FUNDING_HOURS:
        funding_minute = fh * 60
        if abs(minutes_in_day - funding_minute) <= margin_minutes:
            return True
        # Éjfél körüli eset (pl. 23:45 → 00:00)
        if funding_minute == 0 and (1440 - minutes_in_day) <= margin_minutes:
            return True
    return False


# =============================================================================
# TimingScore
# =============================================================================

@dataclass
class TimingScore:
    """Időalapú kereskedési feltételek részletes bontásban."""

    overall: float               # 0.0–1.0 kompozit timing score
    intraday: float              # UTC óra szerinti forgalom profil
    day_of_week: float           # hét napja
    monthly_seasonal: float      # havi szezonalitás (hónap + nap a hónapban)
    day_of_month: float          # hónap napja
    is_funding_window: bool      # ±25 perc funding resettől
    is_weekly_expiry: bool       # péntek = Deribit heti opció lejárat
    is_monthly_expiry: bool      # hónap utolsó péntekje
    is_quarterly_expiry: bool    # Q lejárat (márc/jún/szep/dec utolsó péntek)
    is_weekend: bool
    is_cme_open: bool
    position_size_mult: float    # 0.0–1.0 pozícióméret szorzó
    score_threshold_delta: float # buy/sell küszöb módosítás (pozitív = nehezebb)
    hard_block: bool             # True = HOLD kényszer
    notes: List[str] = field(default_factory=list)

    @property
    def trade_label(self) -> str:
        if self.hard_block:       return "BLOCKED"
        if self.overall >= 0.85:  return "OPTIMAL"
        if self.overall >= 0.72:  return "GOOD"
        if self.overall >= 0.60:  return "SUBOPTIMAL"
        return "POOR"

    def __str__(self) -> str:
        notes = " | ".join(self.notes) if self.notes else "—"
        return (
            f"[{self.trade_label}] overall={self.overall:.2f} "
            f"(intra={self.intraday:.2f} dow={self.day_of_week:.2f} "
            f"seasonal={self.monthly_seasonal:.2f}) "
            f"size×{self.position_size_mult:.2f} "
            f"thresh+{self.score_threshold_delta:.2f} "
            f"| {notes}"
        )


# =============================================================================
# MarketTimingAnalyzer
# =============================================================================

class MarketTimingAnalyzer:
    """
    Időalapú piaci aktivitás elemző.

    Paraméterek:
        intraday_weight:  intraday score súlya a kompozitban
        dow_weight:       hét napja súlya
        seasonal_weight:  havi/éves szezonalitás súlya
        dom_weight:       hónap napja súlya
        funding_penalty:  position_size_mult szorzója funding ablakban
        expiry_penalty:   position_size_mult szorzója lejárat napon
    """

    def __init__(
        self,
        intraday_weight:  float = 0.35,
        dow_weight:       float = 0.25,
        seasonal_weight:  float = 0.25,
        dom_weight:       float = 0.15,
        funding_penalty:  float = 0.40,
        expiry_penalty:   float = 0.65,
        weekend_block_threshold: float = 0.55,
    ):
        self.w_intra    = intraday_weight
        self.w_dow      = dow_weight
        self.w_seasonal = seasonal_weight
        self.w_dom      = dom_weight
        self.funding_penalty = funding_penalty
        self.expiry_penalty  = expiry_penalty
        self.weekend_block   = weekend_block_threshold

    def score(self, dt: Optional[datetime] = None) -> TimingScore:
        """
        Kiszámítja a jelenlegi időpont kereskedési minőségét.

        Paraméter:
            dt: datetime UTC timezone-ban (None = most)
        """
        dt = dt or datetime.now(timezone.utc)

        notes: List[str] = []

        # ── Komponens score-ok ────────────────────────────────────────────
        intraday = _HOUR_SCORE.get(dt.hour, 0.70)
        dow      = _DOW_SCORE.get(dt.weekday(), 0.75)
        seasonal = _MONTH_SCORE.get(dt.month, 0.80)
        dom      = _day_of_month_score(dt.day)

        # ── Speciális státuszok ───────────────────────────────────────────
        is_fund_win      = _is_funding_window(dt)
        is_weekend       = dt.weekday() >= 5
        is_weekly_exp    = dt.weekday() == 4   # péntek = Deribit heti exp
        is_monthly_exp   = _is_deribit_monthly_expiry(dt.date())
        is_quarterly_exp = _is_quarterly_expiry(dt.date())
        is_cme           = _is_cme_open(dt)

        # ── Kompozit alap score ───────────────────────────────────────────
        overall = (
            self.w_intra    * intraday  +
            self.w_dow      * dow       +
            self.w_seasonal * seasonal  +
            self.w_dom      * dom
        )

        # ── Pozícióméret szorzó ───────────────────────────────────────────
        pos_mult = self._score_to_pos_mult(overall)

        # Funding ablak büntetés (legszigorúbb — szinte ne lépj be)
        if is_fund_win:
            pos_mult *= self.funding_penalty
            notes.append("⚠️ Funding ablak — belépés kerülendő")

        # Lejárat nap büntetés
        if is_quarterly_exp:
            pos_mult *= self.expiry_penalty
            notes.append("⚠️ Negyedéves futures lejárat")
        elif is_monthly_exp:
            pos_mult *= self.expiry_penalty
            notes.append("⚠️ Havi opció lejárat")
        elif is_weekly_exp:
            pos_mult *= 0.80
            notes.append("ℹ️ Péntek: heti expiry volatilitás")

        # CME gap (hétfő reggel) — kissé óvatosabb
        if dt.weekday() == 0 and dt.hour < 10:
            pos_mult *= 0.90
            notes.append("ℹ️ Hétfő reggel: CME gap fill kockázat")

        # ── Küszöb módosítás ─────────────────────────────────────────────
        threshold_delta = self._score_to_threshold(overall)
        if is_fund_win:
            threshold_delta += 0.12
        if is_quarterly_exp:
            threshold_delta += 0.08

        # ── Hard block ───────────────────────────────────────────────────
        hard_block = False
        if is_weekend and overall < self.weekend_block:
            hard_block = True
            notes.append("🚫 BLOCKED: hétvége + gyenge timing")
        if overall < 0.45:
            hard_block = True
            notes.append("🚫 BLOCKED: extrém rossz timing")

        # ── Informatív megjegyzések ───────────────────────────────────────
        self._add_context_notes(dt, overall, notes)

        return TimingScore(
            overall              = round(max(0.0, min(1.0, overall)), 3),
            intraday             = round(intraday, 3),
            day_of_week          = round(dow, 3),
            monthly_seasonal     = round(seasonal, 3),
            day_of_month         = round(dom, 3),
            is_funding_window    = is_fund_win,
            is_weekly_expiry     = is_weekly_exp,
            is_monthly_expiry    = is_monthly_exp,
            is_quarterly_expiry  = is_quarterly_exp,
            is_weekend           = is_weekend,
            is_cme_open          = is_cme,
            position_size_mult   = round(max(0.0, min(1.0, pos_mult)), 3),
            score_threshold_delta= round(threshold_delta, 3),
            hard_block           = hard_block,
            notes                = notes,
        )

    # ------------------------------------------------------------------ #
    # Privát segédek
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_to_pos_mult(score: float) -> float:
        """Timing score → pozícióméret szorzó."""
        if score >= 0.88:  return 1.00
        if score >= 0.80:  return 0.90
        if score >= 0.72:  return 0.78
        if score >= 0.62:  return 0.60
        if score >= 0.52:  return 0.35
        return 0.15

    @staticmethod
    def _score_to_threshold(score: float) -> float:
        """Timing score → buy/sell küszöb módosítás (pozitív = nehezebb)."""
        if score >= 0.88:  return 0.00
        if score >= 0.80:  return 0.02
        if score >= 0.72:  return 0.05
        if score >= 0.62:  return 0.09
        if score >= 0.52:  return 0.15
        return 0.25

    @staticmethod
    def _add_context_notes(dt: datetime, score: float, notes: List[str]) -> None:
        """Kontextuális magyarázatok a loghoz / Telegramhoz."""
        hour = dt.hour
        dow  = dt.weekday()
        mon  = dt.month

        if 14 <= hour <= 15:
            notes.append("✅ US nyitás — csúcs likviditás")
        elif 9 <= hour <= 11:
            notes.append("✅ EU session — jó likviditás")
        elif hour in (0, 8, 16):
            notes.append("⚠️ Funding reset óra")
        elif hour <= 3:
            notes.append("😴 Ázsiai éjszaka — vékony piac")

        if dow == 2:
            notes.append("📅 Szerda — heti csúcs forgalom")
        elif dow == 5:
            notes.append("📅 Szombat — hétvégi vékony piac")
        elif dow == 6:
            notes.append("📅 Vasárnap — legkisebb forgalom")

        if mon == 9:
            notes.append("📉 Szeptember — historikusan leggyengébb hónap")
        elif mon == 10:
            notes.append("📈 Október — 'Uptober' historikusan bullish")
        elif mon in (11, 12):
            notes.append("📈 Q4 — historikusan a legerősebb negyedév")


# =============================================================================
# Kényelmi függvények
# =============================================================================

_DEFAULT_ANALYZER = MarketTimingAnalyzer()


def score_now() -> TimingScore:
    """Gyors timing score az aktuális UTC időre."""
    return _DEFAULT_ANALYZER.score()


def next_good_window(from_dt: Optional[datetime] = None,
                     min_score: float = 0.80) -> datetime:
    """
    Megkeresi a következő 'good' timing ablakot (overall >= min_score).
    Maximum 24 órán belül keres, óránkénti felbontással.
    """
    dt = from_dt or datetime.now(timezone.utc)
    for i in range(1, 25):
        candidate = dt + timedelta(hours=i)
        s = _DEFAULT_ANALYZER.score(candidate)
        if s.overall >= min_score and not s.hard_block:
            return candidate
    return dt + timedelta(hours=24)  # fallback


def timing_summary(ts: TimingScore, dt: Optional[datetime] = None) -> str:
    """Telegram-barát összefoglaló."""
    dt = dt or datetime.now(timezone.utc)
    emoji = {"OPTIMAL": "🟢", "GOOD": "🟡", "SUBOPTIMAL": "🟠",
             "POOR": "🔴", "BLOCKED": "⛔"}.get(ts.trade_label, "❓")
    lines = [
        f"{emoji} Kereskedési timing: *{ts.trade_label}* ({ts.overall:.2f})",
        f"   Intraday ({dt.hour:02d}:00 UTC): {ts.intraday:.2f}",
        f"   Hét napja ({dt.strftime('%A')}): {ts.day_of_week:.2f}",
        f"   Szezonalitás ({dt.strftime('%B')}): {ts.monthly_seasonal:.2f}",
        f"   Pozícióméret szorzó: ×{ts.position_size_mult:.2f}",
        f"   Küszöb delta: +{ts.score_threshold_delta:.2f}",
    ]
    for note in ts.notes:
        lines.append(f"   {note}")
    return "\n".join(lines)
