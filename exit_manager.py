"""
exit_manager.py — Professzionális exit logika spot tradinghez

A Trader._check_exits() és _update_trailing() helyett ezt kell használni.
Öt exit-feltétel hierarchikusan, sorrendben:

1. Stop-loss hit  (kötelező, mindig ellenőrizve)
2. Take-profit 1  (TP1): pozíció 50%-ának zárása 1×ATR haszon esetén
   → "partial_tp" után a stop felhúzódik break-even re (entry price)
3. Trailing stop  (a TP1 triggerelése után aktív az egész pozícióra)
4. Profit lock    (ha TP1 már volt, és az ár >0.5×ATR-t esik a helyi csúcsról)
5. Time exit      (max_holding_bars lejártakor mindenképp kizár)

Spot-only megjegyzések:
  - Nincs short pozíció → csak long exit logika
  - A "partial sell" a PaperBroker-ben nincs implementálva natívan,
    ezért az ExitManager jelzést ad vissza, a Trader valósítja meg
  - TWAP exit nem szükséges (kis pozíció spot-on)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("exit_manager")


# =============================================================================
# ExitConfig — exit paraméterek dataclass
# =============================================================================

@dataclass
class ExitConfig:
    """
    Az ExitManager viselkedését vezérlő paraméterek.

    A Trader vagy az Agent példányosítja, és átadja az ExitManager-nek
    belépéskor (on_entry). CycleRegimeParams.max_holding_bars értékét
    érdemes a time_exit_bars mezőbe tölteni.
    """

    # Részleges zárás (TP1) aránya: 0.5 = pozíció felét zárjuk
    partial_tp_fraction: float = 0.5

    # TP1 szintje: entry + tp1_atr_mult × ATR
    tp1_atr_mult: float = 1.0

    # Trailing stop távolsága a helyi csúcstól (ATR-ban)
    trailing_atr_mult: float = 1.5

    # Profit lock: ennyi ATR-t esik a csúcsról → zárunk
    profit_lock_atr_mult: float = 0.5

    # Time exit: 0 = kikapcsolt; CycleRegimeParams.max_holding_bars-ból töltjük
    time_exit_bars: int = 0

    # Ha True, TP1 után a stop = entry (break-even), így nem lehet veszteséges
    breakeven_after_partial: bool = True


# =============================================================================
# ExitSignal — az on_bar() visszatérési értéke
# =============================================================================

@dataclass
class ExitSignal:
    """
    Az ExitManager által visszaadott exit jelzés.

    A Trader ezt kapja, és a mezők alapján hajtja végre az akciót:
      - should_exit=False  → ne csinálj semmit (csak esetleg frissítsd a stopot)
      - is_partial=True    → csak exit_fraction arányú részt add el
      - stop_updated=True  → frissítsd a stop-ot new_stop_price-ra
    """

    # Kell-e bármilyen záró akció?
    should_exit: bool

    # True = részleges zárás (partial_tp esetén), False = teljes zárás
    is_partial: bool

    # 0.0–1.0: a pozíció hány hányadát kell eladni (0.5 = fele)
    exit_fraction: float

    # Az exit oka (loggoláshoz, DB-hez)
    # Lehetséges értékek: "partial_tp", "trailing_stop", "time_exit",
    #                     "profit_lock", "stop_loss", "take_profit"
    reason: str

    # Ha True, a Trader-nek frissítenie kell a belső stop szintjét
    stop_updated: bool

    # Az új stop ár (csak ha stop_updated=True)
    new_stop_price: Optional[float]

    # Emberi olvasható megjegyzés loggoláshoz
    note: str


# Üres jelzés: nincs teendő (factory helper)
def _no_signal() -> ExitSignal:
    """Üres jelzés — nincs exit, nincs stop frissítés."""
    return ExitSignal(
        should_exit=False,
        is_partial=False,
        exit_fraction=0.0,
        reason="",
        stop_updated=False,
        new_stop_price=None,
        note="",
    )


# =============================================================================
# ExitManager — az öt exit-feltétel hierarchikus kezelője
# =============================================================================

class ExitManager:
    """
    Professzionális exit logika spot long pozíciókhoz.

    Használat:
        em = ExitManager(config)
        em.on_entry(entry_price=50000, atr=800, atr_tp_mult=4.0,
                    atr_stop_mult=2.0, max_holding_bars=25)

        # Minden gyertya zárásakor:
        signal = em.on_bar(current_price=51200, current_atr=820)
        if signal.should_exit:
            trader.execute_exit(signal)

    Az öt feltétel sorrendben (az első triggerelt lép életbe):
        1. Stop-loss
        2. TP1 (részleges, ha partial_tp_done még nem igaz)
        3. Trailing stop (csak TP1 után aktív)
        4. Profit lock   (csak TP1 után aktív)
        5. Time exit     (ha time_exit_bars > 0)
    """

    def __init__(self, config: Optional[ExitConfig] = None) -> None:
        self.config: ExitConfig = config or ExitConfig()

        # --- Belső állapot ---
        self._entry_price: Optional[float] = None
        self._initial_atr: float = 0.0
        self._stop_price: Optional[float] = None
        self._tp1_price: Optional[float] = None        # TP1 (részleges zárás szintje)
        self._tp_full_price: Optional[float] = None   # Teljes TP: entry + atr_tp_mult × atr
        self._highest_price: Optional[float] = None
        self._partial_tp_done: bool = False            # Volt-e már TP1 zárás?
        self._bars_held: int = 0
        self._remaining_fraction: float = 1.0         # Kezdetben 100%, TP1 után 50%

    # -------------------------------------------------------------------------
    # Publikus API
    # -------------------------------------------------------------------------

    def on_entry(
        self,
        entry_price: float,
        atr: float,
        atr_tp_mult: float = 4.0,
        atr_stop_mult: float = 2.0,
        max_holding_bars: int = 0,
    ) -> None:
        """
        Belépéskor hívandó. Beállítja az összes stop/TP szintet, nullázza az állapotot.

        Paraméterek:
            entry_price      : Belépési ár
            atr              : Az aktuális ATR érték belépéskor
            atr_tp_mult      : Teljes TP szorzója (CycleRegimeParams.atr_tp_mult)
            atr_stop_mult    : Stop-loss szorzója (CycleRegimeParams.atr_stop_mult)
            max_holding_bars : Max tartási idő gyertyákban (0 = kikapcsolt)
        """
        self.reset()

        if atr <= 0:
            logger.warning(
                "on_entry: atr=%.4f <= 0, stop/TP szintek nincsenek beállítva.", atr
            )

        self._entry_price = entry_price
        self._initial_atr = atr

        # Stop-loss: entry - atr_stop_mult × ATR
        self._stop_price = entry_price - atr_stop_mult * atr

        # TP1: entry + tp1_atr_mult × ATR (részleges zárás)
        self._tp1_price = entry_price + self.config.tp1_atr_mult * atr

        # Teljes TP: entry + atr_tp_mult × ATR (a CycleRegimeParams szerinti)
        self._tp_full_price = entry_price + atr_tp_mult * atr

        # Legmagasabb ár: belépési ártól indul
        self._highest_price = entry_price

        # Time exit: az on_entry max_holding_bars felülírja a config értékét,
        # ha meg van adva (> 0). Így a CycleRegimeParams dinamikusan átadható.
        if max_holding_bars > 0:
            self.config.time_exit_bars = max_holding_bars

        logger.info(
            "ExitManager.on_entry | entry=%.4f atr=%.4f "
            "SL=%.4f TP1=%.4f TP_full=%.4f max_bars=%d",
            entry_price, atr,
            self._stop_price,
            self._tp1_price,
            self._tp_full_price,
            self.config.time_exit_bars,
        )

    def on_bar(self, current_price: float, current_atr: float) -> ExitSignal:
        """
        Minden gyertya zárásakor hívandó. Ellenőrzi az öt exit-feltételt sorban.

        Visszatérési érték:
            ExitSignal — a Trader alapján hajtja végre (vagy hagyja figyelmen kívül)

        Fontos: a _bars_held számlálót mindig növeli, így a time exit helyesen működik.
        """
        # Ha nincs aktív belépés, üres jelzéssel térünk vissza
        if self._entry_price is None:
            return _no_signal()

        # Legmagasabb ár frissítése (minden logika előtt)
        self._update_highest_price(current_price)

        # Gyertyaszámláló növelése
        self._bars_held += 1

        # ── 1. Stop-loss ──────────────────────────────────────────────────────
        signal = self._check_stop_loss(current_price)
        if signal is not None:
            logger.info(
                "EXIT [stop_loss] ár=%.4f stop=%.4f bars=%d",
                current_price, self._stop_price, self._bars_held,
            )
            return signal

        # ── 2. TP1 (részleges zárás) ─────────────────────────────────────────
        signal = self._check_partial_tp(current_price)
        if signal is not None:
            logger.info(
                "EXIT [partial_tp] ár=%.4f tp1=%.4f fraction=%.2f bars=%d",
                current_price, self._tp1_price, signal.exit_fraction, self._bars_held,
            )
            return signal

        # ── 3. Trailing stop (csak TP1 után) ─────────────────────────────────
        signal = self._check_trailing_stop(current_price, current_atr)
        if signal is not None:
            logger.info(
                "EXIT [trailing_stop] ár=%.4f highest=%.4f bars=%d",
                current_price, self._highest_price, self._bars_held,
            )
            return signal

        # ── 4. Profit lock (csak TP1 után) ───────────────────────────────────
        signal = self._check_profit_lock(current_price, current_atr)
        if signal is not None:
            logger.info(
                "EXIT [profit_lock] ár=%.4f highest=%.4f bars=%d",
                current_price, self._highest_price, self._bars_held,
            )
            return signal

        # ── 5. Time exit ──────────────────────────────────────────────────────
        signal = self._check_time_exit(current_price)
        if signal is not None:
            logger.info(
                "EXIT [time_exit] ár=%.4f bars=%d / max=%d",
                current_price, self._bars_held, self.config.time_exit_bars,
            )
            return signal

        # Egyik feltétel sem teljesült
        return _no_signal()

    def reset(self) -> None:
        """Teljes állapot törlése (pozíció zárása után hívandó)."""
        self._entry_price = None
        self._initial_atr = 0.0
        self._stop_price = None
        self._tp1_price = None
        self._tp_full_price = None
        self._highest_price = None
        self._partial_tp_done = False
        self._bars_held = 0
        self._remaining_fraction = 1.0
        logger.debug("ExitManager.reset — állapot törölve.")

    def describe(self) -> str:
        """Emberi olvasható állapot összefoglaló (loggoláshoz / Telegramhoz)."""
        if self._entry_price is None:
            return "ExitManager: nincs aktív pozíció"

        partial_str = "IGEN" if self._partial_tp_done else "NEM"
        return (
            f"ExitManager | "
            f"entry={self._entry_price:.4f} "
            f"atr0={self._initial_atr:.4f} "
            f"stop={self._stop_price:.4f} "
            f"tp1={self._tp1_price:.4f} "
            f"tp_full={self._tp_full_price:.4f} "
            f"highest={self._highest_price:.4f} "
            f"partial_tp_done={partial_str} "
            f"bars={self._bars_held}/{self.config.time_exit_bars or '∞'} "
            f"remaining={self._remaining_fraction:.2f}"
        )

    # -------------------------------------------------------------------------
    # Privát exit-feltételek
    # -------------------------------------------------------------------------

    def _check_stop_loss(self, price: float) -> Optional[ExitSignal]:
        """
        1. Stop-loss: ha az ár a stop alá esik, teljes pozíció zárása.

        Mindig aktív, TP1 előtt és után egyaránt.
        """
        if self._stop_price is None:
            return None

        if price <= self._stop_price:
            return ExitSignal(
                should_exit=True,
                is_partial=False,
                exit_fraction=self._remaining_fraction,
                reason="stop_loss",
                stop_updated=False,
                new_stop_price=None,
                note=(
                    f"Stop-loss: ár={price:.4f} <= stop={self._stop_price:.4f} "
                    f"| bars={self._bars_held}"
                ),
            )
        return None

    def _check_partial_tp(self, price: float) -> Optional[ExitSignal]:
        """
        2. TP1 (részleges zárás): ha az ár eléri a tp1_price szintet
           és még nem volt részleges zárás.

        Hatás:
          - partial_tp_fraction arányú részt zárjuk (alapértelmezett: 50%)
          - remaining_fraction csökken
          - Ha breakeven_after_partial=True, a stop = entry price-ra húzódik fel
          - partial_tp_done = True (a trailing + profit lock ettől aktiválódik)
        """
        if self._partial_tp_done:
            return None
        if self._tp1_price is None:
            return None

        if price >= self._tp1_price:
            # Break-even stop beállítása
            new_stop: Optional[float] = None
            stop_updated = False
            if self.config.breakeven_after_partial and self._entry_price is not None:
                new_stop = self._entry_price
                self._stop_price = new_stop
                stop_updated = True

            # Belső állapot frissítése
            self._partial_tp_done = True
            fraction = self.config.partial_tp_fraction
            self._remaining_fraction = round(1.0 - fraction, 8)

            return ExitSignal(
                should_exit=True,
                is_partial=True,
                exit_fraction=fraction,
                reason="partial_tp",
                stop_updated=stop_updated,
                new_stop_price=new_stop,
                note=(
                    f"TP1: ár={price:.4f} >= tp1={self._tp1_price:.4f} "
                    f"| zárjuk: {fraction*100:.0f}% "
                    f"| új stop (break-even)={new_stop} "
                    f"| bars={self._bars_held}"
                ),
            )
        return None

    def _check_trailing_stop(self, price: float, atr: float) -> Optional[ExitSignal]:
        """
        3. Trailing stop: csak TP1 után aktív.

        A trailing stop = highest_price - trailing_atr_mult × atr.
        A stop csak felfele húzódhat (soha nem lazul).
        Ha az ár a trailing stop alá esik, a maradék pozíciót zárjuk.
        """
        if not self._partial_tp_done:
            return None
        if self._highest_price is None or atr <= 0:
            return None

        trailing_stop = self._highest_price - self.config.trailing_atr_mult * atr

        # Trailing stop csak felfele húzható
        if self._stop_price is None or trailing_stop > self._stop_price:
            self._stop_price = trailing_stop
            logger.debug(
                "Trailing stop frissítve: %.4f (highest=%.4f atr=%.4f mult=%.2f)",
                self._stop_price, self._highest_price, atr, self.config.trailing_atr_mult,
            )

        if price <= self._stop_price:
            return ExitSignal(
                should_exit=True,
                is_partial=False,
                exit_fraction=self._remaining_fraction,
                reason="trailing_stop",
                stop_updated=False,
                new_stop_price=None,
                note=(
                    f"Trailing stop: ár={price:.4f} <= trailing={self._stop_price:.4f} "
                    f"| highest={self._highest_price:.4f} "
                    f"| bars={self._bars_held}"
                ),
            )
        return None

    def _check_profit_lock(self, price: float, atr: float) -> Optional[ExitSignal]:
        """
        4. Profit lock: csak TP1 után aktív.

        Ha az ár a helyi csúcshoz képest profit_lock_atr_mult × ATR-t esik,
        a maradék pozíciót zárjuk (profitot rögzítjük).

        Ez durvább szűrő, mint a trailing stop — gyorsabban reagál egy
        hirtelen visszaesésre, nem vár a folyamatos ATR-alapú számításra.
        """
        if not self._partial_tp_done:
            return None
        if self._highest_price is None or atr <= 0:
            return None

        profit_lock_level = self._highest_price - self.config.profit_lock_atr_mult * atr

        if price < profit_lock_level:
            return ExitSignal(
                should_exit=True,
                is_partial=False,
                exit_fraction=self._remaining_fraction,
                reason="profit_lock",
                stop_updated=False,
                new_stop_price=None,
                note=(
                    f"Profit lock: ár={price:.4f} < lock_level={profit_lock_level:.4f} "
                    f"| highest={self._highest_price:.4f} "
                    f"| esés={self._highest_price - price:.4f} "
                    f"| bars={self._bars_held}"
                ),
            )
        return None

    def _check_time_exit(self, price: float) -> Optional[ExitSignal]:
        """
        5. Time exit: ha a pozíció bars_held >= time_exit_bars gyertyán át nyitva volt.

        Ha time_exit_bars == 0, a feltétel ki van kapcsolva.
        """
        if self.config.time_exit_bars <= 0:
            return None

        if self._bars_held >= self.config.time_exit_bars:
            return ExitSignal(
                should_exit=True,
                is_partial=False,
                exit_fraction=self._remaining_fraction,
                reason="time_exit",
                stop_updated=False,
                new_stop_price=None,
                note=(
                    f"Time exit: bars={self._bars_held} >= max={self.config.time_exit_bars} "
                    f"| ár={price:.4f}"
                ),
            )
        return None

    # -------------------------------------------------------------------------
    # Segédfüggvény
    # -------------------------------------------------------------------------

    def _update_highest_price(self, price: float) -> None:
        """
        A belső legmagasabb ár frissítése.

        Az on_bar() minden hívás elején meghívja, mielőtt bármilyen
        exit-feltételt ellenőriz. Így a trailing stop és profit lock
        mindig a valódi lokális csúcshoz viszonyít.
        """
        if self._highest_price is None or price > self._highest_price:
            self._highest_price = price
