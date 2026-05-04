"""
twap.py — TWAP (Time-Weighted Average Price) végrehajtó, spot tradinghez

Nagyobb spot vételi megbízások felosztása időben egyenletesen,
az átlagos teljesítési ár javítása érdekében.

Alapértelmezett konfiguráció:
  - 6 egyenlő szelet (slice)
  - 30 percen belül
  - 5 percenként egy szelet

Abort feltételek:
  - Az ár X%-nál többet mozdul a kezdeti ártól → abort (drágult)
  - Manuális abort() hívás

Spot-only szempontok:
  - Csak vételi TWAP (spot long-only, nincs short)
  - Szeletek piaci megbízásként kerülnek végrehajtásra
  - PaperBroker esetén a buy() nem enged be ha már pozícióban vagyunk,
    ezért a TWAP a belső coin_balance-t kezeli maga (nem a Broker.buy()-t)
  - Valódi éles brókernél a CCXT create_market_buy_order-t hívjuk közvetlen

Használat (Trader.step()-ben):
    if decision.action == "BUY" and twap_config.enabled:
        twap = TWAPExecutor(total_size_usd=notional, config=twap_config)
        twap.start(current_price, timestamp)
        # Következő iterációkban:
        slice_result = twap.tick(current_price, timestamp)
        if slice_result and slice_result.executed:
            # ... logolás, ertesítés ...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

# Pandas opcionális függőség — csak ha elérhető importáljuk
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_AVAILABLE = False

logger = logging.getLogger("twap")


# --------------------------------------------------------------------------- #
# Konfiguráció
# --------------------------------------------------------------------------- #

@dataclass
class TWAPConfig:
    """
    TWAP végrehajtó beállításai.

    Alapértelmezetten ki van kapcsolva (enabled=False); csak nagy megbízásoknál
    érdemes bekapcsolni, ahol az egyszeri piaci ár jelentős slippage-et okozna.
    """

    # Bekapcsolt állapot — False esetén a TWAPExecutor.should_use_twap() False-t ad
    enabled: bool = False

    # Szeletek száma (pl. 6 × 5 perc = 30 perc)
    num_slices: int = 6

    # Teljes végrehajtási ablak másodpercben (alapból 30 perc)
    total_duration_sec: int = 1800

    # Maximális megengedett áreltérés a kezdeti ártól (1.5% felett abort)
    max_price_drift_pct: float = 0.015

    # Minimális megbízásméret USD-ben, ami alatt TWAP nem éri meg
    min_order_size_usd: float = 500.0

    # E felett az összeg felett automatikusan TWAP-ot ajánlott használni
    use_twap_above_usd: float = 2000.0


# --------------------------------------------------------------------------- #
# Adatstruktúrák
# --------------------------------------------------------------------------- #

@dataclass
class TWAPSlice:
    """
    Egy TWAP szelet — a teljes megbízás egy időarányos töredéke.

    A slice_num 1-alapú: az első szelet azonnal végrehajtódik a start() híváskor,
    a többi a tick() hívásokban, amint elérkezik a target_time.
    """

    slice_num: int          # 1-alapú sorszám
    total_slices: int       # összes szelet száma (= TWAPConfig.num_slices)
    size_usd: float         # szelet névértéke USD-ben
    target_time: float      # tervezett végrehajtási unix timestamp

    # Kitöltve végrehajtás után
    executed: bool = False
    fill_price: Optional[float] = None
    fill_time: Optional[float] = None
    note: str = ""


@dataclass
class TWAPResult:
    """
    A TWAP végrehajtás teljes összefoglalója — logoláshoz és audithoz.

    price_improvement_pct > 0  → jobb átlagáron vettük, mint az induláskor
    price_improvement_pct < 0  → drágábban vettük (rosszabb)
    """

    completed: bool                  # minden szelet végrehajtódott
    aborted: bool                    # abort() hívás vagy ár-drift miatt megszakítva
    abort_reason: Optional[str]      # miért állt meg (None ha normálisan befejezte)

    slices_executed: int             # ténylegesen végrehajtott szeletek száma
    slices_total: int                # tervezett összes szelet

    avg_fill_price: float            # súlyozott átlagos teljesítési ár
    total_size_usd: float            # ténylegesen elköltött összeg USD-ben

    total_fee_estimate: float        # becsült összdíj (size_usd × fee_rate × szeletek)

    start_price: float               # ár a TWAP indulásakor
    price_improvement_pct: float     # (start_price - avg_fill) / start_price


# --------------------------------------------------------------------------- #
# Fő végrehajtó osztály
# --------------------------------------------------------------------------- #

class TWAPExecutor:
    """
    TWAP (Time-Weighted Average Price) végrehajtó — spot long-only.

    Egy nagyobb USD megbízást egyenlő méretű szeletekre oszt fel, és időben
    egyenletesen hajtja végre azokat. Az első szelet azonnal végrehajtódik
    (start() híváskor), a többi a tick() hívásokban ütemezetten.

    Abort feltételek:
      1. Az aktuális ár > max_price_drift_pct %-kal eltér az indulóártól
      2. Manuális abort() hívás

    PaperBroker kompatibilitás:
      A PaperBroker.buy() csak akkor enged pozíciót nyitni, ha még nincs nyitva.
      TWAP esetén több szeletet veszünk, ezért a végrehajtást a broker buy()-jától
      FÜGGETLENÜL kell kezelni — a hívó felelőssége a coin_balance és cash
      manuális frissítése, vagy közvetlen CCXT hívás BybitBroker esetén.

    Használat:
        config = TWAPConfig(enabled=True, num_slices=6, total_duration_sec=1800)
        twap = TWAPExecutor(total_size_usd=5000.0, config=config)
        first_slice = twap.start(current_price=45000.0, timestamp=ts)
        # ... következő Trader.step() hívásokban:
        result = twap.tick(current_price, timestamp)
        if result and result.executed:
            broker.coin_balance += result.size_usd / result.fill_price
    """

    def __init__(
        self,
        total_size_usd: float,
        config: Optional[TWAPConfig] = None,
        fee_rate: float = 0.001,
    ) -> None:
        self.config: TWAPConfig = config if config is not None else TWAPConfig()
        self._total_size_usd: float = total_size_usd
        self.fee_rate: float = fee_rate

        # Belső állapot — start() inicializálja
        self._slices: List[TWAPSlice] = []
        self._executed_slices: List[TWAPSlice] = []
        self._start_price: float = 0.0
        self._start_time: float = 0.0
        self._slice_interval_sec: float = 0.0
        self._aborted: bool = False
        self._abort_reason: Optional[str] = None

        # Még nem indult el
        self._started: bool = False

    # ----------------------------------------------------------------------- #
    # Publikus API
    # ----------------------------------------------------------------------- #

    def start(self, current_price: float, timestamp) -> TWAPSlice:
        """
        Inicializálja az ütemtervet és azonnal végrehajtja az első szeletet.

        Az ütemterv egyenlő időközönként osztja el a szeleteket:
          interval = total_duration_sec / (num_slices - 1)
        Az első szelet target_time = start_time (azonnal), az utolsó
        target_time = start_time + total_duration_sec.

        Visszatér az első (már végrehajtott) szelettel.
        """
        self._start_price = current_price
        self._start_time = self._to_unix(timestamp)
        n = self.config.num_slices

        # Szelet méret egyenlő elosztással
        slice_size_usd = self._total_size_usd / n

        # Intervallum: ha n == 1, nincs közte szünet
        if n > 1:
            self._slice_interval_sec = self.config.total_duration_sec / (n - 1)
        else:
            self._slice_interval_sec = 0.0

        # Szelet lista felépítése
        self._slices = [
            TWAPSlice(
                slice_num=i + 1,
                total_slices=n,
                size_usd=slice_size_usd,
                target_time=self._start_time + i * self._slice_interval_sec,
            )
            for i in range(n)
        ]

        self._started = True
        logger.info(
            "TWAP indult: %.2f USD, %d szelet, interval=%.0fs, start_price=%.4f",
            self._total_size_usd,
            n,
            self._slice_interval_sec,
            current_price,
        )

        # Első szelet azonnali végrehajtása
        first = self._slices[0]
        self._execute_slice(first, current_price, self._start_time, note="első szelet")
        return first

    def tick(self, current_price: float, timestamp) -> Optional[TWAPSlice]:
        """
        Ellenőrzi, hogy esedékes-e a következő szelet; ha igen, végrehajtja.

        Abort feltételek (sorrendben ellenőrizve):
          1. Már abortálva van → None
          2. Minden szelet kész → None
          3. Áreltérés > max_price_drift_pct → auto-abort, None

        Visszatér a végrehajtott szelettel, vagy None-nal ha nincs teendő.
        """
        if not self._started:
            logger.warning("tick() hívva start() előtt — figyelmen kívül hagyva.")
            return None

        if self._aborted:
            return None

        # Ár-drift ellenőrzés (szimmetrikus: felfelé és lefelé egyaránt)
        # Felfelé drift: drágábban vásárolunk — abort (eredeti logika)
        # Lefelé drift: veszélyes piaci esés közben veszünk — abort (crash védelem)
        drift = (current_price - self._start_price) / self._start_price
        if abs(drift) > self.config.max_price_drift_pct:
            direction = "emelkedett" if drift > 0 else "esett"
            reason = (
                f"Áreltérés {drift * 100:.2f}% ({direction}) > ±limit "
                f"{self.config.max_price_drift_pct * 100:.2f}% "
                f"(start={self._start_price:.4f}, now={current_price:.4f})"
            )
            self.abort(reason)
            return None

        # Következő nem végrehajtott szelet keresése
        next_slice = self._next_pending_slice()
        if next_slice is None:
            # Minden szelet kész
            return None

        now = self._to_unix(timestamp)
        if now < next_slice.target_time:
            # Még nem esedékes
            return None

        # Esedékes — végrehajtás
        self._execute_slice(next_slice, current_price, now)
        return next_slice

    def abort(self, reason: str) -> None:
        """
        Megszakítja a TWAP végrehajtást.

        A már végrehajtott szeletek érvényesek maradnak; a hátralévők törlésre
        kerülnek (nem lesznek végrehajtva). A get_result().aborted True lesz.
        """
        self._aborted = True
        self._abort_reason = reason
        pending = sum(1 for s in self._slices if not s.executed)
        logger.warning(
            "TWAP abort: %s | Végrehajtva: %d/%d szelet | Hátralévő: %d",
            reason,
            len(self._executed_slices),
            self.config.num_slices,
            pending,
        )

    def is_complete(self) -> bool:
        """
        True ha a TWAP befejezte a működését — akár sikeres befejezéssel,
        akár abort miatt.

        Meg nem indított (start() előtti) executor esetén False-t ad vissza,
        mert a _slices üres lista → all() True lenne (Python vacuous truth),
        ami félrevezető lenne.
        """
        if not self._started:
            return False
        if self._aborted:
            return True
        return all(s.executed for s in self._slices)

    def get_result(self) -> TWAPResult:
        """
        Visszaadja a TWAP végrehajtás teljes összefoglalóját.

        Meghívható bármikor — akár végrehajtás közben is (részleges állapot).
        """
        executed = self._executed_slices
        n_exec = len(executed)

        # Átlagos teljesítési ár (mennyiséggel súlyozva — egyenlő szeleteknél
        # ez ugyanaz mint a számtani átlag, de általánosan is helyes)
        if n_exec > 0:
            total_spent_usd = sum(s.size_usd for s in executed)
            avg_fill = sum(
                s.fill_price * s.size_usd for s in executed if s.fill_price is not None
            ) / total_spent_usd
        else:
            total_spent_usd = 0.0
            avg_fill = self._start_price  # nincs adat, fallback

        # Díjbecslés
        total_fee = total_spent_usd * self.fee_rate

        # Árjavulás: pozitív = olcsóbban vettük, mint a nyitóáron
        if self._start_price > 0:
            improvement = (self._start_price - avg_fill) / self._start_price
        else:
            improvement = 0.0

        completed = (not self._aborted) and all(s.executed for s in self._slices)

        return TWAPResult(
            completed=completed,
            aborted=self._aborted,
            abort_reason=self._abort_reason,
            slices_executed=n_exec,
            slices_total=self.config.num_slices,
            avg_fill_price=avg_fill,
            total_size_usd=total_spent_usd,
            total_fee_estimate=total_fee,
            start_price=self._start_price,
            price_improvement_pct=improvement,
        )

    def describe(self) -> str:
        """
        Ember által olvasható státusz — logoláshoz és Telegram értesítőhöz.

        Például:
          TWAP [3/6] avg=44980.00 drift=+0.12% | next in 287s
        """
        if not self._started:
            return "TWAP [nem indult]"

        result = self.get_result()
        n_exec = result.slices_executed
        n_total = result.slices_total
        avg = result.avg_fill_price
        drift = (
            (self._start_price and
             (avg - self._start_price) / self._start_price * 100)
            or 0.0
        )

        status_parts = [f"TWAP [{n_exec}/{n_total}] avg={avg:.4f} drift={drift:+.2f}%"]

        if self._aborted:
            status_parts.append(f"ABORT: {self._abort_reason}")
        elif result.completed:
            status_parts.append("KÉSZ")
        else:
            # Következő szelet hátralévő ideje
            next_slice = self._next_pending_slice()
            if next_slice is not None:
                now = time.time()
                remaining = max(0.0, next_slice.target_time - now)
                status_parts.append(f"következő {remaining:.0f}s múlva")

        return " | ".join(status_parts)

    # ----------------------------------------------------------------------- #
    # Osztály-szintű segédmetódus
    # ----------------------------------------------------------------------- #

    @classmethod
    def should_use_twap(cls, order_size_usd: float, config: TWAPConfig) -> bool:
        """
        Visszaadja, hogy érdemes-e TWAP-ot használni az adott megbízásmérethez.

        True ha:
          - config.enabled == True
          - order_size_usd >= config.use_twap_above_usd
          - order_size_usd >= config.min_order_size_usd

        Használat a Trader.step()-ben:
            if TWAPExecutor.should_use_twap(notional, twap_config):
                twap = TWAPExecutor(notional, twap_config)
                ...
        """
        if not config.enabled:
            return False
        if order_size_usd < config.min_order_size_usd:
            return False
        if order_size_usd < config.use_twap_above_usd:
            return False
        return True

    # ----------------------------------------------------------------------- #
    # Belső segédmetódusok
    # ----------------------------------------------------------------------- #

    def _execute_slice(
        self,
        slc: TWAPSlice,
        fill_price: float,
        fill_time: float,
        note: str = "",
    ) -> None:
        """Megjelöli a szeletet végrehajtottnak és felveszi az executed listába."""
        slc.executed = True
        slc.fill_price = fill_price
        slc.fill_time = fill_time
        slc.note = note
        self._executed_slices.append(slc)
        logger.info(
            "TWAP szelet %d/%d: %.2f USD @ %.4f | %s",
            slc.slice_num,
            slc.total_slices,
            slc.size_usd,
            fill_price,
            note or "végrehajtva",
        )

    def _next_pending_slice(self) -> Optional[TWAPSlice]:
        """Az első még nem végrehajtott szelet, vagy None ha mind kész."""
        for slc in self._slices:
            if not slc.executed:
                return slc
        return None

    @staticmethod
    def _to_unix(timestamp) -> float:
        """
        Konvertál pd.Timestamp-et vagy float unix értéket float unix mp-re.

        Elfogad:
          - pd.Timestamp (pandas datetime) → .timestamp() metódussal
          - int / float → unix másodperc, változatlanul visszaadja
          - Bármilyen más típus → time.time() fallback + figyelmeztetés
        """
        # pd.Timestamp esetén
        if _PANDAS_AVAILABLE and isinstance(timestamp, pd.Timestamp):
            return timestamp.timestamp()

        # Numerikus — már unix másodperc
        if isinstance(timestamp, (int, float)):
            return float(timestamp)

        # Fallback: próbáljuk meg a .timestamp() metódust (datetime.datetime is)
        try:
            return float(timestamp.timestamp())
        except AttributeError:
            logger.warning(
                "Ismeretlen timestamp típus: %s — time.time() fallback használva.",
                type(timestamp).__name__,
            )
            return time.time()
