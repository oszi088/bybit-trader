"""
stress_test.py — Historikus stresszteszt

Ismert black-swan szcenáriók szimulálása a modell viselkedésének
vizsgálatához. Nem igényel valódi historikus adatot: szintetikus
ársorozatot generál a dokumentált piaci mozgások alapján.

Szcenáriók:
  1. covid_2020:  BTC -60% 14 nap alatt, majd +300% 12 hónapig
  2. luna_2022:   LUNA/altok -90% 3 nap, BTC -40% 1 hónap
  3. ftx_2022:    BTC -30% 7 nap, alts -50%, majd oldalazás
  4. bear_2018:   BTC -84% 12 hónap, folyamatos grinding

A szimulált ársorozatból a Trader döntési logikáját futtatjuk
(ATR-stop, TP, drawdown-limit) anélkül, hogy ML modellt igényelnénk.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("stress_test")


# ---------------------------------------------------------------------------
# Szintetikus árútvonalak segédfüggvényei
# ---------------------------------------------------------------------------

def _log_linspace(start: float, end: float, n: int) -> list[float]:
    """
    Logaritmikusan lineáris (exponenciálisan sima) árút generál.
    start → end értékek n lépésben, log-térben lineárisan interpolálva.
    """
    if n <= 1:
        return [start]
    log_start = math.log(start)
    log_end = math.log(end)
    step = (log_end - log_start) / (n - 1)
    return [math.exp(log_start + step * i) for i in range(n)]


def _sideways(base: float, n: int, amplitude: float = 0.05, seed: int = 42) -> list[float]:
    """Oldalazó ármozgás ±amplitude véletlenszerű ingadozással."""
    rng = random.Random(seed)
    result = []
    price = base
    for _ in range(n):
        pct = rng.uniform(-amplitude, amplitude)
        price = price * (1.0 + pct)
        result.append(price)
    return result


def _grind_down(
    start: float,
    end: float,
    n: int,
    bounce_every: int = 30,
    bounce_pct: float = 0.20,
    seed: int = 99,
) -> list[float]:
    """
    Lassú lefelé grinding bounce-okkal.
    Általában start→end log-lineáris lejtő, de bounce_every-nként
    bounce_pct felfelé visszapattanás simítva.
    """
    base_path = _log_linspace(start, end, n)
    rng = random.Random(seed)
    result = list(base_path)
    i = bounce_every
    while i < n - 5:
        # rövid bounce: néhány bárban +bounce_pct, majd visszaesik
        duration = rng.randint(3, 8)
        peak_i = min(i + duration, n - 1)
        peak_val = result[i] * (1.0 + rng.uniform(bounce_pct * 0.5, bounce_pct))
        # sima interpoláció a bounce csúcsig
        for j in range(i, peak_i + 1):
            frac = (j - i) / max(peak_i - i, 1)
            result[j] = result[i] + (peak_val - result[i]) * frac
        # visszaesés a base_path szintjéhez
        trough_i = min(peak_i + duration, n - 1)
        for j in range(peak_i, trough_i + 1):
            frac = (j - peak_i) / max(trough_i - peak_i, 1)
            result[j] = peak_val + (base_path[min(trough_i, n - 1)] - peak_val) * frac
        i += bounce_every + rng.randint(-5, 5)
    return result


# ---------------------------------------------------------------------------
# Szcenárió konfigurációk
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """Egy stresszteszt-szcenárió leírása."""
    name: str
    description: str
    price_path: list[float]          # normalizált: kezdőár = 1.0
    n_bars: int
    timeframe: str = "1d"


def _build_covid_2020() -> ScenarioConfig:
    # 1.0 → 0.40 in 14 bars (exponential drop)
    drop = _log_linspace(1.0, 0.40, 14)
    # 0.40 → 2.0 in 60 bars (V-recovery)
    recovery = _log_linspace(0.40, 2.0, 61)[1:]   # skip duplicate 0.40
    # 2.0 → 4.0 in 120 bars (bull run)
    bull = _log_linspace(2.0, 4.0, 121)[1:]
    path = drop + recovery + bull
    return ScenarioConfig(
        name="covid_2020",
        description="BTC -60% 14 nap alatt (Covid-crash), majd +300% bull run",
        price_path=path,
        n_bars=len(path),
        timeframe="1d",
    )


def _build_luna_2022() -> ScenarioConfig:
    # 1.0 → 1.1 enyhe emelkedés 10 bárban
    pre = _log_linspace(1.0, 1.1, 10)
    # 1.1 → 0.10 összeomlás 10 bárban
    crash = _log_linspace(1.1, 0.10, 11)[1:]
    # 0.10 → 0.20 gyenge visszapattanás 30 bárban
    bounce = _log_linspace(0.10, 0.20, 31)[1:]
    # 0.20 → 0.15 további lecsorgás 29 bárban
    grind = _log_linspace(0.20, 0.15, 30)[1:]
    path = pre + crash + bounce + grind
    return ScenarioConfig(
        name="luna_2022",
        description="LUNA/alt -90% 3 nap, gyenge visszapattanás, majd grind",
        price_path=path,
        n_bars=len(path),
        timeframe="1d",
    )


def _build_ftx_2022() -> ScenarioConfig:
    # 1.0 → 1.15 emelkedés 10 bárban
    rise = _log_linspace(1.0, 1.15, 10)
    # 1.15 → 0.80 összeomlás 7 bárban
    crash = _log_linspace(1.15, 0.80, 8)[1:]
    # oldalazás ±5% 62 bárban
    side = _sideways(0.80, 62, amplitude=0.05, seed=17)
    path = rise + crash + side
    return ScenarioConfig(
        name="ftx_2022",
        description="FTX-összeomlás: BTC -30% 7 nap, majd tartós oldalazás",
        price_path=path,
        n_bars=len(path),
        timeframe="1d",
    )


def _build_bear_2018() -> ScenarioConfig:
    # 1.0 → 1.3 rövid csúcs 10 bárban
    peak = _log_linspace(1.0, 1.3, 10)
    # 1.3 → 0.16 lassú lefelé grind 250 bárban bounce-okkal
    grind = _grind_down(1.3, 0.16, 251, bounce_every=35, bounce_pct=0.20, seed=2018)[1:]
    path = peak + grind
    return ScenarioConfig(
        name="bear_2018",
        description="2018-as medvepiac: BTC -84% 12 hónap, visszapattanásokkal",
        price_path=path,
        n_bars=len(path),
        timeframe="1d",
    )


SCENARIOS: Dict[str, ScenarioConfig] = {
    "covid_2020": _build_covid_2020(),
    "luna_2022":  _build_luna_2022(),
    "ftx_2022":   _build_ftx_2022(),
    "bear_2018":  _build_bear_2018(),
}


# ---------------------------------------------------------------------------
# Stresszteszt eredmény
# ---------------------------------------------------------------------------

@dataclass
class StressResult:
    """Egy stresszteszt-futás összefoglalója."""
    scenario: str
    max_drawdown_pct: float
    max_consecutive_losses: int
    total_trades: int
    win_rate: float
    final_equity_pct: float          # 1.0 = nincs változás a kezdőtőkéhez képest
    kill_switch_triggered: bool
    kill_switch_reason: Optional[str]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stressztesztelő osztály
# ---------------------------------------------------------------------------

class StressTester:
    """
    Szintetikus ársorozaton futó egyszerű stratégia-szimulátor.

    Stratégia: 10/30 SMA keresztezés ATR-stopokkal.
    Spot-only: nincs short, csak long pozíciók.
    """

    def __init__(
        self,
        fee_rate: float = 0.001,
        initial_cash: float = 10_000.0,
        atr_stop_mult: float = 2.0,
        atr_tp_mult: float = 4.0,
        max_drawdown_pct: float = 0.15,
    ):
        self.fee_rate = fee_rate
        self.initial_cash = initial_cash
        self.atr_stop_mult = atr_stop_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_drawdown_pct = max_drawdown_pct

    # ------------------------------------------------------------------
    # OHLCV generálás
    # ------------------------------------------------------------------

    def _generate_ohlcv(
        self, price_path: list[float], start_price: float = 10_000.0
    ) -> pd.DataFrame:
        """
        Szintetikus OHLCV DataFrame a normalizált árútból.

        A close-ár = price_path[i] * start_price.
        High = close × (1 + rand × 0.01)
        Low  = close × (1 - rand × 0.01)
        Open = előző close (első bár: close-szal egyezik)
        Volume = 1_000_000
        ATR = 14-bár rolling True Range átlaga
        """
        rng = np.random.default_rng(seed=42)
        closes = np.array(price_path) * start_price
        n = len(closes)

        highs  = closes * (1.0 + rng.uniform(0.0, 0.01, size=n))
        lows   = closes * (1.0 - rng.uniform(0.0, 0.01, size=n))
        opens  = np.empty(n)
        opens[0] = closes[0]
        opens[1:] = closes[:-1]

        df = pd.DataFrame({
            "open":   opens,
            "high":   highs,
            "low":    lows,
            "close":  closes,
            "volume": np.full(n, 1_000_000.0),
        })

        # True Range és 14-bár rolling ATR
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14, min_periods=1).mean()

        # SMA-k
        df["sma_fast"] = df["close"].rolling(10, min_periods=1).mean()
        df["sma_slow"] = df["close"].rolling(30, min_periods=1).mean()

        return df

    # ------------------------------------------------------------------
    # Egy szcenárió futtatása
    # ------------------------------------------------------------------

    def run_scenario(
        self, scenario_name: str, start_price: float = 10_000.0
    ) -> StressResult:
        """
        Egyszerű MA-keresztezéses stratégia szimulációja a szcenárión.

        Vételi jel:  sma_fast keresztezi felfelé a sma_slow-t
        Zárási jel:  stop (entry - atr*mult) vagy TP (entry + atr*tp_mult)
                     vagy sma_fast kereszezi lefelé a sma_slow-t

        Kill switch: ha az equity esik max_drawdown_pct-nél jobban a csúcshoz képest.
        """
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Ismeretlen szcenárió: {scenario_name}")

        scenario = SCENARIOS[scenario_name]
        df = self._generate_ohlcv(scenario.price_path, start_price)

        cash = self.initial_cash
        equity = cash
        peak_equity = cash
        position_size = 0.0   # coin mennyiség
        entry_price = 0.0
        stop_price = 0.0
        tp_price = 0.0

        trades: list[dict] = []
        kill_switch = False
        kill_reason: Optional[str] = None
        notes: list[str] = []

        equity_curve: list[float] = [equity]
        max_dd = 0.0

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            close = float(row["close"])
            atr   = float(row["atr"])

            # Equity frissítés
            if position_size > 0:
                equity = cash + position_size * close
            else:
                equity = cash

            equity_curve.append(equity)

            if equity > peak_equity:
                peak_equity = equity

            # Drawdown számítás
            if peak_equity > 0:
                dd = 1.0 - equity / peak_equity
                max_dd = max(max_dd, dd)
                if dd >= self.max_drawdown_pct and not kill_switch:
                    kill_switch = True
                    kill_reason = (
                        f"max drawdown elérve: -{dd:.1%} "
                        f"(peak {peak_equity:.2f})"
                    )
                    logger.warning("[%s] Kill switch: %s", scenario_name, kill_reason)

            if kill_switch:
                # Kényszerzárás
                if position_size > 0:
                    proceeds = position_size * close * (1.0 - self.fee_rate)
                    pnl = proceeds - (position_size * entry_price * (1.0 + self.fee_rate))
                    trades.append({"pnl": pnl, "win": pnl > 0})
                    cash = proceeds
                    position_size = 0.0
                break

            # ---- Nyitott pozíció kezelése ----
            if position_size > 0:
                # Stop-loss
                if close <= stop_price:
                    proceeds = position_size * close * (1.0 - self.fee_rate)
                    pnl = proceeds - (position_size * entry_price * (1.0 + self.fee_rate))
                    trades.append({"pnl": pnl, "win": pnl > 0})
                    cash += proceeds
                    position_size = 0.0
                    logger.debug("[%s] Stop-loss @ %.2f (pnl: %+.2f)", scenario_name, close, pnl)
                    continue

                # Take-profit
                if close >= tp_price:
                    proceeds = position_size * close * (1.0 - self.fee_rate)
                    pnl = proceeds - (position_size * entry_price * (1.0 + self.fee_rate))
                    trades.append({"pnl": pnl, "win": pnl > 0})
                    cash += proceeds
                    position_size = 0.0
                    logger.debug("[%s] TP @ %.2f (pnl: %+.2f)", scenario_name, close, pnl)
                    continue

                # SMA cross zárás (fast < slow)
                if float(row["sma_fast"]) < float(row["sma_slow"]):
                    proceeds = position_size * close * (1.0 - self.fee_rate)
                    pnl = proceeds - (position_size * entry_price * (1.0 + self.fee_rate))
                    trades.append({"pnl": pnl, "win": pnl > 0})
                    cash += proceeds
                    position_size = 0.0
                    logger.debug("[%s] SMA-zárás @ %.2f (pnl: %+.2f)", scenario_name, close, pnl)

            # ---- Belépési jel (csak ha nincs pozíció) ----
            elif position_size == 0:
                fast_cross_up = (
                    float(row["sma_fast"])  > float(row["sma_slow"]) and
                    float(prev["sma_fast"]) <= float(prev["sma_slow"])
                )
                if fast_cross_up and i >= 30:    # legalább 30 bár kell az SMA-hoz
                    # Teljes cash-t fektetjük be (spot: max 95%, díjjal együtt is belefér)
                    invest = cash / (1.0 + self.fee_rate)
                    position_size = invest / close
                    entry_price   = close
                    stop_price    = close - atr * self.atr_stop_mult
                    tp_price      = close + atr * self.atr_tp_mult
                    cash         -= invest * (1.0 + self.fee_rate)
                    if cash < 0:
                        cash = 0.0
                    logger.debug(
                        "[%s] Vétel @ %.2f | stop=%.2f | tp=%.2f",
                        scenario_name, close, stop_price, tp_price,
                    )

        # Nyitott pozíció zárása szcenárió végén
        if position_size > 0:
            last_close = float(df.iloc[-1]["close"])
            proceeds = position_size * last_close * (1.0 - self.fee_rate)
            pnl = proceeds - position_size * entry_price * (1.0 + self.fee_rate)
            trades.append({"pnl": pnl, "win": pnl > 0})
            cash = proceeds
            position_size = 0.0

        # Statisztikák
        total_trades = len(trades)
        wins = sum(1 for t in trades if t["win"])
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        final_equity = cash
        final_equity_pct = final_equity / self.initial_cash

        # Max egymást követő vesztes trade
        max_consec = 0
        consec = 0
        for t in trades:
            if not t["win"]:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        # Megjegyzések
        scenario_notes: list[str] = []
        if kill_switch:
            scenario_notes.append(f"Kill switch aktiválva: {kill_reason}")
        if total_trades == 0:
            scenario_notes.append("Nem volt egyetlen trade sem (pl. nincs SMA-keresztezés).")
        if final_equity_pct < 0.5:
            scenario_notes.append("A végső tőke az induló felénél is kevesebb.")
        if win_rate > 0.6:
            scenario_notes.append(f"Viszonylag jó win rate: {win_rate:.0%}.")

        logger.info(
            "[%s] trades=%d  win_rate=%.0f%%  max_dd=%.1f%%  final=%.1f%%  kill=%s",
            scenario_name, total_trades, win_rate * 100,
            max_dd * 100, final_equity_pct * 100, kill_switch,
        )

        return StressResult(
            scenario=scenario_name,
            max_drawdown_pct=max_dd,
            max_consecutive_losses=max_consec,
            total_trades=total_trades,
            win_rate=win_rate,
            final_equity_pct=final_equity_pct,
            kill_switch_triggered=kill_switch,
            kill_switch_reason=kill_reason,
            notes=scenario_notes,
        )

    # ------------------------------------------------------------------
    # Összes szcenárió
    # ------------------------------------------------------------------

    def run_all(self, start_price: float = 10_000.0) -> list[StressResult]:
        """Mind a négy szcenáriót lefuttatja és visszaadja az eredményeket."""
        results = []
        for name in SCENARIOS:
            logger.info("Stresszteszt futtatása: %s", name)
            result = self.run_scenario(name, start_price=start_price)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Eredmény formázás
    # ------------------------------------------------------------------

    def format_results(self, results: list[StressResult]) -> str:
        """
        Táblázatos összefoglaló az összes stresszteszt-eredményről.
        """
        header = (
            f"{'Szcenárió':<15} "
            f"{'MaxDD':>7} "
            f"{'Trades':>7} "
            f"{'WinRate':>8} "
            f"{'FinalEq':>8} "
            f"{'MaxConsecL':>11} "
            f"{'KillSwitch':>11}"
        )
        sep = "-" * len(header)
        lines = ["", "=== STRESSZTESZT EREDMÉNYEK ===", sep, header, sep]

        for r in results:
            kill_icon = "YES" if r.kill_switch_triggered else "no"
            lines.append(
                f"{r.scenario:<15} "
                f"{r.max_drawdown_pct:>6.1%} "
                f"{r.total_trades:>7d} "
                f"{r.win_rate:>7.0%}  "
                f"{r.final_equity_pct:>7.1%}  "
                f"{r.max_consecutive_losses:>10d} "
                f"{kill_icon:>11}"
            )

        lines.append(sep)

        # Részletes megjegyzések
        for r in results:
            if r.notes or r.kill_switch_reason:
                lines.append(f"\n[{r.scenario}]")
                if r.kill_switch_reason:
                    lines.append(f"  Kill-switch oka: {r.kill_switch_reason}")
                for note in r.notes:
                    lines.append(f"  - {note}")

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parancssori futtatás
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    tester = StressTester()
    results = tester.run_all(start_price=10_000.0)
    print(tester.format_results(results))
