"""
performance_attribution.py — Teljesítmény attribúció

A trade log elemzése szegmensek szerint:
  - Per-coin:   melyik szimbólum volt a legjövedelmezőbb?
  - Per-cycle:  melyik piaci ciklusban volt a legjobb eredmény?
  - Per-signal: melyik jelzés (note mező alapján) hozta a profitot?
  - Per-hour:   melyik napszakban zártak a trade-ek nyereséggel?
  - Összesített: win rate, profit factor, avg holding time, Sharpe

Bemenet: TradeDb.list_trades() által visszaadott dict-ek listája
  {id, timestamp, symbol, side, size, price, fee, pnl, note}

A párosítás: BUY + következő SELL = egy kereskedés (round-trip).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("performance_attribution")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TradeStats:
    """Aggregált statisztikák egy trade-szegmensre."""

    count: int          # lezárt trade-ek száma
    total_pnl: float
    win_count: int
    loss_count: int
    win_rate: float
    avg_pnl: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    best_trade: float
    worst_trade: float


@dataclass
class AttributionReport:
    """Teljes attribúciós riport."""

    by_symbol: Dict[str, TradeStats]
    by_cycle: Dict[str, TradeStats]          # "cycle:bull_mid" stílusú note mezőből
    by_exit_reason: Dict[str, TradeStats]    # "stop_loss", "take_profit", "signal", stb.
    by_hour: Dict[int, TradeStats]           # UTC óra a SELL záráskor
    overall: TradeStats
    generated_at: str                        # ISO formátum
    trade_count: int                         # lezárt round-trip trade-ek száma


# ---------------------------------------------------------------------------
# Attributor
# ---------------------------------------------------------------------------

class PerformanceAttributor:
    """
    Trade log elemzése P&L attribúció szerint.

    Példa:
        attributor = PerformanceAttributor()
        trades = db.list_trades(limit=500)
        report = attributor.generate_report(trades)
        print(attributor.format_report(report))
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_report(self, trades: list[dict]) -> AttributionReport:
        """
        Teljes attribúciós riport generálása a nyers trade listából.

        A lista a TradeDb.list_trades() által visszaadott dict-ek sorozata.
        BUY+SELL round-tripeket párosítunk, majd szegmensek szerint
        aggregáljuk a P&L-t.
        """
        round_trips = self._pair_trades(trades)
        logger.info(
            "Attribution: %d raw trade-ből %d round-trip párosítva.",
            len(trades),
            len(round_trips),
        )

        if not round_trips:
            empty = self._compute_stats([])
            return AttributionReport(
                by_symbol={},
                by_cycle={},
                by_exit_reason={},
                by_hour={},
                overall=empty,
                generated_at=datetime.now(timezone.utc).isoformat(),
                trade_count=0,
            )

        # --- csoportosítás ---
        by_symbol: dict[str, list[float]] = defaultdict(list)
        by_cycle: dict[str, list[float]] = defaultdict(list)
        by_exit_reason: dict[str, list[float]] = defaultdict(list)
        by_hour: dict[int, list[float]] = defaultdict(list)
        all_pnls: list[float] = []

        for rt in round_trips:
            pnl: float = rt["pnl"]
            symbol: str = rt["symbol"]
            note: str = rt.get("note") or ""
            exit_time: Optional[datetime] = rt.get("exit_time")

            all_pnls.append(pnl)
            by_symbol[symbol].append(pnl)

            cycle = self._extract_cycle(note)
            by_cycle[cycle].append(pnl)

            exit_reason = self._extract_exit_reason(note)
            by_exit_reason[exit_reason].append(pnl)

            if exit_time is not None:
                by_hour[exit_time.hour].append(pnl)

        return AttributionReport(
            by_symbol={k: self._compute_stats(v) for k, v in by_symbol.items()},
            by_cycle={k: self._compute_stats(v) for k, v in by_cycle.items()},
            by_exit_reason={k: self._compute_stats(v) for k, v in by_exit_reason.items()},
            by_hour={k: self._compute_stats(v) for k, v in by_hour.items()},
            overall=self._compute_stats(all_pnls),
            generated_at=datetime.now(timezone.utc).isoformat(),
            trade_count=len(round_trips),
        )

    def _pair_trades(self, trades: list[dict]) -> list[dict]:
        """
        BUY-t párosít a következő azonos szimbólumú SELL-lel.

        A listát időrend szerint dolgozzuk fel. Minden szimbólumhoz
        fenntartunk egy FIFO sort a nyitott BUY-okból.

        Returns:
            round-trip dict-ek listája:
            {symbol, entry_time, exit_time, pnl, entry_price, exit_price, note}
        """
        # Rendezés timestamp szerint (list_trades DESC-ben adja vissza)
        sorted_trades = sorted(trades, key=lambda t: t.get("timestamp", ""))

        open_buys: dict[str, list[dict]] = defaultdict(list)
        round_trips: list[dict] = []

        for trade in sorted_trades:
            side = (trade.get("side") or "").upper()
            symbol = trade.get("symbol", "")

            if side == "BUY":
                open_buys[symbol].append(trade)

            elif side == "SELL" and open_buys.get(symbol):
                buy = open_buys[symbol].pop(0)  # FIFO

                entry_time = _parse_ts(buy.get("timestamp"))
                exit_time = _parse_ts(trade.get("timestamp"))

                # A SELL sorban tárolt pnl-t vesszük, ha van; egyébként
                # a buy és sell ár különbségéből közelítjük.
                pnl: float = float(trade.get("pnl") or 0.0)
                if pnl == 0.0:
                    buy_price = float(buy.get("price") or 0.0)
                    sell_price = float(trade.get("price") or 0.0)
                    size = float(buy.get("size") or 0.0)
                    buy_fee = float(buy.get("fee") or 0.0)
                    sell_fee = float(trade.get("fee") or 0.0)
                    pnl = (sell_price - buy_price) * size - buy_fee - sell_fee

                # A note mezőt a SELL-ből vesszük (ott van az exit reason);
                # ha üres, a BUY-ból próbáljuk.
                note = trade.get("note") or buy.get("note") or ""

                round_trips.append({
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "pnl": pnl,
                    "entry_price": float(buy.get("price") or 0.0),
                    "exit_price": float(trade.get("price") or 0.0),
                    "note": note,
                })

        unpaired = sum(len(v) for v in open_buys.values())
        if unpaired:
            logger.debug("%d nyitott BUY maradt párosítatlan.", unpaired)

        return round_trips

    def _compute_stats(self, pnls: list[float]) -> TradeStats:
        """Aggregált statisztikákat számít a P&L listából."""
        if not pnls:
            return TradeStats(
                count=0,
                total_pnl=0.0,
                win_count=0,
                loss_count=0,
                win_rate=0.0,
                avg_pnl=0.0,
                avg_winner=0.0,
                avg_loser=0.0,
                profit_factor=0.0,
                best_trade=0.0,
                worst_trade=0.0,
            )

        winners = [p for p in pnls if p > 0.0]
        losers = [p for p in pnls if p < 0.0]
        n = len(pnls)

        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0.0
            else (float("inf") if gross_profit > 0.0 else 0.0)
        )

        return TradeStats(
            count=n,
            total_pnl=sum(pnls),
            win_count=len(winners),
            loss_count=len(losers),
            win_rate=len(winners) / n,
            avg_pnl=sum(pnls) / n,
            avg_winner=sum(winners) / len(winners) if winners else 0.0,
            avg_loser=sum(losers) / len(losers) if losers else 0.0,
            profit_factor=profit_factor,
            best_trade=max(pnls),
            worst_trade=min(pnls),
        )

    def _extract_cycle(self, note: str) -> str:
        """
        Kinyeri a 'cycle:<name>' mintát a note mezőből.

        Példa: "cycle:bull_mid entry=signal" → "bull_mid"
        Returns "unknown" ha nincs ilyen minta.
        """
        if not note:
            return "unknown"
        match = re.search(r"cycle:(\w+)", note)
        return match.group(1) if match else "unknown"

    def _extract_exit_reason(self, note: str) -> str:
        """
        Kinyeri az exit reason-t a note mezőből.

        Ismert kulcsszavak (sorrendben): stop_loss, take_profit, signal,
        trailing_stop, timeout, manual.
        Returns "other" ha egyik sem található.
        """
        if not note:
            return "other"
        # FIX #2: az ExitManager által generált note-okhoz igazítva.
        # A "|cycle:xxx" utótagot levágjuk az elemzés előtt.
        note_clean = note.split("|")[0].strip().lower()
        for keyword in (
            "stop_loss",
            "take_profit",
            "trailing_stop",
            "partial_tp",    # ExitManager részleges TP
            "profit_lock",   # ExitManager profit lock
            "time_exit",     # ExitManager időalapú exit
            "timeout",
            "manual",
            "signal",
        ):
            if keyword in note_clean:
                return keyword
        return "other"

    def format_report(self, report: AttributionReport) -> str:
        """
        Emberi olvashatóságú, Telegram-barát formátumú riport.

        Emojik: 📊 fejléc, ✅ nyerő, ❌ veszítő, 🏆 legjobb.
        """
        lines: list[str] = []

        lines.append(f"📊 *Performance Attribution Report*")
        lines.append(f"Generated: {report.generated_at}")
        lines.append(f"Total round-trips: {report.trade_count}")
        lines.append("")

        # --- Összesített ---
        ov = report.overall
        lines.append("📊 *Overall*")
        lines.append(
            f"  Trades: {ov.count} | "
            f"✅ {ov.win_count} | ❌ {ov.loss_count} | "
            f"WR: {ov.win_rate:.1%}"
        )
        lines.append(
            f"  Total PnL: {ov.total_pnl:+.4f} | "
            f"Avg: {ov.avg_pnl:+.4f} | "
            f"PF: {ov.profit_factor:.3f}"
        )
        lines.append(
            f"  🏆 Best: {ov.best_trade:+.4f} | "
            f"Worst: {ov.worst_trade:+.4f}"
        )
        lines.append("")

        # --- Per-symbol ---
        if report.by_symbol:
            lines.append("📊 *By Symbol*")
            sorted_symbols = sorted(
                report.by_symbol.items(),
                key=lambda kv: kv[1].total_pnl,
                reverse=True,
            )
            for symbol, st in sorted_symbols:
                lines.append(
                    f"  {symbol}: {st.count} trades | "
                    f"PnL {st.total_pnl:+.4f} | "
                    f"WR {st.win_rate:.1%} | "
                    f"PF {st.profit_factor:.2f}"
                )
            lines.append("")

        # --- Per-cycle ---
        if report.by_cycle:
            lines.append("📊 *By Market Cycle*")
            sorted_cycles = sorted(
                report.by_cycle.items(),
                key=lambda kv: kv[1].total_pnl,
                reverse=True,
            )
            for cycle, st in sorted_cycles:
                lines.append(
                    f"  {cycle}: {st.count} trades | "
                    f"PnL {st.total_pnl:+.4f} | "
                    f"WR {st.win_rate:.1%}"
                )
            lines.append("")

        # --- Per-exit-reason ---
        if report.by_exit_reason:
            lines.append("📊 *By Exit Reason*")
            sorted_reasons = sorted(
                report.by_exit_reason.items(),
                key=lambda kv: kv[1].total_pnl,
                reverse=True,
            )
            for reason, st in sorted_reasons:
                icon = "✅" if st.total_pnl >= 0 else "❌"
                lines.append(
                    f"  {icon} {reason}: {st.count} trades | "
                    f"PnL {st.total_pnl:+.4f} | "
                    f"WR {st.win_rate:.1%}"
                )
            lines.append("")

        # --- Per-hour ---
        if report.by_hour:
            lines.append("📊 *By UTC Hour (top 5)*")
            sorted_hours = sorted(
                report.by_hour.items(),
                key=lambda kv: kv[1].total_pnl,
                reverse=True,
            )
            for hour, st in sorted_hours[:5]:
                lines.append(
                    f"  {hour:02d}:00 UTC — {st.count} trades | "
                    f"PnL {st.total_pnl:+.4f} | "
                    f"WR {st.win_rate:.1%}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """ISO timestamp stringet datetime-má alakít (UTC-aware)."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        logger.debug("Nem sikerült parse-olni a timestampot: %r", ts_str)
        return None
