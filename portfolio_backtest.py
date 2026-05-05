"""
Portfolió-backtester: fix tőkével, párhuzamos multi-coin szimuláció.

Egy közös tőkekészletből (initial_balance) egyszerre több coin OHLCV
adatán kereskedik — max max_positions nyitott pozíció megengedett.
Az equity curve az összes pozíció mark-to-market értékét tükrözi.

Különbség a multi_backtest.py-tól:
  multi_backtest     : minden coin KÜLÖN tőkéből fut (batch statisztika)
  portfolio_backtest : MEGOSZTOTT tőke, valós portfólió-hatások

Slot-alapú allokáció:
  slot_size = initial_balance / max_positions
  Minden belépéskor egy slot értékét allokáljuk — ha nincs elég szabad
  cash (< slot_size * 0.5), nem lépünk be újabb pozícióba.

CLI:
    python main.py portbt --data-dir data
    python main.py portbt --data-dir data --max-positions 3 --objective calmar
    python main.py portbt --data-dir data --initial-balance 5000 --max-positions 5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from agent import TradingAgent
from backtest import (
    _apply_slippage, _check_sl_tp, _close_position,
    _initial_stops, _max_drawdown, Trade,
)
from config import TradingConfig

logger = logging.getLogger("portfolio_backtest")


# ============================================================================
# Adatosztalyok
# ============================================================================

@dataclass
class PortfolioTrade(Trade):
    """Trade kiterjesztve a coin szimbólumával."""
    symbol: str = ""


@dataclass
class PortfolioBacktestResult:
    equity_curve: pd.Series
    trades: List[PortfolioTrade]
    per_symbol: Dict[str, List[PortfolioTrade]]
    final_balance: float
    total_return_pct: float
    initial_balance: float

    def summary(self) -> str:
        wins = [t for t in self.trades if t.pnl > 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0.0
        max_dd = _max_drawdown(self.equity_curve)
        calmar = self.total_return_pct / max_dd if max_dd > 0.01 else 0.0
        lines = [
            f"Trades: {len(self.trades)} | Win%: {win_rate:.1f}% | "
            f"Final: ${self.final_balance:,.2f} | "
            f"Return: {self.total_return_pct:+.2f}% | "
            f"Max DD: {max_dd:.1f}% | Calmar: {calmar:.2f}",
            "",
            f"  {'Coin':<6} {'Trades':>6} {'PnL $':>10} {'Win%':>6}",
            "  " + "-" * 34,
        ]
        for sym in sorted(self.per_symbol):
            ts = self.per_symbol[sym]
            if not ts:
                lines.append(f"  {sym:<6} {'0':>6} {'—':>10} {'—':>6}")
                continue
            sym_wins = sum(1 for t in ts if t.pnl > 0)
            sym_pnl = sum(t.pnl for t in ts)
            sym_win_pct = sym_wins / len(ts) * 100
            lines.append(
                f"  {sym:<6} {len(ts):>6} {sym_pnl:>+10.2f} {sym_win_pct:>5.1f}%"
            )
        return "\n".join(lines)


@dataclass
class _Slot:
    """Egy nyitott pozíció belső nyilvántartása."""
    trade: PortfolioTrade
    stop_price: Optional[float]
    tp_price: Optional[float]
    highest_price: float


# ============================================================================
# Portfolió-backteszter
# ============================================================================

class PortfolioBacktester:
    """
    Megosztott tőkéjű, egyidejű multi-coin backteszter.

    Paraméterek:
        base_config     : alap TradingConfig (minden coinra ugyanaz)
        initial_balance : induló tőke (USD)
        max_positions   : max egyidejű nyitott pozíció
        per_symbol_cfg  : opcionális coin-specifikus config felülírás
    """

    def __init__(
        self,
        base_config: TradingConfig,
        initial_balance: float = 10_000.0,
        max_positions: int = 3,
        per_symbol_cfg: Optional[Dict[str, TradingConfig]] = None,
    ):
        self.base_config = base_config
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.per_symbol_cfg = per_symbol_cfg or {}

    def run(self, datasets: Dict[str, pd.DataFrame]) -> PortfolioBacktestResult:
        cfg = self.base_config
        bt_cfg = cfg.backtest
        stops = cfg.stops

        # ── 1. Ügynökök felkészítése ─────────────────────────────────────────
        agents: Dict[str, TradingAgent] = {}
        enriched: Dict[str, pd.DataFrame] = {}
        ts_to_idx: Dict[str, Dict[pd.Timestamp, int]] = {}

        for sym, df in datasets.items():
            sym_cfg = self.per_symbol_cfg.get(sym, cfg)
            agent = TradingAgent(sym_cfg)
            agent.config.mtf.enabled = False   # portfolió módban MTF ki (sebesség)
            enr = agent.prepare(df)
            agents[sym] = agent
            enriched[sym] = enr
            ts_to_idx[sym] = {ts: i for i, ts in enumerate(enr.index)}

        # ── 2. Közös idővonal ────────────────────────────────────────────────
        all_ts: List[pd.Timestamp] = sorted(
            set().union(*[set(enr.index) for enr in enriched.values()])
        )

        # ── 3. Szimuláció ────────────────────────────────────────────────────
        cash = float(self.initial_balance)
        slot_size = self.initial_balance / self.max_positions

        slots: Dict[str, _Slot] = {}           # symbol -> aktív slot
        all_trades: List[PortfolioTrade] = []
        equity_history: List[float] = []
        equity_index: List[pd.Timestamp] = []

        for ts in all_ts:
            just_exited: Set[str] = set()

            # --- SL / TP ellenőrzés (bar high/low alapján) ------------------
            for sym in list(slots.keys()):
                if ts not in ts_to_idx.get(sym, {}):
                    continue
                idx = ts_to_idx[sym][ts]
                row = enriched[sym].iloc[idx]
                slot = slots[sym]
                price = float(row["close"])
                atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0

                # Trailing stop frissítés
                if stops.use_trailing_stop:
                    if price > slot.highest_price:
                        slot.highest_price = price
                    if atr > 0:
                        new_stop = slot.highest_price - stops.trailing_atr_mult * atr
                        if slot.stop_price is None or new_stop > slot.stop_price:
                            slot.stop_price = new_stop

                exit_reason = _check_sl_tp(row, slot.stop_price, slot.tp_price)
                if exit_reason is not None:
                    fill = slot.stop_price if exit_reason == "stop_loss" else slot.tp_price
                    cash = _close_position(
                        slot.trade, fill or price, ts, cash, cfg, bt_cfg, exit_reason
                    )
                    all_trades.append(slot.trade)
                    del slots[sym]
                    just_exited.add(sym)

            # --- Ügynök döntések és belépések / jelzés alapú kilépések ------
            for sym, agent in agents.items():
                if ts not in ts_to_idx.get(sym, {}):
                    continue
                idx = ts_to_idx[sym][ts]
                row = enriched[sym].iloc[idx]
                price = float(row["close"])
                atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
                decision = agent.decide_at(idx)

                if (decision.action == "BUY"
                        and sym not in slots
                        and sym not in just_exited
                        and len(slots) < self.max_positions
                        and cash >= slot_size * 0.5):

                    atr_pct = atr / price if price > 0 and atr > 0 else 0.0
                    if atr_pct >= cfg.risk.max_atr_pct:
                        continue   # túl volatilis

                    alloc = min(slot_size, cash)
                    size_mult = abs(decision.score) if cfg.risk.score_proportional_size else 1.0
                    fill = _apply_slippage(price, "BUY", bt_cfg)
                    size = alloc * size_mult / (fill * (1 + cfg.fee_rate))

                    trade = PortfolioTrade(
                        entry_time=ts,
                        entry_price=fill,
                        size=size,
                        symbol=sym,
                    )
                    cash -= size * fill * (1 + cfg.fee_rate)
                    sl, tp = _initial_stops(fill, atr, stops)
                    slots[sym] = _Slot(trade, sl, tp, fill)

                elif decision.action == "SELL" and sym in slots:
                    slot = slots[sym]
                    cash = _close_position(
                        slot.trade, price, ts, cash, cfg, bt_cfg, "signal"
                    )
                    all_trades.append(slot.trade)
                    del slots[sym]

            # --- Equity mark-to-market --------------------------------------
            mark = cash
            for sym, slot in slots.items():
                if ts in ts_to_idx.get(sym, {}):
                    idx = ts_to_idx[sym][ts]
                    mark += slot.trade.size * float(enriched[sym].iloc[idx]["close"])
            equity_history.append(mark)
            equity_index.append(ts)

        # ── 4. Nyitott pozíciók zárása (end-of-data) ─────────────────────────
        for sym, slot in list(slots.items()):
            last_price = float(enriched[sym].iloc[-1]["close"])
            last_ts = enriched[sym].index[-1]
            cash = _close_position(
                slot.trade, last_price, last_ts, cash, cfg, bt_cfg, "end_of_data"
            )
            all_trades.append(slot.trade)

        # ── 5. Eredmény ──────────────────────────────────────────────────────
        equity_curve = pd.Series(equity_history, index=equity_index, name="equity")

        per_symbol: Dict[str, List[PortfolioTrade]] = {sym: [] for sym in datasets}
        for t in all_trades:
            per_symbol.setdefault(t.symbol, []).append(t)

        total_return_pct = (cash / self.initial_balance - 1) * 100
        return PortfolioBacktestResult(
            equity_curve=equity_curve,
            trades=all_trades,
            per_symbol=per_symbol,
            final_balance=cash,
            total_return_pct=total_return_pct,
            initial_balance=self.initial_balance,
        )
