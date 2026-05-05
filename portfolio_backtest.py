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
    _apply_slippage, _calc_size,
    _check_sl_tp, _check_sl_tp_short,
    _close_position, _close_short,
    _initial_stops, _initial_stops_short,
    _mark_position, _max_drawdown, Trade,
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
    lowest_price: float = 0.0   # short trailing
    bars: int = 0               # max_holding számlálóhoz


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

            # --- Döntések előre (cycle_params kell a max_holding-hoz is) -----
            decisions = {}
            for sym, agent in agents.items():
                if ts in ts_to_idx.get(sym, {}):
                    decisions[sym] = agent.decide_at(ts_to_idx[sym][ts])

            # --- Trailing stop + SL/TP + max_holding -------------------------
            for sym in list(slots.keys()):
                if ts not in ts_to_idx.get(sym, {}):
                    continue
                idx = ts_to_idx[sym][ts]
                row = enriched[sym].iloc[idx]
                slot = slots[sym]
                price = float(row["close"])
                atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
                slot.bars += 1

                # Cycle params a max_holding-hoz
                dec = decisions.get(sym)
                cp = dec.cycle_params if dec else None
                max_hold = cp.max_holding_bars if cp else 9_999

                # Trailing stop frissítés
                if stops.use_trailing_stop:
                    if slot.trade.direction == "long":
                        if price > slot.highest_price:
                            slot.highest_price = price
                        if atr > 0:
                            new_sl = slot.highest_price - stops.trailing_atr_mult * atr
                            if slot.stop_price is None or new_sl > slot.stop_price:
                                slot.stop_price = new_sl
                    else:  # short
                        if slot.lowest_price == 0.0 or price < slot.lowest_price:
                            slot.lowest_price = price
                        if atr > 0:
                            new_sl = slot.lowest_price + stops.trailing_atr_mult * atr
                            if slot.stop_price is None or new_sl < slot.stop_price:
                                slot.stop_price = new_sl

                # SL/TP
                if slot.trade.direction == "long":
                    exit_r = _check_sl_tp(row, slot.stop_price, slot.tp_price)
                else:
                    exit_r = _check_sl_tp_short(row, slot.stop_price, slot.tp_price)

                if exit_r is None and slot.bars >= max_hold:
                    exit_r = "max_holding"

                if exit_r is not None:
                    fill = (slot.stop_price if exit_r == "stop_loss"
                            else slot.tp_price if exit_r == "take_profit"
                            else price)
                    if slot.trade.direction == "long":
                        cash = _close_position(slot.trade, fill or price, ts,
                                               cash, cfg, bt_cfg, exit_r)
                    else:
                        cash = _close_short(slot.trade, fill or price, ts,
                                            cash, cfg, bt_cfg, exit_r)
                    all_trades.append(slot.trade)
                    del slots[sym]
                    just_exited.add(sym)

            # --- Belépések / jelzés alapú kilépések --------------------------
            for sym, dec in decisions.items():
                if sym in just_exited:
                    continue
                if ts not in ts_to_idx.get(sym, {}):
                    continue
                idx = ts_to_idx[sym][ts]
                row = enriched[sym].iloc[idx]
                price = float(row["close"])
                atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0

                cp = dec.cycle_params
                allow_long  = cp.allow_long  if cp else True
                allow_short = cp.allow_short if cp else False

                if dec.action == "BUY":
                    if sym in slots and slots[sym].trade.direction == "short":
                        # Short zárása
                        cash = _close_short(slots[sym].trade, price, ts,
                                            cash, cfg, bt_cfg, "signal")
                        all_trades.append(slots[sym].trade)
                        del slots[sym]

                    elif (sym not in slots and allow_long
                          and len(slots) < self.max_positions
                          and cash >= slot_size * 0.5):
                        atr_pct = atr / price if price > 0 and atr > 0 else 0.0
                        if atr_pct < cfg.risk.max_atr_pct:
                            fill = _apply_slippage(price, "BUY", bt_cfg)
                            sl, tp = _initial_stops(fill, atr, stops)
                            alloc = min(slot_size, cash)
                            # Risk-alapú méret a slot-on belül
                            size = _calc_size(fill, sl, alloc, cfg, bt_cfg, dec.score)
                            if size > 0:
                                trade = PortfolioTrade(entry_time=ts, entry_price=fill,
                                                       size=size, symbol=sym,
                                                       direction="long")
                                cash -= size * fill * (1 + cfg.fee_rate)
                                slots[sym] = _Slot(trade, sl, tp,
                                                   highest_price=fill,
                                                   lowest_price=fill)

                elif dec.action == "SELL":
                    if sym in slots and slots[sym].trade.direction == "long":
                        # Long zárása
                        cash = _close_position(slots[sym].trade, price, ts,
                                               cash, cfg, bt_cfg, "signal")
                        all_trades.append(slots[sym].trade)
                        del slots[sym]

                    elif (sym not in slots and allow_short
                          and len(slots) < self.max_positions
                          and cash >= slot_size * 0.5):
                        atr_pct = atr / price if price > 0 and atr > 0 else 0.0
                        if atr_pct < cfg.risk.max_atr_pct:
                            fill = _apply_slippage(price, "SELL", bt_cfg)
                            sl, tp = _initial_stops_short(fill, atr, stops)
                            alloc = min(slot_size, cash)
                            size = _calc_size(fill, sl, alloc, cfg, bt_cfg, dec.score)
                            if size > 0:
                                trade = PortfolioTrade(entry_time=ts, entry_price=fill,
                                                       size=size, symbol=sym,
                                                       direction="short")
                                cash -= size * fill * (1 + cfg.fee_rate)
                                slots[sym] = _Slot(trade, sl, tp,
                                                   highest_price=fill,
                                                   lowest_price=fill)

            # --- Equity mark-to-market --------------------------------------
            mark = cash
            for sym, slot in slots.items():
                if ts in ts_to_idx.get(sym, {}):
                    idx = ts_to_idx[sym][ts]
                    cur = float(enriched[sym].iloc[idx]["close"])
                    mark += _mark_position(slot.trade, cur)
            equity_history.append(mark)
            equity_index.append(ts)

        # ── 4. Nyitott pozíciók zárása (end-of-data) ─────────────────────────
        for sym, slot in list(slots.items()):
            last_price = float(enriched[sym].iloc[-1]["close"])
            last_ts = enriched[sym].index[-1]
            if slot.trade.direction == "long":
                cash = _close_position(slot.trade, last_price, last_ts,
                                       cash, cfg, bt_cfg, "end_of_data")
            else:
                cash = _close_short(slot.trade, last_price, last_ts,
                                    cash, cfg, bt_cfg, "end_of_data")
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
