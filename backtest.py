"""
Backteszt motor - reszletesebb fill modell + walk-forward analizis.

Bovitesek:
  * Slippage es spread modellezes (basis pontban)
  * ATR-alapu stopok + trailing stop
  * Walk-forward: a teljes idosort egymast koveto ablakokra bontja
  * Realisabb (kovetkezo gyertya nyitoara fill)
  * SHORT pozicio tamogatas (bear cycle-ban SELL = short nyitas)
  * Fix kockazat per trade: meret = kockazat / stop_tavolsag
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from agent import Decision, TradingAgent
from config import TradingConfig
from mtf import resample_ohlcv


# ============================================================================
# Adatosztalyok
# ============================================================================

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    reason: str = ""
    direction: str = "long"   # "long" | "short"


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Trade]
    final_balance: float
    total_return_pct: float

    def summary(self) -> str:
        wins = [t for t in self.trades if t.pnl > 0]
        win_rate = (len(wins) / len(self.trades) * 100) if self.trades else 0.0
        max_dd = _max_drawdown(self.equity_curve)
        longs  = [t for t in self.trades if t.direction == "long"]
        shorts = [t for t in self.trades if t.direction == "short"]
        return (
            f"Trades: {len(self.trades)} (long={len(longs)}, short={len(shorts)}) | "
            f"Win rate: {win_rate:.1f}% | "
            f"Final: ${self.final_balance:,.2f} | "
            f"Total return: {self.total_return_pct:+.2f}% | "
            f"Max drawdown: {max_dd:.1f}%"
        )


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak
    return float(dd.max() * 100)


# ============================================================================
# Helper fuggvenyek
# ============================================================================

def _apply_slippage(price: float, side: str, bt_cfg) -> float:
    """A megrendelesi arat slippage + spread fele tolja el (a kereskedo karara)."""
    bps = (bt_cfg.slippage_bps + bt_cfg.spread_bps / 2) / 10_000
    return price * (1 + bps) if side == "BUY" else price * (1 - bps)


def _calc_size(fill: float, stop_price: Optional[float],
               cash: float, cfg, bt_cfg, score: float) -> float:
    """
    Poziciomeret kiszamitasa.

    Ha use_fixed_risk_sizing=True es van stop_price:
        size = (capital * risk_pct * |score|) / stop_tavolsag
        + max cap: position_size * capital

    Egyebkent (fallback):
        size = capital * position_size * |score| / fill_price
    """
    size_mult = abs(score) if cfg.risk.score_proportional_size else 1.0
    if cfg.risk.use_fixed_risk_sizing and stop_price is not None:
        stop_dist = abs(fill - stop_price)
        if stop_dist > 0:
            risk_amount = cash * cfg.risk.risk_per_trade_pct
            size = risk_amount * size_mult / stop_dist
            # Felso korlat: ne legyen nagyobb mint position_size * cash-bol kijovo meret
            max_size = cash * cfg.position_size / (fill * (1 + cfg.fee_rate))
            return min(size, max_size)
        return 0.0
    # Fallback: notional-alapu (regi logika)
    notional = cash * cfg.position_size * size_mult
    return notional / (fill * (1 + cfg.fee_rate))


def _initial_stops(entry: float, atr: float, stops) -> Tuple[float, float]:
    """Long stop-loss es take-profit."""
    if stops.use_atr_stops and atr > 0:
        return entry - stops.atr_stop_mult * atr, entry + stops.atr_tp_mult * atr
    return entry * (1 - stops.stop_loss_pct), entry * (1 + stops.take_profit_pct)


def _initial_stops_short(entry: float, atr: float, stops) -> Tuple[float, float]:
    """Short stop-loss (entry felett) es take-profit (entry alatt)."""
    if stops.use_atr_stops and atr > 0:
        return entry + stops.atr_stop_mult * atr, entry - stops.atr_tp_mult * atr
    return entry * (1 + stops.stop_loss_pct), entry * (1 - stops.take_profit_pct)


def _check_sl_tp(row, stop_price, tp_price) -> Optional[str]:
    """Long SL/TP: stop ha low <= stop, TP ha high >= tp."""
    if stop_price is not None and row["low"] <= stop_price:
        return "stop_loss"
    if tp_price is not None and row["high"] >= tp_price:
        return "take_profit"
    return None


def _check_sl_tp_short(row, stop_price, tp_price) -> Optional[str]:
    """Short SL/TP: stop ha high >= stop (felette), TP ha low <= tp (alatta)."""
    if stop_price is not None and row["high"] >= stop_price:
        return "stop_loss"
    if tp_price is not None and row["low"] <= tp_price:
        return "take_profit"
    return None


def _open_position(price: float, stop_price: Optional[float],
                   timestamp, cash: float, cfg, bt_cfg,
                   score: float = 1.0) -> Trade:
    """Long pozicio nyitasa. Meret: fix kockazat / stop_tavolsag."""
    fill = _apply_slippage(price, "BUY", bt_cfg)
    size = _calc_size(fill, stop_price, cash, cfg, bt_cfg, score)
    return Trade(entry_time=timestamp, entry_price=fill, size=size, direction="long")


def _open_short(price: float, stop_price: Optional[float],
                timestamp, cash: float, cfg, bt_cfg,
                score: float = 1.0) -> Trade:
    """
    Short pozicio nyitasa (futures/perp).
    A margin = size * entry_price * (1+fee) levodasra kerul a cash-bol.
    """
    fill = _apply_slippage(price, "SELL", bt_cfg)
    size = _calc_size(fill, stop_price, cash, cfg, bt_cfg, score)
    return Trade(entry_time=timestamp, entry_price=fill, size=size, direction="short")


def _close_position(trade: Trade, price: float, timestamp, cash: float,
                    cfg, bt_cfg, reason: str) -> float:
    """Long pozicio zarasa."""
    fill = _apply_slippage(price, "SELL", bt_cfg)
    proceeds = trade.size * fill * (1 - cfg.fee_rate)
    trade.exit_time = timestamp
    trade.exit_price = fill
    trade.reason = reason
    trade.pnl = proceeds - trade.size * trade.entry_price * (1 + cfg.fee_rate)
    return cash + proceeds


def _close_short(trade: Trade, price: float, timestamp, cash: float,
                 cfg, bt_cfg, reason: str) -> float:
    """
    Short pozicio zarasa (visszavasarlas).

    Nyitaskor levodt margin = size * entry * (1+fee).
    Zaraskor visszakapjuk a margint + PnL (ha az ar esett, PnL pozitiv).
    """
    fill = _apply_slippage(price, "BUY", bt_cfg)   # vasarlas: slippage felfelé
    pnl = (trade.size * (trade.entry_price - fill)
           - trade.size * trade.entry_price * cfg.fee_rate
           - trade.size * fill * cfg.fee_rate)
    trade.exit_time = timestamp
    trade.exit_price = fill
    trade.reason = reason
    trade.pnl = pnl
    margin = trade.size * trade.entry_price * (1 + cfg.fee_rate)
    return cash + margin + pnl


def _mark_position(position: Trade, price: float) -> float:
    """Pozicio mark-to-market erteke (long: size*price, short: margin+unrealized)."""
    if position.direction == "long":
        return position.size * price
    # Short: margin + unrealized PnL
    return position.size * (2 * position.entry_price - price)


# ============================================================================
# Backtester
# ============================================================================

class Backtester:
    """Realisabb gyertyaszintu backteszt motor, long + short tamogatassal."""

    def __init__(self, agent: TradingAgent, config: Optional[TradingConfig] = None):
        self.agent = agent
        self.config = config or agent.config

    def run(self, ohlcv: pd.DataFrame) -> BacktestResult:
        cfg = self.config
        bt_cfg = cfg.backtest
        stops = cfg.stops

        cash = cfg.initial_balance
        position: Optional[Trade] = None
        stop_price: Optional[float] = None
        tp_price: Optional[float] = None
        highest_price: Optional[float] = None   # long trailing
        lowest_price: Optional[float] = None    # short trailing
        bars_held: int = 0

        trades: List[Trade] = []
        equity_history: List[float] = []
        just_exited_sl_tp = False

        # MTF adatok betoltese
        if self.agent.mtf is not None:
            for tf in self.agent.config.mtf.timeframes:
                try:
                    self.agent.mtf.set_data(tf, resample_ohlcv(ohlcv, tf))
                except Exception:
                    pass

        enriched = self.agent.prepare(ohlcv)

        for i, (timestamp, row) in enumerate(enriched.iterrows()):
            price = float(row["close"])
            atr   = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
            just_exited_sl_tp = False

            if position is not None:
                bars_held += 1

            # ── 1. Ügynök döntése (cycle_params-hoz is kell) ─────────────────
            decision = self.agent.decide_at(i)
            cp = decision.cycle_params
            allow_long  = cp.allow_long  if cp else True
            allow_short = cp.allow_short if cp else False
            max_hold    = cp.max_holding_bars if cp else 9_999

            # ── 2. Trailing stop frissítés ───────────────────────────────────
            if position is not None and stops.use_trailing_stop:
                if position.direction == "long":
                    if highest_price is None or price > highest_price:
                        highest_price = price
                    if atr > 0 and highest_price is not None:
                        new_sl = highest_price - stops.trailing_atr_mult * atr
                        if stop_price is None or new_sl > stop_price:
                            stop_price = new_sl
                else:  # short
                    if lowest_price is None or price < lowest_price:
                        lowest_price = price
                    if atr > 0 and lowest_price is not None:
                        new_sl = lowest_price + stops.trailing_atr_mult * atr
                        if stop_price is None or new_sl < stop_price:
                            stop_price = new_sl

            # ── 3. SL / TP ellenőrzés ────────────────────────────────────────
            if position is not None:
                if position.direction == "long":
                    exit_r = _check_sl_tp(row, stop_price, tp_price)
                else:
                    exit_r = _check_sl_tp_short(row, stop_price, tp_price)

                if exit_r is not None:
                    fill = stop_price if exit_r == "stop_loss" else tp_price
                    if position.direction == "long":
                        cash = _close_position(position, fill or price, timestamp,
                                               cash, cfg, bt_cfg, exit_r)
                    else:
                        cash = _close_short(position, fill or price, timestamp,
                                            cash, cfg, bt_cfg, exit_r)
                    trades.append(position)
                    position = None
                    stop_price = tp_price = highest_price = lowest_price = None
                    bars_held = 0
                    just_exited_sl_tp = True

            # ── 4. Max tartási idő ───────────────────────────────────────────
            if position is not None and bars_held >= max_hold:
                if position.direction == "long":
                    cash = _close_position(position, price, timestamp,
                                           cash, cfg, bt_cfg, "max_holding")
                else:
                    cash = _close_short(position, price, timestamp,
                                        cash, cfg, bt_cfg, "max_holding")
                trades.append(position)
                position = None
                stop_price = tp_price = highest_price = lowest_price = None
                bars_held = 0

            # ── 5. Belépés / kilépés ─────────────────────────────────────────
            if decision.action == "BUY":
                if position is not None and position.direction == "short":
                    # Short zárása (BUY = visszavásárlás)
                    cash = _close_short(position, price, timestamp,
                                        cash, cfg, bt_cfg, "signal")
                    trades.append(position)
                    position = None
                    stop_price = tp_price = highest_price = lowest_price = None
                    bars_held = 0

                elif (position is None and allow_long and not just_exited_sl_tp):
                    atr_pct = atr / price if price > 0 and atr > 0 else 0.0
                    if atr_pct < cfg.risk.max_atr_pct:
                        fill = _apply_slippage(price, "BUY", bt_cfg)
                        sl, tp = _initial_stops(fill, atr, stops)
                        position = _open_position(price, sl, timestamp,
                                                  cash, cfg, bt_cfg, decision.score)
                        if position.size > 0:
                            cash -= position.size * position.entry_price * (1 + cfg.fee_rate)
                            stop_price, tp_price = sl, tp
                            highest_price = position.entry_price
                            bars_held = 0
                        else:
                            position = None  # érvénytelen méret

            elif decision.action == "SELL":
                if position is not None and position.direction == "long":
                    # Long zárása (SELL = eladás)
                    cash = _close_position(position, price, timestamp,
                                           cash, cfg, bt_cfg, "signal")
                    trades.append(position)
                    position = None
                    stop_price = tp_price = highest_price = lowest_price = None
                    bars_held = 0

                elif (position is None and allow_short and not just_exited_sl_tp):
                    atr_pct = atr / price if price > 0 and atr > 0 else 0.0
                    if atr_pct < cfg.risk.max_atr_pct:
                        fill = _apply_slippage(price, "SELL", bt_cfg)
                        sl, tp = _initial_stops_short(fill, atr, stops)
                        position = _open_short(price, sl, timestamp,
                                               cash, cfg, bt_cfg, decision.score)
                        if position.size > 0:
                            cash -= position.size * position.entry_price * (1 + cfg.fee_rate)
                            stop_price, tp_price = sl, tp
                            lowest_price = position.entry_price
                            bars_held = 0
                        else:
                            position = None

            # ── 6. Equity mark-to-market ─────────────────────────────────────
            mark = cash + (_mark_position(position, price) if position else 0.0)
            equity_history.append(mark)

        # ── Nyitott pozíció zárása az adat végén ─────────────────────────────
        if position is not None:
            last_price = float(enriched.iloc[-1]["close"])
            last_ts    = enriched.index[-1]
            if position.direction == "long":
                cash = _close_position(position, last_price, last_ts,
                                       cash, cfg, bt_cfg, "end_of_data")
            else:
                cash = _close_short(position, last_price, last_ts,
                                    cash, cfg, bt_cfg, "end_of_data")
            trades.append(position)
            if equity_history:
                equity_history[-1] = cash

        equity_curve = pd.Series(equity_history, index=enriched.index, name="equity")
        total_return_pct = (cash / cfg.initial_balance - 1) * 100
        return BacktestResult(equity_curve, trades, cash, total_return_pct)


# ============================================================================
# Walk-forward analizis
# ============================================================================

@dataclass
class WalkForwardResult:
    folds: List[BacktestResult] = field(default_factory=list)

    def summary(self) -> str:
        if not self.folds:
            return "Nincs fold."
        returns = [f.total_return_pct for f in self.folds]
        positive = sum(1 for r in returns if r > 0)
        avg = sum(returns) / len(returns)
        return (
            f"Folds: {len(self.folds)} | nyereseges: {positive}/{len(self.folds)} | "
            f"atlag hozam: {avg:+.2f}% | min: {min(returns):+.2f}% | "
            f"max: {max(returns):+.2f}%"
        )


def walk_forward(
    agent: TradingAgent,
    ohlcv: pd.DataFrame,
    fold_size: int = 500,
    step: Optional[int] = None,
) -> WalkForwardResult:
    if step is None:
        step = fold_size
    result = WalkForwardResult()
    bt = Backtester(agent)
    n = len(ohlcv)
    start = 0
    while start < n:
        end = min(start + fold_size, n)
        chunk = ohlcv.iloc[start:end]
        if len(chunk) < 100:
            break
        result.folds.append(bt.run(chunk))
        if end == n:
            break
        start += step
    return result
