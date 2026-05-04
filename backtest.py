"""
Backteszt motor - reszletesebb fill modell + walk-forward analizis.

Bovitesek:
  * Slippage es spread modellezes (basis pontban)
  * ATR-alapu stopok + trailing stop
  * Walk-forward: a teljes idosort egymast koveto ablakokra bontja
  * Realisabb (kovetkezo gyertya nyitoara fill)
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
        return (
            f"Trades: {len(self.trades)} | Win rate: {win_rate:.1f}% | "
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
# Backtester
# ============================================================================

class Backtester:
    """Realisabb gyertyaszintu backteszt motor."""

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
        highest_price: Optional[float] = None

        trades: List[Trade] = []
        equity_history: List[float] = []
        # SL/TP ütés után nem nyitunk újat ugyanazon a gyertyán (#7 fix)
        just_exited_sl_tp = False

        # MTF: resample-eljuk a CSV-bol a magasabb timeframe-eket es feltoltjuk
        # az analyzert. Nincs lookahead: az mtf.analyze(as_of=timestamp) a baron
        # belul csak az as_of-ig levo barokat latja (df[df.index <= as_of]).
        if self.agent.mtf is not None:
            for tf in self.agent.config.mtf.timeframes:
                try:
                    self.agent.mtf.set_data(tf, resample_ohlcv(ohlcv, tf))
                except Exception:
                    pass

        enriched = self.agent.prepare(ohlcv)

        for i, (timestamp, row) in enumerate(enriched.iterrows()):
            price = float(row["close"])
            atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
            just_exited_sl_tp = False

            # 1. Trailing frissites
            if position is not None and stops.use_trailing_stop:
                if highest_price is None or price > highest_price:
                    highest_price = price
                if atr > 0 and highest_price is not None:
                    new_stop = highest_price - stops.trailing_atr_mult * atr
                    if stop_price is None or new_stop > stop_price:
                        stop_price = new_stop

            # 2. SL / TP ellenorzes (a gyertya high/low alapjan)
            if position is not None:
                exit_reason = _check_sl_tp(row, stop_price, tp_price)
                if exit_reason is not None:
                    fill_price = stop_price if exit_reason == "stop_loss" else tp_price
                    cash = _close_position(
                        position, fill_price or price, timestamp, cash, cfg,
                        bt_cfg, exit_reason,
                    )
                    trades.append(position)
                    position = None
                    stop_price = tp_price = highest_price = None
                    just_exited_sl_tp = True   # nem nyitunk ugyanazon a gyertyán

            # 3. Az ugynok dontese
            decision = self.agent.decide_at(i)

            # 4. Belepes / kilepes
            # SL/TP utan ugyanazon a gyertyan nem lepunk be ujra: a gyertya
            # high/low-ja mar lefutott, az uj belepesre valojaban bar i+1-en
            # kerulne sor. Az azonnali ujrabelepes optimista backteszt torzitas.
            if decision.action == "BUY" and position is None and not just_exited_sl_tp:
                # Volatilitás szűrő: túl magas relatív ATR esetén kihagyjuk a belépést
                atr_pct = atr / price if price > 0 and atr > 0 else 0.0
                if atr_pct >= cfg.risk.max_atr_pct:
                    pass  # skip: túl volatilis
                else:
                    position = _open_position(
                        decision.price, timestamp, cash, cfg, bt_cfg, decision.score
                    )
                    cash -= position.size * position.entry_price * (1 + cfg.fee_rate)
                    stop_price, tp_price = _initial_stops(position.entry_price, atr, stops)
                    highest_price = position.entry_price

            elif decision.action == "SELL" and position is not None:
                cash = _close_position(position, price, timestamp, cash, cfg, bt_cfg, "signal")
                trades.append(position)
                position = None
                stop_price = tp_price = highest_price = None

            # 5. Equity vezetese
            mark = cash + (position.size * price if position else 0.0)
            equity_history.append(mark)

        if position is not None:
            last_price = float(enriched.iloc[-1]["close"])
            cash = _close_position(position, last_price, enriched.index[-1], cash, cfg, bt_cfg, "end_of_data")
            trades.append(position)
            # Equity curve utolsó pontjának frissítése az end_of_data zárás után,
            # hogy equity_curve.iloc[-1] == final_balance legyen (konzisztens)
            if equity_history:
                equity_history[-1] = cash

        equity_curve = pd.Series(equity_history, index=enriched.index, name="equity")
        total_return_pct = (cash / cfg.initial_balance - 1) * 100
        return BacktestResult(equity_curve, trades, cash, total_return_pct)


# ============================================================================
# Helper fuggvenyek (modulszintu, hogy walk-forward is tudja hasznalni)
# ============================================================================

def _initial_stops(entry: float, atr: float, stops) -> Tuple[float, float]:
    if stops.use_atr_stops and atr > 0:
        return entry - stops.atr_stop_mult * atr, entry + stops.atr_tp_mult * atr
    return entry * (1 - stops.stop_loss_pct), entry * (1 + stops.take_profit_pct)


def _check_sl_tp(row, stop_price, tp_price) -> Optional[str]:
    if stop_price is not None and row["low"] <= stop_price:
        return "stop_loss"
    if tp_price is not None and row["high"] >= tp_price:
        return "take_profit"
    return None


def _apply_slippage(price: float, side: str, bt_cfg) -> float:
    """A megrendelesi arat slippage + spread fele tolja el (a kereskedo karara)."""
    bps = (bt_cfg.slippage_bps + bt_cfg.spread_bps / 2) / 10_000
    return price * (1 + bps) if side == "BUY" else price * (1 - bps)


def _open_position(price: float, timestamp, cash: float, cfg, bt_cfg,
                   score: float = 1.0) -> Trade:
    fill_price = _apply_slippage(price, "BUY", bt_cfg)
    # Score-arányos méretezés (Kelly-szerű): ha be van kapcsolva, a pozíció
    # mérete arányos a jelzés erősségével (|score| ∈ [threshold, 1.0])
    size_mult = abs(score) if cfg.risk.score_proportional_size else 1.0
    notional = cash * cfg.position_size * size_mult
    size = notional / (fill_price * (1 + cfg.fee_rate))
    return Trade(entry_time=timestamp, entry_price=fill_price, size=size)


def _close_position(trade: Trade, price: float, timestamp, cash: float,
                    cfg, bt_cfg, reason: str) -> float:
    fill_price = _apply_slippage(price, "SELL", bt_cfg)
    proceeds = trade.size * fill_price * (1 - cfg.fee_rate)
    trade.exit_time = timestamp
    trade.exit_price = fill_price
    trade.reason = reason
    trade.pnl = proceeds - trade.size * trade.entry_price * (1 + cfg.fee_rate)
    return cash + proceeds


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
    """
    Egyszeru walk-forward: egymast koveto fold_size-os ablakokra futtatja a
    backtesztet. Ez NEM optimalizal parameterre, csak megmutatja, mennyire
    stabil a strategia kulonbozo idoszakokon.
    """
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
            break   # elértük az adat végét
        start += step
    return result
