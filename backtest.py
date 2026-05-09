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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent import Decision, TradingAgent
from config import ScaleInConfig, TradingConfig
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


# ============================================================================
# Vektorizált SL/TP keresők (numpy, C-sebességű)
# ============================================================================

def _find_exit_bar_long(
    lows: np.ndarray,
    highs: np.ndarray,
    sl: float,
    tp: float,
    start: int,
    end: int,
) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    """
    Vektorizált előre-keresés long SL/TP találatra [start, end) tartományban.

    Returns (bar_index, reason, fill_price) vagy (None, None, None) ha nincs találat.
    A „same-bar tie" (SL és TP egyszerre teljesül) konzervatívan SL győz.
    """
    if start >= end:
        return None, None, None
    seg_low  = lows[start:end]
    seg_high = highs[start:end]

    sl_hits = np.where(seg_low  <= sl)[0]
    tp_hits = np.where(seg_high >= tp)[0]

    sl_idx = int(sl_hits[0]) if len(sl_hits) else (end - start)
    tp_idx = int(tp_hits[0]) if len(tp_hits) else (end - start)

    if sl_idx < tp_idx:
        return start + sl_idx, "stop_loss",   sl
    if tp_idx < sl_idx:
        return start + tp_idx, "take_profit", tp
    if sl_idx == tp_idx < (end - start):
        return start + sl_idx, "stop_loss",   sl   # konzervatív
    return None, None, None


def _find_exit_bar_short(
    lows: np.ndarray,
    highs: np.ndarray,
    sl: float,
    tp: float,
    start: int,
    end: int,
) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    """
    Vektorizált előre-keresés short SL/TP találatra [start, end) tartományban.

    Short SL: high >= sl (ár felülről üti a stopot)
    Short TP: low  <= tp (ár leesett a célra)
    """
    if start >= end:
        return None, None, None
    seg_low  = lows[start:end]
    seg_high = highs[start:end]

    sl_hits = np.where(seg_high >= sl)[0]
    tp_hits = np.where(seg_low  <= tp)[0]

    sl_idx = int(sl_hits[0]) if len(sl_hits) else (end - start)
    tp_idx = int(tp_hits[0]) if len(tp_hits) else (end - start)

    if sl_idx < tp_idx:
        return start + sl_idx, "stop_loss",   sl
    if tp_idx < sl_idx:
        return start + tp_idx, "take_profit", tp
    if sl_idx == tp_idx < (end - start):
        return start + sl_idx, "stop_loss",   sl
    return None, None, None


# ============================================================================
# VectorizedBacktester
# ============================================================================

class VectorizedBacktester:
    """
    Gyors backteszter 1s / nagy adathalmazhoz (~100× Backtester-nél).

    decide_at() per-bar hívás helyett vektorizált pipeline:
      1. compute_all()                   — indikátorok (pandas rolling, vektorizált)
      2. compute_signal_matrix()         — szignálmátrix egy numpy pass-ban
      3. compute_scores_with_regime()    — np.dot batch score, ADX-alapú súlyváltással
      4. Fő loop: O(N_trades), nem O(N_bars)
      5. SL/TP keresés: numpy argwhere → O(K) C-loop a trade tartamán belül

    Korlátok:
      * Trailing stop NEM támogatott. Ha use_trailing_stop=True, automatikusan
        a hagyományos Backtester fut le.
      * Cycle params a prepare() egyszeri eredménye (nem változik bar-onként).
      * Belépési feltételek: raw score küszöb + ATR/ár szűrő. Timing-blokkok,
        altseason-szűrő és ML-prob-szűrő nem futnak (backtest sebesség miatt).
      * MTF: ha be van töltve, a resample_ohlcv() előszámítja az adatot;
        a composite_score azonban nem kerül a score-ba (nincs per-bar MTF call).
    """

    def __init__(self, agent: TradingAgent, config: Optional[TradingConfig] = None):
        self.agent  = agent
        self.config = config or agent.config

    def run(self, ohlcv: pd.DataFrame) -> BacktestResult:
        cfg = self.config

        # Trailing stop: visszaesés a standard backtesterre
        if cfg.stops.use_trailing_stop:
            return Backtester(self.agent, cfg).run(ohlcv)

        bt_cfg = cfg.backtest
        stops  = cfg.stops

        # ── 1. Indikátorok + előkészítés ─────────────────────────────────
        if self.agent.mtf is not None:
            for tf in self.agent.config.mtf.timeframes:
                try:
                    self.agent.mtf.set_data(tf, resample_ohlcv(ohlcv, tf))
                except Exception:
                    pass

        enriched = self.agent.prepare(ohlcv)
        n = len(enriched)
        if n == 0:
            return BacktestResult(pd.Series(dtype=float, name="equity"),
                                  [], cfg.initial_balance, 0.0)

        # ── 2. Vektorizált szignálmátrix + score ─────────────────────────
        from signals import compute_signal_matrix, compute_scores_with_regime
        from config import TREND_WEIGHTS, RANGE_WEIGHTS, DEFAULT_WEIGHTS

        fg_value = self.agent._fg_for_ts(enriched.index[0])
        sig_mat  = compute_signal_matrix(enriched, cfg.indicators, fg_value)
        scores   = compute_scores_with_regime(
            sig_mat, enriched, cfg.regime,
            DEFAULT_WEIGHTS, TREND_WEIGHTS, RANGE_WEIGHTS,
        )  # float64 array, len=n

        # ── 3. Numpy price arrays ─────────────────────────────────────────
        lows   = enriched["low"].values
        highs  = enriched["high"].values
        closes = enriched["close"].values
        atrs   = enriched["atr"].values if "atr" in enriched.columns else np.zeros(n)

        # ── 4. Ciklus-paraméterek (egyszer, az egész futásra) ─────────────
        cycle_state = self.agent._cycle_state
        if cycle_state is not None:
            from adaptive_strategy import get_params
            cp = get_params(cycle_state.cycle)
        else:
            cp = None
        allow_long  = cp.allow_long        if cp else True
        allow_short = cp.allow_short       if cp else False
        max_hold    = cp.max_holding_bars  if cp else 9_999

        # ── 5. Fő loop: skip-ahead pattern ───────────────────────────────
        cash = float(cfg.initial_balance)
        equity_arr = np.full(n, np.nan, dtype=np.float64)
        trades: List[Trade] = []
        i = 0

        while i < n:
            # ── Következő belépési jelölt megkeresése (numpy) ─────────────
            rem_scores = scores[i:]
            cand_long  = np.where(rem_scores >= cfg.buy_threshold)[0]  if allow_long  else np.array([], dtype=int)
            cand_short = np.where(rem_scores <= cfg.sell_threshold)[0] if allow_short else np.array([], dtype=int)

            first_long  = (i + int(cand_long[0]))  if len(cand_long)  else n
            first_short = (i + int(cand_short[0])) if len(cand_short) else n

            if first_long >= n and first_short >= n:
                break   # nincs több belépés

            entry_i = min(first_long, first_short)
            direction = "long" if first_long <= first_short else "short"

            # ── Kitöltjük a pozíció nélküli sávot ──────────────────────
            equity_arr[i:entry_i] = cash

            # ── Volatilitás-szűrő ────────────────────────────────────────
            atr_v = float(atrs[entry_i]) if not np.isnan(atrs[entry_i]) else 0.0
            price = float(closes[entry_i])
            atr_pct = atr_v / price if price > 0 and atr_v > 0 else 0.0
            if atr_pct >= cfg.risk.max_atr_pct or atr_v <= 0:
                equity_arr[entry_i] = cash
                i = entry_i + 1
                continue

            # ── Pozíció nyitása ──────────────────────────────────────────
            score_v = float(scores[entry_i])
            if direction == "long":
                fill = _apply_slippage(price, "BUY", bt_cfg)
                sl, tp = _initial_stops(fill, atr_v, stops)
            else:
                fill = _apply_slippage(price, "SELL", bt_cfg)
                sl, tp = _initial_stops_short(fill, atr_v, stops)

            size = _calc_size(fill, sl, cash, cfg, bt_cfg, score_v)
            if size <= 0:
                equity_arr[entry_i] = cash
                i = entry_i + 1
                continue

            cash -= size * fill * (1.0 + cfg.fee_rate)   # = cash after buy
            trade = Trade(
                entry_time=enriched.index[entry_i],
                entry_price=fill, size=size, direction=direction,
            )

            # ── Kilépés meghatározása (vektorizált) ───────────────────────
            max_bar = min(entry_i + 1 + max_hold, n)

            if direction == "long":
                exit_i, reason, exit_fill = _find_exit_bar_long(
                    lows, highs, sl, tp, entry_i + 1, max_bar)
                sig_rev = np.where(scores[entry_i + 1:max_bar] <= cfg.sell_threshold)[0]
            else:
                exit_i, reason, exit_fill = _find_exit_bar_short(
                    lows, highs, sl, tp, entry_i + 1, max_bar)
                sig_rev = np.where(scores[entry_i + 1:max_bar] >= cfg.buy_threshold)[0]

            sig_exit_i = (entry_i + 1 + int(sig_rev[0])) if len(sig_rev) else n

            # SL/TP és signal exit közül a korábbi nyer
            if exit_i is None:
                if sig_exit_i < n:
                    exit_i = sig_exit_i; reason = "signal"; exit_fill = float(closes[exit_i])
                else:
                    exit_i = max_bar - 1; reason = "max_holding"; exit_fill = float(closes[exit_i])
            elif sig_exit_i < exit_i:
                exit_i = sig_exit_i; reason = "signal"; exit_fill = float(closes[exit_i])

            # ── Equity a trade alatt (vektorizált) ────────────────────────
            # cash itt már post-buy; pozíció mark-to-market = cash + size * close
            equity_arr[entry_i:exit_i + 1] = cash + size * closes[entry_i:exit_i + 1]

            # ── Pozíció zárása ───────────────────────────────────────────
            exit_ts = enriched.index[exit_i]
            if direction == "long":
                cash = _close_position(trade, exit_fill, exit_ts, cash, cfg, bt_cfg, reason)
            else:
                cash = _close_short(trade, exit_fill, exit_ts, cash, cfg, bt_cfg, reason)
            equity_arr[exit_i] = cash   # post-close override az exit bárra
            trades.append(trade)
            i = exit_i + 1

        # Kitöltjük a trade utáni maradék sávot
        equity_arr[i:] = cash

        # Nyitott pozíció zárása az adat végén (ha a last trade az utolsó bár)
        # Ez a skip-ahead loopban nem fordulhat elő (max_hold védi), de biztonsági net:
        if np.isnan(equity_arr[-1]):
            equity_arr[np.isnan(equity_arr)] = cash

        equity_curve = pd.Series(equity_arr, index=enriched.index, name="equity")
        total_return_pct = (cash / cfg.initial_balance - 1) * 100
        return BacktestResult(equity_curve, trades, cash, total_return_pct)


# ============================================================================
# Portfolio backteszter — párhuzamos multi-symbol figyelés + scale-in
# ============================================================================

@dataclass
class _OpenPos:
    """Belső állapot egy nyitott portfolió-pozícióhoz."""
    sym: str
    direction: str       # "long" | "short"
    entry_i: int         # bársorszám a szimbólum saját indexén
    sl: float
    tp: float
    size: float          # jelenlegi összesített mennyiség (tranche-ok után)
    trade: Trade
    tranche_idx: int     # hány tranche-t vettünk már
    last_buy_price: float
    next_dca_bar: Optional[int] = None  # cache: köv. scale-in trigger bar (O(1) lookup)


@dataclass
class PortfolioBacktestResult:
    """
    Portfolió backteszt eredménye.

    Tartalmaz:
      combined_equity  — közös cash + mark-to-market equity görbe
      per_symbol       — szimbólumok szerinti trade-lista + hozam
      all_trades       — összes kötés időrendi sorrendben
      final_balance    — végső egyenleg
      total_return_pct — százalékos hozam
    """
    combined_equity: pd.Series
    per_symbol: Dict[str, List[Trade]]
    all_trades: List[Trade]
    final_balance: float
    total_return_pct: float

    def summary(self) -> str:
        wins = [t for t in self.all_trades if t.pnl > 0]
        win_rate = len(wins) / len(self.all_trades) * 100 if self.all_trades else 0.0
        max_dd = _max_drawdown(self.combined_equity)
        by_sym = {s: len(ts) for s, ts in self.per_symbol.items() if ts}
        return (
            f"Trades: {len(self.all_trades)} {by_sym} | "
            f"Win rate: {win_rate:.1f}% | "
            f"Final: ${self.final_balance:,.2f} | "
            f"Return: {self.total_return_pct:+.2f}% | "
            f"Max DD: {max_dd:.1f}%"
        )


class PortfolioBacktester:
    """
    Párhuzamos multi-symbol backteszter.

    Algoritmus:
      1. Minden szimbólumra előszámítja a vektorizált score-tömböt egyszer
         (compute_signal_matrix + compute_scores_with_regime).
      2. Event-driven skip-ahead loop:
         - Ha nincs nyitott pozíció: numpy.where -> ugrás a következő entry jelre
         - Ha van nyitott pozíció: _find_exit_bar_long/short -> leghamarabb kiváltódó
           SL/TP eseményhez ugrik; közben vectorized mark-to-market equity kitöltés
      3. ScaleIn:
         - "dca" mode: ha az ár lesüllyed trigger_pct-tel az utolsó vásárláshoz képest,
           új tranche vásárlása (átlagolás lefelé)
         - "pyramid" mode: ha az ár felemelkedik trigger_pct-tel, hozzáad a pozícióhoz

    max_concurrent: egyszerre hány szimbólum tarthat nyitott pozíciót.
    """

    def __init__(
        self,
        agents: Dict[str, TradingAgent],
        config: TradingConfig,
        max_concurrent: int = 3,
        scale_in: Optional[ScaleInConfig] = None,
    ):
        self.agents = agents
        self.config = config
        self.max_concurrent = max_concurrent
        self.scale_in = scale_in or config.scale_in

    # ------------------------------------------------------------------
    # Segéd: score-tömb előszámítása egy szimbólumra
    # ------------------------------------------------------------------

    def _precompute(self, sym: str, ohlcv: pd.DataFrame) -> dict:
        from signals import compute_signal_matrix, compute_scores_with_regime
        from config import DEFAULT_WEIGHTS, TREND_WEIGHTS, RANGE_WEIGHTS

        agent = self.agents[sym]
        cfg = self.config
        enriched = agent.prepare(ohlcv)
        n = len(enriched)
        fg = agent._fg_for_ts(enriched.index[0])
        sig_mat = compute_signal_matrix(enriched, cfg.indicators, fg)
        scores = compute_scores_with_regime(
            sig_mat, enriched, cfg.regime,
            DEFAULT_WEIGHTS, TREND_WEIGHTS, RANGE_WEIGHTS,
        )
        return {
            "enriched": enriched,
            "scores":   scores,
            "lows":     enriched["low"].values.astype(np.float64),
            "highs":    enriched["high"].values.astype(np.float64),
            "closes":   enriched["close"].values.astype(np.float64),
            "atrs":     (enriched["atr"].values.astype(np.float64)
                         if "atr" in enriched.columns else np.zeros(n)),
            "n":        n,
        }

    # ------------------------------------------------------------------
    # DCA / pyramid trigger bar keresése
    # ------------------------------------------------------------------

    @staticmethod
    def _find_scale_trigger(closes: np.ndarray, last_price: float,
                            trigger_pct: float, mode: str,
                            start: int, end: int) -> Optional[int]:
        """Visszaadja az első bar indexét ahol a scale-in feltétel teljesül."""
        seg = closes[start:end]
        if mode == "dca":
            thresh = last_price * (1.0 - trigger_pct)
            hits = np.where(seg <= thresh)[0]
        else:  # pyramid
            thresh = last_price * (1.0 + trigger_pct)
            hits = np.where(seg >= thresh)[0]
        return (start + int(hits[0])) if len(hits) else None

    # ------------------------------------------------------------------
    # Fő futtatás
    # ------------------------------------------------------------------

    def run(self, ohlcv_dict: Dict[str, pd.DataFrame]) -> PortfolioBacktestResult:
        cfg = self.config
        bt_cfg = cfg.backtest
        stops = cfg.stops
        sc = self.scale_in
        symbols = list(ohlcv_dict.keys())

        # 1. Előszámítás minden szimbólumra
        sym_data: Dict[str, dict] = {}
        for sym in symbols:
            sym_data[sym] = self._precompute(sym, ohlcv_dict[sym])

        # Közös hossz: rövidebb szimbólumhoz igazítunk
        n = min(d["n"] for d in sym_data.values())
        # Közös idő-index (az első szimbólum alapján)
        common_index = sym_data[symbols[0]]["enriched"].index[:n]

        # 2. Inicializálás
        cash = float(cfg.initial_balance)
        open_pos: Dict[str, _OpenPos] = {}   # sym -> _OpenPos
        all_trades: List[Trade] = []
        per_symbol: Dict[str, List[Trade]] = {s: [] for s in symbols}
        equity_arr = np.full(n, np.nan, dtype=np.float64)
        i = 0

        def _mtm_at(bar: int) -> float:
            return sum(
                p.size * float(sym_data[p.sym]["closes"][bar])
                for p in open_pos.values()
            )

        def _open_new(sym: str, bar: int, score: float) -> None:
            nonlocal cash
            d = sym_data[sym]
            price = float(d["closes"][bar])
            atr_v = float(d["atrs"][bar])
            if atr_v <= 0 or price <= 0:
                return
            fill = _apply_slippage(price, "BUY", bt_cfg)
            sl, tp = _initial_stops(fill, atr_v, stops)
            if sc.enabled:
                alloc = cash * sc.first_tranche_pct
                size = alloc / (fill * (1.0 + cfg.fee_rate))
            else:
                size = _calc_size(fill, sl, cash, cfg, bt_cfg, abs(score))
            if size <= 0:
                return
            cost = size * fill * (1.0 + cfg.fee_rate)
            if cost > cash:
                return
            cash -= cost
            trade = Trade(
                entry_time=d["enriched"].index[bar],
                entry_price=fill,
                size=size,
                direction="long",
            )
            open_pos[sym] = _OpenPos(
                sym=sym, direction="long",
                entry_i=bar, sl=sl, tp=tp,
                size=size, trade=trade,
                tranche_idx=1, last_buy_price=fill,
            )

        def _close(sym: str, pos: _OpenPos, fill: float, bar: int, reason: str) -> None:
            nonlocal cash
            d = sym_data[sym]
            ts = d["enriched"].index[bar]
            cash = _close_position(pos.trade, fill, ts, cash, cfg, bt_cfg, reason)
            all_trades.append(pos.trade)
            per_symbol[sym].append(pos.trade)
            del open_pos[sym]

        while i < n:
            # ── A) Kilépések az aktuális bárra ──────────────────────────
            to_close = []
            for sym, pos in open_pos.items():
                d = sym_data[sym]
                low_i  = float(d["lows"][i])
                high_i = float(d["highs"][i])
                if low_i <= pos.sl:
                    to_close.append((sym, pos, pos.sl, "stop_loss"))
                elif high_i >= pos.tp:
                    to_close.append((sym, pos, pos.tp, "take_profit"))
            for sym, pos, fill, reason in to_close:
                _close(sym, pos, fill, i, reason)

            # ── B) Scale-in ──────────────────────────────────────────────
            if sc.enabled:
                for sym, pos in list(open_pos.items()):
                    if pos.tranche_idx >= sc.n_tranches:
                        continue
                    curr = float(sym_data[sym]["closes"][i])
                    if sc.mode == "dca":
                        triggered = curr <= pos.last_buy_price * (1.0 - sc.trigger_pct)
                    else:
                        triggered = curr >= pos.last_buy_price * (1.0 + sc.trigger_pct)
                    if triggered:
                        alloc = cash * sc.add_tranche_pct
                        add_fill = _apply_slippage(curr, "BUY", bt_cfg)
                        add_size = alloc / (add_fill * (1.0 + cfg.fee_rate))
                        cost = add_size * add_fill * (1.0 + cfg.fee_rate)
                        if add_size > 0 and cost <= cash:
                            cash -= cost
                            pos.size += add_size
                            pos.tranche_idx += 1
                            pos.last_buy_price = add_fill

            # ── C) Új belépések ──────────────────────────────────────────
            while len(open_pos) < self.max_concurrent:
                avail = [s for s in symbols if s not in open_pos]
                if not avail:
                    break
                best_sym: Optional[str] = None
                best_score = cfg.buy_threshold - 1e-9
                for sym in avail:
                    s = float(sym_data[sym]["scores"][i])
                    if s > best_score:
                        best_score = s
                        best_sym = sym
                if best_sym is None:
                    break
                _open_new(best_sym, i, best_score)
                if best_sym not in open_pos:
                    break  # nem sikerült (pl. nincs elég cash)

            # ── D) Equity ────────────────────────────────────────────────
            equity_arr[i] = cash + _mtm_at(i)

            # ── E) Skip-ahead ────────────────────────────────────────────
            if not open_pos:
                min_next = n
                for sym in symbols:
                    scr = sym_data[sym]["scores"]
                    cands = np.where(scr[i + 1:] >= cfg.buy_threshold)[0]
                    if len(cands):
                        min_next = min(min_next, i + 1 + int(cands[0]))
                if min_next >= n:
                    equity_arr[i:] = cash
                    break
                equity_arr[i + 1:min_next] = cash
                i = min_next
            else:
                # SL/TP / scale-trigger / új entry közül legkorábbi
                events: List[Tuple[int, str, str, float]] = []

                for sym, pos in open_pos.items():
                    d = sym_data[sym]
                    ex_i, ex_r, ex_f = _find_exit_bar_long(
                        d["lows"], d["highs"], pos.sl, pos.tp, i + 1, n)
                    if ex_i is None:
                        ex_i = n - 1; ex_r = "max_holding"
                        ex_f = float(d["closes"][n - 1])
                    events.append((ex_i, "exit", sym, ex_f))

                    if sc.enabled and pos.tranche_idx < sc.n_tranches:
                        trig_i = self._find_scale_trigger(
                            d["closes"], pos.last_buy_price,
                            sc.trigger_pct, sc.mode, i + 1, n)
                        if trig_i is not None:
                            events.append((trig_i, "scale", sym,
                                           float(d["closes"][trig_i])))

                if len(open_pos) < self.max_concurrent:
                    for sym in symbols:
                        if sym not in open_pos:
                            scr = sym_data[sym]["scores"]
                            cands = np.where(scr[i + 1:] >= cfg.buy_threshold)[0]
                            if len(cands):
                                e_i = i + 1 + int(cands[0])
                                events.append((e_i, "entry", sym,
                                               float(scr[e_i])))

                if not events:
                    equity_arr[i:] = cash + _mtm_at(i)
                    break

                next_i = min(ev[0] for ev in events)

                # Vektorizált equity kitöltés i+1 .. next_i-1
                if next_i > i + 1:
                    eq_seg = np.full(next_i - i - 1, cash)
                    for sym, pos in open_pos.items():
                        eq_seg += pos.size * sym_data[sym]["closes"][i + 1:next_i]
                    equity_arr[i + 1:next_i] = eq_seg

                i = next_i

        # Maradék NaN kitöltés
        if np.any(np.isnan(equity_arr)):
            last_val = cash + _mtm_at(min(i, n - 1)) if i < n else cash
            equity_arr[np.isnan(equity_arr)] = last_val

        # Nyitott pozíciók lezárása az adatsor végén
        for sym, pos in list(open_pos.items()):
            fill = float(sym_data[sym]["closes"][n - 1])
            _close(sym, pos, fill, n - 1, "end_of_data")

        equity_arr[-1] = cash

        equity_curve = pd.Series(equity_arr, index=common_index, name="equity")
        total_return_pct = (cash / cfg.initial_balance - 1) * 100
        return PortfolioBacktestResult(
            combined_equity=equity_curve,
            per_symbol=per_symbol,
            all_trades=all_trades,
            final_balance=cash,
            total_return_pct=total_return_pct,
        )
