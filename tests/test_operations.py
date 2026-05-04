"""
Működési mód tesztek — pytest

Fedett modulok:
  signals, regime, backtest, risk_manager, orderbook_features,
  cost_model, exit_manager, adaptive_strategy, portfolio_risk,
  override_engine, mtf, triple_barrier, drift_detector, agent

Futtatás:
  pytest tests/test_operations.py -v
"""

from __future__ import annotations

import sys
import os
import math
import logging

import numpy as np
import pandas as pd
import pytest

# Projekt gyökér hozzáadása az importálási úthoz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.disable(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Modul-szintű importok
# ---------------------------------------------------------------------------

from signals import (
    signal_sma_cross, signal_ema_cross, signal_macd, signal_adx,
    signal_rsi, signal_stochastic, signal_cci, signal_bollinger,
    signal_atr, signal_obv, signal_vwap, signal_mfi,
    signal_fear_greed, signal_golden_death, signal_long_trend,
)
from regime import detect_regime
from backtest import (
    Backtester, _max_drawdown, _initial_stops, _check_sl_tp,
    _apply_slippage, _open_position,
)
from risk_manager import RiskManager
from orderbook_features import OrderBookSnapshot
from cost_model import CostModel, CostConfig
from exit_manager import ExitManager, ExitConfig
from adaptive_strategy import get_params, apply_to_config, CYCLE_PARAMS, describe
from portfolio_risk import PortfolioRiskManager, PortfolioRiskConfig
from override_engine import OverrideEngine, BlockReason, _OVERRIDE_THRESHOLDS
from mtf import MTFAnalyzer
from triple_barrier import make_labels
from drift_detector import DriftDetector, DriftConfig
from agent import TradingAgent
from config import (
    TradingConfig, RiskConfig, StopConfig, RegimeConfig, IndicatorParams,
    BacktestConfig, TREND_WEIGHTS, RANGE_WEIGHTS, DEFAULT_WEIGHTS,
)
from market_cycle import MarketCycle
from indicators import compute_all as _compute_all_raw

def compute_all(df):
    """Wrapper: alapértelmezett IndicatorParams-szal hívja a compute_all-t."""
    return _compute_all_raw(df, IndicatorParams())


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 300, drift: float = 0.001, sigma: float = 0.015,
           start: float = 30_000.0, seed: int = 0) -> pd.DataFrame:
    """Szintetikus 1h OHLCV DataFrame, n gyertya."""
    np.random.seed(seed)
    r = np.random.normal(drift, sigma, n)
    close = start * np.exp(np.cumsum(r))
    high   = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low    = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_  = np.r_[close[0], close[:-1]]
    volume = np.random.uniform(100, 1_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _row(**kwargs) -> pd.Series:
    """Egyetlen szintetikus sor; nem adott mezők NaN."""
    defaults = {
        "open": 30000.0, "high": 30300.0, "low": 29700.0,
        "close": 30100.0, "volume": 500.0,
        "sma_fast": np.nan, "sma_slow": np.nan, "sma_long": np.nan,
        "ema_fast": np.nan, "ema_slow": np.nan,
        "macd_hist": np.nan,
        "adx": np.nan, "plus_di": np.nan, "minus_di": np.nan,
        "rsi": np.nan, "stoch_k": np.nan, "stoch_d": np.nan,
        "cci": np.nan,
        "bb_upper": np.nan, "bb_lower": np.nan,
        "atr": 300.0, "obv": np.nan, "vwap": np.nan, "mfi": np.nan,
        "recent_golden": 0, "recent_death": 0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def _make_agent(mtf_enabled: bool = False, **cfg_attrs) -> tuple[TradingAgent, TradingConfig]:
    cfg = TradingConfig()
    cfg.mtf.enabled = mtf_enabled
    for k, v in cfg_attrs.items():
        setattr(cfg, k, v)
    return TradingAgent(cfg), cfg


# ===========================================================================
# 1. signals.py
# ===========================================================================

class TestSignals:

    def test_sma_cross_bull(self):
        assert signal_sma_cross(_row(sma_fast=101, sma_slow=100)) == 1

    def test_sma_cross_bear(self):
        assert signal_sma_cross(_row(sma_fast=99, sma_slow=100)) == -1

    def test_sma_cross_nan_returns_0(self):
        assert signal_sma_cross(_row()) == 0

    def test_ema_cross_bull(self):
        assert signal_ema_cross(_row(ema_fast=200, ema_slow=199)) == 1

    def test_ema_cross_bear(self):
        assert signal_ema_cross(_row(ema_fast=198, ema_slow=199)) == -1

    def test_macd_positive(self):
        assert signal_macd(_row(macd_hist=0.5)) == 1

    def test_macd_negative(self):
        assert signal_macd(_row(macd_hist=-0.5)) == -1

    def test_macd_nan(self):
        assert signal_macd(_row()) == 0

    def test_adx_weak_trend_neutral(self):
        assert signal_adx(_row(adx=15, plus_di=25, minus_di=10)) == 0

    def test_adx_strong_bull(self):
        assert signal_adx(_row(adx=30, plus_di=25, minus_di=10)) == 1

    def test_adx_strong_bear(self):
        assert signal_adx(_row(adx=30, plus_di=10, minus_di=25)) == -1

    def test_rsi_oversold(self):
        assert signal_rsi(_row(rsi=25), IndicatorParams()) == 1

    def test_rsi_overbought(self):
        assert signal_rsi(_row(rsi=80), IndicatorParams()) == -1

    def test_rsi_neutral(self):
        assert signal_rsi(_row(rsi=50), IndicatorParams()) == 0

    def test_stoch_oversold_cross_up(self):
        # k < oversold(20) és k > d → buy
        assert signal_stochastic(_row(stoch_k=15, stoch_d=12), IndicatorParams()) == 1

    def test_stoch_overbought_cross_down(self):
        # k > overbought(80) és k < d → sell
        assert signal_stochastic(_row(stoch_k=85, stoch_d=90), IndicatorParams()) == -1

    def test_cci_oversold(self):
        assert signal_cci(_row(cci=-150)) == 1

    def test_cci_overbought(self):
        assert signal_cci(_row(cci=150)) == -1

    def test_bollinger_below_lower(self):
        assert signal_bollinger(_row(close=29000, bb_lower=29500, bb_upper=31000)) == 1

    def test_bollinger_above_upper(self):
        assert signal_bollinger(_row(close=32000, bb_lower=29500, bb_upper=31000)) == -1

    def test_atr_always_zero(self):
        assert signal_atr(_row(atr=500)) == 0
        assert signal_atr(_row(atr=0)) == 0

    def test_obv_rising(self):
        assert signal_obv(_row(obv=1000), prev_obv=900) == 1

    def test_obv_falling(self):
        assert signal_obv(_row(obv=900), prev_obv=1000) == -1

    def test_obv_no_prev(self):
        assert signal_obv(_row(obv=1000), prev_obv=None) == 0

    def test_vwap_above(self):
        assert signal_vwap(_row(close=31000, vwap=30000)) == 1

    def test_vwap_below(self):
        assert signal_vwap(_row(close=29000, vwap=30000)) == -1

    def test_mfi_oversold(self):
        assert signal_mfi(_row(mfi=15)) == 1

    def test_mfi_overbought(self):
        assert signal_mfi(_row(mfi=85)) == -1

    def test_fear_greed_extreme_fear(self):
        assert signal_fear_greed(10) == 1

    def test_fear_greed_extreme_greed(self):
        assert signal_fear_greed(85) == -1

    def test_fear_greed_neutral(self):
        assert signal_fear_greed(50) == 0

    def test_golden_cross(self):
        assert signal_golden_death(_row(recent_golden=1, recent_death=0)) == 1

    def test_death_cross(self):
        assert signal_golden_death(_row(recent_golden=0, recent_death=1)) == -1

    def test_long_trend_bull(self):
        assert signal_long_trend(_row(sma_slow=31000, sma_long=29000)) == 1

    def test_long_trend_bear(self):
        assert signal_long_trend(_row(sma_slow=29000, sma_long=31000)) == -1

    def test_all_signals_valid_set(self):
        """Minden signal {-1, 0, +1} értéket ad vissza."""
        valid = {-1, 0, 1}
        p = IndicatorParams()
        row = _row(
            sma_fast=101, sma_slow=100, ema_fast=101, ema_slow=100,
            macd_hist=0.3, adx=30, plus_di=25, minus_di=10,
            rsi=40, stoch_k=15, stoch_d=12, cci=-120,
            close=29000, bb_lower=29500, bb_upper=31000,
            atr=200, obv=1000, vwap=30000, mfi=50,
            recent_golden=0, recent_death=0, sma_long=29000,
        )
        for fn, args in [
            (signal_sma_cross,   (row,)),
            (signal_ema_cross,   (row,)),
            (signal_macd,        (row,)),
            (signal_adx,         (row,)),
            (signal_rsi,         (row, p)),
            (signal_stochastic,  (row, p)),
            (signal_cci,         (row,)),
            (signal_bollinger,   (row,)),
            (signal_atr,         (row,)),
            (signal_vwap,        (row,)),
            (signal_mfi,         (row,)),
            (signal_golden_death,(row,)),
            (signal_long_trend,  (row,)),
        ]:
            result = fn(*args)
            assert result in valid, f"{fn.__name__} érvénytelen értéket adott: {result}"


# ===========================================================================
# 2. regime.py
# ===========================================================================

class TestRegime:

    def test_trend_above_threshold(self):
        reading = detect_regime(_row(adx=30), RegimeConfig())
        assert reading.label == "trend"
        assert reading.weights == TREND_WEIGHTS

    def test_range_below_threshold(self):
        reading = detect_regime(_row(adx=12), RegimeConfig())
        assert reading.label == "range"
        assert reading.weights == RANGE_WEIGHTS

    def test_neutral_between_thresholds(self):
        reading = detect_regime(_row(adx=21), RegimeConfig())
        assert reading.label == "neutral"
        assert reading.weights == DEFAULT_WEIGHTS

    def test_nan_adx_neutral(self):
        reading = detect_regime(_row(adx=np.nan), RegimeConfig())
        assert reading.label == "neutral"

    def test_adx_value_preserved(self):
        reading = detect_regime(_row(adx=35.5), RegimeConfig())
        assert reading.adx == pytest.approx(35.5)


# ===========================================================================
# 3. backtest.py — helper függvények
# ===========================================================================

class TestBacktestHelpers:

    def test_max_drawdown_empty(self):
        assert _max_drawdown(pd.Series([], dtype=float)) == 0.0

    def test_max_drawdown_flat(self):
        assert _max_drawdown(pd.Series([10000.0] * 100)) == pytest.approx(0.0)

    def test_max_drawdown_50_pct(self):
        eq = pd.Series([10000.0, 8000.0, 6000.0, 5000.0, 7000.0])
        assert _max_drawdown(eq) == pytest.approx(50.0)

    def test_max_drawdown_monotone_up(self):
        assert _max_drawdown(pd.Series([10000.0, 11000.0, 12000.0])) == pytest.approx(0.0)

    def test_initial_stops_atr_based(self):
        stops = StopConfig(use_atr_stops=True, atr_stop_mult=2.0, atr_tp_mult=4.0)
        sl, tp = _initial_stops(30000.0, 300.0, stops)
        assert sl == pytest.approx(30000.0 - 2.0 * 300.0)
        assert tp == pytest.approx(30000.0 + 4.0 * 300.0)

    def test_initial_stops_fixed_pct(self):
        stops = StopConfig(use_atr_stops=False, stop_loss_pct=0.02, take_profit_pct=0.04)
        sl, tp = _initial_stops(30000.0, 0.0, stops)
        assert sl == pytest.approx(30000.0 * 0.98)
        assert tp == pytest.approx(30000.0 * 1.04)

    def test_check_sl_hit(self):
        row = pd.Series({"low": 28000.0, "high": 30500.0})
        assert _check_sl_tp(row, 29000.0, 32000.0) == "stop_loss"

    def test_check_tp_hit(self):
        row = pd.Series({"low": 30000.0, "high": 33000.0})
        assert _check_sl_tp(row, 28000.0, 32000.0) == "take_profit"

    def test_check_sl_tp_neither(self):
        row = pd.Series({"low": 29500.0, "high": 30500.0})
        assert _check_sl_tp(row, 29000.0, 32000.0) is None

    def test_slippage_buy_increases_price(self):
        bt = BacktestConfig(slippage_bps=5, spread_bps=4)
        assert _apply_slippage(30000.0, "BUY", bt) > 30000.0

    def test_slippage_sell_decreases_price(self):
        bt = BacktestConfig(slippage_bps=5, spread_bps=4)
        assert _apply_slippage(30000.0, "SELL", bt) < 30000.0

    def test_open_position_score_proportional(self):
        """score=0.5 → felakkora pozíció mint score=1.0"""
        cfg = TradingConfig()
        cfg.risk.score_proportional_size = True
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        pos_full = _open_position(30000.0, ts, 10000.0, cfg, cfg.backtest, score=1.0)
        pos_half = _open_position(30000.0, ts, 10000.0, cfg, cfg.backtest, score=0.5)
        assert pos_full.size > pos_half.size
        assert pos_half.size == pytest.approx(pos_full.size * 0.5, rel=0.01)

    def test_open_position_no_kelly_ignores_score(self):
        """score_proportional_size=False → score-tól független méret"""
        cfg = TradingConfig()
        cfg.risk.score_proportional_size = False
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        p1 = _open_position(30000.0, ts, 10000.0, cfg, cfg.backtest, score=0.8)
        p2 = _open_position(30000.0, ts, 10000.0, cfg, cfg.backtest, score=0.4)
        assert p1.size == pytest.approx(p2.size, rel=0.001)


# ===========================================================================
# 4. backtest.py — integrációs tesztek
# ===========================================================================

class TestBacktestIntegration:

    def _run(self, df, **cfg_overrides):
        cfg = TradingConfig()
        cfg.mtf.enabled = False
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
        return Backtester(TradingAgent(cfg), cfg).run(df)

    def test_equity_never_negative(self):
        result = self._run(_ohlcv(300, seed=1))
        assert (result.equity_curve >= 0).all()

    def test_final_balance_matches_equity_last(self):
        result = self._run(_ohlcv(300, seed=2))
        assert result.final_balance == pytest.approx(result.equity_curve.iloc[-1], rel=1e-6)

    def test_total_return_consistent_with_balance(self):
        cfg = TradingConfig()
        cfg.mtf.enabled = False
        result = Backtester(TradingAgent(cfg), cfg).run(_ohlcv(300, seed=3))
        expected = (result.final_balance / cfg.initial_balance - 1) * 100
        assert result.total_return_pct == pytest.approx(expected, rel=1e-5)

    def test_trade_pnl_not_nan(self):
        result = self._run(_ohlcv(300, seed=4))
        for t in result.trades:
            assert not math.isnan(t.pnl)

    def test_no_duplicate_entry_times(self):
        """SL/TP után ugyanazon a gyertyán nem nyit újra."""
        result = self._run(_ohlcv(300, seed=5))
        if len(result.trades) > 1:
            times = [t.entry_time for t in result.trades]
            assert len(times) == len(set(times))

    def test_very_strict_atr_filter_blocks_entries(self):
        """Extrém szoros ATR szűrő → nincs vagy alig van belépés"""
        cfg = TradingConfig()
        cfg.mtf.enabled = False
        cfg.risk.max_atr_pct = 0.0001  # szinte minden belépést szűr
        result = Backtester(TradingAgent(cfg), cfg).run(_ohlcv(300, sigma=0.02, seed=6))
        assert len(result.trades) < 5

    def test_bull_market_positive_return(self):
        result = self._run(_ohlcv(300, drift=0.003, sigma=0.008, seed=7))
        assert result.total_return_pct > 0

    def test_drawdown_below_100_pct(self):
        result = self._run(_ohlcv(300, drift=-0.002, sigma=0.015, seed=8))
        assert _max_drawdown(result.equity_curve) < 100.0

    def test_deterministic(self):
        df = _ohlcv(200, seed=42)
        r1 = self._run(df)
        r2 = self._run(df)
        assert r1.total_return_pct == pytest.approx(r2.total_return_pct)
        assert len(r1.trades) == len(r2.trades)


# ===========================================================================
# 5. risk_manager.py
# ===========================================================================

class TestRiskManager:

    def _rm(self, **kwargs):
        return RiskManager(RiskConfig(**kwargs))

    def test_should_open_ok(self):
        ok, _ = self._rm(max_order_value_usd=1000).should_open(30000, 0.01, 0)
        assert ok is True

    def test_should_open_blocks_high_order_value(self):
        ok, reason = self._rm(max_order_value_usd=100).should_open(30000, 0.01, 0)
        assert ok is False
        assert "order value" in reason

    def test_should_open_blocks_high_atr(self):
        ok, reason = self._rm(max_atr_pct=0.02, max_order_value_usd=10_000).should_open(
            price=30000, size=0.001, atr=1500  # ATR/price = 5% > 2%
        )
        assert ok is False
        assert "volatility" in reason

    def test_halted_blocks_all(self):
        rm = self._rm(max_order_value_usd=10_000)
        rm.state.halted = True
        rm.state.halt_reason = "test"
        ok, reason = rm.should_open(100, 1, 0)
        assert ok is False
        assert "halted" in reason

    def test_daily_loss_halts(self):
        rm = self._rm(daily_loss_limit_usd=100, max_order_value_usd=10_000)
        rm.register_trade_pnl(-101)
        assert rm.state.halted is True
        assert "daily" in rm.state.halt_reason

    def test_drawdown_halts(self):
        rm = self._rm(max_drawdown_pct=0.10)
        rm.state.peak_equity = 10_000
        rm.update_equity(8_000)  # 20% drawdown
        assert rm.state.halted is True

    def test_3_losses_halves_size(self):
        rm = self._rm()
        for _ in range(3):
            rm.register_trade_pnl(-10)
        assert rm.state.consecutive_loss_size_mult == pytest.approx(0.5)

    def test_7_losses_trigger_halt(self):
        rm = self._rm()
        for _ in range(7):
            rm.register_trade_pnl(-10)
        assert rm.state.halted is True

    def test_win_resets_consecutive_losses(self):
        rm = self._rm()
        for _ in range(3):
            rm.register_trade_pnl(-10)
        rm.register_trade_pnl(50)
        assert rm.state.consecutive_losses == 0
        assert rm.state.consecutive_loss_size_mult == pytest.approx(1.0)

    def test_cap_size_score_proportional(self):
        cfg = RiskConfig(max_order_value_usd=10_000, score_proportional_size=True)
        rm = RiskManager(cfg)
        full = rm.cap_size_to_limit(1000, 5.0, score=1.0)
        half = rm.cap_size_to_limit(1000, 5.0, score=0.5)
        assert half == pytest.approx(full * 0.5, rel=0.01)

    def test_cap_size_no_kelly(self):
        cfg = RiskConfig(max_order_value_usd=10_000, score_proportional_size=False)
        rm = RiskManager(cfg)
        s1 = rm.cap_size_to_limit(1000, 5.0, score=0.4)
        s2 = rm.cap_size_to_limit(1000, 5.0, score=1.0)
        assert s1 == pytest.approx(s2, rel=0.001)


# ===========================================================================
# 6. orderbook_features.py
# ===========================================================================

class TestOrderBookFeatures:

    def _snap(self, bids, asks):
        return OrderBookSnapshot(bids=bids, asks=asks)

    def test_imbalance_equal(self):
        bids = [(100, 1.0)] * 10
        asks = [(101, 1.0)] * 10
        assert self._snap(bids, asks).imbalance() == pytest.approx(0.0)

    def test_imbalance_only_bids(self):
        bids = [(100, 5.0)] * 10
        asks = [(101, 0.0)] * 10
        assert self._snap(bids, asks).imbalance() == pytest.approx(1.0)

    def test_imbalance_only_asks(self):
        bids = [(100, 0.0)] * 10
        asks = [(101, 5.0)] * 10
        assert self._snap(bids, asks).imbalance() == pytest.approx(-1.0)

    def test_imbalance_in_range(self):
        bids = [(100 - i * 0.1, 10.0 + i) for i in range(20)]
        asks = [(101 + i * 0.1, 8.0 + i) for i in range(20)]
        obi = self._snap(bids, asks).imbalance(levels=10)
        assert -1.0 <= obi <= 1.0

    def test_depth_ratio_equal(self):
        bids = [(100, 1.0)] * 20
        asks = [(101, 1.0)] * 20
        assert self._snap(bids, asks).depth_ratio() == pytest.approx(1.0)

    def test_depth_ratio_more_bids(self):
        bids = [(100, 2.0)] * 20
        asks = [(101, 1.0)] * 20
        assert self._snap(bids, asks).depth_ratio() > 1.0

    def test_large_order_buy_wall(self):
        bids = [(100, 1.0)] * 10
        bids[0] = (100, 100.0)  # top bid >> átlag
        asks = [(101, 1.0)] * 10
        assert self._snap(bids, asks).large_order_signal(top_levels=5, size_multiplier=5.0) == 1

    def test_large_order_sell_wall(self):
        bids = [(100, 1.0)] * 10
        asks = [(101, 1.0)] * 10
        asks[0] = (101, 100.0)
        assert self._snap(bids, asks).large_order_signal(top_levels=5, size_multiplier=5.0) == -1

    def test_large_order_no_wall(self):
        bids = [(100 - i * 0.1, 1.0) for i in range(10)]
        asks = [(101 + i * 0.1, 1.0) for i in range(10)]
        assert self._snap(bids, asks).large_order_signal() == 0

    def test_no_self_comparison_bias(self):
        """
        bids[0] kizárva az átlagból → a nagy top bid helyesen detektálható.
        Ha bids[0] benne lenne az átlagban, saját értékét növelné az átlagot,
        nehezítve a detektálást.
        """
        bids = [(100 - i * 0.1, 1.0) for i in range(10)]
        bids[0] = (100, 10.0)  # 10× az átlag (1.0) → egyértelműen > 5× küszöb
        asks = [(101 + i * 0.1, 1.0) for i in range(10)]
        result = self._snap(bids, asks).large_order_signal(top_levels=5, size_multiplier=5.0)
        assert result == 1

    def test_spread_pct(self):
        bids = [(29990, 1.0)]
        asks = [(30010, 1.0)]
        ob = self._snap(bids, asks)
        assert ob.spread_pct == pytest.approx(20 / 30000, rel=0.01)

    def test_best_bid_ask_mid(self):
        bids = [(100, 1.0), (99, 1.0)]
        asks = [(101, 1.0), (102, 1.0)]
        ob = self._snap(bids, asks)
        assert ob.best_bid == 100
        assert ob.best_ask == 101
        assert ob.mid_price == pytest.approx(100.5)


# ===========================================================================
# 7. cost_model.py
# ===========================================================================

class TestCostModel:

    def _model(self, **kwargs):
        return CostModel(CostConfig(**kwargs))

    def test_slippage_bounded(self):
        model = self._model()
        for size in [100, 1_000, 5_000, 50_000]:
            s = model.estimate_slippage(size, volume_24h_usd=1_000_000, atr_pct=0.01)
            assert 0.0001 <= s <= 0.01, f"Slippage kívül van a tartományon: {s}, méret={size}"

    def test_total_cost_positive(self):
        bd = self._model().calculate(30000, 300, 500, 5_000_000, 0.02)
        assert bd.total_cost_pct > 0

    def test_break_even_less_than_min_return(self):
        bd = self._model().calculate(30000, 300, 500, 5_000_000, 0.05)
        assert bd.break_even_pct < bd.min_return_pct

    def test_filter_trade_rejects_low_return(self):
        ok, _ = self._model(min_return_multiplier=1.5).filter_trade(
            30000, 300, 500, 5_000_000, expected_return_pct=0.001,
        )
        assert ok is False

    def test_filter_trade_accepts_high_return(self):
        ok, _ = self._model(min_return_multiplier=1.5).filter_trade(
            30000, 300, 500, 5_000_000, expected_return_pct=0.05,
        )
        assert ok is True

    def test_large_order_higher_slippage(self):
        model = self._model()
        s_small = model.estimate_slippage(100, 1_000_000, 0.01)
        s_large = model.estimate_slippage(10_000, 1_000_000, 0.01)
        assert s_large >= s_small


# ===========================================================================
# 8. exit_manager.py
# ===========================================================================

class TestExitManager:

    def _em(self, **cfg_kw):
        return ExitManager(ExitConfig(**cfg_kw))

    def test_stop_loss_triggers(self):
        em = self._em()
        em.on_entry(30000, atr=300, atr_stop_mult=2.0, atr_tp_mult=4.0)
        # Stop = 30000 - 600 = 29400 → ár 29000 → stop-loss
        sig = em.on_bar(29000, 300)
        assert sig.should_exit is True
        assert sig.is_partial is False
        assert sig.reason == "stop_loss"

    def test_no_exit_in_normal_range(self):
        em = self._em()
        em.on_entry(30000, atr=300, atr_stop_mult=2.0, atr_tp_mult=4.0)
        sig = em.on_bar(30200, 300)
        assert sig.should_exit is False

    def test_partial_tp1_triggers(self):
        em = self._em(tp1_atr_mult=1.0, partial_tp_fraction=0.5)
        em.on_entry(30000, atr=300, atr_tp_mult=4.0, atr_stop_mult=2.0)
        # TP1 = 30000 + 300 = 30300 → ár 30400 → partial exit
        sig = em.on_bar(30400, 300)
        assert sig.should_exit is True
        assert sig.is_partial is True
        assert sig.exit_fraction == pytest.approx(0.5)
        assert sig.reason == "partial_tp"

    def test_breakeven_after_partial(self):
        em = self._em(tp1_atr_mult=1.0, breakeven_after_partial=True)
        em.on_entry(30000, atr=300, atr_tp_mult=4.0, atr_stop_mult=2.0)
        sig = em.on_bar(30400, 300)
        assert sig.stop_updated is True
        assert sig.new_stop_price == pytest.approx(30000.0)

    def test_time_exit_triggers(self):
        em = self._em(time_exit_bars=3)
        em.on_entry(30000, atr=300, atr_tp_mult=4.0, atr_stop_mult=2.0, max_holding_bars=3)
        for _ in range(3):
            sig = em.on_bar(30100, 300)
        assert sig.should_exit is True
        assert sig.reason == "time_exit"

    def test_reset_clears_state(self):
        em = self._em()
        em.on_entry(30000, atr=300, atr_tp_mult=4.0, atr_stop_mult=2.0)
        em.reset()
        assert em._entry_price is None
        assert em._bars_held == 0
        assert em._partial_tp_done is False


# ===========================================================================
# 9. adaptive_strategy.py
# ===========================================================================

class TestAdaptiveStrategy:

    def test_all_cycles_have_params(self):
        for cycle in MarketCycle:
            assert get_params(cycle) is not None

    def test_positive_delta_raises_buy_threshold(self):
        cfg = TradingConfig()
        base = cfg.buy_threshold
        apply_to_config(cfg, get_params(MarketCycle.RISK_OFF))  # delta = +0.20
        assert cfg.buy_threshold > base

    def test_positive_delta_raises_sell_threshold(self):
        """Pozitív delta → sell_threshold nő (kevésbé negatív = nehezebben ad el)"""
        cfg = TradingConfig()
        base = cfg.sell_threshold
        apply_to_config(cfg, get_params(MarketCycle.RISK_OFF))
        assert cfg.sell_threshold > base

    def test_negative_delta_lowers_thresholds(self):
        cfg = TradingConfig()
        base = cfg.buy_threshold
        apply_to_config(cfg, get_params(MarketCycle.BULL_MID))  # delta = -0.03
        assert cfg.buy_threshold < base

    def test_bear_cycles_forbid_long(self):
        for cycle in [
            MarketCycle.BEAR_EARLY, MarketCycle.BEAR_MID,
            MarketCycle.DISTRIBUTION, MarketCycle.RISK_OFF,
        ]:
            assert get_params(cycle).allow_long is False, (
                f"{cycle.value} tévesen engedi a long-ot"
            )

    def test_bull_mid_largest_position(self):
        bull_mid_pos = get_params(MarketCycle.BULL_MID).max_position_pct
        for cycle in MarketCycle:
            p = get_params(cycle)
            assert p.max_position_pct <= bull_mid_pos + 0.001

    def test_apply_sets_atr_multipliers(self):
        cfg = TradingConfig()
        params = get_params(MarketCycle.BULL_LATE)
        apply_to_config(cfg, params)
        assert cfg.risk.atr_stop_mult == pytest.approx(params.atr_stop_mult)
        assert cfg.risk.atr_tp_mult   == pytest.approx(params.atr_tp_mult)

    def test_describe_contains_cycle_name(self):
        assert "ACCUMULATION" in describe(MarketCycle.ACCUMULATION).upper()


# ===========================================================================
# 10. portfolio_risk.py
# ===========================================================================

class TestPortfolioRisk:

    def _rm(self, **kwargs):
        return PortfolioRiskManager(PortfolioRiskConfig(**kwargs))

    def test_cluster_btc(self):
        assert self._rm().get_cluster("BTC/USDT") == "btc_cluster"

    def test_cluster_eth(self):
        assert self._rm().get_cluster("ETH/USDT") is not None

    def test_cluster_unknown_none(self):
        assert self._rm().get_cluster("UNKNOWN/USDT") is None

    def test_var_zero_for_flat_returns(self):
        assert self._rm().compute_var(np.zeros(100)) == pytest.approx(0.0)

    def test_var_positive_for_volatile_returns(self):
        np.random.seed(0)
        returns = np.random.normal(0, 0.02, 100)
        assert self._rm().compute_var(returns) > 0

    def test_cluster_exposure_sums_to_one(self):
        rm = self._rm()
        exp = rm.cluster_exposure(
            ["BTC/USDT", "ETH/USDT"], [5_000.0, 5_000.0], 10_000.0
        )
        assert sum(exp.values()) <= 1.001  # legfeljebb 100% (lebegőpontos tűréssel)

    def test_high_cluster_blocks_entry(self):
        rm = self._rm(max_cluster_exposure_pct=0.30)
        np.random.seed(0)
        r = np.random.normal(0, 0.01, 50)
        result = rm.check_new_position(
            symbol="BTC/USDT",
            open_symbols=["BTC/USDT"],
            returns_dict={"BTC/USDT": r},
            total_portfolio_usd=10_000,
            new_position_usd=9_000,  # 90% → felülmúlja a 30% limitet
        )
        assert result.can_open is False


# ===========================================================================
# 11. override_engine.py
# ===========================================================================

class TestOverrideEngine:

    def test_no_override_when_disabled(self):
        dec = OverrideEngine(enabled=False).evaluate(
            "BUY", [BlockReason.TIMING_THRESHOLD],
            {"sma_cross": 1}, 0.9, 0.9, 10, _row(),
        )
        assert dec.triggered is False

    def test_no_override_without_blocks(self):
        dec = OverrideEngine(enabled=True).evaluate(
            "BUY", [],
            {"sma_cross": 1}, 0.9, 0.9, 10, _row(),
        )
        assert dec.triggered is False

    def test_low_conviction_no_override(self):
        dec = OverrideEngine(enabled=True).evaluate(
            "BUY", [BlockReason.TIMING_THRESHOLD],
            {"sma_cross": -1, "rsi": -1, "macd": -1},
            score=0.41, ml_prob=0.52, fg_value=50, row=_row(),
        )
        assert dec.triggered is False

    def test_altseason_halving_harder_than_tier(self):
        assert (
            _OVERRIDE_THRESHOLDS[BlockReason.ALTSEASON_HALVING]
            > _OVERRIDE_THRESHOLDS[BlockReason.ALTSEASON_TIER]
        )

    def test_conviction_total_bounded(self):
        dec = OverrideEngine(enabled=True).evaluate(
            "BUY", [BlockReason.TIMING_THRESHOLD],
            {"sma_cross": 1, "rsi": 1, "macd": 1, "adx": 1},
            score=1.0, ml_prob=1.0, fg_value=5,
            row=_row(volume=5000),
        )
        assert 0.0 <= dec.conviction.total <= 1.0


# ===========================================================================
# 12. mtf.py
# ===========================================================================

class TestMTF:

    def test_composite_score_bounded(self):
        a = MTFAnalyzer(["4h"], {"4h": 1.0})
        a.set_data("4h", _ohlcv(100, seed=10))
        assert -1.0 <= a.analyze().composite_score <= 1.0

    def test_label_matches_score_sign(self):
        a = MTFAnalyzer(["4h"], {"4h": 1.0})
        a.set_data("4h", _ohlcv(200, drift=0.01, sigma=0.002, seed=11))
        r = a.analyze()
        if r.composite_score > 0:
            assert r.label == "bullish"
        elif r.composite_score < 0:
            assert r.label == "bearish"
        else:
            assert r.label == "mixed"

    def test_no_data_returns_neutral(self):
        a = MTFAnalyzer(["4h"], {"4h": 1.0})
        assert a.analyze().composite_score == pytest.approx(0.0)

    def test_short_data_valid_signal(self):
        """5 gyertya esetén a jel érvényes értéket ad vissza (kevés adat is kezelt)"""
        a = MTFAnalyzer(["4h"], {"4h": 1.0}, fast=20, slow=50)
        a.set_data("4h", _ohlcv(5, seed=12))
        sig = a.analyze().timeframe_signals.get("4h", 0)
        assert sig in {-1, 0, 1}

    def test_signals_in_valid_set(self):
        a = MTFAnalyzer(["4h", "1d"], {"4h": 1.0, "1d": 1.0})
        a.set_data("4h", _ohlcv(200, seed=13))
        a.set_data("1d", _ohlcv(60, seed=14))
        for tf, sig in a.analyze().timeframe_signals.items():
            assert sig in {-1, 0, 1}


# ===========================================================================
# 13. triple_barrier.py
# ===========================================================================

class TestTripleBarrier:

    @pytest.fixture(scope="class")
    def enriched_bull(self):
        return compute_all(_ohlcv(250, drift=0.002, sigma=0.005, seed=20))

    @pytest.fixture(scope="class")
    def enriched_bear(self):
        return compute_all(_ohlcv(250, drift=-0.002, sigma=0.02, seed=21))

    def test_labels_in_valid_set(self, enriched_bull):
        lb = make_labels(enriched_bull, enriched_bull["atr"],
                         StopConfig(atr_stop_mult=2.0, atr_tp_mult=4.0), 20)
        assert set(lb["label"].unique()).issubset({-1, 0, 1})

    def test_holding_nonnegative(self, enriched_bull):
        lb = make_labels(enriched_bull, enriched_bull["atr"], StopConfig(), 20)
        assert (lb["holding_period"] >= 0).all()

    def test_nan_atr_gives_zero_label(self):
        df = _ohlcv(50, seed=22)
        nan_atr = pd.Series(np.nan, index=df.index)
        lb = make_labels(df, nan_atr, StopConfig(), 10)
        assert (lb["label"] == 0).all()
        assert (lb["holding_period"] == 0).all()

    def test_tp_rows_have_positive_return(self, enriched_bull):
        lb = make_labels(enriched_bull, enriched_bull["atr"],
                         StopConfig(atr_tp_mult=3.0, atr_stop_mult=1.5), 20)
        tp = lb[lb["label"] == 1]
        if len(tp) > 0:
            assert (tp["ret"] > 0).all()

    def test_sl_rows_have_negative_return(self, enriched_bear):
        lb = make_labels(enriched_bear, enriched_bear["atr"],
                         StopConfig(atr_tp_mult=3.0, atr_stop_mult=1.5), 20)
        sl = lb[lb["label"] == -1]
        if len(sl) > 0:
            assert (sl["ret"] < 0).all()


# ===========================================================================
# 14. drift_detector.py
# ===========================================================================

class TestDriftDetector:

    def _dd(self, **kw):
        return DriftDetector(DriftConfig(**kw))

    def test_ok_on_good_varied_trades(self):
        """Változatos nyereséges trade-ek → 'ok' állapot.
        Megjegyzés: ha minden trade azonos értékű, std=0 → Sharpe=0 → 'warn'.
        Ezért kis szórással generálunk trade-eket."""
        dd = self._dd(window=20, min_trades=10)
        np.random.seed(0)
        for pnl in np.random.uniform(5.0, 15.0, 15):  # pozitív, de változatos
            status = dd.update(float(pnl))
        assert status.action == "ok"
        assert status.size_multiplier == pytest.approx(1.0)

    def test_degradation_after_heavy_losses(self):
        dd = self._dd(window=20, min_trades=10)
        for _ in range(3):
            dd.update(5.0)
        for _ in range(12):
            status = dd.update(-10.0)
        assert status.action in {"warn", "reduce_size", "pause", "stop"}

    def test_size_zero_on_pause_or_stop(self):
        dd = self._dd(window=20, min_trades=10, pause_threshold=1, stop_threshold=1)
        for _ in range(15):
            dd.update(-10.0)
        status = dd.check()
        if status.action in {"pause", "stop"}:
            assert status.size_multiplier == pytest.approx(0.0)

    def test_no_detection_below_min_trades(self):
        dd = self._dd(min_trades=10)
        for _ in range(5):
            status = dd.update(-100.0)
        assert status.action == "ok"

    def test_size_half_on_reduce(self):
        dd = self._dd(window=20, min_trades=5, reduce_threshold=2)
        for _ in range(2):
            dd.update(1.0)
        for _ in range(10):
            dd.update(-5.0)
        status = dd.check()
        if status.action == "reduce_size":
            assert status.size_multiplier == pytest.approx(0.5)


# ===========================================================================
# 15. agent.py — end-to-end döntéshozatal
# ===========================================================================

class TestAgentEndToEnd:

    def test_action_always_valid(self):
        agent, _ = _make_agent()
        df = _ohlcv(250, seed=30)
        agent.prepare(df)
        for i in range(100, 250):
            dec = agent.decide_at(i)
            assert dec.action in {"BUY", "SELL", "HOLD"}, (
                f"index {i}: érvénytelen action: {dec.action!r}"
            )

    def test_price_always_positive(self):
        agent, _ = _make_agent()
        df = _ohlcv(250, seed=31)
        agent.prepare(df)
        for i in range(200, 250):
            assert agent.decide_at(i).price > 0

    def test_score_in_range(self):
        agent, _ = _make_agent()
        df = _ohlcv(250, seed=32)
        agent.prepare(df)
        for i in range(200, 250):
            s = agent.decide_at(i).score
            assert -1.0 <= s <= 1.0, f"Score kívül van: {s}"

    def test_no_crash_on_minimal_data(self):
        """Kevés adat (< warmup) sem okoz kivételt"""
        agent, _ = _make_agent()
        df = _ohlcv(10, seed=33)
        agent.prepare(df)
        dec = agent.decide_at(0)
        assert dec.action == "HOLD"

    def test_bull_market_generates_buys(self):
        agent, _ = _make_agent()
        df = _ohlcv(300, drift=0.005, sigma=0.005, seed=34)
        agent.prepare(df)
        actions = [agent.decide_at(i).action for i in range(210, 300)]
        assert "BUY" in actions

    def test_second_prepare_resets_state(self):
        agent, _ = _make_agent()
        agent.prepare(_ohlcv(250, seed=35))
        p1 = agent.decide_at(240).price
        agent.prepare(_ohlcv(250, seed=36))
        p2 = agent.decide_at(240).price
        assert p1 != pytest.approx(p2, rel=1e-6)

    def test_regime_field_always_valid(self):
        agent, _ = _make_agent()
        df = _ohlcv(250, seed=37)
        agent.prepare(df)
        for i in range(220, 250):
            assert agent.decide_at(i).regime in {"trend", "range", "neutral"}

    def test_atr_field_nonnegative(self):
        agent, _ = _make_agent()
        df = _ohlcv(250, seed=38)
        agent.prepare(df)
        for i in range(220, 250):
            assert agent.decide_at(i).atr >= 0
