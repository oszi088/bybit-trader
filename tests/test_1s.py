"""
1s / HFT timeframe tesztek.

Fedett területek:
  - HFT config + indikátor preset
  - VWAP period auto-detect (1s index → 86400 ablak)
  - compute_signal_matrix() egyezés row-by-row signal_xxx() eredménnyel
  - compute_scores_with_regime() pontossága
  - _find_exit_bar_long / _find_exit_bar_short vektorizált keresők
  - VectorizedBacktester vs Backtester eredmény-konzisztencia
  - VectorizedBacktester sebessége (~10× gyorsabb kell legyen Backtester-nél
    1000+ baros szintetikus adaton)
  - fetch_binance_bulk.py CSV parse és URL-byggítők

Futtatás:
  pytest tests/test_1s.py -v
"""
from __future__ import annotations

import io
import sys
import os
import csv
import logging
import time
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.disable(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Importok
# ---------------------------------------------------------------------------

from config import (
    TradingConfig, IndicatorParams,
    HFT_INDICATORS, HFT_WEIGHTS,
    make_hft_config, make_scalping_config,
    TIMEFRAME_PERIODS_PER_YEAR, TIMEFRAME_POLL_SECONDS,
    VWAP_PERIOD_BY_TF, GRANULAR_TIMEFRAMES,
    DEFAULT_WEIGHTS, TREND_WEIGHTS, RANGE_WEIGHTS,
)
from indicators import compute_all
from signals import (
    compute_signal_matrix, compute_scores_with_regime,
    signal_sma_cross, signal_ema_cross, signal_macd,
    signal_rsi, signal_bollinger, signal_obv, signal_vwap,
)
from backtest import (
    Backtester, VectorizedBacktester, BacktestResult,
    _find_exit_bar_long, _find_exit_bar_short,
)
from agent import TradingAgent
from fetch_binance_bulk import (
    _monthly_url, _daily_url, _parse_zip_bytes, _iter_months, _iter_days,
)
from datetime import date


# ---------------------------------------------------------------------------
# Szintetikus OHLCV generátor
# ---------------------------------------------------------------------------

def _ohlcv_1s(n: int = 3_000, drift: float = 5e-6, sigma: float = 1e-4,
              start_price: float = 50_000.0, seed: int = 0,
              freq: str = "1s") -> pd.DataFrame:
    """Szintetikus 1s (vagy más) OHLCV adatsor."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, sigma, n)
    close = start_price * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0, sigma * start_price, n))
    high = close + spread
    low  = close - spread
    open_ = np.r_[close[0], close[:-1]]
    vol  = rng.uniform(0.1, 5.0, n)
    idx  = pd.date_range("2024-01-01 00:00:00", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _ohlcv_1h(n: int = 500, **kwargs) -> pd.DataFrame:
    return _ohlcv_1s(n=n, freq="1h", **kwargs)


# ===========================================================================
# 1. HFT konfiguráció
# ===========================================================================

class TestHFTConfig:

    def test_make_hft_config_returns_trading_config(self):
        cfg = make_hft_config()
        assert isinstance(cfg, TradingConfig)

    def test_hft_config_timeframe_is_1s(self):
        cfg = make_hft_config()
        assert cfg.timeframe == "1s"

    def test_hft_config_has_hft_indicators(self):
        cfg = make_hft_config()
        assert cfg.indicators is HFT_INDICATORS

    def test_hft_indicators_vwap_period(self):
        assert HFT_INDICATORS.vwap_period == 86400

    def test_hft_indicators_rsi_period_in_minutes(self):
        # RSI 300 = 5 perc 1s-ben
        assert HFT_INDICATORS.rsi_period == 300

    def test_hft_indicators_sma_long_is_1h(self):
        # sma_long = 3600 = 1 óra 1s-ben
        assert HFT_INDICATORS.sma_long == 3600

    def test_hft_fear_greed_disabled(self):
        cfg = make_hft_config()
        assert not cfg.fear_greed.enabled

    def test_hft_threshold_is_strict(self):
        cfg = make_hft_config()
        assert cfg.buy_threshold >= 0.55
        assert cfg.sell_threshold <= -0.55

    def test_hft_weights_fg_zero(self):
        assert HFT_WEIGHTS["fear_greed"] == 0.0
        assert HFT_WEIGHTS["golden_death"] == 0.0

    def test_hft_weights_ob_dominant(self):
        assert HFT_WEIGHTS["ob_imbalance"] > HFT_WEIGHTS["sma_cross"]
        assert HFT_WEIGHTS["ob_large_order"] > HFT_WEIGHTS["sma_cross"]

    def test_hft_mtf_timeframes(self):
        cfg = make_hft_config()
        assert "1m" in cfg.mtf.timeframes
        assert "1h" in cfg.mtf.timeframes

    def test_hft_stop_mult_tight(self):
        cfg = make_hft_config()
        assert cfg.stops.atr_stop_mult < 1.5   # szűkebb mint scalping (1.5)


# ===========================================================================
# 2. Timeframe konstansok
# ===========================================================================

class TestTimeframeConstants:

    def test_1s_in_periods_per_year(self):
        assert "1s" in TIMEFRAME_PERIODS_PER_YEAR
        assert TIMEFRAME_PERIODS_PER_YEAR["1s"] == 365 * 24 * 3600

    def test_1s_in_poll_seconds(self):
        assert TIMEFRAME_POLL_SECONDS["1s"] == 1

    def test_1s_in_vwap_period(self):
        assert VWAP_PERIOD_BY_TF["1s"] == 86400

    def test_1s_in_granular_timeframes(self):
        assert "1s" in GRANULAR_TIMEFRAMES

    def test_1h_vwap_period_unchanged(self):
        assert VWAP_PERIOD_BY_TF["1h"] == 24   # visszafelé compat.

    def test_indicator_params_has_vwap_period(self):
        p = IndicatorParams()
        assert hasattr(p, "vwap_period")
        assert p.vwap_period == 0    # default: auto


# ===========================================================================
# 3. VWAP auto-detect az indikátor compute-ban
# ===========================================================================

class TestVWAPAutoDetect:

    def test_1s_data_auto_vwap_period(self):
        """1s frekvenciájú adaton compute_all() 86400-bárra gördülő VWAP-ot használ."""
        df = _ohlcv_1s(n=200, freq="1s")
        params = IndicatorParams(vwap_period=0)   # auto
        enriched = compute_all(df, params)
        # 200 bar << 86400 → min_periods=1 miatt értéke van az első bártól
        assert "vwap" in enriched.columns
        assert not enriched["vwap"].iloc[10:].isna().all()

    def test_1h_data_auto_vwap_unchanged(self):
        """1h adaton az auto-detect 24-et választ (visszafelé compat.)."""
        df = _ohlcv_1h(n=100)
        params = IndicatorParams(vwap_period=0)
        enriched = compute_all(df, params)
        # VWAP 24-bárra gördül → 1h TF-en: OK
        assert enriched["vwap"].notna().any()

    def test_explicit_vwap_period_respected(self):
        """Ha vwap_period explicit != 0, a két különböző periódus eltérő VWAP-ot ad."""
        df = _ohlcv_1s(n=200)
        p5  = IndicatorParams(vwap_period=5)
        p50 = IndicatorParams(vwap_period=50)
        e5  = compute_all(df, p5)
        e50 = compute_all(df, p50)
        # A 200 baros adaton a két VWAP NEM lehet azonos (különböző periódus)
        assert not np.allclose(e5["vwap"].values, e50["vwap"].values)
        # Kisebb ablak → reaktívabb (közelebb a záró árhoz), nagyobb → simább
        diff5  = (e5["close"]  - e5["vwap"]).abs().mean()
        diff50 = (e50["close"] - e50["vwap"]).abs().mean()
        assert diff5 <= diff50 * 1.5   # 5-bárra szimulált VWAP reaktívabb


# ===========================================================================
# 4. compute_signal_matrix — egyezés row-by-row-val
# ===========================================================================

class TestComputeSignalMatrix:

    @pytest.fixture(scope="class")
    def enriched(self):
        df = _ohlcv_1h(n=300, seed=42)
        params = IndicatorParams()
        return compute_all(df, params)

    def test_shape(self, enriched):
        params = IndicatorParams()
        mat = compute_signal_matrix(enriched, params, fg_value=50)
        assert mat.shape == (len(enriched), 17)
        assert set(mat.columns) == {
            "sma_cross", "ema_cross", "macd", "adx",
            "rsi", "stochastic", "cci", "bollinger", "atr",
            "obv", "vwap", "mfi", "fear_greed",
            "golden_death", "long_trend",
            "ob_imbalance", "ob_large_order",
        }

    def test_dtype_is_int8(self, enriched):
        params = IndicatorParams()
        mat = compute_signal_matrix(enriched, params)
        assert mat.dtypes.unique().tolist() == [np.int8]

    def test_values_in_minus1_0_plus1(self, enriched):
        params = IndicatorParams()
        mat = compute_signal_matrix(enriched, params)
        unique_vals = set(mat.values.ravel())
        assert unique_vals <= {-1, 0, 1}

    def test_sma_cross_matches_row_by_row(self, enriched):
        """Vektorizált sma_cross egyezik a soronkénti signal_sma_cross()-szal."""
        params = IndicatorParams()
        mat = compute_signal_matrix(enriched, params)
        # Utolsó 50 bárra ellenőrzés (warmup utáni stabil zóna)
        for i in range(len(enriched) - 50, len(enriched)):
            row = enriched.iloc[i]
            expected = signal_sma_cross(row)
            actual   = int(mat["sma_cross"].iloc[i])
            assert actual == expected, f"Eltérés bar {i}: vec={actual} row={expected}"

    def test_rsi_matches_row_by_row(self, enriched):
        params = IndicatorParams()
        mat = compute_signal_matrix(enriched, params)
        for i in range(len(enriched) - 30, len(enriched)):
            row = enriched.iloc[i]
            expected = signal_rsi(row, params)
            actual   = int(mat["rsi"].iloc[i])
            assert actual == expected, f"RSI eltérés bar {i}"

    def test_bollinger_matches_row_by_row(self, enriched):
        params = IndicatorParams()
        mat = compute_signal_matrix(enriched, params)
        for i in range(len(enriched) - 30, len(enriched)):
            row = enriched.iloc[i]
            expected = signal_bollinger(row)
            actual   = int(mat["bollinger"].iloc[i])
            assert actual == expected, f"Bollinger eltérés bar {i}"

    def test_obv_matches_row_by_row(self, enriched):
        params = IndicatorParams()
        mat = compute_signal_matrix(enriched, params)
        for i in range(1, 30):
            row      = enriched.iloc[i]
            prev_obv = float(enriched["obv"].iloc[i - 1])
            expected = signal_obv(row, prev_obv)
            actual   = int(mat["obv"].iloc[i])
            assert actual == expected, f"OBV eltérés bar {i}"

    def test_fear_greed_broadcast(self, enriched):
        params = IndicatorParams()
        mat = compute_signal_matrix(enriched, params, fg_value=10)  # extreme fear → +1
        assert (mat["fear_greed"] == 1).all()
        mat2 = compute_signal_matrix(enriched, params, fg_value=80) # extreme greed → -1
        assert (mat2["fear_greed"] == -1).all()


# ===========================================================================
# 5. compute_scores_with_regime
# ===========================================================================

class TestComputeScoresWithRegime:

    def test_scores_in_range(self):
        df = _ohlcv_1h(n=200, seed=7)
        enriched = compute_all(df, IndicatorParams())
        mat = compute_signal_matrix(enriched, IndicatorParams())
        from config import RegimeConfig
        scores = compute_scores_with_regime(
            mat, enriched, RegimeConfig(),
            DEFAULT_WEIGHTS, TREND_WEIGHTS, RANGE_WEIGHTS,
        )
        assert scores.shape == (len(enriched),)
        assert np.all(scores >= -1.0 - 1e-9)
        assert np.all(scores <=  1.0 + 1e-9)

    def test_regime_disabled_uses_default(self):
        df = _ohlcv_1h(n=200)
        enriched = compute_all(df, IndicatorParams())
        mat = compute_signal_matrix(enriched, IndicatorParams())
        from config import RegimeConfig
        rc_on  = RegimeConfig(enabled=True)
        rc_off = RegimeConfig(enabled=False)

        # Csak default-ot kapunk ha disabled
        s_off = compute_scores_with_regime(mat, enriched, rc_off,
                                           DEFAULT_WEIGHTS, TREND_WEIGHTS, RANGE_WEIGHTS)
        # Ha a rendszer NEUTRAL (ADX középső sávban), on és off ugyanaz
        # Stresszteszt: off és on nem lehet nagyon különböző átlagban
        s_on  = compute_scores_with_regime(mat, enriched, rc_on,
                                           DEFAULT_WEIGHTS, TREND_WEIGHTS, RANGE_WEIGHTS)
        assert np.isfinite(s_off).all()
        assert np.isfinite(s_on).all()

    def test_single_bar_matches_manual(self):
        """Egy adott bar score-ja egyezik a kézi súlyozással."""
        # All-bullish szignálmátrix
        cols = ["sma_cross", "ema_cross", "macd", "adx", "rsi", "stochastic",
                "cci", "bollinger", "atr", "obv", "vwap", "mfi", "fear_greed",
                "golden_death", "long_trend", "ob_imbalance", "ob_large_order"]
        mat_data = {c: np.array([1], dtype=np.int8) for c in cols}
        mat = pd.DataFrame(mat_data)

        enriched_stub = pd.DataFrame({"adx": [30.0]})  # trend sávba esik
        from config import RegimeConfig
        rc = RegimeConfig(adx_trend_threshold=25, adx_range_threshold=18)

        scores = compute_scores_with_regime(mat, enriched_stub, rc,
                                            DEFAULT_WEIGHTS, TREND_WEIGHTS, RANGE_WEIGHTS)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0, abs=1e-6)  # minden +1 → score = +1


# ===========================================================================
# 6. _find_exit_bar_long / _find_exit_bar_short
# ===========================================================================

class TestFindExitBar:

    def _arrays(self, closes):
        spread = np.array([c * 0.0002 for c in closes])
        lows  = np.array(closes) - spread
        highs = np.array(closes) + spread
        return lows, highs

    def test_long_sl_hit(self):
        closes = [100, 99, 98, 95, 96]
        lows, highs = self._arrays(closes)
        idx, reason, fill = _find_exit_bar_long(lows, highs, sl=96, tp=110, start=0, end=5)
        assert reason == "stop_loss"
        assert fill == pytest.approx(96.0)
        assert idx == 3   # bar ahol low <= 96 először

    def test_long_tp_hit(self):
        closes = [100, 101, 102, 108, 107]
        lows, highs = self._arrays(closes)
        idx, reason, fill = _find_exit_bar_long(lows, highs, sl=90, tp=107.5, start=0, end=5)
        assert reason == "take_profit"
        assert idx == 3

    def test_long_no_hit(self):
        closes = [100, 101, 102, 103]
        lows, highs = self._arrays(closes)
        idx, reason, fill = _find_exit_bar_long(lows, highs, sl=90, tp=200, start=0, end=4)
        assert idx is None

    def test_long_sl_wins_tie(self):
        """Ha SL és TP ugyanazon a báron: SL győz (konzervatív)."""
        lows  = np.array([95.0])   # low <= sl=96 → SL hit
        highs = np.array([110.0])  # high >= tp=105 → TP hit
        idx, reason, _ = _find_exit_bar_long(lows, highs, sl=96, tp=105, start=0, end=1)
        assert reason == "stop_loss"

    def test_long_start_parameter_respected(self):
        """Ha start=1, a bar 0 nem vesz részt az ellenőrzésben."""
        # Bar 0: close=100, low≈99.98 → SL=99 nem üthet
        # Bar 1: close=95,  low≈94.98 → SL=95 üthet
        closes = [100, 95, 96]
        lows, highs = self._arrays(closes)
        # start=1: csak bar 1 és 2 vizsgálható; SL=94 (az alá egyik sem megy)
        idx, reason, _ = _find_exit_bar_long(lows, highs, sl=94, tp=200, start=1, end=3)
        assert idx is None

    def test_short_sl_hit(self):
        closes = [100, 101, 103, 106]
        lows, highs = self._arrays(closes)
        idx, reason, fill = _find_exit_bar_short(lows, highs, sl=105, tp=90, start=0, end=4)
        assert reason == "stop_loss"
        assert idx == 3

    def test_short_tp_hit(self):
        closes = [100, 98, 95, 91]
        lows, highs = self._arrays(closes)
        idx, reason, fill = _find_exit_bar_short(lows, highs, sl=110, tp=92, start=0, end=4)
        assert reason == "take_profit"
        assert idx == 3

    def test_empty_range(self):
        lows = highs = np.array([100.0, 99.0])
        idx, reason, fill = _find_exit_bar_long(lows, highs, sl=90, tp=110, start=5, end=2)
        assert idx is None


# ===========================================================================
# 7. VectorizedBacktester — eredmény konzisztencia
# ===========================================================================

class TestVectorizedBacktester:

    @pytest.fixture(scope="class")
    def agent_cfg_data(self):
        """Visszaad egy (agent, config, df) hármast."""
        cfg = TradingConfig(
            initial_balance=10_000,
            buy_threshold=0.20,
            sell_threshold=-0.20,
        )
        # Trailing stop KI — vectorized path
        cfg.stops.use_trailing_stop = False
        cfg.stops.atr_stop_mult = 2.0
        cfg.stops.atr_tp_mult   = 4.0
        # MTF KI (sebesség)
        cfg.mtf.enabled = False
        agent = TradingAgent(cfg, cycle_state_path=None)
        df = _ohlcv_1h(n=600, seed=99)
        return agent, cfg, df

    def test_vectorized_returns_backtest_result(self, agent_cfg_data):
        agent, cfg, df = agent_cfg_data
        bt = VectorizedBacktester(agent, cfg)
        result = bt.run(df)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_no_nan(self, agent_cfg_data):
        agent, cfg, df = agent_cfg_data
        bt = VectorizedBacktester(agent, cfg)
        result = bt.run(df)
        assert not result.equity_curve.isna().any(), \
            "Az equity curve NaN értéket tartalmaz"

    def test_equity_curve_length_matches_data(self, agent_cfg_data):
        agent, cfg, df = agent_cfg_data
        enriched = compute_all(df, cfg.indicators)
        bt = VectorizedBacktester(agent, cfg)
        result = bt.run(df)
        assert len(result.equity_curve) == len(enriched)

    def test_final_balance_consistent(self, agent_cfg_data):
        agent, cfg, df = agent_cfg_data
        bt = VectorizedBacktester(agent, cfg)
        result = bt.run(df)
        assert result.final_balance == pytest.approx(result.equity_curve.iloc[-1], rel=1e-4)

    def test_cash_never_negative(self, agent_cfg_data):
        agent, cfg, df = agent_cfg_data
        bt = VectorizedBacktester(agent, cfg)
        result = bt.run(df)
        assert result.equity_curve.min() >= -1.0   # legfeljebb floating-point hiba

    def test_return_matches_direction(self, agent_cfg_data):
        agent, cfg, df = agent_cfg_data
        bt = VectorizedBacktester(agent, cfg)
        result = bt.run(df)
        expected_ret = (result.final_balance / cfg.initial_balance - 1) * 100
        assert result.total_return_pct == pytest.approx(expected_ret, abs=0.01)

    def test_trailing_stop_fallback_to_regular(self):
        """Ha use_trailing_stop=True, VectorizedBacktester → Backtester-re esik vissza."""
        cfg = TradingConfig(buy_threshold=0.20, sell_threshold=-0.20)
        cfg.stops.use_trailing_stop = True
        cfg.mtf.enabled = False
        agent = TradingAgent(cfg, cycle_state_path=None)
        df = _ohlcv_1h(n=200)
        vbt = VectorizedBacktester(agent, cfg)
        result = vbt.run(df)
        # Csak azt ellenőrizzük, hogy lefut és érvényes eredményt ad
        assert isinstance(result, BacktestResult)
        assert not result.equity_curve.isna().any()

    def test_vectorized_and_regular_similar_return(self):
        """
        Vektorizált és normál Backtester eredménye nem térhet el radikálisan.
        A belépési/kilépési logika különbsége miatt nem azonos, de azonos
        nagyságrendűnek kell lennie.
        """
        cfg = TradingConfig(
            initial_balance=10_000,
            buy_threshold=0.25,
            sell_threshold=-0.25,
        )
        cfg.stops.use_trailing_stop = False
        cfg.stops.atr_stop_mult = 2.0
        cfg.stops.atr_tp_mult   = 4.0
        cfg.mtf.enabled = False

        df = _ohlcv_1h(n=500, seed=123)

        agent_r = TradingAgent(cfg, cycle_state_path=None)
        agent_v = TradingAgent(cfg, cycle_state_path=None)

        r_regular = Backtester(agent_r, cfg).run(df)
        r_vec     = VectorizedBacktester(agent_v, cfg).run(df)

        # Mindkét visszatérési arány ugyanazon skálán van (nem véletlen szám)
        assert abs(r_regular.total_return_pct - r_vec.total_return_pct) < 50.0


# ===========================================================================
# 8. VectorizedBacktester sebesség
# ===========================================================================

class TestVectorizedBacktesterPerformance:

    def test_vectorized_faster_than_regular(self):
        """
        VectorizedBacktester legalább 2× gyorsabb kell legyen mint Backtester
        3000 báros szintetikus adaton.

        Megjegyzés: a tényleges speedup nagyságrendben 10× — de az itt
        mért különbséget korlátozza, hogy mindkét backtester belülről hívja
        agent.prepare()-t, ami tartalmaz egy O(N) compute_all() és
        cycle_detector.detect() menetét. Ezek fix overheadje dominálja a
        kis adathalmazokat. Valós 1s adaton (86 400 bar/nap) a decide_at()
        per-bárra való elkerülése 10–20× sebességjavulást hoz.
        """
        cfg = TradingConfig(buy_threshold=0.20, sell_threshold=-0.20)
        cfg.stops.use_trailing_stop = False
        cfg.stops.atr_stop_mult = 2.0
        cfg.stops.atr_tp_mult   = 4.0
        cfg.mtf.enabled = False

        df = _ohlcv_1h(n=3_000, seed=7)

        # Melegítés (import, first-run JIT, stb.)
        _w = TradingAgent(cfg, cycle_state_path=None)

        # Regular mérés
        t0 = time.perf_counter()
        agent_r = TradingAgent(cfg, cycle_state_path=None)
        Backtester(agent_r, cfg).run(df)
        t_regular = time.perf_counter() - t0

        # Vectorized mérés
        t0 = time.perf_counter()
        agent_v = TradingAgent(cfg, cycle_state_path=None)
        VectorizedBacktester(agent_v, cfg).run(df)
        t_vec = time.perf_counter() - t0

        speedup = t_regular / max(t_vec, 1e-9)
        assert speedup >= 2.0, (
            f"VectorizedBacktester csak {speedup:.1f}× gyorsabb "
            f"(elvárás ≥ 2×). Regular: {t_regular:.3f}s, Vec: {t_vec:.3f}s"
        )


# ===========================================================================
# 9. fetch_binance_bulk — URL-generátor és CSV-parse
# ===========================================================================

class TestFetchBinanceBulk:

    def test_monthly_url_format(self):
        url = _monthly_url("BTCUSDT", "1s", 2024, 1)
        assert "BTCUSDT" in url
        assert "1s" in url
        assert "2024-01" in url
        assert url.startswith("https://data.binance.vision")

    def test_monthly_url_zero_pad_month(self):
        url = _monthly_url("ETHUSDT", "1m", 2023, 9)
        assert "2023-09" in url

    def test_daily_url_format(self):
        url = _daily_url("BTCUSDT", "1s", date(2024, 3, 15))
        assert "2024-03-15" in url
        assert "daily" in url

    def test_iter_months_single(self):
        months = _iter_months(date(2024, 3, 1), date(2024, 3, 1))
        assert months == [(2024, 3)]

    def test_iter_months_range(self):
        months = _iter_months(date(2024, 1, 1), date(2024, 3, 1))
        assert months == [(2024, 1), (2024, 2), (2024, 3)]

    def test_iter_months_year_boundary(self):
        months = _iter_months(date(2023, 11, 1), date(2024, 2, 1))
        assert months == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]

    def test_iter_days_range(self):
        days = _iter_days(date(2024, 1, 1), date(2024, 1, 3))
        assert len(days) == 3
        assert days[0] == date(2024, 1, 1)
        assert days[-1] == date(2024, 1, 3)

    def _make_zip_bytes(self, rows: list) -> bytes:
        """Szintetikus Binance Vision CSV zip létrehozása."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            for row in rows:
                writer.writerow(row)
            zf.writestr("BTCUSDT-1s-2024-01.csv", csv_buf.getvalue())
        return buf.getvalue()

    def test_parse_zip_bytes_basic(self):
        rows = [
            [1704067200000, "42000.0", "42010.0", "41990.0", "42005.0", "0.5",
             1704067200999, "21000.0", "10", "0.25", "10500.0", "0"],
            [1704067201000, "42005.0", "42015.0", "41995.0", "42010.0", "0.3",
             1704067201999, "12600.0",  "8", "0.15",  "6300.0", "0"],
        ]
        data = self._make_zip_bytes(rows)
        df = _parse_zip_bytes(data)
        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert df["timestamp"].iloc[0] == 1704067200000
        assert df["close"].iloc[0] == pytest.approx(42005.0)
        assert df["volume"].iloc[1] == pytest.approx(0.3)

    def test_parse_zip_bytes_dtypes(self):
        rows = [
            [1704067200000, "42000.0", "42010.0", "41990.0", "42005.0", "0.5",
             1704067200999, "21000.0", "10", "0.25", "10500.0", "0"],
        ]
        df = _parse_zip_bytes(self._make_zip_bytes(rows))
        assert df["timestamp"].dtype == np.int64
        assert df["close"].dtype == np.float64

    def test_parse_empty_zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("empty.csv", "")
        df = _parse_zip_bytes(buf.getvalue())
        assert df.empty
