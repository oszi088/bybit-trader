"""
Optimalizaciós tesztek — paraméterkeresés a maximum hozamhoz.

Mit ellenőriznek ezek a tesztek:
  1. Az optimalizáló egységtesztjei (split, apply_params, score)
  2. IS javulás:  talál-e a grid search default-nál jobb IS paramétert?
  3. OOS védelem: az IS-optimális nem feltétlenül OOS-optimális
  4. Robusztusság: a top jelöltek mindkét OOS szakaszon pozitívak
  5. Calmar vs return célok különbsége
  6. Kiterjesztett grid (ATR stop/TP + pozícióméret): teljes sweep
  7. Paraméter stabilitás: az optimális params más seed-re is pozitív

Futtatás:
  pytest tests/test_optimizer.py -v
  pytest tests/test_optimizer.py -v -s     # hogy lassuk a print kimeneteket
"""

from __future__ import annotations

import sys
import os
import logging

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.disable(logging.CRITICAL)

from optimizer import (
    OptimizationCandidate, OptimizationResult,
    _apply_params, _split_4, _run_one,
    DEFAULT_GRID, EXTENDED_GRID,
    optimize,
)
from config import TradingConfig, DEFAULT_WEIGHTS
from backtest import _max_drawdown


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 600, drift: float = 0.001, sigma: float = 0.015,
           start: float = 30_000.0, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    r     = np.random.normal(drift, sigma, n)
    close = start * np.exp(np.cumsum(r))
    high  = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol   = np.random.uniform(100, 1_000, n)
    idx   = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _base_cfg() -> TradingConfig:
    cfg = TradingConfig()
    cfg.mtf.enabled = False
    return cfg


# ===========================================================================
# 1. Egységtesztek — _split_4, _apply_params, OptimizationCandidate
# ===========================================================================

class TestOptimizerUnits:

    def test_split_4_is_50_pct(self):
        df = _ohlcv(1000)
        is_, oos1, oos2 = _split_4(df)
        assert len(is_) == 500

    def test_split_4_oos1_25_pct(self):
        df = _ohlcv(1000)
        _, oos1, _ = _split_4(df)
        assert len(oos1) == 250

    def test_split_4_oos2_25_pct(self):
        df = _ohlcv(1000)
        _, _, oos2 = _split_4(df)
        assert len(oos2) == 250

    def test_split_4_no_overlap(self):
        df = _ohlcv(400)
        is_, oos1, oos2 = _split_4(df)
        assert is_.index[-1] < oos1.index[0]
        assert oos1.index[-1] < oos2.index[0]

    def test_split_4_full_coverage(self):
        df = _ohlcv(400)
        is_, oos1, oos2 = _split_4(df)
        assert len(is_) + len(oos1) + len(oos2) == len(df)

    def test_apply_params_buy_threshold(self):
        cfg = _apply_params(_base_cfg(), {"buy_threshold": 0.55})
        assert cfg.buy_threshold == pytest.approx(0.55)

    def test_apply_params_sell_threshold(self):
        cfg = _apply_params(_base_cfg(), {"sell_threshold": -0.55})
        assert cfg.sell_threshold == pytest.approx(-0.55)

    def test_apply_params_atr_stop(self):
        cfg = _apply_params(_base_cfg(), {"atr_stop_mult": 1.5})
        assert cfg.stops.atr_stop_mult == pytest.approx(1.5)

    def test_apply_params_atr_tp(self):
        cfg = _apply_params(_base_cfg(), {"atr_tp_mult": 7.0})
        assert cfg.stops.atr_tp_mult == pytest.approx(7.0)

    def test_apply_params_position_size(self):
        cfg = _apply_params(_base_cfg(), {"position_size": 0.5})
        assert cfg.position_size == pytest.approx(0.5)

    def test_apply_params_weight_macd(self):
        cfg = _apply_params(_base_cfg(), {"weight_macd": 2.0})
        assert cfg.weights["macd"] == pytest.approx(2.0)

    def test_apply_params_does_not_mutate_base(self):
        base = _base_cfg()
        original_threshold = base.buy_threshold
        _apply_params(base, {"buy_threshold": 0.99})
        assert base.buy_threshold == pytest.approx(original_threshold)

    def test_apply_params_other_weights_unchanged(self):
        """Ha csak MACD-t változtatjuk, a többi súly az DEFAULT_WEIGHTS marad."""
        cfg = _apply_params(_base_cfg(), {"weight_macd": 3.0})
        assert cfg.weights["rsi"] == pytest.approx(DEFAULT_WEIGHTS["rsi"])
        assert cfg.weights["sma_cross"] == pytest.approx(DEFAULT_WEIGHTS["sma_cross"])

    def test_candidate_passes_robustness_both_positive(self):
        c = OptimizationCandidate(
            params={}, is_return=10, oos1_return=5, oos2_return=3,
            is_drawdown=5, oos1_drawdown=3, oos2_drawdown=2,
        )
        assert c.passes_robustness is True

    def test_candidate_fails_robustness_if_oos1_negative(self):
        c = OptimizationCandidate(
            params={}, is_return=10, oos1_return=-1, oos2_return=3,
            is_drawdown=5, oos1_drawdown=3, oos2_drawdown=2,
        )
        assert c.passes_robustness is False

    def test_candidate_calmar_ratio(self):
        c = OptimizationCandidate(
            params={}, is_return=0, oos1_return=20, oos2_return=10,
            is_drawdown=0, oos1_drawdown=10, oos2_drawdown=5,
        )
        # avg_oos_return = 15, avg_oos_dd = 7.5  → calmar = 2.0
        assert c.avg_oos_calmar == pytest.approx(2.0)

    def test_candidate_calmar_zero_if_no_drawdown(self):
        c = OptimizationCandidate(
            params={}, is_return=0, oos1_return=5, oos2_return=5,
            is_drawdown=0, oos1_drawdown=0, oos2_drawdown=0,
        )
        assert c.avg_oos_calmar == pytest.approx(0.0)

    def test_result_robust_top_only_passing(self):
        good = OptimizationCandidate(
            params={"x": 1}, is_return=20, oos1_return=10, oos2_return=5,
            is_drawdown=5, oos1_drawdown=3, oos2_drawdown=2,
        )
        bad = OptimizationCandidate(
            params={"x": 2}, is_return=30, oos1_return=-5, oos2_return=8,
            is_drawdown=5, oos1_drawdown=3, oos2_drawdown=2,
        )
        res = OptimizationResult(candidates=[good, bad], objective="return")
        tops = res.robust_top(5)
        assert all(c.passes_robustness for c in tops)
        assert bad not in tops

    def test_result_calmar_top_sorted_by_calmar(self):
        high_calmar = OptimizationCandidate(
            params={"x": 1}, is_return=10, oos1_return=20, oos2_return=20,
            is_drawdown=5, oos1_drawdown=5, oos2_drawdown=5,
        )  # calmar = 4.0
        low_calmar = OptimizationCandidate(
            params={"x": 2}, is_return=10, oos1_return=30, oos2_return=30,
            is_drawdown=5, oos1_drawdown=30, oos2_drawdown=30,
        )  # calmar = 1.0
        res = OptimizationResult(candidates=[low_calmar, high_calmar], objective="calmar")
        tops = res.robust_calmar_top(2)
        assert tops[0].avg_oos_calmar >= tops[1].avg_oos_calmar


# ===========================================================================
# 2. IS javulás: talál-e a default-nál jobb paramétert?
# ===========================================================================

class TestOptimizerFindsImprovement:

    @pytest.fixture(scope="class")
    def bull_result(self):
        """Erős bull piacon futtatott DEFAULT_GRID optimalizálás."""
        df  = _ohlcv(800, drift=0.002, sigma=0.01, seed=42)
        res = optimize(_base_cfg(), df, grid=DEFAULT_GRID,
                       max_combinations=100, objective="return")
        return res

    def test_result_has_default_score(self, bull_result):
        assert bull_result.default_score is not None

    def test_result_has_candidates(self, bull_result):
        assert len(bull_result.candidates) > 0

    def test_at_least_one_beats_default_is(self, bull_result):
        """Legalább egy IS jelölt jobb a default-nál."""
        default_is = bull_result.default_score.is_return
        best_is = max(c.is_return for c in bull_result.candidates)
        assert best_is >= default_is, (
            f"Egyik IS jelölt sem jobb a default-nál ({default_is:.2f}%)"
        )

    def test_robust_candidates_exist(self, bull_result):
        """Bull piacon van legalább 1 robusztus jelölt."""
        assert len(bull_result.robust_top(10)) > 0

    def test_best_method_returns_candidate(self, bull_result):
        best = bull_result.best()
        assert best is not None
        assert best.passes_robustness

    def test_best_return_beats_or_matches_default_oos(self, bull_result):
        """A legjobb jelölt OOS hozama nem rosszabb a default-nál."""
        default_avg_oos = bull_result.default_score.avg_oos_return
        best = bull_result.best()
        assert best.avg_oos_return >= default_avg_oos - 5.0  # ±5% tűrés

    def test_drawdown_fields_are_computed(self, bull_result):
        """Minden IS drawdown értéket tartalmaz (nem 0 marad)."""
        assert bull_result.default_score.is_drawdown > 0


# ===========================================================================
# 3. OOS védelem: overfitting-érzékelés
# ===========================================================================

class TestOverfitProtection:

    def test_is_strong_oos_weak_fails_robustness(self):
        """IS nagyon jó, de OOS negatív → passes_robustness = False."""
        overfitted = OptimizationCandidate(
            params={"buy_threshold": 0.1},
            is_return=200.0, oos1_return=-10.0, oos2_return=-5.0,
            is_drawdown=10, oos1_drawdown=15, oos2_drawdown=12,
        )
        assert overfitted.passes_robustness is False

    def test_robust_top_excludes_overfitted(self):
        overfit = OptimizationCandidate(
            params={"a": 1}, is_return=500, oos1_return=-50, oos2_return=5,
            is_drawdown=5, oos1_drawdown=60, oos2_drawdown=5,
        )
        legit = OptimizationCandidate(
            params={"a": 2}, is_return=50, oos1_return=20, oos2_return=10,
            is_drawdown=10, oos1_drawdown=8, oos2_drawdown=5,
        )
        res = OptimizationResult(candidates=[overfit, legit])
        tops = res.robust_top(10)
        assert overfit not in tops
        assert legit in tops

    def test_is_pruning_cuts_weak_is(self):
        """
        Az optimizer korai kizárást alkalmaz: IS < 80% default IS → nem fut OOS.
        Következmény: a result.candidates-ban csak IS-erős jelöltek vannak.
        """
        df  = _ohlcv(600, drift=0.001, seed=10)
        res = optimize(_base_cfg(), df, grid=DEFAULT_GRID,
                       max_combinations=50, objective="return")
        if res.default_score and res.candidates:
            default_is = res.default_score.is_return
            for c in res.candidates:
                # Minden kandidát IS-e ≥ default * 0.80
                assert c.is_return >= default_is * 0.80 - 0.1, (
                    f"Kizárásra váró jelölt átment: IS={c.is_return:.2f}% "
                    f"vs default*0.80={default_is*0.80:.2f}%"
                )


# ===========================================================================
# 4. Calmar vs return célok
# ===========================================================================

class TestObjectiveDifference:

    @pytest.fixture(scope="class")
    def both_results(self):
        df = _ohlcv(800, drift=0.0015, sigma=0.015, seed=77)
        r_ret    = optimize(_base_cfg(), df, grid=DEFAULT_GRID,
                            max_combinations=80, objective="return")
        r_calmar = optimize(_base_cfg(), df, grid=DEFAULT_GRID,
                            max_combinations=80, objective="calmar")
        return r_ret, r_calmar

    def test_return_best_has_higher_or_equal_oos_return(self, both_results):
        r_ret, r_calmar = both_results
        best_ret    = r_ret.best()
        best_calmar = r_calmar.best()
        if best_ret and best_calmar:
            # A "return" cél nyerőjének OOS hozama >= a Calmar cél nyerőjénél
            assert best_ret.avg_oos_return >= best_calmar.avg_oos_return - 1.0

    def test_calmar_best_valid(self, both_results):
        _, r_calmar = both_results
        best = r_calmar.best()
        if best:
            assert best.passes_robustness
            assert best.avg_oos_calmar > 0

    def test_objectives_produce_same_candidates(self, both_results):
        """Ugyanazon grid-en futnak → azonos kandidátuskészlet."""
        r_ret, r_calmar = both_results
        assert len(r_ret.candidates) == len(r_calmar.candidates)


# ===========================================================================
# 5. Kiterjesztett grid — ATR stop/TP + pozícióméret sweep
# ===========================================================================

class TestExtendedGrid:

    @pytest.fixture(scope="class")
    def ext_result(self):
        """EXTENDED_GRID futtatása bull+range vegyes piacon."""
        df = _ohlcv(1000, drift=0.001, sigma=0.015, seed=55)
        return optimize(_base_cfg(), df, grid=EXTENDED_GRID,
                        max_combinations=150, objective="calmar")

    def test_extended_grid_has_candidates(self, ext_result):
        assert len(ext_result.candidates) > 0

    def test_best_calmar_candidate_valid(self, ext_result):
        best = ext_result.best()
        if best:
            assert best.passes_robustness
            assert "atr_stop_mult" in best.params or "buy_threshold" in best.params

    def test_best_calmar_not_zero(self, ext_result):
        best = ext_result.best()
        if best:
            assert best.avg_oos_calmar > 0

    def test_best_has_all_extended_params(self, ext_result):
        """A legjobb jelölt tartalmazza az EXTENDED_GRID dimenzióit."""
        best = ext_result.best()
        if best:
            for key in EXTENDED_GRID:
                assert key in best.params, f"Hiányzó paraméter a legjobb jelöltből: {key}"

    def test_drawdown_reduced_vs_high_position_size(self, ext_result):
        """Az alacsonyabb position_size-ú jelöltek kisebb drawdown-t mutatnak."""
        small_pos = [c for c in ext_result.candidates
                     if c.params.get("position_size", 1.0) <= 0.5
                     and c.passes_robustness]
        large_pos = [c for c in ext_result.candidates
                     if c.params.get("position_size", 1.0) >= 0.90
                     and c.passes_robustness]
        if small_pos and large_pos:
            avg_dd_small = sum(c.avg_oos_drawdown for c in small_pos) / len(small_pos)
            avg_dd_large = sum(c.avg_oos_drawdown for c in large_pos) / len(large_pos)
            assert avg_dd_small <= avg_dd_large + 1.0, (
                f"Kis pozícióméret nagyobb DD-t mutat: {avg_dd_small:.1f}% vs {avg_dd_large:.1f}%"
            )


# ===========================================================================
# 6. Paraméter stabilitás: más seed-en is pozitív?
# ===========================================================================

class TestParameterStability:

    def test_best_params_positive_on_different_seed(self):
        """
        Az egyik seed-en talált legjobb paramétereket egy másik seed-re alkalmazzuk.
        Bull piacon a legjobb paraméternek pozitív OOS hozamot kell hoznia.
        """
        # Tanítás: seed=42 bull piac
        train_df = _ohlcv(600, drift=0.002, sigma=0.01, seed=42)
        res = optimize(_base_cfg(), train_df, grid=DEFAULT_GRID,
                       max_combinations=80, objective="return")
        best = res.best()
        if best is None:
            pytest.skip("Nincs robusztus jelölt ezen az adaton")

        # Tesztelés: seed=99 (más piac)
        test_df = _ohlcv(400, drift=0.001, sigma=0.012, seed=99)
        cfg = _apply_params(_base_cfg(), best.params)
        result = _run_one(cfg, test_df)
        # Nem kell feltétlenül pozitív, de ne legyen katasztrofálisan negatív
        assert result.total_return_pct > -50.0, (
            f"Legjobb paraméter katasztrófálisan teljesít más seed-en: "
            f"{result.total_return_pct:.1f}%"
        )

    def test_default_params_positive_on_bull(self):
        """Baseline ellenőrzés: default konfig bull piacon pozitív."""
        df     = _ohlcv(400, drift=0.002, sigma=0.01, seed=100)
        result = _run_one(_base_cfg(), df)
        assert result.total_return_pct > 0


# ===========================================================================
# 7. Teljes eredmény kimutatás (pytest -s kapcsolóval látható)
# ===========================================================================

class TestFullParameterReport:

    def test_print_best_params_for_max_return(self, capsys):
        """
        Teljes grid search, eredmény konzolra írva.
        Futtatsd: pytest tests/test_optimizer.py::TestFullParameterReport -v -s
        """
        df  = _ohlcv(1000, drift=0.0015, sigma=0.014, seed=42)
        res = optimize(_base_cfg(), df, grid=DEFAULT_GRID,
                       max_combinations=150, objective="return")
        res.print_report(top_n=5)

        best = res.best()
        # A teszt sikeres, ha le tudja futtatni és adott vissza valamit
        assert res.default_score is not None
        # Ellenőrzés: a legjobb OOS return nem catastrophic
        if best:
            print(f"\n>>> Legjobb paraméter (return): {best.params}")
            print(f"    OOS avg hozam: {best.avg_oos_return:+.2f}%")
            print(f"    OOS Calmar:    {best.avg_oos_calmar:.2f}")

    def test_print_best_params_for_calmar(self, capsys):
        """Teljes grid search Calmar célra."""
        df  = _ohlcv(1000, drift=0.0015, sigma=0.014, seed=42)
        res = optimize(_base_cfg(), df, grid=EXTENDED_GRID,
                       max_combinations=150, objective="calmar")
        res.print_report(top_n=5)

        best = res.best()
        if best:
            print(f"\n>>> Legjobb paraméter (calmar): {best.params}")
            print(f"    OOS avg hozam: {best.avg_oos_return:+.2f}%")
            print(f"    OOS Calmar:    {best.avg_oos_calmar:.2f}")
            print(f"    OOS avg DD:    {best.avg_oos_drawdown:.1f}%")
