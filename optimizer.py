"""
Optimalizalo - parameter-hangolas overfit-szuressel.

Walk-forward megkozelites:
    - A teljes idosort 4 egyenlo szakaszra bontjuk: A, B, C, D
    - Tanitas: A+B (in-sample, "IS")
    - Validacio: C (out-of-sample 1, "OOS1")
    - Validacio: D (out-of-sample 2, "OOS2")
    - Csak akkor fogadjuk el az uj parametert, ha:
        IS ertek > default IS ertek
        OOS1 hozam >= 0  (vagy a default-tol nem rosszabb)
        OOS2 hozam >= 0
    - Ezzel kiszurjuk azokat a "csodalat", amelyek csak az IS adaton fenyesek.

Optimalizalasi cel (objective):
    "return"  — max OOS atlaghozam (legmagasabb hozam)
    "calmar"  — max OOS atlag Calmar ratio (hozam / drawdown; stabil hozam)

Keresesi terek:
    DEFAULT_GRID  — threshold + sulyok (kicsi grid, gyors)
    EXTENDED_GRID — + ATR stop/TP + poziciomeret (bovebb, lassabb)
"""

from __future__ import annotations

import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import pandas as pd

from agent import TradingAgent
from backtest import Backtester, BacktestResult, _max_drawdown
from config import DEFAULT_WEIGHTS, TradingConfig

logger = logging.getLogger("optimizer")

Objective = Literal["return", "calmar"]


# ============================================================================
# Keresesi terek
# ============================================================================

DEFAULT_GRID: Dict[str, List[float]] = {
    "buy_threshold":    [0.25, 0.35, 0.45],
    "sell_threshold":   [-0.45, -0.35, -0.25],
    "weight_macd":      [1.0, 1.5, 2.0],
    "weight_rsi":       [1.0, 1.5, 2.0],
    "weight_bollinger": [0.5, 1.0, 1.5],
}

# Kibovitett grid: a legfontosabb kockazatkezelesi parameterek is benne vannak.
# ~192 kombináció (gyors, de lefedi a legfontosabb dimenziókat).
EXTENDED_GRID: Dict[str, List[float]] = {
    "buy_threshold":    [0.25, 0.35, 0.45],
    "sell_threshold":   [-0.45, -0.35, -0.25],
    "atr_stop_mult":    [1.5, 2.5, 3.5],
    "atr_tp_mult":      [3.0, 5.0, 7.0],
    "position_size":    [0.5, 0.75, 0.95],
    "weight_macd":      [1.0, 2.0],
    "weight_rsi":       [1.0, 2.0],
}


# ============================================================================
# Eredmenyek
# ============================================================================

@dataclass
class OptimizationCandidate:
    params: Dict[str, float]
    is_return: float          # in-sample hozam %
    oos1_return: float        # out-of-sample 1 hozam %
    oos2_return: float        # out-of-sample 2 hozam %
    is_drawdown: float        # in-sample max DD %
    oos1_drawdown: float      # OOS1 max DD %
    oos2_drawdown: float      # OOS2 max DD %

    @property
    def passes_robustness(self) -> bool:
        """Csak akkor 'eletkepes', ha mindket OOS foldon nem-negativ a hozam."""
        return self.oos1_return >= 0 and self.oos2_return >= 0

    @property
    def avg_oos_return(self) -> float:
        return (self.oos1_return + self.oos2_return) / 2

    @property
    def avg_oos_drawdown(self) -> float:
        return (self.oos1_drawdown + self.oos2_drawdown) / 2

    @property
    def avg_oos_calmar(self) -> float:
        """OOS atlag Calmar: avg_return / avg_DD. Ha DD=0, 0.0."""
        dd = self.avg_oos_drawdown
        return self.avg_oos_return / dd if dd > 0.01 else 0.0

    @property
    def is_calmar(self) -> float:
        return self.is_return / self.is_drawdown if self.is_drawdown > 0.01 else 0.0

    def summary(self) -> str:
        flag = "OK " if self.passes_robustness else "x  "
        return (
            f"{flag}IS={self.is_return:+7.2f}%(DD={self.is_drawdown:.1f}%) "
            f"OOS1={self.oos1_return:+7.2f}% OOS2={self.oos2_return:+7.2f}% "
            f"avgOOS={self.avg_oos_return:+7.2f}% calmar={self.avg_oos_calmar:.2f} "
            f"| {self.params}"
        )


@dataclass
class OptimizationResult:
    candidates: List[OptimizationCandidate] = field(default_factory=list)
    default_score: OptimizationCandidate | None = None
    objective: str = "return"

    def robust_top(self, n: int = 5) -> List[OptimizationCandidate]:
        """Csak a robusztus jeloltek, atlag OOS hozam szerint csokkenoen."""
        good = [c for c in self.candidates if c.passes_robustness]
        return sorted(good, key=lambda c: c.avg_oos_return, reverse=True)[:n]

    def robust_calmar_top(self, n: int = 5) -> List[OptimizationCandidate]:
        """Csak a robusztus jeloltek, OOS Calmar ratio szerint csokkenoen."""
        good = [c for c in self.candidates if c.passes_robustness]
        return sorted(good, key=lambda c: c.avg_oos_calmar, reverse=True)[:n]

    def best(self) -> OptimizationCandidate | None:
        """A legjobb robusztus jelolt az objective szerint."""
        if self.objective == "calmar":
            tops = self.robust_calmar_top(1)
        else:
            tops = self.robust_top(1)
        return tops[0] if tops else None

    def print_report(self, top_n: int = 5) -> None:
        """Rendezett eredmeny kiiras."""
        print(f"\n=== Optimalizalas eredmenye (cel: {self.objective}) ===")
        if self.default_score:
            print(f"  Default:  {self.default_score.summary()}")
        print(f"  Osszes jelolt: {len(self.candidates)}")
        robust = [c for c in self.candidates if c.passes_robustness]
        print(f"  Robusztus:     {len(robust)}")

        tops = self.robust_calmar_top(top_n) if self.objective == "calmar" else self.robust_top(top_n)
        if not tops:
            print("  [!] Nincs robusztus jelolt — default parameterek maradnak.")
            return

        print(f"\n  Top {len(tops)} ({self.objective} szerint):")
        for i, c in enumerate(tops, 1):
            print(f"    #{i}: {c.summary()}")


# ============================================================================
# Belso segedelyfuggvenyek
# ============================================================================

def _apply_params(base: TradingConfig, params: Dict[str, float]) -> TradingConfig:
    """Egy parameter-csomag alkalmazasa egy fris config-masolatra."""
    cfg = deepcopy(base)
    if "buy_threshold"  in params: cfg.buy_threshold  = params["buy_threshold"]
    if "sell_threshold" in params: cfg.sell_threshold = params["sell_threshold"]
    if "position_size"  in params: cfg.position_size  = params["position_size"]
    if "atr_stop_mult"  in params: cfg.stops.atr_stop_mult = params["atr_stop_mult"]
    if "atr_tp_mult"    in params: cfg.stops.atr_tp_mult   = params["atr_tp_mult"]
    # Sulyok: default-bol indulunk, csak a hangolt dimenziot csereljuk
    cfg.weights = dict(DEFAULT_WEIGHTS)
    if "weight_macd"      in params: cfg.weights["macd"]      = params["weight_macd"]
    if "weight_rsi"       in params: cfg.weights["rsi"]       = params["weight_rsi"]
    if "weight_bollinger" in params: cfg.weights["bollinger"] = params["weight_bollinger"]
    if "weight_adx"       in params: cfg.weights["adx"]       = params["weight_adx"]
    return cfg


def _split_4(ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    A teljes adat 4 reszre osztva:
      A+B (50%) = IS, C (25%) = OOS1, D (25%) = OOS2.
    """
    n = len(ohlcv)
    is_end   = n // 2
    oos1_end = (3 * n) // 4
    return ohlcv.iloc[:is_end], ohlcv.iloc[is_end:oos1_end], ohlcv.iloc[oos1_end:]


def _run_one(cfg: TradingConfig, data: pd.DataFrame) -> BacktestResult:
    cfg.mtf.enabled = False   # gyorsitas
    return Backtester(TradingAgent(cfg), cfg).run(data)


def _score(candidate: OptimizationCandidate, objective: Objective) -> float:
    """Rendezesi kulcs: magasabb = jobb."""
    if objective == "calmar":
        return candidate.avg_oos_calmar
    return candidate.avg_oos_return


# ============================================================================
# Fo optimalizalo
# ============================================================================

def optimize(
    base_config: TradingConfig,
    ohlcv: pd.DataFrame,
    grid: Dict[str, List[float]] | None = None,
    max_combinations: int = 300,
    objective: Objective = "return",
) -> OptimizationResult:
    """
    Walk-forward grid search robustness-szuressel.

    Parametrek:
        base_config     : kiindulo TradingConfig (nem modositja)
        ohlcv           : teljes OHLCV idosor
        grid            : keresesi ter (None = DEFAULT_GRID)
        max_combinations: max probalt kombinacio (tobb → pontosabb, lassabb)
        objective       : "return" = max OOS hozam | "calmar" = max OOS Calmar

    Visszateres:
        OptimizationResult  (result.best() adja a legjobb robusztus jeloltet)
    """
    grid = grid or DEFAULT_GRID
    is_data, oos1, oos2 = _split_4(ohlcv)
    logger.info("Adat split: IS=%d  OOS1=%d  OOS2=%d",
                len(is_data), len(oos1), len(oos2))

    # --- Default referencia ---
    def_cfg = deepcopy(base_config)
    def_is   = _run_one(def_cfg, is_data)
    def_oos1 = _run_one(def_cfg, oos1)
    def_oos2 = _run_one(def_cfg, oos2)
    default = OptimizationCandidate(
        params={"<default>": 0},
        is_return=def_is.total_return_pct,
        oos1_return=def_oos1.total_return_pct,
        oos2_return=def_oos2.total_return_pct,
        is_drawdown=_max_drawdown(def_is.equity_curve),
        oos1_drawdown=_max_drawdown(def_oos1.equity_curve),
        oos2_drawdown=_max_drawdown(def_oos2.equity_curve),
    )
    logger.info("Default: %s", default.summary())

    # --- Kombinaciok ---
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if len(combos) > max_combinations:
        step   = len(combos) // max_combinations + 1
        combos = combos[::step]
    logger.info("Grid: %d/%d kombinacio vizsgalva", len(combos),
                len(list(itertools.product(*[grid[k] for k in keys]))))

    result = OptimizationResult(default_score=default, objective=objective)

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        cfg    = _apply_params(base_config, params)

        is_r = _run_one(cfg, is_data)
        # Korai kizaras: IS nem jobb a default-nal → nem erdemli meg az OOS-t
        if is_r.total_return_pct < default.is_return * 0.80:
            continue

        oos1_r = _run_one(cfg, oos1)
        oos2_r = _run_one(cfg, oos2)

        cand = OptimizationCandidate(
            params=params,
            is_return=is_r.total_return_pct,
            oos1_return=oos1_r.total_return_pct,
            oos2_return=oos2_r.total_return_pct,
            is_drawdown=_max_drawdown(is_r.equity_curve),
            oos1_drawdown=_max_drawdown(oos1_r.equity_curve),
            oos2_drawdown=_max_drawdown(oos2_r.equity_curve),
        )
        result.candidates.append(cand)
        if i % 50 == 0:
            logger.info("  ...%d/%d", i, len(combos))

    return result
