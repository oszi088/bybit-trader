"""
Optimalizalo - parameter-hangolas overfit-szuressel.

Ezt NEM ML-nek hivjuk: csak nehany parametert hangolunk grid search-csel,
es csak akkor fogadjuk el a uj parametert, ha **mindketto** validacios
foldon legalabb annyira jol teljesit, mint a default.

Walk-forward megkozelites:
    - A teljes idosort 4 egyenlo szakaszra bontjuk: A, B, C, D
    - Tanitas: A+B (in-sample, "IS")
    - Validacio: C (out-of-sample 1, "OOS1")
    - Validacio: D (out-of-sample 2, "OOS2")
    - Csak akkor fogadjuk el az uj parametereket, ha:
        IS hozam > default IS hozam
        OOS1 hozam >= 0  (vagy a default-tol nem rosszabb)
        OOS2 hozam >= 0
    - Ezzel kiszurjuk azokat a "csodakat", amelyek csak az IS adaton fenyesek.

A keresesi ter szandekosan KICSI: csak a buy/sell threshold + 3 fo sulyt
hangoljuk. Igy keves a tulilesztes lehetosege, es a futasi ido is rovid.
"""

from __future__ import annotations

import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

from agent import TradingAgent
from backtest import Backtester, BacktestResult
from config import DEFAULT_WEIGHTS, TradingConfig

logger = logging.getLogger("optimizer")


# ============================================================================
# Keresesi ter (szandekosan kicsi: ne overfitolj!)
# ============================================================================

DEFAULT_GRID: Dict[str, List[float]] = {
    # Donteshozo kuszobok
    "buy_threshold":   [0.20, 0.25, 0.30],
    "sell_threshold":  [-0.30, -0.25, -0.20],
    # 3 fo sulyt hangolunk; a tobbi a default
    "weight_macd":     [1.0, 1.5, 2.0],
    "weight_rsi":      [1.0, 1.5, 2.0],
    "weight_bollinger":[0.5, 1.0, 1.5],
}


# ============================================================================
# Eredmenyek
# ============================================================================

@dataclass
class OptimizationCandidate:
    params: Dict[str, float]
    is_return: float          # in-sample hozam %
    oos1_return: float        # out-of-sample 1 %
    oos2_return: float        # out-of-sample 2 %
    is_drawdown: float        # max DD %
    oos1_drawdown: float
    oos2_drawdown: float

    @property
    def passes_robustness(self) -> bool:
        """A jelolt 'eletkepes' csak akkor, ha mindket OOS foldon nem-negativ."""
        return self.oos1_return >= 0 and self.oos2_return >= 0

    @property
    def avg_oos_return(self) -> float:
        return (self.oos1_return + self.oos2_return) / 2

    def summary(self) -> str:
        flag = "OK " if self.passes_robustness else "x  "
        return (
            f"{flag}IS={self.is_return:+6.2f}% "
            f"OOS1={self.oos1_return:+6.2f}% OOS2={self.oos2_return:+6.2f}% "
            f"avgOOS={self.avg_oos_return:+6.2f}% "
            f"params={self.params}"
        )


@dataclass
class OptimizationResult:
    candidates: List[OptimizationCandidate] = field(default_factory=list)
    default_score: OptimizationCandidate | None = None

    def robust_top(self, n: int = 5) -> List[OptimizationCandidate]:
        """Csak a robusztus (mind OOS pozitiv) jelolteket, atlag OOS hozam szerint."""
        good = [c for c in self.candidates if c.passes_robustness]
        return sorted(good, key=lambda c: c.avg_oos_return, reverse=True)[:n]


# ============================================================================
# Maga az optimalizalo
# ============================================================================

def _apply_params(base: TradingConfig, params: Dict[str, float]) -> TradingConfig:
    """Egy parameter-csomag alkalmazasa egy fris config-ra."""
    cfg = deepcopy(base)
    if "buy_threshold" in params:    cfg.buy_threshold = params["buy_threshold"]
    if "sell_threshold" in params:   cfg.sell_threshold = params["sell_threshold"]
    # Sulyok: a default-bol indulunk, csak amit hangolunk azt valtoztatjuk
    cfg.weights = dict(DEFAULT_WEIGHTS)
    if "weight_macd" in params:      cfg.weights["macd"] = params["weight_macd"]
    if "weight_rsi" in params:       cfg.weights["rsi"] = params["weight_rsi"]
    if "weight_bollinger" in params: cfg.weights["bollinger"] = params["weight_bollinger"]
    # Ezt a regime is hasznalja: ha be van kapcsolva, a default sulyokat
    # a regime felulirja - de a thresholdok mindenhol ervenyesek.
    return cfg


def _split_4(ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    A teljes adat 4 reszre osztva:
      A+B (50%) = IS, C (25%) = OOS1, D (25%) = OOS2.
    """
    n = len(ohlcv)
    is_end = n // 2
    oos1_end = (3 * n) // 4
    is_data = ohlcv.iloc[:is_end]
    oos1 = ohlcv.iloc[is_end:oos1_end]
    oos2 = ohlcv.iloc[oos1_end:]
    return is_data, oos1, oos2


def _run_one(cfg: TradingConfig, data: pd.DataFrame) -> BacktestResult:
    agent = TradingAgent(cfg)
    return Backtester(agent, cfg).run(data)


def optimize(
    base_config: TradingConfig,
    ohlcv: pd.DataFrame,
    grid: Dict[str, List[float]] | None = None,
    max_combinations: int = 200,
) -> OptimizationResult:
    """
    Walk-forward grid search robustness-szuressel.
    """
    grid = grid or DEFAULT_GRID
    is_data, oos1, oos2 = _split_4(ohlcv)
    logger.info("Adat split: IS=%d, OOS1=%d, OOS2=%d",
                len(is_data), len(oos1), len(oos2))

    # Default eredmeny (referencia)
    default_cfg = deepcopy(base_config)
    default = OptimizationCandidate(
        params={"<default>": True},
        is_return=_run_one(default_cfg, is_data).total_return_pct,
        oos1_return=_run_one(default_cfg, oos1).total_return_pct,
        oos2_return=_run_one(default_cfg, oos2).total_return_pct,
        is_drawdown=0, oos1_drawdown=0, oos2_drawdown=0,
    )
    logger.info("Default: %s", default.summary())

    # Combinaciok
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if len(combos) > max_combinations:
        # Egyszeru sub-sample, ne pofazzon orakat
        step = len(combos) // max_combinations + 1
        combos = combos[::step]
    logger.info("Grid meret: %d kombinacio", len(combos))

    result = OptimizationResult(default_score=default)
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        cfg = _apply_params(base_config, params)
        is_r = _run_one(cfg, is_data)
        # Csak akkor folytatjuk a OOS-ben, ha az IS legalabb akkora
        # mint a default - igy gyorsul
        if is_r.total_return_pct < default.is_return:
            continue
        oos1_r = _run_one(cfg, oos1)
        oos2_r = _run_one(cfg, oos2)
        cand = OptimizationCandidate(
            params=params,
            is_return=is_r.total_return_pct,
            oos1_return=oos1_r.total_return_pct,
            oos2_return=oos2_r.total_return_pct,
            is_drawdown=0, oos1_drawdown=0, oos2_drawdown=0,
        )
        result.candidates.append(cand)
        if i % 20 == 0:
            logger.info("...%d/%d kesz", i, len(combos))

    return result
