"""
Drawdown-tudatos benchmark: tobbfele konfig kombinacio kiprobalasa
ugyanazon az idosoron. Calmar ratio (return / max_drawdown) szerint
rangsorol - igy egy magas hozam-de-magas-drawdown strategia rosszabb,
mint egy kozepesen jo, de stabil.

A szintetikus idosor 4 szakaszbol all:
  bull (felfele trend) -> range (oldalazas) -> bear (drawdown) -> recovery
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent import TradingAgent
from backtest import Backtester, _max_drawdown
from config import DEFAULT_WEIGHTS, RANGE_WEIGHTS, TREND_WEIGHTS, TradingConfig

logging.disable(logging.CRITICAL)


# ============================================================================
# Realistic szintetikus adat: 4 piaci rezsim
# ============================================================================

def generate_multi_regime(seed: int = 42) -> pd.DataFrame:
    """
    5000 1h gyertya, 4 fazisban:
      0..1500    : bull   (drift +0.0008, sigma 0.012)
      1500..3000 : range  (drift 0,        sigma 0.015)
      3000..4000 : bear   (drift -0.0010, sigma 0.020)
      4000..5000 : recovery (drift +0.0012, sigma 0.014)
    """
    np.random.seed(seed)

    def segment(n, drift, sigma):
        return np.random.normal(drift, sigma, n)

    returns = np.concatenate([
        segment(500,  0.0008, 0.012),  # bull
        segment(500,  0.0000, 0.015),  # range
        segment(500, -0.0010, 0.020),  # bear
        segment(500,  0.0012, 0.014),  # recovery
    ])
    close = 30000.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.008, len(close))))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, len(close))))
    open_ = np.r_[close[0], close[:-1]]
    volume = np.random.uniform(100, 1000, len(close))
    idx = pd.date_range("2024-01-01", periods=len(close), freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


# ============================================================================
# Preset konfiguraciok
# ============================================================================

@dataclass
class Preset:
    name: str
    apply: callable   # (cfg) -> None  - in-place modositja a configot


def _no_op(cfg):
    pass


def _atr_tight(cfg):
    cfg.stops.atr_stop_mult = 1.5
    cfg.stops.atr_tp_mult = 4.0


def _atr_loose(cfg):
    cfg.stops.atr_stop_mult = 3.0
    cfg.stops.atr_tp_mult = 5.0


def _no_trailing(cfg):
    cfg.stops.use_trailing_stop = False


def _mtf_gate(cfg):
    cfg.mtf.mode = "gate"
    cfg.mtf.gate_threshold = 0.2


def _mtf_strong(cfg):
    cfg.mtf.mode = "weighted"
    cfg.mtf.composite_weight = 2.5


def _vol_strict(cfg):
    cfg.risk.max_atr_pct = 0.025


def _no_kelly(cfg):
    cfg.risk.score_proportional_size = False


def _conservative_size(cfg):
    cfg.position_size = 0.5


def _high_threshold(cfg):
    cfg.buy_threshold = 0.40
    cfg.sell_threshold = -0.40


def _trend_only(cfg):
    cfg.regime.enabled = False
    cfg.weights = dict(TREND_WEIGHTS)


def _range_only(cfg):
    cfg.regime.enabled = False
    cfg.weights = dict(RANGE_WEIGHTS)


PRESETS = [
    Preset("default",          _no_op),
    Preset("atr_tight",        _atr_tight),
    Preset("atr_loose",        _atr_loose),
    Preset("no_trailing",      _no_trailing),
    Preset("mtf_gate",         _mtf_gate),
    Preset("mtf_strong",       _mtf_strong),
    Preset("vol_strict",       _vol_strict),
    Preset("no_kelly",         _no_kelly),
    Preset("conservative_50%", _conservative_size),
    Preset("high_threshold",   _high_threshold),
    Preset("trend_only",       _trend_only),
    Preset("range_only",       _range_only),
]




# --- Kombinalt preset-ek (drawdown-tudatos) ---

def _ht_atr_loose(cfg):
    cfg.buy_threshold = 0.40; cfg.sell_threshold = -0.40
    cfg.stops.atr_stop_mult = 3.0; cfg.stops.atr_tp_mult = 5.0


def _ht_conservative(cfg):
    cfg.buy_threshold = 0.40; cfg.sell_threshold = -0.40
    cfg.position_size = 0.5


def _ht_atr_conservative(cfg):
    cfg.buy_threshold = 0.40; cfg.sell_threshold = -0.40
    cfg.stops.atr_stop_mult = 3.0; cfg.stops.atr_tp_mult = 5.0
    cfg.position_size = 0.5


def _trend_high_thr(cfg):
    cfg.regime.enabled = False
    cfg.weights = dict(TREND_WEIGHTS)
    cfg.buy_threshold = 0.40; cfg.sell_threshold = -0.40


def _trend_atr_loose(cfg):
    cfg.regime.enabled = False
    cfg.weights = dict(TREND_WEIGHTS)
    cfg.stops.atr_stop_mult = 3.0; cfg.stops.atr_tp_mult = 5.0


def _trend_conservative(cfg):
    cfg.regime.enabled = False
    cfg.weights = dict(TREND_WEIGHTS)
    cfg.position_size = 0.5


PRESETS.extend([
    Preset("HT+atr_loose",           _ht_atr_loose),
    Preset("HT+conservative_50%",    _ht_conservative),
    Preset("HT+atr_loose+cons50%",   _ht_atr_conservative),
    Preset("trend+high_threshold",   _trend_high_thr),
    Preset("trend+atr_loose",        _trend_atr_loose),
    Preset("trend+conservative_50%", _trend_conservative),
])


# ============================================================================
# Fo benchmark
# ============================================================================

def calmar(total_return_pct: float, max_dd_pct: float) -> float:
    """Calmar-szeru metrika: ha a DD = 0 (nincs trade), 0-t adunk."""
    if max_dd_pct <= 0.01:
        return 0.0
    return total_return_pct / max_dd_pct


def run_benchmark(df: pd.DataFrame):
    print(f"Adat: {len(df)} 1h gyertya, ar {df['close'].iloc[0]:.0f} -> {df['close'].iloc[-1]:.0f} "
          f"({(df['close'].iloc[-1]/df['close'].iloc[0] - 1)*100:+.1f}% buy-and-hold)")
    print()
    rows = []
    for p in PRESETS:
        cfg = TradingConfig()
        # Benchmark gyorsitasahoz az MTF-et kikapcsoljuk
        # (a strategia magjat hasonlitjuk, nem a MTF rateget)
        cfg.mtf.enabled = False
        p.apply(cfg)
        agent = TradingAgent(cfg)
        result = Backtester(agent, cfg).run(df)
        max_dd = _max_drawdown(result.equity_curve)
        rows.append({
            "preset": p.name,
            "trades": len(result.trades),
            "return_pct": result.total_return_pct,
            "max_dd_pct": max_dd,
            "calmar": calmar(result.total_return_pct, max_dd),
            "win_rate": (sum(1 for t in result.trades if t.pnl > 0)
                         / max(1, len(result.trades)) * 100),
        })

    # Sorrend: Calmar szerint csokkeno
    rows.sort(key=lambda r: r["return_pct"], reverse=True)

    print(f"{'preset':<20} {'trades':>6} {'return':>10} "
          f"{'maxDD':>8} {'calmar':>8} {'win%':>6}")
    print("-" * 64)
    for r in rows:
        print(f"{r['preset']:<20} {r['trades']:>6} "
              f"{r['return_pct']:>+9.2f}% {r['max_dd_pct']:>7.1f}% "
              f"{r['calmar']:>8.2f} {r['win_rate']:>5.1f}%")
    print()
    best = rows[0]
    print(f"Legjobb (HOZAM alapjan): {best['preset']}")
    print(f"  hozam={best['return_pct']:+.2f}%  maxDD={best['max_dd_pct']:.1f}%  "
          f"calmar={best['calmar']:.2f}")
    return rows


if __name__ == "__main__":
    df = generate_multi_regime(seed=42)
    run_benchmark(df)
