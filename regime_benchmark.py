"""
Rezsim-szintu benchmark: harom kulonbozo karakteru szintetikus piacon
hasonlit ossze 5 strategia-konfiguraciot. Megmutatja, hogy melyik
strategia melyik rezsimben mukodik jol es melyikben eseik.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from agent import TradingAgent
from backtest import Backtester, _max_drawdown
from config import DEFAULT_WEIGHTS, RANGE_WEIGHTS, TREND_WEIGHTS, TradingConfig

logging.disable(logging.CRITICAL)


# ----------------------------------------------------------------------------
# Adatgenerator: tisztan bull / range / bear piac, mindegyik 1500 1h gyertya
# ----------------------------------------------------------------------------

def gen_market(name: str, drift: float, sigma: float, n: int = 1500,
               init: float = 30000.0, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    returns = np.random.normal(drift, sigma, n)
    close = init * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = np.random.uniform(100, 1000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)
    df.attrs["name"] = name
    df.attrs["buy_hold"] = (close[-1] / close[0] - 1) * 100
    return df


MARKETS = {
    "BULL ":  gen_market("BULL",  drift=+0.0012, sigma=0.014),
    "RANGE":  gen_market("RANGE", drift= 0.0000, sigma=0.018),
    "BEAR ":  gen_market("BEAR",  drift=-0.0010, sigma=0.018),
}


# ----------------------------------------------------------------------------
# Strategia konfigok
# ----------------------------------------------------------------------------

def make_default():
    cfg = TradingConfig()
    cfg.mtf.enabled = False  # gyorsitas
    return cfg


def make_trend_only():
    cfg = make_default()
    cfg.regime.enabled = False
    cfg.weights = dict(TREND_WEIGHTS)
    return cfg


def make_range_only():
    cfg = make_default()
    cfg.regime.enabled = False
    cfg.weights = dict(RANGE_WEIGHTS)
    return cfg


def make_conservative():
    cfg = make_default()
    cfg.position_size = 0.50
    cfg.buy_threshold = 0.45
    cfg.sell_threshold = -0.45
    cfg.stops.atr_stop_mult = 2.0
    cfg.stops.atr_tp_mult = 3.0
    cfg.stops.use_trailing_stop = True
    return cfg


def make_bear_safe():
    """Ha a long-term trend negatív (SMA50<SMA200), nehezebben lép be."""
    cfg = make_default()
    # A long_trend sulyat abszolút magasra emeljük: ha negatív, lényegében
    # blokkolja a BUY-t; ha pozitív, megerősíti
    cfg.weights = dict(DEFAULT_WEIGHTS)
    cfg.weights["long_trend"] = 3.0
    cfg.weights["golden_death"] = 2.5
    return cfg


STRATEGIES = {
    "default":      make_default,
    "trend_only":   make_trend_only,
    "range_only":   make_range_only,
    "conservative": make_conservative,
    "bear_safe":    make_bear_safe,
}


# ----------------------------------------------------------------------------
# Futtatas
# ----------------------------------------------------------------------------

def run_one(cfg, df) -> dict:
    agent = TradingAgent(cfg)
    result = Backtester(agent, cfg).run(df)
    max_dd = _max_drawdown(result.equity_curve)
    win_rate = (sum(1 for t in result.trades if t.pnl > 0)
                / max(1, len(result.trades)) * 100)
    calmar = (result.total_return_pct / max_dd) if max_dd > 0.01 else 0.0
    return {
        "trades": len(result.trades),
        "return": result.total_return_pct,
        "dd":     max_dd,
        "calmar": calmar,
        "win":    win_rate,
    }


def main():
    print(f"{'='*76}\nREZSIM BENCHMARK - 3 piac x 5 strategia\n{'='*76}\n")

    print(f"{'Piac':<8} {'buy-hold':>12}")
    print("-" * 22)
    for name, df in MARKETS.items():
        print(f"{name:<8} {df.attrs['buy_hold']:>+11.1f}%")
    print()

    # Egy nagy tablazat: piac x strategia
    print(f"{'Strategia':<14}", end="")
    for mname in MARKETS:
        print(f" |{mname:^28}", end="")
    print()
    print(f"{'':<14}", end="")
    for _ in MARKETS:
        print(f" |{'return':>8} {'DD':>5} {'cal':>5} {'tr':>4}", end="")
    print()
    print("-" * (14 + 30 * len(MARKETS)))

    # Es mellette osszesites
    overall = {}
    for sname, mkfunc in STRATEGIES.items():
        print(f"{sname:<14}", end="")
        rets, dds = [], []
        for mname, df in MARKETS.items():
            r = run_one(mkfunc(), df)
            print(f" |{r['return']:>+7.1f}% {r['dd']:>4.1f}% "
                  f"{r['calmar']:>+5.2f} {r['trades']:>4d}", end="")
            rets.append(r['return'])
            dds.append(r['dd'])
        avg_ret = sum(rets) / len(rets)
        avg_dd = sum(dds) / len(dds)
        avg_cal = avg_ret / avg_dd if avg_dd > 0.01 else 0
        overall[sname] = (avg_ret, avg_dd, avg_cal)
        print(f"  |  avg ret={avg_ret:+.1f}%  dd={avg_dd:.1f}%  calmar={avg_cal:+.2f}")
    print()

    print("=== ATLAGOLVA HARMOM PIACON ===")
    sortd = sorted(overall.items(), key=lambda kv: kv[1][0], reverse=True)
    print(f"{'Strategia':<14} {'avg return':>12} {'avg DD':>10} {'avg calmar':>12}")
    print("-" * 50)
    for sname, (ret, dd, cal) in sortd:
        print(f"{sname:<14} {ret:>+11.1f}% {dd:>+9.1f}% {cal:>+11.2f}")


if __name__ == "__main__":
    main()
