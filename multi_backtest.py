"""
Multi-asset batch backteszt: egy mappa osszes CSV-jet lefuttatja
es egy aggregalt teljesitmeny-tablazatot ad.

Hasznalat:
    # Eloszor letoltjuk az adatokat (sajat gepen):
    python fetch_history.py --symbols top20 --timeframes 1h,4h,1d --years 2

    # Aztan lefuttatjuk az osszes letoltott CSV-n:
    python multi_backtest.py --data-dir data

A kimenet: minden coin/timeframe-re egy sor (hozam, DD, Calmar, win%),
plusz egy aggregalt osszesites: hany coinon pozitiv a hozam, mennyi a median.

Igy egyetlen parancs eldonti, hogy a strategia
  * minden coinon megy-e (robusztus), vagy csak BTC-en (overfitted),
  * minden timeframe-en (multi-tf hibakezelt), vagy csak 1h-n megy,
  * mennyire egyenletes a teljesitmeny.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from agent import TradingAgent
from backtest import Backtester, _max_drawdown
from config import TradingConfig
from data_source import load_csv

logging.disable(logging.CRITICAL)


@dataclass
class FileResult:
    """Egy CSV-fajl backteszt-eredmenye."""
    path: str
    symbol: str
    timeframe: str
    bars: int
    return_pct: float
    max_dd_pct: float
    calmar: float
    win_rate: float
    trades: int
    buy_hold_pct: float

    def row(self) -> str:
        return (
            f"{self.symbol:<12} {self.timeframe:>4} {self.bars:>5}  "
            f"return={self.return_pct:>+7.1f}%  "
            f"DD={self.max_dd_pct:>5.1f}%  "
            f"calmar={self.calmar:>5.2f}  "
            f"win={self.win_rate:>4.1f}%  "
            f"trades={self.trades:>4}  "
            f"vs BH={self.return_pct - self.buy_hold_pct:>+6.1f}%"
        )


# Filename patterns: BTC_USDT_1h_1.0y.csv -> symbol=BTC/USDT, tf=1h
FNAME_RE = re.compile(r"^([A-Z0-9]+)_([A-Z0-9]+)_([0-9]+[mhdwM])_.*\.csv$")


def parse_filename(filename: str) -> tuple[str, str]:
    """Visszaadja a (symbol, timeframe) parosat a fajlnevbol."""
    base = os.path.basename(filename)
    m = FNAME_RE.match(base)
    if not m:
        return base, "?"
    return f"{m.group(1)}/{m.group(2)}", m.group(3)


def calmar(return_pct: float, dd_pct: float) -> float:
    """Hozam / drawdown arany. Ha a DD ~0, 0-t adunk vissza."""
    return return_pct / dd_pct if dd_pct > 0.01 else 0.0


def run_one(path: str, base_config: TradingConfig) -> Optional[FileResult]:
    try:
        df = load_csv(path)
    except Exception as e:
        print(f"  [HIBA] {path}: {e}", file=sys.stderr)
        return None
    if len(df) < 100:
        print(f"  [SKIP] {path}: tul keves gyertya ({len(df)})", file=sys.stderr)
        return None

    sym, tf = parse_filename(path)
    cfg = TradingConfig()
    # A base_config-ot tukrozzuk a fontos beallitasokra
    cfg.weights = dict(base_config.weights)
    cfg.buy_threshold = base_config.buy_threshold
    cfg.sell_threshold = base_config.sell_threshold
    cfg.stops = base_config.stops
    cfg.regime = base_config.regime
    cfg.mtf.enabled = False  # gyorsitasul; egyenkent be lehet kapcsolni

    agent = TradingAgent(cfg)
    result = Backtester(agent, cfg).run(df)
    max_dd = _max_drawdown(result.equity_curve)
    win_rate = (sum(1 for t in result.trades if t.pnl > 0)
                / max(1, len(result.trades)) * 100)
    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    return FileResult(
        path=path, symbol=sym, timeframe=tf, bars=len(df),
        return_pct=result.total_return_pct,
        max_dd_pct=max_dd, calmar=calmar(result.total_return_pct, max_dd),
        win_rate=win_rate, trades=len(result.trades),
        buy_hold_pct=bh,
    )


def aggregate(results: List[FileResult]) -> str:
    if not results:
        return "Nincs eredmeny."
    n = len(results)
    pos = sum(1 for r in results if r.return_pct > 0)
    returns = sorted([r.return_pct for r in results])
    dds = sorted([r.max_dd_pct for r in results])
    calmars = sorted([r.calmar for r in results])

    median = lambda xs: xs[len(xs) // 2]
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0

    return (
        f"\n=== AGGREGAL ({n} CSV) ===\n"
        f"  Pozitiv hozamu:   {pos}/{n} ({pos/n*100:.0f}%)\n"
        f"  Hozam:            atl={avg(returns):+.1f}%  med={median(returns):+.1f}%  "
        f"min={min(returns):+.1f}%  max={max(returns):+.1f}%\n"
        f"  Max drawdown:     atl={avg(dds):.1f}%  med={median(dds):.1f}%  "
        f"min={min(dds):.1f}%  max={max(dds):.1f}%\n"
        f"  Calmar:           atl={avg(calmars):.2f}  med={median(calmars):.2f}\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Multi-asset batch backteszt")
    parser.add_argument("--data-dir", default="data",
                        help="A CSV-k konyvtara")
    parser.add_argument("--pattern", default="*.csv",
                        help="Glob pattern (pl. 'BTC*.csv')")
    parser.add_argument("--sort", choices=["calmar", "return", "dd", "symbol"],
                        default="calmar")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.data_dir):
        print(f"Hiba: a {args.data_dir} mappa nem letezik. Hozd letre es futtass\n"
              f"   python fetch_history.py --symbols top20 --timeframes 1h,4h,1d",
              file=sys.stderr)
        return 1

    import glob
    paths = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not paths:
        print(f"Nincs CSV a {args.data_dir} mappaban (pattern: {args.pattern}).",
              file=sys.stderr)
        return 1

    print(f"Talaltam {len(paths)} CSV-t a {args.data_dir} mappaban\n")
    base_config = TradingConfig()

    results: List[FileResult] = []
    for path in paths:
        r = run_one(path, base_config)
        if r is not None:
            results.append(r)
            print(f"  {r.row()}")

    # Sorrendezes
    if args.sort == "calmar":
        results.sort(key=lambda r: r.calmar, reverse=True)
    elif args.sort == "return":
        results.sort(key=lambda r: r.return_pct, reverse=True)
    elif args.sort == "dd":
        results.sort(key=lambda r: r.max_dd_pct)
    elif args.sort == "symbol":
        results.sort(key=lambda r: (r.symbol, r.timeframe))

    print()
    print(f"=== TOP 10 a(z) '{args.sort}' szerint ===")
    print(f"{'symbol':<12} {'tf':>4} {'bars':>5}")
    print("-" * 100)
    for r in results[:10]:
        print(r.row())

    print(aggregate(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
