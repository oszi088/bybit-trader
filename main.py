"""
CLI belepopont a kripto trader ugynokhoz (Bybit kiadas).

Peldak:
    python main.py backtest --csv synthetic_btc.csv
    python main.py walkforward --csv synthetic_btc.csv --fold-size 250
    python main.py optimize --csv synthetic_btc.csv
    python main.py paper --symbol BTC/USDT
    python main.py portfolio paper --symbols top20
    python main.py portfolio live --testnet --symbols top5
    python main.py live --testnet --symbol BTC/USDT
    python main.py live --live --execute --max-order 25
    python main.py decide --symbol ETH/USDT --timeframe 4h --endpoint eu
    python main.py trades --limit 50
    python main.py fg
"""

from __future__ import annotations

import argparse
import logging
import sys

from agent import TradingAgent
from backtest import Backtester, walk_forward
from coins import parse_symbol_list
from config import (
    TradingConfig, load_api_credentials,
    make_scalping_config, is_granular_timeframe, poll_interval_for_timeframe,
)
from data_source import CcxtDataSource, load_csv
from db import TradeDb
from fear_greed import FearGreedSource
from ml_model import MLConfig, MetaLabelModel
from ml_train import run_training
from optimizer import optimize, DEFAULT_GRID, EXTENDED_GRID
from portfolio import PortfolioTrader
from portfolio_backtest import PortfolioBacktester
from trader import build_bybit_trader, build_paper_trader


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_config(args) -> TradingConfig:
    """
    Egyseges config epites:
      * --scalping flag eseten a make_scalping_config() faktoryt hasznaljuk
        (sub-5min indikator periodusok, MTF cascade, szukebb stopok, gyors poll)
      * --timeframe 1m/3m/5m automatikusan scalping-szeru hangolasra valt,
        ha a --scalping flag nincs explicit megadva
      * Egyebkent a default TradingConfig-ot adjuk vissza
    """
    tf = getattr(args, "timeframe", None)
    sym = getattr(args, "symbol", None) or "BTC/USDT"
    endp = getattr(args, "endpoint", None)

    use_scalping = getattr(args, "scalping", False) or (
        tf is not None and is_granular_timeframe(tf)
    )

    if use_scalping:
        cfg = make_scalping_config(
            timeframe=(tf or "1m"),
            symbol=sym,
            bybit_endpoint=(endp or "testnet"),
        )
        print(f"[SCALPING MOD] timeframe={cfg.timeframe} | "
              f"poll={cfg.poll_interval_sec}s | "
              f"MTF={cfg.mtf.timeframes} | "
              f"ATR-stop={cfg.stops.atr_stop_mult}x / TP={cfg.stops.atr_tp_mult}x | "
              f"threshold=±{cfg.buy_threshold}")
        return cfg

    cfg = TradingConfig()
    if sym:  cfg.symbol = sym
    if tf:   cfg.timeframe = tf
    if endp: cfg.bybit_endpoint = endp
    return cfg


# ============================================================================
# Subcommandok
# ============================================================================

def cmd_backtest(args: argparse.Namespace) -> int:
    config = _build_config(args)
    if args.slippage_bps is not None:
        config.backtest.slippage_bps = float(args.slippage_bps)

    print(f"Adatok betoltese: {args.csv}")
    df = load_csv(args.csv)
    print(f"  {len(df)} gyertya, {df.index.min()} -> {df.index.max()}")

    agent = TradingAgent(config)
    result = Backtester(agent, config).run(df)
    print("\n=== BACKTEST EREDMENY ===")
    print(result.summary())

    if args.show_trades:
        print("\n--- Trade lista ---")
        for t in result.trades:
            print(
                f"  {t.entry_time}  BUY @ {t.entry_price:>9.2f}  -> "
                f"{t.exit_time}  SELL @ {t.exit_price:>9.2f}  "
                f"({t.reason}, pnl={t.pnl:+.2f})"
            )
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            result.equity_curve.plot(title="Equity gorbe")
            plt.tight_layout(); plt.show()
        except ImportError:
            print("(matplotlib nincs)")
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    config = _build_config(args)

    df = load_csv(args.csv)
    print(f"Adatok: {len(df)} gyertya")
    print(f"Walk-forward: fold_size={args.fold_size}, step={args.step or args.fold_size}")
    agent = TradingAgent(config)
    result = walk_forward(agent, df, fold_size=args.fold_size, step=args.step)
    print("\n=== WALK-FORWARD EREDMENY ===")
    print(result.summary())
    for i, fold in enumerate(result.folds, 1):
        s = fold.equity_curve.index[0].date()
        e = fold.equity_curve.index[-1].date()
        print(f"  Fold {i:2d} | {s} -> {e} | {fold.summary()}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Walk-forward grid search overfit-szuressel."""
    config = _build_config(args)
    df = load_csv(args.csv)
    print(f"Adatok: {len(df)} gyertya, {df.index.min()} -> {df.index.max()}")

    grid = EXTENDED_GRID if args.extended else DEFAULT_GRID
    grid_name = "EXTENDED" if args.extended else "DEFAULT"
    print(f"Grid: {grid_name} | cel: {args.objective} | max_combos: {args.max_combinations}")
    print("Optimalizalas inditasa - ez perceket vehet...\n")

    result = optimize(
        config,
        df,
        grid=grid,
        max_combinations=args.max_combinations,
        objective=args.objective,
    )
    result.print_report(top_n=args.top)

    best = result.best()
    if best is None:
        return 1

    print("\nLegjobb parameter-keszlet alkalmazashoz:")
    for k, v in best.params.items():
        print(f"  {k} = {v}")
    return 0


def cmd_portbt(args: argparse.Namespace) -> int:
    """Portfolió-backteszt: megosztott tőke, párhuzamos multi-coin."""
    import os, re
    config = _build_config(args)

    # CSV-ek betöltése a megadott mappából
    data_dir = args.data_dir
    pattern = re.compile(r"([A-Z]+)_USDT.*\.csv$", re.IGNORECASE)
    datasets = {}
    for fname in sorted(os.listdir(data_dir)):
        m = pattern.match(fname)
        if m:
            sym = m.group(1).upper()
            if args.symbols and sym not in [s.upper() for s in args.symbols.split(",")]:
                continue
            path = os.path.join(data_dir, fname)
            datasets[sym] = load_csv(path)

    if not datasets:
        print(f"Nem talalhato CSV a mappaban: {data_dir}")
        return 1

    print(f"Betoltott coinok: {', '.join(sorted(datasets))} "
          f"({sum(len(d) for d in datasets.values())} osszbar)")
    print(f"Kezdotoke: ${args.initial_balance:,.0f} | "
          f"Max pozicio: {args.max_positions} | "
          f"Slot: ${args.initial_balance / args.max_positions:,.0f}")
    print()

    bt = PortfolioBacktester(
        base_config=config,
        initial_balance=args.initial_balance,
        max_positions=args.max_positions,
    )
    result = bt.run(datasets)

    print("=== PORTFOLIO BACKTEST EREDMENY ===")
    print(result.summary())

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            result.equity_curve.plot(title="Portfolio equity gorbe")
            plt.tight_layout(); plt.show()
        except ImportError:
            print("(matplotlib nincs telepitve)")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    config = _build_config(args)
    _apply_risk_overrides(config, args)
    build_paper_trader(config).run_forever()
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    if not args.testnet and not args.live:
        print("Hiba: kotelezoen --testnet vagy --live.", file=sys.stderr); return 2
    if args.testnet and args.live:
        print("Hiba: --testnet es --live egyszerre nem.", file=sys.stderr); return 2

    # Endpointot most kezeljuk: a _build_config az args.endpoint-ot olvassa
    if not args.endpoint:
        args.endpoint = "testnet" if args.testnet else "eu"
    config = _build_config(args)
    _apply_risk_overrides(config, args)

    if config.is_live and not args.execute:
        config.risk.dry_run = True

    api_key, api_secret = load_api_credentials()
    if not api_key or not api_secret:
        print("Hiba: BYBIT_API_KEY es BYBIT_API_SECRET nincs beallitva.", file=sys.stderr)
        return 3

    if config.is_live and not config.risk.dry_run:
        print(f"\nFIGYELEM: Bybit {config.bybit_endpoint.upper()} VALODI penz mod\n")
        if not args.yes_i_know:
            if input("Add meg a 'YES' szot: ").strip() != "YES":
                return 4
    build_bybit_trader(config).run_forever()
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Top 20 (vagy custom) coin parhuzamos kereskedese."""
    config = _build_config(args)
    _apply_risk_overrides(config, args)

    symbols = parse_symbol_list(args.symbols)
    print(f"Portfolio: {len(symbols)} symbol -> {symbols}")

    if args.subcommand == "live":
        if not args.testnet and not args.live:
            print("portfolio live: --testnet vagy --live kell.", file=sys.stderr); return 2
        config.bybit_endpoint = "testnet" if args.testnet else "eu"
        if args.endpoint:  config.bybit_endpoint = args.endpoint
        if config.is_live and not args.execute:
            config.risk.dry_run = True
        api_key, api_secret = load_api_credentials()
        if not api_key or not api_secret:
            print("Hiba: BYBIT_API_KEY/SECRET nincs.", file=sys.stderr); return 3
        if config.is_live and not config.risk.dry_run and not args.yes_i_know:
            print(f"\nVALODI penz a Bybit {config.bybit_endpoint.upper()} endpointon "
                  f"{len(symbols)} symbolon.")
            if input("Add meg a 'YES' szot: ").strip() != "YES":
                return 4
        pt = PortfolioTrader(symbols, config, live_broker=True,
                             max_open_positions=args.max_positions)
    else:
        pt = PortfolioTrader(symbols, config, live_broker=False,
                             max_open_positions=args.max_positions)

    pt.run_forever()
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    config = _build_config(args)

    source = CcxtDataSource(
        config.exchange_id, config.symbol, config.timeframe,
        endpoint=config.bybit_endpoint, market_type=config.market_type,
    )
    print(f"OHLCV: {config.symbol} ({config.timeframe}) Bybit/{config.bybit_endpoint}")
    ohlcv = source.fetch_ohlcv(limit=200)

    agent = TradingAgent(config)
    decision = agent.decide(ohlcv)
    print(f"\n=== DONTES @ {ohlcv.index[-1]} ===")
    print(decision.explain())
    print(f"\nFear & Greed: {decision.fear_greed} ({_fg_label(decision.fear_greed)})")
    print("\nIndikator szavazatok:")
    for name, sig in decision.reasons.items():
        glyph = "+" if sig > 0 else ("-" if sig < 0 else ".")
        print(f"  {glyph}  {name:<12}  signal={sig:+d}")
    return 0


def cmd_trades(args: argparse.Namespace) -> int:
    config = TradingConfig()
    db = TradeDb(config.db.db_path, enabled=True)
    rows = db.list_trades(limit=args.limit)
    if not rows:
        print("Nincs trade az SQLite logban.")
        return 0
    print(f"Utolso {len(rows)} trade:")
    for r in rows:
        print(f"  {r['timestamp']}  {r['side']:<4} {r['size']:.6f} {r['symbol']} "
              f"@ {r['price']:.2f}  pnl={r['pnl']:+.2f}  ({r['note']})")
    return 0


def cmd_fg(args: argparse.Namespace) -> int:
    """A jelenlegi Fear & Greed Index lekerese."""
    fg = FearGreedSource()
    r = fg.get()
    print(f"Fear & Greed Index: {r.value} ({r.classification})")
    print(f"  Idobelyeg: {r.timestamp}")
    return 0


def cmd_ml_train(args: argparse.Namespace) -> int:
    """Meta-label XGBoost modell tanítása CSV historikus adaton."""
    config = _build_config(args)
    ml_cfg = MLConfig(
        model_path       = args.model_out,
        n_folds          = args.folds,
        embargo_pct      = args.embargo,
        min_train_size   = args.min_train,
        n_estimators     = args.n_estimators,
        max_depth        = args.max_depth,
        learning_rate    = args.lr,
    )
    print(f"Tanítás: {args.csv}  →  {args.model_out}")
    print(f"  Folds={args.folds}  embargo={args.embargo:.1%}  "
          f"max_holding={args.max_holding} gyertya")
    run_training(
        csv_path    = args.csv,
        config      = config,
        ml_config   = ml_cfg,
        model_out   = args.model_out,
        max_holding = args.max_holding,
        top_n_features = 20,
    )
    return 0


# ============================================================================
# Kozos segedek
# ============================================================================

def _fg_label(value: int) -> str:
    if value <= 24: return "Extreme Fear"
    if value <= 44: return "Fear"
    if value <= 55: return "Neutral"
    if value <= 74: return "Greed"
    return "Extreme Greed"


def _apply_risk_overrides(config, args) -> None:
    if getattr(args, "max_order", None) is not None:
        config.risk.max_order_value_usd = float(args.max_order)
    if getattr(args, "daily_loss", None) is not None:
        config.risk.daily_loss_limit_usd = float(args.daily_loss)
    if getattr(args, "max_drawdown", None) is not None:
        config.risk.max_drawdown_pct = float(args.max_drawdown)
    if getattr(args, "dry_run", False):
        config.risk.dry_run = True
    if getattr(args, "no_telegram", False):
        config.notify.enabled = False


def _add_runtime_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None,
                   help="Timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w, 1M)")
    p.add_argument("--endpoint", choices=["eu", "global", "testnet"], default=None)
    p.add_argument("--scalping", action="store_true",
                   help="Sub-5min hangolas: gyorsabb periodusok, alacsonyabb MTF, "
                        "szukebb stopok, ±0.55 threshold")
    p.add_argument("--max-order", type=float, default=None)
    p.add_argument("--daily-loss", type=float, default=None)
    p.add_argument("--max-drawdown", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-telegram", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kripto trader AI ugynok (Bybit)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backtest")
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None,
                   help="Csak meta: a config-hoz; a CSV adat onmagaban dont")
    p.add_argument("--endpoint", choices=["eu", "global", "testnet"], default=None)
    p.add_argument("--scalping", action="store_true",
                   help="Scalping (sub-5min) hangolas")
    p.add_argument("--show-trades", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--slippage-bps", type=float, default=None)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("walkforward")
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--endpoint", choices=["eu", "global", "testnet"], default=None)
    p.add_argument("--scalping", action="store_true")
    p.add_argument("--fold-size", type=int, default=500)
    p.add_argument("--step", type=int, default=None)
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("optimize", help="Walk-forward grid search overfit-szuressel")
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--endpoint", choices=["eu", "global", "testnet"], default=None)
    p.add_argument("--scalping", action="store_true")
    p.add_argument("--top", type=int, default=5,
                   help="Hany legjobb jeloltet mutasson (default: 5)")
    p.add_argument("--objective", choices=["return", "calmar"], default="return",
                   help="Optimalizalasi cel: 'return' = max OOS hozam, "
                        "'calmar' = max OOS Calmar (hozam/drawdown, stabil)")
    p.add_argument("--extended", action="store_true",
                   help="Kibovitett grid: + ATR stop/TP + poziciomeret "
                        "(tobb kombinacio, lassabb)")
    p.add_argument("--max-combinations", type=int, default=300,
                   help="Max probalt parameter-kombinacio (default: 300)")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("portbt",
                       help="Portfolio-backteszt: megosztott toke, parhuzamos multi-coin")
    p.add_argument("--data-dir", default="data",
                   help="Mappa a CSV fajlokkal (default: data/)")
    p.add_argument("--symbols", default=None,
                   help="Vesszos coin lista szuroshoz, pl. BTC,ETH,SOL (default: mind)")
    p.add_argument("--initial-balance", type=float, default=10_000.0,
                   help="Kezdotoke USD-ban (default: 10000)")
    p.add_argument("--max-positions", type=int, default=3,
                   help="Max egyideju nyitott pozicio (default: 3)")
    p.add_argument("--plot", action="store_true",
                   help="Equity curve rajzolasa (matplotlib)")
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--endpoint", choices=["eu", "global", "testnet"], default=None)
    p.add_argument("--scalping", action="store_true")
    p.set_defaults(func=cmd_portbt)

    p = sub.add_parser("paper")
    _add_runtime_flags(p)
    p.set_defaults(func=cmd_paper)

    p = sub.add_parser("live")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--testnet", action="store_true")
    grp.add_argument("--live", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--yes-i-know", action="store_true")
    _add_runtime_flags(p)
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("portfolio", help="Multi-symbol kereskedes (top 20 default)")
    p.add_argument("subcommand", choices=["paper", "live"])
    p.add_argument("--symbols", default="top20",
                   help="vesszos lista vagy 'top5' / 'top20'")
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--testnet", action="store_true")
    p.add_argument("--live", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--yes-i-know", action="store_true")
    _add_runtime_flags(p)
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("decide")
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--endpoint", choices=["eu", "global", "testnet"], default=None)
    p.add_argument("--scalping", action="store_true",
                   help="Scalping (sub-5min) hangolas")
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("trades")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_trades)

    p = sub.add_parser("fg", help="Aktualis Fear & Greed Index")
    p.set_defaults(func=cmd_fg)

    p = sub.add_parser("ml-train", help="Meta-label XGBoost modell tanítása")
    p.add_argument("--csv",          required=True,  help="Historikus OHLCV CSV")
    p.add_argument("--model-out",    default="ml_model.pkl")
    p.add_argument("--symbol",       default=None)
    p.add_argument("--timeframe",    default=None)
    p.add_argument("--endpoint",     choices=["eu", "global", "testnet"], default=None)
    p.add_argument("--scalping",     action="store_true")
    p.add_argument("--folds",        type=int,   default=5)
    p.add_argument("--embargo",      type=float, default=0.01,
                   help="Embargo arány (0..1) a fold határán")
    p.add_argument("--max-holding",  type=int,   default=20,
                   help="Max gyertyák száma a triple barrier függőleges korlátjához")
    p.add_argument("--min-train",    type=int,   default=300)
    p.add_argument("--n-estimators", type=int,   default=300)
    p.add_argument("--max-depth",    type=int,   default=4)
    p.add_argument("--lr",           type=float, default=0.03)
    p.set_defaults(func=cmd_ml_train)

    return parser


def main(argv=None) -> int:
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
