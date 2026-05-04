"""
Historikus OHLCV letolto Bybit / Binance-rol (CCXT-vel, publikus, kulcs nelkul).

A top 20 (vagy custom) USDT spot par tobb timeframe-en valo letoltese
CSV formatumban a `data/` mappaba. A backteszt es az `optimize`/`walkforward`
parancsok ezt fogyaszthatjak.

Pelda hasznalat:
    # Top 20 coin, 1h gyertya, utolso 1 ev (Bybit, alapertelmezett):
    python fetch_history.py --symbols top20 --timeframe 1h --years 1

    # Binance-rol BTC 2017 ota (~9 ev 1d), Bybit-en csak 2018 ota van:
    python fetch_history.py --exchange binance --symbols BTC/USDT --timeframe 1d --years 9

    # Csak BTC es ETH, 4h, utolso 2 ev, eu endpoint (csak Bybit-en):
    python fetch_history.py --symbols BTC/USDT,ETH/USDT \\
        --timeframe 4h --years 2 --endpoint eu

    # Tobb timeframe egyszerre Binance-rol:
    python fetch_history.py --exchange binance --symbols top5 \\
        --timeframes 1h,4h,1d --years 5

A publikus endpoint NEM ker API kulcsot, tehat csak `pip install ccxt`
szukseges. A rate limit-et a CCXT automatikusan kezeli.

Megjegyzes - exchange-valasztas:
  * bybit:   2018 ota van adat, EU-konform endpoint (bybit.eu) is van
  * binance: 2017 augusztusatol BTC/ETH, sok altcoin hamarabb listazva,
             de nehany regio (US/UK) blokkolt — VPN szukseges lehet.
             Az arak ~5-15 bps szpredet mutathatnak Bybithez kepest.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

from coins import parse_symbol_list
from config import BYBIT_HOSTS

logger = logging.getLogger("fetch")


# Timeframe -> milliszekundum (a cursor-ugraskent hasznaljuk)
TIMEFRAME_MS = {
    "1m": 60_000,        "3m": 180_000,       "5m": 300_000,
    "15m": 900_000,      "30m": 1_800_000,    "1h": 3_600_000,
    "2h": 7_200_000,     "4h": 14_400_000,    "6h": 21_600_000,
    "8h": 28_800_000,    "12h": 43_200_000,   "1d": 86_400_000,
    "1w": 604_800_000,   "1M": 30 * 86_400_000,
}


def make_exchange(endpoint: str, exchange: str = "bybit"):
    """CCXT publikus peldany Bybit-hez vagy Binance-hez (kulcs nelkul)."""
    try:
        import ccxt
    except ImportError as e:
        raise ImportError("Telepitsd: pip install ccxt") from e

    if exchange == "binance":
        ex = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if endpoint and endpoint not in ("global", "binance"):
            logger.warning("Binance forrasnal a --endpoint %s figyelmen kivul marad",
                           endpoint)
        return ex

    # Default: bybit
    ex = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    if endpoint == "testnet":
        ex.set_sandbox_mode(True)
    elif endpoint in ("eu", "global"):
        host = BYBIT_HOSTS[endpoint]
        # Az urls['api'] dict atirasa - minden kategoriara ugyanaz a host
        try:
            api_urls = ex.urls.get("api", {})
            if isinstance(api_urls, dict):
                for key in list(api_urls.keys()):
                    api_urls[key] = host
                ex.urls["api"] = api_urls
            else:
                ex.urls["api"] = host
        except Exception as e:
            logger.warning("Endpoint atiras meghiusult: %s (alapertelmezett marad)", e)
    return ex


def fetch_full_history(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    end_ms: Optional[int] = None,
    batch_limit: int = 1000,
) -> List[list]:
    """
    Tobb batch-on at lekeri a teljes idoszakot.
    Bybit max 1000 gyertya per request - ezert iteralni kell.
    """
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Ismeretlen timeframe: {timeframe}")
    interval = TIMEFRAME_MS[timeframe]
    end_ms = end_ms or int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows: List[list] = []
    cursor = since_ms
    _MAX_RETRIES = 4          # maximalis ujraprobalkozasok szama
    _RETRY_SLEEP = 5          # alap varakozas masodpercben (exponencialis)

    while cursor < end_ms:
        retries = 0
        while retries < _MAX_RETRIES:
            try:
                rows = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=batch_limit)
                break   # sikeres lekerdes: kilepes a retry-loopbol
            except Exception as e:
                retries += 1
                wait = _RETRY_SLEEP * retries   # 5s, 10s, 15s, 20s
                logger.warning(
                    "Hiba %s @ %s (kiserlet %d/%d): %s — varakozas %ds",
                    symbol,
                    datetime.fromtimestamp(cursor / 1000, tz=timezone.utc),
                    retries, _MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
        else:
            # Minden retry kiment — leallas ezzel a symbolnal
            logger.error(
                "Max ujraprobalkozas (%d) elerest: %s @ %s. Leallas.",
                _MAX_RETRIES, symbol,
                datetime.fromtimestamp(cursor / 1000, tz=timezone.utc),
            )
            break

        if not rows:
            # nem jott vissza adat ezen az idointervallumon
            break

        all_rows.extend(rows)
        last_ts = rows[-1][0]
        next_cursor = last_ts + interval

        # Ha nem haladtunk elore (pl. ismetelt utolso gyertya), megszakitas
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        # CCXT enableRateLimit kezeli, de plusz buffer
        time.sleep(max(0.05, exchange.rateLimit / 1000))

    return all_rows


def to_dataframe(rows: List[list]) -> pd.DataFrame:
    """Duplikatumok szurese, idorend szerinti rendezes."""
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Az `agent`/`backtest` load_csv() formatuma: timestamp + ohlcv oszlopok."""
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Fo
# --------------------------------------------------------------------------- #

def parse_timeframes(arg: str) -> List[str]:
    """Vesszovel elvalasztott timeframe-lista parsolasa."""
    return [tf.strip() for tf in arg.split(",") if tf.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Historikus OHLCV letolto Bybit / Binance-rol")
    parser.add_argument("--exchange", choices=["bybit", "binance"], default="bybit",
                        help="Adatforras (bybit alapertelmezett; binance-en hosszabb tortenet)")
    parser.add_argument("--symbols", default="top20",
                        help="Vesszos lista vagy 'top5' / 'top20'")
    parser.add_argument("--timeframe", default=None,
                        help="Egy timeframe (alternativ a --timeframes-nek)")
    parser.add_argument("--timeframes", default=None,
                        help="Vesszos timeframe lista, pl. '1h,4h,1d'")
    parser.add_argument("--years", type=float, default=1.0,
                        help="Hany evre vissza menjunk (default 1)")
    parser.add_argument("--endpoint", choices=["eu", "global", "testnet"],
                        default="global", help="Bybit endpoint (binance-nel ignoralt)")
    parser.add_argument("--out-dir", default="data",
                        help="Kimeneti CSV-k konyvtara")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Ne irja felul a meglevo CSV-ket")
    args = parser.parse_args(argv)

    # Timeframe-ek osszegyujtese
    if args.timeframes:
        timeframes = parse_timeframes(args.timeframes)
    elif args.timeframe:
        timeframes = [args.timeframe]
    else:
        timeframes = ["1h"]

    for tf in timeframes:
        if tf not in TIMEFRAME_MS:
            print(f"Ismeretlen timeframe: {tf}", file=sys.stderr)
            return 2

    symbols = parse_symbol_list(args.symbols)
    os.makedirs(args.out_dir, exist_ok=True)

    exchange = make_exchange(args.endpoint, exchange=args.exchange)
    since = datetime.now(timezone.utc) - timedelta(days=args.years * 365)
    since_ms = int(since.timestamp() * 1000)

    if args.exchange == "binance":
        print(f"Adatforras: Binance global (kulcs nelkul)")
    else:
        print(f"Adatforras: Bybit / endpoint={args.endpoint}")
    print(f"Symbolok ({len(symbols)}): {symbols}")
    print(f"Timeframe-ek: {timeframes}")
    print(f"Idoszak: {since.date()} -> ma ({args.years} ev)")
    print()

    # Binance-rol jovo CSV-ket toldalekoljuk, hogy keverhetoek legyenek a bybit-tel
    src_tag = "" if args.exchange == "bybit" else f"_{args.exchange}"

    total = 0
    failed: List[str] = []
    for sym in symbols:
        for tf in timeframes:
            fname = f"{sym.replace('/', '_')}_{tf}_{args.years}y{src_tag}.csv"
            path = os.path.join(args.out_dir, fname)

            if args.skip_existing and os.path.exists(path):
                print(f"  KIHAGYVA (van mar): {path}")
                continue

            print(f"  Letoltes: {sym:<12} {tf:<4} -> {path}")
            try:
                rows = fetch_full_history(exchange, sym, tf, since_ms)
                df = to_dataframe(rows)
                if df.empty:
                    print(f"    figyelmeztetes: ures eredmeny")
                    continue
                save_csv(df, path)
                first = pd.to_datetime(df["timestamp"].iloc[0], unit="ms", utc=True)
                last  = pd.to_datetime(df["timestamp"].iloc[-1], unit="ms", utc=True)
                print(f"    OK: {len(df)} gyertya, {first.date()} -> {last.date()}")
                total += len(df)
            except Exception as e:
                print(f"    HIBA: {e}")
                failed.append(f"{sym} ({tf})")

    print()
    print(f"Osszesen letoltve: {total} gyertya")
    if failed:
        print(f"Sikertelen ({len(failed)}): {failed}")
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    sys.exit(main())
