"""
Binance Vision bulk historikus letöltő — 1s (és más) timeframe-ek.

A Binance REST API nem nyújt 1s OHLCV-t (minimum 1m). A historikus
1s adatokat a Binance Vision bulk adatbázison keresztül lehet letölteni:
  https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/

Elérhető intervallumok (bulk): 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h,
  8h, 12h, 1d, 3d, 1w, 1M.

Megjegyzés az 1s adatról:
  * Elérhetőség: általában 2023. januártól van adat (symbolonként eltérhet).
  * Fájlméret: ~100-300 MB/hó tömörítetlen, zip-ben ~20-50 MB/hó.
  * 1 havi fájl ≈ 2,6 millió sor (86 400 bar/nap × 30 nap).
  * Letöltés sebessége: ~5-15 MB/s jó kapcsolaton; 1 év ≈ 10-15 perc.

CSV oszlopok (Binance Vision formátum):
  0  open_time            — Unix ms (nyitó időbélyeg)
  1  open
  2  high
  3  low
  4  close
  5  volume
  6  close_time           — Unix ms (záró időbélyeg)
  7  quote_asset_volume
  8  number_of_trades
  9  taker_buy_base_asset_volume
  10 taker_buy_quote_asset_volume
  11 ignore

Kimenet: ugyanaz a CSV formátum mint fetch_history.py-ból
  timestamp,open,high,low,close,volume

CLI példák:
    # BTC 1s adat 2024 januártól decemberig:
    python fetch_binance_bulk.py --symbol BTCUSDT --interval 1s \\
        --start 2024-01 --end 2024-12

    # ETH 1m adat, utolsó 3 hónap:
    python fetch_binance_bulk.py --symbol ETHUSDT --interval 1m --months 3

    # Több symbol:
    python fetch_binance_bulk.py --symbols BTCUSDT,ETHUSDT --interval 1s \\
        --start 2024-06 --end 2024-06 --out-dir data

    # Napi fájlok letöltése (pontosabb, de több HTTP kérés):
    python fetch_binance_bulk.py --symbol BTCUSDT --interval 1s \\
        --start 2024-01-01 --end 2024-01-31 --granularity daily
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
import zipfile
from datetime import date, timedelta
from typing import List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request

import pandas as pd

logger = logging.getLogger("binance_bulk")

_BASE_URL = "https://data.binance.vision/data/spot"

# Binance Vision által ismert intervallumok
KNOWN_INTERVALS = {
    "1s", "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}


# ============================================================================
# URL-építők
# ============================================================================

def _monthly_url(symbol: str, interval: str, year: int, month: int) -> str:
    fname = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    return f"{_BASE_URL}/monthly/klines/{symbol}/{interval}/{fname}"


def _daily_url(symbol: str, interval: str, d: date) -> str:
    fname = f"{symbol}-{interval}-{d.year}-{d.month:02d}-{d.day:02d}.zip"
    return f"{_BASE_URL}/daily/klines/{symbol}/{interval}/{fname}"


# ============================================================================
# HTTP letöltő
# ============================================================================

def _download_zip(url: str, retries: int = 4, backoff: float = 5.0) -> Optional[bytes]:
    """Letölti a zip fájlt és visszaadja a byte-okat, vagy None ha nem létezik."""
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "binance-bulk-downloader/1.0"})
            with urlopen(req, timeout=60) as resp:
                return resp.read()
        except HTTPError as e:
            if e.code == 404:
                return None   # fájl nem létezik (még nincs adat arra a hónapra)
            wait = backoff * attempt
            logger.warning("HTTP %d @ %s — %ds múlva újrapróbálás (%d/%d)",
                           e.code, url, wait, attempt, retries)
            time.sleep(wait)
        except URLError as e:
            wait = backoff * attempt
            logger.warning("URLError @ %s: %s — %ds múlva (%d/%d)",
                           url, e.reason, wait, attempt, retries)
            time.sleep(wait)
    logger.error("Max újrapróbálkozás elérve: %s", url)
    return None


# ============================================================================
# Zip → DataFrame
# ============================================================================

def _parse_zip_bytes(data: bytes) -> pd.DataFrame:
    """
    Binance Vision zip tartalmának parse-olása.

    A zip egyetlen CSV-t tartalmaz. Visszaadja az OHLCV DataFrame-et
    (timestamp, open, high, low, close, volume) milliszekundumos időbélyeggel.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if not names:
            return pd.DataFrame()
        csv_name = names[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f,
                header=None,
                usecols=[0, 1, 2, 3, 4, 5],
                names=["timestamp", "open", "high", "low", "close", "volume"],
                dtype={
                    "timestamp": "int64",
                    "open": "float64", "high": "float64",
                    "low": "float64",  "close": "float64",
                    "volume": "float64",
                },
            )
    return df


def _dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Duplikátumok eltávolítása és időrend szerinti rendezés."""
    if df.empty:
        return df
    return (df.drop_duplicates(subset=["timestamp"], keep="last")
              .sort_values("timestamp")
              .reset_index(drop=True))


# ============================================================================
# Fő letöltő logika
# ============================================================================

def _iter_months(start: date, end: date) -> List[Tuple[int, int]]:
    """(year, month) párok listája start..end (inkluzív)."""
    months = []
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        months.append((cur.year, cur.month))
        # következő hónap
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return months


def _iter_days(start: date, end: date) -> List[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def fetch_monthly(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    verbose: bool = True,
) -> pd.DataFrame:
    """Havi zip fájlok letöltése és összefűzése."""
    months = _iter_months(start, end)
    frames: List[pd.DataFrame] = []

    for year, month in months:
        url = _monthly_url(symbol, interval, year, month)
        if verbose:
            print(f"  [{symbol} {interval}] {year}-{month:02d} ... ", end="", flush=True)
        data = _download_zip(url)
        if data is None:
            if verbose:
                print("KIHAGYVA (nincs adat)")
            continue
        df = _parse_zip_bytes(data)
        if df.empty:
            if verbose:
                print("ÜRES")
            continue
        frames.append(df)
        if verbose:
            print(f"OK ({len(df):,} sor)")

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return _dedup_sort(pd.concat(frames, ignore_index=True))


def fetch_daily(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    verbose: bool = True,
) -> pd.DataFrame:
    """Napi zip fájlok letöltése és összefűzése (pontosabb, de több HTTP kérés)."""
    days = _iter_days(start, end)
    frames: List[pd.DataFrame] = []

    for d in days:
        url = _daily_url(symbol, interval, d)
        if verbose:
            print(f"  [{symbol} {interval}] {d} ... ", end="", flush=True)
        data = _download_zip(url)
        if data is None:
            if verbose:
                print("KIHAGYVA")
            continue
        df = _parse_zip_bytes(data)
        if df.empty:
            if verbose:
                print("ÜRES")
            continue
        frames.append(df)
        if verbose:
            print(f"OK ({len(df):,} sor)")

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return _dedup_sort(pd.concat(frames, ignore_index=True))


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Ment a fetch_history.py-kompatibilis CSV formátumba."""
    df.to_csv(path, index=False)


# ============================================================================
# CLI
# ============================================================================

def _parse_month(s: str) -> date:
    """'YYYY-MM' → date(YYYY, MM, 1)"""
    parts = s.strip().split("-")
    if len(parts) < 2:
        raise ValueError(f"Érvénytelen hónap formátum: '{s}' (várt: YYYY-MM)")
    return date(int(parts[0]), int(parts[1]), 1)


def _last_day_of_month(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    return date(d.year, d.month + 1, 1) - timedelta(days=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Binance Vision bulk historikus OHLCV letöltő (1s és egyéb TF)"
    )
    sym_grp = parser.add_mutually_exclusive_group(required=True)
    sym_grp.add_argument("--symbol",  help="Egy symbol (pl. BTCUSDT)")
    sym_grp.add_argument("--symbols", help="Vesszővel elválasztott lista (pl. BTCUSDT,ETHUSDT)")

    parser.add_argument("--interval", default="1s",
                        choices=sorted(KNOWN_INTERVALS),
                        help="Gyertya-intervallum (alapértelmezett: 1s)")

    time_grp = parser.add_mutually_exclusive_group()
    time_grp.add_argument("--start",  help="Kezdő hónap YYYY-MM formátumban")
    time_grp.add_argument("--months", type=int,
                          help="Az utolsó N teljes hónap letöltése")

    parser.add_argument("--end",      help="Záró hónap YYYY-MM (alapértelmezett: aktuális hónap)",
                        default=None)
    parser.add_argument("--granularity", choices=["monthly", "daily"], default="monthly",
                        help="Havi (kevesebb kérés) vagy napi fájlok (alapértelmezett: monthly)")
    parser.add_argument("--out-dir",  default="data",
                        help="Kimeneti könyvtár (alapértelmezett: data)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Ne írja felül a már létező CSV-ket")
    parser.add_argument("--quiet",    action="store_true", help="Csendes mód")

    args = parser.parse_args(argv)

    # Symbolok
    if args.symbol:
        symbols = [args.symbol.strip().upper()]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Időszak
    today = date.today()
    if args.end:
        end_month = _parse_month(args.end)
    else:
        end_month = date(today.year, today.month, 1)

    if args.months:
        # Utolsó N teljes hónap
        d = end_month
        for _ in range(args.months - 1):
            d = date(d.year, d.month, 1) - timedelta(days=1)
            d = date(d.year, d.month, 1)
        start_month = d
    elif args.start:
        start_month = _parse_month(args.start)
    else:
        start_month = date(today.year, today.month, 1)

    os.makedirs(args.out_dir, exist_ok=True)
    verbose = not args.quiet

    if verbose:
        print(f"Binance Vision bulk letöltő")
        print(f"  Interval:   {args.interval}")
        print(f"  Granularity: {args.granularity}")
        print(f"  Időszak:    {start_month.strftime('%Y-%m')} → {end_month.strftime('%Y-%m')}")
        print(f"  Symbolok:   {symbols}")
        print()

    failed: List[str] = []
    total_rows = 0

    for sym in symbols:
        out_name = f"{sym}_{args.interval}_{start_month.strftime('%Y%m')}_" \
                   f"{end_month.strftime('%Y%m')}_binance.csv"
        out_path = os.path.join(args.out_dir, out_name)

        if args.skip_existing and os.path.exists(out_path):
            if verbose:
                print(f"  KIHAGYVA (már létezik): {out_path}")
            continue

        if verbose:
            print(f"Letöltés: {sym}")

        try:
            if args.granularity == "daily":
                start_d = start_month
                end_d   = _last_day_of_month(end_month)
                df = fetch_daily(sym, args.interval, start_d, end_d, verbose=verbose)
            else:
                df = fetch_monthly(sym, args.interval, start_month, end_month, verbose=verbose)

            if df.empty:
                if verbose:
                    print(f"  FIGYELMEZTETÉS: üres eredmény ({sym})")
                failed.append(sym)
                continue

            save_csv(df, out_path)
            total_rows += len(df)
            if verbose:
                first_ts = pd.to_datetime(df["timestamp"].iloc[0],  unit="ms", utc=True)
                last_ts  = pd.to_datetime(df["timestamp"].iloc[-1], unit="ms", utc=True)
                print(f"  → {out_path}  [{len(df):,} sor | {first_ts.date()} – {last_ts.date()}]\n")

        except Exception as e:
            logger.exception("Hiba %s: %s", sym, e)
            failed.append(sym)

    if verbose:
        print(f"Összesen letöltve: {total_rows:,} sor")
        if failed:
            print(f"Sikertelen ({len(failed)}): {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    sys.exit(main())
