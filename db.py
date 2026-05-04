"""
SQLite-alapu trade log es allapot persistencia.

Ha az ugynok osszeomlik vagy ujraindul, innen tudja, hogy mi az utolso
nyitott pozicio, es a trade history elerheto kesobbi elemzeshez (adozas,
teljesitmeny-merres).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Iterator, List, Optional

logger = logging.getLogger("db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    size          REAL NOT NULL,
    price         REAL NOT NULL,
    fee           REAL NOT NULL,
    pnl           REAL NOT NULL DEFAULT 0,
    note          TEXT
);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);
"""


class TradeDb:
    """Vekony SQLite wrapper trade log-hoz es allapot persistenciahoz."""

    def __init__(self, path: str = "trader_state.db", enabled: bool = True):
        self.enabled = enabled
        self.path = path
        if not enabled:
            return
        try:
            with self._connect() as conn:
                conn.executescript(SCHEMA)
        except sqlite3.Error as e:
            # Pl. read-only fajlrendszer vagy I/O hiba: letiltjuk magunkat,
            # de az ugynok mukodjon tovabb hibatlanul.
            logger.warning("SQLite init sikertelen (%s) - DB log letiltva.", e)
            self.enabled = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Trade log
    # ------------------------------------------------------------------ #

    def log_fill(self, symbol: str, side: str, size: float, price: float,
                 fee: float, pnl: float = 0.0, note: str = "",
                 timestamp: Optional[datetime] = None) -> None:
        if not self.enabled:
            return
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, side, size, price, fee, pnl, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, symbol, side, size, price, fee, pnl, note),
            )

    def list_trades(self, limit: int = 100) -> List[dict]:
        if not self.enabled:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Kulcs-ertek allapot tarolas (pl. nyitott pozicio, peak equity)
    # ------------------------------------------------------------------ #

    def set_state(self, key: str, value) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def get_state(self, key: str, default=None):
        if not self.enabled:
            return default
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
            if row is None:
                return default
            try:
                return json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                return default
