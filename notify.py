"""
Egyszeru ertesito modul: Telegram bot vagy log fallback.

Ha a TELEGRAM_BOT_TOKEN es TELEGRAM_CHAT_ID kornyezeti valtozok be vannak
allitva, akkor a Telegram bot api-jara kuldjuk az uzenetet. Egyebkent csak
loggolunk - igy az ugynok mindig fut, a telegram csak nice-to-have.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger("notify")


def _escape_md(text: str) -> str:
    """Markdown v1 speciális karakterek escape-elése dinamikus szövegben."""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


class Notifier:
    """Telegram alapu ertesito; ha nincs beallitva, csak loggol."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        self._available = bool(self.enabled and self.token and self.chat_id)
        if self.enabled and not self._available:
            logger.info(
                "Telegram nincs beallitva (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID hianyzik) - "
                "csak loggolunk."
            )

    def send(self, message: str, level: str = "INFO") -> None:
        """Egy uzenet kuldese. Soha ne dobjon kivetelt - inkabb csak logol."""
        # Mindig loggolunk, hogy backupkent meglegyen
        log_fn = getattr(logger, level.lower(), logger.info)
        log_fn(message)

        if not self._available:
            return

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            logger.warning("Telegram kuldes sikertelen: %s", e)

    # ------------------------------------------------------------------ #
    # Kenyelmi metodusok az ertesites tartalmara
    # ------------------------------------------------------------------ #

    def fill(self, side: str, symbol: str, size: float, price: float,
             pnl: Optional[float] = None) -> None:
        emoji = "BUY" if side == "BUY" else "SELL"
        msg = f"*{emoji}* `{symbol}` size={size:.6f} @ {price:.2f}"
        if pnl is not None:
            msg += f"\nPnL: `{pnl:+.2f}` USD"
        self.send(msg)

    def kill_switch(self, reason: str) -> None:
        self.send(f"KILL SWITCH AKTIV\n`{_escape_md(reason)}`", level="ERROR")

    def error(self, message: str) -> None:
        self.send(f"HIBA: {_escape_md(message)}", level="ERROR")

    def heartbeat(self, equity: float, status: str) -> None:
        self.send(f"Heartbeat: equity=${equity:.2f}\n{status}")
