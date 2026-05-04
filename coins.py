"""
Top 20 USDT spot par a Bybiten (kapitalizacio + likviditas alapjan).

Nem garantaltan a jelenleg pontos sorrend, de valamennyi par
ellenorzotten elerheto a Bybit spot piacon. A lista konfigbol
felulirhato, ha mas coineket szeretnel kereskedni.

Megjegyzes: a meme coinok (DOGE, SHIB, PEPE) jellemzoen volatilisabbak,
es a F&G index, valamint a volatilitas-szuro inkabb felulik rajuk.
"""

from __future__ import annotations

from typing import List


# Hagyomanyos top 20 USDT spot par (likviditas + market cap alapjan).
# Kornyezeti valtozoval / config-bol felulirhato.
TOP_20_USDT: List[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "TRX/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "MATIC/USDT",
    "TON/USDT",
    "SHIB/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "NEAR/USDT",
    "UNI/USDT",
    "XLM/USDT",
    "ATOM/USDT",
]


# Konzervativabb kis-kosar: csak a legfolyekonyabb 5 par
TOP_5_USDT: List[str] = TOP_20_USDT[:5]


def parse_symbol_list(arg: str | None, default: List[str] | None = None) -> List[str]:
    """
    CLI parameterek feldolgozasara: vesszovel elvalasztott symbol lista,
    vagy 'top5' / 'top20' presetek, vagy None -> default.
    """
    if not arg:
        return default if default is not None else list(TOP_20_USDT)
    arg = arg.strip().lower()
    if arg == "top5":
        return list(TOP_5_USDT)
    if arg == "top20":
        return list(TOP_20_USDT)
    # vesszos lista
    return [s.strip().upper() for s in arg.split(",") if s.strip()]
