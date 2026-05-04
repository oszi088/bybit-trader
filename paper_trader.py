"""
Visszafele kompatibilis ujra-export.

A paper trader logikaja atkoltozott a trader.py-ba, hogy kozos vazon
osztozzon a Bybit eles modu traderrel. Itt csak alias marad
visszafele kompatibilitas miatt.
"""

from trader import build_paper_trader as _build_paper_trader  # noqa: F401
from trader import Trader as PaperTrader  # noqa: F401
