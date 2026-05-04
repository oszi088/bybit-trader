"""
Bybit-on elérhető USDT spot párok, marketcap tier szerint csoportosítva.

Tier definíciók:
  LARGE (~top 15):  legjobb likviditás, legalacsonyabb spread, legkevésbé
                    manipulálható — ez az alap kereskedési univerzum
  MID   (~rank 15–60): jó mozgás alt szezonban, de kisebb bid/ask mélység
  SMALL (~rank 60–200): magas hozam potenciál, magas kockázat, kis likviditás

Altseason szabály (altcoin_filter.py részletezi):
  - Csak halving-túlélő coinok vásárolhatók alt szezonban
  - Tier fokozatos engedélyezés: LARGE → MID → SMALL (az altseason korával)
"""

from __future__ import annotations
from typing import List


# =============================================================================
# LARGE CAP — top ~15, nagyon likvid, alacsony spread
# =============================================================================
LARGE_CAP: List[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "TRX/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "TON/USDT",
    "NEAR/USDT",
    "MATIC/USDT",   # POL-ra átnevezve, de még elérhető
    "UNI/USDT",
]

# =============================================================================
# MID CAP — rank ~15–60, jó altseason mozgás
# =============================================================================
MID_CAP: List[str] = [
    "ATOM/USDT",
    "LTC/USDT",
    "APT/USDT",
    "ARB/USDT",
    "OP/USDT",
    "INJ/USDT",
    "STX/USDT",
    "HBAR/USDT",
    "FIL/USDT",
    "IMX/USDT",
    "THETA/USDT",
    "VET/USDT",
    "ALGO/USDT",
    "EGLD/USDT",
    "XTZ/USDT",
    "AAVE/USDT",
    "SAND/USDT",
    "MANA/USDT",
    "AXS/USDT",
    "MKR/USDT",
    "GRT/USDT",
    "FLOW/USDT",
    "LDO/USDT",
    "SNX/USDT",
    "CRV/USDT",
    "KAVA/USDT",
    "ENJ/USDT",
]

# =============================================================================
# SMALL CAP — rank ~60–200, magas volatilitás, kisebb likviditás
# =============================================================================
SMALL_CAP: List[str] = [
    "RUNE/USDT",
    "ZIL/USDT",
    "ONE/USDT",
    "S/USDT",       # Sonic (korábbi FTM/Fantom → token migration 2024)
    "ROSE/USDT",
    "BAT/USDT",
    "LRC/USDT",
    "1INCH/USDT",
    "COMP/USDT",
    "YFI/USDT",
    "ONT/USDT",
    "ETC/USDT",
    "XMR/USDT",
    "DASH/USDT",
    "NEO/USDT",
    "CFX/USDT",
]

# =============================================================================
# Összesített listák
# =============================================================================

ALL_SYMBOLS: List[str] = LARGE_CAP + MID_CAP + SMALL_CAP

# Csak altcoinok (BTC kizárva)
ALTCOINS_LARGE: List[str] = [s for s in LARGE_CAP if s != "BTC/USDT"]
ALTCOINS_MID:   List[str] = list(MID_CAP)
ALTCOINS_SMALL: List[str] = list(SMALL_CAP)

# Hagyományos preset-ek (visszafelé kompatibilis)
TOP_5_USDT: List[str] = LARGE_CAP[:5]
TOP_20_USDT: List[str] = LARGE_CAP + MID_CAP[:5]


# =============================================================================
# Segédfüggvények
# =============================================================================

def get_tier(symbol: str) -> str:
    """Visszaadja a coin tier-jét ('large'/'mid'/'small'/'unknown')."""
    if symbol in LARGE_CAP:  return "large"
    if symbol in MID_CAP:    return "mid"
    if symbol in SMALL_CAP:  return "small"
    return "unknown"


def symbols_by_tier(
    include_large: bool = True,
    include_mid: bool = False,
    include_small: bool = False,
    exclude_btc: bool = False,
) -> List[str]:
    """Tier-alapú szimbólum szűrő."""
    result: List[str] = []
    if include_large:
        result.extend(LARGE_CAP)
    if include_mid:
        result.extend(MID_CAP)
    if include_small:
        result.extend(SMALL_CAP)
    if exclude_btc and "BTC/USDT" in result:
        result.remove("BTC/USDT")
    return result


def parse_symbol_list(arg: str | None, default: List[str] | None = None) -> List[str]:
    """
    CLI paraméterek feldolgozása: vesszővel elválasztott symbol lista,
    vagy 'top5' / 'top20' / 'large' / 'mid' / 'small' / 'all' preset,
    vagy None → default.
    """
    if not arg:
        return default if default is not None else list(TOP_20_USDT)
    arg = arg.strip().lower()
    presets = {
        "top5":  TOP_5_USDT,
        "top20": TOP_20_USDT,
        "large": LARGE_CAP,
        "mid":   MID_CAP,
        "small": SMALL_CAP,
        "all":   ALL_SYMBOLS,
    }
    if arg in presets:
        return list(presets[arg])
    return [s.strip().upper() for s in arg.split(",") if s.strip()]
