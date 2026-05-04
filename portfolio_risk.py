"""
portfolio_risk.py — Portfolio-szintű kockázatkezelés spot tradinghez

Funkciók:
  1. Korreláció-mátrix: nyitott pozíciók közötti összefüggések
  2. Portfolio VaR (Historical Simulation, 95% CI, 1 napos horizont)
  3. Max korrelált kitettség: ha új belépés >X% a portfólióban, és
     erősen korrelál egy meglévő pozícióval → blokkol
  4. Coin-cluster alapú limitek:
       BTC-cluster  (BTC, ETH, SOL, BNB, AVAX, ...)
       Alt-cluster  (LINK, DOT, ADA, UNI, MATIC, ...)
       DeFi-cluster (AAVE, MKR, CRV, SNX, LDO, ...)
       GameFi       (AXS, SAND, MANA, ENJ, IMX, ...)

Spot-only megjegyzések:
  - Nincs short, tehát csak long-long korreláció számít
  - A VaR a spot portfólió piaci értékére vonatkozik
  - Egy cluster max kitettség: max_cluster_exposure_pct (pl. 40%)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("portfolio_risk")

# ---------------------------------------------------------------------------
# Coin klaszterek
# ---------------------------------------------------------------------------

COIN_CLUSTERS: Dict[str, List[str]] = {
    "btc_cluster": [
        "BTC", "ETH", "SOL", "BNB", "AVAX", "MATIC", "DOT", "NEAR", "TON",
    ],
    "alt_cluster": [
        "XRP", "ADA", "TRX", "DOGE", "LINK", "UNI", "ATOM", "LTC", "HBAR",
        "VET", "ALGO",
    ],
    "defi_cluster": [
        "AAVE", "MKR", "CRV", "SNX", "LDO", "GRT", "COMP", "YFI", "1INCH",
        "KAVA",
    ],
    "gamefi_cluster": [
        "AXS", "SAND", "MANA", "ENJ", "IMX", "FLOW",
    ],
    "infra_cluster": [
        "FIL", "THETA", "INJ", "STX", "APT", "ARB", "OP", "EGLD",
    ],
}


# ---------------------------------------------------------------------------
# Konfiguráció és eredmény dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PortfolioRiskConfig:
    """Portfolio-szintű kockázatkezelési paraméterek."""
    # Max 8% napi portfólió VaR (95% CI, Historical Simulation)
    max_var_95_pct: float = 0.08
    # Max korreláció az új és egy meglévő pozíció hozamai között
    max_position_correlation: float = 0.80
    # Max egyetlen klaszterbe rakott tőke aránya
    max_cluster_exposure_pct: float = 0.40
    # Legalább ennyi historikus return kell a VaR számításhoz
    min_history_bars: int = 30
    # VaR konfidencia-szint
    var_confidence: float = 0.95


@dataclass
class PortfolioRiskResult:
    """check_new_position() eredménye."""
    can_open: bool
    reason: str
    var_95_pct: float
    # Új pozíció maximális korrelációja a nyitott pozíciókkal
    max_corr_with_existing: float
    # Melyik szimbólummal korrelál legjobban
    correlated_symbol: Optional[str]
    # Klaszter neve → portfólió arány
    cluster_exposure: Dict[str, float]
    note: str


# ---------------------------------------------------------------------------
# Fő menedzser osztály
# ---------------------------------------------------------------------------

class PortfolioRiskManager:
    """
    Portfolio-szintű kockázatkezelő spot tradinghez.

    Ellenőrzések check_new_position() hívásakor:
      1. VaR-limit: a bővített portfólió VaR-ja nem haladhatja meg a konfigban
         beállított max_var_95_pct értéket.
      2. Korreláció-limit: ha az új coin hozamai erősen korrelálnak egy
         meglévő pozícióéval, a belépést blokkolja.
      3. Cluster-limit: egy klaszterbe nem kerülhet a portfólió
         max_cluster_exposure_pct-nél nagyobb hányada.
    """

    def __init__(self, config: PortfolioRiskConfig = None):
        self.config = config or PortfolioRiskConfig()

    # ------------------------------------------------------------------
    # Publikus API
    # ------------------------------------------------------------------

    def check_new_position(
        self,
        symbol: str,
        open_symbols: list[str],
        returns_dict: dict[str, list[float]],
        total_portfolio_usd: float,
        new_position_usd: float,
    ) -> PortfolioRiskResult:
        """
        Ellenőrzi, hogy nyitható-e az új pozíció a portfólió szempontjából.

        Paraméterek
        -----------
        symbol              : az új coin szimbóluma (pl. "BTC/USDT" vagy "BTC")
        open_symbols        : a már nyitott pozíciók szimbólumai
        returns_dict        : szimbólum → napi hozamok listája (pl. 0.02 = +2%)
                              Mind az új, mind a meglévő szimbólumokhoz tartalmaz
                              adatot (ha rendelkezésre áll).
        total_portfolio_usd : jelenlegi portfólió értéke USD-ben (bővítés előtt)
        new_position_usd    : az új pozíció mérete USD-ben

        Visszatér
        ---------
        PortfolioRiskResult
        """
        cfg = self.config
        new_sym = self._strip_suffix(symbol)

        # 1. Cluster kitettség (az új pozícióval együtt)
        open_values = [1.0] * len(open_symbols)          # nominális egységek
        extended_symbols = open_symbols + [symbol]
        total_extended = total_portfolio_usd + new_position_usd
        # Arányos értékek: meglévők egyenlő súllyal, az új a tényleges méretével
        if total_portfolio_usd > 0 and len(open_symbols) > 0:
            existing_value_each = total_portfolio_usd / len(open_symbols)
            open_values_usd = [existing_value_each] * len(open_symbols)
        else:
            open_values_usd = []

        all_values = open_values_usd + [new_position_usd]
        cluster_exp = self.cluster_exposure(extended_symbols, all_values, total_extended)

        # Klaszter-limit ellenőrzés
        for cluster_name, exp_frac in cluster_exp.items():
            if exp_frac > cfg.max_cluster_exposure_pct:
                return PortfolioRiskResult(
                    can_open=False,
                    reason=(
                        f"cluster limit exceeded: {cluster_name} "
                        f"= {exp_frac:.1%} > {cfg.max_cluster_exposure_pct:.1%}"
                    ),
                    var_95_pct=0.0,
                    max_corr_with_existing=0.0,
                    correlated_symbol=None,
                    cluster_exposure=cluster_exp,
                    note="Csökkentsd a klaszterben lévő pozíciók számát/méretét.",
                )

        # 2. Korreláció ellenőrzés
        new_returns = returns_dict.get(symbol) or returns_dict.get(new_sym, [])
        max_corr = 0.0
        correlated_sym: Optional[str] = None

        for osym in open_symbols:
            oret = returns_dict.get(osym) or returns_dict.get(self._strip_suffix(osym), [])
            if len(new_returns) < cfg.min_history_bars or len(oret) < cfg.min_history_bars:
                logger.debug(
                    "Nem elég adat a korreláció számításhoz: %s (%d) vs %s (%d)",
                    symbol, len(new_returns), osym, len(oret),
                )
                continue
            c = self.compute_correlation(new_returns, oret)
            if abs(c) > abs(max_corr):
                max_corr = c
                correlated_sym = osym

        if max_corr > cfg.max_position_correlation:
            return PortfolioRiskResult(
                can_open=False,
                reason=(
                    f"high correlation with {correlated_sym}: "
                    f"{max_corr:.2f} > {cfg.max_position_correlation:.2f}"
                ),
                var_95_pct=0.0,
                max_corr_with_existing=max_corr,
                correlated_symbol=correlated_sym,
                cluster_exposure=cluster_exp,
                note=(
                    f"Az új pozíció ({symbol}) erősen korrelál a "
                    f"meglévő {correlated_sym} pozícióval. "
                    "Spot portfólióban ez koncentrált kockázatot jelent."
                ),
            )

        # 3. Portfolio VaR (historikus szimuláció)
        # A bővített portfólió egyenlő súlyú hozamaiból számítjuk
        all_return_series: list[list[float]] = []
        for sym in extended_symbols:
            r = returns_dict.get(sym) or returns_dict.get(self._strip_suffix(sym), [])
            if len(r) >= cfg.min_history_bars:
                all_return_series.append(r)

        portfolio_var = 0.0
        if all_return_series:
            portfolio_returns = self._portfolio_returns(all_return_series)
            portfolio_var = self.compute_var(portfolio_returns, cfg.var_confidence)
        else:
            logger.warning(
                "Nincs elegendő hozam-adat a VaR számításhoz (%s + meglévők).", symbol
            )

        if portfolio_var > cfg.max_var_95_pct:
            return PortfolioRiskResult(
                can_open=False,
                reason=(
                    f"portfolio VaR {portfolio_var:.2%} > limit {cfg.max_var_95_pct:.2%}"
                ),
                var_95_pct=portfolio_var,
                max_corr_with_existing=max_corr,
                correlated_symbol=correlated_sym,
                cluster_exposure=cluster_exp,
                note=(
                    "A portfólió VaR (95%, 1 napos horizont) meghaladja a limitet. "
                    "Csökkentsd a pozíció méretét vagy az összkitettséget."
                ),
            )

        return PortfolioRiskResult(
            can_open=True,
            reason="ok",
            var_95_pct=portfolio_var,
            max_corr_with_existing=max_corr,
            correlated_symbol=correlated_sym,
            cluster_exposure=cluster_exp,
            note="",
        )

    # ------------------------------------------------------------------
    # VaR – Historical Simulation
    # ------------------------------------------------------------------

    def compute_var(self, returns: list[float], confidence: float = 0.95) -> float:
        """
        Historikus szimulációs VaR.

        Visszatér: pozitív tört (pl. 0.05 = 5% VaR).
        Ha nincs elég adat, 0.0-t ad vissza.
        """
        if len(returns) < 2:
            return 0.0
        sorted_returns = sorted(returns)
        idx = int((1.0 - confidence) * len(sorted_returns))
        idx = max(0, min(idx, len(sorted_returns) - 1))
        worst = sorted_returns[idx]
        # VaR pozitív szám: a veszteség mértéke
        return max(0.0, -worst)

    # ------------------------------------------------------------------
    # Pearson korreláció
    # ------------------------------------------------------------------

    def compute_correlation(
        self, returns_a: list[float], returns_b: list[float]
    ) -> float:
        """
        Pearson-korreláció két hozam-sorozat között.
        Visszatér 0.0-val, ha nincs elegendő adat vagy nincs szórásnégyzet.
        """
        n = min(len(returns_a), len(returns_b))
        if n < 2:
            return 0.0

        a = returns_a[-n:]
        b = returns_b[-n:]

        mean_a = sum(a) / n
        mean_b = sum(b) / n

        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
        var_a = sum((x - mean_a) ** 2 for x in a)
        var_b = sum((x - mean_b) ** 2 for x in b)

        denom = math.sqrt(var_a * var_b)
        if denom == 0.0:
            return 0.0
        return cov / denom

    # ------------------------------------------------------------------
    # Cluster kezelés
    # ------------------------------------------------------------------

    def get_cluster(self, symbol: str) -> Optional[str]:
        """Visszaadja a szimbólum klaszterét (pl. 'btc_cluster')."""
        base = self._strip_suffix(symbol).upper()
        for cluster_name, members in COIN_CLUSTERS.items():
            if base in members:
                return cluster_name
        return None

    def cluster_exposure(
        self,
        open_symbols: list[str],
        open_values: list[float],
        total_value: float,
    ) -> dict[str, float]:
        """
        Klaszterenkénti kitettség kiszámítása.

        Paraméterek
        -----------
        open_symbols : nyitott (és esetleg az új) pozíciók szimbólumai
        open_values  : az egyes pozíciók USD-értéke (azonos sorrendben)
        total_value  : a teljes portfólió USD értéke (nevező)

        Visszatér
        ---------
        {klaszter_neve: portfólió_arány}  — csak nemüres klaszterek
        """
        if total_value <= 0:
            return {}

        agg: Dict[str, float] = {}
        for sym, val in zip(open_symbols, open_values):
            cluster = self.get_cluster(sym)
            if cluster is None:
                cluster = "other"
            agg[cluster] = agg.get(cluster, 0.0) + val

        return {k: v / total_value for k, v in agg.items()}

    # ------------------------------------------------------------------
    # Belső segédfüggvények
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_suffix(symbol: str) -> str:
        """'BTC/USDT' → 'BTC', 'ETHUSDT' → 'ETH' (egyszerű közelítés)."""
        s = symbol.upper()
        for suffix in ("/USDT", "USDT", "/BTC", "BTC", "/ETH", "ETH"):
            if s.endswith(suffix):
                return s[: -len(suffix)]
        return s

    @staticmethod
    def _portfolio_returns(series_list: list[list[float]]) -> list[float]:
        """
        Egyenlő súlyú portfólió napi hozamait számítja ki.
        Az egyes sorozatokból a közös hosszú részt veszi figyelembe.
        """
        min_len = min(len(s) for s in series_list)
        n = len(series_list)
        portfolio = []
        for i in range(min_len):
            avg = sum(s[-(min_len - i)] for s in series_list) / n
            portfolio.append(avg)
        return portfolio
