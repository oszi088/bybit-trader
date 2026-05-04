"""
Globalis konfiguracio a kripto trader ugynokhoz (Bybit spot kiadas).

API kulcsok kornyezeti valtozokbol:
  BYBIT_API_KEY, BYBIT_API_SECRET
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (opcionalis, ertesitesekhez)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


# ============================================================================
# Indikator parameterek
# ============================================================================

@dataclass(frozen=True)
class IndicatorParams:
    # Trend
    sma_fast: int = 20
    sma_slow: int = 50
    sma_long: int = 200      # death/golden cross hosszu atlag
    cross_lookback: int = 5  # az utolso N gyertyan keresunk atkeresztelest
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14

    # Momentum
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    cci_period: int = 20

    # Volatilitas
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # Volumen
    mfi_period: int = 14


# Gyorsitott parameterek scalpinghoz (1m / 3m / 5m timeframe-ekhez).
# Minden periodus kb. 50%-kal rovidebb -> az indikator hamarabb reagal.
SCALPING_INDICATORS = IndicatorParams(
    sma_fast=9, sma_slow=21, sma_long=50,
    cross_lookback=3,
    ema_fast=5, ema_slow=13, macd_signal=4,
    adx_period=7,
    rsi_period=7, rsi_oversold=25.0, rsi_overbought=75.0,
    stoch_k=7, stoch_d=3, cci_period=10,
    bb_period=14, bb_std=2.0,
    atr_period=7,
    mfi_period=7,
)


# ============================================================================
# Sulyok (rezsimek nelkul)
# ============================================================================

DEFAULT_WEIGHTS: Dict[str, float] = {
    "sma_cross": 1.0, "ema_cross": 1.0, "macd": 1.5, "adx": 0.5,
    "rsi": 1.5, "stochastic": 1.0, "cci": 0.5,
    "bollinger": 1.0, "atr": 0.5,
    "obv": 0.7, "vwap": 0.8, "mfi": 0.5,
    "fear_greed": 0.8,
    "golden_death": 1.5,
    "long_trend": 0.7,
}

# Trend rezsimben a trendkoveto indikatorok dominalnak
TREND_WEIGHTS: Dict[str, float] = {
    "sma_cross": 1.5, "ema_cross": 1.5, "macd": 2.0, "adx": 1.0,
    "rsi": 0.5, "stochastic": 0.3, "cci": 0.3,
    "bollinger": 0.3, "atr": 0.0,
    "obv": 1.0, "vwap": 1.0, "mfi": 0.5,
    "fear_greed": 0.5,
    "golden_death": 1.5,
    "long_trend": 1.0,
}

# Range/sav rezsimben a mean-reversion indikatorok dominalnak
RANGE_WEIGHTS: Dict[str, float] = {
    "sma_cross": 0.3, "ema_cross": 0.3, "macd": 0.5, "adx": 0.0,
    "rsi": 2.0, "stochastic": 1.5, "cci": 1.0,
    "bollinger": 2.0, "atr": 0.0,
    "obv": 0.5, "vwap": 1.0, "mfi": 1.0,
    "fear_greed": 1.2,
    "golden_death": 1.0,
    "long_trend": 0.3,
}

# Scalping rezsim sub-5min timeframe-ekhez:
#   * a momentum-jelzok (RSI, Stoch, CCI, MACD) dominalnak (gyors fordulat)
#   * a Bollinger es VWAP fontos (mean-reversion + intraday referencia)
#   * a golden/death cross es long_trend irrelevans 1m skalan -> kis suly
#   * a fear & greed napi -> minimalis hatas
SCALPING_WEIGHTS: Dict[str, float] = {
    "sma_cross": 0.7, "ema_cross": 1.5, "macd": 1.5, "adx": 0.5,
    "rsi": 2.0, "stochastic": 2.0, "cci": 1.0,
    "bollinger": 1.8, "atr": 0.0,
    "obv": 0.8, "vwap": 1.5, "mfi": 0.8,
    "fear_greed": 0.2,
    "golden_death": 0.2,
    "long_trend": 0.2,
}


# ============================================================================
# Bybit endpointok
# ============================================================================

BybitEndpoint = Literal["eu", "global", "testnet"]

BYBIT_HOSTS: Dict[str, str] = {
    "eu":      "https://api.bybit.eu",
    "global":  "https://api.bybit.com",
    "testnet": "https://api-testnet.bybit.com",
}


def load_api_credentials():
    """Bybit API kulcsok kornyezeti valtozokbol."""
    return (
        os.environ.get("BYBIT_API_KEY"),
        os.environ.get("BYBIT_API_SECRET"),
    )


# ============================================================================
# Almodulok
# ============================================================================

@dataclass
class RiskConfig:
    """Vedelmek az eles kereskedeshez."""
    # Belepes-szintu plafon
    max_order_value_usd: float = 50.0

    # Napi realizalt veszteseg limit (kill switch)
    daily_loss_limit_usd: float = 100.0

    # Drawdown limit: ha az equity X%-ot esik a csucsrol, leallunk
    max_drawdown_pct: float = 0.10  # 10%

    # Score-aranyos pozicio: ha True, a position_size meg a |score|-val is szorzodik
    # (azaz 0.6-os score 60%-os meretu belepest ad)
    score_proportional_size: bool = True

    # Volatilitas-szuro: ha az ATR/ar arany >= ez az ertek, ne nyissunk uj poziciot
    max_atr_pct: float = 0.05   # 5%-nal magasabb relativ ATR -> kihagyjuk

    # Ha True, a broker NEM kuld valodi megbizast (csak logol)
    dry_run: bool = False


@dataclass
class StopConfig:
    """Stop-loss / take-profit / trailing parameterek."""
    # ATR-alapu dinamikus stopok (ha False, a fix % stop_loss_pct / take_profit_pct megy)
    use_atr_stops: bool = True
    atr_stop_mult: float = 3.0   # benchmark szerinti optimum (3*ATR a 2*ATR helyett)
    atr_tp_mult: float = 5.0     # benchmark szerinti optimum (5*ATR a 3*ATR helyett)

    # Fix % stopok (fallback, ha use_atr_stops False)
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06

    # Trailing stop
    use_trailing_stop: bool = False  # benchmark: a trailing kiveri a winner-eket; nelkulle jobb a hozam
    trailing_atr_mult: float = 2.0   # ha bekapcsolod, ennyi ATR-tol a max ar alatt zar


@dataclass
class RegimeConfig:
    """Piaci rezsim detektor."""
    enabled: bool = True
    adx_trend_threshold: float = 25.0   # ADX e folott trend
    adx_range_threshold: float = 18.0   # ADX e alatt range
    # 25 felett trend, 18 alatt range, koztuk neutral (DEFAULT_WEIGHTS)


@dataclass
class BacktestConfig:
    """Backteszt-specifikus beallitasok."""
    # Slippage (basis pont, 1 bp = 0.01%) - belepes utan/elott a piac elcsuszik
    slippage_bps: float = 5.0   # 5 bp = 0.05%
    # Spread (a ket oldal kozott) - de facto extra koltseg
    spread_bps: float = 2.0


@dataclass
class NotifyConfig:
    """Telegram ertesites."""
    enabled: bool = True   # ha env nincs beallitva, csak loggol
    notify_on_fill: bool = True
    notify_on_kill_switch: bool = True
    notify_on_error: bool = True


@dataclass
class DbConfig:
    """SQLite trade log es allapot persistencia."""
    enabled: bool = True
    db_path: str = "trader_state.db"






@dataclass
class MTFConfig:
    """Multi-timeframe megerosites: 6h, 12h, 1d, 1w, 1M trendje."""
    enabled: bool = True
    # Mode:
    #   "weighted" - composite score plus szavazokent a fo dontesben
    #   "gate"     - ha az MTF eros ellentmondast mutat, decision -> HOLD
    #   "off"      - nem hat semmire
    mode: str = "weighted"

    # A megfigyelt timeframe-ek.
    # 1M = havi gyertya; ev mint 12 honap szerepelhet.
    timeframes: list = field(default_factory=lambda: ["6h", "12h", "1d", "1w", "1M"])

    # Sulyozas: hosszabb tf nagyobb suly (long-term trend dominal)
    weights: dict = field(default_factory=lambda: {
        "6h":  0.5,
        "8h":  0.5,
        "12h": 0.7,
        "1d":  1.0,
        "1w":  1.5,
        "1M":  2.0,
    })

    # weighted modban: az MTF composite_score sulya a fo aggregaltban
    composite_weight: float = 1.5

    # gate modban: a kuszob, ami felett az MTF "ellentmondas" blokkolo
    gate_threshold: float = 0.3

    # SMA periodusok az egyes tf-ek tendenciajahoz
    sma_fast: int = 20
    sma_slow: int = 50


@dataclass
class FearGreedConfig:
    """Makro hangulat indikator (alternative.me/fng)."""
    enabled: bool = True
    cache_ttl_sec: int = 3600   # naponta 1x frissul, 1 ora cache eleg


@dataclass
class WatchdogConfig:
    """Heartbeat / iteracio watchdog."""
    enabled: bool = True
    max_silent_seconds: int = 600   # 10 perc utan riasztas


# ============================================================================
# Fokonfiguracio
# ============================================================================

@dataclass
class TradingConfig:
    # Penzugyi parameterek (paper / backtest modhoz)
    initial_balance: float = 10_000.0
    fee_rate: float = 0.001
    position_size: float = 0.95

    # Dontesi kuszobok ([-1, +1])
    buy_threshold: float = 0.40   # benchmark: HT+atr_loose preset
    sell_threshold: float = -0.40

    # Megorzott legacy mezok (a StopConfig hasznaloja az alapertelmezett)
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06

    # Piaci beallitasok
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    poll_interval_sec: int = 60

    # Bybit specifikus
    exchange_id: str = "bybit"
    bybit_endpoint: BybitEndpoint = "testnet"
    market_type: Literal["spot", "linear"] = "spot"

    # Beagyazott alkonfiguraciok
    indicators: IndicatorParams = field(default_factory=IndicatorParams)
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    risk: RiskConfig = field(default_factory=RiskConfig)
    stops: StopConfig = field(default_factory=StopConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    db: DbConfig = field(default_factory=DbConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    fear_greed: FearGreedConfig = field(default_factory=FearGreedConfig)
    mtf: MTFConfig = field(default_factory=MTFConfig)

    @property
    def bybit_host(self) -> str:
        return BYBIT_HOSTS[self.bybit_endpoint]

    @property
    def is_live(self) -> bool:
        return self.bybit_endpoint in ("eu", "global")


# ============================================================================
# Granularitas / scalping helperek
# ============================================================================

# Hany masodperc varakozas iteraciok kozott egy adott timeframe-en.
# Cel: a poll-ok finoman beleerjenek a gyertyak frissulesi ritmusaba,
# de ne pazaroljuk a CCXT rate-limitet (Bybit ~120 req/sec/IP, de spot
# 50 req/sec biztos hatar).
TIMEFRAME_POLL_SECONDS: Dict[str, int] = {
    "1m": 10,    "3m": 20,    "5m": 30,
    "15m": 60,   "30m": 90,   "1h": 60,
    "2h": 120,   "4h": 180,   "6h": 300,
    "8h": 300,   "12h": 600,  "1d": 900,
    "1w": 3600,  "1M": 7200,
}


def poll_interval_for_timeframe(tf: str) -> int:
    """Optimalis poll periodus egy adott timeframe-hez."""
    return TIMEFRAME_POLL_SECONDS.get(tf, 60)


# Sub-5min timeframe-ek (scalping)
GRANULAR_TIMEFRAMES = {"1m", "3m", "5m"}


def is_granular_timeframe(tf: str) -> bool:
    """True ha a timeframe sub-5min (scalping mod ajanlott)."""
    return tf in GRANULAR_TIMEFRAMES


def make_scalping_config(
    timeframe: str = "1m",
    symbol: str = "BTC/USDT",
    bybit_endpoint: BybitEndpoint = "testnet",
) -> "TradingConfig":
    """
    Sub-5min trading-re hangolt TradingConfig.

    Mit valtoztatunk a default-hoz kepest:
      * Indikator periodusok feleznek (RSI=7, MACD=5/13/4, BB=14, ATR=7, ...)
      * Sulyozas a momentum-fele tolva (SCALPING_WEIGHTS)
      * MTF-et alacsonyabb timeframe-ekre cseljuk (5m, 15m, 1h, 4h)
      * ATR-stop szukebb (1.5*ATR / 2.5*ATR) -> sub-5min mozgas kisebb
      * volatilitas-szuro szigorubb (3% relativ ATR a 5% helyett)
      * poll_interval idoaranyos (1m -> 10s)
      * buy/sell threshold magasabb (a fee + slippage 1m skalan jelentos -
        csak eros konszenzussal nyitunk poziciot)
      * trailing stop bekapcsolva: scalpingben muszaj winnert lockolni
      * F&G cache es signal sulya minimalis (napi adat 1m-en irrelevans)
      * regime threshold-ok kissebb ADX-re vannak hangolva
    """
    cfg = TradingConfig(
        symbol=symbol,
        timeframe=timeframe,
        bybit_endpoint=bybit_endpoint,
        poll_interval_sec=poll_interval_for_timeframe(timeframe),
        # Scalpingben szigorubb threshold (a kereskedesi koltseg miatt)
        buy_threshold=0.55,
        sell_threshold=-0.55,
        position_size=0.40,   # max 40% notional egy belepesnel (gyakori re-entry)
        indicators=SCALPING_INDICATORS,
        weights=dict(SCALPING_WEIGHTS),
    )
    # Szukebb ATR stopok
    cfg.stops.atr_stop_mult = 1.5
    cfg.stops.atr_tp_mult = 2.5
    cfg.stops.use_trailing_stop = True
    cfg.stops.trailing_atr_mult = 1.0
    # Szigorubb vol-szuro: scalping nem tudja megfinanszirozni a 5%-os ATR/ar arany
    cfg.risk.max_atr_pct = 0.03
    # Kisebb max notional: scalpingben sok kis trade
    cfg.risk.max_order_value_usd = 25.0
    # Regime: alacsonyabb ADX kuszobok 1m skalara
    cfg.regime.adx_trend_threshold = 20.0
    cfg.regime.adx_range_threshold = 15.0
    # F&G: napi adat irrelevans 1m skalan
    cfg.fear_greed.cache_ttl_sec = 3600
    # MTF: sub-5min cascade
    cfg.mtf.timeframes = ["5m", "15m", "1h", "4h"]
    cfg.mtf.weights = {
        "5m":  0.5,
        "15m": 1.0,
        "1h":  1.5,
        "4h":  2.0,
    }
    cfg.mtf.composite_weight = 1.5
    cfg.mtf.sma_fast = 9
    cfg.mtf.sma_slow = 21
    # Watchdog: 1m-en max 5 perc nemasag mar gyanus
    cfg.watchdog.max_silent_seconds = 300
    return cfg
