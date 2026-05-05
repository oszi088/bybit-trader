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

    # VWAP ablak (0 = auto: az index time-delta alapjan 24 ora)
    # 1h TF: 24, 1m TF: 1440, 1s TF: 86400 — lasd VWAP_PERIOD_BY_TF
    vwap_period: int = 0


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

# HFT parameterek 1s timeframe-hez.
# 1 perc = 60 bar, 5 perc = 300, 1 ora = 3600.
# Az indikatorok "hagyomanyos" (pl. RSI-14 = 14 masodperc) helyett
# percben/oraban ertelmezett ekvivalenst hasznalnak.
HFT_INDICATORS = IndicatorParams(
    sma_fast=60,    sma_slow=300,    sma_long=3600,   # 1min / 5min / 1h
    cross_lookback=60,                                # 1 perces lookback
    ema_fast=30,    ema_slow=120,    macd_signal=20,  # 30s / 2min
    adx_period=120,                                   # 2 perces ADX
    rsi_period=300, rsi_oversold=25.0, rsi_overbought=75.0,  # 5min RSI
    stoch_k=300,    stoch_d=60,                       # 5min %K, 1min %D
    stoch_oversold=15.0, stoch_overbought=85.0,       # szukebb savok
    cci_period=300,                                   # 5min CCI
    bb_period=300,  bb_std=1.5,                       # 5min BB, szukebb
    atr_period=120,                                   # 2min ATR
    mfi_period=300,                                   # 5min MFI
    vwap_period=86400,                                # 24 oras VWAP
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
    # Orderflow
    "ob_imbalance":   1.2,
    "ob_large_order": 0.8,
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
    # Trendben az OBI fontos (impulzus megerosites)
    "ob_imbalance":   1.5,
    "ob_large_order": 1.0,
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
    # Range-ben az OBI kevesbe megbizhatو, de large_order wall szint fontos
    "ob_imbalance":   0.8,
    "ob_large_order": 1.2,
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
    # Scalpingban az OBI a legfontosabb mikrostruktura signal
    "ob_imbalance":   2.0,
    "ob_large_order": 1.5,
}


# HFT sulyozas 1s timeframe-hez:
#   - Orderflow (OBI, large_order) a legfontosabb mikrostruktura jelzo
#   - RSI / Stoch / Bollinger: gyors mean-reversion 1s-en is mukodik
#   - Fear & Greed, golden/death cross: napi adatok, irrelevansak 1s-en
HFT_WEIGHTS: Dict[str, float] = {
    "sma_cross":    0.5,   # 1min/5min SMA → zajos, de trendet jelez
    "ema_cross":    1.5,   # 30s/2min EMA → gyorsabb reakcio
    "macd":         1.0,
    "adx":          0.5,
    "rsi":          2.0,   # oversold/overbought kritikus
    "stochastic":   2.0,
    "cci":          1.0,
    "bollinger":    1.5,   # mean-reversion 1s-en is mukodik
    "atr":          0.0,
    "obv":          0.5,
    "vwap":         1.5,   # intraday referencia
    "mfi":          0.8,
    "fear_greed":   0.0,   # napi adat irrelevans 1s-en
    "golden_death": 0.0,   # 1h/4h cross → nem hat 1s-re
    "long_trend":   0.3,   # 1h trend taktiorizalas
    "ob_imbalance":   2.5, # legfontosabb HFT mikrostruktura jel
    "ob_large_order": 2.0, # intezmenyi fal
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

    # Fix kockazat per trade (profi modszer):
    #   Ha True, a poziciomeret a stop tavolsagabol kovetkezik:
    #     size = (capital * risk_per_trade_pct) / stop_distance
    #   Ha False (fallback): notional = capital * position_size (regi logika)
    use_fixed_risk_sizing: bool = True
    risk_per_trade_pct: float = 0.01   # toke 1%-at kockaztatjuk tradeenkent

    # Volatilitas-szuro: ha az ATR/ar arany >= ez az ertek, ne nyissunk uj poziciot
    max_atr_pct: float = 0.05   # 5%-nal magasabb relativ ATR -> kihagyjuk

    # Ha True, a broker NEM kuld valodi megbizast (csak logol)
    dry_run: bool = False

    # --- Egymast koveto veszteseg circuit breaker ---
    # N veszteseg utan: meret felezese / szunet / leallas
    consecutive_loss_half_at: int = 3    # 3 veszteseg -> 50% meret
    consecutive_loss_pause_at: int = 5   # 5 veszteseg -> X oras szunet
    consecutive_loss_pause_hours: float = 4.0
    consecutive_loss_stop_at: int = 7    # 7 veszteseg -> teljes STOP


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
    # Csak az aktív timeframes-ben szereplő tf-ek legyenek itt!
    weights: dict = field(default_factory=lambda: {
        "6h":  0.5,
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
# Spot-only bővítmény konfigurációk
# ============================================================================

@dataclass
class SpotExitConfig:
    """
    Professzionalis exit stratégia spot tradinghez (exit_manager.py).

    TP1 (reszleges zarás) + trailing stop + profit lock + idobazisu exit.
    Az atr_tp_mult a CycleRegimeParams-ból felülíródik, ha adaptív ciklus aktív.
    """
    enabled: bool = True
    partial_tp_fraction: float = 0.50   # TP1-nel eladott hanyad (50%)
    tp1_atr_mult: float = 1.0           # TP1 szint: entry + 1×ATR
    trailing_atr_mult: float = 1.5     # trailing stop: legmagasabb ar - 1.5×ATR
    profit_lock_atr_mult: float = 0.5  # profit lock: legmagasabb ar - 0.5×ATR
    time_exit_bars: int = 0            # 0 = CycleRegimeParams.max_holding_bars-ból
    breakeven_after_partial: bool = True  # TP1 utan stop = entry (breakevenre huz)


@dataclass
class SpotLiquidityConfig:
    """Likviditasi szuro spot tradinghez (liquidity_filter.py)."""
    enabled: bool = True
    min_volume_usd: float = 500_000.0       # minimum napi forgalom USD-ban
    max_volume_impact_pct: float = 0.01     # max pozicio: 1% a napi forgalombol
    max_spread_pct: float = 0.002           # max bid/ask spread 0.2%


@dataclass
class SpotCostConfig:
    """Kereskedesi koltseg modell spot tradinghez (cost_model.py)."""
    enabled: bool = True
    # FIX #6: fee_rate-t a Trader mindig a TradingConfig.fee_rate-bol veszi at,
    # ezt a mezot a _init_spot_modules() felulirja — ne allitsd be manualis!
    slippage_small_usd: float = 1_000.0    # ezalatt kis slippage becsles
    slippage_base_pct: float = 0.0003      # alap slippage (0.03%)
    slippage_impact_pct: float = 0.005     # piaci impact szorzo
    min_return_multiplier: float = 1.5     # vart hozam >= koltseg x 1.5


@dataclass
class SpotTWAPConfig:
    """TWAP vegrehajtasi konfig spot tradinghez (twap.py)."""
    enabled: bool = False                  # nagy pozicional kapcsold be
    num_slices: int = 6
    total_duration_sec: int = 1800         # 30 perc
    max_price_drift_pct: float = 0.015    # 1.5% felet abort
    min_order_size_usd: float = 500.0
    use_twap_above_usd: float = 2000.0    # ennyi felett auto-TWAP


@dataclass
class SpotDriftConfig:
    """Modell drift detektalas (drift_detector.py)."""
    enabled: bool = True
    window: int = 20                  # utolso N trade ablaka
    min_trades: int = 10             # ennyi alatt nem detektalunk
    min_sharpe: float = 0.30
    min_win_rate: float = 0.40
    min_profit_factor: float = 1.0
    min_edge_ratio: float = 0.80
    # FIX #5: 0.0 = auto (a Trader _init_spot_modules szamitja a timeframe alapjan)
    # Pl. 1h TF: 365*24=8760, 4h: 2190, 1d: 365
    annualization_factor: float = 0.0  # 0 = auto kiszamitas a timeframe-bol


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

    # --- Spot-only profi bővítmények ---
    spot_exit: SpotExitConfig = field(default_factory=SpotExitConfig)
    spot_liquidity: SpotLiquidityConfig = field(default_factory=SpotLiquidityConfig)
    spot_cost: SpotCostConfig = field(default_factory=SpotCostConfig)
    spot_twap: SpotTWAPConfig = field(default_factory=SpotTWAPConfig)
    spot_drift: SpotDriftConfig = field(default_factory=SpotDriftConfig)

    @property
    def bybit_host(self) -> str:
        return BYBIT_HOSTS[self.bybit_endpoint]

    @property
    def is_live(self) -> bool:
        return self.bybit_endpoint in ("eu", "global")


# ============================================================================
# Granularitas / scalping helperek
# ============================================================================

# FIX #5: Evi periodus szam timeframe-enkent (Sharpe annualizaciohoz).
# Crypto 24/7 piac: 365 nap, nem 252 munkanap.
TIMEFRAME_PERIODS_PER_YEAR: Dict[str, int] = {
    "1s":  365 * 24 * 3600,  # 31 536 000
    "1m":  365 * 24 * 60,    #    525 600
    "3m":  365 * 24 * 20,    #    175 200
    "5m":  365 * 24 * 12,    #    105 120
    "15m": 365 * 24 * 4,     #     35 040
    "30m": 365 * 24 * 2,     #     17 520
    "1h":  365 * 24,         #      8 760
    "2h":  365 * 12,         #      4 380
    "4h":  365 * 6,          #      2 190
    "6h":  365 * 4,          #      1 460
    "8h":  365 * 3,          #      1 095
    "12h": 365 * 2,          #        730
    "1d":  365,              #        365
    "1w":  52,
    "1M":  12,
}


# Hany masodperc varakozas iteraciok kozott egy adott timeframe-en.
# Cel: a poll-ok finoman beleerjenek a gyertyak frissulesi ritmusaba,
# de ne pazaroljuk a CCXT rate-limitet (Bybit ~120 req/sec/IP, de spot
# 50 req/sec biztos hatar).
TIMEFRAME_POLL_SECONDS: Dict[str, int] = {
    "1s": 1,
    "1m": 10,    "3m": 20,    "5m": 30,
    "15m": 60,   "30m": 90,   "1h": 60,
    "2h": 120,   "4h": 180,   "6h": 300,
    "8h": 300,   "12h": 600,  "1d": 900,
    "1w": 3600,  "1M": 7200,
}


# VWAP gorduloablak timeframe-enkent (24 oras ekvivalens, kerekitve).
# compute_all() automatikusan kivalasztja a helyes erteket ha vwap_period=0.
VWAP_PERIOD_BY_TF: Dict[str, int] = {
    "1s":  86400,  "1m":  1440,  "3m":  480,  "5m":  288,
    "15m": 96,     "30m": 48,    "1h":  24,   "2h":  12,
    "4h":  6,      "6h":  4,     "8h":  3,    "12h": 2,
    "1d":  1,      "1w":  1,     "1M":  1,
}


def poll_interval_for_timeframe(tf: str) -> int:
    """Optimalis poll periodus egy adott timeframe-hez."""
    return TIMEFRAME_POLL_SECONDS.get(tf, 60)


# Sub-1min timeframe-ek (scalping + HFT)
GRANULAR_TIMEFRAMES = {"1s", "1m", "3m", "5m"}


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


def make_hft_config(
    symbol: str = "BTC/USDT",
    timeframe: str = "1s",
    exchange_id: str = "binance",
) -> "TradingConfig":
    """
    1 masodperces HFT-re hangolt TradingConfig.

    Binance az elodleges forrás: az egyetlen exchange, amelyen 1s OHLCV
    historikus adat letoltheto (data.binance.vision bulk download).

    Mit valtoztatunk a scalping confighoz kepest:
      * HFT_INDICATORS: periodusok percekre/orakra skalazva (lasd fent)
      * HFT_WEIGHTS: F&G=0, golden/death=0; OBI+large_order dominalnak
      * MTF: 1m, 5m, 15m, 1h (minden TF feljebb tolva eggyel)
      * ATR stopok meg szukebbek (0.8* / 1.2*ATR) — 1s savon kicsi a mozgas
      * max_atr_pct: 0.01 (1%) — 1s bar szinte sosem mozog 1%-ot
      * threshold: 0.60 / -0.60 — nagyon eros konszenzus kell (zaj miatt)
      * position_size: 0.20 — kis egyseg, gyors re-entry
      * VectorizedBacktester ajanlott: 86400 bar/nap → lassu az alap Backtester
    """
    cfg = TradingConfig(
        symbol=symbol,
        timeframe=timeframe,
        exchange_id=exchange_id,
        bybit_endpoint="global",      # Binance-hez nem relevans, de beallitjuk
        poll_interval_sec=1,
        buy_threshold=0.60,
        sell_threshold=-0.60,
        position_size=0.20,
        indicators=HFT_INDICATORS,
        weights=dict(HFT_WEIGHTS),
    )
    # Szukebb ATR stopok: 1s sav kicsiny
    cfg.stops.atr_stop_mult = 0.8
    cfg.stops.atr_tp_mult   = 1.2
    cfg.stops.use_trailing_stop = False   # VectorizedBacktester nem tamogatja
    # Volatilitas szuro: 1s bar ATR/ar aranyan kell szurni
    cfg.risk.max_atr_pct = 0.01
    cfg.risk.max_order_value_usd = 20.0
    # Regime: 1s-en az ADX ertekelese nehezebb, tolekcetesebb kuszobok
    cfg.regime.adx_trend_threshold = 25.0
    cfg.regime.adx_range_threshold = 15.0
    # F&G: nem hasznalt 1s-en
    cfg.fear_greed.enabled = False
    # MTF: 1s-es cascade (1m, 5m, 15m, 1h)
    cfg.mtf.timeframes = ["1m", "5m", "15m", "1h"]
    cfg.mtf.weights = {
        "1m":  0.5,
        "5m":  1.0,
        "15m": 1.5,
        "1h":  2.0,
    }
    cfg.mtf.composite_weight = 1.0
    cfg.mtf.sma_fast = 60     # 1 perces SMA
    cfg.mtf.sma_slow = 300    # 5 perces SMA
    # Watchdog: 1s-en 30 masodperc nemasag mar gyanus
    cfg.watchdog.max_silent_seconds = 30
    return cfg
