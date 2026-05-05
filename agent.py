"""
TradingAgent - regime + Fear & Greed + Multi-TimeFrame megerosites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger("agent")

from adaptive_strategy import CycleRegimeParams, get_params
from altcoin_filter import AltseasonValidator, ValidationResult, CapTier, get_eligible_symbols, HALVING_SURVIVORS
from config import TradingConfig
from market_timing import MarketTimingAnalyzer, TimingScore
from override_engine import BlockReason, OverrideDecision, OverrideEngine, evaluate_override
from fear_greed import FearGreedReading, FearGreedSource
from fear_greed_history import FearGreedHistory
from indicators import compute_all
from market_cycle import CycleState, MarketCycle, MarketCycleDetector
from ml_features import build_feature_matrix
from ml_model import MLScore, MetaLabelModel
from mtf import MTFAnalyzer, MTFReading
from regime import RegimeReading, detect_regime
from orderbook_features import OrderBookFetcher
from signals import (
    signal_adx, signal_atr, signal_bollinger, signal_cci,
    signal_ema_cross, signal_fear_greed, signal_macd, signal_mfi,
    signal_golden_death, signal_long_trend,
    signal_ob_imbalance, signal_ob_large_order,
    signal_obv, signal_rsi, signal_sma_cross, signal_stochastic, signal_vwap,
)


@dataclass
class Decision:
    action: str                       # "BUY", "SELL", "HOLD"
    score: float                      # [-1.0, +1.0]
    price: float
    atr: float
    regime: str = "neutral"
    fear_greed: int = 50
    mtf_label: str = "n/a"
    mtf_score: float = 0.0
    mtf_signals: Dict[str, int] = field(default_factory=dict)
    reasons: Dict[str, int] = field(default_factory=dict)
    ml_score: Optional[MLScore] = None                    # meta-label konfidencia
    cycle: Optional[CycleState] = None                    # piaci ciklus állapot
    cycle_params: Optional[CycleRegimeParams] = None      # adaptív paraméterek
    altseason_result: Optional[ValidationResult] = None   # valódi vs. false altseason
    timing: Optional[TimingScore] = None                  # időalapú aktivitás score
    override: Optional[OverrideDecision] = None           # volt-e szabály felülírás

    def explain(self) -> str:
        bullish = [n for n, s in self.reasons.items() if s > 0]
        bearish = [n for n, s in self.reasons.items() if s < 0]
        ml_part = ""
        if self.ml_score and self.ml_score.fitted:
            ml_part = (f" | ML p={self.ml_score.probability:.2f} "
                       f"bet={self.ml_score.bet_size:.2f}")
        cycle_part = ""
        if self.cycle:
            cycle_part = (f" | cycle={self.cycle.cycle.value}"
                          f"(conf={self.cycle.confidence:.2f}"
                          f",rem~{self.cycle.days_remaining_est}d)")
        timing_part = ""
        if self.timing:
            timing_part = (f" | timing={self.timing.trade_label}"
                           f"({self.timing.overall:.2f}×{self.timing.position_size_mult:.2f})")
        override_part = ""
        if self.override and self.override.triggered:
            rules = [r.value for r in self.override.overridden_rules]
            override_part = (f" | ⚠️OVERRIDE({','.join(rules)})"
                             f"conv={self.override.conviction.total:.3f}"
                             f"×{self.override.position_size_mult:.2f}")
        return (
            f"{self.action} @ {self.price:.2f} | score={self.score:+.2f} "
            f"| regime={self.regime} | F&G={self.fear_greed} "
            f"| MTF={self.mtf_label}({self.mtf_score:+.2f})"
            f"{ml_part}{cycle_part}{timing_part}{override_part}"
            f" | bullish={bullish} bearish={bearish}"
        )


class TradingAgent:
    def __init__(self, config: Optional[TradingConfig] = None,
                 fear_greed_source: Optional[FearGreedSource] = None,
                 fg_history: Optional[FearGreedHistory] = None,
                 mtf_analyzer: Optional[MTFAnalyzer] = None,
                 ml_model: Optional[MetaLabelModel] = None,
                 cycle_detector: Optional[MarketCycleDetector] = None,
                 cycle_state_path: Optional[str] = "data/cycle_state.json",
                 symbol: Optional[str] = None,
                 ob_fetcher: Optional[OrderBookFetcher] = None):
        self.config = config or TradingConfig()
        self._enriched: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._cycle_state: Optional[CycleState] = None
        self._altseason_result: Optional[ValidationResult] = None
        self._symbol = symbol   # aktuálisan kereskedett szimbólum

        # Élő orderbook fetcher (None = backtest / nem elérhető)
        self._ob_fetcher: Optional[OrderBookFetcher] = ob_fetcher
        # Cache: egy döntési cikluson belül egyszer lekért OB feature-ök
        self._live_ob_features: Optional[dict] = None

        # Historikus F&G lookup (ML train loophoz — lookahead mentesítés).
        # Ha meg van adva: decide_at() a gyertya dátumához tartozó értéket
        # keresi, NEM a live API-t hívja.
        self._fg_history: Optional[FearGreedHistory] = fg_history

        # Meta-label ML modell (opcionális)
        self.ml: Optional[MetaLabelModel] = ml_model

        # Piaci ciklus detektor
        self.cycle_detector = cycle_detector or MarketCycleDetector(
            state_path=cycle_state_path,
            smoothing=3,
        )

        # Alt szezon validátor
        altseason_state = (
            cycle_state_path.replace("cycle_state.json", "altseason_state.json")
            if cycle_state_path else None
        )
        self.altseason_validator = AltseasonValidator(state_path=altseason_state)

        # Időalapú aktivitás elemző
        self.timing_analyzer = MarketTimingAnalyzer()

        # Fear & Greed forras
        if self.config.fear_greed.enabled:
            self._fg = fear_greed_source or FearGreedSource(
                ttl_sec=self.config.fear_greed.cache_ttl_sec
            )
        else:
            self._fg = None

        # Multi-timeframe analyzer
        if self.config.mtf.enabled and self.config.mtf.mode != "off":
            self.mtf = mtf_analyzer or MTFAnalyzer(
                timeframes=self.config.mtf.timeframes,
                weights=self.config.mtf.weights,
                fast=self.config.mtf.sma_fast,
                slow=self.config.mtf.sma_slow,
            )
        else:
            self.mtf = None

    # ------------------------------------------------------------------ #
    # Belso seged
    # ------------------------------------------------------------------ #

    def _gather_signals(self, row: pd.Series, prev_obv: Optional[float],
                        fg_value: int) -> Dict[str, int]:
        params = self.config.indicators

        # Ha van élő OB fetcher, a row-t kibővítjük a friss OB feature-ökkel
        # (csak ha már le lett kérve a _live_ob_features a decide() hívásban)
        effective_row = row
        if self._live_ob_features is not None:
            effective_row = row.copy()
            for k, v in self._live_ob_features.items():
                effective_row[k] = v

        return {
            "sma_cross":    signal_sma_cross(effective_row),
            "ema_cross":    signal_ema_cross(effective_row),
            "macd":         signal_macd(effective_row),
            "adx":          signal_adx(effective_row),
            "rsi":          signal_rsi(effective_row, params),
            "stochastic":   signal_stochastic(effective_row, params),
            "cci":          signal_cci(effective_row),
            "bollinger":    signal_bollinger(effective_row),
            "atr":          signal_atr(effective_row),
            "obv":          signal_obv(effective_row, prev_obv),
            "vwap":         signal_vwap(effective_row),
            "mfi":          signal_mfi(effective_row),
            "fear_greed":   signal_fear_greed(fg_value),
            "golden_death": signal_golden_death(effective_row),
            "long_trend":   signal_long_trend(effective_row),
            # Orderflow szignalok
            "ob_imbalance":   signal_ob_imbalance(effective_row),
            "ob_large_order": signal_ob_large_order(effective_row),
        }

    def _aggregate(self, signals: Dict[str, int], weights: Dict[str, float],
                   mtf_score: float = 0.0) -> float:
        """Sulyozott szavazas + opcionalis MTF composite score."""
        weighted_sum = sum(weights.get(name, 0.0) * sig for name, sig in signals.items())
        weight_total = sum(abs(w) for w in weights.values()) or 1.0

        # MTF mint plus szavazo (csak weighted modban)
        if self.mtf is not None and self.config.mtf.mode == "weighted":
            mw = self.config.mtf.composite_weight
            weighted_sum += mw * mtf_score
            weight_total += abs(mw)

        return weighted_sum / weight_total

    def _score_to_action(self, score: float, mtf: Optional[MTFReading]) -> str:
        action = "HOLD"
        if score >= self.config.buy_threshold:
            action = "BUY"
        elif score <= self.config.sell_threshold:
            action = "SELL"

        # Gate mod: ha az MTF erosen ellentmond, override HOLD-ra
        if (self.mtf is not None
                and self.config.mtf.mode == "gate"
                and mtf is not None):
            t = self.config.mtf.gate_threshold
            if action == "BUY" and mtf.composite_score < -t:
                action = "HOLD"
            elif action == "SELL" and mtf.composite_score > t:
                action = "HOLD"

        return action

    # ------------------------------------------------------------------ #
    # Publikus API
    # ------------------------------------------------------------------ #

    def prepare(self, ohlcv: pd.DataFrame,
                fg_value: int = 50,
                funding_rate: float = 0.0,
                btc_dominance: float = 50.0,
                eth_btc_ratio: Optional[float] = None,
                vix: float = 20.0) -> pd.DataFrame:
        self._enriched = compute_all(ohlcv, self.config.indicators)
        if self.ml is not None:
            self._features = build_feature_matrix(ohlcv, self.config.indicators)

        # logger már elérhető modulszinten

        # Piaci ciklus detektálás a teljes OHLCV history alapján
        try:
            self._cycle_state = self.cycle_detector.detect(
                ohlcv=ohlcv,
                fg_value=fg_value,
                funding_rate=funding_rate,
                btc_dominance=btc_dominance,
                vix=vix,
            )
        except Exception as e:
            logger.warning("Ciklus detektálás hiba: %s", e)
            self._cycle_state = None

        # Alt szezon validáció (csak ha a ciklus azt mutatja)
        self._altseason_result = None
        if (self._cycle_state is not None
                and self._cycle_state.cycle.value == "altseason"):
            try:
                self._altseason_result = self.altseason_validator.validate(
                    ohlcv_btc=ohlcv,
                    btc_dominance=btc_dominance,
                    eth_btc_ratio=eth_btc_ratio,
                    vix=vix,
                )
            except Exception as e:
                logger.warning("Altseason validáció hiba: %s", e)

        return self._enriched

    def _current_fg(self) -> FearGreedReading:
        """Live F&G – csak élő kereskedésnél. Training-ben _fg_for_ts()-t használj."""
        if self._fg is None:
            from fear_greed import FearGreedReading as _FGR
            from datetime import datetime, timezone
            return _FGR(50, "Disabled", datetime.now(timezone.utc))
        return self._fg.get()

    def _fg_for_ts(self, timestamp) -> int:
        """
        Lookahead-mentes F&G érték lekérése.

        Ha van historikus adatforrás (fg_history), a gyertya dátumához
        tartozó értéket adja vissza — training loopban ezt kell hívni.
        Ha nincs (élő kereskedés), a live API-t hívja meg.
        """
        from datetime import datetime as _dt, timezone as _tz
        if self._fg_history is not None:
            # Historikus lookup: a timestamp dátumából keresi az értéket
            if hasattr(timestamp, "to_pydatetime"):
                ts_dt = timestamp.to_pydatetime()
            elif isinstance(timestamp, _dt):
                ts_dt = timestamp
            else:
                ts_dt = _dt.now(_tz.utc)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=_tz.utc)
            return self._fg_history.get_value_for_ts(ts_dt)
        return self._current_fg().value

    def decide_at(self, index: int, funding_rate: float = 0.0) -> Decision:
        if self._enriched is None:
            raise RuntimeError("Eloszor hivd meg a prepare(ohlcv) metodust.")

        row = self._enriched.iloc[index]
        timestamp = self._enriched.index[index]
        prev_obv = self._enriched["obv"].iloc[index - 1] if index > 0 else None

        regime: RegimeReading = detect_regime(row, self.config.regime)
        fg_value = self._fg_for_ts(timestamp)
        # FearGreedReading csak a teljes Decision objektumhoz kell (fear_greed mező)
        from fear_greed import FearGreedReading as _FGR
        from datetime import datetime as _dt, timezone as _tz
        fg = _FGR(fg_value, "", _dt.now(_tz.utc))
        signals = self._gather_signals(row, prev_obv, fg.value)

        # ── Időalapú timing score ────────────────────────────────────────
        from datetime import datetime as _dt, timezone as _tz
        if hasattr(timestamp, "to_pydatetime"):
            _ts_dt = timestamp.to_pydatetime()
            if _ts_dt.tzinfo is None:
                _ts_dt = _ts_dt.replace(tzinfo=_tz.utc)
        elif isinstance(timestamp, _dt):
            _ts_dt = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=_tz.utc)
        else:
            # Integer vagy egyéb index: nincs értelmes dátum → UTC epoch + index másodperc
            # Backtestben ez nem fordul elő ha az enriched DatetimeIndex-es, élőben soha.
            logger.debug("decide_at: nem-datetime index (%r) → datetime.now fallback", timestamp)
            _ts_dt = _dt.now(_tz.utc)
        timing = self.timing_analyzer.score(_ts_dt)

        # MTF analizis (csak ha be van kapcsolva)
        mtf_reading: Optional[MTFReading] = None
        if self.mtf is not None:
            mtf_reading = self.mtf.analyze(as_of=timestamp)

        score = self._aggregate(
            signals, regime.weights,
            mtf_score=(mtf_reading.composite_score if mtf_reading else 0.0),
        )

        # Az eredeti szándékolt irány (mielőtt bármit blokkolnánk)
        intended_action = self._score_to_action(score, mtf_reading)
        action = intended_action
        blocked_reasons: list[BlockReason] = []

        # ── Timing blokkok ───────────────────────────────────────────────
        if timing.hard_block and action in ("BUY", "SELL"):
            blocked_reasons.append(BlockReason.TIMING_HARD)
            action = "HOLD"
        elif action in ("BUY", "SELL"):
            effective_base = self.config.buy_threshold + timing.score_threshold_delta
            if action == "BUY" and score < effective_base:
                blocked_reasons.append(BlockReason.TIMING_THRESHOLD)
                action = "HOLD"

        # ── Ciklus-adaptív szűrők ────────────────────────────────────────
        cycle_state = self._cycle_state
        cycle_params: Optional[CycleRegimeParams] = None
        if cycle_state is not None:
            cycle_params = get_params(cycle_state.cycle)

            # Irány-tilalom
            if intended_action == "BUY" and not cycle_params.allow_long:
                blocked_reasons.append(BlockReason.CYCLE_DIRECTION)
                action = "HOLD"
            if intended_action == "SELL" and not cycle_params.allow_short:
                blocked_reasons.append(BlockReason.CYCLE_DIRECTION)
                action = "HOLD"

            # Score threshold (ciklus + timing összesítve, ha még nem volt blokk)
            if action == "BUY":
                effective_threshold = (
                    self.config.buy_threshold
                    + cycle_params.score_threshold_delta
                    + timing.score_threshold_delta
                )
                if score < effective_threshold:
                    blocked_reasons.append(BlockReason.CYCLE_THRESHOLD)
                    action = "HOLD"

        # ── Altseason validáció ──────────────────────────────────────────
        altseason_result = self._altseason_result
        if (intended_action == "BUY"
                and cycle_state is not None
                and cycle_state.cycle.value == "altseason"
                and cycle_params is not None
                and cycle_params.altseason_validate):

            if altseason_result is None or not altseason_result.confirmed:
                blocked_reasons.append(BlockReason.ALTSEASON_FALSE)
                action = "HOLD"
            elif self._symbol is not None:
                days_in = cycle_state.days_in_cycle
                allowed_tiers: list[CapTier] = [CapTier.LARGE]
                if days_in >= 14: allowed_tiers.append(CapTier.MID)
                if days_in >= 30: allowed_tiers.append(CapTier.SMALL)

                # Halving-ellenőrzést külön végezzük, hogy a helyes BlockReason
                # kerüljön az override motorba: ALTSEASON_HALVING küszöb 0.95
                # (szinte sosem engedjük felül), ALTSEASON_TIER küszöb 0.82
                # (tier fokozatos nyitásnál felülírható erős meggyőzéssel).
                sym_halvings = HALVING_SURVIVORS.get(self._symbol, 0)
                if sym_halvings < cycle_params.altseason_min_halvings:
                    blocked_reasons.append(BlockReason.ALTSEASON_HALVING)
                    action = "HOLD"
                else:
                    eligible = get_eligible_symbols(
                        min_halvings=cycle_params.altseason_min_halvings,
                        cap_tiers=allowed_tiers,
                    )
                    if self._symbol not in eligible:
                        blocked_reasons.append(BlockReason.ALTSEASON_TIER)
                        action = "HOLD"

        # ── Meta-label ML szűrő ──────────────────────────────────────────
        ml_score: Optional[MLScore] = None
        if self.ml is not None and self._features is not None:
            ml_score = self.ml.predict(self._features.iloc[[index]])
            min_prob = cycle_params.min_ml_prob if cycle_params is not None else 0.55
            if (cycle_state is not None
                    and cycle_state.cycle.value == "altseason"
                    and self._symbol is not None):
                from altcoin_filter import COIN_DB, CapTier as CT
                coin_info = COIN_DB.get(self._symbol, {})
                if coin_info.get("tier") == CT.SMALL:
                    min_prob = max(min_prob, 0.65)
                elif coin_info.get("tier") == CT.MID:
                    min_prob = max(min_prob, 0.60)

            if intended_action in ("BUY", "SELL") and ml_score.probability < min_prob:
                blocked_reasons.append(BlockReason.ML_PROBABILITY)
                action = "HOLD"

        # ── Override motor ───────────────────────────────────────────────
        # Ha valami blokkolt, megvizsgáljuk, hogy az evidence elég erős-e
        # az override-hoz. Az override sosem törli a stop-loss kötelezettséget
        # és sosem növeli a max pozícióhoz képest.
        override_dec: Optional[OverrideDecision] = None
        if action == "HOLD" and blocked_reasons and intended_action != "HOLD":
            override_dec = evaluate_override(
                intended_action=intended_action,
                blocked_reasons=blocked_reasons,
                signals=signals,
                score=score,
                ml_prob=(ml_score.probability if ml_score is not None else None),
                fg_value=fg.value,
                row=row,
                funding_rate=funding_rate,
            )
            if override_dec.triggered:
                action = override_dec.action

        return Decision(
            action=action,
            score=score,
            price=float(row["close"]),
            atr=float(row["atr"]) if not pd.isna(row["atr"]) else 0.0,
            regime=regime.label,
            fear_greed=fg.value,
            mtf_label=(mtf_reading.label if mtf_reading else "off"),
            mtf_score=(mtf_reading.composite_score if mtf_reading else 0.0),
            mtf_signals=(mtf_reading.timeframe_signals if mtf_reading else {}),
            reasons=signals,
            ml_score=ml_score,
            cycle=cycle_state,
            cycle_params=cycle_params,
            altseason_result=altseason_result,
            timing=timing,
            override=override_dec,
        )

    def decide(self, ohlcv: pd.DataFrame) -> Decision:
        self.prepare(ohlcv)

        # Élő kereskedésben: OB lekérés és cache reset döntés előtt
        if self._ob_fetcher is not None:
            raw = self._ob_fetcher.feature_dict()
            # Normalizálás: az ob_imbalance_10 → ob_imbalance (amit a signal-ok keresnek)
            self._live_ob_features = {
                "ob_imbalance":  raw.get("ob_imbalance_10", 0.0),
                "ob_large_order": raw.get("ob_large_order", 0),
                # extra meta (loghoz, de signal nem használja)
                "ob_imbalance_20":   raw.get("ob_imbalance_20", 0.0),
                "ob_depth_ratio_20": raw.get("ob_depth_ratio_20", 1.0),
                "ob_spread_pct":     raw.get("ob_spread_pct", 0.0),
            }
            logger.debug(
                "OB features: imb=%.3f large=%d spread=%.5f",
                self._live_ob_features["ob_imbalance"],
                self._live_ob_features["ob_large_order"],
                self._live_ob_features["ob_spread_pct"],
            )
        else:
            self._live_ob_features = None

        decision = self.decide_at(len(ohlcv) - 1)
        self._live_ob_features = None   # döntés után töröljük a cache-t
        return decision
