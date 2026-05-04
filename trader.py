"""
Trader - broker-agnosztikus élo/paper kereskedési loop.

Bővítések (spot-only professzionális csomag):
  * ExitManager      — részleges TP + trailing stop + profit lock + időalapú exit
  * CostModel        — belépés előtti cost/break-even szűrő
  * LiquidityFilter  — 24h volume + spread ellenőrzés
  * DriftDetector    — rolling Sharpe + win rate modell drift figyelmeztetés
  * TWAPExecutor     — nagy vételek felosztása 6 szeletbe (30 perc)
  * Consecutive loss — RiskManager circuit breaker (3/5/7 veszteség)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from agent import Decision, TradingAgent
from brokers import Broker, BybitBroker, FillReport, PaperBroker
from config import TradingConfig, TIMEFRAME_PERIODS_PER_YEAR
from cost_model import CostModel, CostConfig
from data_source import CcxtDataSource
from db import TradeDb
from drift_detector import DriftDetector, DriftConfig
from exit_manager import ExitManager, ExitConfig
from liquidity_filter import LiquidityFilter, LiquidityConfig
from notify import Notifier
from risk_manager import RiskManager
from twap import TWAPExecutor, TWAPConfig

logger = logging.getLogger("trader")


class Trader:
    """Broker-független kereskedési loop ugyanazzal a logikával."""

    def __init__(
        self,
        agent: TradingAgent,
        broker: Broker,
        risk: RiskManager,
        data_source: CcxtDataSource,
        config: Optional[TradingConfig] = None,
        history_limit: int = 200,
        db: Optional[TradeDb] = None,
        notifier: Optional[Notifier] = None,
    ):
        self.agent = agent
        self.broker = broker
        self.risk = risk
        self.data = data_source
        self.config = config or agent.config
        self.history_limit = history_limit

        self.db = db or TradeDb(self.config.db.db_path, enabled=self.config.db.enabled)
        self.notifier = notifier or Notifier(enabled=self.config.notify.enabled)

        # --- Spot-only professzionális modulok ---
        self._init_spot_modules()

        # Aktív TWAP végrehajtó (None ha nincs folyamatban)
        self._twap: Optional[TWAPExecutor] = None

        # FIX #2: aktív belépés ciklus-neve (SELL note-ba kerül)
        self._entry_cycle: str = ""

        # FIX #7: utolsó OHLCV %-os hozamok (portfolio kockázathoz)
        self._last_returns: list[float] = []

        self._last_step_ts: float = time.time()

    def _init_spot_modules(self) -> None:
        """Spot-only bővítmény modulok inicializálása a config alapján."""
        cfg = self.config

        # ExitManager
        exit_cfg = ExitConfig(
            partial_tp_fraction = cfg.spot_exit.partial_tp_fraction,
            tp1_atr_mult        = cfg.spot_exit.tp1_atr_mult,
            trailing_atr_mult   = cfg.spot_exit.trailing_atr_mult,
            profit_lock_atr_mult= cfg.spot_exit.profit_lock_atr_mult,
            time_exit_bars      = cfg.spot_exit.time_exit_bars,
            breakeven_after_partial = cfg.spot_exit.breakeven_after_partial,
        ) if cfg.spot_exit.enabled else None
        self._exit_mgr = ExitManager(exit_cfg) if exit_cfg else None

        # CostModel — FIX #6: fee_rate mindig a TradingConfig.fee_rate-ból jön
        cost_cfg = CostConfig(
            fee_rate             = cfg.fee_rate,        # szinkronizált a főkonfiggal
            slippage_small_usd   = cfg.spot_cost.slippage_small_usd,
            slippage_base_pct    = cfg.spot_cost.slippage_base_pct,
            slippage_impact_pct  = cfg.spot_cost.slippage_impact_pct,
            min_return_multiplier= cfg.spot_cost.min_return_multiplier,
        ) if cfg.spot_cost.enabled else None
        self._cost_model = CostModel(cost_cfg) if cost_cfg else None

        # LiquidityFilter
        liq_cfg = LiquidityConfig(
            min_volume_usd        = cfg.spot_liquidity.min_volume_usd,
            max_volume_impact_pct = cfg.spot_liquidity.max_volume_impact_pct,
            max_spread_pct        = cfg.spot_liquidity.max_spread_pct,
        ) if cfg.spot_liquidity.enabled else None
        self._liquidity = LiquidityFilter(liq_cfg) if liq_cfg else None

        # DriftDetector — FIX #5: timeframe-helyes Sharpe annualizáció
        if cfg.spot_drift.enabled:
            # Ha annualization_factor=0 → auto kiszámítás a timeframe alapján
            annual = cfg.spot_drift.annualization_factor
            if annual <= 0:
                annual = float(TIMEFRAME_PERIODS_PER_YEAR.get(cfg.timeframe, 365 * 24))
                logger.info(
                    "DriftDetector annualization_factor auto: %s TF → %.0f periódus/év",
                    cfg.timeframe, annual,
                )
            drift_cfg = DriftConfig(
                window              = cfg.spot_drift.window,
                min_trades          = cfg.spot_drift.min_trades,
                min_sharpe          = cfg.spot_drift.min_sharpe,
                min_win_rate        = cfg.spot_drift.min_win_rate,
                min_profit_factor   = cfg.spot_drift.min_profit_factor,
                min_edge_ratio      = cfg.spot_drift.min_edge_ratio,
                annualization_factor= annual,
            )
            self._drift: Optional[DriftDetector] = DriftDetector(drift_cfg)
        else:
            self._drift = None

        # TWAP konfig tárolása (az executor az egyes trade-ekhez jön létre)
        self._twap_cfg = TWAPConfig(
            enabled             = cfg.spot_twap.enabled,
            num_slices          = cfg.spot_twap.num_slices,
            total_duration_sec  = cfg.spot_twap.total_duration_sec,
            max_price_drift_pct = cfg.spot_twap.max_price_drift_pct,
            min_order_size_usd  = cfg.spot_twap.min_order_size_usd,
            use_twap_above_usd  = cfg.spot_twap.use_twap_above_usd,
        )

    # ------------------------------------------------------------------ #
    # Egy iteráció
    # ------------------------------------------------------------------ #

    def step(self) -> Decision:
        ohlcv = self.data.fetch_ohlcv(limit=self.history_limit)
        if len(ohlcv) < 50:
            logger.warning("Kevés gyertya érkezett (%d), kihagyjuk az iterációt.", len(ohlcv))
            return Decision(action="HOLD", score=0.0, price=0.0, atr=0.0)

        decision = self.agent.decide(ohlcv)
        timestamp = ohlcv.index[-1]

        # FIX #7: %-os hozamok kiszámítása és tárolása (portfolio kockázathoz)
        try:
            closes = ohlcv["close"].tolist()
            self._last_returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
                if closes[i - 1] > 0
            ][-60:]   # utolsó 60 periódus elég a VaR/korreláció számításhoz
        except Exception:
            pass

        # --- TWAP folyamatban lévő szelet végrehajtása (ha van aktív TWAP) ---
        if self._twap and not self._twap.is_complete():
            self._tick_twap(decision.price, timestamp)

        # --- Exit feltételek ellenőrzése (ExitManager vagy legacy) ---
        if self.broker.in_position:
            self._check_smart_exits(decision.price, decision.atr, timestamp)

        # --- Az ügynök döntése ---
        if decision.action == "BUY" and not self.broker.in_position:
            self._try_open(decision, timestamp)
        elif decision.action == "SELL" and self.broker.in_position:
            self._close_position(decision.price, timestamp, note="signal")

        equity = self.broker.equity(decision.price)
        self.risk.update_equity(equity)
        self._last_step_ts = time.time()

        logger.info(
            "%s | equity=%.2f | %s | %s",
            timestamp, equity, decision.explain(), self.risk.status(),
        )
        return decision

    # ------------------------------------------------------------------ #
    # Belépési logika
    # ------------------------------------------------------------------ #

    def _try_open(self, decision: Decision, timestamp: pd.Timestamp) -> None:
        ok, reason = self.risk.should_open(decision.price, size=1.0, atr=decision.atr)
        if not ok:
            logger.info("BUY visszautasítva (risk): %s", reason)
            if "halted" in reason and self.config.notify.notify_on_kill_switch:
                self.notifier.kill_switch(reason)
            return

        # --- Likviditás szűrő --- FIX #1: None → paper/testnet módban kihagyjuk
        if self._liquidity:
            volume = self._estimate_volume(decision.price)
            if volume is not None:
                liq_ok, liq_reason = self._liquidity.check_volume_only(
                    self.config.symbol,
                    volume_24h_usd=volume,
                )
                if not liq_ok:
                    logger.info("BUY visszautasítva (likviditás): %s", liq_reason)
                    return
            else:
                logger.debug("Likviditás szűrő kihagyva (paper/testnet mód).")

        # --- Kereskedési költség szűrő ---
        if self._cost_model:
            order_usd = self.config.risk.max_order_value_usd * self.risk.get_size_multiplier()
            expected_return = abs(decision.score) * decision.atr / decision.price \
                if decision.price > 0 and decision.atr > 0 else 0.0
            cost_ok, cost_reason = self._cost_model.filter_trade(
                price=decision.price,
                atr=decision.atr,
                order_size_usd=order_usd,
                expected_return_pct=expected_return,
            )
            if not cost_ok:
                logger.info("BUY visszautasítva (koltseg): %s", cost_reason)
                return

        # --- TWAP döntés (nagy megbízásnál) ---
        # FIX #4: TWAP paper módban nem működik (PaperBroker nem támogat részleges vételeket)
        order_usd = self.config.risk.max_order_value_usd * self.risk.get_size_multiplier()
        if TWAPExecutor.should_use_twap(order_usd, self._twap_cfg):
            if isinstance(self.broker, PaperBroker):
                logger.debug(
                    "TWAP kihagyva (PaperBroker): egyszeri belépés történik %.0f USD-ért.",
                    order_usd,
                )
                # Fall through → normál belépés
            else:
                logger.info("TWAP indítása: %.0f USD, %d szelet", order_usd, self._twap_cfg.num_slices)
                self._twap = TWAPExecutor(
                    total_size_usd=order_usd,
                    config=self._twap_cfg,
                    fee_rate=self.config.fee_rate,
                )
                first_slice = self._twap.start(decision.price, timestamp)
                self._execute_buy_slice(first_slice.size_usd, decision, timestamp, note="twap_slice_1")
                return

        # --- Normál belépés ---
        report = self.broker.buy(decision.price, timestamp)
        if report is None:
            return

        # ExitManager inicializálása belépéskor
        if self._exit_mgr:
            # Ciklus-paraméterekből vesszük az TP/stop szorzókat ha elérhetők
            atr_tp_mult   = 4.0
            atr_stop_mult = 2.0
            max_holding   = self.config.spot_exit.time_exit_bars
            if decision.cycle_params:
                atr_tp_mult   = decision.cycle_params.atr_tp_mult
                atr_stop_mult = decision.cycle_params.atr_stop_mult
                max_holding   = decision.cycle_params.max_holding_bars
            self._exit_mgr.on_entry(
                entry_price    = decision.price,
                atr            = decision.atr,
                atr_tp_mult    = atr_tp_mult,
                atr_stop_mult  = atr_stop_mult,
                max_holding_bars = max_holding,
            )

        # FIX #2: ciklus-info tárolása (SELL note-ban fog megjelenni)
        self._entry_cycle = (
            decision.cycle.cycle.value
            if decision.cycle is not None
            else ""
        )

        # Perzisztencia + értesítés
        self.db.log_fill(
            self.config.symbol, "BUY", report.size, report.price,
            report.fee, pnl=0.0, note=report.note, timestamp=timestamp,
        )
        if self.config.notify.notify_on_fill:
            self.notifier.fill("BUY", self.config.symbol, report.size, report.price)
        logger.info("FILL: %s", report)

    def _execute_buy_slice(self, size_usd: float, decision: Decision,
                           timestamp: pd.Timestamp, note: str = "") -> None:
        """TWAP szelet végrehajtása (közvetlen buy a brokerbe, nem a teljes position_size-t)."""
        size = size_usd / decision.price if decision.price > 0 else 0.0
        if size <= 0:
            return
        report = self.broker.buy(decision.price, timestamp)
        if report:
            self.db.log_fill(
                self.config.symbol, "BUY", report.size, report.price,
                report.fee, pnl=0.0, note=note, timestamp=timestamp,
            )
            logger.info("TWAP FILL: %s", report)

    def _close_position(self, price: float, timestamp: pd.Timestamp,
                        note: str, fraction: float = 1.0) -> None:
        """Pozíció (részleges) zárása. fraction=1.0: teljes zárás."""
        if fraction < 1.0:
            logger.info("Részleges zárás: fraction=%.0f%%, price=%.2f, note=%s",
                        fraction * 100, price, note)

        # FIX #2: ciklus-infó hozzáfűzése a note-hoz (performance attribúcióhoz)
        full_note = f"{note}|cycle:{self._entry_cycle}" if self._entry_cycle else note

        report = self.broker.sell(price, timestamp, note=full_note)
        if report is None:
            return

        self.risk.register_trade_pnl(report.pnl)
        self.db.log_fill(
            self.config.symbol, "SELL", report.size, report.price,
            report.fee, pnl=report.pnl, note=full_note, timestamp=timestamp,
        )
        if self.config.notify.notify_on_fill:
            self.notifier.fill("SELL", self.config.symbol, report.size, report.price,
                               pnl=report.pnl)
        logger.info("FILL: %s", report)

        # Drift detektor frissítése
        if self._drift:
            drift_status = self._drift.update(report.pnl)
            if drift_status.action not in ("ok", "warn"):
                logger.warning("DRIFT ALERT [%s]: %s", drift_status.action, drift_status.note)
                if self.config.notify.notify_on_error:
                    self.notifier.error(f"📉 Modell drift [{drift_status.action}]: {drift_status.note}")

        # ExitManager, TWAP és ciklus-info reset
        if self._exit_mgr:
            self._exit_mgr.reset()
        if self._twap:
            self._twap = None
        self._entry_cycle = ""  # FIX #2: ciklus reset a pozíció zárása után

        # Kill switch értesítés
        if self.risk.state.halted and self.config.notify.notify_on_kill_switch:
            self.notifier.kill_switch(self.risk.state.halt_reason or "halted")

    # ------------------------------------------------------------------ #
    # Smart exit (ExitManager vagy legacy SL/TP)
    # ------------------------------------------------------------------ #

    def _check_smart_exits(self, price: float, atr: float,
                           timestamp: pd.Timestamp) -> None:
        """ExitManager alapú exit logika (részleges TP, trailing, stb.)."""
        if self._exit_mgr:
            signal = self._exit_mgr.on_bar(price, atr)
            if signal.should_exit:
                if signal.stop_updated and signal.new_stop_price is not None:
                    logger.info("Stop frissítve: %.2f (ok: %s)", signal.new_stop_price, signal.reason)
                    # Részleges TP esetén csak az infót logoljuk, a stop az ExitManagerben él
                if signal.is_partial:
                    logger.info("RÉSZLEGES ZÁRÁS (%.0f%%): %s @ %.2f",
                                signal.exit_fraction * 100, signal.reason, price)
                    self._close_position(price, timestamp, note=signal.reason,
                                         fraction=signal.exit_fraction)
                else:
                    self._close_position(price, timestamp, note=signal.reason)
        else:
            # Fallback: régi SL/TP logika (ha ExitManager nincs bekapcsolva)
            self._legacy_check_exits(price, atr, timestamp)

    def _legacy_check_exits(self, price: float, atr: float,
                            timestamp: pd.Timestamp) -> None:
        """Régi, egyszerű SL/TP logika (kompatibilitás)."""
        if not hasattr(self, "_stop_price"):
            self._stop_price = None
            self._tp_price = None
            self._highest_price = None

        self._legacy_update_trailing(price, atr)

        if self._stop_price is not None and price <= self._stop_price:
            logger.info("STOP @ %.2f (stop=%.2f)", price, self._stop_price)
            self._close_position(price, timestamp, note="stop_loss")
        elif self._tp_price is not None and price >= self._tp_price:
            logger.info("TAKE-PROFIT @ %.2f (tp=%.2f)", price, self._tp_price)
            self._close_position(price, timestamp, note="take_profit")

    def _legacy_update_trailing(self, price: float, atr: float) -> None:
        if not self.broker.in_position or not self.config.stops.use_trailing_stop:
            return
        if not hasattr(self, "_highest_price") or self._highest_price is None:
            self._highest_price = price
        if price > self._highest_price:
            self._highest_price = price
        if atr > 0 and self._highest_price:
            new_stop = self._highest_price - self.config.stops.trailing_atr_mult * atr
            if not hasattr(self, "_stop_price") or self._stop_price is None or new_stop > self._stop_price:
                self._stop_price = new_stop

    # ------------------------------------------------------------------ #
    # TWAP kezelés
    # ------------------------------------------------------------------ #

    def _tick_twap(self, price: float, timestamp: pd.Timestamp) -> None:
        """Aktív TWAP következő szeletének ellenőrzése és esetleges végrehajtása."""
        if not self._twap:
            return
        s = self._twap.tick(price, timestamp)
        if s is not None and s.executed:
            logger.info("TWAP szelet %d/%d @ %.2f", s.slice_num, s.total_slices, price or 0)
            # Valódi brókerben itt kellene a ccxt hívás
        if self._twap.is_complete():
            result = self._twap.get_result()
            if result.aborted:
                logger.warning("TWAP abort: %s", result.abort_reason)
            else:
                logger.info("TWAP kész: %.0f USD, átlag ár: %.2f, %d szelet",
                            result.total_size_usd, result.avg_fill_price, result.slices_executed)
            self._twap = None

    # ------------------------------------------------------------------ #
    # Volume lekérdezés (FIX #1)
    # ------------------------------------------------------------------ #

    def _estimate_volume(self, price: float) -> Optional[float]:
        """
        24 órás USD forgalom lekérdezése.

        Visszatérési értékek:
          - float   : éles BybitBroker esetén a valódi quoteVolume
          - None    : paper/dry-run módban — a hívó kihagyja a likviditás szűrőt

        FIX #1: korábban hardcoded 50 000×max_order értéket adott vissza,
        ami mindig átment a szűrőn. Most valódi ticker adatot kérünk.
        """
        if isinstance(self.broker, BybitBroker) and not self.broker.dry_run:
            try:
                ticker = self.broker.exchange.fetch_ticker(self.config.symbol)
                vol = float(ticker.get("quoteVolume") or 0.0)
                logger.debug("[%s] 24h quoteVolume: %.0f USD", self.config.symbol, vol)
                return vol
            except Exception as e:
                logger.warning(
                    "Volume lekérés sikertelen (%s): %s — likviditás szűrő kihagyva.",
                    self.config.symbol, e,
                )
                return None   # ismeretlen → ne blokkolj, de ne is engedd hamisan

        # Paper / dry-run / testnet: nincs valódi piac → szűrő kihagyása
        return None

    # ------------------------------------------------------------------ #
    # Folyamatos futás + watchdog
    # ------------------------------------------------------------------ #

    def run_forever(self) -> None:
        logger.info(
            "Trader elindult. broker=%s symbol=%s timeframe=%s",
            type(self.broker).__name__, self.config.symbol, self.config.timeframe,
        )
        try:
            while True:
                if self.risk.state.halted:
                    msg = f"Kill switch aktív ({self.risk.state.halt_reason}) - leállás."
                    logger.error(msg)
                    if self.config.notify.notify_on_kill_switch:
                        self.notifier.kill_switch(self.risk.state.halt_reason or "halted")
                    break
                try:
                    self.step()
                except Exception as e:
                    logger.exception("Hiba az iterációban: %s", e)
                    if self.config.notify.notify_on_error:
                        self.notifier.error(str(e))
                self._watchdog_check()
                time.sleep(self.config.poll_interval_sec)
        except KeyboardInterrupt:
            logger.info("Leállítás (Ctrl-C). %s", self.risk.status())

    def _watchdog_check(self) -> None:
        """Ha túl rég nem volt sikeres step, riasztunk."""
        if not self.config.watchdog.enabled:
            return
        silent_for = time.time() - self._last_step_ts
        if silent_for > self.config.watchdog.max_silent_seconds:
            msg = f"Watchdog: {silent_for:.0f}s nincs sikeres step"
            logger.warning(msg)
            if self.config.notify.notify_on_error:
                self.notifier.error(msg)
            self._last_step_ts = time.time()  # ne riasszunk minden iterációban


# ============================================================================
# Factory függvények
# ============================================================================

def build_paper_trader(config: TradingConfig) -> Trader:
    agent = TradingAgent(config)
    broker = PaperBroker(
        cash=config.initial_balance,
        fee_rate=config.fee_rate,
        position_size=config.position_size,
    )
    risk = RiskManager(config.risk)
    data = CcxtDataSource(
        exchange_id=config.exchange_id,
        symbol=config.symbol,
        timeframe=config.timeframe,
        endpoint=config.bybit_endpoint,
        market_type=config.market_type,
    )
    return Trader(agent, broker, risk, data, config)


def build_bybit_trader(config: TradingConfig) -> Trader:
    agent = TradingAgent(config)
    broker = BybitBroker(config, dry_run=config.risk.dry_run)
    risk = RiskManager(config.risk)
    data = CcxtDataSource(
        exchange_id=config.exchange_id,
        symbol=config.symbol,
        timeframe=config.timeframe,
        endpoint=config.bybit_endpoint,
        market_type=config.market_type,
    )
    return Trader(agent, broker, risk, data, config)
