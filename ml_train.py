"""
ML tanítási pipeline.

Munkafolyamat:
  1. CSV betöltés (fetch_history.py-jal letöltött adat)
  2. Indikátorok számítása
  3. Triple barrier labeling (ATR-alapú TP/SL/időlimit)
  4. Feature matrix építés
  5. Elsődleges jelrendszer futtatása (lookahead-mentes F&G)
  6. Meta-label: primary_signal == triple_barrier_label?
  7. Stacking ensemble tanítás purged walk-forward CV-vel
  8. Modell mentés (joblib)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agent import TradingAgent
from config import TradingConfig
from data_source import load_csv
from fear_greed_history import FearGreedHistory
from ml_features import build_feature_matrix
from ml_model import MLConfig, MetaLabelModel
from triple_barrier import make_labels

logger = logging.getLogger("ml_train")


def run_training(
    csv_path: str,
    config: Optional[TradingConfig] = None,
    ml_config: Optional[MLConfig] = None,
    model_out: str = "ml_model.joblib",
    max_holding: int = 20,
    top_n_features: int = 20,
) -> MetaLabelModel:
    """
    Teljes tanítási pipeline egy CSV fájlból.

    Visszatér a betanított MetaLabelModel példánnyal.
    """
    cfg    = config    or TradingConfig()
    # embargo_bars automatikusan max_holding + 5 buffer, hogy az átfedő
    # pozíciók ne szivárogassanak info-t a training set-be (#3 fix).
    if ml_config is None:
        ml_cfg = MLConfig(
            model_path   = model_out,
            embargo_bars = max_holding + 5,
        )
    else:
        ml_cfg = ml_config
        if ml_cfg.embargo_bars < max_holding:
            logger.warning(
                "embargo_bars=%d < max_holding=%d — automatikus korrekció.",
                ml_cfg.embargo_bars, max_holding,
            )
            ml_cfg.embargo_bars = max_holding + 5

    # 1. Adat betöltés
    logger.info("Adatok betöltése: %s", csv_path)
    ohlcv = load_csv(csv_path)
    logger.info("  %d gyertya, %s → %s", len(ohlcv),
                ohlcv.index.min(), ohlcv.index.max())

    # 2. Indikátorok + ATR
    from indicators import compute_all
    enriched = compute_all(ohlcv, cfg.indicators)
    atr      = enriched["atr"]

    # 3. Triple barrier labelek
    logger.info("Triple barrier labeling (max_holding=%d gyertya)...", max_holding)
    barrier_df = make_labels(ohlcv, atr, cfg.stops, max_holding=max_holding)
    label_dist = barrier_df["label"].value_counts().to_dict()
    logger.info("  Label eloszlás: +1=%d  0=%d  -1=%d",
                label_dist.get(1, 0), label_dist.get(0, 0), label_dist.get(-1, 0))

    # 4. Feature matrix
    logger.info("Feature matrix építése...")
    X = build_feature_matrix(ohlcv, cfg.indicators)
    logger.info("  %d sor × %d feature", *X.shape)

    # 5. Elsődleges jelrendszer futtatása minden gyertyán
    logger.info("Elsődleges jelrendszer futtatása...")

    # Historikus F&G betöltése — lookahead bias megelőzése.
    # A training loopban minden gyertyához a saját napjának F&G értékét
    # keressük, NEM a live API aktuális értékét.
    logger.info("Historikus Fear & Greed adatok betöltése...")
    fg_history = FearGreedHistory.load()
    date_min, date_max = fg_history.date_range()
    if len(fg_history) > 0:
        logger.info("  F&G history: %d nap (%s → %s)", len(fg_history), date_min, date_max)
    else:
        logger.warning("  F&G history üres — fallback 50 (Neutral) minden gyertyán.")

    agent = TradingAgent(cfg, fg_history=fg_history)
    agent.prepare(ohlcv)

    primary = np.zeros(len(ohlcv), dtype=np.int8)
    _errors = 0
    for i in range(len(ohlcv)):
        try:
            dec = agent.decide_at(i)
            if dec.action == "BUY":
                primary[i] = 1
            elif dec.action == "SELL":
                primary[i] = -1
        except Exception as exc:
            _errors += 1
            if _errors <= 5:
                logger.warning("decide_at(%d) hiba: %s", i, exc)

    if _errors > 0:
        logger.warning("Összesen %d hiba a signal generálásnál (%d sor)", _errors, len(ohlcv))

    n_signals = int((primary != 0).sum())
    if n_signals == 0:
        raise RuntimeError(
            "Egyetlen primary signal sem keletkezett (mind HOLD). "
            "Ellenőrizd az agent konfigurációt és az adatot."
        )
    if n_signals < 50:
        logger.warning("Csak %d signal (ajánlott >300) — modell valószínűleg megbízhatatlan.", n_signals)

    primary_series = pd.Series(primary, index=ohlcv.index, name="primary")
    signal_dist    = pd.Series(primary).value_counts().to_dict()
    logger.info("  Jelek: BUY=%d  HOLD=%d  SELL=%d",
                signal_dist.get(1, 0), signal_dist.get(0, 0), signal_dist.get(-1, 0))

    # 6. Tanítás
    logger.info(
        "Stacking ensemble tanítás (%d fold, embargo=%d bar / %.1f%%)...",
        ml_cfg.n_folds, ml_cfg.embargo_bars, ml_cfg.embargo_pct * 100,
    )
    model = MetaLabelModel(ml_cfg)
    result = model.fit(X, barrier_df["label"], primary_series)

    logger.info("=== TANÍTÁS EREDMÉNY ===")
    logger.info("  Train sorok:    %d", result["train_samples"])
    logger.info("  Pozitív arány:  %.1f%%", result["positive_rate"] * 100)
    logger.info("  Base learnerek: %d", result["n_base_models"])
    logger.info("  CV accuracy:    %.3f ± %.3f",
                result["mean_accuracy"], result["std_accuracy"])
    for base in result.get("base_models", []):
        logger.info("    [%s] OOF acc=%.3f", base["name"], base["oof_accuracy"])

    # 7. Feature importance
    imp = model.feature_importance(top_n=top_n_features)
    if not imp.empty:
        logger.info("Top %d feature (MDI):", top_n_features)
        for feat, score in imp.items():
            logger.info("  %-30s %.4f", feat, score)

    # 8. Mentés
    model.save(model_out)
    logger.info("Modell mentve: %s", model_out)

    return model
