"""
train.py — ML modell tanítási belépési pont.

Futtatás:
    python train.py --csv data/BTCUSDT_1h.csv
    python train.py --csv data/BTCUSDT_1h.csv --optuna --trials 50
    python train.py --csv data/BTCUSDT_1h.csv --no-catboost  # csak LGBM+XGB

Lépések:
    1. Függőségek ellenőrzése
    2. Historikus OHLCV adat betöltése CSV-ből
    3. Historikus F&G adat letöltése (alternative.me, cache: data/fear_greed_history.csv)
    4. Triple barrier labeling (ATR-alapú TP/SL/időlimit)
    5. Feature matrix építés
    6. Elsődleges jelrendszer futtatása (lookahead-mentes F&G)
    7. Stacking ensemble tanítás purged walk-forward CV-vel
       Level 0: CatBoost (ordered) + LightGBM + XGBoost
       Level 1: LogisticRegression az OOF prob-okon
    8. Modell mentése (.joblib)

Szükséges csomagok:
    pip install catboost lightgbm xgboost optuna scikit-learn joblib
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Függőség-ellenőrzés
# ---------------------------------------------------------------------------

REQUIRED = {
    "xgboost":     "pip install xgboost",
    "lightgbm":    "pip install lightgbm",
    "catboost":    "pip install catboost",
    "optuna":      "pip install optuna          # csak --optuna esetén kötelező",
    "sklearn":     "pip install scikit-learn",
    "joblib":      "pip install joblib           # általában a scikit-learn hozza",
}

def check_deps(need_optuna: bool = False) -> bool:
    missing = []
    skip    = {"optuna"} if not need_optuna else set()
    for pkg, hint in REQUIRED.items():
        if pkg in skip:
            continue
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"  {pkg:12s}  →  {hint}")
    if missing:
        logger.error("Hiányzó csomagok:\n%s", "\n".join(missing))
        return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stacking ensemble meta-label modell tanítása",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Adat ---
    p.add_argument("--csv",        required=True,
                   help="OHLCV CSV fájl elérési útja (pl. data/BTCUSDT_1h.csv)")
    p.add_argument("--symbol",     default="BTC/USDT",
                   help="Kereskedési pár (csak logoláshoz)")
    p.add_argument("--timeframe",  default="1h",
                   help="Gyertya timeframe (pl. 1h, 4h, 1d)")

    # --- Modell kimenet ---
    p.add_argument("--out",        default="ml_model.joblib",
                   help="Kimeneti modell fájl neve")

    # --- Triple barrier ---
    p.add_argument("--max-holding", type=int, default=20,
                   help="Maximális tartási idő gyertyában (embargo alap)")

    # --- Jel küszöb ---
    p.add_argument("--threshold", type=float, default=0.20,
                   help="Elsődleges jel küszöb (buy_threshold). "
                        "0.50 nagyon szűrős 1h-n, 0.20 kb. 300+ signal")

    # --- Ensemble bekapcsolók ---
    p.add_argument("--no-catboost",  action="store_true",
                   help="CatBoost kihagyása az ensemble-ből")
    p.add_argument("--no-lightgbm",  action="store_true",
                   help="LightGBM kihagyása az ensemble-ből")
    p.add_argument("--no-xgboost",   action="store_true",
                   help="XGBoost kihagyása az ensemble-ből")

    # --- Optuna ---
    p.add_argument("--optuna",       action="store_true",
                   help="Optuna hyperparameter tuning bekapcsolása (lassabb, pontosabb)")
    p.add_argument("--trials",       type=int, default=50,
                   help="Optuna trial-ok száma modellenként")
    p.add_argument("--timeout",      type=int, default=300,
                   help="Optuna max másodperc modellenként")

    # --- CV ---
    p.add_argument("--folds",        type=int, default=5,
                   help="Walk-forward CV fold-ok száma")

    # --- Feature importance ---
    p.add_argument("--top-features", type=int, default=20,
                   help="Top N feature importance megjelenítése")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Fő futtatás
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ML TANÍTÁS INDUL")
    logger.info("  CSV:         %s", args.csv)
    logger.info("  Symbol:      %s", args.symbol)
    logger.info("  Timeframe:   %s", args.timeframe)
    logger.info("  Max holding: %d gyertya", args.max_holding)
    logger.info("  Optuna:      %s", "igen" if args.optuna else "nem")
    logger.info("  Kimenet:     %s", args.out)
    logger.info("=" * 60)

    # --- 1. Függőségek ---
    if not check_deps(need_optuna=args.optuna):
        sys.exit(1)

    # --- 2. CSV ellenőrzés ---
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV fájl nem található: %s", csv_path)
        logger.error(
            "Tölts le historikus adatot:\n"
            "  python fetch_history.py --symbol %s --timeframe %s",
            args.symbol, args.timeframe,
        )
        sys.exit(1)

    # --- 3. Config összeállítás ---
    from config import TradingConfig
    from ml_model import MLConfig, MetaLabelModel
    from ml_train import run_training

    cfg = TradingConfig()
    cfg.symbol         = args.symbol
    cfg.timeframe      = args.timeframe
    cfg.buy_threshold  = args.threshold
    cfg.sell_threshold = -args.threshold

    ml_cfg = MLConfig(
        model_path          = args.out,
        n_folds             = args.folds,
        embargo_bars        = args.max_holding + 5,
        use_catboost        = not args.no_catboost,
        use_lightgbm        = not args.no_lightgbm,
        use_xgboost         = not args.no_xgboost,
        use_optuna          = args.optuna,
        n_trials            = args.trials,
        optuna_timeout_sec  = args.timeout,
    )

    active = [
        n for n, on in [
            ("CatBoost", ml_cfg.use_catboost),
            ("LightGBM", ml_cfg.use_lightgbm),
            ("XGBoost",  ml_cfg.use_xgboost),
        ] if on
    ]
    logger.info("Aktív base learnerek: %s", " + ".join(active))
    if len(active) < 2:
        logger.warning(
            "Stacking legalább 2 base learnerrel optimális. "
            "Jelenleg csak: %s", active
        )

    # --- 4. Tanítás ---
    try:
        model = run_training(
            csv_path        = str(csv_path),
            config          = cfg,
            ml_config       = ml_cfg,
            model_out       = args.out,
            max_holding     = args.max_holding,
            top_n_features  = args.top_features,
        )
    except KeyboardInterrupt:
        logger.info("Megszakítva (Ctrl-C).")
        sys.exit(0)
    except Exception as e:
        logger.exception("Tanítási hiba: %s", e)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("KÉSZ — modell elmentve: %s", args.out)
    logger.info(
        "Betöltés élő kereskedéshez:\n"
        "  from ml_model import MetaLabelModel\n"
        "  model = MetaLabelModel()\n"
        "  model.load('%s')",
        args.out,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
