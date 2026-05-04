"""
Meta-labeling ML modell — Stacking Ensemble.

Architektúra (Lopez de Prado AFML + modern ML best practices):

  Level 0  (base learners — purged OOF predikciókat adnak):
    ① CatBoostClassifier  – ordered boosting, idősorhoz optimális,
                            csökkentett target leakage
    ② LGBMClassifier      – leaf-wise, gyors, pontos tabular adaton
    ③ XGBClassifier       – depth-wise, jó kis mintákon

  Level 1  (meta-learner — az OOF prob-okon tanul):
    LogisticRegression L2 – egyszerű, nem illeszkedik túl a stacken,
                            inherensen kalibrált valószínűségeket ad

  Purged walk-forward CV minden szinten → nincs target leakage a
  stacking szintjén sem (minden OOF pred az adott bar jövőjéből jön).

  Optuna hyperparameter tuning (use_optuna=True esetén):
    Mindhárom base learner külön study-val, purged CV mint belső loop.

  Mentés: joblib (nem pickle — biztonságosabb, numpy-optimált).

Fix-ek:
  #1  – nincs use_label_encoder=False (XGBoost 1.6+ kompatibilis)
  #3  – embargo_bars garantál min. max_holding méretű embargót
  #9  – joblib mentés pickle helyett
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_model")


# ============================================================================
# Konfiguráció
# ============================================================================

@dataclass
class MLConfig:
    model_path: str = "ml_model.joblib"    # .pkl → .joblib (#9)
    n_folds: int = 5
    embargo_pct: float = 0.01
    # Abszolút minimum embargo bars (#3 fix).
    # Állítsd legalább max_holding + 5-re, hogy az átfedő pozíciók ne
    # szivárogassanak info-t a training set-be.
    embargo_bars: int = 25
    min_train_size: int = 300

    # --- Base learner bekapcsolók ---
    use_catboost: bool = True
    use_lightgbm: bool = True
    use_xgboost: bool = True

    # --- Optuna hyperparameter tuning ---
    # False = gyors, konzervatív default paraméterek
    # True  = lassabb, de potenciálisan pontosabb (ajánlott prodban)
    use_optuna: bool = False
    n_trials: int = 50
    optuna_timeout_sec: int = 300          # max 5 perc modellenként

    # --- Default base model hiperparaméterek (use_optuna=False esetén) ---
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5


@dataclass
class MLScore:
    """Az ensemble kimenete egyetlen döntési ponthoz."""
    probability: float       # P(primary jel helyes) — 0..1
    bet_size: float          # javasolt pozícióméret-szorzó — 0..1
    feature_count: int = 0
    fitted: bool = False

    @property
    def is_confident(self) -> bool:
        return self.fitted and self.probability >= 0.55


# ============================================================================
# Helpers
# ============================================================================

def _purged_kfold_splits(
    n: int,
    n_folds: int,
    embargo_pct: float,
    embargo_bars: int = 25,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Idősorban walk-forward split garantált embargo zónával.

    Az embargo legalább embargo_bars sor (tipikusan max_holding + buffer),
    de legalább az adat 1%-a — amelyik a nagyobb.
    """
    embargo = max(embargo_bars, int(n * embargo_pct))
    fold_size = n // n_folds
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold in range(1, n_folds):
        test_start = fold * fold_size
        test_end   = test_start + fold_size if fold < n_folds - 1 else n
        train_end  = max(0, test_start - embargo)

        if train_end < 50:
            continue

        splits.append((
            np.arange(0, train_end),
            np.arange(test_start, test_end),
        ))

    return splits


def _prob_to_bet_size(prob: float, threshold: float = 0.5) -> float:
    """Lineáris Kelly-proxy: prob=0.5→0.0, prob=1.0→1.0."""
    if prob <= threshold:
        return 0.0
    return min(1.0, 2.0 * (prob - threshold))


def _check_library(name: str) -> bool:
    """Visszaadja, hogy a könyvtár importálható-e."""
    import importlib
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        logger.warning("%s nem elérhető (pip install %s)", name, name)
        return False


# ============================================================================
# Stacking Ensemble
# ============================================================================

class MetaLabelModel:
    """
    Stacking ensemble meta-labeling modell.

    Level 0: CatBoost (ordered) + LightGBM + XGBoost → purged OOF prob-ok
    Level 1: LogisticRegression az OOF prob-okon

    Felhasználás (backward-kompatibilis):
        model = MetaLabelModel(MLConfig(use_optuna=True))
        result = model.fit(X, triple_barrier_labels, primary_signals)
        score  = model.predict(X_row)
        if score.is_confident:
            size *= score.bet_size
    """

    def __init__(self, config: Optional[MLConfig] = None) -> None:
        self.config = config or MLConfig()
        # (name, fitted_clf, oof_accuracy) hármasok
        self._base_models: List[Tuple[str, object, float]] = []
        self._meta_clf = None           # LogisticRegression
        self._feats: List[str] = []
        self._fitted = False
        self._best_params: Dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Tanítás — fő belépési pont
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: pd.DataFrame,
        labels: pd.Series,
        primary_signals: pd.Series,
    ) -> Dict:
        """
        Stacking ensemble tanítása purged walk-forward CV-vel.

        Paraméterek:
            X               -- feature matrix
            labels          -- triple barrier label (+1 / -1 / 0)
            primary_signals -- elsődleges jelrendszer jelei (+1 / -1 / 0)
        """
        # Meta-label: egyezett-e a primary signal a tényleges kimenettel?
        same_sign  = np.sign(primary_signals) == np.sign(labels)
        has_signal = primary_signals != 0
        meta_y     = (has_signal & same_sign).astype(int)

        mask      = (primary_signals != 0).values
        X_f       = X[mask].reset_index(drop=True)
        y_f       = meta_y[mask].reset_index(drop=True)
        self._feats = list(X_f.columns)

        n = len(X_f)
        if n < self.config.min_train_size:
            logger.warning(
                "Kevés tanítóadat: %d sor (min: %d) — modell megbízhatatlan lehet.",
                n, self.config.min_train_size,
            )

        splits = _purged_kfold_splits(
            n, self.config.n_folds,
            self.config.embargo_pct,
            self.config.embargo_bars,
        )
        if not splits:
            raise ValueError("Nem jöttek létre purged CV splits. Növeld az adatmennyiséget.")

        # ── 1. Optuna tuning (opcionális) ─────────────────────────────
        if self.config.use_optuna:
            self._best_params = self._optuna_tune_all(X_f, y_f, splits)

        # ── 2. OOF predikciók base learnerenként ──────────────────────
        oof_probs: Dict[str, np.ndarray] = {}
        base_info: List[Dict] = []

        for maker_name, make_fn in self._active_makers():
            clf_params = self._best_params.get(maker_name, {})
            oof, acc, fold_accs = self._generate_oof(
                X_f, y_f, splits, make_fn, clf_params,
            )
            oof_probs[maker_name] = oof
            base_info.append({
                "name": maker_name,
                "oof_accuracy": acc,
                "fold_accuracies": fold_accs,
            })
            logger.info(
                "[%s] OOF accuracy: %.3f ± %.3f",
                maker_name, acc, float(np.std(fold_accs)),
            )

        if not oof_probs:
            raise RuntimeError(
                "Egy sem elérhető base learner. "
                "pip install catboost lightgbm xgboost"
            )

        # ── 3. Meta-learner tanítása OOF prob-okon ────────────────────
        oof_X = np.column_stack(list(oof_probs.values()))  # (n, n_models)
        self._meta_clf = self._fit_meta(oof_X, y_f.values)

        # ── 4. Base learnerek újratanítása teljes adaton ──────────────
        self._base_models = []
        for maker_name, make_fn in self._active_makers():
            clf_params = self._best_params.get(maker_name, {})
            clf = make_fn(**clf_params)
            clf.fit(X_f, y_f)
            oof_acc = next(
                (b["oof_accuracy"] for b in base_info if b["name"] == maker_name),
                0.0,
            )
            self._base_models.append((maker_name, clf, oof_acc))
            logger.info("[%s] Teljes adaton újratanítva.", maker_name)

        self._fitted = True

        # ── 5. CV statisztikák összesítése ────────────────────────────
        all_fold_accs = [acc for b in base_info for acc in b["fold_accuracies"]]
        return {
            "train_samples":   n,
            "positive_rate":   float(y_f.mean()),
            "n_features":      len(self._feats),
            "n_base_models":   len(self._base_models),
            "base_models":     base_info,
            "mean_accuracy":   float(np.mean(all_fold_accs)) if all_fold_accs else 0.0,
            "std_accuracy":    float(np.std(all_fold_accs))  if all_fold_accs else 0.0,
            "fold_accuracies": all_fold_accs,
        }

    # ------------------------------------------------------------------ #
    # Predikció
    # ------------------------------------------------------------------ #

    def predict(self, X_row: pd.DataFrame) -> MLScore:
        """Egyetlen gyertya meta-label valószínűsége az ensemble-ből."""
        if not self._fitted or not self._base_models:
            return MLScore(probability=0.5, bet_size=0.0, fitted=False)

        try:
            X_aligned = X_row.reindex(columns=self._feats, fill_value=0.0)
            base_probs = np.array([
                float(clf.predict_proba(X_aligned)[0][1])
                for _, clf, _ in self._base_models
            ]).reshape(1, -1)

            if self._meta_clf is not None and len(self._base_models) > 1:
                prob = float(self._meta_clf.predict_proba(base_probs)[0][1])
            else:
                # Egyetlen model: OOF-súlyozott átlag
                weights = np.array([acc for _, _, acc in self._base_models])
                weights = weights / (weights.sum() or 1.0)
                prob = float((base_probs[0] * weights).sum())

            bet = _prob_to_bet_size(prob)
            return MLScore(
                probability=prob,
                bet_size=bet,
                feature_count=len(self._feats),
                fitted=True,
            )
        except Exception as e:
            logger.warning("Ensemble predikció hiba: %s", e)
            return MLScore(probability=0.5, bet_size=0.0, fitted=False)

    # ------------------------------------------------------------------ #
    # Feature importance
    # ------------------------------------------------------------------ #

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        """
        OOF accuracy-val súlyozott átlagos feature importance.

        Minden base learner feature_importances_-ét normalizálja,
        majd OOF pontosságuk szerint súlyozza.
        """
        if not self._fitted or not self._base_models:
            return pd.Series(dtype=float)

        total_weight = sum(acc for _, _, acc in self._base_models) or 1.0
        agg = np.zeros(len(self._feats))

        for name, clf, acc in self._base_models:
            if not hasattr(clf, "feature_importances_"):
                continue
            imp = np.array(clf.feature_importances_, dtype=float)
            imp_sum = imp.sum()
            if imp_sum > 0:
                imp /= imp_sum
            agg += imp * (acc / total_weight)

        return (
            pd.Series(agg, index=self._feats)
            .sort_values(ascending=False)
            .head(top_n)
        )

    # ------------------------------------------------------------------ #
    # Mentés / betöltés — joblib (#9 fix)
    # ------------------------------------------------------------------ #

    def save(self, path: Optional[str] = None) -> None:
        try:
            import joblib
        except ImportError:
            raise ImportError("pip install scikit-learn  # tartalmazza a joblibet")

        p = Path(path or self.config.model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "base_models": self._base_models,
            "meta_clf":    self._meta_clf,
            "feats":       self._feats,
            "best_params": self._best_params,
        }
        joblib.dump(state, p, compress=3)
        logger.info("Ensemble modell mentve: %s (%d base learner)", p, len(self._base_models))

    def load(self, path: Optional[str] = None) -> bool:
        try:
            import joblib
        except ImportError:
            logger.error("joblib nem elérhető: pip install scikit-learn")
            return False

        p = Path(path or self.config.model_path)
        if not p.exists():
            logger.info("Nincs mentett modell: %s", p)
            return False
        try:
            state = joblib.load(p)
            self._base_models = state["base_models"]
            self._meta_clf    = state["meta_clf"]
            self._feats       = state["feats"]
            self._best_params = state.get("best_params", {})
            self._fitted      = True
            logger.info(
                "Ensemble modell betöltve: %s (%d base learner, %d feature)",
                p, len(self._base_models), len(self._feats),
            )
            return True
        except Exception as e:
            logger.warning("Modell betöltés sikertelen: %s", e)
            return False

    # ------------------------------------------------------------------ #
    # Base learner gyárak
    # ------------------------------------------------------------------ #

    def _active_makers(self) -> List[Tuple[str, callable]]:
        """Visszaadja az elérhető és bekapcsolt base learner gyárfüggvényeket."""
        candidates = []
        if self.config.use_catboost and _check_library("catboost"):
            candidates.append(("catboost", self._make_catboost))
        if self.config.use_lightgbm and _check_library("lightgbm"):
            candidates.append(("lightgbm", self._make_lightgbm))
        if self.config.use_xgboost and _check_library("xgboost"):
            candidates.append(("xgboost", self._make_xgboost))
        return candidates

    def _make_catboost(self, **kw) -> object:
        from catboost import CatBoostClassifier
        cfg = self.config
        params = dict(
            iterations      = kw.get("iterations", cfg.n_estimators),
            learning_rate   = kw.get("learning_rate", cfg.learning_rate),
            depth           = kw.get("depth", cfg.max_depth),
            l2_leaf_reg     = kw.get("l2_leaf_reg", 3.0),
            min_data_in_leaf= kw.get("min_data_in_leaf", cfg.min_child_weight * 2),
            boosting_type   = "Ordered",      # idősor-optimális
            eval_metric     = "Accuracy",
            auto_class_weights = "Balanced",
            random_seed     = 42,
            verbose         = 0,
        )
        return CatBoostClassifier(**params)

    def _make_lightgbm(self, **kw) -> object:
        from lightgbm import LGBMClassifier
        cfg = self.config
        return LGBMClassifier(
            n_estimators    = kw.get("n_estimators", cfg.n_estimators),
            learning_rate   = kw.get("learning_rate", cfg.learning_rate),
            num_leaves      = kw.get("num_leaves", 2 ** cfg.max_depth - 1),
            min_child_samples = kw.get("min_child_samples", cfg.min_child_weight * 4),
            subsample       = kw.get("subsample", cfg.subsample),
            colsample_bytree= kw.get("colsample_bytree", cfg.colsample_bytree),
            reg_lambda      = kw.get("reg_lambda", 1.0),
            class_weight    = "balanced",
            random_state    = 42,
            n_jobs          = -1,
            verbose         = -1,
        )

    def _make_xgboost(self, **kw) -> object:
        import xgboost as xgb
        cfg = self.config
        return xgb.XGBClassifier(
            n_estimators    = kw.get("n_estimators", cfg.n_estimators),
            max_depth       = kw.get("max_depth", cfg.max_depth),
            learning_rate   = kw.get("learning_rate", cfg.learning_rate),
            subsample       = kw.get("subsample", cfg.subsample),
            colsample_bytree= kw.get("colsample_bytree", cfg.colsample_bytree),
            min_child_weight= kw.get("min_child_weight", cfg.min_child_weight),
            reg_lambda      = kw.get("reg_lambda", 1.0),
            eval_metric     = "logloss",      # use_label_encoder eltávolítva (#1)
            scale_pos_weight= kw.get("scale_pos_weight", 1.0),
            random_state    = 42,
            n_jobs          = -1,
            verbosity       = 0,
        )

    def _fit_meta(self, oof_X: np.ndarray, y: np.ndarray) -> object:
        """LogisticRegression tanítása az OOF prob-mátrixon."""
        from sklearn.linear_model import LogisticRegression
        meta = LogisticRegression(
            C            = 1.0,
            class_weight = "balanced",
            max_iter     = 500,
            random_state = 42,
        )
        meta.fit(oof_X, y)
        return meta

    # ------------------------------------------------------------------ #
    # OOF generálás (egy base learnerhez)
    # ------------------------------------------------------------------ #

    def _generate_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        make_fn: callable,
        clf_params: dict,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Purged walk-forward OOF predikciók generálása.

        Returns:
            (oof_probs, mean_accuracy, fold_accuracies)
        """
        n = len(X)
        oof = np.full(n, 0.5)
        fold_accs: List[float] = []

        for tr_idx, te_idx in splits:
            clf = make_fn(**clf_params)
            clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            probs = clf.predict_proba(X.iloc[te_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            acc   = float((preds == y.iloc[te_idx].values).mean())
            fold_accs.append(acc)
            oof[te_idx] = probs

        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        return oof, mean_acc, fold_accs

    # ------------------------------------------------------------------ #
    # Optuna hyperparameter tuning
    # ------------------------------------------------------------------ #

    def _optuna_tune_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        splits: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, dict]:
        """Mindhárom base learner hyperparamétereit hangolja Optunával."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna nem elérhető (pip install optuna) — default params.")
            return {}

        best: Dict[str, dict] = {}

        tune_map = []
        if self.config.use_catboost and _check_library("catboost"):
            tune_map.append(("catboost", self._optuna_catboost_objective))
        if self.config.use_lightgbm and _check_library("lightgbm"):
            tune_map.append(("lightgbm", self._optuna_lgbm_objective))
        if self.config.use_xgboost and _check_library("xgboost"):
            tune_map.append(("xgboost", self._optuna_xgb_objective))

        for name, obj_factory in tune_map:
            logger.info("Optuna tuning: %s (%d trial, max %ds)...",
                        name, self.config.n_trials, self.config.optuna_timeout_sec)
            study = optuna.create_study(direction="maximize")
            study.optimize(
                obj_factory(X, y, splits),
                n_trials   = self.config.n_trials,
                timeout    = self.config.optuna_timeout_sec,
                show_progress_bar = False,
            )
            best[name] = study.best_params
            logger.info("  %s legjobb params: %s  (val_acc=%.4f)",
                        name, study.best_params, study.best_value)

        return best

    def _optuna_catboost_objective(self, X, y, splits):
        def objective(trial):
            params = dict(
                iterations       = trial.suggest_int("iterations", 100, 800),
                learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                depth            = trial.suggest_int("depth", 3, 8),
                l2_leaf_reg      = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 50),
            )
            _, acc, _ = self._generate_oof(X, y, splits, self._make_catboost, params)
            return acc
        return objective

    def _optuna_lgbm_objective(self, X, y, splits):
        def objective(trial):
            params = dict(
                n_estimators     = trial.suggest_int("n_estimators", 100, 800),
                learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                num_leaves       = trial.suggest_int("num_leaves", 8, 128),
                min_child_samples= trial.suggest_int("min_child_samples", 10, 80),
                subsample        = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda       = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            )
            _, acc, _ = self._generate_oof(X, y, splits, self._make_lightgbm, params)
            return acc
        return objective

    def _optuna_xgb_objective(self, X, y, splits):
        def objective(trial):
            params = dict(
                n_estimators     = trial.suggest_int("n_estimators", 100, 800),
                learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                max_depth        = trial.suggest_int("max_depth", 2, 8),
                min_child_weight = trial.suggest_int("min_child_weight", 1, 20),
                subsample        = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda       = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            )
            _, acc, _ = self._generate_oof(X, y, splits, self._make_xgboost, params)
            return acc
        return objective
