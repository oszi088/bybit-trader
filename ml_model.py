"""
Meta-labeling ML modell (Lopez de Prado, AFML 10. fejezet).

A meta-labeling lényege:
  1. Az elsődleges modell (meglévő súlyozott szavazó) ad egy irányjelet.
  2. Az ML modell CSAK azt tanulja: az adott piaci helyzetben az elsődleges
     jel megbízható-e? (bináris: 1 = igen, 0 = ne menj bele)

Miért jobb ez mint egyenesen az irányt tanítani?
  * Az irány-predikció nagyon zajos (közel 50/50 random walk)
  * A "mikor megbízható a jelrendszerem?" kérdés pontosabban megtanulható
  * Az elsődleges jelrendszer logikája megmarad — az ML csak szűr + méretez

Purged Walk-Forward CV:
  * A train/test határán embargo zóna van
  * Megelőzi, hogy az átfedő nyitott pozíciók label-információt
    szivárogtassanak a training set-be (ez a klasszikus backteszt-overfit forrása)
"""

from __future__ import annotations

import logging
import pickle
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
    model_path: str = "ml_model.pkl"
    n_folds: int = 5
    # Embargo: a fold határán ennyi % adatot kihagyunk a train-ből,
    # hogy az átfedő pozíciók ne szivárogjon info a training set-be
    embargo_pct: float = 0.01
    min_train_size: int = 300
    # XGBoost hiperparaméterek (szándékosan konzervatív, overfit ellen)
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    min_child_weight: int = 5


@dataclass
class MLScore:
    """Az ML modell kimenete egyetlen döntési ponthoz."""
    probability: float   # P(primary jel helyes) — 0..1
    bet_size: float      # javasolt pozícióméret-szorzó — 0..1
    feature_count: int = 0
    fitted: bool = False

    @property
    def is_confident(self) -> bool:
        """Csak akkor érdemes belépni, ha az ML elég magabiztos."""
        return self.fitted and self.probability >= 0.55


# ============================================================================
# Purged Walk-Forward K-Fold
# ============================================================================

def _purged_kfold_splits(
    n: int,
    n_folds: int,
    embargo_pct: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Idősorban walk-forward split embargo zónával.

    Minden fold:
      train = az idősorban korábbi adatok (embargo nélküli rész)
      test  = a fold maga
      embargo = test_start előtt [embargo] sor kihagyva a train-ből
    """
    embargo    = max(1, int(n * embargo_pct))
    fold_size  = n // n_folds
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold in range(1, n_folds):          # fold=0 nem jó: nincs előtte train
        test_start = fold * fold_size
        test_end   = test_start + fold_size if fold < n_folds - 1 else n
        train_end  = max(0, test_start - embargo)

        if train_end < 50:
            continue

        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


# ============================================================================
# Bet sizing: Kelly-szerű valószínűség → pozícióméret
# ============================================================================

def _prob_to_bet_size(prob: float, threshold: float = 0.5) -> float:
    """
    Lineáris Kelly-proxy:
      prob = 0.50 -> bet = 0.0  (semleges)
      prob = 0.75 -> bet = 0.5
      prob = 1.00 -> bet = 1.0
    """
    if prob <= threshold:
        return 0.0
    return min(1.0, 2.0 * (prob - threshold))


# ============================================================================
# A modell
# ============================================================================

class MetaLabelModel:
    """
    XGBoost meta-labeling modell.

    Tanítás:
        result = model.fit(X, triple_barrier_labels, primary_signals)

    Élő predikció:
        score = model.predict(X_row)
        if score.is_confident:
            position_size *= score.bet_size
    """

    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self._clf    = None
        self._feats: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Tanítás
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: pd.DataFrame,
        labels: pd.Series,
        primary_signals: pd.Series,
    ) -> Dict:
        """
        Tanítás purged walk-forward CV-vel.

        Paraméterek:
            X               -- feature matrix (build_feature_matrix kimenete)
            labels          -- triple barrier label (+1 / -1 / 0)
            primary_signals -- az elsődleges rendszer jelei (+1 / -1 / 0)

        Meta-label definíció:
            1 ha a primary signal és a triple barrier label azonos előjelű
            0 egyébként (rossz jel, vagy időlimites döntetlen)
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("pip install xgboost")

        # Meta-label: egyezett-e a primary signal a tényleges kimenettel?
        same_sign   = np.sign(primary_signals) == np.sign(labels)
        has_signal  = primary_signals != 0
        meta_labels = (has_signal & same_sign).astype(int)

        # Csak olyan sorokon tanítunk ahol volt primary signal
        mask       = (primary_signals != 0).values
        X_f        = X[mask].copy()
        y_f        = meta_labels[mask].copy()
        self._feats = list(X_f.columns)

        if len(X_f) < self.config.min_train_size:
            logger.warning(
                "Kevés tanítóadat: %d sor (minimum: %d) — modell lehet megbízhatatlan",
                len(X_f), self.config.min_train_size,
            )

        n      = len(X_f)
        splits = _purged_kfold_splits(n, self.config.n_folds, self.config.embargo_pct)
        fold_accs: List[float] = []

        for i, (tr_idx, te_idx) in enumerate(splits):
            clf = self._make_clf(xgb)
            clf.fit(X_f.iloc[tr_idx], y_f.iloc[tr_idx])
            acc = float((clf.predict(X_f.iloc[te_idx]) == y_f.iloc[te_idx]).mean())
            fold_accs.append(acc)
            pos_rate = float(y_f.iloc[te_idx].mean())
            logger.info("Fold %d/%d  acc=%.3f  pos_rate=%.3f",
                        i + 1, len(splits), acc, pos_rate)

        # Végső modell: összes adat
        self._clf = self._make_clf(xgb)
        self._clf.fit(X_f, y_f)
        self._fitted = True

        return {
            "fold_accuracies":  fold_accs,
            "mean_accuracy":    float(np.mean(fold_accs)) if fold_accs else 0.0,
            "std_accuracy":     float(np.std(fold_accs))  if fold_accs else 0.0,
            "train_samples":    len(X_f),
            "positive_rate":    float(y_f.mean()),
            "n_features":       len(self._feats),
        }

    def _make_clf(self, xgb):
        return xgb.XGBClassifier(
            n_estimators      = self.config.n_estimators,
            max_depth         = self.config.max_depth,
            learning_rate     = self.config.learning_rate,
            subsample         = self.config.subsample,
            colsample_bytree  = self.config.colsample_bytree,
            min_child_weight  = self.config.min_child_weight,
            eval_metric       = "logloss",
            use_label_encoder = False,
            verbosity         = 0,
        )

    # ------------------------------------------------------------------ #
    # Predikció
    # ------------------------------------------------------------------ #

    def predict(self, X_row: pd.DataFrame) -> MLScore:
        """
        Egy sor (= egy gyertya) meta-label valószínűsége.

        Visszatér: MLScore — ha is_confident=True, érdemes belépni
        és bet_size-szal skálázni a pozíciót.
        """
        if not self._fitted or self._clf is None:
            return MLScore(probability=0.5, bet_size=0.0, fitted=False)

        try:
            X_aligned = X_row.reindex(columns=self._feats, fill_value=0.0)
            prob      = float(self._clf.predict_proba(X_aligned)[0][1])
            bet       = _prob_to_bet_size(prob)
            return MLScore(probability=prob, bet_size=bet,
                           feature_count=len(self._feats), fitted=True)
        except Exception as e:
            logger.warning("ML predikció hiba: %s", e)
            return MLScore(probability=0.5, bet_size=0.0, fitted=False)

    # ------------------------------------------------------------------ #
    # Feature importance
    # ------------------------------------------------------------------ #

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        """Top N legfontosabb feature MDI (Mean Decrease Impurity) alapján."""
        if not self._fitted or self._clf is None:
            return pd.Series(dtype=float)
        return (
            pd.Series(self._clf.feature_importances_, index=self._feats)
            .sort_values(ascending=False)
            .head(top_n)
        )

    # ------------------------------------------------------------------ #
    # Perzisztencia
    # ------------------------------------------------------------------ #

    def save(self, path: Optional[str] = None) -> None:
        p = Path(path or self.config.model_path)
        with open(p, "wb") as f:
            pickle.dump({"clf": self._clf, "feats": self._feats}, f)
        logger.info("ML modell mentve: %s", p)

    def load(self, path: Optional[str] = None) -> bool:
        p = Path(path or self.config.model_path)
        if not p.exists():
            logger.info("Nincs mentett ML modell: %s", p)
            return False
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            self._clf    = data["clf"]
            self._feats  = data["feats"]
            self._fitted = True
            logger.info("ML modell betöltve: %s (%d feature)", p, len(self._feats))
            return True
        except Exception as e:
            logger.warning("ML modell betöltés sikertelen: %s", e)
            return False
