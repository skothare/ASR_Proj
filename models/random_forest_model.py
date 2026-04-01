"""
models/random_forest_model.py
==============================
Random Forest base learner wrapper for pool-based active learning.

WHY A WRAPPER CLASS?
--------------------
The active learning loop (al_loop.py) calls `.fit()`, `.predict_proba()`, and
`.uncertainty()` on whatever base learner it receives.  Wrapping sklearn's RF
in this interface means we can swap in the MPNN later without changing al_loop.

UNCERTAINTY ESTIMATION FOR RF
------------------------------
We cannot use the softmax output as the model's
confidence.  For Random Forest the principled uncertainty signal is:

    entropy of the per-tree vote distribution

Each tree votes 0 or 1.  Let p = fraction of trees voting "active" (class 1).
Then Shannon entropy  H = -p·log(p) - (1-p)·log(1-p)  is highest when
p ≈ 0.5 (maximum disagreement) and lowest when all trees agree (p ≈ 0 or 1).

This is *epistemic* uncertainty: it measures disagreement in the model
ensemble, not just proximity to the decision boundary.

Your teammate already uses this in uncertainty sampling (Section 3.6.1 of
the milestone), so this is consistent with existing code.

CLASS IMBALANCE HANDLING
------------------------
With 96.5% inactive compounds, a naive RF will predict "inactive" for nearly
everything and achieve 96.5% accuracy while being useless.

We use class_weight='balanced' which automatically sets the weight of class i
to  n_samples / (n_classes * n_samples_in_class_i).  For HIV this amplifies
the active class by a factor of roughly 27×.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone


class RandomForestModel:
    """
    Thin wrapper around sklearn RandomForestClassifier.

    Parameters
    ----------
    n_estimators : int   – number of trees (100 is usually enough; your
                           teammate found performance plateaus around 100-200)
    max_depth    : int   – tree depth limit (None = grow fully; use 20-30 to
                           prevent severe overfitting on small initial sets)
    seed         : int   – random state for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.seed         = seed

        self._model = RandomForestClassifier(
            n_estimators  = n_estimators,
            max_depth     = max_depth,
            class_weight  = "balanced",   # handles 96/4 imbalance
            n_jobs        = -1,           # use all CPU cores (important on Colab)
            random_state  = seed,
        )

    # ── standard sklearn interface ────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Train on the current labelled set."""
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities shape (N, 2).
        Column 0 = P(inactive), column 1 = P(active).
        """
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard 0/1 predictions."""
        return self._model.predict(X)

    # ── uncertainty interface (used by al_loop.py) ────────────────────────────

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-molecule epistemic uncertainty = Shannon entropy of the
        fraction of trees voting 'active'.

        Returns shape (N,)  float32  in range [0, log(2) ≈ 0.693]
        High value → the trees strongly disagree → query this molecule.
        """
        proba = self.predict_proba(X)[:, 1]   # P(active) from each tree's vote
        # Clip to avoid log(0)
        p = np.clip(proba, 1e-9, 1 - 1e-9).astype(np.float64)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return entropy.astype(np.float32)

    def clone_untrained(self) -> "RandomForestModel":
        """Return a fresh (untrained) copy with the same hyperparameters."""
        return RandomForestModel(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            seed=self.seed,
        )
