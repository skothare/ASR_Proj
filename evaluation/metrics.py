"""
evaluation/metrics.py
======================
Evaluation functions used at every AL iteration and in the final report.

WHY AUPRC IS THE PRIMARY METRIC
---------------------------------
With 96.5% inactive compounds, AUROC can look deceptively good (a model that
predicts everything as inactive still gets AUROC > 0.5).  AUPRC is much more
informative under severe class imbalance because it focuses on the minority
(active) class: precision = how many of your predicted actives are truly active;
recall = how many of all true actives you've found.

A random classifier achieves AUPRC ≈ prevalence ≈ 0.035 on HIV.
Your teammate's RF achieves ~0.47–0.50.  This is the meaningful bar.

HIT RECOVERY RATE
-----------------
An additional metric particularly suited to drug discovery active learning:
after labelling k molecules, what fraction of all true actives have been
recovered?  This directly answers: "did active learning find actives faster
than random labelling would?"
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalResult:
    """All metrics from a single evaluation call, bundled together."""
    auprc: float
    auroc: float
    n_labeled: int        # how many molecules were in the training set
    hit_recovery: Optional[float] = None  # fraction of all actives found


def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_labeled: int,
    total_actives_in_pool: Optional[int] = None,
    labeled_y: Optional[np.ndarray] = None,
) -> EvalResult:
    """
    Compute AUPRC, AUROC, and (optionally) hit recovery rate.

    Parameters
    ----------
    model         : any model with .predict_proba(X) method
    X_test        : test fingerprints
    y_test        : test labels
    n_labeled     : number of molecules in current training set (for x-axis)
    total_actives_in_pool : total actives in the full training pool
                            (needed for hit_recovery; pass None to skip)
    labeled_y     : labels of the currently labeled training set
                    (needed for hit_recovery)
    """
    proba = model.predict_proba(X_test)[:, 1]   # P(active)

    # ── AUPRC ─────────────────────────────────────────────────────────────────
    # average_precision_score computes the area under the precision-recall curve
    # using the trapezoidal rule.  It is equivalent to AUPRC.
    auprc = float(average_precision_score(y_test, proba))

    # ── AUROC ─────────────────────────────────────────────────────────────────
    # Only defined when both classes present; guard for early AL iterations
    # where the labelled set might be all-inactive.
    try:
        auroc = float(roc_auc_score(y_test, proba))
    except ValueError:
        auroc = float("nan")

    # ── Hit recovery rate ─────────────────────────────────────────────────────
    hit_recovery = None
    if total_actives_in_pool is not None and labeled_y is not None:
        actives_found = int(labeled_y.sum())
        if total_actives_in_pool > 0:
            hit_recovery = actives_found / total_actives_in_pool

    return EvalResult(
        auprc=auprc,
        auroc=auroc,
        n_labeled=n_labeled,
        hit_recovery=hit_recovery,
    )
