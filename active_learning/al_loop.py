"""
active_learning/al_loop.py
===========================
Core active learning simulation loop.

HOW THE LOOP WORKS — STEP BY STEP
----------------------------------
This mirrors exactly what your teammate described in Section 3.5-3.6 of the
milestone, but is now a standalone function that accepts any initialization
strategy and any base learner.

  Given:
    - Full training pool   (N molecules, all with known labels in simulation)
    - Fixed test set       (never modified)
    - Initial labelled set (chosen by DOE or random strategy)
    - Batch size b         (number of molecules to query per iteration)

  Repeat until pool is exhausted or budget reached:
    1. Train model on current labelled set
    2. Evaluate on test set  → record AUPRC, AUROC
    3. Run query strategy on the *unlabelled* pool → pick b molecules
    4. "Label" them (in simulation: just look up their true y values)
    5. Move them from unlabelled pool to labelled set

  Key point on "labelling":
    In a real drug discovery setting, step 4 would mean running an assay
    (expensive, takes weeks).  In simulation, we already have all labels, so
    we just reveal them.  The simulation is still informative because the
    MODEL only sees the labelled subset — it has to generalise to the rest.

QUERY STRATEGY: UNCERTAINTY SAMPLING
--------------------------------------
We select the b molecules from the unlabelled pool that the current model is
MOST UNCERTAIN about (highest entropy of the RF vote distribution).

This is the active learning acquisition function.  The DOE comparison only
changes the *initialisation*, not this step — so any difference in the AUPRC
curves is purely attributable to the starting set.

PARAMETERS
----------
model             : base learner (must have .fit, .predict_proba, .uncertainty,
                    .clone_untrained methods)
X_pool, y_pool    : full training pool (fingerprints + labels)
X_test, y_test    : held-out test set (never touched during AL)
init_indices      : np.ndarray  – indices selected by initialisation strategy
batch_size        : int  – molecules to add per AL iteration (500 from your
                            teammate's grid search)
seed              : int  – for reproducibility

RETURNS
-------
List[EvalResult]  – one entry per AL iteration (for plotting)
"""

import numpy as np
from typing import List

from evaluation.metrics import evaluate, EvalResult


def run_active_learning(
    model,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    init_indices: np.ndarray,
    batch_size: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> List[EvalResult]:
    """
    Run the full active learning simulation and return per-iteration metrics.
    """
    N = len(y_pool)
    total_actives_in_pool = int(y_pool.sum())

    # ── Initialise labelled / unlabelled bookkeeping ──────────────────────────
    # We track which pool indices are labelled using a boolean mask.
    labelled_mask = np.zeros(N, dtype=bool)
    labelled_mask[init_indices] = True

    results: List[EvalResult] = []
    iteration = 0

    if verbose:
        n_init_active = y_pool[init_indices].sum()
        print(f"\nAL loop starting")
        print(f"  Init set : {labelled_mask.sum():,} molecules  "
              f"({n_init_active} actives = "
              f"{100*n_init_active/labelled_mask.sum():.1f}%)")
        print(f"  Pool size: {N:,}   Batch size: {batch_size}")
        print(f"  Max iterations: {(N - labelled_mask.sum()) // batch_size + 1}")

    while True:
        # ── 1. Get current labelled set ───────────────────────────────────────
        X_labeled = X_pool[labelled_mask]
        y_labeled = y_pool[labelled_mask]

        # Guard: need at least 2 classes to train a classifier
        if len(np.unique(y_labeled)) < 2:
            if verbose:
                print(f"  Iter {iteration}: skipping eval "
                      "(only one class in labelled set so far)")
            # Still add more molecules before evaluating
        else:
            # ── 2. Train on labelled set ──────────────────────────────────────
            fresh_model = model.clone_untrained()
            fresh_model.fit(X_labeled, y_labeled)

            # ── 3. Evaluate on test set ───────────────────────────────────────
            result = evaluate(
                model=fresh_model,
                X_test=X_test,
                y_test=y_test,
                n_labeled=int(labelled_mask.sum()),
                total_actives_in_pool=total_actives_in_pool,
                labeled_y=y_labeled,
            )
            results.append(result)

            if verbose and (iteration % 5 == 0 or iteration < 3):
                print(f"  Iter {iteration:3d} | "
                      f"labeled={result.n_labeled:6,} | "
                      f"AUPRC={result.auprc:.4f} | "
                      f"AUROC={result.auroc:.4f} | "
                      f"hits={result.hit_recovery:.3f}" if result.hit_recovery else
                      f"  Iter {iteration:3d} | "
                      f"labeled={result.n_labeled:6,} | "
                      f"AUPRC={result.auprc:.4f} | "
                      f"AUROC={result.auroc:.4f}")

        # ── 4. Check if pool is exhausted ─────────────────────────────────────
        unlabelled_indices = np.where(~labelled_mask)[0]
        if len(unlabelled_indices) == 0:
            if verbose:
                print("  Pool exhausted.")
            break

        # ── 5. Query: pick batch_size most uncertain unlabelled molecules ─────
        actual_batch = min(batch_size, len(unlabelled_indices))
        X_unlabelled = X_pool[unlabelled_indices]

        # Uncertainty requires a trained model; skip on very first iteration
        # if model wasn't trained (single-class guard above)
        if len(np.unique(y_labeled)) < 2:
            # Fall back to random selection if no trained model yet
            rng = np.random.default_rng(seed + iteration)
            chosen_positions = rng.choice(len(unlabelled_indices),
                                          size=actual_batch, replace=False)
        else:
            unc = fresh_model.uncertainty(X_unlabelled)   # shape (M,)
            # argsort descending: positions of most uncertain molecules
            chosen_positions = np.argsort(unc)[::-1][:actual_batch]

        # Convert local positions back to pool indices
        newly_labelled = unlabelled_indices[chosen_positions]
        labelled_mask[newly_labelled] = True
        iteration += 1

    if verbose:
        print(f"  AL complete. {len(results)} evaluation points recorded.")

    return results
