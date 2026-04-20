"""
active_learning/al_loop.py

Core active learning simulation loop — updated to handle both
fingerprint-based models (RandomForestModel) and graph-based models (MPNNModel).

What changed from the original:
The original loop passed X_pool (numpy fingerprint arrays) to every model.
MPNN models need PyG graph objects instead.  We handle this cleanly by:

  1. Detecting whether the model is a graph model via model.is_graph_model
  2. Accepting an optional graphs_pool parameter alongside X_pool
  3. When model.is_graph_model is True, passing graph subsets instead of
     fingerprint subsets to fit/predict_proba/uncertainty

Everything else: the labelled mask, hit recovery tracking, evaluation —
is unchanged. RandomForestModel still works as before with zero modifications.

ACQUISITION MODES:
acquisition : str (entropy, bald, weighted)
  'entropy'  — standard uncertainty sampling (RF default, MPNN baseline)
  'bald'     — BALD epistemic uncertainty via MC Dropout (MPNN only)
  'weighted' — imbalance-aware: entropy × p̂_active 
               Works for BOTH RF and MPNN:
               - RF: entropy(x) × predict_proba(x)[:,1]
               - MPNN: handled inside MPNNModel.uncertainty('weighted')
"""

import numpy as np
from typing import List, Optional

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
    # new parameters 
    graphs_pool: Optional[List] = None,   # PyG graphs for MPNN
    graphs_test: Optional[List] = None,   # PyG test graphs for MPNN
    acquisition: str = 'entropy', # 'entropy', 'bald', 'weighted'
    graphs_val: Optional[List] = None, 
    y_val: Optional[np.ndarray] = None,
) -> List[EvalResult]:
    """
    Run the full active learning simulation and return per-iteration metrics.

    For RF: pass X_pool and X_test (fingerprint arrays), leave graphs=None.
    For MPNN: pass graphs_pool and graphs_test, X_pool/X_test still needed
              for bookkeeping (y_pool indexing).
    """
    N = len(y_pool)
    total_actives_in_pool = int(y_pool.sum())
    is_graph_model = getattr(model, 'is_graph_model', False)

    # Validate: graph model needs graphs
    if is_graph_model:
        assert graphs_pool is not None, \
            "MPNNModel requires graphs_pool to be provided to run_active_learning"
        assert graphs_test is not None, \
            "MPNNModel requires graphs_test to be provided to run_active_learning"
        assert len(graphs_pool) == N, \
            f"graphs_pool length {len(graphs_pool)} != y_pool length {N}"

    # initialize labelled mask 
    labelled_mask = np.zeros(N, dtype=bool)
    labelled_mask[init_indices] = True

    results: List[EvalResult] = []
    iteration = 0

    if verbose:
        n_init_active = y_pool[init_indices].sum()
        model_type = "MPNN" if is_graph_model else "RF"
        print(f"\nAL loop starting ({model_type}, acquisition={acquisition})")
        print(f"  Init set : {labelled_mask.sum():,} molecules  "
              f"({n_init_active} actives = "
              f"{100*n_init_active/labelled_mask.sum():.1f}%)")
        print(f"  Pool size: {N:,}   Batch size: {batch_size}")
        print(f"  Max iterations: {(N - labelled_mask.sum()) // batch_size + 1}")

    while True:
        # 1. Get current labeled set 
        labeled_indices = np.where(labelled_mask)[0]
        y_labeled = y_pool[labeled_indices]

        if is_graph_model:
            data_labeled = [graphs_pool[i] for i in labeled_indices]
        else:
            X_labeled = X_pool[labeled_indices]

        # Guard: need at least 2 classes to train a classifier
        if len(np.unique(y_labeled)) < 2:
            if verbose:
                print(f"  Iter {iteration}: skipping eval "
                      "(only one class in labelled set so far)")
        else:
            # 2. Train on labeled set
            fresh_model = model.clone_untrained()
            if is_graph_model:
                fresh_model.fit(
                    data_labeled, y_labeled,
                    graphs_val=graphs_val,
                    y_val=y_val,
                    patience=10,
                )
            else:
                fresh_model.fit(X_labeled, y_labeled)

            # 3. Evaluate on test set 
            if is_graph_model:
                result = evaluate(
                    model=fresh_model,
                    X_test=graphs_test,# graphs for MPNN
                    y_test=y_test,
                    n_labeled=int(labelled_mask.sum()),
                    total_actives_in_pool=total_actives_in_pool,
                    labeled_y=y_labeled,
                )
            else:
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
                hr = f"hits={result.hit_recovery:.3f}" if result.hit_recovery is not None else ""
                print(f"  Iter {iteration:3d} | "
                      f"labeled={result.n_labeled:6,} | "
                      f"AUPRC={result.auprc:.4f} | "
                      f"AUROC={result.auroc:.4f} | {hr}")

        # 4. Check if pool is exhausted 
        unlabelled_indices = np.where(~labelled_mask)[0]
        if len(unlabelled_indices) == 0:
            if verbose:
                print("  Pool exhausted.")
            break

        # 5. Query: pick batch_size most informative unlabeled molecules 
        actual_batch = min(batch_size, len(unlabelled_indices))

        if len(np.unique(y_labeled)) < 2:
            # Fall back to random if no trained model yet
            rng = np.random.default_rng(seed + iteration)
            chosen_positions = rng.choice(len(unlabelled_indices),
                                          size=actual_batch, replace=False)
        else:
            if is_graph_model:
                unlabelled_graphs = [graphs_pool[i] for i in unlabelled_indices]
                unc = fresh_model.uncertainty(unlabelled_graphs,
                                              acquisition=acquisition)
            else:
                X_unlabelled = X_pool[unlabelled_indices]
                if acquisition == 'weighted':
                    # Imbalance-aware for RF:
                    # score = entropy × p̂_active
                    p_active = fresh_model.predict_proba(X_unlabelled)[:, 1]
                    p_clip   = np.clip(p_active, 1e-9, 1-1e-9)
                    entropy  = -(p_clip * np.log(p_clip) +
                                (1-p_clip) * np.log(1-p_clip))
                    unc = (entropy * p_active).astype(np.float32)
                else:
                    # Standard entropy (default RF behavior)
                    unc = fresh_model.uncertainty(X_unlabelled)

            chosen_positions = np.argsort(unc)[::-1][:actual_batch]

        # Convert local positions back to pool indices
        newly_labelled = unlabelled_indices[chosen_positions]
        labelled_mask[newly_labelled] = True
        iteration += 1

    if verbose:
        print(f"  AL complete. {len(results)} evaluation points recorded.")

    return results