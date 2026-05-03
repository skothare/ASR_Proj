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
  'diversity' — MiniBatch K-Means clustering:
               - RF: clusters on Morgan fingerprints (X_pool)
               - MPNN: clusters on learned graph embeddings
  'density'  — uncertainty × density in feature space:
               - RF: entropy × cosine similarity in Morgan fingerprint space
               - MPNN: entropy × cosine similarity in graph embedding space
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
    acquisition: str = 'entropy',
    graphs_val: Optional[List] = None, 
    y_val: Optional[np.ndarray] = None,
    checkpoint_dir: Optional[str] = None,
    cond_key: Optional[str] = None,
    run_seed: Optional[int] = None,
    warm_start: bool = True,
) -> List[EvalResult]:
    """
    Run the full active learning simulation and return per-iteration metrics.
 
    For RF: pass X_pool and X_test (fingerprint arrays), leave graphs=None.
    For MPNN: pass graphs_pool and graphs_test, X_pool/X_test still needed
              for bookkeeping (y_pool indexing) and diversity/density sampling.
 
    Note: diversity and density acquisition always use Morgan fingerprints
    (X_pool) for both RF and MPNN, so that diversity is measured in a
    consistent molecular feature space rather than model-dependent embeddings.
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
 
    # Validate: diversity/density always need X_pool (fingerprints)
    if acquisition in ('diversity', 'density'):
        assert X_pool is not None, \
            f"acquisition='{acquisition}' requires X_pool (Morgan fingerprints)"
 
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

    # Initialize warm start state (before the while loop)
    warm_model = None

    while True:
        labeled_indices = np.where(labelled_mask)[0]
        y_labeled       = y_pool[labeled_indices]

        if is_graph_model:
            data_labeled = [graphs_pool[i] for i in labeled_indices]
        else:
            X_labeled = X_pool[labeled_indices]
        if len(np.unique(y_labeled)) < 2:
            if verbose:
                print(f"  Iter {iteration}: skipping (only one class)")
        else:
            # ── Model selection: warm or cold start ───────────────────────
            if warm_start and warm_model is not None:
                # Warm: continue fine-tuning from previous iteration's weights
                current_model = warm_model
                lr_now        = 3e-4 if iteration < 3 else 1e-4
                patience_now  = 10   if iteration < 5 else 7
            else:
                # Cold: fresh random weights every iteration
                current_model = model.clone_untrained()
                lr_now        = None   # use model's default LR
                patience_now  = 10

            if is_graph_model:
                current_model.fit(
                    data_labeled, y_labeled,
                    graphs_val=graphs_val,
                    y_val=y_val,
                    patience=patience_now,
                    lr_override=lr_now,
                )
            else:
                current_model.fit(X_labeled, y_labeled)

            warm_model  = current_model   # always save — only used if warm_start=True
            fresh_model = current_model


            # ── Evaluate on test set ───────────────────────────────────
            if is_graph_model:
                result = evaluate(
                    model=fresh_model,
                    X_test=graphs_test,
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
 
            # Log to WandB after each iteration
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "al/n_labeled":    result.n_labeled,
                        "al/auprc":        result.auprc,
                        "al/auroc":        result.auroc,
                        "al/hit_recovery": result.hit_recovery or 0,
                    })
            except Exception:
                pass
 
            # Checkpoint: save results and labeled mask after every iteration
            try:
                import json, os
                ckpt_dir = "/content/drive/MyDrive/ASR_Proj/checkpoints"
                os.makedirs(ckpt_dir, exist_ok=True)
                
                ckpt = {
                    "iteration": iteration,
                    "labelled_indices": np.where(labelled_mask)[0].tolist(),
                    "results": {
                        "n_labeled":    [r.n_labeled for r in results],
                        "auprc":        [r.auprc for r in results],
                        "auroc":        [r.auroc for r in results],
                        "hit_recovery": [r.hit_recovery or 0 for r in results],
                    }
                }
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_iter{iteration}.json")
                with open(ckpt_path, "w") as f:
                    json.dump(ckpt, f, indent=2)
            except Exception as e:
                print(f"  Checkpoint save failed: {e}")
 
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
 
        # Morgan fingerprints for the unlabelled subset — used by
        # diversity and density for both RF and MPNN
        X_unlabelled = X_pool[unlabelled_indices]
 
        if len(np.unique(y_labeled)) < 2:
            # Fall back to random if no trained model yet
            rng = np.random.default_rng(seed + iteration)
            chosen_positions = rng.choice(len(unlabelled_indices),
                                          size=actual_batch, replace=False)
        else:
            # Compute uncertainty scores for uncertainty-based acquisitions
            if is_graph_model:
                unlabelled_graphs = [graphs_pool[i] for i in unlabelled_indices]
                if acquisition not in ('diversity', 'density', 'random'):
                    unc = fresh_model.uncertainty(unlabelled_graphs,
                                                  acquisition=acquisition)
            else:
                if acquisition == 'weighted':
                    p_active = fresh_model.predict_proba(X_unlabelled)[:, 1]
                    p_clip   = np.clip(p_active, 1e-9, 1-1e-9)
                    entropy  = -(p_clip * np.log(p_clip) +
                                (1-p_clip) * np.log(1-p_clip))
                    unc = (entropy * p_active).astype(np.float32)
                elif acquisition not in ('diversity', 'density', 'random'):
                    unc = fresh_model.uncertainty(X_unlabelled)
 
            if acquisition == 'random':
                rng = np.random.default_rng(seed + iteration)
                chosen_positions = rng.choice(len(unlabelled_indices),
                                              size=actual_batch, replace=False)
 
            elif acquisition == 'diversity':
                # Cluster in Morgan fingerprint space for both RF and MPNN.
                # Diversity should reflect structural diversity in molecular
                # feature space, not model-dependent learned embeddings.
                from sklearn.cluster import MiniBatchKMeans
                k = min(actual_batch, len(unlabelled_indices))
                km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3)
                km.fit(X_unlabelled)
 
                chosen_positions = []
                used = set()
                for center in km.cluster_centers_:
                    dists = np.linalg.norm(X_unlabelled - center, axis=1)
                    # rank candidates, pick the closest not already chosen
                    for idx in np.argsort(dists):
                        if idx not in used:
                            chosen_positions.append(idx)
                            used.add(idx)
                            break
                chosen_positions = np.array(chosen_positions)
 
            elif acquisition == 'density':
                # Density computed via cosine similarity in Morgan fingerprint
                # space for both RF and MPNN. Uncertainty signal comes from
                # the respective model's entropy estimate.
                from sklearn.metrics.pairwise import cosine_similarity
 
                if is_graph_model:
                    unc = fresh_model.uncertainty(unlabelled_graphs,
                                                  acquisition='entropy')
                else:
                    p_active = fresh_model.predict_proba(X_unlabelled)[:, 1]
                    p_clip = np.clip(p_active, 1e-9, 1-1e-9)
                    unc = -(p_clip * np.log(p_clip) +
                            (1-p_clip) * np.log(1-p_clip)).astype(np.float32)
 
                max_density_sample = 2000
                if len(unlabelled_indices) > max_density_sample:
                    rng = np.random.default_rng(seed + iteration)
                    sample_idx = rng.choice(len(unlabelled_indices),
                                            size=max_density_sample, replace=False)
                    sim_matrix = cosine_similarity(X_unlabelled,
                                                   X_unlabelled[sample_idx])
                else:
                    sim_matrix = cosine_similarity(X_unlabelled)
                density = sim_matrix.mean(axis=1)
                scores = unc * density
                chosen_positions = np.argsort(scores)[::-1][:actual_batch]
 
            elif acquisition in ('entropy', 'bald', 'weighted'):
                chosen_positions = np.argsort(unc)[::-1][:actual_batch]
 
            else:
                raise ValueError(f"Unknown acquisition: {acquisition}. "
                                 f"Choose from 'entropy', 'bald', 'weighted', "
                                 f"'diversity', 'density', 'random'")
 
        # Convert local positions back to pool indices
        newly_labelled = unlabelled_indices[chosen_positions]
        labelled_mask[newly_labelled] = True
        iteration += 1
 
    if verbose:
        print(f"  AL complete. {len(results)} evaluation points recorded.")
 
    return results