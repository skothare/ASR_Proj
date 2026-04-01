"""
initialization/kmedoids_init.py
================================
DOE Strategy B: k-Medoids Clustering Initialization.

CONCEPTUAL BACKGROUND
---------------------
Where MaxMin picks points greedily by maximising the minimum distance,
k-Medoids takes a global approach: it partitions the entire pool into k
clusters, then selects the medoid (most central actual data point) of each
cluster.

This is the molecular analog of *Latin Hypercube Sampling* – you tile the
chemical space into k roughly equal regions and draw exactly one representative
from each region.  The key difference from k-Means is that medoids must be
*actual data points* (molecules that exist in the pool), whereas k-Means
centroids can be imaginary points in feature space.

WHY MEDOIDS OVER MEANS?
  Morgan fingerprints are binary.  The arithmetic mean of binary vectors
  produces fractional values that don't correspond to any real molecule.
  Medoids are always real molecules, so every selected point is a valid SMILES.

DISTANCE METRIC
  Tanimoto distance, same as MaxMin.  Two molecules that share all their
  active substructure bits have distance 0; two that share nothing have
  distance 1.

ALGORITHM USED: CLARA (Clustering Large Applications)
  Full k-Medoids is O(N²) per iteration and too slow for N≈33K.
  CLARA repeatedly runs k-Medoids on random subsamples and keeps the best
  partition.  We use the `kmedoids` package which implements PAM + CLARA.

PARAMETERS
----------
X_pool  : np.ndarray  shape (N, 2048) – fingerprint matrix
n_init  : int  – number of clusters (= number of molecules to select)
seed    : int  – random seed for CLARA subsampling

RETURNS
-------
selected_indices : np.ndarray shape (n_init,) – indices of cluster medoids
"""

import numpy as np


def _tanimoto_distance_matrix_batched(
    X: np.ndarray,
    batch_size: int = 1000,
) -> np.ndarray:
    """
    Compute the full N×N Tanimoto distance matrix in row-batches.

    For N=33K this is a 33K×33K float32 matrix = ~4 GB, which will OOM on
    Colab.  We therefore only use this for small subsamples (CLARA style).

    X : shape (N, D) bool
    Returns : shape (N, N) float32
    """
    N = X.shape[0]
    X_bool = X.astype(bool)
    D = np.zeros((N, N), dtype=np.float32)

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        batch = X_bool[start:end]                     # (batch, D)
        inter = (batch[:, np.newaxis, :] & X_bool[np.newaxis, :, :]).sum(axis=2)  # (batch, N)
        a_sum = batch.sum(axis=1)                      # (batch,)
        b_sum = X_bool.sum(axis=1)                     # (N,)
        union = a_sum[:, np.newaxis] + b_sum[np.newaxis, :] - inter
        union = np.where(union == 0, 1, union)
        D[start:end] = 1.0 - inter / union

    return D


def kmedoids_initialization(
    X_pool: np.ndarray,
    n_init: int,
    seed: int = 42,
    subsample_size: int = 5000,
    verbose: bool = True,
) -> np.ndarray:
    """
    Select `n_init` medoids from X_pool using k-Medoids on Tanimoto distance.

    Because full k-Medoids is O(N²) and N≈33K, we use CLARA-style subsampling:
      1. Draw `subsample_size` molecules randomly
      2. Compute Tanimoto distance matrix within the subsample
      3. Run PAM k-Medoids on the subsample to get n_init cluster centres
      4. For each centre, find the closest molecule in the FULL pool
         (this is the step that maps subsample results back to real pool indices)

    Parameters
    ----------
    subsample_size : int
        How many molecules to subsample for the distance-matrix computation.
        5000 gives a 5K×5K matrix = ~100 MB, which is fine on Colab.
        Larger → better coverage but slower.  Trade-off.
    """
    try:
        import kmedoids as km_lib
    except ImportError:
        raise ImportError(
            "kmedoids package not installed.\n"
            "Run:  pip install kmedoids --quiet"
        )

    N = X_pool.shape[0]
    rng = np.random.default_rng(seed)

    # ── Step 1: subsample ─────────────────────────────────────────────────────
    sub_size = min(subsample_size, N)
    sub_idx  = rng.choice(N, size=sub_size, replace=False)
    X_sub    = X_pool[sub_idx].astype(bool)

    if verbose:
        print(f"k-Medoids init: {n_init:,} clusters, "
              f"subsample={sub_size:,} / {N:,}")
        print("  Computing Tanimoto distance matrix on subsample ...")

    # ── Step 2: distance matrix on subsample ──────────────────────────────────
    D_sub = _tanimoto_distance_matrix_batched(X_sub)

    # ── Step 3: PAM k-Medoids on subsample ────────────────────────────────────
    if verbose:
        print(f"  Running PAM k-Medoids (k={n_init}) ...")

    result = km_lib.fasterpam(D_sub, n_init, random_state=seed)
    # result.medoids contains indices INTO the subsample
    sub_medoid_positions = result.medoids  # shape (n_init,)

    # ── Step 4: map back to full-pool indices ─────────────────────────────────
    # The medoid positions are indices into X_sub.
    # sub_idx[pos] gives the original pool index.
    pool_medoid_indices = sub_idx[sub_medoid_positions]

    if verbose:
        print(f"  Done.  {len(pool_medoid_indices)} medoids selected.")

    return pool_medoid_indices.astype(np.int64)
