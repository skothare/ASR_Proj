"""
initialization/maxmin_init.py
==============================
DOE Strategy A: Greedy MaxMin Tanimoto Diversity Initialization.

CONCEPTUAL BACKGROUND
---------------------
Classical DOE (D-optimality, from your Lec 15 slides) selects k design points
that maximize det(X'X) — which geometrically means maximising the "spread" of
the selected set in feature space so that the confidence ellipsoid around the
OLS coefficient estimate is as small as possible.

We cannot apply D-optimality directly to 2048-bit binary fingerprints because:
  1. The space is discrete (we pick from existing molecules, not place points)
  2. 2048 dimensions makes det(X'X) numerically unstable
  3. D-optimality assumes a parametric linear model; we use RF / MPNN

MaxMin diversity selection is the established molecular-space analog of D-
optimality.  It was formalised for compound library selection in:

  Snarey et al. (1997) "Comparison of algorithms for dissimilarity-based
  compound selection", J. Mol. Graph. Model. 15(6):372-385.

  Taylor (1995) "Simulation Analysis of Experimental Design Strategies for
  Screening Random Compounds", J. Chem. Inf. Comput. Sci. 35(1):59-67.

THE ALGORITHM
-------------
1. Pick one molecule (randomly, or the most central one) → selected set S
2. For i = 1 … k:
     a. For each candidate x NOT yet in S:
            min_dist(x) = min over s∈S  Tanimoto_distance(x, s)
        This is "how far is x from its nearest already-selected neighbour?"
     b. Pick  x* = argmax_{x not in S}  min_dist(x)
        i.e. the molecule that is farthest from anything already selected
     c. Add x* to S

TANIMOTO DISTANCE (for binary fingerprints)
-------------------------------------------
Tanimoto similarity between two bit-vectors a and b:
    sim(a,b) = |a AND b| / |a OR b|   (fraction of shared ON-bits)
Tanimoto distance:
    dist(a,b) = 1 – sim(a,b)

COMPLEXITY
----------
Naïve: O(k × N) similarity computations per step → O(k² × N) total, where
N ≈ 33K and k ≈ 6.6K (20% of pool).  This is slow in pure Python.

We use two tricks to make this tractable on Colab CPU:
  1. Represent fingerprints as uint8 numpy arrays; use bitwise ops for fast
     counting of AND and OR bits.
  2. Maintain a `min_dist` vector that is updated incrementally: after adding
     x* to S, only update candidates where dist(candidate, x*) < their
     current min_dist.  This avoids recomputing all pairwise distances.

PARAMETERS
----------
X_pool  : np.ndarray  shape (N, 2048), float32 or bool – fingerprint matrix
n_init  : int  – number of molecules to select
seed    : int  – selects the first molecule randomly (for reproducibility)

RETURNS
-------
selected_indices : np.ndarray shape (n_init,) – indices into X_pool
"""

import numpy as np


# ── fast Tanimoto utilities ───────────────────────────────────────────────────

def _to_bool(X: np.ndarray) -> np.ndarray:
    """Ensure fingerprint matrix is boolean for fast bitwise operations."""
    return X.astype(bool)


def _tanimoto_sim_one_vs_all(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """
    Compute Tanimoto similarity between ONE query vector and ALL rows in pool.

    query : shape (D,)  bool
    pool  : shape (N, D) bool

    Returns similarities : shape (N,)  float32
    """
    # Number of bits set in the AND (intersection)
    intersection = pool & query[np.newaxis, :]          # (N, D) bool
    inter_count  = intersection.sum(axis=1).astype(np.float32)  # (N,)

    # Number of bits set in the OR (union) = |a| + |b| - |a AND b|
    query_count   = query.sum().astype(np.float32)
    pool_counts   = pool.sum(axis=1).astype(np.float32)  # (N,)
    union_count   = pool_counts + query_count - inter_count      # (N,)

    # Avoid division by zero (all-zero fingerprint → similarity 0)
    union_count   = np.where(union_count == 0, 1.0, union_count)

    return inter_count / union_count


# ── main function ─────────────────────────────────────────────────────────────

def maxmin_initialization(
    X_pool: np.ndarray,
    n_init: int,
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """
    Select `n_init` indices from X_pool using greedy MaxMin Tanimoto diversity.

    This is the DOE-inspired strategy: label-blind, coverage-maximising.
    """
    N = X_pool.shape[0]
    assert n_init <= N, f"n_init ({n_init}) cannot exceed pool size ({N})"

    X_bool = _to_bool(X_pool)   # convert once up front

    # ── Step 1: pick first molecule at random ─────────────────────────────────
    rng = np.random.default_rng(seed)
    first_idx = int(rng.integers(0, N))

    selected = [first_idx]
    # `min_dist_to_selected[i]` = distance from molecule i to its nearest
    # already-selected molecule.  Initialise as distance to first selected.
    sim_to_first = _tanimoto_sim_one_vs_all(X_bool[first_idx], X_bool)
    min_dist = (1.0 - sim_to_first).astype(np.float32)
    min_dist[first_idx] = -1.0   # mark as already selected (exclude from future picks)

    if verbose:
        print(f"MaxMin init: selecting {n_init:,} molecules from pool of {N:,}")

    # ── Step 2: greedily add the most distant molecule ────────────────────────
    for step in range(1, n_init):
        # Pick candidate with largest min-distance to selected set
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

        # Update min_dist incrementally:
        # Only molecules whose distance to next_idx is SMALLER than their
        # current recorded min_dist need an update.
        sim_to_new  = _tanimoto_sim_one_vs_all(X_bool[next_idx], X_bool)
        dist_to_new = (1.0 - sim_to_new).astype(np.float32)
        min_dist    = np.minimum(min_dist, dist_to_new)
        min_dist[next_idx] = -1.0   # mark as selected

        if verbose and (step % 500 == 0 or step == n_init - 1):
            print(f"  Step {step:,}/{n_init:,}  |  max min-dist = {min_dist.max():.4f}")

    return np.array(selected, dtype=np.int64)
