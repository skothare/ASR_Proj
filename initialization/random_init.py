"""
initialization/random_init.py
==============================
Random initialization strategy – the baseline every DOE method is compared to.

WHAT IT DOES
------------
Selects `n_init` indices uniformly at random from the training pool, with no
knowledge of fingerprints, chemical space, or labels.

WHY IT IS THE CORRECT BASELINE
--------------------------------
Your teammate's starting-fraction experiments already established that 20% (≈
8,000 molecules) is a sensible initialization size – performance later converges
regardless of starting fraction, but 20% gives enough signal to start.

If DOE initializations improve over this, it means structured coverage of
chemical space genuinely helps.  If they don't, random is sufficient and DOE
adds overhead without benefit – that null result is also publishable.

PARAMETERS
----------
n_total  : int   – size of the full training pool
n_init   : int   – how many indices to select
seed     : int   – random seed for reproducibility

RETURNS
-------
selected_indices : np.ndarray shape (n_init,)   – indices into the training pool
"""

import numpy as np


def random_initialization(n_total: int, n_init: int, seed: int = 42) -> np.ndarray:
    """
    Draw `n_init` indices uniformly at random from [0, n_total).

    This is the *label-blind* baseline.  No fingerprints needed.
    """
    rng = np.random.default_rng(seed)
    selected = rng.choice(n_total, size=n_init, replace=False)
    return selected
