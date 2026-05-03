"""
initialization/supervised_init.py

# Summary created using Claude 4.6 Sonnet:
Supervised Transfer Initialization for Molecular Active Learning.

WHAT THIS DOES:
---------------
Standard diversity-based initialization (MaxMin, k-Medoids) is label-blind —
it treats actives and inactives identically and picks structurally diverse
molecules. Under severe class imbalance (96.5% inactives), this produces
initial sets that are ~96.5% inactive — identical to random.

This module trains a Random Forest classifier on an external, more balanced
dataset (BACE: ~46% active, ~1,500 molecules) to learn which chemical
features distinguish active from inactive compounds. It then uses that
classifier to score all HIV pool molecules and bias initialization toward
predicted-active regions of chemical space.

PIPELINE:
---------
1. Load BACE dataset from TDC (balanced binary classification, 46% active)
2. Compute Morgan fingerprints for BACE molecules
3. Train RF classifier on BACE fingerprints (class_weight='balanced')
4. Decontaminate: remove any BACE molecules whose scaffold appears in HIV pool
5. Score all HIV pool molecules: p_active = RF.predict_proba(X_pool)[:,1]
6. Filter to top candidates by predicted activity (default top 50%)
7. Run MaxMin diversity selection within the filtered candidate set
8. Return indices into the original pool

WHY BACE?
---------
BACE (beta-secretase inhibitor dataset) has ~46% active rate vs HIV's 3.5%.
A classifier trained on BACE learns general drug-likeness features: aromatic
rings, hydrogen bond acceptors, appropriate molecular weight — properties
that correlate with bioactivity broadly, not just for HIV. This gives the
initialization a better prior over "what active compounds look like" compared
to random or diversity-based selection.

LIMITATION:
-----------
BACE inhibitors bind a different protein than HIV targets. The transfer
captures general drug-likeness but not HIV-specific pharmacophores.
The scientific claim is precise: "BACE-pretrained initialization selects
more drug-like molecules at iteration 0 than random or diversity-only
initialization, providing a better starting distribution."

DECONTAMINATION:
----------------
We remove any BACE molecule whose Bemis-Murcko scaffold appears in the
HIV pool to prevent data leakage. Scaffold-level decontamination is
sufficient for non-HIV datasets; exact SMILES decontamination is added
as an additional check.

USAGE:
------
    from initialization.supervised_init import supervised_transfer_initialization

    init_idx = supervised_transfer_initialization(
        X_pool       = data.X_train_pool,   # (N, 2048) Morgan fingerprints
        y_pool       = data.y_train_pool,   # only used for post-hoc evaluation
        smiles_pool  = data.smiles_train,   # for scaffold decontamination
        n_init       = 5757,
        seed         = 0,
        verbose      = True,
    )
"""

import numpy as np
from typing import Optional, List


def _smiles_to_fp(smiles_list: List[str], radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """Convert a list of SMILES to Morgan fingerprint array."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
            fps.append(np.array(fp, dtype=np.float32))
        else:
            fps.append(np.zeros(nbits, dtype=np.float32))
    return np.array(fps)


def _get_scaffold(smi: str) -> Optional[str]:
    """Return Bemis-Murcko scaffold SMILES for a molecule.
    References:
    1. https://docs.chemaxon.com/latest/jklustor_bemis-murcko-clustering.html
    2. https://github.com/rdkit/rdkit/discussions/6844
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def _canonical_smiles(smi: str) -> Optional[str]:
    """Return canonical SMILES."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def load_bace_dataset():
    """
    Load BACE dataset from TDC.

    Returns:
        smiles_train : list of SMILES strings (training split)
        y_train      : np.ndarray of binary labels (1=active, 0=inactive)
        active_rate  : float — fraction of actives in training set
    """
    from tdc.single_pred import ADMET
    print("  Loading BACE dataset from TDC...")
    bace = ADMET(name='BACE_classification')
    split = bace.get_split(method='scaffold')

    smiles_train = split['train']['Drug'].tolist()
    y_train      = split['train']['Y'].values.astype(int)

    active_rate = y_train.mean()
    print(f"  BACE train: {len(smiles_train)} molecules, "
          f"{y_train.sum()} actives ({active_rate:.1%})")
    return smiles_train, y_train, active_rate


def decontaminate_bace(
    bace_smiles: List[str],
    bace_labels: np.ndarray,
    pool_smiles: List[str],
    verbose: bool = True,
) -> tuple:
    """
    Remove BACE molecules whose scaffold or exact SMILES appears in the HIV pool.

    This prevents data leakage: if a BACE molecule with a known label appears
    in the HIV pool, the transfer classifier would effectively have seen that
    molecule's label during pretraining.

    Args:
        bace_smiles   : SMILES for BACE training set
        bace_labels   : binary labels for BACE training set
        pool_smiles   : SMILES for the HIV pool molecules
        verbose       : print decontamination statistics

    Returns:
        clean_smiles  : decontaminated BACE SMILES
        clean_labels  : corresponding labels
    """
    # Build sets of pool canonical SMILES and scaffolds
    print("  Building pool scaffold/SMILES index for decontamination...")
    pool_canonical = set()
    pool_scaffolds = set()

    for smi in pool_smiles:
        c = _canonical_smiles(smi)
        if c:
            pool_canonical.add(c)
        s = _get_scaffold(smi)
        if s:
            pool_scaffolds.add(s)

    print(f"  Pool: {len(pool_canonical)} canonical SMILES, "
          f"{len(pool_scaffolds)} unique scaffolds")

    # Check each BACE molecule
    keep_mask = []
    n_exact = 0
    n_scaffold = 0

    for smi in bace_smiles:
        c = _canonical_smiles(smi)
        s = _get_scaffold(smi)

        if c and c in pool_canonical:
            keep_mask.append(False)
            n_exact += 1
        elif s and s in pool_scaffolds:
            keep_mask.append(False)
            n_scaffold += 1
        else:
            keep_mask.append(True)

    keep_mask = np.array(keep_mask)
    clean_smiles = [s for s, k in zip(bace_smiles, keep_mask) if k]
    clean_labels = bace_labels[keep_mask]

    if verbose:
        print(f"  Decontamination: removed {n_exact} exact + "
              f"{n_scaffold} scaffold matches = "
              f"{(~keep_mask).sum()} total")
        print(f"  Clean BACE train: {len(clean_smiles)} molecules, "
              f"{clean_labels.sum()} actives ({clean_labels.mean():.1%})")

    return clean_smiles, clean_labels


def train_transfer_classifier(
    smiles_train: List[str],
    y_train: np.ndarray,
    n_estimators: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> object:
    """
    Train a Random Forest on external (BACE) fingerprints.

    Uses class_weight='balanced' to handle any residual imbalance.

    Args:
        smiles_train  : SMILES for external training set
        y_train       : binary labels
        n_estimators  : number of RF trees
        seed          : random seed for reproducibility

    Returns:
        Trained RandomForestClassifier
    """
    from sklearn.ensemble import RandomForestClassifier

    print(f"  Computing Morgan fingerprints for {len(smiles_train)} BACE molecules...")
    X_train = _smiles_to_fp(smiles_train)

    print(f"  Training transfer RF ({n_estimators} trees, class_weight=balanced)...")
    rf = RandomForestClassifier(
        n_estimators  = n_estimators,
        class_weight  = 'balanced',
        n_jobs        = -1,
        random_state  = seed,
    )
    rf.fit(X_train, y_train)

    # Quick self-evaluation
    from sklearn.metrics import average_precision_score
    p_train = rf.predict_proba(X_train)[:, 1]
    auprc_train = average_precision_score(y_train, p_train)
    if verbose:
        print(f"  Transfer RF train AUPRC: {auprc_train:.4f} "
              f"(expected ~0.9+ with balanced BACE)")

    return rf


def score_pool_with_transfer(
    rf_transfer,
    X_pool: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """
    Score all HIV pool molecules using the BACE-trained transfer classifier.

    Args:
        rf_transfer   : trained RandomForestClassifier (from train_transfer_classifier)
        X_pool        : (N, 2048) Morgan fingerprints for HIV pool molecules- the actual unlabeled pool data in principle

    Returns:
        p_active      : (N,) float array, predicted P(active) for each pool molecule
    """
    print(f"  Scoring {len(X_pool):,} HIV pool molecules with transfer RF...")
    p_active = rf_transfer.predict_proba(X_pool)[:, 1]
    if verbose:
        print(f"  Score range: [{p_active.min():.4f}, {p_active.max():.4f}]")
        print(f"  Molecules predicted active (>0.5): "
              f"{(p_active > 0.5).sum():,} ({(p_active > 0.5).mean():.1%})")
    return p_active


def supervised_transfer_initialization(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    smiles_pool: List[str],
    n_init: int,
    seed: int = 0,
    candidate_percentile: float = 50.0,
    n_estimators: int = 200,
    verbose: bool = True,
) -> np.ndarray:
    """
    Full supervised transfer initialization pipeline.

    1. Load and decontaminate BACE dataset
    2. Train transfer RF on BACE
    3. Score HIV pool with transfer RF
    4. Filter to top candidates (default: top 50% by predicted activity)
    5. Run MaxMin diversity within filtered candidates
    6. Return selected pool indices

    Args:
        X_pool               : (N, 2048) Morgan fingerprints for HIV pool
        y_pool               : (N,) true HIV labels (used only for reporting)
        smiles_pool          : list of N SMILES for HIV pool (for decontamination)
        n_init               : number of molecules to select
        seed                 : random seed
        candidate_percentile : top X percentile by predicted activity to keep
                               as candidates before MaxMin (default 50%)
        n_estimators         : trees in transfer RF
        verbose              : print progress

    Returns:
        init_idx : (n_init,) indices into X_pool of selected molecules
    """
    from initialization.maxmin_init import maxmin_initialization

    if verbose:
        print(f"\nSupervised Transfer Initialization")
        print(f"  n_init               = {n_init:,}")
        print(f"  candidate_percentile = top {100-candidate_percentile:.0f}% "
              f"predicted active")
        print(f"  seed                 = {seed}")
        print()

    # Step 1: Load BACE
    bace_smiles, bace_labels, _ = load_bace_dataset()

    # Step 2: Decontaminate
    clean_smiles, clean_labels = decontaminate_bace(
        bace_smiles, bace_labels, smiles_pool, verbose=verbose
    )

    # Step 3: Train transfer classifier
    rf_transfer = train_transfer_classifier(
        clean_smiles, clean_labels,
        n_estimators=n_estimators, seed=seed, verbose=verbose
    )

    # Step 4: Score pool
    p_active = score_pool_with_transfer(rf_transfer, X_pool, verbose=verbose)

    # Step 5: Filter to top candidates
    threshold      = np.percentile(p_active, candidate_percentile)
    candidate_mask = p_active >= threshold
    candidate_idx  = np.where(candidate_mask)[0]
    X_candidates   = X_pool[candidate_mask]

    if verbose:
        true_active_rate_candidates = y_pool[candidate_mask].mean()
        true_active_rate_pool       = y_pool.mean()
        print(f"\n  Candidate pool: {candidate_mask.sum():,} molecules "
              f"(top {100-candidate_percentile:.0f}% by predicted activity)")
        print(f"  True active rate in candidates : {true_active_rate_candidates:.2%}")
        print(f"  True active rate in full pool  : {true_active_rate_pool:.2%}")
        enrichment = true_active_rate_candidates / true_active_rate_pool
        print(f"  Enrichment factor              : {enrichment:.2f}x")

    # Step 6: MaxMin diversity within candidates
    if verbose:
        print(f"\n  Running MaxMin on {len(candidate_idx):,} candidates "
              f"to select {n_init:,} diverse molecules...")

    local_idx = maxmin_initialization(
        X_pool  = X_candidates,
        n_init  = n_init,
        seed    = seed,
        verbose = verbose,
    )

    # Map local indices back to full pool indices
    init_idx = candidate_idx[local_idx]

    # Step 7: Report final statistics
    if verbose:
        n_actives_selected    = y_pool[init_idx].sum()
        expected_random       = y_pool.mean() * n_init
        print(f"\n  ── Final Initialization Statistics ──")
        print(f"  Selected       : {len(init_idx):,} molecules")
        print(f"  Actives found  : {n_actives_selected} "
              f"({100*n_actives_selected/n_init:.1f}%)")
        print(f"  Expected random: {expected_random:.1f} "
              f"({100*y_pool.mean():.1f}%)")
        print(f"  Improvement    : "
              f"{n_actives_selected/expected_random:.2f}x over random")

    return init_idx