"""
data/data_loader.py
===================
Loads the TDC HIV dataset, computes Morgan fingerprints for every molecule,
and returns the scaffold-split train/validation/test pools as numpy arrays.

WHY THIS FILE EXISTS
--------------------
All other modules (initialization strategies, active learning loop, etc.) need
the same data in the same format.  Centralising loading here means changing one
line (e.g. switching to BACE) automatically propagates everywhere.

WHAT IT RETURNS
---------------
DataBundle  – a plain dataclass with:
    X_train_pool  : float32 numpy array  (N_train, 2048)  – Morgan fingerprints
    y_train_pool  : int numpy array      (N_train,)        – 0/1 labels
    X_val         : float32 numpy array  (N_val,   2048)
    y_val         : int numpy array      (N_val,)
    X_test        : float32 numpy array  (N_test,  2048)
    y_test        : int numpy array      (N_test,)
    smiles_train  : list[str]            – kept for optional Chemprop use later
    smiles_val    : list[str]
    smiles_test   : list[str]
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem



# helper

def smiles_to_fingerprint(smi: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Convert one SMILES string → binary Morgan fingerprint (numpy float32 array).

    radius=2  means each bit encodes substructures up to 2 bonds from a centre
              atom – this is the "ECFP4" convention used in your milestone report.
    n_bits    matches your teammate's implementation (2048-bit vectors).

    Returns a zero vector if the SMILES cannot be parsed so that a single bad
    molecule never crashes the whole dataset load.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


# ── return type ──────────────────────────────────────────────────────────────

@dataclass
class DataBundle:
    """All arrays you will ever need, in one object."""
    X_train_pool: np.ndarray   # full labelled training pool (AL picks from here)
    y_train_pool: np.ndarray
    X_val: np.ndarray          # fixed validation set  (never queried, only evaluated)
    y_val: np.ndarray
    X_test: np.ndarray         # fixed held-out test set (final evaluation only)
    y_test: np.ndarray
    smiles_train: List[str]    # raw SMILES – needed later when we plug in Chemprop
    smiles_val: List[str]
    smiles_test: List[str]


# ── main function ─────────────────────────────────────────────────────────────

def load_hiv_data(radius: int = 2, n_bits: int = 2048) -> DataBundle:
    """
    Download (or use cached) TDC HIV dataset and return a DataBundle.

    The scaffold split (80/10/10) is the same one used in your milestone report.
    Scaffold splits put molecules with *different* Bemis-Murcko scaffolds into
    different splits, so the test set is genuinely out-of-distribution – a much
    harder and more realistic evaluation than a random split.

    Steps
    -----
    1. Download via PyTDC  (cached after first run in ~/.tdc_data/)
    2. Extract SMILES + labels from each split
    3. Convert every SMILES to a 2048-bit Morgan fingerprint
    4. Pack into DataBundle and return
    """
    try:
        from tdc.single_pred import HTS
    except ImportError:
        raise ImportError(
            "PyTDC not installed.  Run:  pip install PyTDC --quiet"
        )

    print("Loading HIV dataset from TDC ...")
    # Build absolute path to the data folder — works regardless of working directory
    import os as _os
    _DATA_DIR = _os.path.dirname(_os.path.abspath(__file__))
    data = HTS(name="HIV", path=_DATA_DIR)
    split = data.get_split(method="scaffold")   # returns dict with 'train','valid','test'

    # ── unpack splits ─────────────────────────────────────────────────────────
    df_train = split["train"]   # pandas DataFrame with columns: Drug, Y
    df_val   = split["valid"]
    df_test  = split["test"]

    smiles_train = df_train["Drug"].tolist()
    smiles_val   = df_val["Drug"].tolist()
    smiles_test  = df_test["Drug"].tolist()

    y_train = df_train["Y"].values.astype(int)
    y_val   = df_val["Y"].values.astype(int)
    y_test  = df_test["Y"].values.astype(int)

    # ── fingerprints ──────────────────────────────────────────────────────────
    print(f"Computing {n_bits}-bit Morgan fingerprints (radius={radius}) ...")
    print(f"  Train pool : {len(smiles_train):,} molecules")
    print(f"  Validation : {len(smiles_val):,} molecules")
    print(f"  Test       : {len(smiles_test):,} molecules")

    X_train = np.vstack([smiles_to_fingerprint(s, radius, n_bits) for s in smiles_train])
    X_val   = np.vstack([smiles_to_fingerprint(s, radius, n_bits) for s in smiles_val])
    X_test  = np.vstack([smiles_to_fingerprint(s, radius, n_bits) for s in smiles_test])

    # ── quick sanity check ────────────────────────────────────────────────────
    active_frac = y_train.mean()
    print(f"  Active fraction in train pool: {active_frac:.3f}  "
          f"({y_train.sum()} actives / {len(y_train)} total)")

    return DataBundle(
        X_train_pool=X_train,
        y_train_pool=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        smiles_train=smiles_train,
        smiles_val=smiles_val,
        smiles_test=smiles_test,
    )
