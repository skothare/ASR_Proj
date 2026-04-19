"""
preprocessing/data/graph_builder.py

Converts SMILES strings into PyTorch Geometric Data objects.

Takes a SMILES string (e.g. "CC(=O)Oc1ccccc1C(=O)O") and returns a
torch_geometric.data.Data object with:
  - x          : atom feature matrix  shape (num_atoms, 9)
  - edge_index  : bond connectivity    shape (2, 2*num_bonds)  (bidirectional)
  - edge_attr   : bond feature matrix  shape (2*num_bonds, 4)
  - y           : label tensor         shape (1,)  if label provided

Features: 9 (atom features) and 4 (bond features) to match the feature set described in the milestone report (Section 3.4.4) and used in the Chemprop MPNN implementation:
Chemprop: https://chemprop.readthedocs.io/en/main/tutorial/python/featurizers/atom_featurizers.html

  Atom features (9):
    0: atomic number       (integer, encodes element identity)
    1: degree              (number of bonds, captures connectivity)
    2: formal charge       (captures ionic character)
    3: chiral tag          (0=none, 1=CW, 2=CCW, 3=other)
    4: num H               (implicit + explicit hydrogens)
    5: hybridization       (SP=2, SP2=3, SP3=4, ...)
    6: is aromatic         (0/1)
    7: is in ring          (0/1)
    8: atomic mass / 100   (normalised, gives rough size signal)

  Bond features (4):
    0: bond type           (1=single, 2=double, 3=triple, 4=aromatic)
    1: is conjugated       (0/1)
    2: is in ring          (0/1)
    3: stereo              (0=none, 1=any, 2=Z, 3=E, ...)



Usage:
from preprocessing.data.graph_builder import smiles_to_graph, build_graph_dataset

# Single molecule
g = smiles_to_graph("CC(=O)O", label=1)

# Full dataset
graphs = build_graph_dataset(smiles_list, labels)
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
from typing import Optional, List


#atom feature helpers: https://chemprop.readthedocs.io/en/main/tutorial/python/featurizers/atom_featurizers.html

_HYBRIDIZATION_MAP = {
    rdchem.HybridizationType.S:1,
    rdchem.HybridizationType.SP: 2,
    rdchem.HybridizationType.SP2:3,
    rdchem.HybridizationType.SP3:4,
    rdchem.HybridizationType.SP3D: 5,
    rdchem.HybridizationType.SP3D2:6,
}

_BOND_TYPE_MAP = {
    rdchem.BondType.SINGLE:1,
    rdchem.BondType.DOUBLE:2, # double bond
    rdchem.BondType.TRIPLE:3, # triple bond
    rdchem.BondType.AROMATIC:4, # aromatic bond
}

_STEREO_MAP = {
    rdchem.BondStereo.STEREONONE: 0,
    rdchem.BondStereo.STEREOANY:1,
    rdchem.BondStereo.STEREOZ:2,
    rdchem.BondStereo.STEREOE:3,
    rdchem.BondStereo.STEREOCIS: 4,
    rdchem.BondStereo.STEREOTRANS:5,
}

ATOM_FEATURE_DIM = 9
BOND_FEATURE_DIM = 4


def _atom_features(atom: rdchem.Atom) -> List[float]:
    """
    Extract 9 numerical features from an RDKit atom.
    All values are floats so PyTorch can use them directly.
    """
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetChiralTag()),
        float(atom.GetTotalNumHs()),
        float(_HYBRIDIZATION_MAP.get(atom.GetHybridization(), 0)),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetMass() / 100.0,
    ]


def _bond_features(bond: rdchem.Bond) -> List[float]:
    """
    Extract 4 numerical features from an RDKit bond.
    Bonds are stored bidirectionally (both i→j and j→i) with the same features.
    """
    return [
        float(_BOND_TYPE_MAP.get(bond.GetBondType(), 0)),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(_STEREO_MAP.get(bond.GetStereo(), 0)),
    ]


# main conversion function

def smiles_to_graph(
    smiles: str,
    label: Optional[float] = None,
) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Returns None if the SMILES cannot be parsed by RDKit (bad molecule).


    Parameters
    ----------
    smiles : str    SMILES string
    label  : float  optional label (0 or 1 for binary classification)

    Returns
    -------
    Data object with x, edge_index, edge_attr, (y if label provided)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # atom features 
    atom_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)  # (num_atoms, 9)

    # bond features (bidirectional) 
    # For each bond we create two directed edges: i→j and j→i
    edge_index_list = []
    edge_attr_list  = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feats = _bond_features(bond)
        # add both directions
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list  += [feats, feats]

    if len(edge_index_list) == 0:
        # single atom molecule — no bonds
        # still valid, just no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)

    # assemble Data object 
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data


# dataset builder 

def build_graph_dataset(
    smiles_list: List[str],
    labels: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> List[Data]:
    """
    Convert a list of SMILES strings to a list of PyG Data objects.

    Skips molecules that cannot be parsed (returns None from smiles_to_graph).
    The returned list is parallel to smiles_list with None entries removed —should also filter the corresponding labels.

    Parameters:
    smiles_list : list of str
    labels      : np.ndarray shape (N,) or None (0/1 labels)
    verbose     : whether to print parse statistics

    Returns:
    graphs         : list of Data objects (len <= len(smiles_list))
    valid_indices  : np.ndarray of indices into the original smiles_list
                     that successfully parsed
    """
    graphs = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        label = float(labels[i]) if labels is not None else None
        g = smiles_to_graph(smi, label=label)
        if g is not None:
            graphs.append(g)
            valid_indices.append(i)

    if verbose:
        n_failed = len(smiles_list) - len(graphs)
        print(f"  Parsed {len(graphs):,} / {len(smiles_list):,} molecules "
              f"({n_failed} failed to parse)")

    return graphs, np.array(valid_indices, dtype=np.int64)