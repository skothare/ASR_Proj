"""
models/mpnn_model.py

Message Passing Neural Network built using PyTorch Geometric.

Input: molecular graph (atoms as nodes, bonds as edges)
Output: binary classification (active or inactive)

Logic:

1. Atom features --> Linear projection --> hidden_dim -->
2. k rounds of NNConv (bond-conditioned message passing) 
    - Each message is W (bond features) * h_neighbor
    -The bond MLP W maps edge_attr (4-dim) --> hiddenxhidden
    - Aggregation: sum over the neighbors
    - Update: h_v = ReLU(h_v + message) + dropout -->
3. Global mean pooling --> molecular vector (hidden_dim) --> 
4. MLP head: Linear --> ReLU --> Dropout --> Linear --> sigmoid --> 
5. p_active (one scalar per molecule)

** Using NNConv (Neural Network Convolution) for message passing instead of GCNConv as the latter treats all bonds identically whereas NNConv uses a small MLP to transform edge features into a weight matrix so each messaage is conditioned on bond type. This is as implemented in ChemProp --> an aromatic ring carbon bond is different from a single bond.

** Uncertainty Quantificaiton:
Dropout layers are kept ON at inference time (model.train() mode). Running T forward passes with different dropout masks gives T slightly different p̂ values. Their variance = epistemic uncertainty.
 
This is placed in BOTH the message-passing layers AND the MLP head.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from typing import List, Optional
 

class _MPNNNet(nn.Module):
    """
    PyTorch Module for MPNN model.

    Parameters:
    - atom_feat_dim: input atom feature dimension
    - bond_feat_dim: input bond feature dimension
    - hidden_dim: hidden dimension
    - num_layers: number of message passing rounds (k=3 for example)
    - dropout_p: dropout probability applied after each MP round and in the MLP head
    """

    def __init__(self, atom_feat_dim: int =9, bond_feat_dim: int =4, hidden_dim: int =128, num_layers: int =3, dropout_p: float =0.3):
        super().__init__()
        self.atom_feat_dim = atom_feat_dim
        self.bond_feat_dim = bond_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.atom_proj = nn.Linear(atom_feat_dim, hidden_dim) # project the raw atom features to hidden_dim
        # message passing layers:
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            # use edge_network to map bond features to hiddenxhidden matrix
            edge_network = nn.Sequential(
                nn.Linear(bond_feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_network, aggr='mean'))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Apply dropout after each message passing layer and in the MLP head
        self.mp_dropout = nn.Dropout(p=dropout_p)

        # Establish and MLP classification head:
        # Input here is the pooled molecule vector (hidden_dim) and the output is a single logit
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p), # MC Dropout in the head too
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass for the MPNN model.

        Parameters:
        data: PyG Batch object (output of DataLoader, contains multiple molecular graphs)

        Outputs:
        logits: shape (num_graphs, )- raw scores before sigmoid
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
 
        #project atom features 
        h = F.relu(self.atom_proj(x))  # (total_atoms, hidden_dim)
 
        # k rounds of message passing 
        # Each round: collect messages from neighbors weighted by bond features, apply residual connection, normalize, dropout
        for conv, norm in zip(self.convs, self.norms):
            h_new = F.relu(conv(h, edge_index, edge_attr))
            h = norm(h + h_new) # residual connection
            h  = self.mp_dropout(h) # dropout after each round
 
        # pool atoms → molecule vector 
        mol_vec = global_mean_pool(h, batch)  # (num_graphs, hidden_dim)
 
        #  classification head 
        logits = self.head(mol_vec).squeeze(-1)  # (num_graphs,)
        return logits

class MPNNModel:
    """
    Active learning wrapper around _MPNNNet.
 
    Exposes the same interface as RandomForestModel so al_loop.py works
    without modification.
 
    The key difference from RandomForestModel:
      - fit() and uncertainty() take List[Data] (PyG graphs), not numpy arrays
      - The al_loop.py must pass smiles_list+y to build graphs, or graphs
        directly — see al_loop.py for how this is handled
 
    Parameters

    hidden_dim    : hidden state size (128)
    num_layers    : message passing rounds (3)
    dropout_p     : dropout rate (0.2)
    n_epochs      : training epochs per AL iteration (50 is default;
                    reduce to 20 for faster iteration during debugging)
    batch_size    : training batch size (32)
    lr            : learning rate (1e-3)
    pos_weight    : BCEWithLogitsLoss weight for actives (27.0 for HIV 96/4)
    mc_samples    : number of MC Dropout forward passes for uncertainty (30)
    seed          : random seed
    """
 
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout_p: float = 0.2,
        n_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        pos_weight: float = 27.0,
        mc_samples: int = 30,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.dropout_p   = dropout_p
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.pos_weight  = pos_weight
        self.mc_samples  = mc_samples
        self.seed        = seed
 
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
 
        self._model: Optional[_MPNNNet] = None
 
    # training
 
    def fit(
        self,
        graphs: List[Data],
        y: np.ndarray,
        verbose: bool = False,
    ) -> "MPNNModel":
        """
        Train the MPNN on the current labeled set.
 
        Parameters
        ----------
        graphs : list of PyG Data objects (from graph_builder.build_graph_dataset)
        y      : np.ndarray shape (N,)  labels (0/1)
                 Note: y values should already be embedded in graph.y,
                 but we accept y separately for interface consistency
        """
        torch.manual_seed(self.seed)
 
        # Attach labels to graphs (in case they're not already there)
        for g, label in zip(graphs, y):
            g.y = torch.tensor([float(label)], dtype=torch.float)
 
        # Build model fresh (called by clone_untrained pattern)
        self._model = _MPNNNet(
            atom_feat_dim=9,
            bond_feat_dim=4,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout_p=self.dropout_p,
        ).to(self.device)
 
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr,
                                     weight_decay=1e-4)
        pos_w  = torch.tensor([self.pos_weight], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
 
        # Learning rate scheduler — reduces LR when loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False
        )
 
        self._model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                logits = self._model(batch)
                labels = batch.y.to(self.device)
                loss   = loss_fn(logits, labels)
                loss.backward()
                # gradient clipping prevents exploding gradients (important
                # with NNConv's edge MLP)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
 
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
 
            if verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                print(f"    Epoch {epoch:3d}/{self.n_epochs} | loss={avg_loss:.4f}")
 
        return self
 
    # inference
 
    def predict_proba(self, graphs: List[Data]) -> np.ndarray:
        """
        Return class probabilities shape (N, 2).
        Column 0 = P(inactive), column 1 = P(active).
 
        Uses model.eval() — dropout OFF — so this is the deterministic
        point estimate, consistent with how RandomForestModel works.
        """
        assert self._model is not None, "Call fit() before predict_proba()"
        self._model.eval()
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        all_probs = []
 
        with torch.no_grad():
            for batch in loader:
                batch  = batch.to(self.device)
                logits = self._model(batch)
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
 
        p_active = np.concatenate(all_probs)  # (N,)
        return np.column_stack([1 - p_active, p_active])  # (N, 2)
 
    def predict(self, graphs: List[Data]) -> np.ndarray:
        """Return hard 0/1 predictions using threshold 0.5."""
        return (self.predict_proba(graphs)[:, 1] > 0.5).astype(int)
 
    #uncertainty (MC Dropout) 
 
    def uncertainty(
        self,
        graphs: List[Data],
        acquisition: str = 'entropy',
    ) -> np.ndarray:
        """
        Compute epistemic uncertainty via MC Dropout.
 
        Sets model to TRAIN mode (dropout ON) and runs mc_samples forward
        passes. The variance across passes is the uncertainty signal.
 
        Parameters
        ----------
        graphs      : list of PyG Data objects
        acquisition : 'entropy'   — Shannon entropy of mean prediction
                                    (standard uncertainty sampling)
                      'bald'      — BALD: mutual info between prediction and
                                    model weights (entropy - mean per-pass entropy)
                                    This is the most principled UQ method
                      'weighted'  — imbalance-aware: entropy × p̂_active
                                    biases queries toward predicted actives
 
        Returns
        -------
        scores : shape (N,)  float32  — higher = more informative to query
        """
        assert self._model is not None, "Call fit() before uncertainty()"
 
        # Keep dropout ON during inference — this is the MC Dropout trick
        self._model.train()
 
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        all_sample_probs = []  # will be (mc_samples, N)
 
        with torch.no_grad():
            for _ in range(self.mc_samples):
                batch_probs = []
                for batch in loader:
                    batch  = batch.to(self.device)
                    logits = self._model(batch)
                    probs  = torch.sigmoid(logits).cpu().numpy()
                    batch_probs.append(probs)
                all_sample_probs.append(np.concatenate(batch_probs))
 
        # Shape: (mc_samples, N)
        samples = np.stack(all_sample_probs, axis=0)
 
        # Mean prediction across MC samples
        p_mean = samples.mean(axis=0)          # (N,)
        p_mean = np.clip(p_mean, 1e-9, 1-1e-9)
 
        if acquisition == 'entropy':
            # Shannon entropy of mean prediction
            # High when model is uncertain (p_mean ≈ 0.5)
            scores = -(p_mean * np.log(p_mean) +
                      (1-p_mean) * np.log(1-p_mean))
 
        elif acquisition == 'bald':
            # BALD = H[y|x,D] - E_theta[H[y|x,theta]]
            # = entropy of mean prediction - mean entropy per sample
            # Isolates epistemic uncertainty from aleatoric noise
            # Reference: Houlsby et al. 2011
            H_mean = -(p_mean * np.log(p_mean) +
                      (1-p_mean) * np.log(1-p_mean))
            s_clip = np.clip(samples, 1e-9, 1-1e-9)
            per_sample_H = -(s_clip * np.log(s_clip) +
                            (1-s_clip) * np.log(1-s_clip))
            E_H = per_sample_H.mean(axis=0)
            scores = H_mean - E_H
 
        elif acquisition == 'weighted':
            # Imbalance-aware: entropy × predicted probability of being active
            # Biases queries toward uncertain molecules that might be actives
            # rather than uncertain inactives near the decision boundary
            H = -(p_mean * np.log(p_mean) +
                 (1-p_mean) * np.log(1-p_mean))
            scores = H * p_mean
 
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}. "
                           f"Choose from 'entropy', 'bald', 'weighted'")
 
        return scores.astype(np.float32)
 
    # clone interface
 
    def clone_untrained(self) -> "MPNNModel":
        """Return a fresh (untrained) copy with same hyperparameters."""
        return MPNNModel(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout_p=self.dropout_p,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            pos_weight=self.pos_weight,
            mc_samples=self.mc_samples,
            seed=self.seed,
            device=str(self.device),
        )
 
    @property
    def is_graph_model(self) -> bool:
        """
        Flag used by al_loop.py to decide whether to pass fingerprint arrays
        or graph lists to the model.  RF returns False, MPNN returns True.
        """
        return True