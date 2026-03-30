"""
gnn_model.py
------------
Graph Neural Network for drug toxicity prediction.

Treats each molecule as a graph:
    - Nodes  = atoms  (feature vector: atomic num, degree, charge, aromaticity, …)
    - Edges  = bonds  (features: bond type, is_in_ring, is_conjugated)

Architecture:
    - 3 × GATConv (Graph Attention Network) layers with residual connections
    - Global mean + max pooling (concatenated)
    - 2-layer MLP head → sigmoid output

Usage:
    python src/train_gnn.py              # train all targets
    python src/train_gnn.py --target SR-MMP --epochs 50
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, DataLoader as PyGLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[WARNING] PyTorch Geometric not found. Run: pip install torch-geometric")

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Atom / Bond feature vectors
# ---------------------------------------------------------------------------

ATOM_TYPES = [
    "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Other"
]
HYBRIDISATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
] if RDKIT_AVAILABLE else []


def one_hot(value, choices):
    """One-hot encode `value` against `choices`, with an unknown bin at end."""
    vec = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        vec[idx] = 1
    except ValueError:
        vec[-1] = 1
    return vec


def atom_features(atom) -> list:
    """Return a 39-dim feature vector for a single RDKit atom."""
    symbol = atom.GetSymbol()
    at_types = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    feats = (
        one_hot(symbol, at_types)                          # 10
        + one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])   # 7
        + [atom.GetFormalCharge()]                          # 1
        + [atom.GetNumImplicitHs()]                         # 1
        + [int(atom.GetIsAromatic())]                       # 1
        + [int(atom.IsInRing())]                            # 1
        + one_hot(atom.GetHybridization(), HYBRIDISATIONS)  # 6
        + [atom.GetMass() / 100.0]                          # 1  (normalised)
        + [int(atom.GetNoImplicit())]                       # 1
    )
    return feats  # 29 dims


def bond_features(bond) -> list:
    """Return a 6-dim feature vector for a single RDKit bond."""
    bt = bond.GetBondTypeAsDouble()
    return [
        int(bt == 1.0),           # single
        int(bt == 2.0),           # double
        int(bt == 3.0),           # triple
        int(bt == 1.5),           # aromatic
        int(bond.IsInRing()),
        int(bond.GetIsConjugated()),
    ]  # 6 dims


def smiles_to_graph(smiles: str) -> "Data | None":
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Returns None for invalid SMILES.
    """
    if not RDKIT_AVAILABLE or not PYG_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    # Edge index + features (both directions for undirected graph)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs  += [bf, bf]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attrs,  dtype=torch.float)
    else:
        # Molecule with no bonds (e.g. single atom)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 6),  dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# GNN Model
# ---------------------------------------------------------------------------

NODE_DIM = 29    # from atom_features()
EDGE_DIM = 6     # from bond_features()


class ToxGNN(nn.Module):
    """
    Graph Attention Network for binary toxicity classification.

    Architecture:
        GATConv × 3 → global pooling (mean + max) → MLP → sigmoid
    """

    def __init__(
        self,
        in_channels: int = NODE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # GAT layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim * heads if i > 0 else hidden_dim
            is_last = (i == num_layers - 1)
            self.conv_layers.append(
                GATConv(in_dim, hidden_dim, heads=heads, concat=not is_last, dropout=dropout)
            )

        # Readout MLP (mean + max pooled → 2 * hidden_dim)
        pool_dim = hidden_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GATConv layers
        for i, conv in enumerate(self.conv_layers):
            x_new = conv(x, edge_index)
            is_last = (i == len(self.conv_layers) - 1)
            if not is_last:
                x_new = F.elu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x_new

        # Global pooling (concat mean + max)
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x      = torch.cat([x_mean, x_max], dim=1)

        # MLP head
        out = self.mlp(x)
        return torch.sigmoid(out).squeeze(1)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

class MoleculeDataset(torch.utils.data.Dataset):
    """Converts (smiles_series, labels_series) into a PyG-compatible dataset."""

    def __init__(self, smiles_series, labels_series):
        self.graphs = []
        self.labels = []
        for smi, lbl in zip(smiles_series, labels_series):
            g = smiles_to_graph(smi)
            if g is not None:
                g.y = torch.tensor([float(lbl)], dtype=torch.float)
                self.graphs.append(g)
                self.labels.append(float(lbl))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# ---------------------------------------------------------------------------
# Training & Inference helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        probs = model(batch).cpu().numpy()
        labels = batch.y.squeeze().cpu().numpy()
        all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
        all_labels.extend(labels.tolist() if labels.ndim > 0 else [labels.item()])
    return np.array(all_probs), np.array(all_labels)


def predict_gnn(model_path: str, smiles_list, device: str = "cpu") -> np.ndarray:
    """
    Load a saved GNN checkpoint and predict probabilities for a list of SMILES.

    Returns array of shape (n_valid,) — invalid SMILES get 0.5 (uncertain).
    """
    if not PYG_AVAILABLE:
        raise RuntimeError("PyTorch Geometric not installed.")

    model = ToxGNN()
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    model.to(device)

    results = []
    graphs = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
            valid_idx.append(i)

    probs = np.full(len(smiles_list), 0.5)    # default: uncertain

    if graphs:
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            p = model(batch).cpu().numpy()
        for pos, idx in enumerate(valid_idx):
            probs[idx] = float(p[pos])

    return probs
