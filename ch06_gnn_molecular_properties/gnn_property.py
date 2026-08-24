"""
Chapter 6 hands-on project: message-passing neural networks (MPNNs) for
molecular property prediction.

Data: MoleculeNet's ESOL (aqueous solubility, Delaney, 2004) and FreeSolv
  (hydration free energy, Mobley & Guthrie, 2014), loaded via PyTorch
  Geometric's built-in MoleculeNet dataset (Fey & Lenssen, 2019), which
  featurizes each SMILES into a molecular graph using the Open Graph
  Benchmark's atom/bond feature scheme (Hu et al., 2020).
Models: GCN (Kipf & Welling, 2017), GAT (Velickovic et al., 2018), and an
  MPNN using edge-conditioned convolution (Gilmer et al., 2017), all
  implemented directly on top of PyTorch Geometric's message-passing
  layers - see README.md for usage and chapter.md Section 6.5 for context.
Split: random and Bemis-Murcko scaffold split (Bemis & Murcko, 1996),
  reimplemented here following the same methodology as Chapter 5's
  scaffold_split, applied to a regression task instead of classification.

See README.md for usage and chapter.md Section 6.5 for full context.
"""
import argparse
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, NNConv, global_mean_pool

RDLogger.DisableLog("rdApp.*")

DATA_DIR = Path(__file__).parent / "data"

# name -> (PyG dataset name, bundled raw CSV filename, property description)
DATASETS = {
    "esol": ("ESOL", "delaney-processed.csv", "log aqueous solubility (mol/L)"),
    "freesolv": ("FreeSolv", "SAMPL.csv", "hydration free energy (kcal/mol)"),
}

# Category sizes for PyG's OGB-style categorical atom/bond features
# (torch_geometric.utils.smiles.x_map / e_map), verified directly against
# the installed torch_geometric version rather than assumed.
NODE_FEATURE_DIMS = [119, 9, 11, 12, 9, 5, 8, 2, 2]
EDGE_FEATURE_DIMS = [22, 6, 2]

DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 3
DEFAULT_EPOCHS = 60
DEFAULT_SEED = 0


# --------------------------------------------------------------------------
# Data loading (with an offline fixture path mirroring Chapters 4-5's
# --use-cached-raw convention, since MoleculeNet otherwise downloads from
# a live S3 bucket on first use)
# --------------------------------------------------------------------------


def load_dataset(name: str, root: Path, use_cached_raw: bool = False) -> MoleculeNet:
    """Load an ESOL/FreeSolv MoleculeNet dataset. If use_cached_raw, seed
    PyG's expected raw-file location from this chapter's bundled CSV
    instead of downloading from the live DeepChem S3 bucket."""
    pyg_name, raw_filename, _ = DATASETS[name]
    root = Path(root)
    if use_cached_raw:
        raw_dir = root / pyg_name.lower() / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / raw_filename
        if not dest.exists():
            shutil.copy(DATA_DIR / raw_filename, dest)
    return MoleculeNet(root=str(root), name=pyg_name)


# --------------------------------------------------------------------------
# Featurization: embed PyG's categorical atom/bond indices (OGB scheme)
# --------------------------------------------------------------------------


class AtomEncoder(nn.Module):
    """Sums one learned embedding per categorical atom feature (atomic
    number, chirality, degree, formal charge, H count, radical electrons,
    hybridization, aromaticity, ring membership) into a single dense
    per-atom vector - the standard OGB-style categorical encoder."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(dim, hidden_dim) for dim in NODE_FEATURE_DIMS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embeddings))


class BondEncoder(nn.Module):
    """Same idea as AtomEncoder, for the three categorical bond features
    (bond type, stereo, conjugation)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(dim, hidden_dim) for dim in EDGE_FEATURE_DIMS)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return sum(emb(edge_attr[:, i]) for i, emb in enumerate(self.embeddings))


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class GCNRegressor(nn.Module):
    """Kipf & Welling (2017) graph convolution, stacked, mean-pooled, and
    regressed to a scalar property."""

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN_DIM, num_layers: int = DEFAULT_NUM_LAYERS):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> torch.Tensor:
        x = self.atom_encoder(data.x)
        for conv in self.convs:
            x = conv(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.readout(x).squeeze(-1)


class GATRegressor(nn.Module):
    """Velickovic et al. (2018) masked self-attention over graph
    neighborhoods, stacked, mean-pooled, and regressed to a scalar
    property. Attention weights are computed from node features only
    (GATConv's original formulation does not condition on edge_attr)."""

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN_DIM, num_layers: int = DEFAULT_NUM_LAYERS, heads: int = 4):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.convs = nn.ModuleList(
            GATConv(hidden_dim, hidden_dim // heads, heads=heads) for _ in range(num_layers)
        )
        self.readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> torch.Tensor:
        x = self.atom_encoder(data.x)
        for conv in self.convs:
            x = conv(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.readout(x).squeeze(-1)


class MPNNRegressor(nn.Module):
    """Gilmer et al. (2017) message passing via edge-conditioned
    convolution (NNConv): each layer's message function is itself a small
    neural network applied to the edge features, exactly as in the
    "edge network" variant of the original MPNN paper. Simplification
    relative to the original paper, stated explicitly: this
    implementation reads out with mean pooling and a plain node update,
    not the GRU node update and Set2Set readout the original paper also
    used - the edge-conditioned message function, which is the
    architecturally distinctive part, is implemented faithfully."""

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN_DIM, num_layers: int = DEFAULT_NUM_LAYERS):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            edge_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr="mean"))
        self.readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> torch.Tensor:
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr).relu()
        x = global_mean_pool(x, data.batch)
        return self.readout(x).squeeze(-1)


MODEL_BUILDERS = {"gcn": GCNRegressor, "gat": GATRegressor, "mpnn": MPNNRegressor}


# --------------------------------------------------------------------------
# Split (random and Bemis-Murcko scaffold split - same methodology as
# Chapter 5's random_split/scaffold_split, reimplemented here for a
# regression dataset of torch_geometric Data objects)
# --------------------------------------------------------------------------


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def random_split(dataset, frac_train: float = 0.8, seed: int = DEFAULT_SEED) -> tuple[list[int], list[int]]:
    idx = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(idx, train_size=frac_train, random_state=seed)
    return sorted(train_idx), sorted(test_idx)


def scaffold_split(dataset, frac_train: float = 0.8, seed: int = DEFAULT_SEED) -> tuple[list[int], list[int]]:
    """Group by Bemis-Murcko scaffold, then greedily fill the training
    set scaffold-group by scaffold-group (largest first) until
    frac_train of the data is assigned - identical methodology to
    Chapter 5's scaffold_split, applied here to a regression dataset."""
    scaffold_to_indices: dict[str, list[int]] = {}
    for i in range(len(dataset)):
        scaffold = bemis_murcko_scaffold(dataset[i].smiles)
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    rng = np.random.RandomState(seed)
    groups = list(scaffold_to_indices.values())
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]
    groups.sort(key=len, reverse=True)

    n_train_target = int(round(frac_train * len(dataset)))
    train_idx: list[int] = []
    test_idx: list[int] = []
    for group in groups:
        if len(train_idx) < n_train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)
    return sorted(train_idx), sorted(test_idx)


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float
    n_train: int
    n_test: int


def _normalize(dataset, train_idx: list[int]) -> tuple[float, float]:
    ys = torch.cat([dataset[i].y for i in train_idx]).view(-1)
    return float(ys.mean()), float(ys.std())


def train_and_evaluate(
    dataset,
    model_type: str = "mpnn",
    split_type: str = "scaffold",
    epochs: int = DEFAULT_EPOCHS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    num_layers: int = DEFAULT_NUM_LAYERS,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = DEFAULT_SEED,
) -> Metrics:
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type {model_type!r}; choose from {list(MODEL_BUILDERS)}")
    if split_type == "random":
        train_idx, test_idx = random_split(dataset, seed=seed)
    elif split_type == "scaffold":
        train_idx, test_idx = scaffold_split(dataset, seed=seed)
    else:
        raise ValueError(f"Unknown split_type {split_type!r}; choose 'random' or 'scaffold'")

    torch.manual_seed(seed)
    mean, std = _normalize(dataset, train_idx)
    train_set = [dataset[i] for i in train_idx]
    test_set = [dataset[i] for i in test_idx]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = MODEL_BUILDERS[model_type](hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch)
            target = (batch.y.view(-1) - mean) / std
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch) * std + mean
            preds.append(pred)
            targets.append(batch.y.view(-1))
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    errors = preds - targets
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return Metrics(rmse=round(rmse, 4), mae=round(mae, 4), r2=round(r2, 4), n_train=len(train_idx), n_test=len(test_idx))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GNN to predict a molecular property.")
    parser.add_argument("--dataset", choices=list(DATASETS), default="esol")
    parser.add_argument("--model", choices=list(MODEL_BUILDERS), default="mpnn")
    parser.add_argument("--split", choices=["random", "scaffold", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "data" / "_pyg_cache"))
    parser.add_argument(
        "--use-cached-raw",
        action="store_true",
        help="Seed PyG's dataset cache from this chapter's bundled CSV instead of downloading from the live DeepChem S3 bucket.",
    )
    args = parser.parse_args()

    pyg_name, _, prop_desc = DATASETS[args.dataset]
    print(f"Loading {pyg_name} ({prop_desc})...")
    dataset = load_dataset(args.dataset, Path(args.out_dir), use_cached_raw=args.use_cached_raw)
    print(f"{len(dataset)} molecular graphs.\n")

    splits = ["random", "scaffold"] if args.split == "both" else [args.split]
    for split_type in splits:
        metrics = train_and_evaluate(
            dataset,
            model_type=args.model,
            split_type=split_type,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            seed=args.seed,
        )
        print(f"[{pyg_name}, {args.model}, {split_type} split]")
        print(f"  n_train={metrics.n_train}  n_test={metrics.n_test}")
        print(f"  RMSE={metrics.rmse}  MAE={metrics.mae}  R2={metrics.r2}\n")


if __name__ == "__main__":
    main()
