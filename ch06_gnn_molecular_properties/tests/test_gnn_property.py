"""
Tests for the Chapter 6 hands-on project (GNN molecular property
prediction).

Model/split-logic tests use a small synthetic set of real molecules
(fast, deterministic) plus a fast subset of the bundled real ESOL/FreeSolv
fixtures (data/delaney-processed.csv, data/SAMPL.csv - the exact raw
MoleculeNet CSVs, bundled so the pipeline runs offline via
--use-cached-raw) for realistic end-to-end checks. One test exercises the
live MoleculeNet download path directly, as a reproducibility check on
that path specifically - see README.md for what to do if it is
unavailable when you run this.

Run with: pytest
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_property import (
    DATASETS,
    AtomEncoder,
    BondEncoder,
    GATRegressor,
    GCNRegressor,
    MPNNRegressor,
    bemis_murcko_scaffold,
    load_dataset,
    random_split,
    scaffold_split,
    train_and_evaluate,
)
from torch_geometric.utils import from_smiles

CH06_DIR = Path(__file__).parent.parent

# A small, deterministic, offline set of real molecular graphs spanning
# several distinct scaffolds, each labeled with an arbitrary but fixed
# regression target - enough to exercise splitting/training end to end
# without touching a live download.
_MOLECULES = [
    ("CC(=O)OC1=CC=CC=C1C(=O)O", -2.1),  # aspirin
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", -3.6),  # ibuprofen
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", -0.9),  # caffeine
    ("C1=CC=C(C=C1)O", 0.1),  # phenol
    ("C1CCCCC1", -3.4),  # cyclohexane
    ("C1CCCCC1C", -3.5),  # methylcyclohexane
    ("c1ccncc1", 0.8),  # pyridine
    ("c1ccncc1C", 0.5),  # methylpyridine
    ("c1ccsc1", -1.5),  # thiophene
    ("c1ccsc1C", -1.9),  # methylthiophene
    ("C1CCNCC1", 0.9),  # piperidine
    ("C1CCNCC1C", 0.6),  # methylpiperidine
]


def make_synthetic_dataset() -> list:
    data_list = []
    for smiles, target in _MOLECULES:
        data = from_smiles(smiles)
        data.y = torch.tensor([[target]], dtype=torch.float)
        data_list.append(data)
    return data_list


# --------------------------------------------------------------------------
# Encoders / models: shapes and gradients
# --------------------------------------------------------------------------


def test_atom_encoder_output_shape():
    data = from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    encoder = AtomEncoder(hidden_dim=16)
    out = encoder(data.x)
    assert out.shape == (data.num_nodes, 16)


def test_bond_encoder_output_shape():
    data = from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    encoder = BondEncoder(hidden_dim=16)
    out = encoder(data.edge_attr)
    assert out.shape == (data.edge_attr.size(0), 16)


@pytest.mark.parametrize("model_cls", [GCNRegressor, GATRegressor, MPNNRegressor])
def test_model_forward_produces_one_scalar_per_graph(model_cls):
    from torch_geometric.loader import DataLoader

    dataset = make_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=4)
    model = model_cls(hidden_dim=8, num_layers=2)
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == (batch.num_graphs,)


@pytest.mark.parametrize("model_cls", [GCNRegressor, GATRegressor, MPNNRegressor])
def test_model_gradients_flow_to_atom_encoder(model_cls):
    from torch_geometric.loader import DataLoader

    dataset = make_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=4)
    model = model_cls(hidden_dim=8, num_layers=2)
    batch = next(iter(loader))
    out = model(batch)
    out.sum().backward()
    grad = model.atom_encoder.embeddings[0].weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0


# --------------------------------------------------------------------------
# bemis_murcko_scaffold / splits
# --------------------------------------------------------------------------


def test_bemis_murcko_scaffold_strips_substituents():
    assert bemis_murcko_scaffold("CC(=O)OC1=CC=CC=C1C(=O)O") == "c1ccccc1"


def test_random_split_covers_all_records_without_overlap():
    dataset = make_synthetic_dataset()
    train_idx, test_idx = random_split(dataset, frac_train=0.75, seed=0)
    assert set(train_idx) & set(test_idx) == set()
    assert set(train_idx) | set(test_idx) == set(range(len(dataset)))


def test_scaffold_split_puts_no_scaffold_in_both_sets():
    dataset = make_synthetic_dataset()
    train_idx, test_idx = scaffold_split(dataset, frac_train=0.75, seed=0)
    train_scaffolds = {bemis_murcko_scaffold(dataset[i].smiles) for i in train_idx}
    test_scaffolds = {bemis_murcko_scaffold(dataset[i].smiles) for i in test_idx}
    assert train_scaffolds & test_scaffolds == set()
    assert set(train_idx) | set(test_idx) == set(range(len(dataset)))


def test_scaffold_split_is_deterministic():
    dataset = make_synthetic_dataset()
    a = scaffold_split(dataset, seed=0)
    b = scaffold_split(dataset, seed=0)
    assert a == b


# --------------------------------------------------------------------------
# train_and_evaluate: fast, synthetic, deterministic
# --------------------------------------------------------------------------


def test_train_and_evaluate_rejects_unknown_model_type():
    dataset = make_synthetic_dataset()
    with pytest.raises(ValueError):
        train_and_evaluate(dataset, model_type="not_a_model", epochs=1)


def test_train_and_evaluate_rejects_unknown_split_type():
    dataset = make_synthetic_dataset()
    with pytest.raises(ValueError):
        train_and_evaluate(dataset, split_type="not_a_split", epochs=1)


def test_train_and_evaluate_returns_finite_metrics():
    dataset = make_synthetic_dataset()
    metrics = train_and_evaluate(dataset, model_type="gcn", split_type="random", epochs=2, hidden_dim=8, seed=0)
    assert metrics.rmse >= 0
    assert metrics.mae >= 0
    assert metrics.n_train + metrics.n_test == len(dataset)


# --------------------------------------------------------------------------
# Real bundled fixture (offline, deterministic)
# --------------------------------------------------------------------------


def test_load_dataset_from_bundled_esol_fixture(tmp_path):
    dataset = load_dataset("esol", tmp_path, use_cached_raw=True)
    assert len(dataset) == 1128  # the real, full ESOL dataset
    assert dataset[0].y.shape == (1, 1)


def test_load_dataset_from_bundled_freesolv_fixture(tmp_path):
    dataset = load_dataset("freesolv", tmp_path, use_cached_raw=True)
    assert len(dataset) == 642  # the real, full FreeSolv dataset


def test_scaffold_split_on_real_esol_subset_has_no_scaffold_leakage(tmp_path):
    dataset = load_dataset("esol", tmp_path, use_cached_raw=True)
    subset = [dataset[i] for i in range(150)]
    train_idx, test_idx = scaffold_split(subset, seed=0)
    train_scaffolds = {bemis_murcko_scaffold(subset[i].smiles) for i in train_idx}
    test_scaffolds = {bemis_murcko_scaffold(subset[i].smiles) for i in test_idx}
    assert train_scaffolds & test_scaffolds == set()


def test_train_and_evaluate_on_real_esol_subset():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dataset = load_dataset("esol", Path(tmp), use_cached_raw=True)
        subset = [dataset[i] for i in range(120)]
        metrics = train_and_evaluate(subset, model_type="mpnn", split_type="scaffold", epochs=3, hidden_dim=8, seed=0)
        assert metrics.n_train + metrics.n_test == 120
        assert metrics.rmse >= 0


# --------------------------------------------------------------------------
# Live MoleculeNet download
# --------------------------------------------------------------------------


def test_load_dataset_from_live_download(tmp_path):
    dataset = load_dataset("freesolv", tmp_path, use_cached_raw=False)
    assert len(dataset) == 642


# --------------------------------------------------------------------------
# CLI (fast, offline via --use-cached-raw and few epochs)
# --------------------------------------------------------------------------


def test_cli_runs_end_to_end_offline():
    result = subprocess.run(
        [
            sys.executable,
            str(CH06_DIR / "gnn_property.py"),
            "--dataset",
            "esol",
            "--model",
            "gcn",
            "--split",
            "random",
            "--epochs",
            "2",
            "--use-cached-raw",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "RMSE=" in result.stdout
    assert "R2=" in result.stdout
