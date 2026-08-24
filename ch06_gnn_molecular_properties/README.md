# Chapter 6 Hands-on: GNN Molecular Property Prediction

Trains a graph neural network (GCN, GAT, or an edge-conditioned MPNN) in
PyTorch Geometric to predict a molecular property directly from the
molecular graph, and evaluates it under both a random train/test split
and a Bemis-Murcko scaffold split. See [`chapter.md`](chapter.md)
Section 6.5 for full context, and the chapter as a whole for the
message-passing theory this project implements directly.

## Setup

```bash
# CPU-only PyTorch (no CUDA GPU required)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run

```bash
# Default: MPNN on ESOL (aqueous solubility), both splits, live download
python gnn_property.py

# A specific model / dataset / split
python gnn_property.py --model gcn --dataset esol --split scaffold
python gnn_property.py --model gat --dataset esol --split random
python gnn_property.py --model mpnn --dataset freesolv --split both

# Offline, deterministic run against the bundled fixture (see below)
python gnn_property.py --use-cached-raw
```

Prints, for each split evaluated: the train/test size and RMSE, MAE, and
R² on the held-out test set.

`data/delaney-processed.csv` and `data/SAMPL.csv` are the real, unmodified
raw MoleculeNet CSVs for ESOL (1,128 compounds) and FreeSolv (642
compounds) respectively, so the pipeline runs fully offline and
deterministically with `--use-cached-raw` (which seeds PyTorch
Geometric's dataset cache from these files instead of downloading from
the live DeepChem S3 bucket).

Training 60 epochs on the full ESOL training split (902 molecules) takes
roughly 2-3 minutes on a single CPU core; FreeSolv (514 training
molecules) is faster. No GPU is required, but one will be used
automatically if available.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Most tests run fast, offline, synthetic-graph checks (2-3 epochs, tiny
hidden dimensions) or a small offline subset of the bundled fixtures;
one test calls the live MoleculeNet/DeepChem download directly, as a
reproducibility check on that path specifically.

## Google Colab

Colab's default runtime does not include `torch_geometric` or `rdkit`;
`torch` is preinstalled. Run
`!pip install torch_geometric rdkit` in the first cell. A GPU runtime
will be used automatically if selected, but is not required - CPU
training completes well within a Colab session's free-tier limits.
