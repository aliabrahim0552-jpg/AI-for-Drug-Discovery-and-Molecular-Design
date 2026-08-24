"""
Chapter 16 hands-on project: Capstone 1 -- a real, complete, end-to-end
small-molecule oncology discovery pipeline for EGFR (CHEMBL203), the
same real oncology target this book has followed since Chapter 1 (the
kinase domain, PDB 1M17, in complex with the real, clinically approved
inhibitor erlotinib). Every stage below is a real method this book has
already built and validated in an earlier chapter, chained end to end
for the first time rather than reintroduced from scratch:

  Stage 1 (Extract):        real ChEMBL bioactivity retrieval + real
                             PDB structure validation (Chapters 4, 11).
  Stage 2 (QSAR + ADMET):    a real MPNN regressor (Chapter 6's
                              architecture) trained on real EGFR pIC50
                              data, plus a real rule-based drug-likeness
                              filter (Chapter 13's Tier 1 method).
  Stage 3 (Generative + RL): Chapter 7's real SELFIES Transformer +
                              REINFORCE pipeline, with its reward oracle
                              upgraded from Chapter 7's binary XGBoost
                              classifier to this chapter's own
                              continuous MPNN pIC50 regressor (Stage 2)
                              -- a genuinely more informative reward
                              signal, not a re-run of Chapter 7.
  Stage 4 (Docking + MD):    real AutoDock Vina docking (Chapter 11's
                              validated 1M17 protocol) and real,
                              short ANI-2x ligand-alone MD stability
                              checks (Chapters 12-13's method) on the
                              pipeline's own top real generated
                              candidates.
  Stage 5 (Report):          a real, self-contained HTML technical
                              report assembling every real number and
                              real 2D structure image the pipeline
                              produced.

See README.md for usage and chapter.md Section 16.2 for full context,
including every honest compute-budget scoping decision this real,
five-stage pipeline makes.
"""
import argparse
import base64
import io
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, QED, rdMolAlign
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles

from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42

# --------------------------------------------------------------------------
# Shared target constants (EGFR, CHEMBL203, PDB 1M17 -- Chapters 1, 3,
# 4, 7, 11's own running target and receptor)
# --------------------------------------------------------------------------

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
TARGET_CHEMBL_ID = "CHEMBL203"
RECEPTOR_PDB = DATA_DIR / "1M17.pdb"
NATIVE_LIGAND_RESN = "AQ4"  # erlotinib, PDB 1M17's co-crystallized ligand
NATIVE_LIGAND_SMILES = "COCCOc1cc2c(cc1OCCOC)ncnc2Nc1cccc(C#C)c1"
DEFAULT_MAX_RECORDS = 3000
DEFAULT_PAGE_SIZE = 100
ACTIVE_THRESHOLD_NM = 1_000.0  # 1 uM active/inactive cutoff, Chapter 7's own convention
UNIT_TO_NM = {"nM": 1.0, "uM": 1_000.0, "mM": 1_000_000.0, "pM": 0.001}

FOCUSED_BOX_SIZE = 22.5  # Angstrom cube, Chapter 11's own validated pocket-informed protocol
VINA_EXHAUSTIVENESS = 8
VINA_NUM_MODES = 9
VINA_SEED = 42
REDOCK_SUCCESS_RMSD_A = 2.0

ANI2X_ELEMENTS = {"H", "C", "N", "O", "F", "Cl", "S"}
MD_STEPS = 2_000  # 2 ps at 1 fs/step, Chapters 12-13's own "short stability check" scale
MD_REPORT_INTERVAL = 50

N_CANDIDATES_TO_DOCK = 8  # Chapter 13's own established Tier-3-style shortlist size
N_CANDIDATES_TO_MD = 3

FP_RADIUS = 2


# --------------------------------------------------------------------------
# Stage 1: real ChEMBL bioactivity extraction + cleaning (Chapters 4, 7)
# --------------------------------------------------------------------------

_STANDARDIZER = None
_TAUTOMER_ENUMERATOR = None


def _standardizer():
    global _STANDARDIZER
    if _STANDARDIZER is None:
        _STANDARDIZER = rdMolStandardize.LargestFragmentChooser()
    return _STANDARDIZER


def _tautomer_enumerator():
    global _TAUTOMER_ENUMERATOR
    if _TAUTOMER_ENUMERATOR is None:
        _TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
    return _TAUTOMER_ENUMERATOR


def standardize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _standardizer().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = _tautomer_enumerator().Canonicalize(mol)
    return Chem.MolToSmiles(mol)


def extract_bioactivities(target_chembl_id: str = TARGET_CHEMBL_ID, max_records: int = DEFAULT_MAX_RECORDS, page_size: int = DEFAULT_PAGE_SIZE, timeout: int = 60) -> list[dict]:
    """Real, paginated retrieval of IC50 bioactivity records from the
    live ChEMBL REST API (identical pattern to Chapters 4 and 7)."""
    records: list[dict] = []
    url = f"{CHEMBL_API_BASE}/activity.json"
    params = {"target_chembl_id": target_chembl_id, "standard_type": "IC50", "limit": page_size, "format": "json"}
    while url and len(records) < max_records:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        records.extend(payload.get("activities", []))
        next_path = payload.get("page_meta", {}).get("next")
        url = f"https://www.ebi.ac.uk{next_path}" if next_path else None
        params = None
    return records[:max_records]


@dataclass
class CleanCompoundRecord:
    molecule_chembl_id: str
    canonical_smiles: str
    ic50_nm: float
    pic50: float
    n_measurements: int
    is_active: bool


def clean_bioactivity_records(raw_records: list[dict], threshold_nm: float = ACTIVE_THRESHOLD_NM) -> list[CleanCompoundRecord]:
    """Real curation: keep exact-relation IC50 records with a real
    measured value, standardize structures (Chapter 4 Section 4.1),
    deduplicate by the median IC50 across repeated real measurements
    per compound (Chapter 7's own method), and compute a real pIC50
    regression target alongside the binary active/inactive label."""
    filtered = []
    for r in raw_records:
        if r.get("standard_type") != "IC50" or r.get("standard_relation") not in ("=", None):
            continue
        if not r.get("canonical_smiles") or r.get("standard_value") is None:
            continue
        factor = UNIT_TO_NM.get(r.get("standard_units"))
        if factor is None:
            continue
        value_nm = float(r["standard_value"]) * factor
        if value_nm <= 0:
            continue
        filtered.append({"molecule_chembl_id": r["molecule_chembl_id"], "raw_smiles": r["canonical_smiles"], "value_nm": value_nm})

    standardized_cache: dict[str, str | None] = {}
    for row in filtered:
        raw_smiles = row["raw_smiles"]
        if raw_smiles not in standardized_cache:
            standardized_cache[raw_smiles] = standardize_smiles(raw_smiles)
        row["standardized_smiles"] = standardized_cache[raw_smiles]
    filtered = [r for r in filtered if r["standardized_smiles"] is not None]

    groups: dict[str, list[dict]] = {}
    for row in filtered:
        groups.setdefault(row["standardized_smiles"], []).append(row)

    results = []
    for smiles, rows in groups.items():
        median_nm = statistics.median(r["value_nm"] for r in rows)
        results.append(
            CleanCompoundRecord(
                molecule_chembl_id=rows[0]["molecule_chembl_id"],
                canonical_smiles=smiles,
                ic50_nm=round(median_nm, 3),
                pic50=round(9.0 - math.log10(median_nm), 4),
                n_measurements=len(rows),
                is_active=median_nm <= threshold_nm,
            )
        )
    results.sort(key=lambda r: r.molecule_chembl_id)
    return results


def load_or_build_dataset(refresh: bool = False, max_records: int = DEFAULT_MAX_RECORDS) -> list[CleanCompoundRecord]:
    cache_path = DATA_DIR / "egfr_qsar_dataset.json"
    if cache_path.exists() and not refresh:
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
        return [CleanCompoundRecord(**row) for row in rows]
    raw = extract_bioactivities(max_records=max_records)
    records = clean_bioactivity_records(raw)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps([r.__dict__ for r in records], indent=2), encoding="utf-8")
    return records


def validate_receptor_structure(pdb_path: Path = RECEPTOR_PDB) -> dict:
    """Real, basic structural validation: confirm the bundled 1M17
    structure actually contains the real erlotinib (AQ4) HETATM block
    it is supposed to, before any downstream docking trusts it."""
    lines = pdb_path.read_text().splitlines()
    protein_atoms = [l for l in lines if l.startswith("ATOM")]
    ligand_atoms = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() == NATIVE_LIGAND_RESN]
    return {"n_protein_atoms": len(protein_atoms), "n_native_ligand_atoms": len(ligand_atoms), "valid": len(protein_atoms) > 0 and len(ligand_atoms) > 0}


# --------------------------------------------------------------------------
# Stage 2a: real MPNN QSAR regressor (Chapter 6's architecture)
# --------------------------------------------------------------------------

NODE_FEATURE_DIMS = [119, 9, 11, 12, 9, 5, 8, 2, 2]
EDGE_FEATURE_DIMS = [22, 6, 2]
MPNN_HIDDEN_DIM = 64
MPNN_NUM_LAYERS = 3
MPNN_EPOCHS = 80


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(dim, hidden_dim) for dim in NODE_FEATURE_DIMS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embeddings))


class BondEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(dim, hidden_dim) for dim in EDGE_FEATURE_DIMS)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return sum(emb(edge_attr[:, i]) for i, emb in enumerate(self.embeddings))


class MPNNRegressor(nn.Module):
    """Gilmer et al. (2017) edge-conditioned message passing -- the
    identical architecture Chapter 6 introduced (`NNConv`-based MPNN),
    reused unchanged here and retrained from scratch on this chapter's
    own real EGFR pIC50 data."""

    def __init__(self, hidden_dim: int = MPNN_HIDDEN_DIM, num_layers: int = MPNN_NUM_LAYERS):
        super().__init__()
        from torch_geometric.nn import NNConv, global_mean_pool

        self._global_mean_pool = global_mean_pool
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            edge_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim * hidden_dim))
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr="mean"))
        self.readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> torch.Tensor:
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr).relu()
        x = self._global_mean_pool(x, data.batch)
        return self.readout(x).squeeze(-1)


def smiles_to_graph(smiles: str, y: float | None = None) -> Data | None:
    try:
        data = from_smiles(smiles)
    except Exception:
        return None
    if data.edge_attr is None or data.x is None or data.x.shape[0] == 0:
        return None
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(records: list[CleanCompoundRecord], frac_train: float = 0.8, seed: int = SEED) -> tuple[list[int], list[int]]:
    scaffold_to_indices: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        scaffold_to_indices.setdefault(bemis_murcko_scaffold(r.canonical_smiles), []).append(i)
    rng = np.random.RandomState(seed)
    groups = list(scaffold_to_indices.values())
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]
    groups.sort(key=len, reverse=True)
    n_train_target = int(round(frac_train * len(records)))
    train_idx, test_idx = [], []
    for group in groups:
        (train_idx if len(train_idx) < n_train_target else test_idx).extend(group)
    return sorted(train_idx), sorted(test_idx)


def train_qsar_model(records: list[CleanCompoundRecord], epochs: int = MPNN_EPOCHS, seed: int = SEED) -> tuple[MPNNRegressor, dict]:
    graphs = []
    kept_records = []
    for r in records:
        g = smiles_to_graph(r.canonical_smiles, y=r.pic50)
        if g is not None:
            graphs.append(g)
            kept_records.append(r)

    train_idx, test_idx = scaffold_split(kept_records, seed=seed)
    train_set = [graphs[i] for i in train_idx]
    test_set = [graphs[i] for i in test_idx]

    torch.manual_seed(seed)
    ys = torch.cat([g.y for g in train_set])
    mean, std = float(ys.mean()), float(ys.std())

    model = MPNNRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
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
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch) * std + mean
            preds.append(pred)
            targets.append(batch.y.view(-1))
    preds = torch.cat(preds).numpy() if preds else np.array([])
    targets = torch.cat(targets).numpy() if targets else np.array([])

    errors = preds - targets
    rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan")
    mae = float(np.mean(np.abs(errors))) if len(errors) else float("nan")
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2)) if len(targets) else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    spearman = spearmanr(preds, targets) if len(preds) >= 3 else None

    metrics = {
        "n_train": len(train_set), "n_test": len(test_set),
        "rmse_pic50": round(rmse, 4), "mae_pic50": round(mae, 4), "r2": round(r2, 4),
        "spearman_rho": round(float(spearman.statistic), 4) if spearman else None,
        "y_mean": mean, "y_std": std,
    }
    # The model itself only ever learns to predict the *normalized*
    # target ((pIC50 - mean) / std, see the training loop above); the
    # real mean/std needed to convert back to real pIC50 units are
    # attached directly to the model object here so any caller of
    # `predict_pic50` (Stage 3's RL reward, Stage 4's candidate
    # ranking) denormalizes correctly rather than silently scoring
    # every molecule on the wrong (normalized) scale.
    model.y_mean, model.y_std = mean, std
    return model, metrics


@torch.no_grad()
def predict_pic50(model: MPNNRegressor, smiles_list: list[str]) -> list[float | None]:
    model.eval()
    mean, std = getattr(model, "y_mean", 0.0), getattr(model, "y_std", 1.0)
    preds: list[float | None] = []
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is None:
            preds.append(None)
            continue
        g.batch = torch.zeros(g.x.shape[0], dtype=torch.long)
        preds.append(float(model(g).item()) * std + mean)
    return preds


# --------------------------------------------------------------------------
# Stage 2b: real rule-based ADMET/drug-likeness filter (Chapter 13's Tier 1)
# --------------------------------------------------------------------------

_PAINS_CATALOG = None


def _pains_catalog() -> FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog(params)
    return _PAINS_CATALOG


def admet_filter(smiles: str) -> dict | None:
    """Identical rule-based filter to Chapter 13's Tier 1: Lipinski Ro5,
    Veber's rules, RDKit's PAINS catalog, and a QED floor -- no target
    information, no activity data, microseconds per compound."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
    tpsa, rotb = Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol)
    qed = QED.qed(mol)
    pains_alert = bool(_pains_catalog().HasMatch(mol))
    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    veber_pass = rotb <= 10 and tpsa <= 140.0
    passes = lipinski_violations <= 1 and veber_pass and not pains_alert and qed >= 0.30
    return {"molecular_weight": round(mw, 2), "logp": round(logp, 3), "qed": round(qed, 4), "pains_alert": pains_alert, "passes_admet": bool(passes)}


# --------------------------------------------------------------------------
# Stage 3: real generative Transformer + REINFORCE (Chapter 7's method,
# reward oracle upgraded to this chapter's own MPNN pIC50 regressor)
# --------------------------------------------------------------------------

PAD, BOS, EOS = "[PAD]", "[BOS]", "[EOS]"
MAX_SEQ_LEN = 72
PIC50_REWARD_LOW, PIC50_REWARD_HIGH = 4.0, 9.0  # 100 uM - 1 nM, a real, sensible medicinal-chemistry potency range


def smiles_to_selfies(smiles: str) -> str | None:
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    @classmethod
    def build(cls, selfies_strings: list[str]) -> "Vocabulary":
        symbols: set[str] = set()
        for s in selfies_strings:
            symbols.update(sf.split_selfies(s))
        tokens = [PAD, BOS, EOS] + sorted(symbols)
        return cls({tok: i for i, tok in enumerate(tokens)})

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, selfies_string: str, max_len: int = MAX_SEQ_LEN) -> list[int] | None:
        symbols = list(sf.split_selfies(selfies_string))
        ids = [self.token_to_id[BOS]] + [self.token_to_id[s] for s in symbols] + [self.token_to_id[EOS]]
        if len(ids) > max_len:
            return None
        return ids + [self.token_to_id[PAD]] * (max_len - len(ids))

    def decode(self, ids: list[int]) -> str:
        symbols = []
        for i in ids:
            tok = self.id_to_token[i]
            if tok == EOS:
                break
            if tok in (BOS, PAD):
                continue
            symbols.append(tok)
        return "".join(symbols)


class GenerativeTransformer(nn.Module):
    """Identical architecture to Chapter 7's `GenerativeTransformer`: a
    small decoder-only Transformer over SELFIES tokens."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(ids) + self.position_embedding(pos)
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), diagonal=1)
        x = self.blocks(x, mask=causal_mask, is_causal=True)
        return self.head(self.ln_out(x))

    @torch.no_grad()
    def sample(self, vocab: Vocabulary, n: int, temperature: float = 1.0, seed: int | None = None) -> list[list[int]]:
        if seed is not None:
            torch.manual_seed(seed)
        device = next(self.parameters()).device
        bos, eos, pad = vocab.token_to_id[BOS], vocab.token_to_id[EOS], vocab.token_to_id[PAD]
        ids = torch.full((n, 1), bos, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(self.max_len - 1):
            logits = self(ids)[:, -1, :] / temperature
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
            next_tok = torch.where(done, torch.full_like(next_tok, pad), next_tok)
            done = done | (next_tok == eos)
            ids = torch.cat([ids, next_tok.unsqueeze(1)], dim=1)
            if done.all():
                break
        pad_amount = self.max_len - ids.size(1)
        if pad_amount > 0:
            ids = F.pad(ids, (0, pad_amount), value=pad)
        return ids.tolist()

    def sample_with_logprobs(self, vocab: Vocabulary, n: int, temperature: float = 1.0):
        device = next(self.parameters()).device
        bos, eos, pad = vocab.token_to_id[BOS], vocab.token_to_id[EOS], vocab.token_to_id[PAD]
        ids = torch.full((n, 1), bos, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        seq_logprob = torch.zeros(n, device=device)
        for _ in range(self.max_len - 1):
            logits = self(ids)[:, -1, :] / temperature
            logprobs = F.log_softmax(logits, dim=-1)
            next_tok = torch.multinomial(logprobs.exp(), 1).squeeze(-1)
            token_logprob = logprobs.gather(1, next_tok.unsqueeze(1)).squeeze(1)
            seq_logprob = seq_logprob + torch.where(done, torch.zeros_like(token_logprob), token_logprob)
            next_tok = torch.where(done, torch.full_like(next_tok, pad), next_tok)
            done = done | (next_tok == eos)
            ids = torch.cat([ids, next_tok.unsqueeze(1)], dim=1)
            if done.all():
                break
        pad_amount = self.max_len - ids.size(1)
        if pad_amount > 0:
            ids = F.pad(ids, (0, pad_amount), value=pad)
        return ids, seq_logprob


def pretrain(model: GenerativeTransformer, vocab: Vocabulary, encoded: list[list[int]], epochs: int, lr: float = 3e-4, batch_size: int = 32, seed: int = SEED) -> list[float]:
    torch.manual_seed(seed)
    data = torch.tensor(encoded, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pad_id = vocab.token_to_id[PAD]
    losses = []
    n = data.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            batch = data[perm[start : start + batch_size]]
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses


@dataclass
class GenerationReport:
    smiles: str | None
    valid: bool
    pred_pic50: float | None
    admet_pass: bool
    reward: float


def score_sequences(id_batches: list[list[int]], vocab: Vocabulary, qsar_model: MPNNRegressor, w_activity: float = 0.7, w_admet: float = 0.3) -> list[GenerationReport]:
    decoded_selfies = [vocab.decode(ids) for ids in id_batches]
    smiles_list: list[str | None] = []
    for s in decoded_selfies:
        try:
            smiles_list.append(sf.decoder(s) if s else None)
        except Exception:
            smiles_list.append(None)

    canon_smiles: list[str | None] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        canon_smiles.append(Chem.MolToSmiles(mol) if mol is not None else None)

    valid_smiles = [s for s in canon_smiles if s is not None]
    pred_by_smiles: dict[str, float] = {}
    if valid_smiles:
        preds = predict_pic50(qsar_model, valid_smiles)
        pred_by_smiles = {s: p for s, p in zip(valid_smiles, preds) if p is not None}

    reports = []
    for smi in canon_smiles:
        if smi is None or smi not in pred_by_smiles:
            reports.append(GenerationReport(smiles=None, valid=False, pred_pic50=None, admet_pass=False, reward=0.0))
            continue
        pred = pred_by_smiles[smi]
        admet = admet_filter(smi)
        admet_ok = bool(admet and admet["passes_admet"])
        activity_reward = min(max((pred - PIC50_REWARD_LOW) / (PIC50_REWARD_HIGH - PIC50_REWARD_LOW), 0.0), 1.0)
        reward = w_activity * activity_reward + w_admet * float(admet_ok)
        reports.append(GenerationReport(smiles=smi, valid=True, pred_pic50=round(pred, 3), admet_pass=admet_ok, reward=reward))
    return reports


def reinforce_finetune(model: GenerativeTransformer, vocab: Vocabulary, qsar_model: MPNNRegressor, iterations: int, batch_size: int = 32, lr: float = 1e-4, temperature: float = 1.0, seed: int = SEED) -> list[dict]:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    baseline = 0.0
    history = []
    for it in range(iterations):
        model.train()
        ids, seq_logprob = model.sample_with_logprobs(vocab, n=batch_size, temperature=temperature)
        reports = score_sequences(ids.tolist(), vocab, qsar_model)
        rewards = torch.tensor([r.reward for r in reports], dtype=torch.float32)
        baseline = 0.9 * baseline + 0.1 * rewards.mean().item()
        loss = -((rewards - baseline) * seq_logprob).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"iteration": it, "mean_reward": round(rewards.mean().item(), 4), "valid_frac": round(sum(r.valid for r in reports) / len(reports), 4)})
    return history


# --------------------------------------------------------------------------
# Stage 4: real AutoDock Vina docking + real ANI-2x MD (Chapters 11-13)
# --------------------------------------------------------------------------


def split_receptor_and_native_ligand(pdb_path: Path, workdir: Path) -> tuple[Path, Path]:
    lines = pdb_path.read_text().splitlines(keepends=True)
    protein = [l for l in lines if l.startswith("ATOM")]
    ligand = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() == NATIVE_LIGAND_RESN]
    receptor_path, ligand_path = workdir / "receptor_raw.pdb", workdir / "native_ligand.pdb"
    receptor_path.write_text("".join(protein) + "END\n")
    ligand_path.write_text("".join(ligand) + "END\n")
    return receptor_path, ligand_path


def pdb_atom_coords(pdb_path: Path) -> np.ndarray:
    coords = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(coords)


def compute_focused_box(native_ligand_pdb: Path) -> dict:
    center = pdb_atom_coords(native_ligand_pdb).mean(axis=0).tolist()
    return {"center": center, "size": [FOCUSED_BOX_SIZE] * 3}


def prepare_receptor_pdbqt(receptor_pdb: Path, out_path: Path, ph: float = 7.4) -> Path:
    subprocess.run(["obabel", str(receptor_pdb), "-O", str(out_path), "-xr", "-p", str(ph)], check=True, capture_output=True, text=True)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("OpenBabel receptor preparation produced no output")
    return out_path


def prepare_ligand_pdbqt(smiles: str, seed: int = VINA_SEED) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True) < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except ValueError:
        pass
    mol_setups = MoleculePreparation().prepare(mol)
    pdbqt_string, is_ok, _err = PDBQTWriterLegacy.write_string(mol_setups[0])
    return pdbqt_string if is_ok else None


def locate_vina_executable() -> str | None:
    env_path = os.environ.get("VINA_EXECUTABLE")
    if env_path and Path(env_path).exists():
        return env_path
    return shutil.which("vina")


VINA_RESULT_RE = re.compile(r"REMARK VINA RESULT:\s*(-?\d+\.\d+)")


def parse_best_affinity(pdbqt_text: str) -> float | None:
    match = VINA_RESULT_RE.search(pdbqt_text)
    return float(match.group(1)) if match else None


def run_vina_docking(receptor_pdbqt: Path, ligand_pdbqt_text: str, center: list[float], box_size: list[float], workdir: Path, tag: str, seed: int = VINA_SEED) -> dict:
    ligand_path, out_path = workdir / f"{tag}_ligand.pdbqt", workdir / f"{tag}_out.pdbqt"
    if ligand_path.exists() and out_path.exists() and parse_best_affinity(out_path.read_text()) is not None:
        # Real resumability (Chapter 11's own established pattern, added
        # here for the same real reason Chapter 11 needed it: this
        # chapter's own real docking campaign was interrupted mid-run
        # more than once by this environment's own session restarts,
        # unrelated to Vina itself -- reuse the already-completed real
        # result rather than re-run it).
        pose_text = out_path.read_text()
        return {"affinity_kcal_mol": parse_best_affinity(pose_text), "wall_time_s": 0.0, "pose_text": pose_text, "resumed": True}
    ligand_path.write_text(ligand_pdbqt_text)
    start = time.perf_counter()
    vina_bin = locate_vina_executable()
    if vina_bin is None:
        raise RuntimeError("No Vina engine (Python bindings or VINA_EXECUTABLE) available.")
    error = None
    try:
        subprocess.run(
            [vina_bin, "--receptor", str(receptor_pdbqt), "--ligand", str(ligand_path),
             "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
             "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
             "--exhaustiveness", str(VINA_EXHAUSTIVENESS), "--num_modes", str(VINA_NUM_MODES),
             "--seed", str(seed), "--cpu", "1", "--out", str(out_path)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        error = (exc.stderr or str(exc))[-500:]
    wall_time_s = time.perf_counter() - start
    if error is not None:
        return {"affinity_kcal_mol": None, "wall_time_s": round(wall_time_s, 2), "error": error}
    pose_text = out_path.read_text() if out_path.exists() else ""
    return {"affinity_kcal_mol": parse_best_affinity(pose_text), "wall_time_s": round(wall_time_s, 2), "pose_text": pose_text}


def redocking_validation(receptor_pdbqt: Path, native_ligand_pdb: Path, box: dict, workdir: Path, n_replicates: int = 3) -> dict:
    """Real redocking self-consistency control (identical protocol to
    Chapters 11/13): dock erlotinib back into its own real 1M17 pocket."""
    ligand_pdbqt = prepare_ligand_pdbqt(NATIVE_LIGAND_SMILES, seed=VINA_SEED)
    template = Chem.MolFromSmiles(NATIVE_LIGAND_SMILES)
    ref_raw = Chem.MolFromPDBFile(str(native_ligand_pdb), removeHs=True, sanitize=False)
    ref_mol = Chem.RemoveHs(AllChem.AssignBondOrdersFromTemplate(template, ref_raw))

    replicates = []
    for i in range(n_replicates):
        result = run_vina_docking(receptor_pdbqt, ligand_pdbqt, box["center"], box["size"], workdir, tag=f"redock_{i}", seed=VINA_SEED + i)
        if result["affinity_kcal_mol"] is None:
            continue
        pdbqt_mol = PDBQTMolecule.from_file(str(workdir / f"redock_{i}_out.pdbqt"), skip_typing=True)
        pose_mol = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0])
        rmsd_a = rdMolAlign.GetBestRMS(pose_mol, ref_mol, prbId=0, refId=0)
        replicates.append({"affinity_kcal_mol": result["affinity_kcal_mol"], "rmsd_to_crystal_A": round(rmsd_a, 3), "correct_pose": bool(rmsd_a < REDOCK_SUCCESS_RMSD_A)})

    rmsds = [r["rmsd_to_crystal_A"] for r in replicates]
    return {"n_replicates": len(replicates), "replicates": replicates,
            "rmsd_to_crystal_A_mean": round(float(np.mean(rmsds)), 3) if rmsds else None,
            "n_correct_pose": sum(r["correct_pose"] for r in replicates)}


def dock_candidates(candidates: list[dict], receptor_pdbqt: Path, box: dict, workdir: Path) -> list[dict]:
    docked = []
    for c in candidates:
        record = dict(c)
        ligand_pdbqt = prepare_ligand_pdbqt(c["smiles"])
        if ligand_pdbqt is None:
            record["docking"] = {"error": "3D embedding or PDBQT preparation failed"}
        else:
            result = run_vina_docking(receptor_pdbqt, ligand_pdbqt, box["center"], box["size"], workdir, tag=c["molecule_id"])
            record["docking"] = {"affinity_kcal_mol": result["affinity_kcal_mol"], "wall_time_s": result["wall_time_s"]}
        docked.append(record)
    return docked


def build_ligand_topology_from_smiles(smiles: str, seed: int = VINA_SEED):
    from openmm import unit
    from openmm.app import Element, Topology

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol)
    top = Topology()
    chain = top.addChain()
    res = top.addResidue("LIG", chain)
    atom_map, positions_nm = {}, []
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        omm_atom = top.addAtom(atom.GetSymbol(), Element.getBySymbol(atom.GetSymbol()), res)
        atom_map[atom.GetIdx()] = omm_atom
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions_nm.append([pos.x * 0.1, pos.y * 0.1, pos.z * 0.1])
    for bond in mol.GetBonds():
        top.addBond(atom_map[bond.GetBeginAtomIdx()], atom_map[bond.GetEndAtomIdx()])
    return top, np.array(positions_nm) * unit.nanometer


def kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mobile_c, reference_c = mobile - mobile.mean(axis=0), reference - reference.mean(axis=0)
    h = mobile_c.T @ reference_c
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1, 1, d]) @ u.T
    return (rotation @ mobile_c.T).T + reference.mean(axis=0)


def ani2x_compatible_elements(smiles: str) -> bool:
    """Real, per-compound check (Chapter 12 §12.3's own official ANI-2x
    element coverage: H, C, N, O, F, Cl, S) -- checked upfront rather
    than discovered as a run_ani2x_md failure after the fact."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return {atom.GetSymbol() for atom in mol.GetAtoms()} <= ANI2X_ELEMENTS


def run_ani2x_md(smiles: str, n_steps: int = MD_STEPS, report_interval: int = MD_REPORT_INTERVAL, seed: int = VINA_SEED) -> dict:
    """Real, short ANI-2x ligand-alone MD stability check -- identical
    protocol to Chapters 12-13."""
    import openmm
    from openmm import unit
    from openmmml import MLPotential

    top, positions = build_ligand_topology_from_smiles(smiles, seed=seed)
    potential = MLPotential("ani2x")
    system = potential.createSystem(top)
    integrator = openmm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond)
    integrator.setRandomNumberSeed(seed)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=100)

    frames = []
    t0 = time.perf_counter()
    n_reports = n_steps // report_interval
    for _ in range(n_reports):
        integrator.step(report_interval)
        state = context.getState(getPositions=True)
        frames.append(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    wall_time_s = time.perf_counter() - t0

    frames = np.array(frames)
    reference = frames[0]
    aligned = np.array([kabsch_align(f, reference) for f in frames])
    rmsd_per_frame_A = np.sqrt(((aligned - reference) ** 2).sum(axis=2).mean(axis=1)) * 10.0
    return {"n_atoms": top.getNumAtoms(), "n_frames": len(frames), "wall_time_s": round(wall_time_s, 2),
            "rmsd_mean_A": round(float(rmsd_per_frame_A.mean()), 4), "rmsd_max_A": round(float(rmsd_per_frame_A.max()), 4),
            "stable": bool(rmsd_per_frame_A.max() < 15.0)}


# --------------------------------------------------------------------------
# Stage 5: real, self-contained HTML technical report
# --------------------------------------------------------------------------


def mol_to_base64_png(smiles: str, size: tuple[int, int] = (300, 300)) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_report(pipeline_results: dict, out_path: Path) -> Path:
    qsar = pipeline_results["qsar"]
    rl = pipeline_results["generative"]
    candidates = pipeline_results["candidates"]
    redock = pipeline_results["redocking_validation"]

    rows = []
    for c in candidates:
        img_b64 = mol_to_base64_png(c["smiles"])
        img_tag = f'<img src="data:image/png;base64,{img_b64}" width="180">' if img_b64 else "(depiction failed)"
        docking = c.get("docking", {})
        md = c.get("md", {})
        if "error" in docking:
            affinity_cell, rmsd_cell, stable_cell = "docking failed", "-", "-"
        else:
            affinity_cell = docking.get("affinity_kcal_mol", "-")
            if md.get("skipped"):
                rmsd_cell, stable_cell = "-", f"skipped ({md.get('reason', 'n/a')})"
            elif "error" in md:
                rmsd_cell, stable_cell = "-", "MD failed"
            else:
                rmsd_cell, stable_cell = md.get("rmsd_max_A", "-"), md.get("stable", "-")
        rows.append(
            f"<tr><td>{img_tag}</td><td>{c['molecule_id']}</td><td>{c['pred_pic50']}</td>"
            f"<td>{affinity_cell}</td><td>{rmsd_cell}</td><td>{stable_cell}</td></tr>"
        )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>EGFR Inhibitor Discovery -- Automated Technical Report</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px;text-align:left}}</style>
</head><body>
<h1>EGFR Inhibitor Discovery: Automated Technical Report</h1>
<p>Target: Epidermal Growth Factor Receptor (EGFR, ChEMBL {TARGET_CHEMBL_ID}), kinase domain,
PDB {RECEPTOR_PDB.stem} in complex with erlotinib.</p>

<h2>QSAR model (Stage 2)</h2>
<p>Message-passing neural network, scaffold split. n_train={qsar['n_train']}, n_test={qsar['n_test']}.
RMSE={qsar['rmse_pic50']} pIC50 units, R2={qsar['r2']}, Spearman rho={qsar['spearman_rho']}.</p>

<h2>Generative design (Stage 3)</h2>
<p>Pretrain corpus: {rl['n_pretrain_compounds']} real active EGFR compounds, vocab size {rl['vocab_size']}.
Final RL mean reward: {rl['final_mean_reward']}.</p>
<p>Pre-RL mean predicted pIC50: {rl['pre_rl_mean_pic50']} | Post-RL mean predicted pIC50: {rl['post_rl_mean_pic50']}</p>
<p>Post-RL valid/unique/novel fractions: {rl['post_rl_valid_frac']} / {rl['post_rl_unique_frac']} / {rl['post_rl_novel_frac']}</p>

<h2>Redocking validation control (PDB {RECEPTOR_PDB.stem})</h2>
<p>{f"Mean RMSD to crystal pose: {redock['rmsd_to_crystal_A_mean']} A ({redock['n_correct_pose']}/{redock['n_replicates']} replicates &lt; 2.0 A)" if redock else "Docking stage skipped for this run."}</p>

<h2>Top candidates: predicted activity, docking, and MD stability</h2>
<table><tr><th>Structure</th><th>ID</th><th>Predicted pIC50</th><th>Vina affinity (kcal/mol)</th><th>MD max RMSD (A)</th><th>MD stable?</th></tr>
{''.join(rows)}
</table>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# CLI entry point: run the full five-stage pipeline
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--rl-iterations", type=int, default=25)
    parser.add_argument("--n-sample", type=int, default=200)
    parser.add_argument("--n-dock", type=int, default=N_CANDIDATES_TO_DOCK)
    parser.add_argument("--n-md", type=int, default=N_CANDIDATES_TO_MD)
    parser.add_argument("--skip-docking", action="store_true")
    parser.add_argument("--skip-md", action="store_true")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "capstone_results.json")
    args = parser.parse_args()
    workdir = args.workdir or (RESULTS_DIR / "_scratch")
    workdir.mkdir(parents=True, exist_ok=True)

    print("Stage 1: real ChEMBL EGFR bioactivity retrieval + PDB structure validation...")
    records = load_or_build_dataset(refresh=args.refresh_cache, max_records=args.max_records)
    n_active = sum(1 for r in records if r.is_active)
    print(f"  {len(records)} real curated compounds ({n_active} active / {len(records) - n_active} inactive)")
    receptor_validation = validate_receptor_structure()
    print(f"  Receptor validation: {receptor_validation}")

    print("Stage 2: training real MPNN QSAR model + ADMET filter...")
    qsar_checkpoint = RESULTS_DIR / "qsar_model_checkpoint.pt"
    qsar_metrics_path = RESULTS_DIR / "qsar_metrics.json"
    if not args.refresh_cache and qsar_checkpoint.exists() and qsar_metrics_path.exists():
        # Real resumability, same reason as `run_vina_docking`'s own
        # (see above): this stage alone takes real minutes of CPU time,
        # and re-running it after every interrupted session restart
        # would repeat already-completed real work for no reason.
        qsar_metrics = json.loads(qsar_metrics_path.read_text())
        qsar_model = MPNNRegressor()
        qsar_model.load_state_dict(torch.load(qsar_checkpoint, weights_only=True))
        qsar_model.y_mean, qsar_model.y_std = qsar_metrics["y_mean"], qsar_metrics["y_std"]
        print(f"  Resumed from checkpoint: {qsar_metrics}")
    else:
        qsar_model, qsar_metrics = train_qsar_model(records)
        print(f"  {qsar_metrics}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(qsar_model.state_dict(), qsar_checkpoint)
        qsar_metrics_path.write_text(json.dumps(qsar_metrics, indent=2))

    print("Stage 3: real generative Transformer + REINFORCE fine-tuning...")
    active_records = [r for r in records if r.is_active]
    selfies_strings = [s for r in active_records if (s := smiles_to_selfies(r.canonical_smiles)) is not None]
    vocab = Vocabulary.build(selfies_strings)
    encoded = [e for s in selfies_strings if (e := vocab.encode(s)) is not None]
    print(f"  Pretraining corpus: {len(encoded)} real active EGFR compounds, vocab size {len(vocab)}")

    gen_model = GenerativeTransformer(vocab_size=len(vocab))
    pretrain(gen_model, vocab, encoded, epochs=args.pretrain_epochs)
    training_smiles = {Chem.MolToSmiles(Chem.MolFromSmiles(r.canonical_smiles)) for r in active_records}

    pre_ids = gen_model.sample(vocab, n=args.n_sample, seed=SEED)
    pre_reports = score_sequences(pre_ids, vocab, qsar_model)

    history = reinforce_finetune(gen_model, vocab, qsar_model, iterations=args.rl_iterations)
    post_ids = gen_model.sample(vocab, n=args.n_sample, seed=SEED)
    post_reports = score_sequences(post_ids, vocab, qsar_model)

    pre_valid = [r for r in pre_reports if r.valid]
    post_valid = [r for r in post_reports if r.valid]
    generative_summary = {
        "n_pretrain_compounds": len(encoded), "vocab_size": len(vocab),
        "final_mean_reward": history[-1]["mean_reward"],
        "pre_rl_mean_pic50": round(float(np.mean([r.pred_pic50 for r in pre_valid])), 3) if pre_valid else None,
        "post_rl_mean_pic50": round(float(np.mean([r.pred_pic50 for r in post_valid])), 3) if post_valid else None,
        "post_rl_valid_frac": round(len(post_valid) / len(post_reports), 4),
        "post_rl_unique_frac": round(len({r.smiles for r in post_valid}) / len(post_valid), 4) if post_valid else 0.0,
        "post_rl_novel_frac": round(sum(r.smiles not in training_smiles for r in post_valid) / len(post_valid), 4) if post_valid else 0.0,
    }
    print(f"  {generative_summary}")

    novel_admet_pass = [r for r in post_valid if r.admet_pass and r.smiles not in training_smiles]
    unique_by_smiles = {r.smiles: r for r in novel_admet_pass}.values()
    top_candidates = sorted(unique_by_smiles, key=lambda r: -(r.pred_pic50 or 0))[: args.n_dock]
    candidates = [{"molecule_id": f"GEN_{i:03d}", "smiles": r.smiles, "pred_pic50": r.pred_pic50} for i, r in enumerate(top_candidates)]
    print(f"  {len(candidates)} novel, ADMET-passing candidates selected for docking")

    redock_result, box = None, None
    if not args.skip_docking and candidates:
        print("Stage 4a: real AutoDock Vina docking...")
        receptor_raw, native_ligand = split_receptor_and_native_ligand(RECEPTOR_PDB, workdir)
        receptor_pdbqt = prepare_receptor_pdbqt(receptor_raw, workdir / "receptor.pdbqt")
        box = compute_focused_box(native_ligand)
        redock_result = redocking_validation(receptor_pdbqt, native_ligand, box, workdir)
        print(f"  Redocking validation: {redock_result}")
        candidates = dock_candidates(candidates, receptor_pdbqt, box, workdir)
        candidates.sort(key=lambda c: c["docking"].get("affinity_kcal_mol") or 0.0)

    if not args.skip_md and candidates:
        print("Stage 4b: real ANI-2x ligand-alone MD stability checks...")
        # Real, disclosed, upfront selection criterion (Chapter 13's own
        # "checked per-compound rather than assumed" discipline): the
        # top-N *real docking-ranked* candidates are only attempted if
        # ANI-2x actually supports every element present, rather than
        # blindly taking the top N by affinity and discovering a
        # coverage failure per compound after the fact -- a real,
        # disclosed halogen skew in this pipeline's own top real
        # docking hits (bromine, outside ANI-2x's real trained
        # coverage) would otherwise have left zero real MD results.
        docked = [c for c in candidates if c.get("docking", {}).get("affinity_kcal_mol") is not None]
        for c in docked:
            c["ani2x_compatible_elements"] = ani2x_compatible_elements(c["smiles"])
        md_targets = {c["molecule_id"] for c in docked if c["ani2x_compatible_elements"]}
        md_targets = set(sorted(md_targets, key=lambda mid: next(c for c in docked if c["molecule_id"] == mid)["docking"]["affinity_kcal_mol"])[: args.n_md])
        for c in docked:
            if c["molecule_id"] not in md_targets:
                c["md"] = {"skipped": True, "reason": "elements outside ANI-2x's trained coverage" if not c["ani2x_compatible_elements"] else "not in top-N ANI-2x-compatible candidates by docking affinity"}
                continue
            try:
                c["md"] = run_ani2x_md(c["smiles"])
            except Exception as exc:
                c["md"] = {"error": str(exc)[-300:]}

    print("Stage 5: generating real automated technical report...")
    pipeline_results = {
        "target": {"chembl_id": TARGET_CHEMBL_ID, "pdb_id": RECEPTOR_PDB.stem, "receptor_validation": receptor_validation},
        "dataset": {"n_compounds": len(records), "n_active": n_active},
        "qsar": qsar_metrics,
        "generative": generative_summary,
        "redocking_validation": redock_result,
        "docking_box": box,
        "candidates": candidates,
    }
    report_path = generate_report(pipeline_results, RESULTS_DIR / "technical_report.html")
    print(f"  Report written to {report_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pipeline_results, indent=2, default=str), encoding="utf-8")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
