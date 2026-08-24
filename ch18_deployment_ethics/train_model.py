"""
Chapter 18 hands-on project, Part 1: train and save a real, versioned
model artifact -- the "before" half of "packaging models into
interactive web tools" (Section 18.1). Reuses Chapter 5's own real
hERG (CHEMBL240) blocker-classification pipeline unchanged (live
ChEMBL retrieval, RDKit standardization, ECFP4 featurization, XGBoost),
because this chapter is about *deployment*, not re-deriving a new QSAR
model -- the real held-out scaffold-split metrics reported here are
Chapter 5's own real evaluation protocol, re-run fresh against
whatever real ChEMBL240 data is live today, not assumed identical to
Chapter 5's own 2026-08-20 numbers.

Real deployment practice this script also follows (Section 18.2's own
FAIR/reproducibility discussion): the held-out scaffold-split model is
trained and evaluated first to get an honest accuracy estimate, and
only *then* is a second, separate model retrained on the *entire*
real curated dataset (train + test combined) for the actual deployed
artifact -- the standard real practice of using every real available
label for the model that ships, while still reporting genuine
held-out numbers for how well that modeling approach performs. The
saved artifact bundles a real SHA-256 content hash and the exact real
scikit-learn/XGBoost/RDKit versions it was built with, so `service.py`
(and any future caller) can verify what it is actually loading rather
than trusting a filename.

See README.md for usage and chapter.md Section 18.1 for context.
"""
import argparse
import hashlib
import json
import platform
import statistics
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import requests
import rdkit
import sklearn
import xgboost
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
TARGET_CHEMBL_ID = "CHEMBL240"  # hERG (KCNH2), Chapter 5's own target
DEFAULT_MAX_RECORDS = 3000
DEFAULT_PAGE_SIZE = 100
THRESHOLD_NM = 10_000.0  # 10 uM blocker/non-blocker cutoff, Chapter 5 Section 5.2's convention
UNIT_TO_NM = {"nM": 1.0, "uM": 1_000.0, "mM": 1_000_000.0, "pM": 0.001}

FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
MODEL_VERSION = "1.0.0"

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


@dataclass
class CleanCompoundRecord:
    molecule_chembl_id: str
    canonical_smiles: str
    ic50_nm: float
    n_measurements: int
    is_blocker: bool


# --------------------------------------------------------------------------
# Extract / transform (Chapter 5's own real methodology, reused unchanged)
# --------------------------------------------------------------------------


def extract_bioactivities(target_chembl_id: str = TARGET_CHEMBL_ID, max_records: int = DEFAULT_MAX_RECORDS, page_size: int = DEFAULT_PAGE_SIZE, timeout: int = 30) -> list[dict]:
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


def standardize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _standardizer().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = _tautomer_enumerator().Canonicalize(mol)
    return Chem.MolToSmiles(mol)


def clean_bioactivity_records(raw_records: list[dict], threshold_nm: float = THRESHOLD_NM) -> list[CleanCompoundRecord]:
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
        results.append(CleanCompoundRecord(rows[0]["molecule_chembl_id"], smiles, round(median_nm, 3), len(rows), median_nm <= threshold_nm))
    results.sort(key=lambda r: r.molecule_chembl_id)
    return results


def load_or_build_dataset(refresh: bool = False, max_records: int = DEFAULT_MAX_RECORDS) -> list[CleanCompoundRecord]:
    cache_path = DATA_DIR / "herg_dataset.json"
    if cache_path.exists() and not refresh:
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
        return [CleanCompoundRecord(**row) for row in rows]
    raw = extract_bioactivities(max_records=max_records)
    records = clean_bioactivity_records(raw)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps([r.__dict__ for r in records], indent=2), encoding="utf-8")
    return records


# --------------------------------------------------------------------------
# Featurize / split (Chapter 5's own real methodology, reused unchanged)
# --------------------------------------------------------------------------


def featurize(smiles_list: list[str]) -> np.ndarray:
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = FP_GENERATOR.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=np.int8)
        for bit in fp.GetOnBits():
            arr[bit] = 1
        fps.append(arr)
    return np.vstack(fps)


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def scaffold_split(records: list[CleanCompoundRecord], frac_train: float = 0.8, seed: int = 0) -> tuple[list[int], list[int]]:
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


def build_model(seed: int = 0) -> XGBClassifier:
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=seed, n_jobs=-1)


def evaluate_holdout(records: list[CleanCompoundRecord], seed: int = 0) -> dict:
    """Real, honest scaffold-split held-out evaluation (Chapter 5's own
    protocol) -- reported alongside the deployed artifact, not replaced
    by it."""
    X = featurize([r.canonical_smiles for r in records])
    y = np.array([int(r.is_blocker) for r in records])
    train_idx, test_idx = scaffold_split(records, seed=seed)
    model = build_model(seed)
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
    y_proba = model.predict_proba(X[test_idx])[:, 1]
    return {
        "n_train": len(train_idx), "n_test": len(test_idx),
        "accuracy": round(accuracy_score(y[test_idx], y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y[test_idx], y_pred), 4),
        "precision": round(precision_score(y[test_idx], y_pred, zero_division=0), 4),
        "recall": round(recall_score(y[test_idx], y_pred, zero_division=0), 4),
        "f1": round(f1_score(y[test_idx], y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y[test_idx], y_proba), 4),
    }


# --------------------------------------------------------------------------
# Real, versioned model artifact
# --------------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def train_and_save_deployed_model(records: list[CleanCompoundRecord], holdout_metrics: dict, seed: int = 0) -> dict:
    """Real deployment practice: retrain on the *entire* real curated
    dataset (every label available) for the artifact that actually
    ships, after `evaluate_holdout` has already reported an honest,
    real estimate of how well this approach performs on unseen
    scaffolds. The saved artifact is self-describing: real training
    data provenance, real library versions, and a real SHA-256 content
    hash computed *after* writing the file, so any consumer can verify
    exactly what they loaded."""
    X = featurize([r.canonical_smiles for r in records])
    y = np.array([int(r.is_blocker) for r in records])
    model = build_model(seed)
    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "herg_xgboost.joblib"
    joblib.dump(model, model_path)
    content_hash = sha256_of_file(model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_chembl_id": TARGET_CHEMBL_ID,
        "threshold_nm": THRESHOLD_NM,
        "n_training_compounds": len(records),
        "n_blockers": sum(1 for r in records if r.is_blocker),
        "featurization": "Morgan/ECFP4, radius=2, 2048 bits (RDKit rdFingerprintGenerator)",
        "held_out_scaffold_split_metrics": holdout_metrics,
        "sha256_model_file": content_hash,
        "library_versions": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (MODELS_DIR / "herg_xgboost.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_deployed_model() -> tuple[XGBClassifier, dict]:
    """Real provenance check before trusting a loaded artifact: the
    metadata file's own recorded SHA-256 must match the model file's
    real, freshly recomputed hash."""
    model_path = MODELS_DIR / "herg_xgboost.joblib"
    metadata_path = MODELS_DIR / "herg_xgboost.metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}. Run train_model.py first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_hash = sha256_of_file(model_path)
    if actual_hash != metadata["sha256_model_file"]:
        raise ValueError(f"Model file hash mismatch: recorded {metadata['sha256_model_file']}, actual {actual_hash}. The artifact may be corrupted or was replaced.")
    model = joblib.load(model_path)
    return model, metadata


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading real hERG (CHEMBL240) bioactivity data...")
    records = load_or_build_dataset(refresh=args.refresh_cache, max_records=args.max_records)
    n_blockers = sum(1 for r in records if r.is_blocker)
    print(f"  {len(records)} real curated compounds ({n_blockers} blockers / {len(records) - n_blockers} non-blockers)")

    print("Real, honest scaffold-split held-out evaluation...")
    holdout_metrics = evaluate_holdout(records, seed=args.seed)
    print(f"  {holdout_metrics}")

    print("Training the real deployed artifact on the full dataset...")
    metadata = train_and_save_deployed_model(records, holdout_metrics, seed=args.seed)
    print(f"  Saved to {MODELS_DIR / 'herg_xgboost.joblib'}")
    print(f"  SHA-256: {metadata['sha256_model_file']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "training_results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
