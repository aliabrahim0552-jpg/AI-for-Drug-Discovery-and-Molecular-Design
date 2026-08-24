"""
Chapter 5 hands-on project: a QSAR classifier for hERG channel toxicity.

Extract: paginated retrieval of raw hERG (CHEMBL240) IC50 bioactivity
  records from the ChEMBL REST API (same pattern as Chapter 4's
  extract_bioactivities, reimplemented here to keep this chapter's
  project self-contained per this repo's convention).
Transform: standardize structures with RDKit (Section 4.1's technique),
  deduplicate repeated measurements via the median, and label each
  compound blocker/non-blocker at a fixed IC50 threshold (Section 5.2).
Featurize: Morgan/ECFP4 fingerprints (Chapter 2's representation).
Split: both a random split and a Bemis-Murcko scaffold split (Section
  5.4), so the same clean/featurize/train pipeline can be run under
  either splitting strategy for a controlled comparison.
Train/evaluate: Random Forest, SVM, or XGBoost (Section 5.3) classifier,
  optionally with SMOTE oversampling of the training set (Section 5.4).

See README.md for usage and chapter.md Section 5.5 for context.
"""
import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_TARGET_CHEMBL_ID = "CHEMBL240"  # hERG (KCNH2), Sanguinetti & Tristani-Firouzi, 2006
DEFAULT_MAX_RECORDS = 3000
DEFAULT_PAGE_SIZE = 100
DEFAULT_TIMEOUT = 30
DEFAULT_THRESHOLD_NM = 10_000.0  # 10 uM blocker/non-blocker cutoff - see chapter.md Section 5.2
ACCEPTED_RELATIONS = {"=", None}
UNIT_TO_NM = {"nM": 1.0, "uM": 1_000.0, "mM": 1_000_000.0, "pM": 0.001}

FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

_STANDARDIZER = None
_TAUTOMER_ENUMERATOR = None


def _standardizer() -> rdMolStandardize.LargestFragmentChooser:
    global _STANDARDIZER
    if _STANDARDIZER is None:
        _STANDARDIZER = rdMolStandardize.LargestFragmentChooser()
    return _STANDARDIZER


def _tautomer_enumerator() -> rdMolStandardize.TautomerEnumerator:
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
# Extract
# --------------------------------------------------------------------------


def extract_bioactivities(
    target_chembl_id: str = DEFAULT_TARGET_CHEMBL_ID,
    max_records: int = DEFAULT_MAX_RECORDS,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict]:
    """Paginated extraction of raw IC50 bioactivity records from the ChEMBL API."""
    records: list[dict] = []
    url = f"{CHEMBL_API_BASE}/activity.json"
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": "IC50",
        "limit": page_size,
        "format": "json",
    }

    while url and len(records) < max_records:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        records.extend(payload.get("activities", []))

        next_path = payload.get("page_meta", {}).get("next")
        url = f"https://www.ebi.ac.uk{next_path}" if next_path else None
        params = None  # the "next" URL already includes query params

    return records[:max_records]


def save_raw_json(records: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"activities": records}, indent=None), encoding="utf-8")
    return out_path


def load_raw_json(path: Path) -> list[dict]:
    """Load raw records from a cached extract (see save_raw_json / bundled fixture)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["activities"]


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


def standardize_smiles(smiles: str) -> str | None:
    """Parse, strip salts/counterions, neutralize charges, canonicalize
    to a single tautomer (Chapter 4, Section 4.1). None if unparsable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _standardizer().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = _tautomer_enumerator().Canonicalize(mol)
    return Chem.MolToSmiles(mol)


def _to_nm(value: float, units: str | None) -> float | None:
    factor = UNIT_TO_NM.get(units)
    return value * factor if factor is not None else None


def clean_bioactivity_records(
    raw_records: list[dict], threshold_nm: float = DEFAULT_THRESHOLD_NM
) -> list[CleanCompoundRecord]:
    """
    Filter to complete, exact-valued IC50 measurements, standardize
    structures, deduplicate repeated measurements of the same compound
    via the median, and label blocker/non-blocker at threshold_nm.
    """
    filtered = []
    for r in raw_records:
        if r.get("standard_type") != "IC50":
            continue
        if r.get("standard_relation") not in ACCEPTED_RELATIONS:
            continue
        if not r.get("canonical_smiles") or r.get("standard_value") is None:
            continue
        value_nm = _to_nm(float(r["standard_value"]), r.get("standard_units"))
        if value_nm is None or value_nm <= 0:
            continue
        filtered.append(
            {
                "molecule_chembl_id": r["molecule_chembl_id"],
                "raw_smiles": r["canonical_smiles"],
                "value_nm": value_nm,
            }
        )

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

    import statistics

    results = []
    for smiles, rows in groups.items():
        values = [r["value_nm"] for r in rows]
        median_nm = statistics.median(values)
        results.append(
            CleanCompoundRecord(
                molecule_chembl_id=rows[0]["molecule_chembl_id"],
                canonical_smiles=smiles,
                ic50_nm=round(median_nm, 3),
                n_measurements=len(values),
                is_blocker=median_nm <= threshold_nm,
            )
        )

    results.sort(key=lambda r: r.ic50_nm)
    return results


# --------------------------------------------------------------------------
# Featurize
# --------------------------------------------------------------------------


def featurize(smiles_list: list[str]) -> np.ndarray:
    """Morgan/ECFP4 fingerprints (radius=2, 2048 bits) - Chapter 2, Section 2.2."""
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
    """The Bemis-Murcko generic scaffold SMILES for a molecule (Bemis &
    Murcko, 1996). Compounds sharing a scaffold are structurally related
    at the ring-system level, which is what scaffold splitting groups on."""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def random_split(
    records: list[CleanCompoundRecord], frac_train: float = 0.8, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Stratified random split by row index (Section 5.4's baseline)."""
    labels = [r.is_blocker for r in records]
    idx = list(range(len(records)))
    train_idx, test_idx = train_test_split(
        idx, train_size=frac_train, random_state=seed, stratify=labels
    )
    return sorted(train_idx), sorted(test_idx)


def scaffold_split(
    records: list[CleanCompoundRecord], frac_train: float = 0.8, seed: int = 0
) -> tuple[list[int], list[int]]:
    """
    Bemis-Murcko scaffold split (Bemis & Murcko, 1996; grouping/greedy-fill
    methodology per Yang et al., 2019 and DeepChem's documented
    ScaffoldSplitter): group compounds by generic scaffold, order the
    groups deterministically, and greedily fill the training set
    scaffold-group by scaffold-group (largest first) until frac_train of
    the data is assigned - so no scaffold appears in both train and test.
    """
    scaffold_to_indices: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        scaffold = bemis_murcko_scaffold(r.canonical_smiles)
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    rng = np.random.RandomState(seed)
    groups = list(scaffold_to_indices.values())
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]
    groups.sort(key=len, reverse=True)

    n_train_target = int(round(frac_train * len(records)))
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

MODEL_BUILDERS = {
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=300, random_state=seed, n_jobs=-1
    ),
    "svm": lambda seed: SVC(kernel="rbf", probability=True, random_state=seed),
    "xgboost": lambda seed: XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    ),
}


def train_and_evaluate(
    records: list[CleanCompoundRecord],
    model_type: str = "xgboost",
    split_type: str = "scaffold",
    use_smote: bool = False,
    seed: int = 0,
) -> dict:
    """Featurize, split, optionally oversample the training set, train
    one classifier, and return held-out test-set metrics plus split
    diagnostics (Sections 5.3-5.5)."""
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type {model_type!r}; choose from {list(MODEL_BUILDERS)}")

    X = featurize([r.canonical_smiles for r in records])
    y = np.array([int(r.is_blocker) for r in records])

    if split_type == "random":
        train_idx, test_idx = random_split(records, seed=seed)
    elif split_type == "scaffold":
        train_idx, test_idx = scaffold_split(records, seed=seed)
    else:
        raise ValueError(f"Unknown split_type {split_type!r}; choose 'random' or 'scaffold'")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if use_smote:
        from imblearn.over_sampling import SMOTE

        X_train, y_train = SMOTE(random_state=seed).fit_resample(X_train, y_train)

    model = MODEL_BUILDERS[model_type](seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model_type": model_type,
        "split_type": split_type,
        "use_smote": use_smote,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_blocker_frac": round(float(np.mean(y_train)), 4),
        "test_blocker_frac": round(float(np.mean(y_test)), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an hERG-blocker QSAR classifier and compare random vs. scaffold split."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET_CHEMBL_ID, help="ChEMBL target ID.")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--threshold-nm", type=float, default=DEFAULT_THRESHOLD_NM)
    parser.add_argument(
        "--model", choices=list(MODEL_BUILDERS), default="xgboost", help="Classifier to train."
    )
    parser.add_argument(
        "--split",
        choices=["random", "scaffold", "both"],
        default="both",
        help="Evaluation split(s) to run.",
    )
    parser.add_argument("--use-smote", action="store_true", help="Oversample the training set with SMOTE.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).parent / "data"), help="Directory to write/read cached data."
    )
    parser.add_argument(
        "--use-cached-raw",
        action="store_true",
        help="Load the bundled raw fixture instead of hitting the live ChEMBL API.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.use_cached_raw:
        fixture_path = out_dir / "raw_herg_bioactivities_sample.json"
        print(f"Loading cached raw extract -> {fixture_path}")
        raw_records = load_raw_json(fixture_path)[: args.max_records]
    else:
        try:
            print(f"Extracting IC50 bioactivities for {args.target} from ChEMBL...")
            raw_records = extract_bioactivities(args.target, max_records=args.max_records)
            save_raw_json(raw_records, out_dir / f"raw_{args.target}_bioactivities.json")
        except requests.exceptions.RequestException as exc:
            print(f"Live ChEMBL extraction failed ({exc}); falling back to bundled fixture.")
            raw_records = load_raw_json(out_dir / "raw_herg_bioactivities_sample.json")[: args.max_records]

    print(f"Extracted {len(raw_records)} raw records.")

    records = clean_bioactivity_records(raw_records, threshold_nm=args.threshold_nm)
    n_blockers = sum(1 for r in records if r.is_blocker)
    print(
        f"Cleaned to {len(records)} unique standardized compounds "
        f"({n_blockers} blockers / {len(records) - n_blockers} non-blockers "
        f"at {args.threshold_nm:.0f} nM)\n"
    )

    splits = ["random", "scaffold"] if args.split == "both" else [args.split]
    for split_type in splits:
        result = train_and_evaluate(
            records,
            model_type=args.model,
            split_type=split_type,
            use_smote=args.use_smote,
            seed=args.seed,
        )
        print(f"[{split_type} split, {args.model}, smote={args.use_smote}]")
        for k, v in result.items():
            if k not in ("model_type", "split_type", "use_smote"):
                print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
