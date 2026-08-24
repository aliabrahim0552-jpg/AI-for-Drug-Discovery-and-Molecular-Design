"""
Chapter 4 hands-on project: an automated ETL pipeline that extracts,
cleans, and formats a bioactivity dataset from ChEMBL.

Extract: paginated retrieval of raw bioactivity records for a target
  from the ChEMBL REST API (building on Chapter 1's single-page fetch).
Transform: drop incomplete/censored records, standardize structures
  with RDKit (salt stripping, uncharging, canonical tautomer - the
  techniques from Section 4.1), normalize units, deduplicate repeated
  measurements of the same compound, and compute Lipinski descriptors
  (Chapter 2's Rule of Five, Lipinski et al., 2001).
Load: write a tidy, one-row-per-compound CSV ready for the QSAR/ADMET
  modeling in Chapter 5.

See README.md for usage and chapter.md Section 4.5 for context.
"""
import argparse
import csv
import json
import statistics
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_TARGET_CHEMBL_ID = "CHEMBL203"  # EGFR, the running example since Chapter 1
DEFAULT_MAX_RECORDS = 500
DEFAULT_PAGE_SIZE = 100
DEFAULT_TIMEOUT = 30
ACCEPTED_STANDARD_TYPES = {"IC50", "Ki", "EC50", "Kd"}
# ChEMBL's standard_value is only a point estimate when the relation is
# exact ("="); ">"/"<"/">="/"<=" mean the true value is censored (e.g. "IC50
# > 1250000 nM" just means "no activity observed up to this concentration").
# Records with no relation recorded are, in ChEMBL's convention, exact.
ACCEPTED_RELATIONS = {"=", None}
UNIT_TO_NM = {"nM": 1.0, "uM": 1_000.0, "mM": 1_000_000.0, "pM": 0.001}

# Lipinski's Rule of Five thresholds (Lipinski et al., 2001) - identical
# to Chapter 2's similarity_search.py, reimplemented here to keep this
# chapter's project self-contained per this repo's convention.
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
    standard_type: str
    standard_value_nm: float
    n_measurements: int
    molecular_weight: float
    logp: float
    h_bond_donors: int
    h_bond_acceptors: int
    passes_lipinski: bool


# --------------------------------------------------------------------------
# Extract
# --------------------------------------------------------------------------


def extract_bioactivities(
    target_chembl_id: str = DEFAULT_TARGET_CHEMBL_ID,
    max_records: int = DEFAULT_MAX_RECORDS,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict]:
    """Paginated extraction of raw bioactivity records from the ChEMBL API."""
    records: list[dict] = []
    url = f"{CHEMBL_API_BASE}/activity.json"
    params = {"target_chembl_id": target_chembl_id, "limit": page_size, "format": "json"}

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
    """
    Standardize a SMILES string: parse, keep the largest fragment (strip
    salts/counterions), neutralize charges, and canonicalize to a single
    tautomer (Section 4.1). Returns None if the input cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _standardizer().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = _tautomer_enumerator().Canonicalize(mol)
    return Chem.MolToSmiles(mol)


def compute_lipinski_descriptors(smiles: str) -> dict | None:
    """Molecular weight, LogP, H-bond donors/acceptors, and Ro5 pass/fail."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "molecular_weight": mw,
        "logp": logp,
        "h_bond_donors": hbd,
        "h_bond_acceptors": hba,
        "passes_lipinski": violations <= 1,
    }


def _to_nm(value: float, units: str | None) -> float | None:
    factor = UNIT_TO_NM.get(units)
    return value * factor if factor is not None else None


def clean_bioactivity_records(raw_records: list[dict]) -> list[CleanCompoundRecord]:
    """
    Filter, standardize, unit-normalize, and deduplicate raw ChEMBL
    bioactivity records into one row per (standardized compound, assay
    type), with Lipinski descriptors computed on the standardized
    structure.
    """
    # Stage 1: filter to complete, exact-valued, in-scope measurements.
    filtered = []
    for r in raw_records:
        if r.get("standard_type") not in ACCEPTED_STANDARD_TYPES:
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
                "standard_type": r["standard_type"],
                "value_nm": value_nm,
            }
        )

    # Stage 2: standardize structures (cached per raw SMILES - re-running
    # tautomer canonicalization on every duplicate row would be wasted work).
    standardized_cache: dict[str, str | None] = {}
    for row in filtered:
        raw_smiles = row["raw_smiles"]
        if raw_smiles not in standardized_cache:
            standardized_cache[raw_smiles] = standardize_smiles(raw_smiles)
        row["standardized_smiles"] = standardized_cache[raw_smiles]
    filtered = [r for r in filtered if r["standardized_smiles"] is not None]

    # Stage 3: deduplicate by (standardized compound, assay type), aggregating
    # repeated measurements with the median (robust to single outlier assays).
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in filtered:
        key = (row["standardized_smiles"], row["standard_type"])
        groups.setdefault(key, []).append(row)

    # Stage 4: compute Lipinski descriptors once per unique standardized structure.
    descriptor_cache: dict[str, dict] = {}
    results = []
    for (smiles, standard_type), rows in groups.items():
        if smiles not in descriptor_cache:
            desc = compute_lipinski_descriptors(smiles)
            if desc is None:
                continue
            descriptor_cache[smiles] = desc
        desc = descriptor_cache[smiles]
        values = [r["value_nm"] for r in rows]
        results.append(
            CleanCompoundRecord(
                molecule_chembl_id=rows[0]["molecule_chembl_id"],
                canonical_smiles=smiles,
                standard_type=standard_type,
                standard_value_nm=round(statistics.median(values), 3),
                n_measurements=len(values),
                molecular_weight=round(desc["molecular_weight"], 2),
                logp=round(desc["logp"], 2),
                h_bond_donors=desc["h_bond_donors"],
                h_bond_acceptors=desc["h_bond_acceptors"],
                passes_lipinski=desc["passes_lipinski"],
            )
        )

    results.sort(key=lambda r: r.standard_value_nm)
    return results


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------


def save_clean_csv(records: list[CleanCompoundRecord], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CleanCompoundRecord.__dataclass_fields__.keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)
    return out_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, clean, and format a ChEMBL bioactivity dataset."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET_CHEMBL_ID, help="ChEMBL target ID.")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).parent / "data"), help="Directory to write output files to."
    )
    parser.add_argument(
        "--use-cached-raw",
        action="store_true",
        help="Load the bundled raw fixture instead of hitting the live ChEMBL API (useful when ChEMBL is unavailable).",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.use_cached_raw:
        fixture_path = out_dir / "raw_egfr_bioactivities_sample.json"
        print(f"Loading cached raw extract -> {fixture_path}")
        raw_records = load_raw_json(fixture_path)
    else:
        try:
            print(f"Extracting bioactivities for {args.target} from ChEMBL...")
            raw_records = extract_bioactivities(args.target, max_records=args.max_records)
            save_raw_json(raw_records, out_dir / f"raw_{args.target}_bioactivities.json")
        except requests.exceptions.RequestException as exc:
            print(f"Live ChEMBL extraction failed ({exc}); falling back to bundled fixture.")
            raw_records = load_raw_json(out_dir / "raw_egfr_bioactivities_sample.json")

    print(f"Extracted {len(raw_records)} raw records.")

    clean_records = clean_bioactivity_records(raw_records)
    csv_path = save_clean_csv(clean_records, out_dir / f"{args.target}_clean.csv")
    n_pass = sum(1 for r in clean_records if r.passes_lipinski)
    print(
        f"Cleaned to {len(clean_records)} unique (compound, assay type) rows -> {csv_path}\n"
        f"  {n_pass}/{len(clean_records)} pass Lipinski's Rule of Five"
    )
    for r in clean_records[:5]:
        print(f"  {r.molecule_chembl_id} {r.standard_type}={r.standard_value_nm}nM (n={r.n_measurements})")


if __name__ == "__main__":
    main()
