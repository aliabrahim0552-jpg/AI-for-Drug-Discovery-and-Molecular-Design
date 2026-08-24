"""
Chapter 1 hands-on project: programmatic retrieval of chemical and
structural data from the ChEMBL and RCSB PDB public REST APIs.

Fetches, for a given target (default: EGFR / CHEMBL203):
  - target metadata and a page of bioactivity measurements from ChEMBL
  - entry metadata and a 3D structure file from the PDB (default: 1M17,
    the EGFR tyrosine kinase domain bound to erlotinib)

See README.md for usage. See chapter.md Section 1.4 for context.
"""
import argparse
import csv
import sys
from pathlib import Path

import requests

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
RCSB_DATA_API_BASE = "https://data.rcsb.org/rest/v1/core"
RCSB_FILES_BASE = "https://files.rcsb.org/download"

DEFAULT_TARGET_CHEMBL_ID = "CHEMBL203"  # EGFR (Homo sapiens)
DEFAULT_PDB_ID = "1M17"  # EGFR tyrosine kinase domain + erlotinib
DEFAULT_TIMEOUT = 30


def fetch_chembl_status(timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Fetch ChEMBL's current release version and database-wide counts."""
    resp = requests.get(f"{CHEMBL_API_BASE}/status.json", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_chembl_target(target_chembl_id: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Fetch target metadata (preferred name, organism, type) from ChEMBL."""
    resp = requests.get(f"{CHEMBL_API_BASE}/target/{target_chembl_id}.json", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_chembl_bioactivities(
    target_chembl_id: str, limit: int = 25, timeout: int = DEFAULT_TIMEOUT
) -> list[dict]:
    """
    Fetch a page of measured bioactivities for a target from ChEMBL.

    Returns a list of dicts with molecule_chembl_id, canonical_smiles,
    standard_type, standard_value, standard_units, and pchembl_value.
    """
    params = {
        "target_chembl_id": target_chembl_id,
        "limit": limit,
        "format": "json",
    }
    resp = requests.get(f"{CHEMBL_API_BASE}/activity.json", params=params, timeout=timeout)
    resp.raise_for_status()
    activities = resp.json().get("activities", [])

    rows = []
    for a in activities:
        rows.append(
            {
                "molecule_chembl_id": a.get("molecule_chembl_id"),
                "canonical_smiles": a.get("canonical_smiles"),
                "standard_type": a.get("standard_type"),
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
                "pchembl_value": a.get("pchembl_value"),
            }
        )
    return rows


def fetch_pdb_entry_metadata(pdb_id: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Fetch entry-level metadata (title, method, release date) from RCSB PDB."""
    resp = requests.get(f"{RCSB_DATA_API_BASE}/entry/{pdb_id}", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    experiment = data.get("rcsb_entry_info", {}).get("experimental_method")
    resolution_list = data.get("rcsb_entry_info", {}).get("resolution_combined")

    return {
        "pdb_id": pdb_id,
        "title": data.get("struct", {}).get("title"),
        "experimental_method": experiment,
        "resolution_angstrom": resolution_list[0] if resolution_list else None,
        "release_date": data.get("rcsb_accession_info", {}).get("initial_release_date"),
    }


def download_pdb_structure(
    pdb_id: str, out_path: Path, file_format: str = "pdb", timeout: int = DEFAULT_TIMEOUT
) -> Path:
    """Download a structure file (default legacy .pdb format) from RCSB PDB."""
    ext = "pdb" if file_format == "pdb" else file_format
    url = f"{RCSB_FILES_BASE}/{pdb_id}.{ext}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return out_path


def save_bioactivities_csv(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_value",
        "standard_units",
        "pchembl_value",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve chemical (ChEMBL) and structural (PDB) data for a target."
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET_CHEMBL_ID, help="ChEMBL target ID (default: EGFR / CHEMBL203)."
    )
    parser.add_argument("--pdb", default=DEFAULT_PDB_ID, help="PDB entry ID (default: 1M17).")
    parser.add_argument("--limit", type=int, default=25, help="Number of bioactivity records to fetch.")
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).parent / "data"), help="Directory to write fetched files to."
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    print(f"ChEMBL status: {fetch_chembl_status()}\n")

    target = fetch_chembl_target(args.target)
    print(f"Target {args.target}: {target['pref_name']} ({target['organism']}, {target['target_type']})")

    bioactivities = fetch_chembl_bioactivities(args.target, limit=args.limit)
    csv_path = save_bioactivities_csv(bioactivities, out_dir / f"{args.target}_bioactivities.csv")
    print(f"Fetched {len(bioactivities)} bioactivity records -> {csv_path}")

    entry = fetch_pdb_entry_metadata(args.pdb)
    print(
        f"\nPDB {args.pdb}: {entry['title']}\n"
        f"  method={entry['experimental_method']} "
        f"resolution={entry['resolution_angstrom']}A "
        f"released={entry['release_date']}"
    )

    structure_path = download_pdb_structure(args.pdb, out_dir / f"{args.pdb}.pdb")
    print(f"Downloaded structure -> {structure_path} ({structure_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
