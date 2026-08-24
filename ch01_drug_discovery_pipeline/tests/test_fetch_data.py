"""
Tests for the Chapter 1 hands-on project (ChEMBL + PDB data retrieval).

These call the live ChEMBL and RCSB PDB REST APIs directly (no mocking):
the whole point of this chapter's project is a *reproducible* retrieval
pipeline, so a passing test suite is itself evidence the pipeline still
works against the real services. Requires network access.

Run with: pytest
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from fetch_data import (
    DEFAULT_PDB_ID,
    DEFAULT_TARGET_CHEMBL_ID,
    download_pdb_structure,
    fetch_chembl_bioactivities,
    fetch_chembl_status,
    fetch_chembl_target,
    fetch_pdb_entry_metadata,
    save_bioactivities_csv,
)


def test_fetch_chembl_status_reports_a_recent_release():
    status = fetch_chembl_status()
    assert status["status"] == "UP"
    assert status["chembl_db_version"].startswith("ChEMBL_")
    assert status["disinct_compounds"] > 1_000_000


def test_fetch_chembl_target_identifies_egfr():
    target = fetch_chembl_target(DEFAULT_TARGET_CHEMBL_ID)
    assert target["pref_name"] == "Epidermal growth factor receptor"
    assert target["organism"] == "Homo sapiens"
    assert target["target_type"] == "SINGLE PROTEIN"


def test_fetch_chembl_bioactivities_returns_valid_rows():
    rows = fetch_chembl_bioactivities(DEFAULT_TARGET_CHEMBL_ID, limit=10)
    assert len(rows) == 10
    for row in rows:
        assert row["molecule_chembl_id"]
        assert row["canonical_smiles"]
        assert row["standard_type"]


def test_save_bioactivities_csv_writes_expected_header(tmp_path):
    rows = fetch_chembl_bioactivities(DEFAULT_TARGET_CHEMBL_ID, limit=3)
    out = save_bioactivities_csv(rows, tmp_path / "out.csv")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == (
        "molecule_chembl_id,canonical_smiles,standard_type,"
        "standard_value,standard_units,pchembl_value"
    )
    assert len(lines) == 4  # header + 3 rows


def test_fetch_pdb_entry_metadata_identifies_1m17():
    entry = fetch_pdb_entry_metadata(DEFAULT_PDB_ID)
    assert "EPIDERMAL GROWTH FACTOR RECEPTOR" in entry["title"].upper()
    assert entry["experimental_method"] == "X-ray"
    assert entry["resolution_angstrom"] is not None
    assert entry["release_date"].startswith("2002")


def test_download_pdb_structure_writes_valid_pdb_file(tmp_path):
    out_path = download_pdb_structure(DEFAULT_PDB_ID, tmp_path / "1m17.pdb")
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8", errors="ignore")
    assert content.startswith("HEADER")
    assert "EPIDERMAL GROWTH FACTOR RECEPTOR" in content
    assert out_path.stat().st_size > 10_000


def test_fetch_chembl_target_rejects_unknown_id():
    with pytest.raises(Exception):
        fetch_chembl_target("CHEMBL_NOT_A_REAL_ID_999999")


def test_fetch_pdb_entry_metadata_rejects_unknown_id():
    with pytest.raises(Exception):
        fetch_pdb_entry_metadata("9ZZZ")
