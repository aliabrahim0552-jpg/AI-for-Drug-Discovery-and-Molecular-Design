"""
Tests for the Chapter 2 hands-on project (molecular similarity search +
Lipinski filtering). Run with: pytest
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent.parent))
from similarity_search import lipinski_pass, load_library, rank_by_similarity

CH02_DIR = Path(__file__).parent.parent
LIBRARY_CSV = CH02_DIR / "molecules.csv"

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
# Long PEG chain: not a real drug, just a molecule large and ether-rich enough
# to fail Lipinski's Rule of Five on two independent axes (MW and HBA), so
# the "fail" branch of lipinski_pass is exercised by a genuine RDKit result
# rather than an invented one.
PEG_LONG_SMILES = "OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO"


def test_load_library_reads_all_entries():
    library = load_library(LIBRARY_CSV)
    names = [name for name, _ in library]
    assert len(library) == 17
    assert "Aspirin" in names
    assert "Warfarin" in names


def test_load_library_skips_blank_trailing_line():
    library = load_library(LIBRARY_CSV)
    assert all(name and smiles for name, smiles in library)


def test_rank_by_similarity_self_match_is_top_and_exact():
    library = load_library(LIBRARY_CSV)
    results = rank_by_similarity(ASPIRIN_SMILES, library)
    assert results[0].name == "Aspirin"
    assert results[0].similarity == pytest.approx(1.0)


def test_rank_by_similarity_is_sorted_descending():
    library = load_library(LIBRARY_CSV)
    results = rank_by_similarity(ASPIRIN_SMILES, library)
    similarities = [c.similarity for c in results]
    assert similarities == sorted(similarities, reverse=True)


def test_rank_by_similarity_all_similarities_in_unit_range():
    library = load_library(LIBRARY_CSV)
    results = rank_by_similarity(ASPIRIN_SMILES, library)
    assert all(0.0 <= c.similarity <= 1.0 for c in results)


def test_rank_by_similarity_rejects_invalid_query():
    library = load_library(LIBRARY_CSV)
    with pytest.raises(ValueError):
        rank_by_similarity("not a smiles string", library)


def test_rank_by_similarity_skips_unparsable_library_entries(capsys):
    library = [("Aspirin", ASPIRIN_SMILES), ("Bad", "not-a-smiles")]
    results = rank_by_similarity(ASPIRIN_SMILES, library)
    assert [c.name for c in results] == ["Aspirin"]
    assert "skipping unparsable entry" in capsys.readouterr().err


def test_lipinski_pass_aspirin_passes():
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    passes, mw, logp, hbd, hba = lipinski_pass(mol)
    assert passes is True
    assert mw == pytest.approx(180.2, abs=0.1)
    assert hbd == 1
    assert hba == 3


def test_lipinski_pass_large_peg_fails():
    mol = Chem.MolFromSmiles(PEG_LONG_SMILES)
    passes, mw, logp, hbd, hba = lipinski_pass(mol)
    assert passes is False
    assert mw > 500
    assert hba > 10


def test_molecules_csv_entries_are_all_valid_smiles():
    with LIBRARY_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row["name"]]
    assert rows, "molecules.csv should not be empty"
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        assert mol is not None, f"{row['name']!r} has an unparsable SMILES: {row['smiles']!r}"


def test_cli_runs_end_to_end_with_default_query():
    result = subprocess.run(
        [sys.executable, str(CH02_DIR / "similarity_search.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Aspirin" in result.stdout
    assert "1.000" in result.stdout


def test_cli_respects_top_argument():
    result = subprocess.run(
        [sys.executable, str(CH02_DIR / "similarity_search.py"), "--top", "3"],
        capture_output=True,
        text=True,
        check=True,
    )
    # header + separator + 3 rows = 5 lines after the "Query:" line and blank line
    table_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(table_lines) == 3 + 2 + 1  # +1 for the "Query: ..." line
