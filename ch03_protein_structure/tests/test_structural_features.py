"""
Tests for the Chapter 3 hands-on project (structural feature extraction).

Most tests use the bundled 1M17.pdb fixture (data/1M17.pdb, the real EGFR
tyrosine kinase domain + erlotinib structure also used in Chapter 1) so
they run offline and deterministically. One test exercises the live
RCSB PDB download path directly, as a reproducibility check.

Run with: pytest
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from structural_features import (
    classify_secondary_structure,
    compute_binding_pocket,
    compute_contact_map,
    compute_phi_psi,
    fetch_pdb_structure,
    find_ligand_residue,
    get_standard_residues,
    load_chain,
    save_contact_map_npy,
    save_phi_psi_csv,
    save_pocket_csv,
    summarize_secondary_structure,
)

FIXTURE_PDB = Path(__file__).parent.parent / "data" / "1M17.pdb"


@pytest.fixture(scope="module")
def chain():
    return load_chain(FIXTURE_PDB, "1M17", "A")


def test_get_standard_residues_excludes_ligand_and_water(chain):
    residues = get_standard_residues(chain)
    resnames = {r.get_resname() for r in residues}
    assert "AQ4" not in resnames
    assert "HOH" not in resnames
    assert len(residues) == 312  # confirmed against the real structure


def test_compute_phi_psi_returns_one_row_per_residue(chain):
    rows = compute_phi_psi(chain)
    assert len(rows) == 312
    # chain N-terminus has no phi
    assert rows[0]["phi"] is None
    assert rows[0]["psi"] is not None
    # an interior residue has both
    interior = rows[5]
    assert interior["phi"] is not None
    assert interior["psi"] is not None
    assert -180.0 <= interior["phi"] <= 180.0
    assert -180.0 <= interior["psi"] <= 180.0


@pytest.mark.parametrize(
    "phi,psi,expected",
    [
        (-57.0, -47.0, "helix"),  # canonical alpha helix (Pauling et al., 1951)
        (-120.0, 120.0, "sheet"),  # canonical beta strand
        (None, -47.0, "coil"),  # chain terminus
        (60.0, 60.0, "coil"),  # disallowed / left-handed region
    ],
)
def test_classify_secondary_structure(phi, psi, expected):
    assert classify_secondary_structure(phi, psi) == expected


def test_summarize_secondary_structure_fractions_sum_to_one(chain):
    rows = compute_phi_psi(chain)
    summary = summarize_secondary_structure(rows)
    assert summary["total"] == 312
    assert sum(summary["counts"].values()) == 312
    assert summary["fractions"]["helix"] + summary["fractions"]["sheet"] + summary["fractions"]["coil"] == pytest.approx(1.0)
    # a kinase domain has a mixed alpha/beta fold, not all one class
    assert summary["fractions"]["helix"] > 0.1
    assert summary["fractions"]["sheet"] > 0.1


def test_compute_contact_map_is_symmetric_with_self_contact(chain):
    residue_ids, contact_map = compute_contact_map(chain, cutoff=8.0)
    assert contact_map.shape == (312, 312)
    assert len(residue_ids) == 312
    assert np.array_equal(contact_map, contact_map.T)
    assert contact_map.diagonal().all()  # every residue contacts itself


def test_compute_contact_map_larger_cutoff_gives_more_contacts(chain):
    _, small = compute_contact_map(chain, cutoff=6.0)
    _, large = compute_contact_map(chain, cutoff=10.0)
    assert large.sum() > small.sum()


def test_find_ligand_residue_locates_erlotinib(chain):
    ligand = find_ligand_residue(chain, "AQ4")
    assert ligand.get_resname() == "AQ4"


def test_find_ligand_residue_autodetects_without_resname(chain):
    ligand = find_ligand_residue(chain, None)
    assert ligand.get_resname() == "AQ4"  # only non-water heteroresidue in this structure


def test_find_ligand_residue_raises_for_missing_ligand(chain):
    with pytest.raises(ValueError):
        find_ligand_residue(chain, "ZZZ")


def test_compute_binding_pocket_finds_known_contact_residue(chain):
    pocket = compute_binding_pocket(chain, ligand_resname="AQ4", cutoff=5.0)
    resnames_and_ids = {(row["resname"], row["resseq"]) for row in pocket}
    # MET769 is the closest pocket residue to the ligand in this structure
    # (~2.7A, computed - not asserted from memory), consistent with a
    # hinge-region hydrogen-bond contact typical of this inhibitor class.
    assert ("MET", 769) in resnames_and_ids
    assert all(row["min_distance_angstrom"] <= 5.0 for row in pocket)
    assert pocket == sorted(pocket, key=lambda row: row["min_distance_angstrom"])


def test_compute_binding_pocket_empty_beyond_zero_cutoff(chain):
    pocket = compute_binding_pocket(chain, ligand_resname="AQ4", cutoff=0.0)
    assert pocket == []


def test_save_functions_write_expected_files(tmp_path, chain):
    phi_psi_rows = compute_phi_psi(chain)
    csv_path = save_phi_psi_csv(phi_psi_rows, tmp_path / "phi_psi.csv")
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "resname,resseq,phi,psi"

    pocket_rows = compute_binding_pocket(chain, "AQ4")
    pocket_path = save_pocket_csv(pocket_rows, tmp_path / "pocket.csv")
    assert pocket_path.exists()

    _, contact_map = compute_contact_map(chain)
    npy_path = save_contact_map_npy(contact_map, tmp_path / "contacts.npy")
    loaded = np.load(npy_path)
    assert np.array_equal(loaded, contact_map)


def test_fetch_pdb_structure_downloads_real_file(tmp_path):
    out_path = fetch_pdb_structure("1M17", tmp_path / "1m17_fresh.pdb")
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8", errors="ignore")
    assert content.startswith("HEADER")
    assert "EPIDERMAL GROWTH FACTOR RECEPTOR" in content
