"""
Tests for the Chapter 11 hands-on project (molecular_docking.py).

Receptor/ligand preparation (RDKit, Meeko, OpenBabel) all run for
real, offline, against the real bundled PDB 1M17 structure -- none of
that needs network access or AutoDock Vina itself. The ChEMBL
curation logic is tested against small, synthetic fixture activity
records (not a live API call) so it stays deterministic and fast.
Tests that require actually running AutoDock Vina (the `vina` Python
bindings or a standalone executable) are skipped automatically when
neither is available on the host -- true on this book's own Windows
authoring environment, where `pip install vina` has no prebuilt wheel
(see chapter.md's feasibility note) unless a standalone binary is
pointed to via `VINA_EXECUTABLE`.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent.parent))
from molecular_docking import (
    BLIND_BOX_PADDING,
    FOCUSED_BOX_SIZE,
    MAX_LIGAND_MW,
    NATIVE_LIGAND_RESN,
    NATIVE_LIGAND_SMILES,
    RECEPTOR_PDB,
    compute_docking_boxes,
    curate_benchmark_set,
    locate_vina_executable,
    parse_best_affinity,
    prepare_ligand_pdbqt,
    prepare_receptor_pdbqt,
    split_receptor_and_native_ligand,
)

DATA_DIR = Path(__file__).parent.parent / "data"

VINA_AVAILABLE = locate_vina_executable() is not None
try:
    import vina  # noqa: F401

    VINA_AVAILABLE = True
except ImportError:
    pass

requires_vina = pytest.mark.skipif(not VINA_AVAILABLE, reason="No AutoDock Vina engine (Python bindings or executable) available on this host")


# --------------------------------------------------------------------------
# curate_benchmark_set: real curation logic, synthetic fixture activities
# --------------------------------------------------------------------------


def _activity(mol_id, smiles, pchembl):
    return {"molecule_chembl_id": mol_id, "canonical_smiles": smiles, "pchembl_value": str(pchembl)}


FIXTURE_ACTIVITIES = [
    _activity("CHEMBL1", "CCO", 4.0),  # ethanol -- low potency placeholder
    _activity("CHEMBL1", "CCO", 4.2),  # repeated measurement of the same molecule
    _activity("CHEMBL2", "c1ccccc1", 5.0),  # benzene
    _activity("CHEMBL3", "CC(=O)Oc1ccccc1C(=O)O", 6.0),  # aspirin
    _activity("CHEMBL4", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 7.0),  # caffeine
    _activity("CHEMBL5", "invalid_smiles_xyz", 8.0),  # unparseable -- must be dropped
    _activity("CHEMBL6", "[Na+].[Cl-]", 9.0),  # two fragments -- must be dropped
]


def test_curate_benchmark_set_deduplicates_repeated_measurements():
    curated = curate_benchmark_set(FIXTURE_ACTIVITIES, n=10)
    chembl1 = next(c for c in curated if c["chembl_id"] == "CHEMBL1")
    assert chembl1["n_measurements"] == 2
    assert chembl1["pchembl_value"] == pytest.approx(4.1)


def test_curate_benchmark_set_drops_unparseable_and_multi_fragment_entries():
    curated = curate_benchmark_set(FIXTURE_ACTIVITIES, n=10)
    ids = {c["chembl_id"] for c in curated}
    assert "CHEMBL5" not in ids  # invalid SMILES
    assert "CHEMBL6" not in ids  # multi-fragment (salt)
    assert ids == {"CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"}


def test_curate_benchmark_set_respects_molecular_weight_filter():
    heavy_smiles = "C" * 60  # a single-fragment, RDKit-valid, unrealistically heavy chain
    activities = FIXTURE_ACTIVITIES + [_activity("CHEMBL_HEAVY", heavy_smiles, 5.0)]
    curated = curate_benchmark_set(activities, n=10)
    ids = {c["chembl_id"] for c in curated}
    assert "CHEMBL_HEAVY" not in ids
    mol = Chem.MolFromSmiles(heavy_smiles)
    from rdkit.Chem import Descriptors

    assert Descriptors.MolWt(mol) > MAX_LIGAND_MW  # sanity check on the fixture itself


def test_curate_benchmark_set_spans_the_full_potency_range_when_n_is_smaller_than_pool():
    curated = curate_benchmark_set(FIXTURE_ACTIVITIES, n=2)
    assert len(curated) == 2
    pchembl_values = [c["pchembl_value"] for c in curated]
    assert min(pchembl_values) == pytest.approx(4.1)  # weakest real compound in the pool
    assert max(pchembl_values) == pytest.approx(7.0)  # most potent real compound in the pool


# --------------------------------------------------------------------------
# Real receptor geometry (PDB 1M17, bundled)
# --------------------------------------------------------------------------


def test_split_receptor_and_native_ligand_finds_the_real_erlotinib_hetatm_block(tmp_path):
    receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    assert receptor_path.exists() and ligand_path.exists()
    receptor_lines = receptor_path.read_text().splitlines()
    ligand_lines = [l for l in ligand_path.read_text().splitlines() if l.startswith("HETATM")]
    assert all(l.startswith("ATOM") for l in receptor_lines if l != "END")
    assert len(ligand_lines) == 29  # real erlotinib heavy-atom count in 1M17's AQ4 record
    assert all(l[17:20].strip() == NATIVE_LIGAND_RESN for l in ligand_lines)


def test_compute_docking_boxes_focused_center_matches_real_ligand_centroid(tmp_path):
    receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    boxes = compute_docking_boxes(receptor_path, ligand_path)
    assert boxes["focused_size"] == [FOCUSED_BOX_SIZE] * 3
    # Real, known erlotinib pocket centroid in 1M17 (approximately (22, 0, 53))
    assert boxes["focused_center"][0] == pytest.approx(22.0, abs=0.5)
    assert boxes["focused_center"][2] == pytest.approx(52.8, abs=0.5)


def test_compute_docking_boxes_blind_covers_the_real_receptor_with_padding(tmp_path):
    receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    boxes = compute_docking_boxes(receptor_path, ligand_path)
    from molecular_docking import pdb_atom_coords

    protein_coords = pdb_atom_coords(receptor_path)
    real_extent = protein_coords.max(axis=0) - protein_coords.min(axis=0)
    for i in range(3):
        assert boxes["blind_size"][i] == pytest.approx(real_extent[i] + 2 * BLIND_BOX_PADDING, abs=1e-6)


# --------------------------------------------------------------------------
# Real ligand preparation (RDKit + Meeko, offline)
# --------------------------------------------------------------------------


def test_prepare_ligand_pdbqt_produces_a_valid_pdbqt_block():
    pdbqt = prepare_ligand_pdbqt("CCO", seed=42)  # ethanol
    assert pdbqt is not None
    assert "ROOT" in pdbqt and "ENDROOT" in pdbqt
    assert pdbqt.count("ATOM") >= 3  # C, C, O heavy atoms at minimum


def test_prepare_ligand_pdbqt_is_deterministic_for_a_fixed_seed():
    first = prepare_ligand_pdbqt(NATIVE_LIGAND_SMILES, seed=42)
    second = prepare_ligand_pdbqt(NATIVE_LIGAND_SMILES, seed=42)
    assert first == second


def test_prepare_ligand_pdbqt_returns_none_for_unparseable_smiles():
    assert prepare_ligand_pdbqt("not_a_real_smiles_string!!", seed=42) is None


# --------------------------------------------------------------------------
# Real receptor preparation (OpenBabel, offline)
# --------------------------------------------------------------------------


def test_prepare_receptor_pdbqt_produces_rigid_pdbqt_from_the_real_receptor(tmp_path):
    receptor_path, _ = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    out_path = prepare_receptor_pdbqt(receptor_path, tmp_path / "receptor.pdbqt")
    text = out_path.read_text()
    assert "ATOM" in text
    assert "ROOT" not in text  # a rigid receptor has no torsion tree


# --------------------------------------------------------------------------
# parse_best_affinity: fixture-based, no Vina execution required
# --------------------------------------------------------------------------


def test_parse_best_affinity_reads_the_first_remark_line():
    fixture = (
        "MODEL 1\n"
        "REMARK VINA RESULT:    -7.440      0.000      0.000\n"
        "REMARK INTER + INTRA:         -13.117\n"
    )
    assert parse_best_affinity(fixture) == pytest.approx(-7.440)


def test_parse_best_affinity_returns_none_when_no_remark_present():
    assert parse_best_affinity("MODEL 1\nATOM ...\n") is None


# --------------------------------------------------------------------------
# Real AutoDock Vina execution (skipped when no engine is available)
# --------------------------------------------------------------------------


@requires_vina
def test_redocking_validation_reproduces_a_plausible_pose(tmp_path):
    from molecular_docking import redocking_validation

    receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    receptor_pdbqt = prepare_receptor_pdbqt(receptor_path, tmp_path / "receptor.pdbqt")
    boxes = compute_docking_boxes(receptor_path, ligand_path)
    result = redocking_validation(receptor_pdbqt, ligand_path, boxes, tmp_path, n_replicates=1)
    assert result["affinity_kcal_mol_mean"] < 0  # a favorable (negative) predicted binding affinity
    assert result["rmsd_to_crystal_A_mean"] >= 0
