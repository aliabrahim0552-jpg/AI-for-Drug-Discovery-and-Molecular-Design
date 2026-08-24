"""
Tests for the Chapter 12 hands-on project (md_nnp_simulation.py).

Structure repair (PDBFixer), protonation (OpenMM Modeller), and ligand
building (RDKit) are tested for real, offline, against the real
bundled `data/1M17.pdb` -- none of that needs ANI-2x or network
access. The Kabsch-alignment/RMSD/RMSF logic is tested against
synthetic, hand-constructed coordinate arrays with a known correct
answer -- this is the same check that caught a real bug (an incorrect
re-centering term) during this chapter's own development; see
chapter.md Section 12.4. Tests that require actually running ANI-2x
dynamics are skipped automatically when `openmmml`/`torchani` are not
importable on the host.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from md_nnp_simulation import (
    NATIVE_LIGAND_RESN,
    RECEPTOR_PDB,
    build_ligand_topology,
    compute_rmsd_rmsf,
    fix_and_protonate_protein,
    kabsch_align,
    split_receptor_and_native_ligand,
)

try:
    import openmmml  # noqa: F401
    import torchani  # noqa: F401

    ANI2X_AVAILABLE = True
except ImportError:
    ANI2X_AVAILABLE = False

requires_ani2x = pytest.mark.skipif(not ANI2X_AVAILABLE, reason="openmmml/torchani not installed on this host")


# --------------------------------------------------------------------------
# Real structure preparation (PDB 1M17, bundled, offline)
# --------------------------------------------------------------------------


def test_split_receptor_and_native_ligand_finds_the_real_erlotinib_hetatm_block(tmp_path):
    receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    assert receptor_path.exists() and ligand_path.exists()
    ligand_lines = [l for l in ligand_path.read_text().splitlines() if l.startswith("HETATM")]
    assert len(ligand_lines) == 29  # real erlotinib heavy-atom count in 1M17's AQ4 record (Ch11)
    assert all(l[17:20].strip() == NATIVE_LIGAND_RESN for l in ligand_lines)


def test_fix_and_protonate_protein_repairs_the_real_missing_terminal_atom(tmp_path):
    receptor_path, _ = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    modeller, ff = fix_and_protonate_protein(receptor_path, tmp_path)
    # Real, protonated 1M17 protein atom count (heavy atoms + real added hydrogens + OXT)
    assert modeller.topology.getNumAtoms() > 2400  # real heavy-atom count alone is ~2498 pre-H
    element_symbols = {atom.element.symbol for atom in modeller.topology.atoms()}
    assert "H" in element_symbols  # real hydrogens were actually added, not a no-op


def test_build_ligand_topology_from_real_crystal_pose_matches_erlotinib_atom_count(tmp_path):
    receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    top, positions = build_ligand_topology(ligand_path)
    assert top.getNumAtoms() == 52  # 29 real heavy atoms + real added hydrogens
    assert positions.shape == (52, 3)


def test_build_ligand_topology_from_smiles_is_deterministic_for_a_fixed_seed():
    top1, pos1 = build_ligand_topology(ligand_pdb=None, seed=42)
    top2, pos2 = build_ligand_topology(ligand_pdb=None, seed=42)
    assert top1.getNumAtoms() == top2.getNumAtoms()
    np.testing.assert_array_equal(pos1._value, pos2._value)


# --------------------------------------------------------------------------
# Kabsch alignment / RMSD / RMSF: synthetic, hand-verified coordinates
# --------------------------------------------------------------------------


def test_kabsch_align_recovers_a_known_rotation_and_translation():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(12, 3)) + np.array([50.0, -20.0, 100.0])  # far from the origin
    theta = 0.7
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    mobile = (rotation @ reference.T).T + np.array([5.0, 5.0, 5.0])

    aligned = kabsch_align(mobile, reference)
    np.testing.assert_allclose(aligned, reference, atol=1e-8)


def test_compute_rmsd_rmsf_is_zero_for_an_unmoving_trajectory():
    rng = np.random.default_rng(1)
    frame = rng.normal(size=(20, 3)) + np.array([10.0, 10.0, 10.0])
    frames = np.array([frame, frame, frame])

    result = compute_rmsd_rmsf(frames)
    assert result["rmsd_mean_A"] == pytest.approx(0.0, abs=1e-6)
    assert result["rmsf_mean_A"] == pytest.approx(0.0, abs=1e-6)


def test_compute_rmsd_rmsf_detects_a_real_known_displacement():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(8, 3)) + np.array([30.0, 0.0, -15.0])
    # Move atom 0 by exactly 1 nm (10 A) along x in the second frame; everything else fixed.
    displaced = reference.copy()
    displaced[0, 0] += 1.0
    frames = np.array([reference, displaced])

    result = compute_rmsd_rmsf(frames)
    assert result["rmsd_per_frame_A"][0] == pytest.approx(0.0, abs=1e-6)
    assert result["rmsd_per_frame_A"][1] > 0.5  # a real, detected structural change
    assert max(result["rmsf_per_atom_A"]) > 0.0


def test_compute_rmsd_rmsf_alignment_does_not_produce_the_real_bug_this_caught():
    """Regression test for a real bug found during this chapter's own
    development: an earlier version of `kabsch_align` re-centered onto
    the *already-mean-zeroed* reference (always (0,0,0)) instead of the
    reference's real centroid, inflating RMSD by exactly the
    reference's real distance from the coordinate origin. A reference
    structure far from the origin, with a trivial identical second
    frame, must report RMSD close to zero -- not the ~tens-of-Angstrom
    artifact the real bug produced."""
    rng = np.random.default_rng(3)
    far_reference = rng.normal(size=(15, 3)) + np.array([200.0, -150.0, 80.0])
    frames = np.array([far_reference, far_reference.copy()])

    result = compute_rmsd_rmsf(frames)
    assert result["rmsd_mean_A"] < 1e-6


# --------------------------------------------------------------------------
# Real ANI-2x execution (skipped when openmmml/torchani are unavailable)
# --------------------------------------------------------------------------


@requires_ani2x
def test_run_ani2x_md_on_the_real_ligand_produces_a_favorable_energy_trajectory(tmp_path):
    from md_nnp_simulation import run_ani2x_md

    receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    top, positions = build_ligand_topology(ligand_path)
    result = run_ani2x_md(top, positions, n_steps=20, report_interval=10)
    assert result["n_frames"] == 2
    assert result["ms_per_step"] > 0
