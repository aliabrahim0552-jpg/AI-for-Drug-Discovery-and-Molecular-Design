"""
Tests for the Chapter 9 hands-on project (esmfold_structure_prediction.py).

Data-dependent tests run against real, bundled fixtures fetched from
live services on 2026-08-20 (verified bit-identical to a fresh live
call before bundling -- see chapter.md Section 9.4):
  - data/ubiquitin_esmfold_prediction.pdb / alpha_synuclein_esmfold_prediction.pdb:
    real ESMFold predictions from the live ESM Metagenomic Atlas API.
  - data/1UBQ.pdb: the real 1.8-A ubiquitin crystal structure from RCSB.
Only `test_fold_sequence_against_live_api` makes a network call; every
other test is offline and deterministic.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from esmfold_structure_prediction import (
    ALPHA_SYNUCLEIN_SEQUENCE,
    ESMFOLD_API_URL,
    UBIQUITIN_SEQUENCE,
    compute_ca_accuracy,
    evaluate_confidence_vs_accuracy,
    evaluate_disorder_signal,
    fold_sequence,
    per_residue_plddt,
)

DATA_DIR = Path(__file__).parent.parent / "data"
UBIQUITIN_PDB = (DATA_DIR / "ubiquitin_esmfold_prediction.pdb").read_text()
SYNUCLEIN_PDB = (DATA_DIR / "alpha_synuclein_esmfold_prediction.pdb").read_text()
REFERENCE_PDB_PATH = DATA_DIR / "1UBQ.pdb"


# --------------------------------------------------------------------------
# fold_sequence (network mocked, offline-deterministic)
# --------------------------------------------------------------------------


def test_fold_sequence_posts_the_sequence_and_returns_pdb_text():
    mock_response = Mock()
    mock_response.text = UBIQUITIN_PDB
    mock_response.raise_for_status = Mock()
    with patch("esmfold_structure_prediction.requests.post", return_value=mock_response) as mock_post:
        result = fold_sequence(UBIQUITIN_SEQUENCE)
    mock_post.assert_called_once_with(ESMFOLD_API_URL, data=UBIQUITIN_SEQUENCE, timeout=60)
    assert result == UBIQUITIN_PDB


def test_fold_sequence_rejects_a_non_pdb_response():
    mock_response = Mock()
    mock_response.text = '{"message": "Endpoint request timed out"}'
    mock_response.raise_for_status = Mock()
    with patch("esmfold_structure_prediction.requests.post", return_value=mock_response):
        with pytest.raises(ValueError, match="did not return a PDB structure"):
            fold_sequence(UBIQUITIN_SEQUENCE)


def test_fold_sequence_against_live_api():
    """The one test in this suite that calls the real ESM Metagenomic
    Atlas API directly (matches Chapters 1/7's pattern of one live-
    network test per hands-on project)."""
    pdb_text = fold_sequence(UBIQUITIN_SEQUENCE)
    assert "ATOM" in pdb_text
    assert "ESMFOLD" in pdb_text.upper()


# --------------------------------------------------------------------------
# per_residue_plddt (real bundled fixtures, no network)
# --------------------------------------------------------------------------


def test_per_residue_plddt_covers_every_residue_of_ubiquitin():
    confidences = per_residue_plddt(UBIQUITIN_PDB)
    assert len(confidences) == len(UBIQUITIN_SEQUENCE) == 76
    assert [c.residue_number for c in confidences] == list(range(1, 77))


def test_per_residue_plddt_covers_every_residue_of_alpha_synuclein():
    confidences = per_residue_plddt(SYNUCLEIN_PDB)
    assert len(confidences) == len(ALPHA_SYNUCLEIN_SEQUENCE) == 140


def test_per_residue_plddt_values_are_in_the_zero_to_one_range():
    """This API reports pLDDT already normalized to [0, 1], not the
    [0, 100] scale used in the original AlphaFold/ESMFold papers --
    verified directly against the bundled fixture rather than assumed."""
    confidences = per_residue_plddt(UBIQUITIN_PDB)
    values = [c.plddt for c in confidences]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert max(values) > 0.5  # sanity: not a degenerate all-zero parse


def test_per_residue_plddt_averages_over_atoms_not_just_the_first_atom():
    """Confirms the per-atom-varying B-factor claim in the docstring by
    checking the computed mean differs from any single atom's raw
    value for a residue where the atoms visibly disagree."""
    confidences = {c.residue_number: c.plddt for c in per_residue_plddt(UBIQUITIN_PDB)}
    # Residue 1 in the bundled fixture has atom B-factors 0.90/0.91/0.92/0.90/0.91/0.85/0.85/0.79
    assert confidences[1] == pytest.approx(np.mean([0.90, 0.91, 0.92, 0.90, 0.91, 0.85, 0.85, 0.79]), abs=1e-2)


# --------------------------------------------------------------------------
# compute_ca_accuracy / evaluate_confidence_vs_accuracy (real fixtures)
# --------------------------------------------------------------------------


def test_compute_ca_accuracy_matches_all_76_ubiquitin_residues():
    result = compute_ca_accuracy(UBIQUITIN_PDB, REFERENCE_PDB_PATH)
    assert result["n_residues_compared"] == 76
    assert len(result["per_residue_deviation"]) == 76


def test_compute_ca_accuracy_rmsd_is_small_for_a_well_predicted_fold():
    """ESMFold's real prediction for ubiquitin should land close to the
    real 1.8-A crystal structure (published ESMFold benchmarks report
    sub-2-A backbone accuracy for well-behaved single-domain folds like
    this one); a large RMSD would indicate a parsing or alignment bug,
    not normal model behavior for this protein."""
    result = compute_ca_accuracy(UBIQUITIN_PDB, REFERENCE_PDB_PATH)
    assert 0.0 < result["global_ca_rmsd"] < 3.0


def test_compute_ca_accuracy_rejects_self_alignment_with_near_zero_rmsd():
    """Aligning the reference structure to itself must produce ~0 RMSD --
    a sanity check on the Superimposer wiring itself."""
    reference_text = REFERENCE_PDB_PATH.read_text()
    result = compute_ca_accuracy(reference_text, REFERENCE_PDB_PATH)
    assert result["global_ca_rmsd"] == pytest.approx(0.0, abs=1e-6)


def test_evaluate_confidence_vs_accuracy_result_shape():
    result = evaluate_confidence_vs_accuracy(UBIQUITIN_PDB, REFERENCE_PDB_PATH)
    assert result["n"] == 76
    assert -1.0 <= result["spearman_rho_plddt_vs_deviation"] <= 1.0
    assert 0.0 <= result["spearman_pvalue"] <= 1.0
    assert 0.0 <= result["mean_plddt"] <= 1.0


def test_evaluate_confidence_vs_accuracy_correlation_is_negative():
    """Higher confidence should track *lower* structural error -- a
    negative Spearman correlation is the scientifically expected sign,
    not just any nonzero value."""
    result = evaluate_confidence_vs_accuracy(UBIQUITIN_PDB, REFERENCE_PDB_PATH)
    assert result["spearman_rho_plddt_vs_deviation"] < 0


# --------------------------------------------------------------------------
# evaluate_disorder_signal (real fixtures)
# --------------------------------------------------------------------------


def test_evaluate_disorder_signal_result_shape():
    result = evaluate_disorder_signal(UBIQUITIN_PDB, SYNUCLEIN_PDB)
    assert result["n_ordered"] == 76
    assert result["n_disordered"] == 140
    assert 0.0 <= result["mean_plddt_ordered"] <= 1.0
    assert 0.0 <= result["mean_plddt_disordered"] <= 1.0


def test_evaluate_disorder_signal_ordered_protein_scores_higher():
    """The ordered protein (ubiquitin) must have higher mean pLDDT than
    the DisProt-annotated fully-disordered protein (alpha-synuclein) --
    this is the entire scientific point of Section 9.4's second
    experiment, not an incidental property."""
    result = evaluate_disorder_signal(UBIQUITIN_PDB, SYNUCLEIN_PDB)
    assert result["mean_plddt_ordered"] > result["mean_plddt_disordered"]
    assert result["mannwhitney_pvalue_greater"] < 0.05
