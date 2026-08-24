"""
Tests for the Chapter 10 hands-on project (protein_design.py).

Unlike Chapters 8-9's protein language models, ProteinMPNN's real
pretrained checkpoint is small (~6.7 MB) and fast on CPU (~1s per
design call for these backbones), so these tests run the real,
official model directly -- no tiny/untrained substitute is needed the
way Chapters 8-9 use for their much larger language models. Only
`fold_sequence`'s network call is mocked (plus one live-API test,
matching the pattern established in earlier chapters); everything
model-related runs the real checkpoint against the real bundled PDB
structures.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from protein_design import (
    ESMFOLD_API_URL,
    MDM2_P53_PDB,
    P53_HOTSPOT_INDICES,
    UBIQUITIN_PDB,
    compute_ca_rmsd,
    design_sequences,
    fold_sequence,
    hotspot_recovery,
    load_model,
)

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def model():
    return load_model()


# --------------------------------------------------------------------------
# load_model
# --------------------------------------------------------------------------


def test_load_model_returns_the_real_architecture(model):
    assert model.hidden_dim == 128
    assert len(model.encoder_layers) == 3
    assert len(model.decoder_layers) == 3


# --------------------------------------------------------------------------
# design_sequences: whole-chain redesign (real 1UBQ backbone)
# --------------------------------------------------------------------------


def test_design_sequences_whole_chain_returns_the_real_native_ubiquitin_sequence(model):
    out = design_sequences(UBIQUITIN_PDB, model, designed_chains=None, temperature=0.1, num_samples=1, seed=0)
    assert out["native_sequence"] == "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def test_design_sequences_whole_chain_produces_full_length_designs(model):
    out = design_sequences(UBIQUITIN_PDB, model, designed_chains=None, temperature=0.1, num_samples=2, seed=0)
    for d in out["designs"]:
        assert len(d["sequence"]) == 76
        assert set(d["sequence"]) <= set("ACDEFGHIKLMNPQRSTVWY")  # no 'X'/unknown in a real design


def test_design_sequences_recovery_is_plausible_given_the_published_benchmark(model):
    """Dauparas et al. (2022) report 52.4% average native-sequence
    recovery on native backbones; a single real design on one backbone
    won't hit that exactly, but it should land in a broad plausible
    neighborhood, not near 0% (random) or 100% (a bug re-emitting the
    input)."""
    out = design_sequences(UBIQUITIN_PDB, model, designed_chains=None, temperature=0.1, num_samples=3, seed=0)
    for d in out["designs"]:
        assert 0.25 < d["recovery"] < 0.95


def test_design_sequences_is_deterministic_given_the_same_seed(model):
    out1 = design_sequences(UBIQUITIN_PDB, model, designed_chains=None, temperature=0.2, num_samples=1, seed=7)
    out2 = design_sequences(UBIQUITIN_PDB, model, designed_chains=None, temperature=0.2, num_samples=1, seed=7)
    assert out1["designs"][0]["sequence"] == out2["designs"][0]["sequence"]


def test_design_sequences_different_seeds_usually_differ(model):
    out = design_sequences(UBIQUITIN_PDB, model, designed_chains=None, temperature=0.3, num_samples=5, seed=0)
    sequences = {d["sequence"] for d in out["designs"]}
    assert len(sequences) > 1  # sampling stochasticity actually does something


# --------------------------------------------------------------------------
# design_sequences: binder-only redesign (real 1YCR MDM2-p53 complex)
# --------------------------------------------------------------------------


def test_design_sequences_binder_only_keeps_native_peptide_as_ground_truth(model):
    out = design_sequences(MDM2_P53_PDB, model, designed_chains=["B"], fixed_chains=["A"], temperature=0.2, num_samples=1, seed=0)
    assert out["native_sequence"] == "ETFSDLWKLLPEN"  # real resolved p53 peptide, chain B of 1YCR


def test_design_sequences_binder_only_designs_are_peptide_length_only(model):
    """Only chain B (13 residues) should come back as the 'designed'
    sequence -- chain A (MDM2, 85 residues) is real fixed context, not
    part of the output, confirming the fixed/designed chain split
    actually took effect rather than silently designing everything."""
    out = design_sequences(MDM2_P53_PDB, model, designed_chains=["B"], fixed_chains=["A"], temperature=0.2, num_samples=2, seed=0)
    for d in out["designs"]:
        assert len(d["sequence"]) == 13


def test_design_sequences_binder_only_recovers_the_real_hotspot_triad(model):
    """Real structural finding this chapter reports: ProteinMPNN
    preserves Phe19/Trp23/Leu26 (Kussie et al., 1996's real,
    literature-verified MDM2-binding hot-spot triad) far more reliably
    than solvent-exposed positions, because the fixed backbone
    structurally requires hydrophobic residues there. Checked here
    across several real seeds, not asserted from a single lucky draw."""
    out = design_sequences(MDM2_P53_PDB, model, designed_chains=["B"], fixed_chains=["A"], temperature=0.2, num_samples=5, seed=0)
    hotspot_rates = [hotspot_recovery(out["native_sequence"], d["sequence"])["hotspot_recovery"] for d in out["designs"]]
    assert min(hotspot_rates) >= 2 / 3  # at least 2 of the 3 hot-spot residues recovered, every time


# --------------------------------------------------------------------------
# hotspot_recovery (pure function, no model)
# --------------------------------------------------------------------------


def test_hotspot_recovery_matches_manual_calculation():
    native = "ETFSDLWKLLPEN"
    designed = "ETFEELWSKLPQS"  # a real design from this project's own run
    result = hotspot_recovery(native, designed, hotspot_indices=P53_HOTSPOT_INDICES)
    manual_hotspot = sum(designed[i] == native[i] for i in P53_HOTSPOT_INDICES) / len(P53_HOTSPOT_INDICES)
    assert result["hotspot_recovery"] == pytest.approx(manual_hotspot)


def test_hotspot_recovery_perfect_match_gives_1_0():
    native = "ETFSDLWKLLPEN"
    result = hotspot_recovery(native, native)
    assert result["hotspot_recovery"] == 1.0
    assert result["non_hotspot_recovery"] == 1.0


def test_hotspot_recovery_indices_land_on_the_real_triad():
    """Position 2/6/9 of the real bundled 1YCR peptide sequence must be
    F/W/L -- the exact triad Kussie et al. (1996) report, not an
    off-by-one error in the hard-coded indices."""
    native = "ETFSDLWKLLPEN"
    assert native[P53_HOTSPOT_INDICES[0]] == "F"
    assert native[P53_HOTSPOT_INDICES[1]] == "W"
    assert native[P53_HOTSPOT_INDICES[2]] == "L"


# --------------------------------------------------------------------------
# fold_sequence (network mocked, offline-deterministic) + compute_ca_rmsd
# --------------------------------------------------------------------------


def test_fold_sequence_posts_the_sequence_and_returns_pdb_text():
    pdb_text = "HEADER    ESMFOLD V1 PREDICTION\nATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.90           N\n"
    mock_response = Mock()
    mock_response.text = pdb_text
    mock_response.raise_for_status = Mock()
    with patch("protein_design.requests.post", return_value=mock_response) as mock_post:
        result = fold_sequence("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == ESMFOLD_API_URL
    assert result == pdb_text


def test_fold_sequence_rejects_a_non_pdb_response():
    mock_response = Mock()
    mock_response.text = '{"message": "Endpoint request timed out"}'
    mock_response.raise_for_status = Mock()
    with patch("protein_design.requests.post", return_value=mock_response):
        with pytest.raises(ValueError, match="did not return a PDB structure"):
            fold_sequence("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")


def test_compute_ca_rmsd_self_alignment_is_near_zero():
    reference_text = (DATA_DIR / "1UBQ.pdb").read_text()
    result = compute_ca_rmsd(reference_text, DATA_DIR / "1UBQ.pdb")
    assert result["n_residues_compared"] == 76
    assert result["global_ca_rmsd"] == pytest.approx(0.0, abs=1e-6)


def test_fold_sequence_against_live_api():
    """The one test in this suite that calls the real ESM Metagenomic
    Atlas API directly, using the native ubiquitin sequence -- known
    fast and reliable (Chapter 9; reconfirmed in this chapter's own
    real run) unlike the redesigned/peptide sequences chapter.md
    discusses timing out."""
    pdb_text = fold_sequence("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")
    assert "ATOM" in pdb_text
