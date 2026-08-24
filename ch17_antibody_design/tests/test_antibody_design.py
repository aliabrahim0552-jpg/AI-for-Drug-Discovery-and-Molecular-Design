"""
Tests for the Chapter 17 hands-on project (antibody_design.py).

Like Chapter 10, ProteinMPNN's real pretrained checkpoint is small and
fast on CPU, so these tests run the real, official model directly
against the real bundled PDB structures -- no synthetic substitute.
The PRODIGY reimplementation is validated directly against the
official tool's own published `2oob.pdb` test fixture (Vangone &
Bonvin, 2015; github.com/haddocking/prodigy), fetched byte-identical
from that repository. Only `fold_sequence`'s network call is mocked
(plus one live-API test, matching the pattern established in earlier
chapters).
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from antibody_design import (  # noqa: E402
    ACE2_RBD_PDB,
    ESMFOLD_API_URL,
    NANOBODY_RBD_PDB,
    PRODIGY_VALIDATION_PDB,
    calculate_contacts,
    classify_contacts,
    compute_ca_rmsd,
    design_nanobody_sequences,
    dg_to_kd,
    fold_sequence,
    hotspot_recovery,
    load_proteinmpnn,
    parse_model,
    percent_nis,
    predict_binding_affinity,
    prodigy_predict,
    stage1_hotspot_identification,
)

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def mpnn_model():
    return load_proteinmpnn()


# --------------------------------------------------------------------------
# Stage 1: real, geometric hotspot identification
# --------------------------------------------------------------------------


def test_ace2_rbd_structures_have_no_internal_gaps():
    """Both real RBD chains must be fully contiguous for Stage 3's
    residue-number <-> sequence-index mapping to be valid."""
    model = parse_model(NANOBODY_RBD_PDB)
    chain_b_numbers = sorted(r.id[1] for r in model["B"] if r.id[0] == " ")
    assert chain_b_numbers == list(range(1, 122))  # 121 contiguous residues


def test_stage1_hotspot_identification_finds_real_ace2_epitope():
    result = stage1_hotspot_identification()
    # Real, well-known ACE2-contacting RBD residues (Lan et al., 2020) must
    # appear among the independently, geometrically computed epitope.
    for real_hotspot in (486, 493, 501, 505):
        assert real_hotspot in result["ace2_epitope_residues"]


def test_stage1_hotspot_identification_epitopes_are_nonempty_and_plausible():
    result = stage1_hotspot_identification()
    assert 10 <= result["n_ace2_epitope_residues"] <= 40
    assert 10 <= result["n_nanobody_epitope_residues"] <= 60
    assert 10 <= result["n_nanobody_paratope_residues"] <= 60


def test_stage1_hotspot_identification_finds_real_overlap_with_ace2_site():
    """Ahmad et al. (2021) report Sb45 as 'partially buried' in the real
    ACE2 footprint -- a real, substantial but incomplete overlap, not
    zero and not total."""
    result = stage1_hotspot_identification()
    assert 0.3 < result["overlap_fraction_of_ace2_epitope"] < 1.0
    assert result["n_overlap_residues"] > 0


# --------------------------------------------------------------------------
# Stage 3: ProteinMPNN redesign of the real Sb45 nanobody scaffold
# --------------------------------------------------------------------------


def test_load_proteinmpnn_returns_the_real_architecture(mpnn_model):
    assert mpnn_model.hidden_dim == 128
    assert len(mpnn_model.encoder_layers) == 3
    assert len(mpnn_model.decoder_layers) == 3


def test_design_nanobody_sequences_returns_the_real_native_sb45_sequence(mpnn_model):
    out = design_nanobody_sequences(NANOBODY_RBD_PDB, mpnn_model, designed_chains=["B"], fixed_chains=["A"], temperature=0.1, num_samples=1, seed=0)
    assert out["native_sequence"] == (
        "QVQLVESGGGLVQAGGSLRLSCAASGFPVYRDRMAWYRQAPGKEREWVAAIYSAGQQTRYADSVKGRFTIS"
        "RDNAKNTVYLQMNSLKPEDTAVYYCNVKDVGHHYEYYDYWGQGTQVTVSA"
    )


def test_design_nanobody_sequences_designs_are_chain_b_length_only(mpnn_model):
    """Only chain B (121 residues, the nanobody) should come back as the
    designed sequence -- chain A (RBD, 196 residues) is real fixed
    context, confirming the fixed/designed split took effect."""
    out = design_nanobody_sequences(NANOBODY_RBD_PDB, mpnn_model, designed_chains=["B"], fixed_chains=["A"], temperature=0.2, num_samples=2, seed=0)
    for d in out["designs"]:
        assert len(d["sequence"]) == 121
        assert set(d["sequence"]) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_design_nanobody_sequences_recovery_is_plausible(mpnn_model):
    out = design_nanobody_sequences(NANOBODY_RBD_PDB, mpnn_model, designed_chains=["B"], fixed_chains=["A"], temperature=0.1, num_samples=3, seed=0)
    for d in out["designs"]:
        assert 0.25 < d["recovery"] < 0.95


def test_design_nanobody_sequences_is_deterministic_given_the_same_seed(mpnn_model):
    out1 = design_nanobody_sequences(NANOBODY_RBD_PDB, mpnn_model, designed_chains=["B"], fixed_chains=["A"], temperature=0.2, num_samples=1, seed=7)
    out2 = design_nanobody_sequences(NANOBODY_RBD_PDB, mpnn_model, designed_chains=["B"], fixed_chains=["A"], temperature=0.2, num_samples=1, seed=7)
    assert out1["designs"][0]["sequence"] == out2["designs"][0]["sequence"]


def test_design_nanobody_sequences_framework_recovery_exceeds_paratope_recovery(mpnn_model):
    """Real, reported finding (chapter.md Section 17.2, Stage 3): unlike
    Chapter 10's single deeply-buried hot-spot triad, the real Sb45
    paratope is a large, mostly solvent-exposed CDR-loop surface, so
    ProteinMPNN's own well-documented core-vs-surface recovery gap
    (Dauparas et al., 2022) shows up as *lower* paratope recovery than
    framework recovery -- checked here across several real seeds."""
    result = stage1_hotspot_identification()
    out = design_nanobody_sequences(NANOBODY_RBD_PDB, mpnn_model, designed_chains=["B"], fixed_chains=["A"], temperature=0.2, num_samples=5, seed=0)
    paratope_rates, framework_rates = [], []
    for d in out["designs"]:
        hs = hotspot_recovery(out["native_sequence"], d["sequence"], result["nanobody_paratope_residues"])
        paratope_rates.append(hs["hotspot_recovery"])
        framework_rates.append(hs["non_hotspot_recovery"])
    assert sum(paratope_rates) / len(paratope_rates) < sum(framework_rates) / len(framework_rates)


# --------------------------------------------------------------------------
# hotspot_recovery (pure function, no model)
# --------------------------------------------------------------------------


def test_hotspot_recovery_perfect_match_gives_1_0():
    native = "ABCDEFGHIJ"
    result = hotspot_recovery(native, native, hotspot_positions_1indexed=[1, 5, 10])
    assert result["hotspot_recovery"] == 1.0
    assert result["non_hotspot_recovery"] == 1.0


def test_hotspot_recovery_matches_manual_calculation():
    native = "ABCDEFGHIJ"
    designed = "ABXDEFXHIX"
    hotspots = [1, 5, 10]
    result = hotspot_recovery(native, designed, hotspots)
    manual = sum(designed[i - 1] == native[i - 1] for i in hotspots) / len(hotspots)
    assert result["hotspot_recovery"] == pytest.approx(manual)


# --------------------------------------------------------------------------
# Stage 4: fold_sequence (network mocked, offline-deterministic) + RMSD
# --------------------------------------------------------------------------


def test_fold_sequence_posts_the_sequence_and_returns_pdb_text():
    pdb_text = "HEADER    ESMFOLD V1 PREDICTION\nATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.90           N\n"
    mock_response = Mock()
    mock_response.text = pdb_text
    mock_response.raise_for_status = Mock()
    with patch("antibody_design.requests.post", return_value=mock_response) as mock_post:
        result = fold_sequence("QVQLVESGGGLVQAGGSLRLSCAAS")
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == ESMFOLD_API_URL
    assert result == pdb_text


def test_fold_sequence_rejects_a_non_pdb_response_after_retries():
    mock_response = Mock()
    mock_response.text = '{"message": "Endpoint request timed out"}'
    mock_response.raise_for_status = Mock()
    with patch("antibody_design.requests.post", return_value=mock_response) as mock_post:
        with pytest.raises(ValueError, match="did not return a PDB structure"):
            fold_sequence("QVQLVESGGGLVQAGGSLRLSCAAS", max_attempts=2)
    assert mock_post.call_count == 2  # real retry logic actually retried


def test_compute_ca_rmsd_self_alignment_is_near_zero():
    reference_text = NANOBODY_RBD_PDB.read_text()
    result = compute_ca_rmsd(reference_text, NANOBODY_RBD_PDB, predicted_chain="B", reference_chain="B")
    assert result["n_residues_compared"] == 121
    assert result["global_ca_rmsd"] == pytest.approx(0.0, abs=1e-6)


def test_fold_sequence_against_live_api():
    """The one test in this suite that calls the real ESM Metagenomic
    Atlas API directly, using the real native Sb45 sequence -- the
    real, retry-enabled `fold_sequence` (chapter.md Section 17.2, Stage
    4 documents the transient 504s this endpoint returns under load)."""
    pdb_text = fold_sequence(
        "QVQLVESGGGLVQAGGSLRLSCAASGFPVYRDRMAWYRQAPGKEREWVAAIYSAGQQTRYADSVKGRFTIS"
        "RDNAKNTVYLQMNSLKPEDTAVYYCNVKDVGHHYEYYDYWGQGTQVTVSA",
        timeout=120,
    )
    assert "ATOM" in pdb_text


# --------------------------------------------------------------------------
# Stage 5: PRODIGY reimplementation, validated against the official tool
# --------------------------------------------------------------------------


def test_calculate_contacts_matches_the_official_2oob_test_case():
    """Exact regression test against github.com/haddocking/prodigy's own
    `test_calculate_ic` (same bundled 2oob.pdb, same 5.5 A cutoff):
    the official test asserts exactly 78 real contacts."""
    model = parse_model(PRODIGY_VALIDATION_PDB)
    contacts = calculate_contacts(model, cutoff=5.5, selection={"A": 0, "B": 1})
    assert len(contacts) == 78


def test_classify_contacts_bins_sum_to_the_total_classifiable_contacts():
    model = parse_model(PRODIGY_VALIDATION_PDB)
    contacts = calculate_contacts(model, cutoff=5.5, selection={"A": 0, "B": 1})
    bins = classify_contacts(contacts)
    assert sum(bins.values()) <= len(contacts)
    assert sum(bins.values()) > 0


def test_predict_binding_affinity_matches_hand_computed_value():
    """Direct arithmetic check of the official PRODIGY Eq. 2 / Model 6
    coefficients (Vangone & Bonvin, 2015), independent of any PDB
    parsing."""
    dg = predict_binding_affinity(ic_cc=3, ic_ac=13, ic_pp=0, ic_ap=2, pct_nis_apolar=31.08, pct_nis_charged=41.89)
    expected = -0.09459 * 3 - 0.10007 * 13 + 0.19577 * 0 - 0.22671 * 2 + 0.18681 * 31.08 + 0.13810 * 41.89 - 15.9433
    assert dg == pytest.approx(expected)


def test_dg_to_kd_recovers_a_known_reference_pair():
    """Kd = 1e-9 M at 25 C corresponds to DeltaG = RT ln(Kd) ~ -12.30 kcal/mol."""
    dg = 0.0019858775 * 298.15 * -20.723  # ln(1e-9)
    kd = dg_to_kd(dg, temp_celsius=25.0)
    assert kd == pytest.approx(1e-9, rel=1e-3)


def test_prodigy_predict_reproduces_the_official_2oob_benchmark():
    """The official PRODIGY test suite asserts ba_val == approx(-6.2,
    abs=1.0) for this exact structure. This reimplementation swaps the
    official `freesasa`/NACCESS SASA engine for BioPython's own
    Shrake-Rupley (no prebuilt `freesasa` wheel exists for this
    environment's Python 3.12/win_amd64 combination; chapter.md Section
    17.2 documents the check), so a slightly wider tolerance is used
    here to allow for that disclosed, real substitution."""
    result = prodigy_predict(PRODIGY_VALIDATION_PDB, selection={"A": 0, "B": 1})
    assert result["n_contacts"] == 78
    assert result["predicted_dg_kcal_mol"] == pytest.approx(-6.2, abs=1.5)


def test_prodigy_predict_on_real_sb45_rbd_complex_is_favorable_and_plausible():
    result = prodigy_predict(NANOBODY_RBD_PDB, selection={"A": 0, "B": 1})
    assert result["predicted_dg_kcal_mol"] < 0  # favorable binding, as it must be for a real, stable complex
    assert 1e-12 < result["predicted_kd_M"] < 1e-6  # broad, physically sane nanomolar-to-picomolar range


def test_prodigy_predict_on_real_ace2_rbd_complex_is_favorable_and_plausible():
    result = prodigy_predict(ACE2_RBD_PDB, selection={"A": 0, "E": 1})
    assert result["predicted_dg_kcal_mol"] < 0
    assert 1e-12 < result["predicted_kd_M"] < 1e-6
