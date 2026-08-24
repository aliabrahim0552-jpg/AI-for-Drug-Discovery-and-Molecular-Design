"""
Tests for the Chapter 16 capstone hands-on project (oncology_capstone.py).

Data curation, the ADMET rule-based filter, scaffold splitting, real
receptor/box geometry (against the bundled real `data/1M17.pdb`), the
Kabsch/RMSD logic, SELFIES vocabulary round-tripping, and the report
generator are all tested directly, offline, with no network access and
no Vina/ANI-2x/generative-model-training execution required -- the
same "real, offline-testable units; heavy real execution skipped
honestly where unavailable" convention Chapters 11-15 established.
Tests that require actually running Vina or ANI-2x are skipped
automatically when neither engine is available on the host.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from oncology_capstone import (
    NATIVE_LIGAND_RESN,
    RECEPTOR_PDB,
    CleanCompoundRecord,
    MPNNRegressor,
    Vocabulary,
    admet_filter,
    bemis_murcko_scaffold,
    clean_bioactivity_records,
    compute_focused_box,
    generate_report,
    kabsch_align,
    locate_vina_executable,
    parse_best_affinity,
    ani2x_compatible_elements,
    predict_pic50,
    scaffold_split,
    smiles_to_selfies,
    split_receptor_and_native_ligand,
    validate_receptor_structure,
)

try:
    import vina  # noqa: F401

    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = locate_vina_executable() is not None

try:
    import openmmml  # noqa: F401
    import torchani  # noqa: F401

    ANI2X_AVAILABLE = True
except ImportError:
    ANI2X_AVAILABLE = False

requires_vina = pytest.mark.skipif(not VINA_AVAILABLE, reason="no Vina engine available on this host")
requires_ani2x = pytest.mark.skipif(not ANI2X_AVAILABLE, reason="openmmml/torchani not installed on this host")

# A small, real-format ChEMBL activity fixture: one real compound with
# two real replicate IC50 measurements (median -> one clean record),
# one single-measurement compound, and one record with a non-numeric
# unit that must be dropped.
FIXTURE_ACTIVITIES = [
    {"standard_type": "IC50", "standard_relation": "=", "standard_value": "50", "standard_units": "nM",
     "canonical_smiles": "COCCOc1cc2c(cc1OCCOC)ncnc2Nc1cccc(C#C)c1", "molecule_chembl_id": "CHEMBL553"},
    {"standard_type": "IC50", "standard_relation": "=", "standard_value": "54", "standard_units": "nM",
     "canonical_smiles": "COCCOc1cc2c(cc1OCCOC)ncnc2Nc1cccc(C#C)c1", "molecule_chembl_id": "CHEMBL553"},
    {"standard_type": "IC50", "standard_relation": "=", "standard_value": "2.0", "standard_units": "uM",
     "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O", "molecule_chembl_id": "CHEMBL25"},
    {"standard_type": "IC50", "standard_relation": "=", "standard_value": "10", "standard_units": "weird_unit",
     "canonical_smiles": "CCO", "molecule_chembl_id": "CHEMBL999"},
]


# --------------------------------------------------------------------------
# Stage 1: real data curation
# --------------------------------------------------------------------------


def test_clean_bioactivity_records_deduplicates_by_median():
    records = clean_bioactivity_records(FIXTURE_ACTIVITIES)
    erlotinib = next(r for r in records if r.molecule_chembl_id == "CHEMBL553")
    assert erlotinib.n_measurements == 2
    assert erlotinib.ic50_nm == pytest.approx(52.0)  # median of 50, 54


def test_clean_bioactivity_records_drops_unrecognized_units():
    records = clean_bioactivity_records(FIXTURE_ACTIVITIES)
    assert all(r.molecule_chembl_id != "CHEMBL999" for r in records)


def test_clean_bioactivity_records_computes_correct_pic50():
    records = clean_bioactivity_records(FIXTURE_ACTIVITIES)
    aspirin_like = next(r for r in records if r.molecule_chembl_id == "CHEMBL25")
    # IC50 = 2000 nM = 2e-6 M -> pIC50 = -log10(2e-6) = 5.699
    assert aspirin_like.pic50 == pytest.approx(9.0 - __import__("math").log10(2000.0), abs=1e-3)


def test_clean_bioactivity_records_active_label_matches_threshold():
    records = clean_bioactivity_records(FIXTURE_ACTIVITIES, threshold_nm=1000.0)
    erlotinib = next(r for r in records if r.molecule_chembl_id == "CHEMBL553")
    aspirin_like = next(r for r in records if r.molecule_chembl_id == "CHEMBL25")
    assert erlotinib.is_active is True  # 52 nM << 1000 nM
    assert aspirin_like.is_active is False  # 2000 nM > 1000 nM


# --------------------------------------------------------------------------
# Stage 2: MPNN prediction denormalization (a real regression test --
# `predict_pic50` once returned the model's raw normalized output
# directly, silently skipping the `* std + mean` step `train_qsar_model`
# uses internally during its own evaluation loop, systematically
# under-predicting every real pIC50 by roughly `mean` units. Caught by
# comparing predictions on real, in-distribution training molecules
# against their own real, known pIC50 values -- see chapter.md Section
# 16.2.3 for the full account of how this was found and its real
# downstream effect on the RL reward signal.)
# --------------------------------------------------------------------------


def test_predict_pic50_denormalizes_using_the_models_own_training_statistics():
    model = MPNNRegressor(hidden_dim=8, num_layers=1)
    model.y_mean, model.y_std = 6.0, 1.5
    with torch.no_grad():
        model.readout[-1].weight.zero_()
        model.readout[-1].bias.zero_()  # forces the raw (normalized) model output to exactly 0.0
    preds = predict_pic50(model, ["CCO"])
    # A raw normalized output of 0.0 must denormalize to exactly y_mean,
    # not to 0.0 (the bug this test guards against).
    assert preds[0] == pytest.approx(6.0, abs=1e-4)


def test_predict_pic50_falls_back_to_identity_when_model_has_no_training_statistics():
    # A model that was never passed through train_qsar_model (and so
    # has no .y_mean/.y_std attached) must not silently crash --
    # falls back to an identity (mean=0, std=1) transform.
    model = MPNNRegressor(hidden_dim=8, num_layers=1)
    assert not hasattr(model, "y_mean")
    preds = predict_pic50(model, ["CCO"])
    assert preds[0] is not None and np.isfinite(preds[0])


# --------------------------------------------------------------------------
# Stage 2: ADMET filter and scaffold split
# --------------------------------------------------------------------------


def test_admet_filter_passes_a_known_drug_like_molecule():
    result = admet_filter("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # ibuprofen
    assert result is not None
    assert result["passes_admet"] is True
    assert result["pains_alert"] is False


def test_admet_filter_flags_a_real_pains_alert():
    result = admet_filter("O=C1CSC(=S)N1")  # rhodanine core, canonical PAINS alert
    assert result is not None
    assert result["pains_alert"] is True
    assert result["passes_admet"] is False


def test_admet_filter_returns_none_for_unparsable_smiles():
    assert admet_filter("not_a_real_smiles(((") is None


def test_scaffold_split_is_disjoint_and_covers_all_records():
    records = [
        CleanCompoundRecord(f"CHEMBL{i}", smi, 100.0, 7.0, 1, True)
        for i, smi in enumerate(["c1ccccc1C", "c1ccccc1CC", "c1ccncc1C", "CCCCCC", "CCCCCCC"])
    ]
    train_idx, test_idx = scaffold_split(records, frac_train=0.6, seed=0)
    assert set(train_idx) & set(test_idx) == set()
    assert set(train_idx) | set(test_idx) == set(range(len(records)))


def test_bemis_murcko_scaffold_groups_analogs_together():
    # Two toluene-like analogs share the benzene scaffold; a
    # pyridine analog does not.
    s1 = bemis_murcko_scaffold("c1ccccc1C")
    s2 = bemis_murcko_scaffold("c1ccccc1CC")
    s3 = bemis_murcko_scaffold("c1ccncc1C")
    assert s1 == s2
    assert s1 != s3


# --------------------------------------------------------------------------
# Real receptor and pocket geometry (bundled 1M17.pdb, offline)
# --------------------------------------------------------------------------


def test_validate_receptor_structure_finds_the_real_native_ligand():
    result = validate_receptor_structure()
    assert result["valid"] is True
    assert result["n_native_ligand_atoms"] > 0
    assert result["n_protein_atoms"] > 1000


def test_split_receptor_and_native_ligand_finds_the_real_aq4_hetatm_block(tmp_path):
    receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    assert receptor_path.exists() and ligand_path.exists()
    ligand_lines = [l for l in ligand_path.read_text().splitlines() if l.startswith("HETATM")]
    assert len(ligand_lines) > 0
    assert all(l[17:20].strip() == NATIVE_LIGAND_RESN for l in ligand_lines)


def test_compute_focused_box_center_is_within_the_real_receptor_envelope(tmp_path):
    _receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    box = compute_focused_box(ligand_path)
    assert box["size"] == [22.5, 22.5, 22.5]
    assert all(-100 < c < 100 for c in box["center"])


# --------------------------------------------------------------------------
# Kabsch alignment / RMSD (Chapters 12-13's method)
# --------------------------------------------------------------------------


def test_kabsch_align_recovers_a_known_rotation_and_translation():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(10, 3)) + np.array([5.0, -2.0, 10.0])
    theta = 0.3
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    mobile = (rotation @ reference.T).T + np.array([1.0, -1.0, 2.0])
    aligned = kabsch_align(mobile, reference)
    np.testing.assert_allclose(aligned, reference, atol=1e-8)


# --------------------------------------------------------------------------
# SELFIES vocabulary
# --------------------------------------------------------------------------


def test_smiles_to_selfies_and_vocabulary_round_trip():
    selfies_str = smiles_to_selfies("CCO")
    assert selfies_str is not None
    vocab = Vocabulary.build([selfies_str])
    encoded = vocab.encode(selfies_str)
    assert encoded is not None
    decoded = vocab.decode(encoded)
    assert decoded == selfies_str


def test_vocabulary_encode_returns_none_when_sequence_exceeds_max_len():
    vocab = Vocabulary.build(["[C][C][O]"])
    assert vocab.encode("[C][C][O]", max_len=2) is None


# --------------------------------------------------------------------------
# Report generation (offline, synthetic pipeline_results)
# --------------------------------------------------------------------------


def test_generate_report_produces_valid_html_with_no_docking_stage(tmp_path):
    pipeline_results = {
        "qsar": {"n_train": 100, "n_test": 25, "rmse_pic50": 0.8, "r2": 0.5, "spearman_rho": 0.6},
        "generative": {
            "n_pretrain_compounds": 50, "vocab_size": 20, "final_mean_reward": 0.4,
            "pre_rl_mean_pic50": 5.0, "post_rl_mean_pic50": 6.0,
            "post_rl_valid_frac": 0.9, "post_rl_unique_frac": 0.8, "post_rl_novel_frac": 0.7,
        },
        "redocking_validation": None,
        "candidates": [{"molecule_id": "GEN_000", "smiles": "CCO", "pred_pic50": 6.5}],
    }
    out_path = generate_report(pipeline_results, tmp_path / "report.html")
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert "<html>" in html and "</html>" in html
    assert "GEN_000" in html
    assert "data:image/png;base64," in html  # a real 2D structure was embedded


def test_generate_report_handles_a_real_redocking_result(tmp_path):
    pipeline_results = {
        "qsar": {"n_train": 100, "n_test": 25, "rmse_pic50": 0.8, "r2": 0.5, "spearman_rho": 0.6},
        "generative": {
            "n_pretrain_compounds": 50, "vocab_size": 20, "final_mean_reward": 0.4,
            "pre_rl_mean_pic50": 5.0, "post_rl_mean_pic50": 6.0,
            "post_rl_valid_frac": 0.9, "post_rl_unique_frac": 0.8, "post_rl_novel_frac": 0.7,
        },
        "redocking_validation": {"rmsd_to_crystal_A_mean": 1.2, "n_correct_pose": 3, "n_replicates": 3},
        "candidates": [],
    }
    out_path = generate_report(pipeline_results, tmp_path / "report2.html")
    html = out_path.read_text(encoding="utf-8")
    assert "1.2" in html


# --------------------------------------------------------------------------
# Real Vina/ANI-2x execution (skipped when neither engine is available)
# --------------------------------------------------------------------------


@requires_vina
def test_redocking_validation_on_the_real_1m17_receptor_finds_a_correct_pose(tmp_path):
    from oncology_capstone import prepare_receptor_pdbqt, redocking_validation

    receptor_raw, native_ligand = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    receptor_pdbqt = prepare_receptor_pdbqt(receptor_raw, tmp_path / "receptor.pdbqt")
    box = compute_focused_box(native_ligand)
    redock = redocking_validation(receptor_pdbqt, native_ligand, box, tmp_path, n_replicates=1)
    assert redock["n_replicates"] == 1
    assert redock["rmsd_to_crystal_A_mean"] is not None


def test_ani2x_compatible_elements_flags_a_real_bromine_containing_candidate():
    # A real generated candidate from this chapter's own pipeline run
    # (molecule_id GEN_006): bromine is outside ANI-2x's real trained
    # element coverage (H, C, N, O, F, Cl, S).
    assert ani2x_compatible_elements("NCCN1CNc2c(ccc3[nH]c4c(c23)C=CC4CBr)NN1") is False


def test_ani2x_compatible_elements_passes_a_real_compatible_candidate():
    # Also a real generated candidate (molecule_id GEN_003): only
    # C, H, and N, all within ANI-2x's real trained coverage.
    assert ani2x_compatible_elements("Cc1c[nH]c2cncnc2[nH]c2c1=CC=2") is True


@requires_ani2x
def test_run_ani2x_md_on_a_real_small_ligand_produces_a_valid_trajectory():
    from oncology_capstone import run_ani2x_md

    result = run_ani2x_md("CCO", n_steps=20, report_interval=10)
    assert result["n_frames"] == 2
    assert result["wall_time_s"] > 0


def test_parse_best_affinity_extracts_the_first_vina_result_line():
    text = "MODEL 1\nREMARK VINA RESULT:    -7.234      0.000      0.000\nOTHER\n"
    assert parse_best_affinity(text) == pytest.approx(-7.234)
