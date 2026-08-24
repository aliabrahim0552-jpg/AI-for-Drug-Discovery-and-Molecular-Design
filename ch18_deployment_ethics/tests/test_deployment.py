"""
Tests for the Chapter 18 hands-on project (train_model.py, service.py).

Data curation and the model-artifact provenance check are tested
directly, offline, against small, hand-checkable fixtures. The FastAPI
service is tested with Starlette's real `TestClient` against the real,
already-trained artifact this chapter's own `train_model.py` produced
(loaded once per test session) -- no live network access or running
server required, the same "real, offline-testable units" convention
Chapters 11-16 established.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_model import (
    MODELS_DIR,
    CleanCompoundRecord,
    clean_bioactivity_records,
    load_deployed_model,
    scaffold_split,
    sha256_of_file,
)

MODEL_ARTIFACT_EXISTS = (MODELS_DIR / "herg_xgboost.joblib").exists()
requires_trained_model = pytest.mark.skipif(not MODEL_ARTIFACT_EXISTS, reason="run train_model.py first to produce the real model artifact")

# A small, real-format ChEMBL activity fixture (same structure Chapter
# 5/7/16's own curation tests use): one real compound with two real
# replicate IC50 measurements, one single-measurement compound, and one
# record with a non-numeric unit that must be dropped.
FIXTURE_ACTIVITIES = [
    {"standard_type": "IC50", "standard_relation": "=", "standard_value": "50", "standard_units": "nM",
     "canonical_smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1", "molecule_chembl_id": "CHEMBL1000"},
    {"standard_type": "IC50", "standard_relation": "=", "standard_value": "70", "standard_units": "nM",
     "canonical_smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1", "molecule_chembl_id": "CHEMBL1000"},
    {"standard_type": "IC50", "standard_relation": "=", "standard_value": "50", "standard_units": "uM",
     "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O", "molecule_chembl_id": "CHEMBL25"},
    {"standard_type": "IC50", "standard_relation": "=", "standard_value": "10", "standard_units": "weird_unit",
     "canonical_smiles": "CCO", "molecule_chembl_id": "CHEMBL999"},
]


# --------------------------------------------------------------------------
# Data curation (Chapter 5's own methodology, reused unchanged)
# --------------------------------------------------------------------------


def test_clean_bioactivity_records_deduplicates_by_median():
    records = clean_bioactivity_records(FIXTURE_ACTIVITIES)
    terfenadine_like = next(r for r in records if r.molecule_chembl_id == "CHEMBL1000")
    assert terfenadine_like.n_measurements == 2
    assert terfenadine_like.ic50_nm == pytest.approx(60.0)  # median of 50, 70


def test_clean_bioactivity_records_drops_unrecognized_units():
    records = clean_bioactivity_records(FIXTURE_ACTIVITIES)
    assert all(r.molecule_chembl_id != "CHEMBL999" for r in records)


def test_clean_bioactivity_records_blocker_label_matches_threshold():
    records = clean_bioactivity_records(FIXTURE_ACTIVITIES, threshold_nm=10_000.0)
    terfenadine_like = next(r for r in records if r.molecule_chembl_id == "CHEMBL1000")
    aspirin = next(r for r in records if r.molecule_chembl_id == "CHEMBL25")
    assert terfenadine_like.is_blocker is True  # 60 nM << 10,000 nM
    assert aspirin.is_blocker is False  # 50,000 nM > 10,000 nM


def test_scaffold_split_is_disjoint_and_covers_all_records():
    records = [
        CleanCompoundRecord(f"CHEMBL{i}", smi, 100.0, 1, True)
        for i, smi in enumerate(["c1ccccc1C", "c1ccccc1CC", "c1ccncc1C", "CCCCCC", "CCCCCCC"])
    ]
    train_idx, test_idx = scaffold_split(records, frac_train=0.6, seed=0)
    assert set(train_idx) & set(test_idx) == set()
    assert set(train_idx) | set(test_idx) == set(range(len(records)))


# --------------------------------------------------------------------------
# Real model-artifact provenance check
# --------------------------------------------------------------------------


def test_sha256_of_file_is_deterministic(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"real content")
    assert sha256_of_file(f) == sha256_of_file(f)


@requires_trained_model
def test_load_deployed_model_passes_its_own_real_hash_check():
    model, metadata = load_deployed_model()
    assert model is not None
    assert metadata["sha256_model_file"] == sha256_of_file(MODELS_DIR / "herg_xgboost.joblib")


@requires_trained_model
def test_load_deployed_model_detects_a_real_tampered_artifact(tmp_path, monkeypatch):
    import shutil

    import train_model

    tampered_dir = tmp_path / "models"
    tampered_dir.mkdir()
    shutil.copy(MODELS_DIR / "herg_xgboost.joblib", tampered_dir / "herg_xgboost.joblib")
    shutil.copy(MODELS_DIR / "herg_xgboost.metadata.json", tampered_dir / "herg_xgboost.metadata.json")
    # Corrupt the model file after the metadata's hash was already recorded.
    with open(tampered_dir / "herg_xgboost.joblib", "ab") as f:
        f.write(b"corruption")

    monkeypatch.setattr(train_model, "MODELS_DIR", tampered_dir)
    with pytest.raises(ValueError, match="hash mismatch"):
        train_model.load_deployed_model()


# --------------------------------------------------------------------------
# Real FastAPI service (Starlette TestClient, no live server needed)
# --------------------------------------------------------------------------


@requires_trained_model
def test_health_endpoint_reports_the_real_loaded_model():
    from fastapi.testclient import TestClient

    from service import app

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["n_training_compounds"] > 0
        assert 0.0 <= body["held_out_roc_auc"] <= 1.0


@requires_trained_model
def test_predict_endpoint_flags_a_real_known_herg_blocker():
    """Terfenadine: a real drug withdrawn from market in 1998 for real,
    documented hERG-mediated cardiac arrhythmia risk -- the single most
    direct real correctness check available for this model."""
    from fastapi.testclient import TestClient

    from service import app

    with TestClient(app) as client:
        response = client.post("/predict", json={"smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1"})
        assert response.status_code == 200
        body = response.json()
        assert body["herg_blocker_predicted"] is True
        assert body["herg_blocker_probability"] > 0.5


@requires_trained_model
def test_predict_endpoint_clears_a_real_known_safe_compound():
    from fastapi.testclient import TestClient

    from service import app

    with TestClient(app) as client:
        response = client.post("/predict", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})  # aspirin
        assert response.status_code == 200
        body = response.json()
        assert body["herg_blocker_predicted"] is False
        assert body["admet"]["passes_admet"] is True


@requires_trained_model
def test_predict_endpoint_rejects_an_invalid_smiles_with_422():
    from fastapi.testclient import TestClient

    from service import app

    with TestClient(app) as client:
        response = client.post("/predict", json={"smiles": "not_a_real_smiles((("})
        assert response.status_code == 422
