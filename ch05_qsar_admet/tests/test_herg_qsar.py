"""
Tests for the Chapter 5 hands-on project (hERG QSAR classifier).

Cleaning/splitting/featurization tests use small synthetic records and
molecules so the logic is tested precisely and fast, plus a slice of the
bundled real ChEMBL hERG extract (data/raw_herg_bioactivities_sample.json,
3000 real IC50 records for CHEMBL240 fetched 2026-08-19) for realistic
end-to-end checks. One test exercises the live ChEMBL extract_bioactivities()
call directly, as a reproducibility check on that path specifically - see
README.md for what to do if ChEMBL is unavailable when you run this.

Run with: pytest
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from herg_qsar import (
    CleanCompoundRecord,
    bemis_murcko_scaffold,
    clean_bioactivity_records,
    extract_bioactivities,
    featurize,
    load_raw_json,
    random_split,
    scaffold_split,
    standardize_smiles,
    train_and_evaluate,
)

CH05_DIR = Path(__file__).parent.parent
FIXTURE_PATH = CH05_DIR / "data" / "raw_herg_bioactivities_sample.json"

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
IBUPROFEN_SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
ASPIRIN_SALT_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O.[Na+]"


def make_record(**overrides) -> dict:
    base = {
        "molecule_chembl_id": "CHEMBL_TEST",
        "canonical_smiles": ASPIRIN_SMILES,
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "5000.0",
        "standard_units": "nM",
    }
    base.update(overrides)
    return base


def make_clean_records(n: int, scaffold_smiles: list[str]) -> list[CleanCompoundRecord]:
    """n synthetic clean records cycling through the given scaffold-bearing
    SMILES, half labeled blocker and half non-blocker."""
    records = []
    for i in range(n):
        records.append(
            CleanCompoundRecord(
                molecule_chembl_id=f"CHEMBL_SYN_{i}",
                canonical_smiles=scaffold_smiles[i % len(scaffold_smiles)],
                ic50_nm=float(1000 + i),
                n_measurements=1,
                is_blocker=(i % 2 == 0),
            )
        )
    return records


# --------------------------------------------------------------------------
# standardize_smiles
# --------------------------------------------------------------------------


def test_standardize_smiles_strips_salt_fragment():
    result = standardize_smiles(ASPIRIN_SALT_SMILES)
    assert result is not None
    assert "Na" not in result


def test_standardize_smiles_returns_none_for_garbage():
    assert standardize_smiles("not a smiles") is None


# --------------------------------------------------------------------------
# clean_bioactivity_records
# --------------------------------------------------------------------------


def test_clean_drops_records_missing_standard_value():
    records = [make_record(standard_value=None)]
    assert clean_bioactivity_records(records) == []


def test_clean_drops_censored_inequality_records():
    records = [make_record(standard_relation=">")]
    assert clean_bioactivity_records(records) == []


def test_clean_keeps_exact_and_unspecified_relation():
    records = [make_record(standard_relation="="), make_record(standard_relation=None)]
    cleaned = clean_bioactivity_records(records)
    assert len(cleaned) == 1  # both are the same compound, so they merge
    assert cleaned[0].n_measurements == 2


def test_clean_drops_non_ic50_records():
    records = [make_record(standard_type="EC50")]
    assert clean_bioactivity_records(records) == []


def test_clean_drops_unparsable_smiles():
    records = [make_record(canonical_smiles="not a smiles")]
    assert clean_bioactivity_records(records) == []


def test_clean_converts_micromolar_to_nanomolar():
    records = [make_record(standard_value="1.0", standard_units="uM")]
    cleaned = clean_bioactivity_records(records)
    assert cleaned[0].ic50_nm == pytest.approx(1000.0)


def test_clean_deduplicates_via_median():
    records = [
        make_record(standard_value="1000.0"),
        make_record(standard_value="3000.0"),
        make_record(standard_value="2000.0"),
    ]
    cleaned = clean_bioactivity_records(records)
    assert len(cleaned) == 1
    assert cleaned[0].ic50_nm == pytest.approx(2000.0)
    assert cleaned[0].n_measurements == 3


def test_clean_labels_blocker_at_threshold():
    below = make_record(standard_value="9999.0")  # 9999 nM < 10000 nM default threshold
    above = make_record(standard_value="10001.0", molecule_chembl_id="OTHER", canonical_smiles=IBUPROFEN_SMILES)
    cleaned = clean_bioactivity_records([below, above])
    by_id = {r.molecule_chembl_id: r for r in cleaned}
    assert by_id["CHEMBL_TEST"].is_blocker is True
    assert by_id["OTHER"].is_blocker is False


def test_clean_respects_custom_threshold():
    record = make_record(standard_value="5000.0")
    assert clean_bioactivity_records([record], threshold_nm=1000.0)[0].is_blocker is False
    assert clean_bioactivity_records([record], threshold_nm=10000.0)[0].is_blocker is True


# --------------------------------------------------------------------------
# featurize / bemis_murcko_scaffold
# --------------------------------------------------------------------------


def test_featurize_returns_correct_shape():
    X = featurize([ASPIRIN_SMILES, IBUPROFEN_SMILES])
    assert X.shape == (2, 2048)
    assert set(X.flatten().tolist()) <= {0, 1}


def test_bemis_murcko_scaffold_strips_substituents():
    # aspirin's scaffold is the bare benzene ring, without the ester/acid substituents
    scaffold = bemis_murcko_scaffold(ASPIRIN_SMILES)
    assert scaffold == "c1ccccc1"


def test_bemis_murcko_scaffold_groups_analogues_together():
    # aspirin and salicylic acid share a benzene-ring scaffold
    salicylic_acid = "C1=CC=C(C(=C1)C(=O)O)O"
    assert bemis_murcko_scaffold(ASPIRIN_SMILES) == bemis_murcko_scaffold(salicylic_acid)


# --------------------------------------------------------------------------
# random_split / scaffold_split
# --------------------------------------------------------------------------

# 8 distinct ring scaffolds (so the greedy scaffold-split fill has enough
# groups to approximate an 80/20 split, rather than a handful of large,
# tied groups that would all land in the same partition), 3 analogues
# each (methyl substituents, which do not change the Murcko scaffold).
_SCAFFOLD_CORES = [
    "c1ccccc1",        # benzene
    "C1CCCCC1",        # cyclohexane
    "c1ccncc1",        # pyridine
    "c1ccoc1",         # furan
    "c1ccsc1",         # thiophene
    "C1CCNCC1",        # piperidine
    "c1ccc2ccccc2c1",  # naphthalene
    "C1CCOCC1",        # tetrahydropyran
]
_MIXED_SMILES = [core + "C" * i for core in _SCAFFOLD_CORES for i in range(3)]


def test_random_split_covers_all_records_without_overlap():
    records = make_clean_records(24, _MIXED_SMILES)
    train_idx, test_idx = random_split(records, frac_train=0.75, seed=0)
    assert set(train_idx) & set(test_idx) == set()
    assert set(train_idx) | set(test_idx) == set(range(24))
    assert len(train_idx) == 18


def test_random_split_is_deterministic():
    records = make_clean_records(24, _MIXED_SMILES)
    a = random_split(records, seed=0)
    b = random_split(records, seed=0)
    assert a == b


def test_scaffold_split_puts_no_scaffold_in_both_sets():
    records = make_clean_records(24, _MIXED_SMILES)
    train_idx, test_idx = scaffold_split(records, frac_train=0.75, seed=0)
    train_scaffolds = {bemis_murcko_scaffold(records[i].canonical_smiles) for i in train_idx}
    test_scaffolds = {bemis_murcko_scaffold(records[i].canonical_smiles) for i in test_idx}
    assert train_scaffolds & test_scaffolds == set()
    assert set(train_idx) | set(test_idx) == set(range(24))


def test_scaffold_split_is_deterministic():
    records = make_clean_records(24, _MIXED_SMILES)
    a = scaffold_split(records, seed=0)
    b = scaffold_split(records, seed=0)
    assert a == b  # same seed -> identical partition, every time


def test_scaffold_split_produces_a_nonempty_test_set():
    # 8 distinct scaffold groups of size 3 give the greedy fill enough
    # granularity to leave a real held-out set, unlike a handful of large groups.
    records = make_clean_records(24, _MIXED_SMILES)
    train_idx, test_idx = scaffold_split(records, frac_train=0.8, seed=0)
    assert len(test_idx) > 0
    assert len(train_idx) > 0


# --------------------------------------------------------------------------
# train_and_evaluate
# --------------------------------------------------------------------------


def test_train_and_evaluate_rejects_unknown_model_type():
    records = make_clean_records(24, _MIXED_SMILES)
    with pytest.raises(ValueError):
        train_and_evaluate(records, model_type="not_a_model")


def test_train_and_evaluate_rejects_unknown_split_type():
    records = make_clean_records(24, _MIXED_SMILES)
    with pytest.raises(ValueError):
        train_and_evaluate(records, split_type="not_a_split")


def test_train_and_evaluate_returns_metrics_in_unit_range():
    records = make_clean_records(40, _MIXED_SMILES)
    result = train_and_evaluate(records, model_type="random_forest", split_type="scaffold", seed=0)
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= result[key] <= 1.0
    assert result["n_train"] + result["n_test"] == 40


def test_train_and_evaluate_smote_changes_train_set_only():
    records = make_clean_records(40, _MIXED_SMILES)
    baseline = train_and_evaluate(records, model_type="random_forest", split_type="random", seed=0)
    smoted = train_and_evaluate(
        records, model_type="random_forest", split_type="random", use_smote=True, seed=0
    )
    assert smoted["n_test"] == baseline["n_test"]  # SMOTE never touches the held-out test set
    assert smoted["test_blocker_frac"] == baseline["test_blocker_frac"]


# --------------------------------------------------------------------------
# Real bundled fixture (offline, deterministic) - a fast slice, not the
# full 3000-record extract, to keep the suite quick; see README.md for
# running the full pipeline.
# --------------------------------------------------------------------------


def test_clean_bioactivity_records_on_real_fixture_slice():
    raw = load_raw_json(FIXTURE_PATH)[:300]
    cleaned = clean_bioactivity_records(raw)
    assert len(cleaned) > 100  # 300 raw records dedupe down, but well above zero
    assert all(r.ic50_nm > 0 for r in cleaned)
    assert any(r.is_blocker for r in cleaned)
    assert any(not r.is_blocker for r in cleaned)


def test_scaffold_split_on_real_fixture_slice_has_no_scaffold_leakage():
    raw = load_raw_json(FIXTURE_PATH)[:300]
    cleaned = clean_bioactivity_records(raw)
    train_idx, test_idx = scaffold_split(cleaned, seed=0)
    train_scaffolds = {bemis_murcko_scaffold(cleaned[i].canonical_smiles) for i in train_idx}
    test_scaffolds = {bemis_murcko_scaffold(cleaned[i].canonical_smiles) for i in test_idx}
    assert train_scaffolds & test_scaffolds == set()


def test_train_and_evaluate_on_real_fixture_slice():
    raw = load_raw_json(FIXTURE_PATH)[:300]
    cleaned = clean_bioactivity_records(raw)
    result = train_and_evaluate(cleaned, model_type="xgboost", split_type="scaffold", seed=0)
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert result["n_train"] + result["n_test"] == len(cleaned)


# --------------------------------------------------------------------------
# Live ChEMBL extraction
# --------------------------------------------------------------------------


def test_extract_bioactivities_returns_real_herg_records():
    records = extract_bioactivities(max_records=50)
    assert len(records) == 50
    assert all(r["standard_type"] == "IC50" for r in records)


# --------------------------------------------------------------------------
# CLI (fast subset via --max-records, offline via --use-cached-raw)
# --------------------------------------------------------------------------


def test_cli_runs_end_to_end_on_a_fast_offline_subset():
    result = subprocess.run(
        [
            sys.executable,
            str(CH05_DIR / "herg_qsar.py"),
            "--use-cached-raw",
            "--max-records",
            "300",
            "--model",
            "xgboost",
            "--split",
            "scaffold",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scaffold split" in result.stdout
    assert "roc_auc" in result.stdout
