"""
Tests for the Chapter 4 hands-on project (ChEMBL ETL pipeline).

Cleaning/standardization tests use synthetic records with deliberately
messy properties (censored values, missing fields, unusual units, salts,
duplicate measurements) so the logic is tested precisely, plus the
bundled real raw ChEMBL extract (data/raw_egfr_bioactivities_sample.json,
200 real EGFR bioactivity records fetched 2026-08-16) for an end-to-end,
realistic check. One test exercises the live ChEMBL extract() call
directly, as a reproducibility check on that path specifically - see
README.md for what to do if ChEMBL is unavailable when you run this.

Run with: pytest
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from etl_pipeline import (
    ACCEPTED_STANDARD_TYPES,
    CleanCompoundRecord,
    clean_bioactivity_records,
    compute_lipinski_descriptors,
    extract_bioactivities,
    load_raw_json,
    save_clean_csv,
    save_raw_json,
    standardize_smiles,
)

FIXTURE_PATH = Path(__file__).parent.parent / "data" / "raw_egfr_bioactivities_sample.json"
ASPIRIN_SALT_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O.[Na+]"  # sodium aspirin salt


def make_record(**overrides) -> dict:
    base = {
        "molecule_chembl_id": "CHEMBL_TEST",
        "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "100.0",
        "standard_units": "nM",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# standardize_smiles / compute_lipinski_descriptors
# --------------------------------------------------------------------------


def test_standardize_smiles_strips_salt_fragment():
    result = standardize_smiles(ASPIRIN_SALT_SMILES)
    assert result is not None
    assert "Na" not in result


def test_standardize_smiles_returns_none_for_garbage():
    assert standardize_smiles("not a smiles") is None


def test_standardize_smiles_canonicalizes_tautomers_to_same_output():
    # cyclohexanone keto and enol tautomers should standardize identically
    keto = standardize_smiles("O=C1CCCCC1")
    enol = standardize_smiles("OC1=CCCCC1")
    assert keto == enol


def test_compute_lipinski_descriptors_matches_known_aspirin_values():
    desc = compute_lipinski_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert desc["molecular_weight"] == pytest.approx(180.2, abs=0.1)
    assert desc["h_bond_donors"] == 1
    assert desc["h_bond_acceptors"] == 3
    assert desc["passes_lipinski"] is True


def test_compute_lipinski_descriptors_returns_none_for_garbage():
    assert compute_lipinski_descriptors("not a smiles") is None


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
    assert len(cleaned) == 1  # both are the same compound+assay, so they merge
    assert cleaned[0].n_measurements == 2


def test_clean_drops_records_outside_accepted_standard_types():
    records = [make_record(standard_type="Solubility")]
    assert clean_bioactivity_records(records) == []
    assert "Solubility" not in ACCEPTED_STANDARD_TYPES


def test_clean_drops_unparsable_smiles():
    records = [make_record(canonical_smiles="not a smiles")]
    assert clean_bioactivity_records(records) == []


def test_clean_converts_micromolar_to_nanomolar():
    records = [make_record(standard_value="1.0", standard_units="uM")]
    cleaned = clean_bioactivity_records(records)
    assert cleaned[0].standard_value_nm == pytest.approx(1000.0)


def test_clean_drops_unrecognized_units():
    records = [make_record(standard_units="mg/mL")]
    assert clean_bioactivity_records(records) == []


def test_clean_deduplicates_same_compound_via_median():
    records = [
        make_record(standard_value="10.0"),
        make_record(standard_value="30.0"),
        make_record(standard_value="20.0"),
    ]
    cleaned = clean_bioactivity_records(records)
    assert len(cleaned) == 1
    assert cleaned[0].standard_value_nm == 20.0
    assert cleaned[0].n_measurements == 3


def test_clean_separates_different_assay_types_for_same_compound():
    records = [make_record(standard_type="IC50"), make_record(standard_type="Ki")]
    cleaned = clean_bioactivity_records(records)
    assert len(cleaned) == 2
    assert {r.standard_type for r in cleaned} == {"IC50", "Ki"}


def test_clean_output_sorted_by_potency():
    records = [
        make_record(
            molecule_chembl_id="A",
            canonical_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
            standard_value="500.0",
        ),
        make_record(
            molecule_chembl_id="B",
            canonical_smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
            standard_value="5.0",
        ),
    ]
    cleaned = clean_bioactivity_records(records)
    assert [r.molecule_chembl_id for r in cleaned] == ["B", "A"]


def test_clean_produces_dataclass_instances_with_lipinski_fields():
    cleaned = clean_bioactivity_records([make_record()])
    assert isinstance(cleaned[0], CleanCompoundRecord)
    assert cleaned[0].passes_lipinski is True


# --------------------------------------------------------------------------
# Real fixture end-to-end (offline, deterministic)
# --------------------------------------------------------------------------


def test_full_pipeline_on_real_bundled_fixture():
    raw = load_raw_json(FIXTURE_PATH)
    assert len(raw) == 200  # confirmed against the real bundled extract
    cleaned = clean_bioactivity_records(raw)
    assert 0 < len(cleaned) < len(raw)  # cleaning must actually filter/dedupe something
    for row in cleaned:
        assert row.standard_value_nm > 0
        assert row.n_measurements >= 1
        assert row.molecular_weight > 0


def test_save_clean_csv_writes_expected_header(tmp_path):
    cleaned = clean_bioactivity_records(load_raw_json(FIXTURE_PATH))
    out = save_clean_csv(cleaned, tmp_path / "clean.csv")
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "molecule_chembl_id,canonical_smiles,standard_type,standard_value_nm,"
        "n_measurements,molecular_weight,logp,h_bond_donors,h_bond_acceptors,passes_lipinski"
    )


def test_save_and_load_raw_json_roundtrip(tmp_path):
    raw = load_raw_json(FIXTURE_PATH)[:5]
    path = save_raw_json(raw, tmp_path / "roundtrip.json")
    reloaded = load_raw_json(path)
    assert reloaded == raw


# --------------------------------------------------------------------------
# Live API (reproducibility check on the extract path itself)
# --------------------------------------------------------------------------


def test_extract_bioactivities_paginates_against_live_api():
    records = extract_bioactivities("CHEMBL203", max_records=150, page_size=100)
    assert len(records) == 150  # confirms pagination actually spans >1 page
    assert all(r.get("target_chembl_id") == "CHEMBL203" for r in records)
