"""
Tests for the Chapter 7 hands-on project (generative Transformer + RL
for de novo EGFR inhibitor design).

Cleaning/tokenization/model tests use small synthetic molecule sets and
tiny models (fast, deterministic) plus a fast subset of the bundled real
EGFR ChEMBL extract (data/raw_egfr_bioactivities_sample.json, 3000 real
IC50 records for CHEMBL203 fetched 2026-08-20) for realistic end-to-end
checks. One test exercises the live ChEMBL extract_bioactivities() call
directly, as a reproducibility check on that path specifically - see
README.md for what to do if ChEMBL is unavailable when you run this.

Run with: pytest
"""
import subprocess
import sys
from pathlib import Path

import pytest
import selfies as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from gen_transformer import (
    BOS,
    EOS,
    PAD,
    GenerativeTransformer,
    RewardOracle,
    Vocabulary,
    bemis_murcko_scaffold,
    clean_bioactivity_records,
    evaluate_oracle,
    extract_bioactivities,
    lipinski_pass,
    load_raw_json,
    pretrain,
    reinforce_finetune,
    scaffold_split,
    score_sequences,
    smiles_to_selfies,
    standardize_smiles,
)
from rdkit import Chem

CH07_DIR = Path(__file__).parent.parent
FIXTURE_PATH = CH07_DIR / "data" / "raw_egfr_bioactivities_sample.json"

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
_MOLECULES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "C1=CC=C(C=C1)O",  # phenol
    "C1CCCCC1",  # cyclohexane
    "c1ccncc1",  # pyridine
    "c1ccsc1",  # thiophene
    "C1CCNCC1",  # piperidine
]


def make_record(**overrides) -> dict:
    base = {
        "molecule_chembl_id": "CHEMBL_TEST",
        "canonical_smiles": ASPIRIN_SMILES,
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "500.0",
        "standard_units": "nM",
    }
    base.update(overrides)
    return base


def make_vocab_and_encoded() -> tuple[Vocabulary, list[list[int]]]:
    selfies_strings = [smiles_to_selfies(s) for s in _MOLECULES]
    selfies_strings = [s for s in selfies_strings if s is not None]
    vocab = Vocabulary.build(selfies_strings)
    encoded = [vocab.encode(s) for s in selfies_strings]
    return vocab, [e for e in encoded if e is not None]


# Module-scoped fixtures: standardizing/fitting against the bundled
# fixture is the slow part of this suite (RDKit tautomer canonicalization,
# then an XGBoost fit); every test that needs real cleaned records or a
# fitted oracle shares one of these instead of repeating that work.


@pytest.fixture(scope="module")
def fixture_records_300() -> list:
    raw = load_raw_json(FIXTURE_PATH)[:300]
    return clean_bioactivity_records(raw)


@pytest.fixture(scope="module")
def fixture_records_200() -> list:
    raw = load_raw_json(FIXTURE_PATH)[:200]
    return clean_bioactivity_records(raw)


@pytest.fixture(scope="module")
def fitted_oracle_200(fixture_records_200) -> RewardOracle:
    oracle = RewardOracle(seed=0)
    oracle.fit(fixture_records_200)
    return oracle


# --------------------------------------------------------------------------
# standardize_smiles / clean_bioactivity_records
# --------------------------------------------------------------------------


def test_standardize_smiles_returns_none_for_garbage():
    assert standardize_smiles("not a smiles") is None


def test_clean_labels_active_at_threshold():
    below = make_record(standard_value="999.0")
    above = make_record(standard_value="1001.0", molecule_chembl_id="OTHER", canonical_smiles="c1ccccc1O")
    cleaned = clean_bioactivity_records([below, above])
    by_id = {r.molecule_chembl_id: r for r in cleaned}
    assert by_id["CHEMBL_TEST"].is_active is True
    assert by_id["OTHER"].is_active is False


def test_clean_deduplicates_via_median():
    records = [make_record(standard_value=v) for v in ("400.0", "600.0", "500.0")]
    cleaned = clean_bioactivity_records(records)
    assert len(cleaned) == 1
    assert cleaned[0].ic50_nm == pytest.approx(500.0)
    assert cleaned[0].n_measurements == 3


def test_clean_drops_censored_inequality_records():
    records = [make_record(standard_relation=">")]
    assert clean_bioactivity_records(records) == []


# --------------------------------------------------------------------------
# SELFIES tokenization / Vocabulary
# --------------------------------------------------------------------------


def test_smiles_to_selfies_roundtrips():
    s = smiles_to_selfies(ASPIRIN_SMILES)
    assert s is not None
    back = sf.decoder(s)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(back)) == Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN_SMILES))


def test_smiles_to_selfies_returns_none_for_garbage():
    assert smiles_to_selfies("not a smiles") is None


def test_vocabulary_encode_decode_roundtrip():
    vocab, encoded = make_vocab_and_encoded()
    for ids in encoded:
        decoded_selfies = vocab.decode(ids)
        assert decoded_selfies  # non-empty
        smiles = sf.decoder(decoded_selfies)
        assert Chem.MolFromSmiles(smiles) is not None


def test_vocabulary_includes_special_tokens():
    vocab, _ = make_vocab_and_encoded()
    for tok in (PAD, BOS, EOS):
        assert tok in vocab.token_to_id


def test_vocabulary_rejects_sequences_longer_than_max_len():
    vocab, _ = make_vocab_and_encoded()
    long_selfies = smiles_to_selfies(ASPIRIN_SMILES) * 20  # deliberately absurdly long
    assert vocab.encode(long_selfies, max_len=10) is None


# --------------------------------------------------------------------------
# bemis_murcko_scaffold / scaffold_split / RewardOracle
# --------------------------------------------------------------------------


def test_bemis_murcko_scaffold_strips_substituents():
    assert bemis_murcko_scaffold(ASPIRIN_SMILES) == "c1ccccc1"


def test_scaffold_split_on_real_fixture_slice_has_no_scaffold_leakage(fixture_records_300):
    train_idx, test_idx = scaffold_split(fixture_records_300, seed=0)
    train_scaffolds = {bemis_murcko_scaffold(fixture_records_300[i].canonical_smiles) for i in train_idx}
    test_scaffolds = {bemis_murcko_scaffold(fixture_records_300[i].canonical_smiles) for i in test_idx}
    assert train_scaffolds & test_scaffolds == set()


def test_evaluate_oracle_on_real_fixture_slice_returns_finite_metrics(fixture_records_300):
    metrics = evaluate_oracle(fixture_records_300, seed=0)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["n_train"] + metrics["n_test"] == len(fixture_records_300)


def test_reward_oracle_predict_proba_in_unit_range(fixture_records_200, fitted_oracle_200):
    proba = fitted_oracle_200.predict_proba([r.canonical_smiles for r in fixture_records_200[:10]])
    assert all(0.0 <= p <= 1.0 for p in proba)


def test_reward_oracle_raises_before_fit():
    oracle = RewardOracle()
    with pytest.raises(RuntimeError):
        oracle.predict_proba([ASPIRIN_SMILES])


# --------------------------------------------------------------------------
# GenerativeTransformer: shapes, sampling, gradients
# --------------------------------------------------------------------------


def test_model_forward_produces_logits_over_vocab():
    vocab, encoded = make_vocab_and_encoded()
    model = GenerativeTransformer(vocab_size=len(vocab), d_model=16, n_layers=2, n_heads=2, max_len=len(encoded[0]))
    batch = torch.tensor(encoded[:4], dtype=torch.long)
    logits = model(batch)
    assert logits.shape == (4, batch.size(1), len(vocab))


def test_model_sample_produces_valid_shape_and_special_tokens():
    vocab, encoded = make_vocab_and_encoded()
    max_len = len(encoded[0])
    model = GenerativeTransformer(vocab_size=len(vocab), d_model=16, n_layers=2, n_heads=2, max_len=max_len)
    ids = model.sample(vocab, n=5, seed=0)
    assert len(ids) == 5
    assert all(len(seq) == max_len for seq in ids)
    assert all(seq[0] == vocab.token_to_id[BOS] for seq in ids)


def test_model_sample_is_deterministic_with_seed():
    vocab, encoded = make_vocab_and_encoded()
    max_len = len(encoded[0])
    model = GenerativeTransformer(vocab_size=len(vocab), d_model=16, n_layers=2, n_heads=2, max_len=max_len)
    a = model.sample(vocab, n=5, seed=0)
    b = model.sample(vocab, n=5, seed=0)
    assert a == b


def test_sample_with_logprobs_gradient_flows():
    vocab, encoded = make_vocab_and_encoded()
    max_len = len(encoded[0])
    model = GenerativeTransformer(vocab_size=len(vocab), d_model=16, n_layers=2, n_heads=2, max_len=max_len)
    ids, logprob = model.sample_with_logprobs(vocab, n=4)
    assert logprob.shape == (4,)
    loss = -logprob.mean()
    loss.backward()
    grad = model.token_embedding.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0


def test_pretrain_reduces_loss():
    vocab, encoded = make_vocab_and_encoded()
    max_len = len(encoded[0])
    model = GenerativeTransformer(vocab_size=len(vocab), d_model=16, n_layers=2, n_heads=2, max_len=max_len)
    losses = pretrain(model, vocab, encoded, epochs=20, seed=0)
    assert losses[-1] < losses[0]


# --------------------------------------------------------------------------
# score_sequences / lipinski_pass / reinforce_finetune
# --------------------------------------------------------------------------


def test_lipinski_pass_aspirin_passes():
    assert lipinski_pass(Chem.MolFromSmiles(ASPIRIN_SMILES)) is True


def test_score_sequences_scores_a_known_valid_molecule(fitted_oracle_200):
    vocab, encoded = make_vocab_and_encoded()
    reports = score_sequences(encoded, vocab, fitted_oracle_200)
    assert len(reports) == len(encoded)
    assert all(r.valid for r in reports)  # all built from real, valid molecules
    assert all(0.0 <= r.p_active <= 1.0 for r in reports)


def test_score_sequences_flags_invalid_sequence(fitted_oracle_200):
    vocab, _ = make_vocab_and_encoded()
    garbage_ids = [vocab.token_to_id[BOS]] + [vocab.token_to_id[PAD]] * (vocab.__len__() and 5)
    reports = score_sequences([garbage_ids], vocab, fitted_oracle_200)
    assert reports[0].valid is False
    assert reports[0].reward == 0.0


def test_reinforce_finetune_runs_and_returns_history(fitted_oracle_200):
    oracle = fitted_oracle_200
    vocab, encoded = make_vocab_and_encoded()
    max_len = len(encoded[0])
    model = GenerativeTransformer(vocab_size=len(vocab), d_model=16, n_layers=2, n_heads=2, max_len=max_len)
    pretrain(model, vocab, encoded, epochs=3, seed=0)
    history = reinforce_finetune(model, vocab, oracle, iterations=3, batch_size=8, seed=0)
    assert len(history) == 3
    for entry in history:
        assert 0.0 <= entry["valid_frac"] <= 1.0


# --------------------------------------------------------------------------
# Live ChEMBL extraction
# --------------------------------------------------------------------------


def test_extract_bioactivities_returns_real_egfr_records():
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
            str(CH07_DIR / "gen_transformer.py"),
            "--use-cached-raw",
            "--max-records",
            "200",
            "--pretrain-epochs",
            "2",
            "--rl-iterations",
            "2",
            "--n-sample",
            "10",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    assert "RL fine-tuned (after RL)" in result.stdout
    assert "mean_reward" in result.stdout
