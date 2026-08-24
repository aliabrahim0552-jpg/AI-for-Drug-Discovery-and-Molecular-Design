"""
Tests for the Chapter 8 hands-on project (esm_variant_effect.py).

Two kinds of fixtures keep this suite fast, offline, and deterministic:
  - `data/gb1_single_mutants_sample.csv`: a real (not synthetic) 77-row
    extract of the Wu et al. (2016) GB1 dataset -- the complete
    wild-type + single-mutant population -- so data-loading tests run
    against genuine experimental values without a live download.
  - `fixtures/esm2_vocab.txt`: the real ESM-2 tokenizer vocabulary (33
    tokens, matching facebook/esm2_t*_UR50D on the Hugging Face Hub)
    paired with a tiny, randomly-initialized (untrained) EsmForMaskedLM
    / EsmModel, so model-dependent tests exercise the real tokenization
    and masked-marginal/embedding-pooling mechanics without downloading
    a multi-hundred-MB pretrained checkpoint.
"""
import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import EsmConfig, EsmForMaskedLM, EsmModel
from transformers.models.esm.tokenization_esm import EsmTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from esm_variant_effect import (
    AMINO_ACIDS,
    GB1_DOMAIN_LENGTH,
    GB1_MUTATED_POSITIONS,
    GB1_WT_VARIANT,
    ESM2VariantScorer,
    _row_to_mutant,
    download_gb1_dataset,
    evaluate_embedding_perturbation,
    evaluate_zero_shot_masked_marginal,
    load_single_mutants,
)

DATA_DIR = Path(__file__).parent.parent / "data"
FIXTURE_CSV = DATA_DIR / "gb1_single_mutants_sample.csv"
VOCAB_FILE = Path(__file__).parent / "fixtures" / "esm2_vocab.txt"


# --------------------------------------------------------------------------
# Data loading (real GB1 fixture, no network)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gb1_fixture():
    return load_single_mutants(FIXTURE_CSV)


def test_wild_type_sequence_matches_known_gb1_domain(gb1_fixture):
    wt_sequence, _ = gb1_fixture
    assert len(wt_sequence) == GB1_DOMAIN_LENGTH
    # positions are 1-indexed; residues at 39, 40, 41, 54 must spell VDGV (Wu et al., 2016)
    assert wt_sequence[38] == "V"
    assert wt_sequence[39] == "D"
    assert wt_sequence[40] == "G"
    assert wt_sequence[53] == "V"


def test_loads_complete_population_of_single_mutants(gb1_fixture):
    _, mutants = gb1_fixture
    assert len(mutants) == 76
    # 4 positions x 19 non-wild-type amino acids = the entire HD=1 neighborhood
    from collections import Counter

    position_counts = Counter(m.position for m in mutants)
    assert set(position_counts) == set(GB1_MUTATED_POSITIONS)
    assert all(count == 19 for count in position_counts.values())


def test_known_mutant_fitness_value_matches_published_data(gb1_fixture):
    _, mutants = gb1_fixture
    idgv = next(m for m in mutants if m.variant_code == "IDGV")
    assert idgv.position == 39
    assert idgv.wt_aa == "V"
    assert idgv.mut_aa == "I"
    assert idgv.fitness == pytest.approx(1.4459050863, abs=1e-6)


def test_mutant_sequence_carries_the_substitution(gb1_fixture):
    wt_sequence, mutants = gb1_fixture
    for m in mutants[:10]:
        assert m.sequence[m.position - 1] == m.mut_aa
        # every other position matches wild type
        for i, (mut_res, wt_res) in enumerate(zip(m.sequence, wt_sequence), start=1):
            if i != m.position:
                assert mut_res == wt_res


def test_row_to_mutant_returns_none_for_wild_type_row():
    row = {"Variants": GB1_WT_VARIANT, "HD": "0", "Fitness": "1.0", "sequence": "M" * GB1_DOMAIN_LENGTH, "keep": "True"}
    assert _row_to_mutant(row) is None


def test_row_to_mutant_rejects_multiple_substitutions():
    bad_code = "IIGV"  # differs from VDGV at two positions (39 and 40)
    seq = list("M" * GB1_DOMAIN_LENGTH)
    seq[38], seq[39] = "I", "I"
    row = {"Variants": bad_code, "HD": "2", "Fitness": "0.5", "sequence": "".join(seq), "keep": "True"}
    with pytest.raises(ValueError, match="exactly one substitution"):
        _row_to_mutant(row)


def test_row_to_mutant_rejects_sequence_variant_mismatch():
    row = {"Variants": "IDGV", "HD": "1", "Fitness": "0.5", "sequence": "M" * GB1_DOMAIN_LENGTH, "keep": "True"}
    with pytest.raises(ValueError, match="mismatch"):
        _row_to_mutant(row)


# --------------------------------------------------------------------------
# download_gb1_dataset (network mocked, offline-deterministic)
# --------------------------------------------------------------------------


def test_download_gb1_dataset_extracts_csv_from_zip(tmp_path):
    fake_csv_bytes = b"Variants,HD,Fitness\nVDGV,0,1.0\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("four_mutations_full_data.csv", fake_csv_bytes)
    mock_response = Mock()
    mock_response.content = buf.getvalue()
    mock_response.raise_for_status = Mock()

    dest = tmp_path / "nested" / "gb1.csv"
    with patch("esm_variant_effect.requests.get", return_value=mock_response) as mock_get:
        result_path = download_gb1_dataset(dest)
    mock_get.assert_called_once()
    assert result_path == dest
    assert dest.read_bytes() == fake_csv_bytes


# --------------------------------------------------------------------------
# ESM2VariantScorer (tiny untrained model, real tokenizer vocab, no network)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_tokenizer():
    return EsmTokenizer(vocab_file=str(VOCAB_FILE))


@pytest.fixture(scope="module")
def tiny_config(tiny_tokenizer):
    return EsmConfig(
        vocab_size=tiny_tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        pad_token_id=tiny_tokenizer.pad_token_id,
        mask_token_id=tiny_tokenizer.mask_token_id,
    )


@pytest.fixture(scope="module")
def tiny_scorer(tiny_tokenizer, tiny_config):
    torch.manual_seed(0)
    mlm_model = EsmForMaskedLM(tiny_config)
    torch.manual_seed(0)
    embedding_model = EsmModel(tiny_config, add_pooling_layer=False)
    return ESM2VariantScorer(
        model_name="tiny-test-model",
        tokenizer=tiny_tokenizer,
        mlm_model=mlm_model,
        embedding_model=embedding_model,
    )


TEST_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # real GB1 domain, 56 aa


def test_masked_marginal_log_probs_covers_all_amino_acids(tiny_scorer):
    log_probs = tiny_scorer.masked_marginal_log_probs(TEST_SEQUENCE, position=39)
    assert set(log_probs) == set(AMINO_ACIDS)
    assert all(torch.isfinite(torch.tensor(v)) for v in log_probs.values())
    assert all(v <= 0.0 for v in log_probs.values())  # log-softmax outputs are never positive


def test_masked_marginal_log_probs_varies_with_position(tiny_scorer):
    at_39 = tiny_scorer.masked_marginal_log_probs(TEST_SEQUENCE, position=39)
    at_54 = tiny_scorer.masked_marginal_log_probs(TEST_SEQUENCE, position=54)
    assert at_39 != at_54  # different local context -> different (untrained but position-dependent) distribution


def test_score_single_mutants_matches_manual_log_odds(tiny_scorer, gb1_fixture):
    wt_sequence, mutants = gb1_fixture
    subset = mutants[:5]
    scores = tiny_scorer.score_single_mutants(wt_sequence, subset)
    assert scores.shape == (5,)
    expected = []
    for m in subset:
        log_probs = tiny_scorer.masked_marginal_log_probs(wt_sequence, m.position)
        expected.append(log_probs[m.mut_aa] - log_probs[m.wt_aa])
    assert scores == pytest.approx(expected, abs=1e-5)


def test_score_single_mutants_makes_one_forward_pass_per_unique_position(tiny_scorer, gb1_fixture):
    wt_sequence, mutants = gb1_fixture
    call_count = 0
    original = tiny_scorer.masked_marginal_log_probs

    def counting_wrapper(sequence, position):
        nonlocal call_count
        call_count += 1
        return original(sequence, position)

    tiny_scorer.masked_marginal_log_probs = counting_wrapper
    try:
        tiny_scorer.score_single_mutants(wt_sequence, mutants)
    finally:
        tiny_scorer.masked_marginal_log_probs = original
    assert call_count == len(GB1_MUTATED_POSITIONS)  # 4, not 76


def test_embed_sequences_shape_and_content_sensitivity(tiny_scorer):
    seq_a = TEST_SEQUENCE
    seq_b = "A" + TEST_SEQUENCE[1:]  # single-residue perturbation
    embeddings = tiny_scorer.embed_sequences([seq_a, seq_b])
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 16  # tiny_config hidden_size
    assert not (embeddings[0] == embeddings[1]).all()


def test_embed_sequences_pools_only_real_residues(tiny_scorer):
    """Manually replicate the mean-pooling over a single un-batched
    sequence and check it matches the batched implementation, i.e. the
    <cls>/<eos> exclusion is doing what it claims."""
    encoded = tiny_scorer.tokenizer(TEST_SEQUENCE, return_tensors="pt")
    with torch.no_grad():
        hidden = tiny_scorer.embedding_model(**encoded).last_hidden_state
    manual_pooled = hidden[0, 1:-1].mean(dim=0).numpy()
    batched_pooled = tiny_scorer.embed_sequences([TEST_SEQUENCE])[0]
    assert batched_pooled == pytest.approx(manual_pooled, abs=1e-5)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def test_evaluate_zero_shot_masked_marginal_result_shape(tiny_scorer, gb1_fixture):
    wt_sequence, mutants = gb1_fixture
    result = evaluate_zero_shot_masked_marginal(tiny_scorer, wt_sequence, mutants)
    assert result["n"] == 76
    assert -1.0 <= result["spearman_rho"] <= 1.0
    assert 0.0 <= result["spearman_pvalue"] <= 1.0
    assert result["n_high_confidence"] <= 76


def test_evaluate_embedding_perturbation_result_shape(tiny_scorer, gb1_fixture):
    wt_sequence, mutants = gb1_fixture
    result = evaluate_embedding_perturbation(tiny_scorer, wt_sequence, mutants[:20])
    assert result["n"] == 20
    assert -1.0 <= result["spearman_rho"] <= 1.0
