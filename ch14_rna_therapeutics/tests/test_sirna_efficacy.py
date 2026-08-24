"""
Tests for the Chapter 14 hands-on project (sirna_efficacy.py).

Data curation and every real feature-computation function (ViennaRNA
thermodynamics, sequence-composition heuristics) are tested directly,
offline, against small, hand-checkable fixtures -- no network access
and no full model training required. This follows the same "real,
offline-testable units" convention as Chapters 11-13.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sirna_efficacy import (
    EXTENDED_LEN,
    SIRNA_LEN,
    build_feature_row,
    curate_huesken_subset,
    design_heuristic_score,
    duplex_features,
    gc_content,
    guide_self_structure,
    internal_4mer_repeats,
    target_accessibility_features,
)

# A small, real-format fixture matching siRBench's own CSV column
# layout, with one real Huesken row (values copied directly from this
# chapter's own live-fetched data/sirna_huesken_train.json), one
# non-Huesken row (must be dropped), and one deliberately malformed
# row (inconsistent target window -- must be dropped and counted).
FIXTURE_CSV = (
    "siRNA,mRNA,extended_mRNA,efficiency,source,cell_line\n"
    "GGAAGGTGATGCTTATATT,AATATAAGCATCACCTTCC,"
    "CTTTGTAGAGATTTTTAAAAATATAAGCATCACCTTCCCATTGAAGAGTGGAGAGAG,0.39,Huesken,h1299\n"
    "AAACGCTGTCGAGCCGGGT,ACCCGGCTCGACAGCGTTT,"
    "CATCCCCAAATACTCCTTCACCCGGCTCGACAGCGTTTCTGAGAAAAGCAGCGTGTC,0.70,reynolds,h1299\n"
    "GGAAGGTGATGCTTATATT,WRONGWRONGWRONGWRON,"
    "CTTTGTAGAGATTTTTAAAAATATAAGCATCACCTTCCCATTGAAGAGTGGAGAGAG,0.50,Huesken,h1299\n"
)


# --------------------------------------------------------------------------
# Real data curation
# --------------------------------------------------------------------------


def test_curate_huesken_subset_keeps_only_huesken_rows():
    records = curate_huesken_subset(FIXTURE_CSV)
    assert len(records) == 1
    assert records[0]["sirna"] == "GGAAGGUGAUGCUUAUAUU"
    assert records[0]["efficiency"] == pytest.approx(0.39)


def test_curate_huesken_subset_converts_dna_to_rna_notation():
    records = curate_huesken_subset(FIXTURE_CSV)
    assert "T" not in records[0]["sirna"]
    assert set(records[0]["sirna"]) <= set("ACGU")


def test_curate_huesken_subset_drops_rows_with_inconsistent_target_window():
    # The third fixture row's mRNA field does not match extended_mRNA's
    # own real [19:38] window -- curation must reject it, not silently
    # trust a corrupted/mismatched row.
    records = curate_huesken_subset(FIXTURE_CSV)
    sirnas = [r["sirna"] for r in records]
    assert sirnas.count("GGAAGGUGAUGCUUAUAUU") == 1  # not duplicated by the malformed row


def test_curate_huesken_subset_enforces_real_sequence_lengths():
    records = curate_huesken_subset(FIXTURE_CSV)
    for r in records:
        assert len(r["sirna"]) == SIRNA_LEN
        assert len(r["extended_target"]) == EXTENDED_LEN


# --------------------------------------------------------------------------
# Sequence-composition heuristics
# --------------------------------------------------------------------------


def test_gc_content_on_a_known_composition():
    assert gc_content("GGGGCCCCAAAAUUUU") == pytest.approx(0.5)
    assert gc_content("AAAAUUUU") == pytest.approx(0.0)
    assert gc_content("GGGGCCCC") == pytest.approx(1.0)


def test_internal_4mer_repeats_detects_a_real_repeated_motif():
    # "AAAA" appears twice, at positions 4-7 and 12-15 (disjoint,
    # non-overlapping): exactly one repeated 4-mer.
    assert internal_4mer_repeats("GGGGAAAACCCCAAAAUUU") == 1
    # A real Huesken sequence (this chapter's own cached data) with no
    # repeated internal 4-mer.
    assert internal_4mer_repeats("GGAAGGUGAUGCUUAUAUU") == 0


def test_design_heuristic_score_is_bounded_and_deterministic():
    seq = "GGAAGGUGAUGCUUAUAUU"
    s1 = design_heuristic_score(seq)
    s2 = design_heuristic_score(seq)
    assert s1 == s2
    assert 0.0 <= s1 <= 4.0


def test_design_heuristic_score_rewards_5prime_AU_end():
    base = "CGCGCGCGCGCGCGCGCGC"  # all-GC control, 5' = C (not A/U)
    au_start = "AGCGCGCGCGCGCGCGCGC"  # same but 5' forced to A
    assert design_heuristic_score(au_start) > design_heuristic_score(base)


# --------------------------------------------------------------------------
# Real ViennaRNA thermodynamic features
# --------------------------------------------------------------------------


def test_guide_self_structure_returns_real_finite_values():
    result = guide_self_structure("ACUUUUUCGCGGUUGUUAC")
    assert result["guide_self_mfe"] <= 0.0  # MFE is never positive by definition
    assert result["guide_ensemble_diversity"] >= 0.0


def test_duplex_features_full_energy_is_more_negative_than_either_end_fragment():
    # A real, physically necessary consistency check: a full 19-bp
    # duplex must be at least as stable (more negative dG) as either
    # of its own short 6-nt terminal fragments alone.
    sirna, target = "ACUUUUUCGCGGUUGUUAC", "GUAACAACCGCGAAAAAGU"
    result = duplex_features(sirna, target)
    assert result["duplex_energy_total"] < 0.0
    assert result["duplex_energy_total"] <= result["duplex_energy_5p_end"]
    assert result["duplex_energy_total"] <= result["duplex_energy_3p_end"]
    assert result["duplex_end_asymmetry"] == pytest.approx(
        result["duplex_energy_5p_end"] - result["duplex_energy_3p_end"], abs=1e-6
    )


def test_target_accessibility_opening_energy_is_non_negative():
    # Real, physically required property: forcing a region unpaired can
    # never lower the free energy below the real global unconstrained
    # minimum, so opening_energy = constrained - unconstrained >= 0.
    extended = "CUUUGUAGAGAUUUUUAAAAAUAUAAGCAUCACCUUCCCAUUGAAGAGUGGAGAGAG"
    result = target_accessibility_features(extended)
    assert result["target_opening_energy"] >= -1e-6


def test_target_accessibility_zero_for_an_already_unpaired_context():
    # A homopolymeric poly-A context has no complementary bases to pair
    # with at all, so it already has zero real secondary structure --
    # opening energy must be exactly zero regardless of which window is
    # constrained.
    extended = "A" * EXTENDED_LEN
    result = target_accessibility_features(extended)
    assert result["target_unconstrained_mfe"] == pytest.approx(0.0, abs=1e-6)
    assert result["target_opening_energy"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Full feature-row assembly
# --------------------------------------------------------------------------


def test_build_feature_row_produces_all_expected_keys():
    record = {
        "sirna": "ACUUUUUCGCGGUUGUUAC",
        "target_site": "GUAACAACCGCGAAAAAGU",
        "extended_target": "CUUUGUAGAGAUUUUUAAAGUAACAACCGCGAAAAAGUUGCGCGGAGGAGUUGUGUU",
        "efficiency": 0.5,
    }
    row = build_feature_row(record)
    expected_keys = {
        "gc_content", "seed_gc_content", "internal_4mer_repeats", "pos1_is_AU", "pos19_is_GC",
        "design_heuristic_score", "duplex_energy_total", "duplex_energy_5p_end", "duplex_energy_3p_end",
        "duplex_end_asymmetry", "guide_self_mfe", "guide_ensemble_diversity",
        "target_unconstrained_mfe", "target_opening_energy", "efficiency",
    }
    assert expected_keys <= set(row.keys())
    assert row["efficiency"] == 0.5
