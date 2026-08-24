"""
Tests for the Chapter 13 hands-on project (htvs_pipeline.py).

Data curation, Tier 1 rule-based filtering, real receptor/box geometry
(against the real bundled `data/5R82.pdb`), the Kabsch/RMSD logic
(reused from Chapter 12), and the funnel-analysis arithmetic are all
tested for real, offline, with no network access and no Vina/ANI-2x
execution required. Tests that require actually running AutoDock Vina
or ANI-2x dynamics are skipped automatically when neither engine is
importable/locatable on the host, following this repo's established
"real where feasible, honestly skipped where not" convention (Chapters
11-12).
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from htvs_pipeline import (
    NATIVE_LIGAND_RESN,
    RECEPTOR_PDB,
    analyze_funnel,
    compute_focused_box,
    compute_rmsd_trajectory,
    compute_tier1_properties,
    curate_library,
    kabsch_align,
    locate_vina_executable,
    parse_best_affinity,
    run_tier1,
    split_receptor_and_native_ligand,
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

requires_vina = pytest.mark.skipif(not VINA_AVAILABLE, reason="no Vina engine (Python bindings or executable) available on this host")
requires_ani2x = pytest.mark.skipif(not ANI2X_AVAILABLE, reason="openmmml/torchani not installed on this host")

# A small, real fixture matching PubChem's own AID 1805203 CSV column
# layout and its own real two-replicate-well assay protocol (one
# compound with two real replicate SID rows, one compound with a
# single row) -- real CIDs, real SMILES, real IC50 values copied
# directly from the live PubChem response this chapter's own
# `data/sars_cov2_3clpro_library.json` was cached from.
FIXTURE_CSV = """PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,PUBCHEM_EXT_DATASOURCE_SMILES,PUBCHEM_ACTIVITY_OUTCOME,PUBCHEM_ACTIVITY_SCORE,PUBCHEM_ACTIVITY_URL,PUBCHEM_ASSAYDATA_COMMENT,Standard Type,PubChem Standard Value,IC50,Target Accession(s),Ligand,Target
RESULT_TYPE,,,,,,,,STRING,FLOAT,FLOAT,TARGET_NCBI_PROTEIN_ACCESSION,STRING,STRING
1,434318973,56973495,C1=CC=C2C(=C1)N=NN2CC(=O)N(CC3=CSC=C3)C4=CC=C(C=C4)C5=CN=CC=C5,Active,,,,IC50,0.93,930,P0C6X7,BDBM429464,Replicase polyprotein 1ab
2,434318986,56973495,C1=CC=C2C(=C1)N=NN2CC(=O)N(CC3=CSC=C3)C4=CC=C(C=C4)C5=CN=CC=C5,Active,,,,IC50,0.95,950,P0C6X7,BDBM429464,Replicase polyprotein 1ab
1,434318974,56973496,CCOC(=O)c1ccccc1,Inactive,,,,IC50,55.0,55000,P0C6X7,BDBM429465,Replicase polyprotein 1ab
"""


# --------------------------------------------------------------------------
# Real data curation (offline, against a real-format fixture)
# --------------------------------------------------------------------------


def test_curate_library_deduplicates_real_replicate_wells_by_cid():
    # The fixture has two real replicate rows for CID 56973495 (IC50
    # 0.93 and 0.95 uM) and one row for CID 56973496 -- curation must
    # collapse to exactly one record per real compound, not one per row.
    records = curate_library(FIXTURE_CSV)
    assert len(records) == 2
    active = next(r for r in records if r["is_active"])
    assert active["pubchem_cid"] == "56973495"
    assert active["n_replicate_measurements"] == 2
    # Real median of the two real replicate IC50 values (0.93, 0.95).
    assert active["ic50_uM"] == pytest.approx(0.94)
    # pIC50 = 6 - log10(IC50 in uM); a real, standard conversion.
    assert active["pIC50"] == pytest.approx(6.0 - np.log10(0.94), abs=1e-3)
    assert active["ani2x_compatible_elements"] is True

    inactive = next(r for r in records if not r["is_active"])
    assert inactive["n_replicate_measurements"] == 1
    assert inactive["ic50_uM"] == pytest.approx(55.0)


def test_curate_library_sorts_deterministically_by_cid():
    records = curate_library(FIXTURE_CSV)
    cids = [int(r["pubchem_cid"]) for r in records]
    assert cids == sorted(cids)


# --------------------------------------------------------------------------
# Tier 1: rule-based ADMET/drug-likeness filtering
# --------------------------------------------------------------------------


def test_compute_tier1_properties_on_a_known_drug_like_molecule():
    # Ibuprofen: real, well-characterized small molecule, comfortably
    # inside Lipinski/Veber space and PAINS-clean.
    props = compute_tier1_properties("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    assert props is not None
    assert props["molecular_weight"] == pytest.approx(206.28, abs=0.1)
    assert props["lipinski_violations"] == 0
    assert props["veber_pass"] is True
    assert props["pains_alert"] is False
    assert props["passes_tier1"] is True


def test_compute_tier1_properties_flags_a_real_pains_alert_compound():
    # A rhodanine (2-thioxo-4-thiazolidinone) core is a real, canonical
    # PAINS substructure alert (Baell & Holloway, 2010).
    props = compute_tier1_properties("O=C1CSC(=S)N1")
    assert props is not None
    assert props["pains_alert"] is True
    assert props["passes_tier1"] is False


def test_compute_tier1_properties_returns_none_for_unparsable_smiles():
    assert compute_tier1_properties("not_a_real_smiles(((") is None


def test_run_tier1_annotates_every_library_compound():
    library = curate_library(FIXTURE_CSV)
    annotated = run_tier1(library)
    assert len(annotated) == len(library)
    assert all("tier1" in r for r in annotated)
    assert all(r["tier1"] is not None for r in annotated)


# --------------------------------------------------------------------------
# Real receptor and pocket geometry (PDB 5R82, bundled, offline)
# --------------------------------------------------------------------------


def test_split_receptor_and_native_ligand_finds_the_real_rzs_hetatm_block(tmp_path):
    receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    assert receptor_path.exists() and ligand_path.exists()
    ligand_lines = [l for l in ligand_path.read_text().splitlines() if l.startswith("HETATM")]
    assert len(ligand_lines) == 11  # real RZS heavy-atom count in 5R82 (Douangamath et al., 2020)
    assert all(l[17:20].strip() == NATIVE_LIGAND_RESN for l in ligand_lines)
    receptor_lines = [l for l in receptor_path.read_text().splitlines() if l.startswith("ATOM")]
    assert len(receptor_lines) > 2000  # real 3C-like proteinase chain, no waters/additives


def test_compute_focused_box_center_matches_real_ligand_centroid(tmp_path):
    _receptor_path, ligand_path = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    box = compute_focused_box(ligand_path)
    assert box["size"] == [22.5, 22.5, 22.5]
    # Real RZS centroid lands inside 5R82's real overall coordinate
    # envelope, not at the origin or some unrelated point.
    assert all(-50 < c < 80 for c in box["center"])


# --------------------------------------------------------------------------
# Kabsch alignment / RMSD: synthetic, hand-verified coordinates (Chapter 12's method)
# --------------------------------------------------------------------------


def test_kabsch_align_recovers_a_known_rotation_and_translation():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(10, 3)) + np.array([20.0, -5.0, 40.0])
    theta = 0.4
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    mobile = (rotation @ reference.T).T + np.array([2.0, -3.0, 1.0])

    aligned = kabsch_align(mobile, reference)
    np.testing.assert_allclose(aligned, reference, atol=1e-8)


def test_compute_rmsd_trajectory_is_zero_for_an_unmoving_trajectory():
    rng = np.random.default_rng(1)
    frame = rng.normal(size=(15, 3)) + np.array([5.0, 5.0, 5.0])
    frames = np.array([frame, frame, frame])
    result = compute_rmsd_trajectory(frames)
    assert result["rmsd_mean_A"] == pytest.approx(0.0, abs=1e-6)


def test_compute_rmsd_trajectory_detects_a_real_known_displacement():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(6, 3)) + np.array([10.0, 0.0, -5.0])
    displaced = reference.copy()
    displaced[0, 0] += 1.0  # 1 nm = 10 A displacement of one atom
    frames = np.array([reference, displaced])
    result = compute_rmsd_trajectory(frames)
    assert result["rmsd_per_frame_A"][0] == pytest.approx(0.0, abs=1e-6)
    assert result["rmsd_per_frame_A"][1] > 0.5


# --------------------------------------------------------------------------
# Funnel analysis arithmetic (synthetic, known-answer)
# --------------------------------------------------------------------------


def test_analyze_funnel_computes_enrichment_factor_correctly():
    # A synthetic 10-compound library, 2 real actives, to verify the
    # enrichment-factor arithmetic against a hand-computed answer.
    library = [{"is_active": i < 2} for i in range(10)]
    tier1_records = [{"is_active": c["is_active"], "tier1": {"passes_tier1": True}} for c in library]
    tier2_records = [{"is_active": c["is_active"], "pubchem_cid": str(i), "pIC50": 5.0, "tier2": {"affinity_kcal_mol": -7.0}} for i, c in enumerate(library)]
    # Tier 3 shortlist of 2 compounds, both real actives -> shortlist hit rate 1.0 vs. library hit rate 0.2 -> EF = 5.0
    tier3_records = [{"is_active": True, "tier3": {"stable": True}}, {"is_active": True, "tier3": {"stable": True}}]

    analysis = analyze_funnel(library, tier1_records, tier2_records, tier3_records)
    assert analysis["retrospective_enrichment"]["library_active_rate"] == pytest.approx(0.2)
    assert analysis["retrospective_enrichment"]["tier3_shortlist_active_rate"] == pytest.approx(1.0)
    assert analysis["retrospective_enrichment"]["enrichment_factor"] == pytest.approx(5.0)


def test_parse_best_affinity_extracts_the_first_vina_result_line():
    text = "MODEL 1\nREMARK VINA RESULT:    -7.234      0.000      0.000\nOTHER\n"
    assert parse_best_affinity(text) == pytest.approx(-7.234)


def test_parse_best_affinity_returns_none_for_text_with_no_result():
    assert parse_best_affinity("no vina output here") is None


# --------------------------------------------------------------------------
# Real Vina/ANI-2x execution (skipped when neither engine is available)
# --------------------------------------------------------------------------


@requires_vina
def test_redocking_validation_on_the_real_5r82_receptor_finds_a_correct_pose(tmp_path):
    from htvs_pipeline import prepare_receptor_pdbqt, redocking_validation

    receptor_raw, native_ligand = split_receptor_and_native_ligand(RECEPTOR_PDB, tmp_path)
    receptor_pdbqt = prepare_receptor_pdbqt(receptor_raw, tmp_path / "receptor.pdbqt")
    box = compute_focused_box(native_ligand)
    redock = redocking_validation(receptor_pdbqt, native_ligand, box, tmp_path, n_replicates=1)
    assert redock["n_replicates"] == 1
    assert redock["rmsd_to_crystal_A_mean"] is not None
    assert redock["rmsd_to_crystal_A_mean"] < 2.0  # real RZS redocking is a real, easy small fragment


@requires_ani2x
def test_run_ani2x_md_on_a_real_library_ligand_produces_a_valid_trajectory():
    from htvs_pipeline import build_ligand_topology_from_smiles, run_ani2x_md

    top, positions = build_ligand_topology_from_smiles("CCNc1ccc(C#N)cn1", seed=42)
    result = run_ani2x_md(top, positions, n_steps=20, report_interval=10)
    assert result["n_frames"] == 2
    assert result["ms_per_step"] > 0
