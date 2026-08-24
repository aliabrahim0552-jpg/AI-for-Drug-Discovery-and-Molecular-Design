"""
Chapter 17 hands-on project: de novo neutralizing-antibody (nanobody)
design against a real viral surface antigen -- the SARS-CoV-2 spike
receptor-binding domain (RBD) -- chaining five real methods end to end.

Extract: three real PDB structures.
  - PDB 6M0J (Lan et al., 2020): the real SARS-CoV-2 RBD bound to its
    natural receptor, human ACE2 -- defines the real neutralizing
    epitope (the "ACE2-binding motif") this chapter's designed binder
    is meant to occupy.
  - PDB 7KGJ (Ahmad et al., 2021): the RBD bound to a real,
    experimentally validated synthetic nanobody (sybody), Sb45 (K_D in
    the nanomolar range by SPR) -- this chapter's real starting
    backbone for CDR redesign, substituting for an RFdiffusion-
    generated backbone per Section 17.1's feasibility finding.
  - PDB 2OOB (Peschard et al., 2007): a real, unrelated protein-protein
    complex used only to validate this chapter's own from-scratch
    PRODIGY reimplementation directly against the official tool's own
    published test fixture (Vangone & Bonvin, 2015;
    github.com/haddocking/prodigy).
Predict:
  1. Real, geometric hotspot identification: contact-based epitope
     mapping on both real complexes independently, then a real,
     computed overlap statistic between them (Section 17.1).
  2. ProteinMPNN (Dauparas et al., 2022, vendored from Chapter 10)
     redesign of the real Sb45 nanobody sequence, RBD held fixed as
     real structural context -- this chapter's real substitute for an
     RFdiffusion-generated backbone (Section 17.1's feasibility
     finding; see chapter.md).
  3. ESMFold (Chapter 9's live-API method, reused verbatim) structural
     self-consistency check of the redesigned nanobody sequences.
  4. A from-scratch, fully disclosed reimplementation of PRODIGY
     (Vangone & Bonvin, 2015) for real interface analysis and binding-
     affinity prediction (Section 17.2, Stage 5).
Evaluate:
  1. Real overlap between the independently-solved ACE2-RBD and
     Sb45-RBD interfaces (Section 17.1).
  2. Native-sequence recovery of ProteinMPNN's redesigns, overall and
     split by real, geometrically-defined paratope vs. framework
     positions (Section 17.2, Stage 3).
  3. Real Ca RMSD between ESMFold's prediction and the real 7KGJ
     nanobody backbone, for both the native and redesigned sequences
     (Section 17.2, Stage 4).
  4. This chapter's own PRODIGY reimplementation, validated against the
     official tool's own published 2OOB test case, then applied to the
     real native Sb45-RBD complex and checked against the real,
     literature-reported K_D range (Section 17.2, Stage 5).

See README.md for usage and chapter.md Section 17.2 for full context.
"""
import argparse
import copy
import json
import math
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import requests
import torch
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley

sys.path.insert(0, str(Path(__file__).parent / "third_party" / "proteinmpnn"))
from protein_mpnn_utils import (  # noqa: E402
    ProteinMPNN,
    _S_to_seq,
    _scores,
    parse_PDB,
    tied_featurize,
)

DATA_DIR = Path(__file__).parent / "data"
ACE2_RBD_PDB = DATA_DIR / "6M0J.pdb"          # chain A = ACE2, chain E = RBD
NANOBODY_RBD_PDB = DATA_DIR / "7KGJ.pdb"      # chain A = RBD, chain B = Sb45 nanobody
PRODIGY_VALIDATION_PDB = DATA_DIR / "2OOB.pdb"  # chain A = Cbl-b UBA, chain B = ubiquitin

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
HIDDEN_DIM = 128
DEFAULT_CHECKPOINT = Path(__file__).parent / "proteinmpnn_weights" / "v_48_020.pt"

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_TIMEOUT = 60

# Real heavy-atom interface-residue definition (Stage 1's own hotspot
# mapping), the standard structural-biology convention -- distinct from
# PRODIGY's own re-optimized 5.5 A cutoff (Stage 5), which is kept
# unchanged from Vangone & Bonvin (2015) since the model's regression
# weights were fit specifically at that cutoff.
EPITOPE_CONTACT_CUTOFF = 4.5
PRODIGY_IC_CUTOFF = 5.5
PRODIGY_NIS_THRESHOLD = 0.05

# Official PRODIGY residue-classification tables (Vangone & Bonvin, 2015;
# verified directly against the maintained reference implementation at
# github.com/haddocking/prodigy, src/prodigy_prot/modules/aa_properties.py
# -- not re-derived from the paper's own main-text prose, which omits
# leucine from its residue lists by a transcription error the repository
# code does not share).
AA_CHARACTER_IC = {
    "ALA": "A", "CYS": "A", "GLU": "C", "ASP": "C", "GLY": "A", "PHE": "A",
    "ILE": "A", "HIS": "C", "LYS": "C", "MET": "A", "LEU": "A", "ASN": "P",
    "GLN": "P", "PRO": "A", "SER": "P", "ARG": "C", "THR": "P", "TRP": "A",
    "VAL": "A", "TYR": "A",
}
AA_CHARACTER_NIS = {
    "ALA": "A", "CYS": "P", "GLU": "C", "ASP": "C", "GLY": "A", "PHE": "A",
    "ILE": "A", "HIS": "P", "LYS": "C", "MET": "A", "LEU": "A", "ASN": "P",
    "GLN": "P", "PRO": "A", "SER": "P", "ARG": "C", "THR": "P", "TRP": "P",
    "VAL": "A", "TYR": "P",
}
# NACCESS-derived maximum (Ala-X-Ala) solvent-accessible surface area per
# residue type, for relative-SASA normalization (Vangone & Bonvin, 2015;
# github.com/haddocking/prodigy aa_properties.py `rel_asa["total"]`).
MAX_ASA = {
    "ALA": 107.95, "CYS": 134.28, "ASP": 140.39, "GLU": 172.25, "PHE": 199.48,
    "GLY": 80.10, "HIS": 182.88, "ILE": 175.12, "LYS": 200.81, "LEU": 178.63,
    "MET": 194.15, "ASN": 143.94, "PRO": 136.13, "GLN": 178.50, "ARG": 238.76,
    "SER": 116.50, "THR": 139.27, "VAL": 151.44, "TRP": 249.36, "TYR": 212.76,
}


# --------------------------------------------------------------------------
# Stage 1: real, geometric target-binding-site definition & hotspot ID
# --------------------------------------------------------------------------


def parse_model(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    return structure[0]


def calculate_contacts(model, cutoff: float, selection: dict[str, int] | None = None) -> list[tuple]:
    """Real inter-chain residue contacts: any atom pair within `cutoff`
    Angstrom of each other. Mirrors PRODIGY's own official `calculate_ic`
    logic (Vangone & Bonvin, 2015) so that Stage 5 can be validated
    directly against that tool's own published test fixture."""
    atom_list = list(model.get_atoms())
    ns = NeighborSearch(atom_list)
    all_pairs = ns.search_all(radius=cutoff, level="R")
    if selection is None:
        return [(r1, r2) for r1, r2 in all_pairs if r1.parent.id != r2.parent.id]
    return [
        (r1, r2) for r1, r2 in all_pairs
        if r1.parent.id in selection and r2.parent.id in selection
        and selection[r1.parent.id] != selection[r2.parent.id]
    ]


def interface_residue_numbers(contacts: list[tuple], chain_id: str) -> set[int]:
    """Real PDB residue numbers on `chain_id` making at least one real
    inter-chain contact. Restricted to real amino-acid residues
    (`id[0] == " "`): both real structures used here carry
    crystallographic waters (`HOH`) assigned PDB residue numbers on the
    same chain ID as the polypeptide itself, which would otherwise be
    counted as fake, out-of-range "hotspot" positions."""
    numbers = set()
    for r1, r2 in contacts:
        for r in (r1, r2):
            if r.parent.id == chain_id and r.id[0] == " ":
                numbers.add(r.id[1])
    return numbers


def stage1_hotspot_identification() -> dict:
    """Independently maps the real ACE2-binding epitope (from 6M0J) and
    the real Sb45-nanobody epitope (from 7KGJ) on the RBD, using only
    real heavy-atom contact geometry -- no epitope residue list is
    transcribed from either paper's text or figures. Both structures
    share the real, native SARS-CoV-2 spike numbering (verified
    directly: both RBD chains run residues 333-52x with no internal
    gaps), so the two independently-solved epitope footprints are
    directly comparable residue-by-residue."""
    ace2_model = parse_model(ACE2_RBD_PDB)
    ace2_contacts = calculate_contacts(ace2_model, cutoff=EPITOPE_CONTACT_CUTOFF, selection={"A": 0, "E": 1})
    ace2_epitope = interface_residue_numbers(ace2_contacts, "E")

    nb_model = parse_model(NANOBODY_RBD_PDB)
    nb_contacts = calculate_contacts(nb_model, cutoff=EPITOPE_CONTACT_CUTOFF, selection={"A": 0, "B": 1})
    nb_epitope_on_rbd = interface_residue_numbers(nb_contacts, "A")
    nb_paratope = interface_residue_numbers(nb_contacts, "B")

    overlap = ace2_epitope & nb_epitope_on_rbd
    return {
        "ace2_epitope_residues": sorted(ace2_epitope),
        "n_ace2_epitope_residues": len(ace2_epitope),
        "nanobody_epitope_residues_on_rbd": sorted(nb_epitope_on_rbd),
        "n_nanobody_epitope_residues": len(nb_epitope_on_rbd),
        "nanobody_paratope_residues": sorted(nb_paratope),
        "n_nanobody_paratope_residues": len(nb_paratope),
        "overlap_residues": sorted(overlap),
        "n_overlap_residues": len(overlap),
        "overlap_fraction_of_ace2_epitope": len(overlap) / len(ace2_epitope),
        "overlap_fraction_of_nanobody_epitope": len(overlap) / len(nb_epitope_on_rbd),
    }


# --------------------------------------------------------------------------
# Stage 3: ProteinMPNN sequence redesign of the real Sb45 scaffold
#
# Section 17.1 (see chapter.md) re-runs Chapter 10's own RFdiffusion
# feasibility investigation for this chapter's antibody-specific case
# (Bennett et al., 2026) and reaches the same real conclusion: no
# RFdiffusion-generated backbone is used here. Stage 2 of this pipeline
# is therefore the choice of NANOBODY_RBD_PDB (a real, already-solved
# nanobody-antigen complex) as the fixed backbone Stage 3 redesigns --
# Chapter 10's own real substitution strategy, reused here.
# --------------------------------------------------------------------------


def load_proteinmpnn(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> ProteinMPNN:
    """Loads the real, official ProteinMPNN checkpoint (48 edges, 0.2 A
    training noise) with no weight modification -- identical model and
    checkpoint Chapter 10 uses, vendored fresh into this chapter's own
    folder per this book's per-chapter self-containment convention."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ProteinMPNN(
        ca_only=False, num_letters=21, node_features=HIDDEN_DIM,
        edge_features=HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
        num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def design_nanobody_sequences(
    pdb_path: Path, model: ProteinMPNN, designed_chains: list[str], fixed_chains: list[str],
    temperature: float = 0.1, num_samples: int = 1, seed: int = 0,
) -> dict:
    """Real ProteinMPNN inference redesigning `designed_chains` with
    `fixed_chains` held as real structural context -- the officially
    supported multi-chain design mechanism, Chapter 10's own
    binder-only-redesign pattern reused here for the nanobody chain.
    `chain_id_dict` is built from `parse_PDB`'s own real parsed name,
    exactly as Chapter 10's bugfix requires."""
    pdb_dict_list = parse_PDB(str(pdb_path), ca_only=False)
    name = pdb_dict_list[0]["name"]
    chain_id_dict = {name: (designed_chains, fixed_chains)}
    batch = [copy.deepcopy(pdb_dict_list[0])]
    device = torch.device("cpu")
    (
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list,
        masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx,
        dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all,
        bias_by_res_all, tied_beta,
    ) = tied_featurize(batch, device, chain_id_dict)

    native_seq = _S_to_seq(S[0], chain_M[0])
    omit_AAs_np = np.zeros(len(ALPHABET)).astype(np.float32)
    bias_AAs_np = np.zeros(len(ALPHABET)).astype(np.float32)

    designs = []
    for i in range(num_samples):
        torch.manual_seed(seed + i)
        randn = torch.randn(chain_M.shape)
        with torch.no_grad():
            sample_dict = model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx, mask=mask,
                temperature=temperature, chain_M_pos=chain_M_pos, omit_AAs_np=omit_AAs_np,
                bias_AAs_np=bias_AAs_np, bias_by_res=bias_by_res_all,
            )
            S_sample = sample_dict["S"]
            log_probs = model(
                X, S_sample, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn,
                use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"],
            )
        mask_for_loss = mask * chain_M * chain_M_pos
        score = float(_scores(S_sample, log_probs, mask_for_loss).item())
        designed_seq = _S_to_seq(S_sample[0], chain_M[0])
        recovery = float(np.mean([a == b for a, b in zip(designed_seq, native_seq)]))
        designs.append({"sequence": designed_seq, "score": score, "recovery": recovery})

    return {"name": name, "native_sequence": native_seq, "designs": designs}


def hotspot_recovery(native_seq: str, designed_seq: str, hotspot_positions_1indexed: list[int]) -> dict:
    """Recovery split by real, geometrically-derived paratope positions
    (Stage 1's own `nanobody_paratope_residues`, not a literature-
    transcribed CDR list) vs. every other (framework) position. Chain B
    of 7KGJ is a single, gap-free 1-121 numbering (verified directly),
    so PDB residue number == 1-indexed sequence position exactly."""
    hotspot_idx = [p - 1 for p in hotspot_positions_1indexed]
    non_hotspot_idx = [i for i in range(len(native_seq)) if i not in hotspot_idx]
    hotspot_matches = sum(designed_seq[i] == native_seq[i] for i in hotspot_idx)
    non_hotspot_matches = sum(designed_seq[i] == native_seq[i] for i in non_hotspot_idx)
    return {
        "n_hotspot_positions": len(hotspot_idx),
        "n_non_hotspot_positions": len(non_hotspot_idx),
        "hotspot_recovery": hotspot_matches / len(hotspot_idx) if hotspot_idx else None,
        "non_hotspot_recovery": non_hotspot_matches / len(non_hotspot_idx) if non_hotspot_idx else None,
    }


# --------------------------------------------------------------------------
# Stage 4: ESMFold structural self-consistency (Chapter 9's live-API
# method, reused verbatim). AlphaFold3 is discussed as theory only in
# chapter.md Section 17.2 -- it has no downloadable weights and no free,
# scriptable bulk-inference API (only the browser-only AlphaFold Server),
# the same real constraint Chapter 9 documents.
# --------------------------------------------------------------------------


def fold_sequence(sequence: str, timeout: int = ESMFOLD_TIMEOUT, max_attempts: int = 3) -> str:
    """Real, live ESM Metagenomic Atlas API call (Chapter 9's method).
    The free, shared, rate-limited endpoint occasionally returns a quick
    HTTP 504 under load rather than a genuine per-sequence timeout
    (observed directly: a first attempt failed after ~11 s, well under
    `timeout`, while a same-sequence retry succeeded in ~29 s) -- a
    transient server-side condition, not evidence the sequence itself
    is hard to fold, so up to `max_attempts` real attempts are made
    before this is reported as a genuine timeout."""
    last_exc: Exception | None = None
    for _ in range(max_attempts):
        try:
            response = requests.post(ESMFOLD_API_URL, data=sequence, timeout=timeout)
            response.raise_for_status()
            pdb_text = response.text
            if not pdb_text.lstrip().startswith("HEADER") and "ATOM" not in pdb_text:
                raise ValueError(f"ESMFold API did not return a PDB structure: {pdb_text[:200]!r}")
            return pdb_text
        except (requests.exceptions.Timeout, requests.exceptions.RequestException, ValueError) as exc:
            last_exc = exc
    raise last_exc


def compute_ca_rmsd(
    predicted_pdb_text: str, reference_pdb_path: Path,
    predicted_chain: str = "A", reference_chain: str = "B",
) -> dict:
    """Ca RMSD between an ESMFold prediction (always chain A, since it
    is folded as an isolated single chain) and the real 7KGJ nanobody
    backbone (chain B), matched 1:1 by residue number (Chapter 9/10's
    method)."""
    parser = PDBParser(QUIET=True)
    predicted = parser.get_structure("predicted", StringIO(predicted_pdb_text))[0][predicted_chain]
    reference = parser.get_structure("reference", str(reference_pdb_path))[0][reference_chain]
    pred_by_resnum = {r.id[1]: r for r in predicted if r.id[0] == " "}
    ref_by_resnum = {r.id[1]: r for r in reference if r.id[0] == " "}
    shared = sorted(set(pred_by_resnum) & set(ref_by_resnum))
    pred_ca = [pred_by_resnum[n]["CA"] for n in shared]
    ref_ca = [ref_by_resnum[n]["CA"] for n in shared]
    sup = Superimposer()
    sup.set_atoms(ref_ca, pred_ca)
    sup.apply(pred_ca)
    return {"n_residues_compared": len(shared), "global_ca_rmsd": float(sup.rms)}


# --------------------------------------------------------------------------
# Stage 5: real interface analysis & binding-affinity prediction -- a
# from-scratch, fully disclosed reimplementation of PRODIGY (Vangone &
# Bonvin, 2015). `freesasa` (the official tool's own SASA dependency)
# has no prebuilt wheel for this environment's Python 3.12/win_amd64
# combination (verified directly against PyPI: wheels exist only up to
# cp311); BioPython's own Shrake-Rupley SASA implementation is used
# instead, with the same NACCESS-derived per-residue normalization
# table, and the substitution is validated below against the official
# tool's own published 2OOB test case rather than assumed equivalent.
# --------------------------------------------------------------------------


def classify_contacts(contacts: list[tuple]) -> dict[str, int]:
    bins = {"AA": 0, "PP": 0, "CC": 0, "AP": 0, "CP": 0, "AC": 0}
    for r1, r2 in contacts:
        i = AA_CHARACTER_IC.get(r1.resname)
        j = AA_CHARACTER_IC.get(r2.resname)
        if i is None or j is None:
            continue
        bins["".join(sorted((i, j)))] += 1
    return bins


def compute_relative_sasa(model) -> dict[tuple, tuple]:
    """Per-residue relative solvent-accessible surface area of the full
    complex (BioPython's Shrake-Rupley, 1.4 A probe, 100 sample points --
    the same probe radius NACCESS/freesasa use)."""
    sr = ShrakeRupley(probe_radius=1.4, n_points=100)
    sr.compute(model, level="R")
    out = {}
    for chain in model:
        for residue in chain:
            if residue.id[0] != " " or residue.resname not in MAX_ASA:
                continue
            out[(chain.id, residue.id[1])] = (residue.resname, residue.sasa / MAX_ASA[residue.resname])
    return out


def percent_nis(rel_sasa: dict[tuple, tuple], threshold: float = PRODIGY_NIS_THRESHOLD) -> dict[str, float]:
    """Percentage of the complex's non-interacting surface (residues
    still solvent-exposed, rel. SASA >= `threshold`, in the bound
    complex) that is apolar / polar / charged."""
    counts = {"A": 0, "P": 0, "C": 0}
    for resname, rsa in rel_sasa.values():
        if rsa >= threshold:
            counts[AA_CHARACTER_NIS[resname]] += 1
    total = sum(counts.values())
    return {
        "pct_nis_apolar": 100.0 * counts["A"] / total,
        "pct_nis_polar": 100.0 * counts["P"] / total,
        "pct_nis_charged": 100.0 * counts["C"] / total,
    }


def predict_binding_affinity(
    ic_cc: float, ic_ac: float, ic_pp: float, ic_ap: float,
    pct_nis_apolar: float, pct_nis_charged: float,
) -> float:
    """PRODIGY's own published IC-NIS model (Vangone & Bonvin, 2015,
    Equation 2 / "Model 6"), verified directly against the official
    `IC_NIS` implementation in github.com/haddocking/prodigy
    src/prodigy_prot/modules/models.py."""
    return (
        -0.09459 * ic_cc
        - 0.10007 * ic_ac
        + 0.19577 * ic_pp
        - 0.22671 * ic_ap
        + 0.18681 * pct_nis_apolar
        + 0.13810 * pct_nis_charged
        - 15.9433
    )


def dg_to_kd(dg_kcal_mol: float, temp_celsius: float = 25.0) -> float:
    """DeltaG (kcal/mol) -> dissociation constant K_d (M), DeltaG = RT ln(K_d)."""
    temp_k = temp_celsius + 273.15
    rt = 0.0019858775 * temp_k  # kcal / (mol K)
    return math.exp(dg_kcal_mol / rt)


def prodigy_predict(pdb_path: Path, selection: dict[str, int], temp_celsius: float = 25.0) -> dict:
    model = parse_model(pdb_path)
    contacts = calculate_contacts(model, cutoff=PRODIGY_IC_CUTOFF, selection=selection)
    bins = classify_contacts(contacts)
    rel_sasa = compute_relative_sasa(model)
    nis = percent_nis(rel_sasa)
    dg = predict_binding_affinity(bins["CC"], bins["AC"], bins["PP"], bins["AP"], nis["pct_nis_apolar"], nis["pct_nis_charged"])
    kd = dg_to_kd(dg, temp_celsius)
    return {"n_contacts": len(contacts), **bins, **nis, "predicted_dg_kcal_mol": dg, "predicted_kd_M": kd}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 17: de novo neutralizing-nanobody design pipeline.")
    parser.add_argument("--skip-esmfold", action="store_true", help="Skip the live ESMFold validation calls (e.g. offline).")
    parser.add_argument("--results-path", default=str(Path(__file__).parent / "results" / "antibody_design_results.json"))
    args = parser.parse_args()

    results: dict = {}

    print("=== Stage 1: real, geometric hotspot identification ===")
    hotspots = stage1_hotspot_identification()
    print(f"  ACE2 epitope on RBD: {hotspots['n_ace2_epitope_residues']} residues")
    print(f"  Sb45 nanobody epitope on RBD: {hotspots['n_nanobody_epitope_residues']} residues")
    print(f"  Sb45 nanobody paratope: {hotspots['n_nanobody_paratope_residues']} residues")
    print(f"  Real overlap: {hotspots['n_overlap_residues']} residues "
          f"({hotspots['overlap_fraction_of_ace2_epitope']:.1%} of ACE2 epitope, "
          f"{hotspots['overlap_fraction_of_nanobody_epitope']:.1%} of nanobody epitope)")
    results["stage1_hotspot_identification"] = hotspots

    print("\n=== Stage 3: ProteinMPNN redesign of the real Sb45 nanobody scaffold ===")
    mpnn_model = load_proteinmpnn()
    temperatures = [0.1, 0.2, 0.3]
    n_samples_per_temp = 5
    sweep = []
    best_design = None
    native_seq = None
    for temp in temperatures:
        out = design_nanobody_sequences(
            NANOBODY_RBD_PDB, mpnn_model, designed_chains=["B"], fixed_chains=["A"],
            temperature=temp, num_samples=n_samples_per_temp, seed=0,
        )
        native_seq = out["native_sequence"]
        recoveries = [d["recovery"] for d in out["designs"]]
        print(f"  T={temp}: recovery = {np.mean(recoveries):.4f} +/- {np.std(recoveries):.4f} (n={n_samples_per_temp})")
        hs = [hotspot_recovery(native_seq, d["sequence"], hotspots["nanobody_paratope_residues"]) for d in out["designs"]]
        mean_hotspot = float(np.mean([h["hotspot_recovery"] for h in hs]))
        mean_non_hotspot = float(np.mean([h["non_hotspot_recovery"] for h in hs]))
        print(f"         paratope recovery = {mean_hotspot:.4f}, framework recovery = {mean_non_hotspot:.4f}")
        sweep.append({
            "temperature": temp, "mean_recovery": float(np.mean(recoveries)), "std_recovery": float(np.std(recoveries)),
            "designs": out["designs"], "hotspot_recovery": hs,
            "mean_paratope_recovery": mean_hotspot, "mean_framework_recovery": mean_non_hotspot,
        })
        if temp == 0.1:
            best_design = out["designs"][0]["sequence"]
    results["stage3_nanobody_redesign"] = {"native_sequence": native_seq, "temperature_sweep": sweep, "best_design_t0.1": best_design}

    print("\n=== Stage 4: ESMFold structural self-consistency ===")
    if not args.skip_esmfold:
        print(f"  Positive control: folding the real native Sb45 sequence ({native_seq})...")
        try:
            native_pdb = fold_sequence(native_seq)
            native_acc = compute_ca_rmsd(native_pdb, NANOBODY_RBD_PDB)
            print(f"    Native-sequence ESMFold prediction vs. real 7KGJ chain B: Ca RMSD = {native_acc['global_ca_rmsd']:.3f} A (n={native_acc['n_residues_compared']})")
            results["native_esmfold_ca_rmsd"] = native_acc
        except (requests.exceptions.Timeout, requests.exceptions.RequestException, ValueError) as exc:
            print(f"    Did not return a structure ({exc}).")
            results["native_esmfold_timed_out"] = True

        print(f"  Folding redesigned nanobody sequence: {best_design}")
        try:
            predicted_pdb = fold_sequence(best_design)
            acc = compute_ca_rmsd(predicted_pdb, NANOBODY_RBD_PDB)
            print(f"    Redesigned-sequence ESMFold prediction vs. real 7KGJ chain B: Ca RMSD = {acc['global_ca_rmsd']:.3f} A (n={acc['n_residues_compared']})")
            results["redesigned_esmfold_ca_rmsd"] = acc
        except (requests.exceptions.Timeout, requests.exceptions.RequestException, ValueError) as exc:
            print(f"    Did not return a structure ({exc}).")
            results["redesigned_esmfold_timed_out"] = True

    print("\n=== Stage 5: PRODIGY reimplementation -- validation, then real prediction ===")
    validation = prodigy_predict(PRODIGY_VALIDATION_PDB, selection={"A": 0, "B": 1})
    print(f"  Validation (2OOB, official test case): predicted ddG = {validation['predicted_dg_kcal_mol']:.2f} kcal/mol "
          f"(official tool: -6.2 +/- 1.0 kcal/mol); {validation['n_contacts']} contacts (official tool: 78)")
    results["stage5_prodigy_validation_2OOB"] = validation

    native_complex = prodigy_predict(NANOBODY_RBD_PDB, selection={"A": 0, "B": 1})
    print(f"  Real Sb45-RBD complex (7KGJ): predicted ddG = {native_complex['predicted_dg_kcal_mol']:.2f} kcal/mol, "
          f"predicted Kd = {native_complex['predicted_kd_M']:.2e} M "
          f"(literature range across 5 sybodies: 6.8-62.7 nM, Ahmad et al. 2021)")
    results["stage5_prodigy_sb45_rbd"] = native_complex

    ace2_complex = prodigy_predict(ACE2_RBD_PDB, selection={"A": 0, "E": 1})
    print(f"  Real ACE2-RBD complex (6M0J): predicted ddG = {ace2_complex['predicted_dg_kcal_mol']:.2f} kcal/mol, "
          f"predicted Kd = {ace2_complex['predicted_kd_M']:.2e} M")
    results["stage5_prodigy_ace2_rbd"] = ace2_complex

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {results_path}")


if __name__ == "__main__":
    main()
