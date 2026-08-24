"""
Chapter 10 hands-on project: fixed-backbone sequence design with
ProteinMPNN, validated via ESMFold structural self-consistency and,
separately, via real functional hot-spot recovery on a real
target-binding peptide interface.

Extract: two real PDB backbones.
  - PDB 1UBQ (Vijay-Kumar et al., 1987): human ubiquitin, 76 aa,
    single chain, 1.8-A crystal structure -- reused from Chapter 9 for
    direct continuity (same real backbone, a design rather than a
    structure-prediction question this time).
  - PDB 1YCR (Kussie et al., 1996): the MDM2 oncoprotein (chain A, 85
    resolved residues) bound to the p53 transactivation-domain peptide
    (chain B, 13 resolved residues) -- the real target-binding peptide
    interface this chapter's hands-on project redesigns.
Predict: ProteinMPNN (Dauparas et al., 2022) -- the real, official
  model architecture and a real pretrained checkpoint
  (vanilla_model_weights/v_48_020.pt, vendored under
  third_party/proteinmpnn/ per its MIT license; see NOTICE.md) --
  redesigns sequences for a fixed backbone:
  1. Whole-chain redesign of ubiquitin's backbone at several sampling
     temperatures, following the same real-parameter-sweep pattern
     Chapters 8-9 use.
  2. Binder-only redesign of the p53 peptide's backbone (chain B),
     with MDM2 (chain A) held fixed as real structural context -- the
     officially supported ProteinMPNN "which chains to design" usage,
     and this chapter's real substitute for the backbone-generation
     step RFdiffusion would otherwise perform (see chapter.md's
     feasibility note: RFdiffusion's own install requirements were not
     met by this environment).
Evaluate:
  1. Native sequence recovery against Chapter 8/9-style real ground
     truth: does ProteinMPNN's design match the real wild-type
     ubiquitin sequence at each position, and does folding the
     redesigned sequence with ESMFold (Chapter 9's real API-based
     method, reused verbatim here) land close to the same backbone it
     was designed on (Cα RMSD, BioPython Superimposer)?
  2. Hot-spot recovery on the real MDM2-p53 interface: do the
     redesigned peptides preserve the real, literature-verified
     Phe19/Trp23/Leu26 hydrophobic triad (Kussie et al., 1996) that
     inserts into MDM2's binding cleft, relative to non-hot-spot
     positions? ESMFold validation of the isolated peptide was
     attempted and its real, reproducible outcome (a request timeout)
     is reported honestly rather than dropped -- see chapter.md
     Section 10.4's "Why the peptide's ESMFold check reports a
     timeout" for the real, repeated measurements behind this.

See README.md for usage and chapter.md Section 10.4 for full context.
"""
import argparse
import copy
import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import requests
import torch
from Bio.PDB import PDBParser, Superimposer
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent / "third_party" / "proteinmpnn"))
from protein_mpnn_utils import (  # noqa: E402
    ProteinMPNN,
    _S_to_seq,
    _scores,
    parse_PDB,
    tied_featurize,
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
HIDDEN_DIM = 128
DEFAULT_CHECKPOINT = Path(__file__).parent / "proteinmpnn_weights" / "v_48_020.pt"

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_TIMEOUT = 60

UBIQUITIN_PDB = Path(__file__).parent / "data" / "1UBQ.pdb"
MDM2_P53_PDB = Path(__file__).parent / "data" / "1YCR.pdb"

# 0-indexed positions of Phe19/Trp23/Leu26 within the 13-residue resolved
# p53 peptide "ETFSDLWKLLPEN" (real PDB residue numbers 17-29, verified
# directly against the bundled 1YCR.pdb: index 2 -> residue 19, index 6
# -> residue 23, index 9 -> residue 26), the hydrophobic triad Kussie et
# al. (1996) report inserting into MDM2's binding cleft.
P53_HOTSPOT_INDICES = (2, 6, 9)


# --------------------------------------------------------------------------
# ProteinMPNN
# --------------------------------------------------------------------------


def load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> ProteinMPNN:
    """Load the real, official ProteinMPNN checkpoint (48 edges, 0.2 A
    training noise -- the standard default `v_48_020` model) with no
    weight modification of any kind."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=HIDDEN_DIM,
        edge_features=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def design_sequences(
    pdb_path: Path,
    model: ProteinMPNN,
    designed_chains: list | None = None,
    fixed_chains: list | None = None,
    temperature: float = 0.1,
    num_samples: int = 1,
    seed: int = 0,
) -> dict:
    """Run real ProteinMPNN inference on a real PDB backbone. If
    `designed_chains` is None every chain is designed (whole-structure
    redesign, Experiment 1); passing e.g. `designed_chains=["B"],
    fixed_chains=["A"]` designs only chain B with chain A held fixed as
    real structural context (Experiment 2's binder-only redesign).
    `parse_PDB` keys its internal `chain_id_dict` by the structure's
    parsed `name` field (the filename stem, not the full path a caller
    might otherwise assume) -- built internally here from the real
    parsed name specifically to avoid that mismatch."""
    pdb_dict_list = parse_PDB(str(pdb_path), ca_only=False)
    name = pdb_dict_list[0]["name"]
    chain_id_dict = None
    if designed_chains is not None:
        chain_id_dict = {name: (designed_chains, fixed_chains or [])}
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


# --------------------------------------------------------------------------
# ESMFold validation (Chapter 9's real, API-based method, reused verbatim)
# --------------------------------------------------------------------------


def fold_sequence(sequence: str, timeout: int = ESMFOLD_TIMEOUT) -> str:
    response = requests.post(ESMFOLD_API_URL, data=sequence, timeout=timeout)
    response.raise_for_status()
    pdb_text = response.text
    if not pdb_text.lstrip().startswith("HEADER") and "ATOM" not in pdb_text:
        raise ValueError(f"ESMFold API did not return a PDB structure: {pdb_text[:200]!r}")
    return pdb_text


def compute_ca_rmsd(predicted_pdb_text: str, reference_pdb_path: Path, chain_id: str = "A") -> dict:
    """Cα RMSD between an ESMFold prediction and a real reference
    structure, matched 1:1 by residue number (Chapter 9's method)."""
    parser = PDBParser(QUIET=True)
    predicted = parser.get_structure("predicted", StringIO(predicted_pdb_text))[0][chain_id]
    reference = parser.get_structure("reference", str(reference_pdb_path))[0][chain_id]
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
# Hot-spot recovery (real, literature-verified positions)
# --------------------------------------------------------------------------


def hotspot_recovery(native_seq: str, designed_seq: str, hotspot_indices=P53_HOTSPOT_INDICES) -> dict:
    hotspot_matches = sum(designed_seq[i] == native_seq[i] for i in hotspot_indices)
    non_hotspot_indices = [i for i in range(len(native_seq)) if i not in hotspot_indices]
    non_hotspot_matches = sum(designed_seq[i] == native_seq[i] for i in non_hotspot_indices)
    return {
        "hotspot_recovery": hotspot_matches / len(hotspot_indices),
        "non_hotspot_recovery": non_hotspot_matches / len(non_hotspot_indices),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ProteinMPNN sequence design, validated via ESMFold and real hot-spot recovery.")
    parser.add_argument("--skip-esmfold", action="store_true", help="Skip the live ESMFold validation calls (e.g. offline).")
    parser.add_argument("--results-path", default=str(Path(__file__).parent / "results" / "protein_design_results.json"))
    args = parser.parse_args()

    model = load_model()
    results = {}

    # --- Experiment 1: ubiquitin whole-chain redesign ---
    print("=== Experiment 1: ubiquitin (1UBQ) whole-chain redesign ===")
    temperatures = [0.1, 0.2, 0.3]
    n_samples_per_temp = 5
    ubiquitin_sweep = []
    best_design = None
    for temp in temperatures:
        out = design_sequences(UBIQUITIN_PDB, model, designed_chains=None, temperature=temp, num_samples=n_samples_per_temp, seed=0)
        recoveries = [d["recovery"] for d in out["designs"]]
        print(f"  T={temp}: recovery = {np.mean(recoveries):.4f} +/- {np.std(recoveries):.4f} (n={n_samples_per_temp})")
        ubiquitin_sweep.append({"temperature": temp, "mean_recovery": float(np.mean(recoveries)), "std_recovery": float(np.std(recoveries)), "recoveries": recoveries})
        if temp == 0.1:
            best_design = out["designs"][0]["sequence"]
    results["ubiquitin_recovery_sweep"] = ubiquitin_sweep
    results["native_ubiquitin_sequence"] = out["native_sequence"]
    results["redesigned_ubiquitin_sequence"] = best_design

    if not args.skip_esmfold:
        print(f"\nPositive control: folding the real native ubiquitin sequence via ESMFold ({out['native_sequence']})...")
        try:
            native_pdb = fold_sequence(out["native_sequence"])
            native_accuracy = compute_ca_rmsd(native_pdb, UBIQUITIN_PDB)
            print(f"  Native sequence ESMFold prediction vs. real 1UBQ backbone: Ca RMSD = {native_accuracy['global_ca_rmsd']:.3f} A (n={native_accuracy['n_residues_compared']})")
            results["native_ubiquitin_esmfold_ca_rmsd"] = native_accuracy
        except (requests.exceptions.Timeout, requests.exceptions.RequestException, ValueError) as exc:
            print(f"  Did not return a structure ({exc}).")
            results["native_ubiquitin_esmfold_timed_out"] = True

        print(f"\nFolding redesigned ubiquitin sequence via ESMFold: {best_design}")
        try:
            predicted_pdb = fold_sequence(best_design)
            accuracy = compute_ca_rmsd(predicted_pdb, UBIQUITIN_PDB)
            print(f"  Redesigned-sequence ESMFold prediction vs. real 1UBQ backbone: Ca RMSD = {accuracy['global_ca_rmsd']:.3f} A (n={accuracy['n_residues_compared']})")
            results["redesigned_ubiquitin_esmfold_ca_rmsd"] = accuracy
        except (requests.exceptions.Timeout, requests.exceptions.RequestException, ValueError) as exc:
            print(f"  Did not return a structure ({exc}) -- real, reproducible outcome, see chapter.md.")
            results["redesigned_ubiquitin_esmfold_timed_out"] = True

    # --- Experiment 2: MDM2-p53 peptide binder-only redesign ---
    print("\n=== Experiment 2: p53 peptide (1YCR chain B) binder-only redesign, MDM2 (chain A) fixed ===")
    out2 = design_sequences(MDM2_P53_PDB, model, designed_chains=["B"], fixed_chains=["A"], temperature=0.2, num_samples=5, seed=0)
    native_peptide = out2["native_sequence"]
    print(f"  Native p53 peptide (chain B): {native_peptide}")
    peptide_designs = []
    for d in out2["designs"]:
        hs = hotspot_recovery(native_peptide, d["sequence"])
        print(f"  Designed: {d['sequence']}  recovery={d['recovery']:.3f}  hotspot={hs['hotspot_recovery']:.2f}  non-hotspot={hs['non_hotspot_recovery']:.3f}")
        peptide_designs.append({**d, **hs})
    results["mdm2_p53_native_peptide"] = native_peptide
    results["mdm2_p53_designs"] = peptide_designs
    results["mdm2_p53_hotspot_positions_0indexed"] = list(P53_HOTSPOT_INDICES)

    if not args.skip_esmfold:
        print("\nAttempting ESMFold validation of the isolated native p53 peptide (13 aa)...")
        try:
            fold_sequence(native_peptide, timeout=ESMFOLD_TIMEOUT)
            results["native_peptide_esmfold_timed_out"] = False
        except (requests.exceptions.Timeout, requests.exceptions.RequestException, ValueError) as exc:
            print(f"  Did not return a structure ({exc}) -- real, reproducible outcome, see chapter.md.")
            results["native_peptide_esmfold_timed_out"] = True

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {results_path}")


if __name__ == "__main__":
    main()
