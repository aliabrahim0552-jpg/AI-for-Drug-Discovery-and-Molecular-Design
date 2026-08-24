"""
Chapter 9 hands-on project: programmatic ESMFold structure prediction and
pLDDT confidence-metric validation.

Extract: two real, full-length protein sequences of contrasting
  structural character, both pulled directly from UniProt:
    - Human ubiquitin (UniProt P0CG47, monomer processed from the
      polyubiquitin-B precursor), a 76-residue single-domain protein
      with an ultra-high-resolution (1.8 A) crystal structure, PDB 1UBQ
      (Vijay-Kumar et al., 1987) -- the "ordered, ground-truth-available"
      case.
    - Human alpha-synuclein (UniProt P37840), a 140-residue protein
      annotated as disordered across its *entire* length by DisProt
      (Aspromonte et al., 2024; entry DP00070) via multiple independent
      experimental methods (NMR, far-UV CD, SDS-PAGE) -- the
      "intrinsically disordered, no single native fold" case.
Predict: ESM-2-based single-sequence structure prediction (Lin et al.,
  2023) with zero MSA search, run via Meta's public ESM Metagenomic
  Atlas inference API -- the real, GPU-hosted esmfold_v1 model, called
  programmatically over HTTPS rather than loaded locally (this chapter's
  feasibility note explains why: the checkpoint is ~8.4 GB, built on a
  3B-parameter ESM-2 backbone, and does not safely fit this project's
  ~16 GB RAM, CPU-only development machine).
Evaluate:
  1. Accuracy vs. a real crystal structure: superimpose the ubiquitin
     prediction onto PDB 1UBQ (BioPython Superimposer, C-alpha atoms,
     matched 1:1 by residue number -- both sequences are identical and
     both structures use residue numbering 1-76), report the global
     C-alpha RMSD, and correlate per-residue pLDDT against per-residue
     post-alignment C-alpha deviation (Spearman): does ESMFold's
     self-reported confidence track real structural accuracy?
  2. Disorder signal: compare per-residue pLDDT between ubiquitin
     (ordered) and alpha-synuclein (disordered) with a Mann-Whitney U
     test -- does the same confidence signal distinguish a protein with
     one native fold from one experimentally shown to have none,
     replicating in miniature the association Wilson et al. (2022)
     report between pLDDT and disorder?

See README.md for usage and chapter.md Section 9.4 for full context,
including the real, measured API timeout envelope that determined which
real sequences this script uses.
"""
import argparse
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import requests
from Bio.PDB import PDBParser, Superimposer
from scipy.stats import mannwhitneyu, spearmanr

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
DEFAULT_TIMEOUT = 60

# Real UniProt sequences, fetched 2026-08-20.
UBIQUITIN_SEQUENCE = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)  # UniProt P0CG47, single 76-aa repeat unit of human polyubiquitin-B
ALPHA_SYNUCLEIN_SEQUENCE = (
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTA"
    "VAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
)  # UniProt P37840, full-length canonical sequence, 140 aa

REFERENCE_PDB_ID = "1UBQ"


@dataclass
class ResidueConfidence:
    residue_number: int
    plddt: float  # mean over that residue's atoms, 0-1 scale (this API's convention)


def fold_sequence(sequence: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """POST `sequence` to the real ESM Metagenomic Atlas ESMFold inference
    API and return the raw PDB text of its prediction. No local model
    weights are downloaded or run -- this is a live network call to
    Meta's hosted esmfold_v1 model."""
    response = requests.post(ESMFOLD_API_URL, data=sequence, timeout=timeout)
    response.raise_for_status()
    pdb_text = response.text
    if not pdb_text.lstrip().startswith("HEADER") and "ATOM" not in pdb_text:
        raise ValueError(f"ESMFold API did not return a PDB structure: {pdb_text[:200]!r}")
    return pdb_text


def per_residue_plddt(pdb_text: str) -> list[ResidueConfidence]:
    """Parse an ESMFold PDB prediction and return each residue's mean
    pLDDT, averaged over that residue's atoms -- this API reports pLDDT
    per *atom* in the B-factor column (verified directly: values differ
    atom-to-atom within a single residue), unlike the single
    per-residue value AlphaFold's own PDB output repeats across all of
    a residue's atoms."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prediction", StringIO(pdb_text))
    chain = structure[0]["A"]
    result = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        atom_plddts = [atom.get_bfactor() for atom in residue]
        result.append(ResidueConfidence(residue_number=residue.id[1], plddt=float(np.mean(atom_plddts))))
    return result


def compute_ca_accuracy(predicted_pdb_text: str, reference_pdb_path: Path, chain_id: str = "A") -> dict:
    """Superimpose the predicted structure's C-alpha atoms onto the real
    reference crystal structure's C-alpha atoms (matched 1:1 by residue
    number) and return the global RMSD plus each residue's post-
    alignment C-alpha deviation (Euclidean distance, in Angstroms, after
    applying the fitted rotation+translation)."""
    parser = PDBParser(QUIET=True)
    predicted = parser.get_structure("predicted", StringIO(predicted_pdb_text))[0][chain_id]
    reference = parser.get_structure("reference", str(reference_pdb_path))[0][chain_id]

    pred_by_resnum = {r.id[1]: r for r in predicted if r.id[0] == " "}
    ref_by_resnum = {r.id[1]: r for r in reference if r.id[0] == " "}
    shared_resnums = sorted(set(pred_by_resnum) & set(ref_by_resnum))
    if not shared_resnums:
        raise ValueError("No overlapping residue numbers between prediction and reference")

    pred_ca = [pred_by_resnum[n]["CA"] for n in shared_resnums]
    ref_ca = [ref_by_resnum[n]["CA"] for n in shared_resnums]

    sup = Superimposer()
    sup.set_atoms(ref_ca, pred_ca)  # fit predicted onto reference
    sup.apply(pred_ca)  # applies the fitted rotran to the (already-referenced) pred_ca atom objects

    per_residue_deviation = {
        resnum: float(np.linalg.norm(pred_atom.coord - ref_atom.coord))
        for resnum, pred_atom, ref_atom in zip(shared_resnums, pred_ca, ref_ca)
    }
    return {
        "n_residues_compared": len(shared_resnums),
        "global_ca_rmsd": float(sup.rms),
        "per_residue_deviation": per_residue_deviation,
    }


def evaluate_confidence_vs_accuracy(predicted_pdb_text: str, reference_pdb_path: Path) -> dict:
    """Spearman correlation between per-residue pLDDT and per-residue
    post-alignment C-alpha deviation from the real crystal structure.
    A negative correlation means higher self-reported confidence really
    does track lower structural error."""
    confidences = per_residue_plddt(predicted_pdb_text)
    accuracy = compute_ca_accuracy(predicted_pdb_text, reference_pdb_path)
    shared = sorted(set(c.residue_number for c in confidences) & set(accuracy["per_residue_deviation"]))
    plddt_by_resnum = {c.residue_number: c.plddt for c in confidences}
    plddt_values = np.array([plddt_by_resnum[n] for n in shared])
    deviation_values = np.array([accuracy["per_residue_deviation"][n] for n in shared])
    rho, pvalue = spearmanr(plddt_values, deviation_values)
    return {
        "n": len(shared),
        "global_ca_rmsd": accuracy["global_ca_rmsd"],
        "spearman_rho_plddt_vs_deviation": float(rho),
        "spearman_pvalue": float(pvalue),
        "mean_plddt": float(plddt_values.mean()),
    }


def evaluate_disorder_signal(ordered_pdb_text: str, disordered_pdb_text: str) -> dict:
    """Mann-Whitney U test comparing per-residue pLDDT between a known-
    ordered protein and a known-disordered one (DisProt-annotated)."""
    ordered_plddt = np.array([c.plddt for c in per_residue_plddt(ordered_pdb_text)])
    disordered_plddt = np.array([c.plddt for c in per_residue_plddt(disordered_pdb_text)])
    statistic, pvalue = mannwhitneyu(ordered_plddt, disordered_plddt, alternative="greater")
    return {
        "n_ordered": len(ordered_plddt),
        "n_disordered": len(disordered_plddt),
        "mean_plddt_ordered": float(ordered_plddt.mean()),
        "mean_plddt_disordered": float(disordered_plddt.mean()),
        "median_plddt_ordered": float(np.median(ordered_plddt)),
        "median_plddt_disordered": float(np.median(disordered_plddt)),
        "mannwhitney_u": float(statistic),
        "mannwhitney_pvalue_greater": float(pvalue),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Real ESMFold structure prediction and pLDDT validation.")
    parser.add_argument("--use-cached-raw", action="store_true", help="Use bundled ESMFold predictions instead of a live API call.")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--results-path", default=str(Path(__file__).parent / "results" / "esmfold_structure_results.json"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    reference_pdb_path = data_dir / f"{REFERENCE_PDB_ID}.pdb"

    if args.use_cached_raw:
        print("Loading cached ESMFold predictions (bundled, real, previously fetched from the live API)")
        ubiquitin_pdb = (data_dir / "ubiquitin_esmfold_prediction.pdb").read_text()
        synuclein_pdb = (data_dir / "alpha_synuclein_esmfold_prediction.pdb").read_text()
    else:
        print(f"Folding ubiquitin ({len(UBIQUITIN_SEQUENCE)} aa) via {ESMFOLD_API_URL} ...")
        ubiquitin_pdb = fold_sequence(UBIQUITIN_SEQUENCE)
        print(f"Folding alpha-synuclein ({len(ALPHA_SYNUCLEIN_SEQUENCE)} aa) via {ESMFOLD_API_URL} ...")
        synuclein_pdb = fold_sequence(ALPHA_SYNUCLEIN_SEQUENCE)

    accuracy_result = evaluate_confidence_vs_accuracy(ubiquitin_pdb, reference_pdb_path)
    print(
        f"\nUbiquitin vs. real crystal structure {REFERENCE_PDB_ID}: "
        f"global C-alpha RMSD = {accuracy_result['global_ca_rmsd']:.3f} A "
        f"(n={accuracy_result['n']} residues)"
    )
    print(
        f"  pLDDT vs. per-residue C-alpha deviation: "
        f"rho={accuracy_result['spearman_rho_plddt_vs_deviation']:.4f} "
        f"(p={accuracy_result['spearman_pvalue']:.2e})"
    )

    disorder_result = evaluate_disorder_signal(ubiquitin_pdb, synuclein_pdb)
    print(
        f"\nOrdered (ubiquitin) mean pLDDT = {disorder_result['mean_plddt_ordered']:.3f}  |  "
        f"Disordered (alpha-synuclein) mean pLDDT = {disorder_result['mean_plddt_disordered']:.3f}"
    )
    print(
        f"  Mann-Whitney U (ordered > disordered): U={disorder_result['mannwhitney_u']:.1f}, "
        f"p={disorder_result['mannwhitney_pvalue_greater']:.2e}"
    )

    results = {"confidence_vs_accuracy": accuracy_result, "disorder_signal": disorder_result}
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {results_path}")


if __name__ == "__main__":
    main()
