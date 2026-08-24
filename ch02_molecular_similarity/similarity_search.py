"""
Chapter 2 hands-on project: molecular similarity search + Lipinski filtering.

Given a query molecule (SMILES), rank a small compound library by Tanimoto
similarity of their Morgan (ECFP4) fingerprints, and flag which candidates
pass Lipinski's Rule of Five.
"""
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass
class Candidate:
    name: str
    smiles: str
    similarity: float
    molecular_weight: float
    logp: float
    h_bond_donors: int
    h_bond_acceptors: int
    passes_lipinski: bool


def load_library(csv_path: Path) -> list[tuple[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(row["name"], row["smiles"]) for row in reader]


def lipinski_pass(mol: Chem.Mol) -> tuple[bool, float, float, int, int]:
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return violations <= 1, mw, logp, hbd, hba


def rank_by_similarity(query_smiles: str, library: list[tuple[str, str]]) -> list[Candidate]:
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        raise ValueError(f"Could not parse query SMILES: {query_smiles!r}")
    query_fp = FP_GENERATOR.GetFingerprint(query_mol)

    results = []
    for name, smiles in library:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"warning: skipping unparsable entry {name!r}", file=sys.stderr)
            continue
        fp = FP_GENERATOR.GetFingerprint(mol)
        similarity = TanimotoSimilarity(query_fp, fp)
        passes, mw, logp, hbd, hba = lipinski_pass(mol)
        results.append(Candidate(name, smiles, similarity, mw, logp, hbd, hba, passes))

    results.sort(key=lambda c: c.similarity, reverse=True)
    return results


def print_table(candidates: list[Candidate], top_n: int) -> None:
    header = f"{'Name':<16}{'Similarity':>11}{'MW':>9}{'LogP':>8}{'HBD':>5}{'HBA':>5}  Lipinski"
    print(header)
    print("-" * len(header))
    for c in candidates[:top_n]:
        flag = "PASS" if c.passes_lipinski else "FAIL"
        print(
            f"{c.name:<16}{c.similarity:>11.3f}{c.molecular_weight:>9.1f}"
            f"{c.logp:>8.2f}{c.h_bond_donors:>5}{c.h_bond_acceptors:>5}  {flag}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Molecular similarity search with Lipinski filtering.")
    parser.add_argument(
        "--query",
        default="CC(=O)OC1=CC=CC=C1C(=O)O",
        help="Query molecule as a SMILES string (default: Aspirin).",
    )
    parser.add_argument(
        "--library",
        default=str(Path(__file__).parent / "molecules.csv"),
        help="Path to a CSV file with 'name' and 'smiles' columns.",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of ranked results to show.")
    args = parser.parse_args()

    library = load_library(Path(args.library))
    candidates = rank_by_similarity(args.query, library)
    print(f"Query: {args.query}\n")
    print_table(candidates, args.top)


if __name__ == "__main__":
    main()
