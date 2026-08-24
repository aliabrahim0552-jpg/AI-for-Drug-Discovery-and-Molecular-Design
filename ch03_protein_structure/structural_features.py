"""
Chapter 3 hands-on project: extracting spatial features and binding
pocket geometry from a real PDB structure.

Computes, for a protein chain:
  - backbone dihedral (phi/psi) angles, and a simplified geometric
    secondary-structure classification from them
  - a CA-CA residue contact map
  - the binding-pocket residues around a bound ligand

Default target: PDB entry 1M17, the EGFR tyrosine kinase domain bound
to erlotinib (ligand code AQ4) - the same structure retrieved and used
as a running example in Chapter 1.

See README.md for usage and chapter.md Section 3.4 for context.
"""
import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np
import requests
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

warnings.simplefilter("ignore", BiopythonWarning)

RCSB_FILES_BASE = "https://files.rcsb.org/download"
DEFAULT_PDB_ID = "1M17"
DEFAULT_CHAIN_ID = "A"
DEFAULT_LIGAND = "AQ4"  # erlotinib, as bound in 1M17
DEFAULT_CONTACT_CUTOFF = 8.0  # angstrom, standard CA-CA contact map cutoff
DEFAULT_POCKET_CUTOFF = 5.0  # angstrom, standard binding-site definition
DEFAULT_TIMEOUT = 30

# Simplified Ramachandran regions for a coarse helix/sheet/coil call from
# (phi, psi) alone. This is a pedagogical approximation, NOT a substitute
# for DSSP (Kabsch & Sander, 1983), which additionally uses backbone
# hydrogen-bonding patterns and is the field's actual standard.
HELIX_PHI_RANGE = (-100.0, -30.0)
HELIX_PSI_RANGE = (-77.0, -5.0)
SHEET_PHI_RANGE = (-180.0, -45.0)
SHEET_PSI_RANGES = [(90.0, 180.0), (-180.0, -170.0)]


def fetch_pdb_structure(pdb_id: str, out_path: Path, timeout: int = DEFAULT_TIMEOUT) -> Path:
    """Download a structure file (legacy .pdb format) from RCSB PDB."""
    resp = requests.get(f"{RCSB_FILES_BASE}/{pdb_id}.pdb", timeout=timeout)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return out_path


def load_chain(pdb_path: Path, pdb_id: str, chain_id: str = DEFAULT_CHAIN_ID):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(pdb_path))
    return structure[0][chain_id]


def get_standard_residues(chain) -> list:
    """Standard amino acid residues only (excludes waters/heteroatoms)."""
    return [r for r in chain if r.id[0] == " "]


def compute_phi_psi(chain) -> list[dict]:
    """
    Backbone (phi, psi) dihedral angles in degrees for every residue in
    the chain with a defined value (chain termini have phi or psi = None).
    """
    ppb = PPBuilder()
    rows = []
    for pp in ppb.build_peptides(chain):
        for residue, (phi, psi) in zip(pp, pp.get_phi_psi_list()):
            rows.append(
                {
                    "resname": residue.get_resname(),
                    "resseq": residue.id[1],
                    "phi": math.degrees(phi) if phi is not None else None,
                    "psi": math.degrees(psi) if psi is not None else None,
                }
            )
    return rows


def _in_range(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def classify_secondary_structure(phi_deg: float | None, psi_deg: float | None) -> str:
    """Coarse helix/sheet/coil call from backbone dihedrals alone (see module docstring)."""
    if phi_deg is None or psi_deg is None:
        return "coil"
    if _in_range(phi_deg, *HELIX_PHI_RANGE) and _in_range(psi_deg, *HELIX_PSI_RANGE):
        return "helix"
    if _in_range(phi_deg, *SHEET_PHI_RANGE) and any(
        _in_range(psi_deg, lo, hi) for lo, hi in SHEET_PSI_RANGES
    ):
        return "sheet"
    return "coil"


def summarize_secondary_structure(phi_psi_rows: list[dict]) -> dict:
    counts = {"helix": 0, "sheet": 0, "coil": 0}
    for row in phi_psi_rows:
        counts[classify_secondary_structure(row["phi"], row["psi"])] += 1
    total = sum(counts.values())
    fractions = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return {"counts": counts, "fractions": fractions, "total": total}


def compute_contact_map(chain, cutoff: float = DEFAULT_CONTACT_CUTOFF):
    """
    CA-CA contact map for standard residues in the chain.

    Returns (residue_ids, contact_map) where residue_ids is a list of
    (resname, resseq) and contact_map is an (N, N) boolean numpy array.
    """
    residues = [r for r in get_standard_residues(chain) if "CA" in r]
    residue_ids = [(r.get_resname(), r.id[1]) for r in residues]
    coords = np.array([r["CA"].coord for r in residues], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    contact_map = dist <= cutoff
    return residue_ids, contact_map


def find_ligand_residue(chain, ligand_resname: str | None = None):
    """
    Find a bound-ligand HETATM residue in the chain, excluding water.

    If ligand_resname is None, returns the non-water heteroresidue with
    the most atoms (a reasonable heuristic for "the ligand of interest"
    vs. ions/buffer components).
    """
    het_residues = [r for r in chain if r.id[0] not in (" ", "W") and r.get_resname() != "HOH"]
    if ligand_resname is not None:
        for r in het_residues:
            if r.get_resname() == ligand_resname:
                return r
        raise ValueError(f"Ligand residue {ligand_resname!r} not found in chain.")
    if not het_residues:
        raise ValueError("No non-water heteroatom residues found in chain.")
    return max(het_residues, key=lambda r: len(list(r.get_atoms())))


def compute_binding_pocket(chain, ligand_resname: str | None = None, cutoff: float = DEFAULT_POCKET_CUTOFF) -> list[dict]:
    """
    Residues with any atom within `cutoff` angstroms of any ligand atom.

    Returns a list of dicts (resname, resseq, min_distance_angstrom),
    sorted by increasing distance to the ligand.
    """
    ligand = find_ligand_residue(chain, ligand_resname)
    ligand_atoms = list(ligand.get_atoms())

    pocket = []
    for residue in get_standard_residues(chain):
        distances = [float(atom - lig_atom) for atom in residue for lig_atom in ligand_atoms]
        min_dist = min(distances) if distances else None
        if min_dist is not None and min_dist <= cutoff:
            pocket.append(
                {
                    "resname": residue.get_resname(),
                    "resseq": residue.id[1],
                    "min_distance_angstrom": round(min_dist, 3),
                }
            )
    pocket.sort(key=lambda row: row["min_distance_angstrom"])
    return pocket


def save_phi_psi_csv(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resname", "resseq", "phi", "psi"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def save_pocket_csv(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resname", "resseq", "min_distance_angstrom"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def save_contact_map_npy(contact_map: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, contact_map)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract dihedral angles, a contact map, and binding-pocket residues from a PDB structure."
    )
    parser.add_argument("--pdb", default=DEFAULT_PDB_ID, help="PDB entry ID (default: 1M17).")
    parser.add_argument("--chain", default=DEFAULT_CHAIN_ID, help="Chain ID to analyze (default: A).")
    parser.add_argument(
        "--ligand", default=DEFAULT_LIGAND, help="Ligand residue name for pocket detection (default: AQ4 / erlotinib)."
    )
    parser.add_argument("--pocket-cutoff", type=float, default=DEFAULT_POCKET_CUTOFF)
    parser.add_argument("--contact-cutoff", type=float, default=DEFAULT_CONTACT_CUTOFF)
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).parent / "data"), help="Directory to read/write structure and output files."
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    pdb_path = out_dir / f"{args.pdb}.pdb"

    if not pdb_path.exists():
        print(f"Fetching {args.pdb} from RCSB PDB...")
        fetch_pdb_structure(args.pdb, pdb_path)
    chain = load_chain(pdb_path, args.pdb, args.chain)

    phi_psi_rows = compute_phi_psi(chain)
    ss_summary = summarize_secondary_structure(phi_psi_rows)
    csv_path = save_phi_psi_csv(phi_psi_rows, out_dir / f"{args.pdb}_phi_psi.csv")
    print(f"Backbone dihedrals for {len(phi_psi_rows)} residues -> {csv_path}")
    print(
        f"  Simplified secondary structure: "
        f"helix={ss_summary['fractions']['helix']:.1%}  "
        f"sheet={ss_summary['fractions']['sheet']:.1%}  "
        f"coil={ss_summary['fractions']['coil']:.1%}"
    )

    residue_ids, contact_map = compute_contact_map(chain, cutoff=args.contact_cutoff)
    npy_path = save_contact_map_npy(contact_map, out_dir / f"{args.pdb}_contact_map.npy")
    mean_contacts = contact_map.sum(axis=1).mean() - 1  # exclude self-contact
    print(
        f"\nContact map ({args.contact_cutoff}A cutoff): {contact_map.shape[0]}x{contact_map.shape[1]} "
        f"-> {npy_path} (mean {mean_contacts:.1f} contacts/residue)"
    )

    pocket_rows = compute_binding_pocket(chain, ligand_resname=args.ligand, cutoff=args.pocket_cutoff)
    pocket_csv_path = save_pocket_csv(pocket_rows, out_dir / f"{args.pdb}_pocket_residues.csv")
    print(
        f"\nBinding pocket around {args.ligand} ({args.pocket_cutoff}A cutoff): "
        f"{len(pocket_rows)} residues -> {pocket_csv_path}"
    )
    for row in pocket_rows[:5]:
        print(f"  {row['resname']}{row['resseq']}: {row['min_distance_angstrom']}A")


if __name__ == "__main__":
    main()
