# Chapter 3 Hands-on: Structural Feature Extraction & Binding Pocket Geometry

Computes backbone dihedral angles, a residue contact map, and
binding-pocket residues directly from a real PDB structure — the EGFR
tyrosine kinase domain bound to erlotinib (PDB `1M17`), the same
structure retrieved in Chapter 1. See [`chapter.md`](chapter.md) Section
3.4 for full context, and the chapter as a whole for the sequence- and
structure-representation background.

## Setup

```bash
pip install -r requirements.txt
```

`data/1M17.pdb` is bundled in this repository, so no network access is
required to run the default example.

## Run

```bash
# Default: EGFR (1M17), chain A, erlotinib (AQ4) pocket
python structural_features.py

# A different structure (fetched automatically if not already in --out-dir)
python structural_features.py --pdb 1M17 --chain A --ligand AQ4 --pocket-cutoff 5.0
```

Prints a summary to stdout and writes, under `data/` (git-ignored beyond
the bundled `1M17.pdb` fixture):

- `<pdb>_phi_psi.csv` — backbone dihedral angles and a coarse
  helix/sheet/coil call per residue.
- `<pdb>_contact_map.npy` — the Cα–Cα contact map (load with
  `numpy.load`).
- `<pdb>_pocket_residues.csv` — binding-pocket residues ranked by
  distance to the ligand.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Most tests run against the bundled `data/1M17.pdb` fixture (offline,
deterministic). One test exercises the live RCSB PDB download path
directly and requires network access.

## Google Colab

Only `biopython` needs installing (`numpy` and `requests` are
preinstalled): `!pip install biopython` in the first cell. No GPU
required.
