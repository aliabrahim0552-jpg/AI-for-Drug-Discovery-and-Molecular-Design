# Chapter 1 Hands-on: Retrieving Chemical & Structural Data

Programmatically retrieves chemical bioactivity data from ChEMBL and a 3D
structure from the Protein Data Bank (PDB) for a real oncology target —
EGFR by default. See [`chapter.md`](chapter.md) Section 1.4 for context,
and the chapter as a whole for the drug discovery pipeline background and
citations this project builds on.

## Setup

```bash
pip install -r requirements.txt
```

Requires internet access (both APIs are public and require no API key).

## Run

```bash
# Default: EGFR (CHEMBL203) bioactivities + PDB structure 1M17
python fetch_data.py

# A different target/structure
python fetch_data.py --target CHEMBL203 --pdb 1M17 --limit 50
```

This prints a summary to stdout and writes, under `data/` (git-ignored —
this is fetched, not source, data):

- `<target>_bioactivities.csv` — measured bioactivities (compound SMILES,
  assay type, value, units) for the target, from ChEMBL.
- `<pdb_id>.pdb` — the structure file, from the PDB.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests call the live ChEMBL and RCSB PDB APIs (no mocking) — they are
themselves a reproducibility check that the retrieval pipeline still
works against the real services. They require network access and will
fail if either service is unreachable.

## Google Colab

This project needs only the `requests` library, which ships preinstalled
on Colab's default runtime. Paste `fetch_data.py`'s contents into a cell
(or `!git clone` this repo) and run — no GPU, no extra installs.
