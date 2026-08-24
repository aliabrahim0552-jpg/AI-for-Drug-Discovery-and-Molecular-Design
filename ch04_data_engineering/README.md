# Chapter 4 Hands-on: Automated ChEMBL ETL Pipeline

Extracts, cleans, and formats a bioactivity dataset from ChEMBL for a
real target — EGFR by default, continuing the running example from
Chapters 1 and 3. See [`chapter.md`](chapter.md) Section 4.5 for full
context, and the chapter as a whole for the RDKit/DeepChem/BioPython/
database background this pipeline builds on.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Default: live-fetch up to 500 EGFR (CHEMBL203) bioactivity records
python etl_pipeline.py

# A different target, or record cap
python etl_pipeline.py --target CHEMBL203 --max-records 200

# If ChEMBL is unavailable, use the bundled real 200-record fixture instead
python etl_pipeline.py --use-cached-raw
```

Prints a summary to stdout and writes, under `data/` (git-ignored beyond
the bundled `raw_egfr_bioactivities_sample.json` fixture):

- `raw_<target>_bioactivities.json` — the raw extract (cached for reuse).
- `<target>_clean.csv` — one row per (standardized compound, assay
  type): SMILES, potency (nM), number of measurements aggregated, and
  Lipinski descriptors.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Most tests use synthetic records or the bundled real fixture (offline,
deterministic). One test exercises the live ChEMBL extraction path
directly and requires network access — if it fails due to a ChEMBL
outage rather than a real bug, `--use-cached-raw` above shows the
pipeline still works end-to-end offline.

## Google Colab

Dependencies (`rdkit`, `requests`) match earlier chapters; install with
`!pip install rdkit` if needed. No GPU required.
