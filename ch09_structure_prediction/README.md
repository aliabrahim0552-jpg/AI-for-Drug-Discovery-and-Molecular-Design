# Chapter 9 Hands-on: Programmatic ESMFold Structure Prediction & pLDDT Validation

Folds two real, full-length proteins with ESMFold (Lin et al., 2023) --
human ubiquitin (76 aa, ordered, validated against its real 1.8-A
crystal structure, PDB 1UBQ) and full-length human alpha-synuclein
(140 aa, DisProt-annotated as disordered across its entire length) --
via Meta's public ESM Metagenomic Atlas inference API, and asks two
real questions of the results: does ESMFold's self-reported pLDDT
confidence track real structural accuracy against experiment, and does
it distinguish an ordered protein from a disordered one? See
[`chapter.md`](chapter.md) Section 9.4 for full context, including why
this project calls a hosted API rather than loading `esmfold_v1`
locally.

## Setup

```bash
pip install -r requirements.txt
```

No GPU and no multi-gigabyte model download are required.

## Run

```bash
# Default: live calls to the ESM Metagenomic Atlas API
python esmfold_structure_prediction.py

# Offline, deterministic run against the bundled real predictions
# (verified bit-identical to a fresh live call before bundling)
python esmfold_structure_prediction.py --use-cached-raw
```

Prints: the global C-alpha RMSD between the real ESMFold ubiquitin
prediction and its real 1UBQ crystal structure, the Spearman
correlation between per-residue pLDDT and per-residue structural
deviation, and a Mann-Whitney U comparison of ubiquitin's (ordered)
vs. alpha-synuclein's (disordered) per-residue pLDDT. Writes all of it
to `results/esmfold_structure_results.json`.

`data/ubiquitin_esmfold_prediction.pdb` and
`data/alpha_synuclein_esmfold_prediction.pdb` bundle the real API
responses (fetched 2026-08-20); `data/1UBQ.pdb` bundles the real RCSB
crystal structure. Both API calls typically complete in 1-3 seconds
each with `--use-cached-raw` omitted.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

All tests run against the real bundled fixtures above except one,
which calls the live API directly (matching the pattern established in
earlier chapters' hands-on projects).

## A note on Google Colab

`requests`, `numpy`, and `scipy` are preinstalled on Colab's default
runtime; only `biopython` needs `!pip install biopython`. No GPU is
needed -- this project never runs a model locally.
