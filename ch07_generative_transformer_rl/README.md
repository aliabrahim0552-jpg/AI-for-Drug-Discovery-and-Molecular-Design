# Chapter 7 Hands-on: Generative Transformer + RL for De Novo EGFR Inhibitor Design

Pretrains a small decoder-only Transformer on SELFIES token sequences of
real, ChEMBL-active EGFR (CHEMBL203) inhibitors, then fine-tunes it with
REINFORCE against a reward combining a trained EGFR-activity classifier
and Lipinski drug-likeness — biasing generation toward novel,
higher-scoring molecules without ever leaving valid chemical space
(SELFIES guarantees that). See [`chapter.md`](chapter.md) Section 7.6
for full context, and the chapter as a whole for the inverse-design,
generative-model, and RL background this project builds on.

## Setup

```bash
# CPU-only PyTorch (no CUDA GPU required)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run

```bash
# Default: fetch live EGFR bioactivity data, pretrain, RL fine-tune, report
python gen_transformer.py

# Offline, deterministic run against the bundled fixture (see below)
python gen_transformer.py --use-cached-raw

# Shorter/longer runs
python gen_transformer.py --use-cached-raw --pretrain-epochs 5 --rl-iterations 5 --n-sample 50
```

Prints: the cleaned dataset's active/inactive balance, the reward
oracle's scaffold-split accuracy/ROC-AUC, the pretraining loss curve's
final value, and — both before and after RL fine-tuning — the fraction
of sampled molecules that are chemically valid, unique, predicted
EGFR-active, and Lipinski-compliant.

`data/raw_egfr_bioactivities_sample.json` bundles a real extract of 3000
EGFR IC50 bioactivity records from ChEMBL (fetched 2026-08-20; cleans to
1508 unique standardized compounds, 860 active / 648 inactive at the
project's 1 uM threshold), so the pipeline runs fully offline and
deterministically with `--use-cached-raw`. `--max-records` caps how many
raw records are used from either the live API or this cached extract.

The full default run (20 pretraining epochs + 25 REINFORCE iterations)
takes several minutes on a single CPU core, dominated by structure
standardization and the RL loop's un-cached autoregressive sampling
(see chapter.md Section 7.6 for why the latter is a deliberate,
disclosed simplification rather than an oversight).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Most individual tests are fast, offline, synthetic-molecule checks
(tiny models, 2-3 epochs/iterations); the full suite takes about 3
minutes, dominated by the handful of tests (and the one CLI subprocess
check) that standardize a real slice of the bundled fixture. One test
calls the live ChEMBL API directly, as a reproducibility check on that
path specifically.

## Google Colab

Colab's default runtime preinstalls `torch` but not `rdkit`, `xgboost`,
or `selfies`; run `!pip install rdkit xgboost selfies` in the first
cell. No GPU is required, though one will be used automatically if
selected.
