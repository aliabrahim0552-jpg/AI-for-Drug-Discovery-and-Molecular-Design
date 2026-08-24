# Chapter 5 Hands-on: hERG Cardiotoxicity QSAR Classifier

Trains a classifier (XGBoost, Random Forest, or SVM) to predict whether
a small molecule blocks the hERG (KCNH2) cardiac potassium channel from
its Morgan/ECFP4 fingerprint, and evaluates it under both a random
train/test split and a Bemis-Murcko scaffold split, so the two split
strategies can be compared on the same data. See
[`chapter.md`](chapter.md) Section 5.5 for full context, and the chapter
as a whole for the QSAR, ADMET, classical-ML-benchmark, and
bioactivity-bias background this project builds on.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Default: fetch live hERG (CHEMBL240) IC50 data, train XGBoost, evaluate
# on both a random split and a scaffold split
python herg_qsar.py

# A specific model / split / threshold, or SMOTE-oversampled training data
python herg_qsar.py --model random_forest --split scaffold
python herg_qsar.py --model svm --split random
python herg_qsar.py --use-smote --split scaffold

# Offline, deterministic run against the bundled fixture (see below)
python herg_qsar.py --use-cached-raw
```

Prints, for each split evaluated: the train/test size, the fraction of
blockers in each set, and accuracy, balanced accuracy, precision,
recall, F1, and ROC-AUC on the held-out test set.

`data/raw_herg_bioactivities_sample.json` bundles a real extract of 3000
hERG IC50 bioactivity records from ChEMBL (fetched 2026-08-19; cleans to
1765 unique standardized compounds), so the pipeline runs fully offline
and deterministically with `--use-cached-raw`. `--max-records` caps how
many raw records are used from either the live API or this cached
extract - useful for a fast subset run (see Tests, below).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Most tests run fast, offline, synthetic-data checks or a small offline
slice of the bundled fixture; one test calls the live ChEMBL API
directly, as a reproducibility check on that path specifically.

## Google Colab

`rdkit`, `xgboost`, and `imbalanced-learn` are not preinstalled on
Colab's default runtime; `scikit-learn`, `numpy`, and `requests` are.
Run `!pip install rdkit xgboost imbalanced-learn` in the first cell. No
GPU is required - training on ~1400 compounds' 2048-bit fingerprints
completes in well under a minute on CPU for all three model types.
