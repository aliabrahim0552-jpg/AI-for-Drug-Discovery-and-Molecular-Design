# Chapter 14 Hands-on: A Real siRNA Knockdown-Efficacy Predictor

Builds a real siRNA knockdown-efficacy predictor from real, published
RNAi silencing data -- Huesken et al. (2005), *Design of a genome-wide
siRNA library using an artificial neural network*, the field's original
large-scale siRNA efficacy training set (2,361 real 19-mer siRNAs
profiled by dual-luciferase reporter assay in H1299 cells) -- and real
RNA secondary-structure/duplex thermodynamics computed directly with
ViennaRNA (Lorenz et al., 2011). See [`chapter.md`](chapter.md) Section
14.3 for full scientific context, data provenance, and real, measured
results.

## Setup

```bash
pip install -r requirements.txt
```

`ViennaRNA`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`,
`torch`, and `requests` are all pip-installable, including on Windows
(unlike Chapter 13's AutoDock Vina, ViennaRNA ships a prebuilt
`win_amd64` wheel -- no separate system binary needed).

## Run

```bash
python sirna_efficacy.py
```

Runs the full real pipeline:
1. Loads the real, cached Huesken-subset siRNA/target/efficiency
   records (2,133 train + 228 test sequences, zero sequence overlap),
   fetched live from a community redistribution of the original
   Huesken et al. (2005) dataset (or re-fetches live with
   `--refresh-cache`; see `chapter.md` Section 14.3 for why the
   original hosting is no longer available and exactly what is/isn't
   trusted from the redistribution).
2. Computes real, self-computed features for every sequence: siRNA
   self-structure and target-site accessibility (ViennaRNA MFE
   folding), siRNA-target duplex hybridization energy and 5'/3'
   terminal-stability asymmetry (ViennaRNA `RNA.duplexfold`), and
   simple established sequence-composition heuristics (GC content,
   seed GC content, terminal base identity, internal repeats).
3. Trains a rule-based composite score, a Random Forest, an XGBoost
   model, and a small PyTorch MLP, and evaluates every model on the
   real, sequence-disjoint held-out test split.
4. Writes every real number, including per-model Spearman/Pearson
   correlation, R², RMSE, MAE, and feature importances, to
   `results/sirna_efficacy_results.json`.

Useful flags:
- `--refresh-cache` — re-fetch the siRBench Huesken splits live instead
  of using the bundled cache.
- `--output PATH` — where to write the results JSON (default
  `results/sirna_efficacy_results.json`).

A full run (2,133 training sequences, 228 held-out test sequences,
four models) takes well under a minute on a free-tier Colab CPU
runtime — no GPU, and no docking/MD-scale compute budget, are required
for this chapter's hands-on project.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Data curation (including the malformed/inconsistent-row rejection
logic) and every real feature-computation function — ViennaRNA MFE
folding, duplex thermodynamics, target-site accessibility, and the
sequence-composition heuristics — are tested directly and offline
against small, hand-checkable fixtures. No network access or full
model training is required to run the test suite.

## A note on Google Colab

```bash
!pip install ViennaRNA xgboost
```

`numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, and `requests` are
preinstalled on Colab's default runtime. No GPU is required.
