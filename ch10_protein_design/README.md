# Chapter 10 Hands-on: Fixed-Backbone Sequence Design with ProteinMPNN

Runs the real, official ProteinMPNN model (Dauparas et al., 2022,
vendored under `third_party/proteinmpnn/`, MIT-licensed) on two real
PDB backbones -- human ubiquitin (PDB 1UBQ) and the MDM2-p53 peptide
interface (PDB 1YCR) -- and validates the results two ways: native
sequence recovery against real ground truth, and (where the real,
public ESMFold API cooperates) structural self-consistency. See
[`chapter.md`](chapter.md) Section 10.4 for full context, including why
this project does not run RFdiffusion locally.

## Setup

```bash
pip install -r requirements.txt
```

No GPU is required. The real ProteinMPNN checkpoint
(`proteinmpnn_weights/v_48_020.pt`, ~6.7 MB) and its real model code
(`third_party/proteinmpnn/`) are both bundled directly in this
repository -- nothing needs to be downloaded at run time.

## Run

```bash
python protein_design.py
```

Runs both real experiments:
1. Whole-chain redesign of ubiquitin's backbone (PDB 1UBQ) at three
   sampling temperatures, reporting real native sequence recovery, then
   attempts real ESMFold validation of both the native sequence
   (expected to succeed quickly, per Chapter 9) and one redesigned
   sequence (see chapter.md for the real, reproducible outcome).
2. Binder-only redesign of the p53 peptide's backbone (PDB 1YCR chain
   B), with MDM2 (chain A) held fixed as real structural context,
   reporting real recovery at the literature-verified
   Phe19/Trp23/Leu26 hot-spot triad (Kussie et al., 1996) vs. the
   rest of the peptide.

Writes all real numbers to `results/protein_design_results.json`.
Network calls (to the ESM Metagenomic Atlas API) are live by default;
pass `--skip-esmfold` to run the ProteinMPNN design steps only, fully
offline.

```bash
python protein_design.py --skip-esmfold
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

ProteinMPNN's real checkpoint is small (~6.7 MB) and fast on CPU
(about 1 second per design call for these backbones), so this suite
runs the real, official model directly against the real bundled PDB
structures -- no synthetic/tiny substitute model is needed. Only the
ESMFold network call is mocked, except one test that calls the live
API directly (matching the pattern established in earlier chapters).

## A note on Google Colab

`torch`, `numpy`, `scipy`, and `requests` are preinstalled on Colab's
default runtime; only `biopython` needs `!pip install biopython`. No
GPU is required for ProteinMPNN inference at this scale.
