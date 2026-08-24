# Chapter 8 Hands-on: Zero-Shot Mutation Effect Prediction with ESM-2

Scores every possible single-point mutation at the four saturation-
mutagenized positions of GB1 (the IgG-Fc-binding domain of
Streptococcal protein G) using three sizes of the ESM-2 protein
language model, with **no training on the GB1 data at all**, and
checks the scores against Wu et al.'s (2016) real experimental binding
fitness measurements. See [`chapter.md`](chapter.md) Section 8.4 for
full context, and the chapter as a whole for the protein-language-model
background this project builds on.

## Setup

```bash
# CPU-only PyTorch (no CUDA GPU required)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run

```bash
# Default: download the full 149,361-row GB1 landscape from FLIP, extract
# the wild type + all 76 single mutants, score with 3 ESM-2 model sizes
python esm_variant_effect.py

# Offline, deterministic run against the bundled fixture
python esm_variant_effect.py --use-cached-raw

# A single, smaller model
python esm_variant_effect.py --models facebook/esm2_t6_8M_UR50D
```

Prints, per model: the masked-marginal zero-shot Spearman correlation
against experimental fitness (both for all 76 single mutants and for
the 27-mutant high-read-count subset), and the embedding-cosine-
distance-vs-fitness-deviation Spearman correlation. Writes both to
`results/esm2_gb1_results.json`.

`data/gb1_single_mutants_sample.csv` bundles the complete, real
wild-type + 76-single-mutant population extracted from Wu et al.'s
(2016) published dataset (via the FLIP benchmark's redistribution,
fetched 2026-08-20), so the pipeline runs fully offline and
deterministically with `--use-cached-raw` — this is not a subsample,
it is every single-point variant the original 160,000-variant
combinatorial screen contains. The ESM-2 checkpoints themselves
(8M / 35M / 150M parameters) still download from the Hugging Face Hub
on first use regardless of `--use-cached-raw`; that costs on the order
of a few minutes total on a fresh runtime, then nothing on subsequent
runs.

With all three checkpoints already cached, the full run (3 models x
2 scoring methods x 76 mutants) takes about a minute on a single CPU
core — no GPU is needed.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Data-loading tests run against the real bundled GB1 fixture. Model-
dependent tests run against a tiny, randomly-initialized (untrained)
ESM architecture built from the real ESM-2 tokenizer vocabulary
(`tests/fixtures/esm2_vocab.txt`), so the suite never downloads a
pretrained checkpoint and runs in well under a minute, fully offline.
`download_gb1_dataset`'s network call is mocked with an in-memory zip,
not skipped.

## Google Colab

Colab's default runtime preinstalls `torch` but not `transformers` or
`scipy`; run `!pip install transformers scipy` in the first cell. No
GPU is required for the model sizes used here (8M-150M parameters);
larger ESM-2 checkpoints (650M and up) would benefit from one.
