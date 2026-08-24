# Chapter 16 Hands-on: Capstone 1 — Small Molecule Oncology Target Discovery

A real, complete, end-to-end small-molecule discovery pipeline for
EGFR (CHEMBL203) — the same real oncology target this book has
followed since Chapter 1 (kinase domain, PDB 1M17, in complex with the
real, clinically approved inhibitor erlotinib). Every stage chains a
real method this book already built and validated in an earlier
chapter: real ChEMBL/PDB data retrieval (Chapters 4, 11), a real MPNN
QSAR model (Chapter 6's architecture) and rule-based ADMET filter
(Chapter 13's Tier 1), a real generative Transformer + REINFORCE
pipeline (Chapter 7, with its reward oracle upgraded to this chapter's
own continuous MPNN regressor), real AutoDock Vina docking and ANI-2x
ligand-alone MD stability checks (Chapters 11-13), and a real,
self-contained automated HTML technical report. See
[`chapter.md`](chapter.md) Section 16.2 for full scientific context,
every honest compute-budget scoping decision, and real, measured
results.

## Setup

```bash
pip install -r requirements.txt
```

`torch`, `torch_geometric`, `rdkit`, `selfies`, `scipy`, `numpy`,
`requests`, `meeko`, `openmm`, `openmmml`, and `torchani` are all
pip-installable. `vina` has a prebuilt wheel on Linux/macOS; on
Windows (this chapter's own authoring environment), install the
official standalone Vina CLI binary and point `VINA_EXECUTABLE` at it
(see Chapter 11's README for the same setup — reused unchanged here):

```bash
export VINA_EXECUTABLE=/path/to/vina.exe   # only needed on platforms with no `vina` wheel
```

## Run

```bash
python oncology_capstone.py
```

Runs the full real, five-stage pipeline:
1. **Data** — live ChEMBL EGFR bioactivity retrieval (cached for
   offline reproducibility) and real 1M17 receptor structure
   validation.
2. **QSAR + ADMET** — trains a real MPNN regressor (scaffold split) on
   real EGFR pIC50 data, and defines the real rule-based drug-likeness
   filter used downstream.
3. **Generative design** — pretrains a real SELFIES Transformer on
   real active EGFR compounds, then RL-fine-tunes it (REINFORCE)
   against a reward built from the Stage 2 QSAR model's own
   predictions plus the ADMET filter.
4. **Docking + MD** — real AutoDock Vina docking (with a real
   redocking validation control) of the pipeline's own top novel,
   ADMET-passing generated candidates against the real 1M17 pocket,
   followed by real, short ANI-2x ligand-alone MD stability checks on
   the top docked candidates.
5. **Report** — a real, self-contained HTML technical report
   (`results/technical_report.html`) assembling every real number and
   real 2D structure image the pipeline produced.

Every real number is also written to `results/capstone_results.json`.

Useful flags:
- `--refresh-cache` — re-fetch the ChEMBL data live instead of using
  the bundled cache.
- `--max-records` — cap on raw ChEMBL records fetched (default 3000,
  Chapter 7's own convention).
- `--pretrain-epochs`, `--rl-iterations`, `--n-sample` — generative
  pipeline settings (defaults match Chapter 7).
- `--n-dock`, `--n-md` — how many top real candidates to carry into
  docking and MD respectively (defaults 8 and 3, Chapter 13's own
  established shortlist scale).
- `--skip-docking`, `--skip-md` — skip the real-but-slow downstream
  verification stages (useful for a fast end-to-end smoke test of
  Stages 1-3, or on a host with no Vina/ANI-2x available).

A full run (default settings) took on the order of 30-40 real minutes
of wall-clock CPU time in this chapter's own authoring environment,
including live data retrieval, QSAR training, and generative
pretraining/RL — real Stage 4 docking+MD time alone measured 10.3
minutes across 7 docked and 3 MD-verified candidates, well under
Chapter 11's own per-compound rate on its larger, more diverse
benchmark set (this chapter's own generated candidates tend to be
smaller, simpler structures, a real, disclosed reason for the faster
real measured rate here, not a different docking protocol).
`chapter.md` Section 16.2 reports the exact real measured numbers.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Data curation, the ADMET filter, scaffold splitting, real receptor/box
geometry (against the bundled real `data/1M17.pdb`), the Kabsch/RMSD
logic, SELFIES vocabulary round-tripping, and the report generator are
all tested directly and offline. The small number of tests that would
otherwise invoke Vina or ANI-2x are skipped automatically when neither
engine is available on the host.

## A note on Google Colab

```bash
!pip install torch_geometric selfies meeko openmmml torchani
```

`torch`, `rdkit`, `scipy`, `numpy`, `requests`, and `openmm` are
preinstalled or trivially available on Colab's default runtime. No GPU
is required for any stage. Vina is not pip-installable inside Colab's
own container image either; download the official Linux Vina binary
(a real, prebuilt release asset) and point `VINA_EXECUTABLE` at it, the
same real workaround this chapter's own Windows environment used.
