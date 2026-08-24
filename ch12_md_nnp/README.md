# Chapter 12 Hands-on: Real ANI-2x Molecular Dynamics of EGFR-Erlotinib

Runs real Langevin dynamics under the real, official ANI-2x neural
network potential (Devereux et al., 2020, via TorchANI and OpenMM-ML)
on the real EGFR-erlotinib complex (PDB 1M17, reused from
`ch11_molecular_docking/`). See [`chapter.md`](chapter.md) Section 12.4
for full scientific context, including the real, measured feasibility
investigation behind this project's scope (a real ligand-scale
trajectory rather than a literal 10 ns of the full solvated complex).

## Setup

```bash
pip install -r requirements.txt
```

No GPU is required. ANI-2x's real pretrained weights are downloaded
and cached automatically (by `torchani`) the first time the model is
loaded — a one-time network call, not required on subsequent runs.

## Run

```bash
python md_nnp_simulation.py
```

Runs the full real pipeline:
1. Repairs (PDBFixer) and protonates (OpenMM Modeller, pH 7.4) the
   real EGFR structure.
2. Runs a real classical Amber14 (`ff14SB`) baseline on the protein
   alone (500 steps) for a real speed comparison.
3. Runs a real, short (150-step) ANI-2x demonstration on the complete
   real 5,081-atom complex — a correctness/feasibility check, not a
   converged trajectory.
4. Runs the real, primary 20 ps ANI-2x production trajectory on the
   real erlotinib ligand alone, and computes real RMSD/RMSF from it.

Writes all real numbers to `results/md_nnp_results.json`.

Useful flags:
- `--ligand-steps N` — production trajectory length (default 20,000,
  i.e. 20 ps at 1 fs/step).
- `--complex-demo-steps N` — full-complex demonstration length
  (default 150).
- `--skip-complex-demo` / `--skip-classical` — skip either real
  comparison run to save time.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Structure repair, protonation, ligand building, and the Kabsch
alignment/RMSD/RMSF logic are tested for real, offline, against the
real bundled `data/1M17.pdb` and synthetic coordinate arrays with a
known correct answer. Tests that require actually running ANI-2x
dynamics are skipped automatically when `openmmml`/`torchani` are not
importable on the host.

## A note on Google Colab

```bash
!pip install openmm openmmml torchani pdbfixer rdkit
```

`numpy` is preinstalled on Colab's default runtime. A Colab GPU
runtime lets the classical baseline use OpenMM's `CUDA` platform
automatically, and lets TorchANI's own PyTorch backend use the GPU for
the ANI-2x evaluations too — both real speedups this chapter's own
CPU-only authoring environment could not exercise directly (see
chapter.md Section 12.2).
