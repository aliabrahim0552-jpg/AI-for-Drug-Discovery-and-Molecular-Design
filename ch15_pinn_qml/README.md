# Chapter 15 Hands-on: Physics-Informed & Quantum Machine Learning

Two real, independent, small-scale demonstrations, one per chapter
section. Part 1 (Section 15.1): a physics-informed neural network
(PINN) that solves a real one-compartment oral pharmacokinetic (PK)
ODE system from eight sparse, noisy plasma-concentration samples,
compared against a plain (physics-free) regression baseline and
against the real closed-form analytical solution (the Bateman
equation). Part 2 (Section 15.2): a real hybrid quantum-classical
Variational Quantum Eigensolver (VQE) that computes the real
ground-state electronic energy of H2 and (active-space-reduced) LiH,
validated against real exact diagonalization of the identical qubit
Hamiltonian. See [`chapter.md`](chapter.md) Sections 15.1-15.2 for full
scientific context and real, measured results.

## Setup

```bash
pip install -r requirements.txt
```

`torch`, `numpy`, and `pennylane` are all pip-installable everywhere,
including on Windows. PennyLane's own differentiable Hartree-Fock
backend builds every molecular Hamiltonian in this chapter -- no
external quantum-chemistry package (PySCF, Psi4) is required (PySCF
has no prebuilt Windows wheel; see `requirements.txt`'s comment and
`chapter.md` Section 15.2 for the full feasibility note).

## Run

```bash
python pinn_qml.py
```

Runs both real experiments:
1. **PINN vs. baseline (Section 15.1)**: generates the real sparse,
   noisy synthetic PK observations, trains a plain regression network
   and a physics-informed network (same architecture, same data — the
   PINN additionally minimizes the real ODE residual at unlabeled
   collocation points plus a real dose-conservation initial
   condition), and reports RMSE against the real analytical solution
   over the full 24-hour time course and in the sparse/high-curvature
   early-absorption window specifically.
2. **VQE (Section 15.2)**: builds the real H2 and active-space-reduced
   LiH qubit Hamiltonians, runs a real singles-and-doubles-excitation
   VQE on PennyLane's `default.qubit` simulator, and reports the
   converged energy against real exact diagonalization of the same
   Hamiltonian.

Writes every real number to `results/pinn_qml_results.json`.

Useful flags:
- `--skip-pinn` — run only the QML experiment (fast, ~2 minutes).
- `--skip-qml` — run only the PINN experiment (~2 minutes on CPU).
- `--output PATH` — where to write the results JSON.

A full run (both experiments, default settings) takes on the order of
2-3 minutes of CPU wall-clock time on a free-tier Colab instance — no
GPU required for either experiment.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The PINN/baseline training functions are tested with a small iteration
count (correctness and shape checks, not convergence quality — the
full run's convergence numbers are in `chapter.md`); the QML functions
run real, fast PennyLane simulations (H2's full VQE takes ~2 seconds)
and are checked against a real, independently computed exact-diagonalization
reference and the variational principle's own real inequality.

## A note on Google Colab

```bash
!pip install pennylane
```

`torch` and `numpy` are preinstalled on Colab's default runtime. No
GPU is required for either experiment — both run entirely on CPU in
well under Colab's free-tier session limits.
