# Chapter 11 Hands-on: Real AutoDock Vina Docking Against EGFR

Runs the real, official AutoDock Vina engine (Trott & Olson, 2010;
Eberhardt et al., 2021) against the real EGFR kinase domain (PDB 1M17,
in complex with erlotinib) and a real, curated benchmark set of
potency-labeled ChEMBL EGFR ligands. See [`chapter.md`](chapter.md)
Section 11.4 for full scientific context, including why this project
runs physics-based docking rather than DiffDock.

## Setup

```bash
pip install -r requirements.txt
```

`openbabel-wheel` provides the `obabel` command-line tool used for
receptor preparation. `vina` (the AutoDock Vina Python bindings) ships
prebuilt wheels for Linux and macOS but not Windows; on Windows,
install the official standalone executable from
[the AutoDock-Vina releases page](https://github.com/ccsb-scripps/AutoDock-Vina/releases)
and either put it on `PATH` as `vina` or point the `VINA_EXECUTABLE`
environment variable at it. On Google Colab, `pip install vina` alone
is sufficient (see "A note on Google Colab" below).

## Run

```bash
python molecular_docking.py
```

Runs the full real pipeline:
1. Loads the real, curated 40-compound ChEMBL EGFR benchmark set
   (`data/egfr_chembl_benchmark.json`, bundled for offline
   reproducibility; pass `--refresh-cache` to re-fetch and re-curate
   live from ChEMBL instead).
2. Prepares the real EGFR receptor (PDB 1M17) with OpenBabel.
3. Runs the redocking validation control (erlotinib into its own real
   crystallographic pocket).
4. Docks all 40 real compounds under both the focused (pocket-informed)
   and blind (whole-receptor) search-box conditions, in parallel.
5. Computes real quantitative analysis: docking-score-vs-potency
   correlation, focused-vs-blind agreement, and timing statistics.

Writes all real numbers to `results/molecular_docking_results.json`.

```bash
python molecular_docking.py --n-molecules 40 --n-workers 4
```

Useful flags:
- `--n-molecules N` — benchmark set size (default 40).
- `--n-workers N` — parallel docking workers (default 4; each AutoDock
  Vina call itself runs single-threaded, `--cpu 1`, for run-to-run
  determinism — see chapter.md's note on Vina's own multi-threaded
  nondeterminism).
- `--skip-blind` — run only the focused condition (roughly half the
  wall-clock time).
- `--refresh-cache` — re-fetch and re-curate the benchmark set live
  from the ChEMBL REST API instead of using the bundled cache.

A full run (40 compounds, both conditions, 4 workers) takes on the
order of tens of minutes of wall-clock CPU time; see chapter.md
Section 11.4 for the real measured timing this chapter reports.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Receptor/ligand preparation (RDKit, Meeko, OpenBabel) and the ChEMBL
curation logic are tested for real, offline, against the real bundled
`data/1M17.pdb` and small synthetic activity-record fixtures. Tests
that require actually running AutoDock Vina are skipped automatically
when no Vina engine (Python bindings or executable) is available on
the host.

## A note on Google Colab

```bash
!pip install rdkit meeko vina
!apt-get install -y openbabel
```

`numpy`, `scipy`, and `requests` are preinstalled on Colab's default
runtime. `vina` installs cleanly from a prebuilt Linux wheel on Colab
(unlike this chapter's own Windows authoring environment — see
chapter.md's feasibility note), so the Python-bindings code path runs
directly with no executable fallback needed. No GPU is required —
every step in this hands-on project runs on CPU.
