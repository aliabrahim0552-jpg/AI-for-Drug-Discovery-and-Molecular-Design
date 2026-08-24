# Chapter 13 Hands-on: A Real Tiered HTVS Funnel Against SARS-CoV-2 Mpro

Runs a real, three-tier high-throughput virtual screening funnel against
a real emerging viral target -- SARS-CoV-2 main protease (Mpro/3CLpro,
PDB 5R82, Douangamath et al., 2020) -- over a real, potency-labeled
screening library (37 real distinct compounds, deduplicated from
PubChem AID 1805203's real two-replicate-well assay rows; Han et al.,
2022). Tier 1 is a fast, rule-based ADMET/drug-likeness filter (no
target information used); Tier 2 is real AutoDock Vina docking
(Chapter 11's method); Tier 3 is a real, short ANI-2x ligand-alone MD
stability check (Chapter 12's method) on the top Tier 2 survivors. See
[`chapter.md`](chapter.md) Section 13.3 for full scientific context and
real, measured results.

## Setup

```bash
pip install -r requirements.txt
```

`rdkit`, `meeko`, `numpy`, `scipy`, `requests`, `openmm`, `openmmml`,
and `torchani` are all pip-installable. **AutoDock Vina** additionally
needs, on platforms without a prebuilt `vina` wheel (this chapter's own
Windows authoring environment; see Chapter 11's feasibility note): the
official standalone Vina CLI binary, pointed to via the
`VINA_EXECUTABLE` environment variable, e.g.

```bash
# Windows example, matching how this chapter's own results were produced:
export VINA_EXECUTABLE=/path/to/vina_1.2.7_win.exe
```

**OpenBabel** is required as a system command (`obabel`) for receptor
protonation -- `apt-get install -y openbabel` on Colab/Linux, or
`conda install -c conda-forge openbabel` elsewhere.

## Run

```bash
python htvs_pipeline.py
```

Runs the full real funnel:
1. Loads the real, cached 37-compound PubChem 3CLpro library, already
   deduplicated by real PubChem CID from the assay's own real
   two-replicate-well rows (or re-fetches and re-curates live with
   `--refresh-cache`).
2. Prepares the real 5R82 receptor and runs a real redocking validation
   control (RZS back into its own real crystal pocket).
3. **Tier 1**: real rule-based filtering (Lipinski Ro5, Veber's rules,
   RDKit PAINS alerts, a QED floor) of all 37 compounds.
4. **Tier 2**: real, parallelized AutoDock Vina docking of every Tier 1
   survivor against the real 5R82 pocket.
5. **Tier 3**: real, short (2 ps) ANI-2x ligand-alone MD on the top 8
   Tier 2 survivors by docking affinity, checked for trajectory
   stability.
6. Writes every real number to `results/htvs_results.json`, including a
   retrospective enrichment-factor analysis against PubChem's own real
   Active/Inactive labels (never used by Tier 1 or Tier 2 themselves).

Useful flags:
- `--top-n-tier3 N` — how many Tier 2 survivors advance to Tier 3
  (default 8).
- `--md-steps N` — Tier 3 trajectory length per compound (default
  2,000, i.e. 2 ps at 1 fs/step).
- `--skip-tier3` — stop after Tier 2 (skips the ANI-2x MD stage).
- `--n-workers N` — parallel Vina worker processes (default 4).
- `--refresh-cache` — re-fetch the PubChem library live instead of
  using the bundled cache.

A full run (37 real compounds through Tier 1, 36 through Tier 2, 8
through Tier 3) took real, measured wall-clock time reported in
`chapter.md` Section 13.3 — this chapter's own results took roughly
2.4 CPU-hours of real Vina search time plus about 74 minutes of real
Tier 3 ANI-2x compute, run on a 2-physical-core CPU-only machine;
expect the total to scale with available cores and with how much of
Tier 2's real per-compound cost the resumable cache in `results/`
already covers.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Data curation, Tier 1 filter logic, real receptor/box geometry
(against the bundled `data/5R82.pdb`), the Kabsch/RMSD logic, and the
funnel-analysis arithmetic are tested offline with no network access.
Tests that require actually running Vina or ANI-2x are skipped
automatically when neither engine is available on the host.

## A note on Google Colab

```bash
!apt-get install -y openbabel
!pip install rdkit meeko openmm openmmml torchani vina
```

`numpy`, `scipy`, and `requests` are preinstalled on Colab's default
runtime. On Colab/Linux, `pip install vina` provides prebuilt Python
bindings directly (no `VINA_EXECUTABLE` CLI fallback needed). No GPU is
required for any tier, though a Colab GPU runtime accelerates the Tier
3 ANI-2x evaluations via TorchANI's PyTorch backend.
