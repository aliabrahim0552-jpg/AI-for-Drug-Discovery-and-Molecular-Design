# Chapter 17 Hands-on: Capstone 2 — De Novo Neutralizing Antibody Design

A real, end-to-end nanobody (single-domain antibody) design and
validation pipeline against a real viral surface antigen — the
SARS-CoV-2 spike receptor-binding domain (RBD) — chaining five real
methods this book has already introduced: real, geometric hotspot
identification on real crystal structures (PDB 6M0J, PDB 7KGJ), a real
ProteinMPNN redesign of a real, experimentally validated nanobody
scaffold (Chapter 10's model), real ESMFold structural self-consistency
checks (Chapter 9's live-API method), and a from-scratch, fully
disclosed reimplementation of PRODIGY for real interface analysis and
binding-affinity prediction. See [`chapter.md`](chapter.md) Sections
17.1-17.2 for full scientific context, every honest feasibility and
substitution decision, and the real, measured results.

## Setup

```bash
pip install -r requirements.txt
```

`torch`, `numpy`, `biopython`, and `requests` are all pip-installable.
No GPU, no RFdiffusion install, and no AlphaFold3 access is required —
see chapter.md Section 17.1's feasibility note (this environment cannot
run RFdiffusion for the same real reasons Chapter 10 §10.1
established, extended here to Bennett et al.'s (2026) antibody-specific
pipeline) and Section 17.2 Stage 5's note (the official PRODIGY tool's
own `freesasa` dependency has no prebuilt wheel for this environment).

## Run

```bash
python antibody_design.py
```

Runs the full real, five-stage pipeline:
1. **Hotspot identification** — real, contact-based epitope mapping on
   both the RBD-ACE2 complex (6M0J) and the RBD-Sb45 nanobody complex
   (7KGJ), independently, then a real, computed overlap statistic
   between them. No epitope residue list is transcribed from either
   paper's text or figures.
2. **Backbone source** — Section 17.1's real feasibility finding rules
   out an RFdiffusion-generated backbone in this environment; the real,
   already-solved Sb45-RBD complex (7KGJ) is used as the fixed backbone
   instead, Chapter 10's own substitution strategy reused here.
3. **Sequence redesign** — real ProteinMPNN (Chapter 10's vendored
   model and checkpoint) redesigns the real Sb45 nanobody sequence at
   three sampling temperatures, RBD held fixed as real structural
   context; recovery is reported overall and split by real,
   geometrically-derived paratope vs. framework positions.
4. **3D validation** — real, live ESMFold structural self-consistency
   check (Ca RMSD) of both the native and redesigned nanobody
   sequences against the real 7KGJ backbone. AlphaFold3 is discussed as
   theory only (chapter.md Section 17.2) — no downloadable weights and
   no free, scriptable bulk-inference API exist for it.
5. **Interface analysis & binding-affinity prediction** — a real,
   from-scratch PRODIGY reimplementation, validated directly against
   the official tool's own published test case (2OOB), then applied to
   the real native Sb45-RBD complex and the real ACE2-RBD complex.

Every real number is also written to
`results/antibody_design_results.json`.

Useful flags:
- `--skip-esmfold` — skip the live ESMFold validation calls (e.g.
  offline; the endpoint is shared and rate-limited and occasionally
  returns a transient HTTP 504 under load — `fold_sequence` retries up
  to 3 times automatically, see chapter.md Section 17.2).
- `--results-path` — where to write the JSON results file.

A full run (default settings) took on the order of a few real minutes
of wall-clock CPU time in this chapter's own authoring environment,
including live ESMFold API calls; the ProteinMPNN redesign stage alone
completes in well under a minute on CPU.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Hotspot identification, ProteinMPNN redesign (the real model against
the real bundled `data/7KGJ.pdb`), the hotspot-recovery arithmetic, and
the PRODIGY reimplementation (validated exactly against the official
tool's own bundled `2oob.pdb` test fixture — 78 real contacts, matching
`github.com/haddocking/prodigy`'s own test suite exactly) are all
tested directly and offline. One test calls the live ESMFold API
directly, matching the pattern established in Chapters 9-10.

## A note on Google Colab

`torch`, `numpy`, and `requests` are preinstalled on Colab's default
runtime; only `biopython` needs `!pip install biopython`. No GPU is
required for any stage — a full ProteinMPNN sampling run over the
121-residue nanobody chain completes in well under a minute on a free
CPU-only Colab instance.
