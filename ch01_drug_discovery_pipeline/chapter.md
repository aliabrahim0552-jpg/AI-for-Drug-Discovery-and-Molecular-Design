# Chapter 1: The Drug Discovery Pipeline & AI Revolution

## 1.1 The Traditional Drug Discovery Pipeline

Bringing a new medicine to market is a multi-stage filtering process. Each
stage exists to kill bad candidates as cheaply as possible, before they
reach a stage where failure is expensive. The stages, in order, are
broadly:

1. **Target identification & validation** — establishing that modulating
   a specific biomolecule (usually a protein) plausibly changes the course
   of a disease, using genetic, biochemical, or clinical evidence.
2. **Hit identification** — screening large compound libraries (physically,
   via high-throughput screening, or virtually, via computational
   docking/similarity search) to find molecules that bind the target.
3. **Hit-to-lead & lead optimization** — iteratively modifying hit
   compounds to improve potency, selectivity, and drug-like properties
   (solubility, metabolic stability, low toxicity) while tracking
   structure–activity relationships (SAR).
4. **Preclinical development** — in vitro and animal studies of efficacy,
   pharmacokinetics (absorption, distribution, metabolism, excretion —
   ADME), and toxicology, required before human dosing.
5. **Clinical trials** — Phase I (safety, small healthy/patient cohorts),
   Phase II (efficacy signal, dose-finding), and Phase III (large,
   randomized, confirmatory efficacy and safety) trials in human subjects.
6. **Regulatory review and approval** — submission of the accumulated
   evidence (e.g., a New Drug Application to the US FDA) for marketing
   authorization.

A candidate can fail at any stage, and the cost of failure grows by orders
of magnitude the later it happens — a compound that fails in early
screening costs a lab reagents and a few days; a compound that fails in
Phase III has consumed years and hundreds of millions of dollars enrolling
and monitoring patients. This asymmetry is the central economic argument
for pushing predictive power as early into the pipeline as possible, which
is exactly where the computational and AI methods covered in this book are
concentrated.

### The economics of attrition: Eroom's Law

Despite decades of technological improvement — combinatorial chemistry,
high-throughput screening, genomics, and now machine learning — the
*inflation-adjusted* number of new drugs approved per billion US dollars
of R&D spending has been in long-term decline. Scannell et al. (2012)
documented that this figure roughly halved every nine years from 1950 to
the early 2010s, an approximately 80-fold fall in R&D efficiency over that
period, in direct contrast to the exponential improvement in computing
power described by Moore's Law. They named this trend **Eroom's Law**
("Moore's Law" spelled backwards) to emphasize the contrast. If $N(t)$ is
the number of new drugs approved per billion inflation-adjusted dollars of
R&D spend at year $t$, the observed trend is approximately:

$$
N(t) \approx N_0 \cdot 2^{-\frac{t - t_0}{T}}, \qquad T \approx 9 \text{ years}
$$

where $T$ is the empirical halving period. Scannell et al. attribute this
decline to four compounding factors rather than a single cause: the
"better than the Beatles" problem (new drugs must beat an ever-improving
standard of care, not merely a placebo), increasingly risk-averse
regulatory and reimbursement environments, the tendency to increase R&D
spending as a response to past failure rather than as a cause of future
success, and a "basic research–brute force" bias toward molecular-target
reductionism at the expense of systems-level understanding of disease.

### The cost and risk of a single approved drug

Two numbers define the economics of drug development: how much it costs,
and how likely a candidate is to survive the pipeline at all.

Wouters, McKee, and Luyten (2020) estimated development costs using
publicly disclosed data (SEC filings, FDA approval records, and
ClinicalTrials.gov) for 63 of the 355 new therapeutic agents the FDA
approved between 2009 and 2018. After capitalizing costs at a real cost of
capital of 10.5% per year to account for the time value of money and the
cost of capital tied up in failed programs, they estimated a **median**
R&D investment of \$985.3 million (95% CI: \$683.6M–\$1228.9M) and a
**mean** of \$1,335.9 million (95% CI: \$1,042.5M–\$1,637.5M) per approved
drug, in 2018 US dollars — figures that varied enormously by therapeutic
area, from a median of \$765.9M for nervous-system drugs to \$2,771.6M for
oncology and immunomodulatory drugs. (A 2022 erratum to this paper
corrected errors found in the underlying source data; the corrected
figures above are the ones reported in that correction. See
[References](#references).)

The reason these figures are so large is that they are *risk-adjusted*:
the direct, out-of-pocket cost of running one clinical trial is much
smaller than the effective cost of getting one drug approved, because most
candidates that enter clinical trials never make it out. A standard way to
express this is to capitalize the cost of each phase forward to the launch
date and divide by the cumulative probability of a candidate that has
reached the start of that phase eventually being approved:

$$
E[\text{Cost}_{\text{launch}}] = \sum_{i=1}^{n} \frac{C_i}{\text{POS}_i} \cdot (1+r)^{\,T_{\text{launch}} - t_i}
$$

where $C_i$ is the direct cost of phase $i$, $\text{POS}_i$ is the
probability that a candidate entering phase $i$ is ultimately approved,
$r$ is the real cost of capital, and $t_i$ is the phase's timing relative
to launch at $T_{\text{launch}}$. This is the general shape of the
risk-adjustment framework used throughout the drug-development-cost
literature (it is a simplified pedagogical form, not a specific paper's
exact computation); the key takeaway is that $\text{POS}_i$ — the
attrition rate — dominates the equation, because it appears in the
denominator of every term.

Attrition itself is substantial and disease-dependent. Using a sample of
over 406,000 clinical trial records for more than 21,000 compounds run
between 2000 and 2015, Wong, Siah, and Lo (2019) found that oncology drugs
had only a 3.4% end-to-end clinical success rate in their sample —
notably lower than the 5.1% figure widely cited from earlier, smaller
industry-curated datasets — though this rate has been highly variable
year to year (it fell to 1.7% in 2012 before rising to 8.3% in 2015). Hay
et al. (2014) independently surveyed clinical success rates across the
broader industry using one of the largest datasets assembled at the time.
Both studies agree on the qualitative picture even where the exact
percentages differ: **the overwhelming majority of clinical candidates
fail**, and failure is concentrated in efficacy (the drug doesn't work
well enough) far more than in safety.

This is the economic backdrop against which every method in this book
should be judged. A computational technique earns its place in the
pipeline only if it plausibly increases $\text{POS}_i$ at some stage, or
reduces $C_i$, or both — not because it is fashionable.

## 1.2 The AI Paradigm Shift

Classical computer-aided drug design relies on hand-engineered
descriptors and rule-based heuristics (Chapter 2 uses one of the oldest
and still most useful examples, Lipinski's Rule of Five). Over the last
decade, machine learning has been layered on top of — and in some cases
has begun to replace — this classical toolkit. Vamathevan et al. (2019),
in a widely cited review, characterize opportunities for ML across every
stage of the pipeline described in §1.1: target identification and
validation, prognostic biomarker discovery, and the analysis of
high-dimensional data such as digital pathology images in clinical
trials. Their central caveat is as important as their central claim: ML
performs best on "well-specified questions with abundant, high-quality
data," and its adoption is currently limited more by interpretability and
repeatability of results than by raw predictive accuracy.

It is useful to organize the computational methods used throughout this
book into three broad, overlapping families:

- **Predictive models** answer a question about a specific molecule or
  protein: will this compound bind this target (Chapters 5–6)? Is this
  molecule likely to be toxic (Chapter 5)? What is this protein's 3D
  structure (Chapter 9)? These are typically supervised learning
  problems — classification or regression on a labeled dataset — and
  their scientific value is bounded by the quality and size of that
  dataset, which is precisely why chemical and structural databases like
  ChEMBL and the PDB (introduced in §1.4) are foundational infrastructure
  for the entire field.
- **Generative models** propose new molecules or sequences rather than
  scoring existing ones: variational autoencoders and diffusion models
  over molecular graphs or 3D coordinates (Chapter 7), autoregressive
  transformers over SMILES/SELFIES strings (Chapter 7), and diffusion
  models over protein backbones (Chapter 10). These models shift the
  bottleneck in early discovery from *screening* a fixed library to
  *designing* candidates directly in the region of chemical or sequence
  space likely to satisfy a set of constraints.
- **Physics-informed models** incorporate known physical or chemical
  constraints — conservation laws, symmetries, or explicit energy
  functions — directly into the model architecture or training
  objective, rather than relying on a purely data-driven fit. Equivariant
  graph neural networks that respect 3D rotational and translational
  symmetry (used in Chapter 6 and Chapter 9) and neural network
  potentials that approximate quantum-mechanical energies at
  classical-simulation speed (Chapter 12) are both examples: the model is
  constrained to only ever produce physically consistent outputs, which
  improves data efficiency and generalization compared with an
  unconstrained model trained on the same data.

The clearest illustration of what this paradigm shift can achieve is
structure prediction. Determining a protein's 3D structure experimentally
(by X-ray crystallography, NMR, or cryo-EM) can take months to years per
structure and does not always succeed. Jumper et al. (2021) showed that
AlphaFold, a deep learning system, could predict protein structures from
amino acid sequence alone at accuracy competitive with experimental
methods for a large fraction of targets in the CASP14 blind assessment —
work that has since made high-confidence structural models available for
essentially the entire sequenced proteome, at a computational cost
orders of magnitude below experimental determination. Chapter 9 covers
AlphaFold and related methods (ESMFold, RoseTTAFold) in depth; it is
introduced here because it is the single clearest data point for the
claim that this book is built on: that the AI paradigm shift is not
merely accelerating the classical pipeline, it is changing which
questions are computationally tractable at all.

None of this makes the classical pipeline in §1.1 obsolete. Every
predictive, generative, and physics-informed model in this book is
trained on data that ultimately comes from wet-lab experiments — assays
run against real targets, structures solved by real crystallographers,
outcomes from real clinical trials. AI does not remove the pipeline; it
changes where effort within the pipeline is spent, ideally shifting
attrition as early and as cheaply as possible, per the economic argument
in §1.1.

## 1.3 Modalities Overview: Small Molecules vs. Macromolecules/Biologics

Everything described so far applies differently depending on what kind of
molecule is being designed. The book covers two broad modalities, which
differ enough in their representation, design constraints, and
manufacturing that they are treated with largely separate computational
toolkits in Parts I–II (small molecules) and Part III (macromolecules).

**Small molecules** are low-molecular-weight organic compounds,
synthesized chemically rather than produced biologically, and — for
orally dosed drugs — usually expected to satisfy heuristics like
Lipinski's Rule of Five: no more than
5 hydrogen-bond donors, no more than 10 hydrogen-bond acceptors, a
molecular weight under 500 Da, and a calculated LogP under 5, on the
premise that violating more than one of these substantially reduces the
probability of good oral absorption and permeability (Lipinski et al.,
2001). Chapter 2's hands-on project computes exactly these four
properties with RDKit. Small molecules are represented computationally as
2D graphs, 1D strings (SMILES, SELFIES, InChI — Chapter 2), fingerprints
(Chapter 2), or 3D conformers (Chapter 2), and their chemical space is
explored using the fingerprint similarity, generative, and docking
methods in Chapters 2, 7, and 11.

**Macromolecules / biologics** — therapeutic proteins, peptides, and
antibodies — are produced biologically (typically via recombinant
expression in engineered cells) rather than synthesized chemically, are
orders of magnitude larger (a typical IgG monoclonal antibody is roughly
150 kDa, versus a few hundred Da for a small molecule), and cannot
generally be dosed orally because they are degraded by the digestive
system, so they are usually delivered by injection or infusion. Their
size and complexity give them properties small molecules structurally
cannot match — very high target specificity and affinity, engineerable
multi-domain architectures (e.g., bispecific antibodies) — at the cost of
manufacturing complexity, immunogenicity risk, and a much larger,
higher-dimensional design space (a single antibody's antigen-binding
region alone involves optimizing complementarity-determining region
loops built from 20 possible amino acids at each position). Part III of
this book (Chapters 8–10) covers the representations (sequences,
contact maps, 3D structural graphs) and generative methods (inverse
folding, backbone diffusion) specific to this modality.

The choice of modality for a given disease target is itself a design
decision with no universally correct answer: small molecules are cheaper
to manufacture and can reach intracellular targets that biologics
generally cannot, while biologics can achieve selectivity against targets
(such as flat protein–protein interaction surfaces) that are notoriously
difficult to drug with small molecules. Recognizing which modality is
appropriate for a given target — and which computational toolchain in
this book therefore applies — is one of the first judgment calls in any
real drug discovery project, which is why this book treats it as the
subject of its very first hands-on exercise, below.

## 1.4 Hands-on Project: Retrieving Real Chemical & Structural Data

The rest of this book works with data pulled from two public
infrastructure databases that anchor the whole field:

- **ChEMBL** (Mendez et al., 2019), maintained by EMBL-EBI, is a curated,
  open-access database of bioactivity data — which compounds were tested
  against which biological targets, and what activity (e.g., IC50, Ki)
  was measured. As of release ChEMBL_37 (May 2026), it holds bioactivity
  data for roughly 2.9 million distinct compounds against over 18,000
  targets, drawn from more than 100,000 publications.
- **The Protein Data Bank (PDB)** (wwPDB consortium, 2019) is the single
  global archive of experimentally determined 3D structures of proteins,
  nucleic acids, and their complexes. As of August 2026 it holds 258,403
  released structures.

The hands-on project for this chapter does not analyze this data yet —
that begins in Chapter 2 (chemical similarity) and Chapter 9 (structure
prediction) — it establishes the environment and the *programmatic*
retrieval pattern that the rest of the book builds on: querying both
databases' public REST APIs directly from Python, rather than manually
downloading files through a web browser. This is a deliberate choice: a
reproducible computational pipeline should be able to re-fetch its own
inputs from source.

The project code lives in this chapter's folder
(`ch01_drug_discovery_pipeline/`) and targets a single, concrete example
throughout: **EGFR** (epidermal growth factor receptor, ChEMBL target
`CHEMBL203`), a receptor tyrosine kinase that is one of the most
extensively studied oncology drug targets and the target used in this
book's Chapter 16 capstone project. It retrieves:

1. **Target metadata and bioactivity data from ChEMBL** — confirming the
   target identity (preferred name, organism, target type) and pulling a
   page of measured bioactivities (assay type, measured value, units, and
   the tested compound's SMILES string) for EGFR.
2. **Structural metadata and a 3D structure file from the PDB** — for PDB
   entry `1M17`, the crystal structure of the EGFR tyrosine kinase domain
   bound to the inhibitor erlotinib, retrieving both the entry's metadata
   (title, resolution, release date) and the structure file itself.

See [`README.md`](README.md) in this folder for setup and usage
instructions, and [`fetch_data.py`](fetch_data.py) for the implementation.
The accompanying test suite (`tests/test_fetch_data.py`) exercises the
same functions against the live APIs, so running the tests is itself a
reproducibility check on the retrieval pipeline.

### A note on Google Colab

Every dependency this project needs (`requests`, part of a standard
Python installation's ecosystem) is preinstalled on Google Colab's
default runtime, so the entire project runs unmodified in a fresh Colab
notebook — install nothing, add a cell with `!git clone` or paste
`fetch_data.py`'s contents, and run. No GPU or special runtime is
required for this chapter's project; later chapters that require a GPU
runtime will say so explicitly.

## References

- Scannell, J. W., Blanckley, A., Boldon, H., & Warrington, B. (2012).
  Diagnosing the decline in pharmaceutical R&D efficiency. *Nature
  Reviews Drug Discovery*, 11(3), 191–200.
  https://doi.org/10.1038/nrd3681
- Wouters, O. J., McKee, M., & Luyten, J. (2020). Estimated Research and
  Development Investment Needed to Bring a New Medicine to Market,
  2009-2018. *JAMA*, 323(9), 844–853.
  https://doi.org/10.1001/jama.2020.1166 (Erratum: Wouters, O. J., McKee,
  M., & Luyten, J. (2022). Errors in Source Data for Study of Drug
  Development Costs. *JAMA*, 328(11), 1110.
  https://doi.org/10.1001/jama.2022.14317)
- Wong, C. H., Siah, K. W., & Lo, A. W. (2019). Estimation of clinical
  trial success rates and related parameters. *Biostatistics*, 20(2),
  273–286. https://doi.org/10.1093/biostatistics/kxx069 (Corrigendum:
  *Biostatistics*, 20(2), 366.
  https://doi.org/10.1093/biostatistics/kxy072)
- Hay, M., Thomas, D. W., Craighead, J. L., Economides, C., & Rosenthal,
  J. (2014). Clinical development success rates for investigational
  drugs. *Nature Biotechnology*, 32(1), 40–51.
  https://doi.org/10.1038/nbt.2786
- Vamathevan, J., Clark, D., Czodrowski, P., Dunham, I., Ferran, E., Lee,
  G., Li, B., Madabhushi, A., Shah, P., Spitzer, M., & Zhao, S. (2019).
  Applications of machine learning in drug discovery and development.
  *Nature Reviews Drug Discovery*, 18(6), 463–477.
  https://doi.org/10.1038/s41573-019-0024-5
- Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger,
  O., et al. (2021). Highly accurate protein structure prediction with
  AlphaFold. *Nature*, 596(7873), 583–589.
  https://doi.org/10.1038/s41586-021-03819-2
- Mendez, D., Gaulton, A., Bento, A. P., Chambers, J., De Veij, M., Félix,
  E., et al. (2019). ChEMBL: towards direct deposition of bioassay data.
  *Nucleic Acids Research*, 47(D1), D930–D940.
  https://doi.org/10.1093/nar/gky1075
- wwPDB consortium. (2019). Protein Data Bank: the single global archive
  for 3D macromolecular structure data. *Nucleic Acids Research*, 47(D1),
  D520–D528. https://doi.org/10.1093/nar/gky949
- Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J. (2001).
  Experimental and computational approaches to estimate solubility and
  permeability in drug discovery and development settings. *Advanced Drug
  Delivery Reviews*, 46(1-3), 3–26.
  https://doi.org/10.1016/S0169-409X(00)00129-0

Live data cited in §1.4 (ChEMBL release version and count, PDB entry
count) were retrieved directly from the ChEMBL and RCSB PDB REST APIs on
2026-08-16; see `fetch_data.py` to reproduce.
