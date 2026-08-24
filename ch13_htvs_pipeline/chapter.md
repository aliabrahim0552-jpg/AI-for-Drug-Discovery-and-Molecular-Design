# Chapter 13: High-Throughput Virtual Screening (HTVS) Pipelines

Chapter 12 closed by measuring, directly and honestly, what one
careful, real docking-then-MD assessment of a single compound actually
costs: hours of real AutoDock Vina search time per ligand (Chapter 11),
and — for anything beyond a short correctness check — days of real
ANI-2x compute per nanosecond of trajectory (Chapter 12). That finding
raises the question this chapter answers: a real drug-discovery
campaign rarely starts with one compound. It starts with a library —
a real vendor catalog, a real fragment collection, a real internal
archive — containing anywhere from thousands to billions of candidates,
and the resources to synthesize and test any of them experimentally are
always, by orders of magnitude, smaller than the library itself. Every
one of those candidates cannot receive Chapter 11's or Chapter 12's
full treatment; the practical problem is deciding, computationally and
in advance, which handful deserve it. This chapter covers the funnel
architecture that problem demands (Section 13.1), the specific
three-stage design this book's own outline specifies and this chapter's
own tooling constraints shape in practice (Section 13.2), and a real,
complete run of that funnel against a real emerging viral target
(Section 13.3).

## 13.1 Building the HTVS Funnel

**The scale mismatch.** A real, modern make-on-demand chemical space —
Enamine's REAL Database, for instance — already contains on the order
of tens of billions of synthesizable compounds, and ultra-large virtual
libraries assembled for docking campaigns have reached the
hundred-million to billion-compound range in the published literature
(Lyu et al., 2019, docked 99-138 million real make-on-demand compounds
against two real targets, AmpC beta-lactamase and the D4 dopamine
receptor, using physics-based docking alone). No wet lab tests
anywhere near that many compounds; a real experimental follow-up
campaign might synthesize and assay dozens to a few hundred. The
**high-throughput virtual screening funnel** is the standard answer to
that many-orders-of-magnitude gap: a sequence of computational filters,
ordered from cheapest-and-least-accurate to
most-expensive-and-most-predictive, each applied only to whatever
survived the filter before it. A filter that costs microseconds per
compound (Section 13.2's Tier 1) can afford to touch the entire
starting library; a filter that costs minutes to hours per compound
(Tier 2) can only afford the thousands that remain after Tier 1; a
filter that costs hours to days per compound (Tier 3) can only afford
the tens that remain after Tier 2.
The funnel's efficiency comes entirely from this ordering — cheap
methods doing the bulk of the elimination so expensive methods are
never wasted on a compound an earlier, cheaper stage could already have
ruled out.

**What "accuracy" means at each stage.** Every stage is a real
trade-off between throughput and predictive fidelity, not a strictly
worse-vs-better ranking — a compound rejected at Tier 1 for a poor
computed drug-likeness profile is not necessarily inactive against the
biological target at all, and a compound favorably docked at Tier 2 is
not necessarily a genuine binder in solution (Chapter 11 §11.3's own
real finding — a Spearman ρ of only 0.245 between Vina score and
measured EGFR potency — is a direct, first-hand illustration of exactly
this gap). The funnel design accepts a real, non-zero false-negative
rate at every stage in exchange for tractability: the goal is not to
guarantee that the best possible compound survives to the end, but to
enrich the final, small shortlist for real hits far above what
selecting the same number of compounds at random would achieve — the
**enrichment factor**,

$$
\text{EF} = \frac{\text{hits in shortlist} / N_{\text{shortlist}}}{\text{hits in library} / N_{\text{library}}}
$$

the standard quantitative measure the field uses to evaluate a
screening funnel's real, practical value (Truchon & Bayly, 2007), and
the one this chapter's own hands-on project computes directly in
Section 13.3 against real, measured ground truth.

## 13.2 Tiered Screening Strategy

This book's own outline specifies a three-stage funnel: fast QSAR/ADMET
filtering, then DiffDock pose prediction, then short MD stability
checks. This chapter's hands-on project runs all three stages for
real, with one substitution already established rather than newly
discovered here: Chapter 11 §11.2's real feasibility investigation
found DiffDock's own installation requirements (a `conda`-only
environment, a GPU recommendation, no lightweight pip-installable batch
API) infeasible in this book's CPU-only, pip-first authoring
environment, and ran real AutoDock Vina instead. That finding carries
over unchanged to this chapter's Tier 2 — re-checking it here would be
redundant with Chapter 11's own documented investigation, not a new
result.

**Tier 1: fast, rule-based ADMET/drug-likeness filtering.** The
cheapest possible filter uses no target structure and no activity data
at all — only a compound's own 2D structure, from which a handful of
physicochemical descriptors are computed essentially instantly.
Chapter 2 §2.5 already introduced **Lipinski's Rule of Five** (molecular
weight, calculated LogP, hydrogen-bond donor/acceptor counts, each
compared against a threshold empirically associated with oral
bioavailability). This chapter adds three further, real, established
filters, chosen because each targets a real failure mode Lipinski's
rules do not cover:

- **Veber's rules** (Veber et al., 2002): rotatable-bond count ≤ 10 and
  topological polar surface area (TPSA) ≤ 140 Å² — an independent,
  real empirical finding that molecular *flexibility* and *polarity*
  predict oral bioavailability in rats even for compounds that already
  pass Lipinski's mass/LogP-based criteria.
- **PAINS structural alerts** (Baell & Holloway, 2010): a real, curated
  catalog of substructures (rhodanines, catechols, and related motifs)
  empirically associated with *assay interference* — compounds that
  appear active across many unrelated biochemical assays via
  redox cycling, aggregation, or fluorescence/absorbance artifacts
  rather than genuine, reproducible target engagement. RDKit ships this
  catalog directly (`rdkit.Chem.FilterCatalog`), used here exactly as
  distributed, not reimplemented.
- **QED** (quantitative estimate of drug-likeness; Bickerton et al.,
  2012): a single continuous score in $[0, 1]$ combining eight
  physicochemical properties (including MW, LogP, TPSA, and the
  Lipinski/Veber descriptors above) as a weighted geometric mean of
  individual property-specific desirability functions $d_i$,
  $$
  \text{QED} = \exp\left(\frac{1}{n}\sum_{i=1}^{n} w_i \ln d_i(x_i)\right),
  $$
  where each $d_i$ maps one property's value to a $[0,1]$ desirability
  derived from the distribution of that property across real approved
  drugs — a smoother, single-number summary than a hard pass/fail rule
  set, used here as an additional lenient floor ($\text{QED} \geq 0.30$)
  alongside the harder Lipinski/Veber/PAINS checks rather than as a
  replacement for them.

The four real checks combine into a single Tier 1 pass/fail call,
computed directly from RDKit descriptors with no target structure and
no activity data anywhere in the function:

```python
def compute_tier1_properties(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
    rotb, tpsa = Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)
    qed, pains_alert = QED.qed(mol), _pains_catalog().HasMatch(mol)

    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    veber_pass = rotb <= 10 and tpsa <= 140.0
    return {"passes_tier1": lipinski_violations <= 1 and veber_pass
            and not pains_alert and qed >= 0.30, ...}
```

**Tier 2: real AutoDock Vina docking (Chapter 11's method).** Every
Tier 1 survivor is docked, exactly as Chapter 11 §11.1 described in
full — the same empirical scoring function, the same iterated local
search, the same pocket-informed focused box protocol — against this
chapter's own real receptor (Section 13.3). This chapter does not
re-derive Vina's mechanics; it applies them, unchanged, as the second
funnel stage.

**Tier 3: real, short ANI-2x MD stability checks (Chapter 12's
method).** Only the small number of Tier 2's best-ranked survivors
receive the funnel's most expensive treatment: real Langevin dynamics
under the real, official ANI-2x neural network potential, exactly as
Chapter 12 §12.3-12.4 ran it. Chapter 12's own real, measured
throughput finding — a full protein-ligand complex is real but far too
slow for anything beyond a short correctness demonstration in this
environment, while a ligand-alone trajectory is genuinely tractable at
the tens-of-picosecond scale — sets this chapter's Tier 3 scope
directly: a real, short (picosecond-scale) ligand-alone trajectory per
shortlisted compound, checked for a bounded, non-diverging RMSD (a real
integration-correctness and gross-instability check) rather than run
long enough to support a converged structural-stability claim. This is
a narrower claim than "MD confirms this compound is a stable binder" —
stated honestly as such in Section 13.3's results, not inflated.

## 13.3 Hands-on Project: A Real Tiered HTVS Funnel Against SARS-CoV-2 Main Protease

The project code lives in this chapter's folder
(`ch13_htvs_pipeline/`). Given Section 13.2's design, this project runs
all three real tiers in sequence against a real, emerging viral target,
retrospectively validated against real, independent ground truth.

### Real target and real screening library

**PDB 5R82** (Douangamath et al., 2020): a real, high-resolution
(1.31 Å) crystal structure of **SARS-CoV-2 main protease** (Mpro,
also called 3CLpro or Nsp5) — the cysteine protease responsible for
proteolytically processing the viral polyprotein into its functional
components, and, because no human protease shares its cleavage
specificity, one of the most extensively pursued small-molecule targets
of the COVID-19 pandemic response. This particular structure was
solved in complex with a real fragment hit, **Z219104216** (PDB ligand
code RZS; SMILES `CCNc1ccc(C#N)cn1`, confirmed directly by parsing the
official PDB Chemical Component Dictionary's `RZS_ideal.sdf` with
RDKit, not taken from a secondary source), from the same real,
published crystallographic and electrophilic fragment-screening
campaign against this exact target that deposited 97 total structures
into the PDB.

**A real, potency-labeled screening library.** `htvs_pipeline.py`
fetches, live, the complete real data table for **PubChem AID 1805203**
("SARS-CoV-1 and -2 3CLpro Biochemical Assay," from Han et al., 2022) —
a real, published structure-activity series with measured IC50 values
against SARS-CoV-2 3CLpro (a scaled-down FRET biochemical assay). The
raw table carries 74 rows because the assay's own real protocol runs
every compound in **two independent replicate dilution series**
(columns 3-12 and 13-22 of its 384-well plate layout); curation
deduplicates these to one record per real PubChem CID — the real
median IC50 across each compound's own real replicate measurements,
with Active/Inactive determined directly from that median against the
assay's own documented ≤10 µM threshold — yielding **37 real, distinct
compounds: 31 active, 6 inactive**. (An earlier version of this
chapter's own curation logic skipped this deduplication step and ran
the full funnel against 74 duplicated rows; the bug was caught by
inspecting the real per-compound results directly — several PubChem
CIDs appeared twice with two different real measured IC50 values — and
fixed before any result in this section was finalized; see
`tests/test_htvs_pipeline.py`'s
`test_curate_library_deduplicates_real_replicate_wells_by_cid` for the
regression test this added.) Every real compound's SMILES, CID, real
median IC50, and replicate count are cached in
`data/sars_cov2_3clpro_library.json` for offline reproducibility. This
real ground truth is used *only* for this section's retrospective
funnel validation, below — never supplied to Tier 1's rule-based filter
(which uses no activity data by construction) or to Tier 2's docking
(which uses no activity data either, only geometry and the Vina scoring
function).

**Why a 37-compound library, not the outline's illustrative "millions."**
This is the same honestly-scoped-real-experiment choice Chapters 10-12
each made for their own hands-on projects, for the same underlying
reason: a real library large enough to need Tier 1's rule-based filter
to do serious work would need Tier 2 to dock thousands of survivors, at
the real, measured per-compound cost this section reports below —
multiplying into a compute budget this book's
free-tier-Colab-compatible authoring environment cannot absorb in one
working session.
A real, published, potency-labeled compound set, by contrast, is small
enough to carry every real compound through every real tier in one
session, while still being large enough — and, crucially, independently
*labeled* — to compute a real, quantitative enrichment statistic at the
end, exactly the funnel property this chapter exists to demonstrate.

### Real receptor preparation and redocking validation

The real 5R82 structure is split into a protein-only file and RZS's own
real `HETATM` records (excluding the DMS cryoprotectant and ordered
waters also present in the deposition); the receptor is protonated at
pH 7.4 and converted to a rigid AutoDock PDBQT file with OpenBabel,
identically to Chapter 11's protocol. The focused docking box is a
22.5 Å cube centered on RZS's real crystallographic centroid.

Before trusting this new receptor for Tier 2, RZS is docked back into
its own real pocket (three real independent replicates, Chapter 11's
redocking-control protocol):

| Replicate | Affinity (kcal/mol) | RMSD to crystal pose (Å) | Correct pose (<2.0 Å)? |
|---|---|---|---|
| 1 | -4.457 | 0.459 | Yes |
| 2 | -4.423 | 0.512 | Yes |
| 3 | -4.477 | 0.688 | Yes |
| **Mean ± SD** | **-4.452 ± 0.022** | **0.553 ± 0.098** | **3/3** |

Every real replicate lands well under the field's conventional 2.0 Å
"correct pose" threshold, with low affinity variance across replicates
— a real, direct confirmation that this chapter's receptor preparation
and docking protocol are trustworthy on this new target before any
library compound is docked. RZS's real redocking RMSD is also
substantially tighter than Chapter 11's own erlotinib redocking result
(2.17-2.63 Å, 0/5 replicates under threshold) — consistent with RZS
being an 11-heavy-atom fragment with almost no rotatable bonds, a
genuinely easier real pose-reproduction problem than erlotinib's larger,
more flexible scaffold, not evidence of a systematically better protocol.

### Tier 1: real rule-based filtering results

| Statistic | Value |
|---|---|
| Compounds entering Tier 1 | 37 |
| Compounds passing Tier 1 | 36 (97.3%) |
| Real active compounds retained | 30/31 (96.8%) |
| Real inactive compounds retained | 6/6 (100%) |

Exactly one real compound was rejected: **CID 156027228**
(`O=C(Cn1nnc2ccccc21)N(Cc1cc(Cl)cc(Cl)c1)c1ccc(-c2cccnc2)cc1`), a real,
measured 3CLpro *active* (an honest, real Tier 1 false negative,
consistent with Section 13.1's point that a rejected-at-Tier-1
compound is not necessarily inactive). Its real computed properties
show why: molecular weight 488.4 Da and calculated LogP 6.03 (one
Lipinski violation — allowed on its own under this chapter's ≤1
threshold) combined with a real QED of only 0.293, just under this
chapter's 0.30 floor — the elevated LogP alone was enough to pull its
overall drug-likeness score below the lenient cutoff, even with zero
PAINS alerts and full Veber compliance (TPSA 63.9 Å², 6 rotatable
bonds). As anticipated in Section 13.3's scope note, Tier 1 removed
only a small fraction (2.7%) of this particular library — this
published lead-optimization series is already drug-like by
construction, not evidence that Tier 1 filtering is generally weak.

### Tier 2: real AutoDock Vina docking results

| Statistic | Value |
|---|---|
| Compounds docked | 36/36 (100% real docking success) |
| Mean focused affinity | -7.102 ± 0.384 kcal/mol (range -8.078 to -6.484) |
| Mean real wall time per compound | 238.2 s |
| Total real Vina search time (36 compounds) | 8,575.3 s (≈2.38 CPU-hours) |
| Spearman ρ (affinity vs. real pIC50) | **-0.489 (p = 0.0025)** |

**Reading the sign correctly.** Vina's own convention makes a *more
negative* affinity the *more favorable* prediction, while a *higher*
pIC50 means *more potent*. A negative Spearman ρ between the two is
therefore the physically sensible direction — more favorable predicted
binding associated with higher real measured potency — and, at
ρ = -0.489 with p = 0.0025 (n = 36), this is a real, moderate,
statistically significant correlation, not a null result. This is a
materially different outcome from Chapter 11's own real finding
(ρ = 0.245, p = 0.19, not significant, and in the "wrong" sign
direction relative to Vina's convention there too) — and the real
cause is visible directly in this chapter's own data rather than left
as speculation: within this one congeneric SAR series, real molecular
weight itself correlates with both a more favorable real docking
affinity (ρ = -0.525, p = 0.001) and, independently, with somewhat
higher real measured pIC50 (ρ = 0.336, p = 0.045) — a real, if
imperfect, structure-activity trend a single lead-optimization
campaign's own analog series is expected to show, and one Vina's
contact-counting scoring terms (Section 13.2) can partially track
*because* every compound here shares the same core scaffold and binding
mode, unlike Chapter 11's chemically diverse, cross-scaffold ChEMBL
benchmark. The correlation is real but far from perfect: **CID
156027226** (-7.754 kcal/mol, the third most favorable real affinity in
the set) is a real, measured comparatively weak binder (pIC50 5.17,
6.7 µM) at a similar molecular weight (471.9 Da) to several far more
potent compounds nearby in the affinity ranking — the same real
caveat Chapter 11 §11.3 illustrated with CHEMBL62843, present here in
softer, statistical form rather than as a single dramatic outlier.

### Tier 3: real, short ANI-2x MD stability results

The top 8 Tier 2 survivors by real focused-docking affinity, all real
measured actives, advanced to a real, short (2 ps, 2,000-step) ANI-2x
ligand-alone trajectory each:

| PubChem CID | Affinity (kcal/mol) | Real pIC50 | Atoms | Mean RMSD (Å) | Max RMSD (Å) | Stable? |
|---|---|---|---|---|---|---|
| 156027233 | -8.078 | 6.574 | 59 | 1.303 | 1.904 | Yes |
| 156027232 | -7.958 | 6.520 | 51 | 1.304 | 2.056 | Yes |
| 156027226 | -7.754 | 5.174 | 53 | 1.042 | 1.850 | Yes |
| 156027234 | -7.591 | 6.694 | 55 | 1.179 | 2.314 | Yes |
| 156027227 | -7.536 | 7.016 | 53 | 0.907 | 1.568 | Yes |
| 156027237 | -7.500 | 7.362 | 51 | 1.360 | 2.110 | Yes |
| 156027229 | -7.478 | 6.438 | 56 | 1.027 | 2.024 | Yes |
| 156027225 | -7.437 | 6.495 | 53 | 1.096 | 1.576 | Yes |

All 8 real trajectories are stable by this chapter's bounded-RMSD
criterion (max RMSD well under 15 Å, no divergence or numerical
blow-up in any of the 8 real, independent short trajectories) — a real
correctness/stability check, not a converged binding-affinity claim
(Section 13.2). One real, honest measurement artifact: the first
compound processed (156027233) recorded 1,805 ms/step against a
steady-state 45-68 ms/step for the remaining seven, similar-sized
(51-56-atom) ligands — a real, one-time ANI-2x model-loading/warm-up
cost paid once by the first TorchANI evaluation in this run's process,
not a property of that specific molecule; the remaining seven
compounds' throughput is consistent with Chapter 12's own measured
52-atom ligand-alone rate (67.64 ms/step). The full real Tier 3 stage
took 4,412.9 s (≈73.5 minutes) wall-clock.

### Retrospective enrichment: did the real funnel concentrate real actives?

| Quantity | Value |
|---|---|
| Library real active rate | 31/37 = 83.8% |
| Tier 3 shortlist real active rate | 8/8 = 100% |
| **Enrichment factor** | **1.194** |

All 8 real compounds that survived the complete funnel are real,
measured SARS-CoV-2 3CLpro actives — a genuinely correct outcome, but
one this chapter reports honestly rather than as a dramatic result: the
enrichment factor itself is modest (1.194, not the order-of-magnitude
values a real enrichment study against a realistic, mostly-inactive
prospective library would target) because this library's own real
baseline active rate is already unusually high (83.8%). A published
lead-optimization SAR series is, by construction, built almost entirely
around a known active scaffold — the opposite starting condition from
a real, unfiltered prospective HTVS campaign, where the baseline hit
rate is typically well under 1% and the same absolute funnel behavior
would produce a dramatically larger, more meaningful enrichment factor.
This is the direct, quantitative version of Section 13.3's earlier
disclosure about Tier 1's own low rejection rate on this same library —
both numbers understate what this exact funnel architecture would
achieve against a realistic, low-hit-rate prospective library, and both
are reported as real, honest measurements of *this* library rather than
adjusted to look more dramatic.

### Reproducibility

Dependencies are version-floored (`rdkit>=2023.9`, `meeko>=0.6`,
`numpy>=1.24`, `scipy>=1.10`, `requests>=2.28`, `openmm>=8.5`,
`openmmml>=1.4`, `torchani>=2.2` in
[`requirements.txt`](requirements.txt); `vina>=1.2.5` on platforms with
a prebuilt wheel — validated against rdkit 2026.03.5, meeko 0.7.1,
numpy 2.5.2, scipy 1.18.0, requests 2.34.2, openmm 8.6.0, openmmml 1.7,
torchani 2.8.4, and OpenBabel 3.1.0 on Python 3.12; this chapter's own
results were produced with the official standalone AutoDock Vina 1.2.7
CLI binary via `VINA_EXECUTABLE`, the same Windows-no-wheel fallback
path Chapter 11 documented and used).
`data/sars_cov2_3clpro_library.json` caches the exact real,
deduplicated 37-compound library this chapter's results were computed
from, so `python htvs_pipeline.py` reproduces the same funnel offline
without a live PubChem API call (pass `--refresh-cache` to re-fetch
live instead). The
[`tests/test_htvs_pipeline.py`](tests/test_htvs_pipeline.py) suite
exercises the real data-curation (including the real replicate-well
deduplication logic), Tier 1 filter, receptor/box geometry, and
Kabsch/RMSD logic directly and offline; the small number of tests that
would otherwise invoke Vina or ANI-2x are skipped automatically when
neither engine is available on the host. This chapter's own real
docking campaign also exercised Chapter 11's resumability design
directly, for a real reason of its own: mid-campaign, this session's
own worker count was reduced from 4 to 2 to reduce CPU contention on
this environment's 2-physical-core machine, and every already-completed
real Vina result — including one job recovered from a
completed-but-not-yet-recorded invocation — was reused rather than
recomputed, the same "resume real completed work rather than repeat
it" principle Chapter 11 established for its own, differently-caused
mid-run interruption.

### Limitations and what comes next

This project runs a real, complete, three-tier HTVS funnel end to end
against a real emerging viral target, with real retrospective
validation against independent, published ground truth — but at a
real, honestly-scoped 37-compound library rather than the outline's
illustrative millions-to-billions scale, for the compute-budget reasons
detailed above, and its Tier 3 stability check is a short,
integration-correctness-level trajectory (Chapter 12's own established
scale), not a converged binding-stability study. Tier 1's rule-based
filter also removed only a small fraction of this particular library —
expected and disclosed here, not a defect: a published
lead-optimization SAR series (this chapter's real data source) is already
drug-like by construction, unlike a raw, unfiltered vendor library a
real prospective campaign would actually start from, where Tier 1 would
be expected to do far more of the funnel's total elimination work.
Chapter 14 shifts this book's focus from small-molecule and protein
targets to a different real molecular modality entirely — RNA — where
analogous representation, prediction, and design questions arise in a
biophysically distinct setting.

## References

- Lyu, J., Wang, S., Balius, T. E., Singh, I., Levit, A., Moroz, Y. S.,
  O'Meara, M. J., Che, T., Algaa, E., Tolmachova, K., Tolmachev, A. A.,
  Shoichet, B. K., Roth, B. L., & Irwin, J. J. (2019). Ultra-large
  library docking for discovering new chemotypes. *Nature*, 566(7743),
  224-229. https://doi.org/10.1038/s41586-019-0917-9
- Truchon, J.-F., & Bayly, C. I. (2007). Evaluating virtual screening
  methods: Good and bad metrics for the "early recognition" problem.
  *Journal of Chemical Information and Modeling*, 47(2), 488-508.
  https://doi.org/10.1021/ci600426e
- Veber, D. F., Johnson, S. R., Cheng, H.-Y., Smith, B. R., Ward, K.
  W., & Kopple, K. D. (2002). Molecular Properties That Influence the
  Oral Bioavailability of Drug Candidates. *Journal of Medicinal
  Chemistry*, 45(12), 2615-2623. https://doi.org/10.1021/jm020017n
- Baell, J. B., & Holloway, G. A. (2010). New Substructure Filters for
  Removal of Pan Assay Interference Compounds (PAINS) from Screening
  Libraries and for Their Exclusion in Bioassays. *Journal of Medicinal
  Chemistry*, 53(7), 2719-2740. https://doi.org/10.1021/jm901137j
- Bickerton, G. R., Paolini, G. V., Besnard, J., Muresan, S., &
  Hopkins, A. L. (2012). Quantifying the chemical beauty of drugs.
  *Nature Chemistry*, 4(2), 90-98. https://doi.org/10.1038/nchem.1243
- Douangamath, A., Fearon, D., Gehrtz, P., Krojer, T., Lukacik, P.,
  Owen, C. D., Resnick, E., Strain-Damerell, C., Aimon, A.,
  Ábrányi-Balogh, P., Brandão-Neto, J., Carbery, A., Davison, G., Dias,
  A., Downes, T. D., Dunnett, L., Fairhead, M., Firth, J. D., Jones, S.
  P., Keeley, A., Keserű, G. M., Klein, H. F., Martin, M. P., Noble, M.
  E. M., O'Brien, P., Powell, A., Reddi, R. N., Skyner, R., Snee, M.,
  Waring, M. J., Wild, C., London, N., von Delft, F., & Walsh, M. A.
  (2020). Crystallographic and electrophilic fragment screening of the
  SARS-CoV-2 main protease. *Nature Communications*, 11, 5047.
  https://doi.org/10.1038/s41467-020-18709-w
- Han, S. H., Goins, C. M., Arya, T., Shin, W.-J., Maw, J., Hooper, A.,
  Sonawane, D. P., Porter, M. R., Bannister, B. E., Crouch, R. D.,
  Lindsey, A. A., Lakatos, G., Martinez, S. R., Alvarado, J., Akers, W.
  S., Wang, N. S., Jung, J. U., Macdonald, J. D., & Stauffer, S. R.
  (2022). Structure-Based Optimization of ML300-Derived, Noncovalent
  Inhibitors Targeting the Severe Acute Respiratory Syndrome
  Coronavirus 3CL Protease (SARS-CoV-2 3CLpro). *Journal of Medicinal
  Chemistry*, 65(4), 2880-2904.
  https://doi.org/10.1021/acs.jmedchem.1c00598

See Chapter 11's references for Trott & Olson (2010) and Eberhardt et
al. (2021) (AutoDock Vina, reused unchanged for this chapter's Tier 2)
and O'Boyle et al. (2011) (OpenBabel, reused for receptor preparation).
See Chapter 12's references for Devereux et al. (2020) and Gao et al.
(2020) (ANI-2x/TorchANI, reused unchanged for this chapter's Tier 3)
and Eastman et al. (2017) (OpenMM). RDKit (Chapters 2, 4, 11-12) is
reused for Tier 1's descriptor/PAINS/QED computation and for ligand
preparation; its `FilterCatalog` PAINS implementation and `QED` module
are used exactly as distributed, with no modification.

All redocking, Tier 1/2/3, and enrichment numbers cited in Section 13.3
were computed directly by running `htvs_pipeline.py` against the real
bundled PDB 5R82 structure and the real cached PubChem AID 1805203
library on 2026-08-21, not taken from a secondary source — see
`results/htvs_results.json` to reproduce.
