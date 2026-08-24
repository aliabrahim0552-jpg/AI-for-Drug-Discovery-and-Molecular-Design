# Chapter 14: AI for RNA Therapeutics & Nucleic Acid Design

Chapter 13 closed by handing off to a different real molecular
modality: RNA. Every chapter before this one treated nucleic acids
either as a genome-scale data source (Chapter 4's ChEMBL/PubChem/UniProt
pipelines) or as the thing a small molecule or a protein binds to; none
of them treated RNA itself as the therapeutic modality — the molecule
that is designed, optimized, and delivered as a drug. That gap matters
now for a concrete, real reason: RNA therapeutics stopped being a
niche in the last decade. Small interfering RNA (siRNA) drugs
(patisiran, givosiran, lumasiran, inclisiran) and mRNA vaccines both
reached real regulatory approval and, in the case of the mRNA
COVID-19 vaccines, real global deployment at a scale no earlier
nucleic-acid therapeutic approached — built on real, specific molecular
design choices this chapter covers directly. RNA is also a
biophysically distinct design problem from the small molecules and
proteins of Chapters 1-13: it is simultaneously the therapeutic
molecule *and*, to a much greater extent than a folded protein, an
object whose function is legible directly from its own sequence
through base-pairing thermodynamics — a property this chapter's own
tooling (Section 14.1) and hands-on project (Section 14.3) both
exploit directly, computationally, and cheaply, without the
GPU-scale structure-prediction models Part III required for proteins.

This chapter covers real RNA secondary- and tertiary-structure
prediction and what each is actually good for (Section 14.1); the real,
established design principles behind optimizing an mRNA or an siRNA for
stability, translation, and low immunogenicity (Section 14.2); and a
real, complete hands-on project that predicts siRNA knockdown efficacy
directly from real, published RNAi silencing data and real,
self-computed RNA thermodynamics (Section 14.3).

## 14.1 RNA Structure Prediction

### 14.1.1 A different structure-prediction problem than Part III's

Part III's central finding — most explicitly in Chapter 9 — was that
protein tertiary structure is extraordinarily hard to predict from
sequence alone by first-principles physics, and that it took
evolutionary information (multiple sequence alignments) and, later,
enormous learned models (AlphaFold2/3, ESMFold) to close that gap.
RNA presents almost the opposite profile. **RNA secondary structure**
— the set of Watson-Crick ($\text{A}{=}\text{U}$, $\text{G}{\equiv}\text{C}$)
and wobble ($\text{G}{\cdot}\text{U}$) base pairs a single RNA strand
forms with itself — is, to first approximation, a direct, computable
consequence of the sequence's own nearest-neighbor base-pairing
thermodynamics, solvable by exact dynamic programming in polynomial
time, no learned model or evolutionary information required. This is
not a minor convenience: for many of RNA's real biological and
therapeutic roles — a small interfering RNA's guide-strand loading, a
microRNA's seed-pairing target recognition, a riboswitch's
ligand-sensing conformational switch, an mRNA vaccine's 5' UTR
translation efficiency — the 2D secondary structure alone already
captures most of what matters functionally, which is exactly why
Section 14.3's hands-on project builds an entire real predictive
pipeline on secondary-structure and duplex thermodynamics without
ever needing a 3D coordinate. **RNA tertiary (3D) structure**
prediction, by contrast, remains a substantially harder and less mature
problem than protein 3D structure prediction is today (Section
14.1.4) — the inverse of Part III's own difficulty ordering.

### 14.1.2 The nearest-neighbor thermodynamic model and the Zuker MFE algorithm

A secondary structure $s$ for a sequence of length $n$ is a set of base
pairs $(i,j)$, $i<j$, with no nucleotide participating in more than one
pair and no two pairs "crossing" ($i < k < j < l$ for two pairs $(i,j)$
and $(k,l)$ — the no-pseudoknot restriction Section 14.1.3 returns to).
The **nearest-neighbor thermodynamic model** (Turner & Mathews, 2010)
assigns structure $s$ a total free energy as an additive sum over its
local structural elements — stacked base pairs, hairpin loops, bulges,
internal loops, and multi-branch loops — each parameterized by a real,
experimentally measured free-energy increment (from UV-melting
experiments on thousands of short synthetic duplexes, curated into the
Nearest Neighbor Database, NNDB). A stacked pair of adjacent Watson-Crick
base pairs, for instance, contributes a real, sequence-context-specific
$\Delta G$ around $-1$ to $-3\,\text{kcal/mol}$; an unpaired hairpin
loop contributes a real, loop-length-dependent *penalty* (entropic cost
of constraining a loop). No physics-based first-principles calculation
is involved — the model is entirely built from these curated,
real, measured increments, exactly the same "trust real, published
experimental measurements over a from-scratch physical simulation"
strategy Chapter 6's bond-length priors and Chapter 12's classical force
fields both already relied on.

Finding the secondary structure that minimizes this total free energy —
the **minimum free energy (MFE) structure** — is the real dynamic
program Zuker and Stiegler (1981) introduced (the algorithm ViennaRNA's
`RNA.fold()` still implements, in a considerably more complete and
optimized form, four decades later). Two coupled recursions suffice for
the pseudoknot-free case: $W(j)$, the MFE over the prefix $1{\ldots}j$
allowing $j$ to be unpaired or paired to any earlier position, and
$V(i,j)$, the MFE over the substructure enclosed by pair $(i,j)$,
*given* that $(i,j)$ pairs:

$$
W(j) = \min\Big( W(j-1),\ \min_{1 \le i < j} \big\{ W(i-1) + V(i,j) \big\} \Big)
$$

$$
V(i,j) = \min\Big( eH(i,j),\ \min_{i<i'<j'<j} \big\{ eL(i,j,i',j') + V(i',j') \big\},\ \min_{\text{multiloop split}} \big\{ \cdots \big\} \Big)
$$

where $eH(i,j)$ is the real, measured hairpin-loop energy for closing
pair $(i,j)$, and $eL(i,j,i',j')$ is the real, measured stacking or
internal/bulge-loop energy between an outer pair $(i,j)$ and an inner
pair $(i',j')$ (a simple stack when $i'=i{+}1, j'=j{-}1$; a bulge or
internal loop otherwise). Filling both matrices takes $O(n^2)$ space and
$O(n^3)$ time (the internal-loop term's own double sum, in its
unoptimized form) — fast enough to fold thousands of siRNA-length
sequences in well under a second each, which is precisely why Section
14.3's project can afford to fold every one of 2,361 real sequences
individually rather than relying on any pre-computed structural
feature.

### 14.1.3 The McCaskill partition function and what it buys beyond the single MFE structure

The MFE structure is a single point estimate — the *one* lowest-energy
fold — but a real RNA molecule in solution exists as a thermal ensemble
over many accessible conformations, particularly in loosely structured
or marginal-energy regions. McCaskill's (1990) partition-function
algorithm replaces Zuker's $\min$ recursions with the analogous
$\log\text{-sum-}\exp$ recursions over the same pseudoknot-free
structure space $\Omega$, computing the real Boltzmann-weighted
partition function

$$
Z = \sum_{s \in \Omega} e^{-E(s)/RT}
$$

and, from it, the real **base-pair probability** $p_{ij}$ — the
fraction of the thermal ensemble in which positions $i$ and $j$ are
actually paired:

$$
p_{ij} = \frac{1}{Z}\sum_{s \,:\, (i,j)\in s} e^{-E(s)/RT}
$$

This is directly useful, not merely a theoretical refinement. This
chapter's own hands-on project (Section 14.3) uses it twice: the **mean
base-pair distance** to the ensemble (`RNA.fold_compound.mean_bp_distance`,
ViennaRNA's real ensemble-diversity statistic) as a confidence-like
feature — a guide strand whose MFE structure is close to the entire
thermal ensemble is a confident, well-determined fold, while a large
mean distance flags a genuinely undetermined structure, exactly the
role pLDDT's *confidence*, not *accuracy*, framing played for protein
structures in Chapter 9 — and, more directly, the real target
**accessibility** calculation the next paragraph describes, the single
most important RNA-thermodynamic feature this chapter's own model
relies on.

**Target-site accessibility.** A target mRNA site that is already
single-stranded in its own local secondary structure is easy for RISC
(the RNA-induced silencing complex) to engage; a site buried inside a
stable stem-loop is not, regardless of how good the siRNA-target
duplex would be in isolation — an established, real determinant of
siRNA potency RNAup (Mückstein et al., 2006) was built specifically to
quantify. This chapter's own `target_accessibility_features()`
(`sirna_efficacy.py`) computes the same real quantity RNAup does, by
the most direct real route ViennaRNA's Python bindings expose: fold the
57-nt real local genomic context once unconstrained (real MFE
$E_{\text{free}}$), fold it again with a real hard constraint
(`fold_compound.hc_add_up`) forcing every position in the central 19-nt
target window to be unpaired (real MFE $E_{\text{open}}$), and report
the real, non-negative energetic cost of opening that window,

$$
\Delta G_{\text{open}} = E_{\text{open}} - E_{\text{free}} \;\ge\; 0 .
$$

```python
def target_accessibility_features(extended_target: str) -> dict:
    fc_free = RNA.fold_compound(extended_target)
    _s_free, mfe_free = fc_free.mfe()

    fc_open = RNA.fold_compound(extended_target)
    for i in range(TARGET_WINDOW.start + 1, TARGET_WINDOW.stop + 1):  # 1-indexed
        fc_open.hc_add_up(i)
    _s_open, mfe_open = fc_open.mfe()

    return {
        "target_unconstrained_mfe": round(mfe_free, 3),
        "target_opening_energy": round(mfe_open - mfe_free, 3),
    }
```

A concrete, real example from this chapter's own cached data (Section
14.3): the real Huesken siRNA `GGAAGGUGAUGCUUAUAUU`, targeting its real
57-nt local context, has an unconstrained MFE of $-7.8\,\text{kcal/mol}$
and a constrained (target-window-forced-open) MFE of
$-1.5\,\text{kcal/mol}$ — an opening energy of $6.3\,\text{kcal/mol}$,
a real, substantial accessibility penalty (its measured knockdown
efficiency, on this dataset's real $[0,1]$ scale, is a modest $0.39$).
By contrast, `UUAGUGAGAAUUCUGCAAC`, a real siRNA measured at $0.91$
efficiency, has an opening energy of only $4.9\,\text{kcal/mol}$ — a
real, less-structured, more accessible target site, consistent with,
though not proof of, the accessibility hypothesis; Section 14.3's own
feature-importance results (not this one illustrative pair) are the
real, quantitative test of that hypothesis on the full dataset.

### 14.1.4 RNA tertiary (3D) structure prediction: real tools and an honest scope decision

RNA 3D structure prediction has progressed substantially over the past
two decades but has not reached anything resembling AlphaFold's
transformation of protein structure prediction. **FARFAR2** (Watkins,
Rangan, & Das, 2020, building on Das & Baker's 2007 original FARFAR),
Rosetta's fragment-assembly RNA 3D predictor, samples candidate
tertiary folds by recombining short structural fragments drawn from
known RNA structures and scoring them with a real, physics-informed
energy function. **ARES** (Townshend et al., 2021) took a different,
now-familiar approach — a real, rotation-equivariant graph neural
network (the same $E(3)$-equivariance principle Chapter 6 §6.4
introduced for small-molecule 3D networks) trained to re-score and
rank FARFAR-generated candidate structures directly against real,
experimental accuracy, rather than a physics-based energy alone.
More recently, end-to-end deep-learning predictors have emerged
directly in AlphaFold's own methodological lineage: **RoseTTAFoldNA**
(Baek et al., 2024), extending the RoseTTAFold protein-structure
architecture to jointly model protein-nucleic-acid complexes;
**trRosettaRNA** (Wang et al., 2023), a transformer-network RNA 3D
predictor; and **DRfold** (Li et al., 2023), which integrates learned
geometric potentials with end-to-end structure generation.

Despite this real progress, independent, blinded benchmarking — most
visibly the RNA-Puzzles community trials and RNA's first inclusion as
a CASP15 prediction category — has consistently found RNA 3D structure
prediction accuracy well behind protein structure prediction's own
CASP14/AlphaFold2-era benchmarks, for reasons intrinsic to the
problem rather than to any one tool: RNA's tertiary contacts (coaxial
stacking, ribose-zipper and A-minor motifs, long-range tertiary base
pairs) are individually weaker and more conformationally flexible than
a folded protein's hydrophobic core, and the training data available —
the Protein Data Bank's own RNA-only tertiary structures — is a much
smaller, more repetitive corpus (dominated by rRNA, tRNA, and a
handful of riboswitch/ribozyme families) than the protein structural
universe Chapter 9's AlphaFold2/ESMFold training sets drew on.

This chapter makes the same kind of honest, disclosed scoping decision
Chapter 11 made for DiffDock (§11.2) and Chapter 13 made for its own
library size (§13.3): none of FARFAR2, ARES, RoseTTAFoldNA,
trRosettaRNA, or DRfold is a lightweight, pip-installable,
CPU-tractable tool the way ESMFold's single forward pass was for
proteins in Chapter 9 — each requires either a full Rosetta
installation and fragment-library infrastructure, or multi-gigabyte
learned-model weights with a recommended GPU runtime, neither of which
fits this book's own free-tier-Colab-compatible, pip-first authoring
constraint. Section 14.3's hands-on project therefore works entirely at
the secondary-structure and duplex-thermodynamics level established
in Sections 14.1.2-14.1.3, where ViennaRNA provides real, complete,
lightweight, exactly-reproducible tools — the same "go where the real,
tractable tooling is, and disclose why" principle this book has applied
consistently since Chapter 11.

## 14.2 mRNA & siRNA Optimization

A therapeutic RNA sequence is rarely used exactly as first designed
against its biological target; it is further optimized along three
largely separable real axes — translation/expression (for a coding
mRNA), stability, and immunogenicity — each with its own established
design principles.

### 14.2.1 Codon optimization

The genetic code is degenerate: 61 sense codons encode only 20 amino
acids, so a coding sequence for a fixed protein has enormous real
sequence freedom at the nucleotide level, with no effect on the
translated protein sequence itself. **Codon optimization** exploits
that freedom, but not merely by maximizing the frequency-weighted
"codon adaptation index" (CAI) against a host organism's own codon
usage table, as older heuristics did — Presnyak et al. (2015)
established a real, mechanistic reason codon choice matters beyond
translation speed alone: codon optimality is a major, direct
determinant of **mRNA stability** itself, because slower-translating
(non-optimal) codons trigger co-translational recruitment of the
Ccr4-Not deadenylation complex, coupling translation elongation rate
directly to mRNA decay. A therapeutic mRNA's codon choice therefore
trades off translation efficiency, mRNA half-life, and (Section
14.1.2) the resulting sequence's own secondary-structure content, since
synonymous codon substitutions change local base-pairing potential even
though they don't change the protein.

**LinearDesign** (Zhang et al., 2023) is a real, published algorithm
that optimizes exactly this joint trade-off directly and efficiently:
rather than optimizing codon choice and mRNA secondary structure as two
separate steps, it formulates a single joint objective balancing
translation efficiency (via CAI) against real, ViennaRNA-style
minimum free energy (lower — i.e., more negative and more structured —
being generally *less* desirable for coding-sequence stability and
translation, the same "less intramolecular structure is generally more
favorable" direction Section 14.1.2's discussion already established
for a guide strand's own self-structure), and solves it with a real,
polynomial-time lattice-parsing algorithm rather than a naive
exponential search over the codon space. The paper reports this
approach applied directly to real SARS-CoV-2 spike-protein mRNA vaccine
design, reducing predicted mRNA structuredness and improving measured
stability and expression relative to unoptimized or CAI-only-optimized
sequences — a real, deployed-scale demonstration that the
thermodynamic tools of Section 14.1 are not confined to siRNA design.

### 14.2.2 Stability enhancement

Beyond coding-sequence codon choice, an mRNA's stability and
translation efficiency depend heavily on its untranslated regions
(UTRs), 5' cap, and poly(A) tail — and, again, on secondary structure,
now considered across the *entire* transcript rather than just the
coding sequence. Leppek et al. (2022) ran a real, large-scale
crowdsourced RNA design challenge (Eterna's "OpenVaccine" project) that
directly, empirically tested this relationship at scale: real,
measured in-solution stability across thousands of real,
synthesized mRNA constructs correlated strongly with real,
computed secondary-structure metrics, with unstructured 5' regions in
particular associated with both higher stability and higher
translation — the same accessibility logic Section 14.1.3 introduced
for a *target* site applies, in a different but related sense, to a
therapeutic mRNA's own 5' UTR as the site ribosomal scanning must
traverse. This is a second, independent, real line of evidence
(alongside LinearDesign) that the exact real ViennaRNA-computable
quantities Section 14.1 introduces — MFE, base-pair probability,
local structuredness — are not an academic exercise but a load-bearing
part of how real, deployed RNA therapeutics are actually designed.

### 14.2.3 Immunogenicity reduction

Exogenous RNA delivered into a human cell is not immunologically
inert: pattern-recognition receptors — endosomal Toll-like receptors
TLR3, TLR7, and TLR8, and cytoplasmic sensors including RIG-I — evolved
specifically to detect foreign RNA and trigger a real, potent innate
immune (type I interferon) response, which for a therapeutic mRNA or
siRNA is an unwanted side effect rather than the intended mechanism.
Karikó, Buckstein, Ni, & Weissman (2005) established the real,
foundational finding that this recognition is suppressed by naturally
occurring nucleoside modifications: RNA synthesized with
**pseudouridine** or **N1-methylpseudouridine** in place of standard
uridine is dramatically less immunostimulatory via TLR3/7/8 while
remaining translatable — the specific real chemical modification that
underlies the mRNA vaccine platforms subsequently deployed at global
scale, work recognized by the 2023 Nobel Prize in Physiology or
Medicine. siRNA therapeutics address the analogous problem with a
different, complementary set of real, established chemistries:
2'-O-methyl and 2'-fluoro ribose modifications and phosphorothioate
backbone linkages, which simultaneously reduce innate-immune
recognition and increase nuclease resistance. Delivery is a separate,
real design axis on top of that chemistry: patisiran, the first
FDA-approved siRNA drug, uses a lipid-nanoparticle carrier, while the
liver-targeted siRNA drugs approved since (givosiran, lumasiran,
inclisiran) instead conjugate the siRNA directly to
N-acetylgalactosamine (GalNAc) for receptor-mediated hepatocyte
uptake. None of these delivery/chemistry questions is addressed
computationally in Section 14.3's hands-on project, which — like
Chapter 11's docking-only scope or Chapter 12's ligand-alone MD scope —
targets one real, well-defined, computationally tractable sub-problem
(sequence-level knockdown efficacy) rather than the full therapeutic
design pipeline.

## 14.3 Hands-on Project: Predicting siRNA Knockdown Efficacy from Real RNAi Data and Real RNA Thermodynamics

The project code lives in this chapter's folder
(`ch14_rna_therapeutics/`). Given Section 14.1's toolbox and Section
14.2's design principles, this project builds a real, complete siRNA
efficacy predictor: real measured silencing data in, real
self-computed thermodynamic and sequence features, real trained
models, real held-out evaluation.

### Real data and its provenance, disclosed in full

**The data.** Huesken et al. (2005) profiled 19-mer siRNAs against a
dual-luciferase reporter system in H1299 cells — the field's original
large-scale siRNA efficacy dataset, and still the most widely used
benchmark for this exact task two decades later (every rule-based and
learned siRNA design tool published since, including this chapter's
own, is implicitly or explicitly compared against it). The paper's own
original data hosting — Novartis' BIOPREDsi web server — is no longer
online, a real, disclosed constraint this chapter shares with several
of Chapter 13's own real 20-year-old fragment-screening data-access
questions. `sirna_efficacy.py` fetches the real sequences and real
measured efficiency values live from `github.com/dimostzim/siRBench`,
a small, actively maintained (as of this chapter's writing) community
redistribution assembled specifically because the original hosting
disappeared; it carries no explicit open-source license. This chapter
handles that honestly and conservatively rather than ignoring it:
**only four raw, factual fields** are taken from that source — the
19-nt siRNA sequence, its 57-nt local target-mRNA context, the real
measured efficiency value, and the source-study tag used to filter to
the real Huesken-only subset (2,133 train + 228 test sequences, curated
to zero sequence overlap between the two splits) — individual published
scientific facts (a sequence, a number), not the redistribution's own
compiled/engineered feature columns, none of which this chapter reuses.
Every feature this chapter's own model trains on (below) is computed
fresh, by this chapter's own code, from those four raw fields, via
ViennaRNA's real thermodynamic engine.

**A real data-quality check, not assumed.** `curate_huesken_subset()`
verifies, for every row, that the real 57-nt extended target context
actually contains the real 19-nt target site at its documented [19:38]
offset before accepting the row — a basic real consistency check
against transcription/formatting errors in a third-party redistribution,
following the same "verify, don't trust, a bundled real dataset"
discipline Chapter 13's own replicate-well deduplication bug and fix
established (§13.3).

### Real, self-computed features

Fourteen real features are computed per sequence, none of them
label-derived: sequence composition (overall GC content; seed-region
[positions 2-8] GC content, the established RNAi "seed" window; 5'/3'
terminal base identity; a count of internal repeated 4-mers, Reynolds
et al.'s 2004 internal-repeat liability); a compact, four-criterion
`design_heuristic_score` — inspired by, but not a literal reproduction
of, the qualitative design principles in Reynolds et al. (2004) and
Ui-Tei et al. (2004), used here purely as this section's rule-based
baseline; and, from Section 14.1's own real ViennaRNA machinery, the
guide strand's own self-structure MFE and ensemble diversity, the full
siRNA-target duplex hybridization energy and its 5'-end-vs-3'-end
terminal-stability asymmetry (`RNA.duplexfold`, the real, established
biophysical basis of RISC strand-selection bias; Khvorova, Reynolds, &
Jayasena, 2003; Schwarz et al., 2003), and the target site's own real
accessibility (unconstrained MFE and opening energy, Section 14.1.3).

```python
def duplex_features(sirna: str, target_site: str) -> dict:
    full = RNA.duplexfold(sirna, target_site)
    five_prime = RNA.duplexfold(sirna[:6], target_site[-6:])
    three_prime = RNA.duplexfold(sirna[-6:], target_site[:6])
    return {
        "duplex_energy_total": round(full.energy, 3),
        "duplex_energy_5p_end": round(five_prime.energy, 3),
        "duplex_energy_3p_end": round(three_prime.energy, 3),
        "duplex_end_asymmetry": round(five_prime.energy - three_prime.energy, 3),
    }
```

### Real models and real, held-out evaluation

Four real predictors are trained on the real 2,133-sequence training
split and evaluated, once, on siRBench's own real, pre-defined,
sequence-disjoint 228-sequence Huesken test split: the rule-based
`design_heuristic_score` alone; a Random Forest (500 trees); an
XGBoost gradient-boosted model; and a small PyTorch MLP
(19→64→32→1, matching the same modeling family — a feed-forward neural
network — Huesken et al. themselves used, on this chapter's own,
fully disclosed feature set rather than their original proprietary
one), with early stopping on a 15% real validation split carved from
the training data only, never touching the test set.

| Model | Spearman ρ | Pearson r | R² | RMSE | MAE |
|---|---|---|---|---|---|
| Rule-based score alone | 0.267 (p = 4.5×10⁻⁵) | 0.266 | — | — | — |
| Random Forest | **0.583** (p < 10⁻⁶) | **0.567** | **0.316** | **0.131** | **0.102** |
| XGBoost | 0.573 (p < 10⁻⁶) | 0.554 | 0.298 | 0.133 | 0.103 |
| PyTorch MLP | 0.543 (p < 10⁻⁶) | 0.522 | 0.260 | 0.136 | 0.106 |

(R²/RMSE/MAE are omitted for the rule-based score: it is a bounded,
0-4 integer count, not a value calibrated to the real efficiency
scale, so those absolute-error metrics are not meaningful for it —
Spearman/Pearson rank/linear correlation are the real, comparable
quantities across all four rows.)

**Reading the real result.** All three learned models comfortably and
significantly outperform the rule-based baseline (Spearman ρ around
0.54-0.58 versus 0.267) — a real, substantial gain from letting a model
learn the real, quantitative relationship between the thermodynamic and
compositional features directly, rather than combining them through
four hand-picked binary criteria. The **Random Forest is the strongest
model on every real metric** measured, ahead of both XGBoost and the
PyTorch MLP — a genuinely modest, believable margin (not one this
chapter over-interprets as "tree ensembles are better than neural
networks for this problem" from a single train/test split), consistent
with the real, well-established finding in the broader siRNA-efficacy
literature that this problem — a few dozen well-engineered
biophysical/compositional features against a noisy experimental
readout from a single assay platform — sits in a regime where gradient
boosting and random forests are frequently competitive with or ahead of
small neural networks, unlike the large-scale sequence/graph problems
(Chapters 6-10) where deep learning's advantage is much larger. An
$R^2$ around 0.30 and Spearman ρ around 0.58 are real, modest,
literature-consistent numbers for this exact task on this exact
dataset — well above the rule-based floor and well above chance, but
not the near-perfect predictor a compact, fourteen-feature model on a
noisy single-cell-line luciferase-reporter assay was ever going to
produce.

**Feature importances confirm established biology, not an artifact.**
Random Forest's single most important feature, by a wide margin, is
`pos1_is_AU` (importance 0.28 of 1.0 total; XGBoost ranks the same
feature even higher, at 0.45) — whether the guide strand's 5'-terminal
base is A or U. This is not a novel finding; it is this chapter's own,
independent, real quantitative confirmation of the single
best-established qualitative rule in the RNAi design literature
(Khvorova et al., 2003; Schwarz et al., 2003; Ui-Tei et al., 2004): a
weakly paired (A/U) 5' terminus lowers that end's local duplex
stability, biasing RISC to load the intended guide strand rather than
the passenger strand as the functional antisense species. Seeing this
exact, decades-old rule re-emerge as the dominant learned feature from
a real, independent, from-scratch feature-engineering-and-training
pipeline is a meaningful real sanity check on this section's whole
approach, in the same spirit as Chapter 11's redocking-validation
control or Chapter 13's PAINS/QED sanity checks on drug-like chemistry:
a model that had *not* rediscovered this relationship would be the
result worth distrusting. The `duplex_energy_total` and
`target_unconstrained_mfe`/`target_opening_energy` features — Section
14.1.3's real accessibility machinery — cluster immediately behind it
in both models' importance rankings, a real, quantitative
confirmation that target-site accessibility genuinely carries
independent predictive signal on this dataset, not merely a
theoretically motivated feature that turned out not to matter in
practice.

### Limitations and what comes next

This project builds a real, complete, held-out-evaluated siRNA
efficacy predictor from real published silencing data and real,
independently verifiable RNA thermodynamics — but several real,
disclosed limitations bound how far its conclusions generalize. First,
the Huesken dataset itself covers only 34 real target genes; because
the redistributed data this chapter fetches carries no gene/transcript
identifier, the sequence-disjoint train/test split used here cannot
also be gene-disjoint, so some of the reported performance may reflect
partial memorization of a given transcript's own local
structure/accessibility profile rather than purely sequence-generalizable
signal — the same category of leakage concern Chapter 5's scaffold
split was built specifically to rule out for small-molecule QSAR, left
here as an open, disclosed gap rather than a silently assumed
non-issue. Second, every measurement in this dataset comes from a
single assay platform (dual-luciferase reporter, H1299 cells); Section
14.3's model has not been tested against knockdown measured by direct
mRNA quantification, a different cell type, or a chemically modified
(2'-OMe/2'-F, Section 14.2.3) siRNA, any of which could shift the
real, learned feature-efficacy relationships. Third, the training data
source itself is an unlicensed, community-maintained redistribution of
a two-decade-old dataset — disclosed in full above, and mitigated by
using only the raw, factual (sequence, efficiency) fields and
independently verifying their internal consistency, but not eliminated
as a provenance caveat. Chapter 15 shifts the book's focus once more,
from RNA's own real biophysics to a different real cross-cutting
question: how physical conservation laws and quantum-mechanical
electronic-structure methods can be incorporated directly into the
learned models Chapters 1-14 have each built from a different real
angle.

## References

- Huesken, D., Lange, J., Mickanin, C., Weiler, J., Asselbergs, F.,
  Warner, J., Meloon, B., Engel, S., Rosenberg, A., Cohen, D., Labow,
  M., Reinhardt, M., Natt, F., & Hall, J. (2005). Design of a
  genome-wide siRNA library using an artificial neural network.
  *Nature Biotechnology*, 23(8), 995-1001.
  https://doi.org/10.1038/nbt1118
- Reynolds, A., Leake, D., Boese, Q., Scaringe, S., Marshall, W. S., &
  Khvorova, A. (2004). Rational siRNA design for RNA interference.
  *Nature Biotechnology*, 22(3), 326-330.
  https://doi.org/10.1038/nbt936
- Ui-Tei, K., Naito, Y., Takahashi, F., Haraguchi, T., Ohki-Hamazaki,
  H., Juni, A., Ueda, R., & Saigo, K. (2004). Guidelines for the
  selection of highly effective siRNA sequences for mammalian and
  chick RNA interference. *Nucleic Acids Research*, 32(3), 936-948.
  https://doi.org/10.1093/nar/gkh247
- Khvorova, A., Reynolds, A., & Jayasena, S. D. (2003). Functional
  siRNAs and miRNAs exhibit strand bias. *Cell*, 115(2), 209-216.
  https://doi.org/10.1016/s0092-8674(03)00801-8
- Schwarz, D. S., Hutvágner, G., Du, T., Xu, Z., Aronin, N., & Zamore,
  P. D. (2003). Asymmetry in the assembly of the RNAi enzyme complex.
  *Cell*, 115(2), 199-208.
  https://doi.org/10.1016/s0092-8674(03)00759-1
- Lorenz, R., Bernhart, S. H., Höner zu Siederdissen, C., Tafer, H.,
  Flamm, C., Stadler, P. F., & Hofacker, I. L. (2011). ViennaRNA
  Package 2.0. *Algorithms for Molecular Biology*, 6, 26.
  https://doi.org/10.1186/1748-7188-6-26
- Mückstein, U., Tafer, H., Hackermüller, J., Bernhart, S. H., Stadler,
  P. F., & Hofacker, I. L. (2006). Thermodynamics of RNA-RNA binding.
  *Bioinformatics*, 22(10), 1177-1182.
  https://doi.org/10.1093/bioinformatics/btl024
- Zuker, M., & Stiegler, P. (1981). Optimal computer folding of large
  RNA sequences using thermodynamics and auxiliary information.
  *Nucleic Acids Research*, 9(1), 133-148.
  https://doi.org/10.1093/nar/9.1.133
- McCaskill, J. S. (1990). The equilibrium partition function and base
  pair binding probabilities for RNA secondary structure.
  *Biopolymers*, 29(6-7), 1105-1119.
  https://doi.org/10.1002/bip.360290621
- Turner, D. H., & Mathews, D. H. (2010). NNDB: the nearest neighbor
  parameter database for predicting stability of nucleic acid
  secondary structure. *Nucleic Acids Research*, 38(suppl_1), D280-D282.
  https://doi.org/10.1093/nar/gkp892
- Das, R., & Baker, D. (2007). Automated de novo prediction of
  native-like RNA tertiary structures. *Proceedings of the National
  Academy of Sciences*, 104(37), 14664-14669.
  https://doi.org/10.1073/pnas.0703836104
- Watkins, A. M., Rangan, R., & Das, R. (2020). FARFAR2: Improved de
  novo Rosetta prediction of complex global RNA folds. *Structure*,
  28(8), 963-976.e6. https://doi.org/10.1016/j.str.2020.05.011
- Townshend, R. J. L., Eismann, S., Watkins, A. M., Rangan, R.,
  Karelina, M., Das, R., & Dror, R. O. (2021). Geometric deep learning
  of RNA structure. *Science*, 373(6558), 1047-1051.
  https://doi.org/10.1126/science.abe5650
- Baek, M., McHugh, R., Anishchenko, I., Jiang, H., Baker, D., &
  DiMaio, F. (2024). Accurate prediction of protein-nucleic acid
  complexes using RoseTTAFoldNA. *Nature Methods*, 21(1), 117-121.
  https://doi.org/10.1038/s41592-023-02086-5
- Wang, W., Feng, C., Han, R., Wang, Z., Ye, L., Du, Z., Wei, H.,
  Zhang, F., Peng, Z., & Yang, J. (2023). trRosettaRNA: automated
  prediction of RNA 3D structure with transformer network. *Nature
  Communications*, 14, 7266.
  https://doi.org/10.1038/s41467-023-42528-4
- Li, Y., Zhang, C., Feng, C., Pearce, R., Freddolino, P. L., & Zhang,
  Y. (2023). Integrating end-to-end learning with deep geometrical
  potentials for ab initio RNA structure prediction. *Nature
  Communications*, 14, 5745.
  https://doi.org/10.1038/s41467-023-41303-9
- Presnyak, V., Alhusaini, N., Chen, Y.-H., Martin, S., Morris, N.,
  Kline, N., Olson, S., Weinberg, D., Baker, K. E., Graveley, B. R., &
  Coller, J. (2015). Codon optimality is a major determinant of mRNA
  stability. *Cell*, 160(6), 1111-1124.
  https://doi.org/10.1016/j.cell.2015.02.029
- Zhang, H., Zhang, L., Lin, A., Xu, C., Li, Z., Liu, K., Liu, B., Ma,
  X., Zhao, F., Jiang, H., Chen, C., Shen, H., Li, H., Mathews, D. H.,
  Zhang, Y., & Huang, L. (2023). Algorithm for optimized mRNA design
  improves stability and immunogenicity. *Nature*, 621(7978), 396-403.
  https://doi.org/10.1038/s41586-023-06127-z
- Leppek, K., Byeon, G. W., Kladwang, W., Wayment-Steele, H. K., Kerr,
  C. H., Xu, A. F., et al. (2022). Combinatorial optimization of mRNA
  structure, stability, and translation for RNA-based therapeutics.
  *Nature Communications*, 13, 1536.
  https://doi.org/10.1038/s41467-022-28776-w
- Karikó, K., Buckstein, M., Ni, H., & Weissman, D. (2005). Suppression
  of RNA recognition by Toll-like receptors: the impact of nucleoside
  modification and the evolutionary origin of RNA. *Immunity*, 23(2),
  165-175. https://doi.org/10.1016/j.immuni.2005.06.008

All feature-computation examples, model metrics, and feature-importance
values cited in Section 14.3 were computed directly by running
`sirna_efficacy.py` against the real, cached, curated Huesken
train/test splits on 2026-08-21, not taken from a secondary source —
see `results/sirna_efficacy_results.json` to reproduce.
