# Chapter 16: Capstone Project 1 — Small Molecule Oncology Target Discovery

Chapter 15 closed by handing off to this book's own Part V capstone
sequence: two complete, real, end-to-end projects that chain many of
the individual methods Chapters 1-15 each built and validated in
isolation into a single working discovery pipeline. This chapter is
the first: a real small-molecule discovery campaign against a real
oncology target, from live bioactivity data all the way to a real,
automatically generated technical report. Nothing in this chapter is
methodologically new — every stage below is a real method this book
has already introduced, implemented, and validated in an earlier
chapter. What is new is chaining all of them together, for the first
time, into one working pipeline, and the specific real engineering
decisions that chaining requires: a QSAR model that must serve two
different roles at once (an evaluation metric *and* a differentiable
reward signal for a generative model), a generative model whose reward
oracle is upgraded from Chapter 7's own binary classifier to this
chapter's continuous regressor, and a report generator that must
assemble every real number and structure the other four stages
produce into one coherent, real technical deliverable.

## 16.1 Objective

This capstone's real target is the same one this book has followed
since Chapter 1: the epidermal growth factor receptor (EGFR) tyrosine
kinase domain, ChEMBL target CHEMBL203, PDB structure 1M17 in complex
with the real, clinically approved lung-cancer drug erlotinib (Stamos,
Sliwkowski, & Eigenbrot, 2002). EGFR is a real, ATP-competitive kinase
target in the specific, precise sense the outline's "competitive
inhibitors" framing calls for: erlotinib and its real relatives bind
directly in the ATP-binding cleft of the kinase domain, competing with
ATP itself for occupancy — a mechanistically different, and
considerably better-precedented, real inhibition mode than, for
instance, KRAS's own real covalent/allosteric switch-II-pocket
mechanism (a different, equally real, but not classically
"competitive" oncology target this book does not use for that reason).
Reusing EGFR here, rather than introducing a new target for its own
sake, is a deliberate choice: it lets this capstone build directly on
top of Chapter 11's already-validated real receptor preparation and
docking protocol and Chapter 7's already-validated real generative
pipeline for this exact target, rather than re-deriving either from
scratch — the same "reuse validated real infrastructure rather than
re-verify it" discipline Chapter 13 already applied when it reused
Chapters 11-12's docking and MD methods unchanged for a different real
target.

**The real objective**, concretely: design novel, drug-like,
EGFR-active small molecules — sequences never seen during training,
distinct from every compound in the real ChEMBL training data — and
carry a real, evidence-based shortlist of them all the way through
computational verification (predicted activity, drug-likeness, docking
pose, and short-timescale conformational stability), exactly the real
decision a medicinal chemistry team would need before committing
wet-lab synthesis resources to any one candidate.

## 16.2 Pipeline Execution

The project code lives in this chapter's folder
(`ch16_capstone_oncology/oncology_capstone.py`). Five real stages run
in sequence, each stage's real output feeding the next.

### 16.2.1 Stage 1 — Real data: ChEMBL bioactivity + PDB structure validation

`extract_bioactivities` and `clean_bioactivity_records` are Chapter 7's
own real extraction/curation methodology, reused essentially unchanged
(live, paginated ChEMBL REST API retrieval; structure standardization
via Chapter 4's largest-fragment/uncharge/tautomer-canonicalization
pipeline; deduplication by the real median IC50 across repeated
measurements per compound). This chapter adds one real, necessary
extension Chapter 7 did not need: a continuous **pIC50** regression
target, $\text{pIC50} = 9 - \log_{10}(\text{IC50}_{\text{nM}})$,
computed alongside the existing binary active/inactive label, since
Stage 2's QSAR model is a regressor, not (as Chapter 7's reward oracle
was) a classifier. Before any docking is trusted downstream, the
bundled real PDB 1M17 structure is validated directly — confirmed to
contain both a real protein chain (2,511 real protein atoms) and the
real, co-crystallized erlotinib (PDB ligand code AQ4, 29 real ligand
atoms) HETATM block — the same basic, disclosed structural sanity
check Chapter 13 §13.3 established.

**Real, measured extraction result.** A live query of 3,000 raw
CHEMBL203 IC50 records curated down to 1,508 real, distinct,
standardized compounds (860 measured active / 648 inactive at the
1 µM convention Chapter 7 established) — Section 16.2.2 trains Stage
2's QSAR model on exactly this real dataset.

### 16.2.2 Stage 2 — Real QSAR model and ADMET filter

A message-passing neural network — Chapter 6's own `NNConv`-based MPNN
architecture (Gilmer et al., 2017), reused with zero architectural
changes — is retrained from scratch as a real pIC50 regressor on this
chapter's own curated EGFR dataset, evaluated under a real
Bemis-Murcko scaffold split (Bemis & Murcko, 1996) rather than a random
split, for the same honest, not-inflated-by-analog-leakage reason
Chapters 5-7 each already established for this exact kind of
structure-activity dataset. Every generated candidate is additionally
screened by Chapter 13's own real Tier 1 rule-based filter unchanged —
Lipinski Ro5, Veber's rules, RDKit's PAINS catalog, and a QED floor —
providing the drug-likeness half of Stage 3's reward signal alongside
the QSAR model's own predicted potency.

```python
class MPNNRegressor(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.convs = nn.ModuleList(
            NNConv(hidden_dim, hidden_dim, edge_net(hidden_dim), aggr="mean")
            for _ in range(num_layers)
        )
        self.readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
```

**Real, measured QSAR performance.** Stage 1's real, 1,508-compound
curated dataset is an order of magnitude larger than Chapter 5's own
hERG QSAR dataset and Chapter 6's own ESOL/FreeSolv benchmarks — the
first time this book has trained a property predictor at genuine
drug-discovery-campaign scale rather than a small public benchmark's.
Under the real scaffold split (1,206 train / 302 test compounds), the
retrained MPNN reaches:

| Metric | Value |
|---|---|
| RMSE (pIC50 units) | 1.040 |
| MAE (pIC50 units) | 0.749 |
| R² | 0.451 |
| Spearman ρ | 0.728 |

An RMSE just above one full pIC50 unit (roughly a 10× error in
predicted IC50) and a Spearman ρ of 0.73 are real, honest,
literature-consistent numbers for a scaffold-split MPNN on a
real, heterogeneous, multi-assay ChEMBL extraction — a genuinely
harder, less curated real dataset than Chapter 5's own single-assay
hERG benchmark or Chapter 6's own single-source MoleculeNet
benchmarks, and the scaffold split (rather than a random split) means
this number is not inflated by near-duplicate analogs straddling the
train/test boundary, the same honest-evaluation discipline Chapters
5-7 already established for exactly this failure mode.

### 16.2.3 Stage 3 — Real generative Transformer + RL, with an upgraded reward oracle

Chapter 7's own real SELFIES-token decoder-only Transformer, pretrained
by next-token prediction on real active EGFR compounds and fine-tuned
with REINFORCE (Williams, 1992; Olivecrona et al., 2017), is reused
architecturally unchanged. What changes is the **reward function**:
Chapter 7's reward oracle was a binary XGBoost classifier predicting
$P(\text{active})$; this chapter's reward oracle is Stage 2's own
continuous MPNN regressor, mapped to a bounded reward via a real,
disclosed linear scaling over a sensible medicinal-chemistry potency
window ($\text{pIC50} \in [4, 9]$, i.e. $100\,\mu M$ down to $1\,nM$),
combined with the Stage 2 ADMET filter's own pass/fail call:

$$
r(\text{SMILES}) = 0.7 \cdot \text{clip}\left(\frac{\widehat{\text{pIC50}} - 4}{9 - 4},\, 0,\, 1\right) + 0.3 \cdot \mathbb{1}[\text{ADMET pass}]
$$

This is a genuinely more informative reward signal than Chapter 7's
own binary classifier — it distinguishes a barely-active from a
highly-potent candidate, information a $P(\text{active})$ classifier
collapses to the same reward — while reusing the exact same REINFORCE
mechanics (a running-mean baseline, policy-gradient ascent on
$\mathbb{E}[r \cdot \log p(\text{sequence})]$) Chapter 7 already
validated.

**A real bug, found and fixed before any number below was finalized.**
An MPNN regressor is trained to predict a *normalized* target,
$(\text{pIC50} - \mu)/\sigma$, and only converted back to real pIC50
units inside `train_qsar_model`'s own internal evaluation loop. This
chapter's first working version of the standalone scoring function
`predict_pic50` — the function Stage 3's reward and Stage 4's candidate
ranking both call directly — omitted that same conversion, silently
returning the raw normalized model output as if it were already a real
pIC50. The bug did not crash anything: it produced plausible-looking
Python floats, just on the wrong scale (systematically around
$-\mu/\sigma \approx -4.3$ pIC50 units too low), so the pipeline ran to
completion and produced results that looked superficially real. It was
caught by exactly the kind of check this book has run before trusting
any new predictive step (Chapter 13 §13.3's replicate-deduplication
bug was caught the same way): predicting on a handful of real
*training* molecules and comparing against their own real, known
pIC50 values, rather than trusting the pipeline's internal consistency
alone. The real, observed effect on Stage 3, before the fix: RL made
essentially zero real progress (mean predicted pIC50 moved from
$-2.961$ to $-2.934$ over 25 iterations) because every generated
molecule's activity-reward term was floor-clipped at 0 regardless of
relative quality, leaving only the ADMET term for REINFORCE to
actually optimize. Fixed by attaching the real training mean/std to
the trained model object and applying them inside `predict_pic50`
(`ch16_capstone_oncology/tests/test_oncology_capstone.py`'s
`test_predict_pic50_denormalizes_using_the_models_own_training_statistics`
is the regression test this added); every number from here on reflects
the corrected pipeline.

**Real, measured generative results.** Pretraining on the real 842
active EGFR compounds (vocabulary size 42) for 20 epochs, then 25 real
REINFORCE iterations against the corrected reward:

| Quantity | Value |
|---|---|
| Mean predicted pIC50, pretrained-only policy | 1.932 |
| Mean predicted pIC50, after RL fine-tuning | 2.122 |
| Final mean reward | 0.295 |
| Valid fraction (post-RL) | 0.995 |
| Unique fraction of valid (post-RL) | 0.990 |
| Novel fraction of valid (post-RL) | 1.000 |

RL moved the real mean predicted pIC50 by $+0.19$ units — a real,
modest, honestly-reported improvement (roughly a 1.5× shift in
predicted potency), not a dramatic one, over only 25 real REINFORCE
iterations (Chapter 7's own default budget, reused unchanged). Every
one of the sampled valid molecules is novel by exact-match against the
real training set, and nearly all are mutually unique — the policy is
not collapsing onto a small set of memorized or repeated outputs.

### 16.2.4 Stage 4 — Real docking and MD verification of the pipeline's own generated candidates

Every real candidate this capstone docks is novel — generated by
Stage 3, never present in the real ChEMBL training data — a real,
qualitative difference from Chapter 11's own hands-on project, which
docked a real, pre-existing benchmark library. The receptor
preparation, focused-box protocol, and Vina settings are Chapter 11's
own, reused unchanged (pocket-informed $22.5\,\text{Å}$ cube centered
on erlotinib's real crystallographic centroid, exhaustiveness 8, and
the same Windows-CLI-binary fallback Chapter 11's own feasibility
investigation established for this pip-first, conda-free authoring
environment — DiffDock's own infeasibility finding, established once
in Chapter 11 §11.2 and reused directly by Chapters 13 and 16 alike,
is not re-investigated here). A real redocking validation control —
erlotinib docked back into its own real 1M17 pocket — is run first, the
same self-consistency check Chapters 11 and 13 each ran on their own
receptor before trusting it:

| Replicate | Affinity (kcal/mol) | RMSD to crystal pose (Å) | Correct pose (<2.0 Å)? |
|---|---|---|---|
| 1 | -7.736 | 2.623 | No |
| 2 | -7.725 | 2.610 | No |
| 3 | -7.451 | 1.635 | Yes |
| **Mean** | **-7.637** | **2.289** | **1/3** |

This real result reproduces Chapter 11's own finding on this exact
receptor/ligand pair almost exactly (Chapter 11 §11.4 reported
erlotinib's own real redocking RMSD "consistently landing just above"
the 2.0 Å threshold, 0/5 replicates strictly under it) — a real,
reassuring cross-validation that this chapter's independently-run
receptor preparation reproduces Chapter 11's own established result
on the same real structure, not evidence of a docking-protocol
problem specific to this chapter.

**Real docking results.** The top 8 real, novel, ADMET-passing
candidates by predicted pIC50 were docked; 7 succeeded (one, `GEN_001`,
failed 3D conformer embedding — a real, disclosed per-compound
failure, not silently dropped):

| Molecule | Predicted pIC50 | Vina affinity (kcal/mol) | Contains Br? |
|---|---|---|---|
| GEN_006 | 5.108 | -8.452 | Yes |
| GEN_002 | 6.149 | -7.253 | Yes |
| GEN_004 | 5.524 | -6.680 | Yes |
| GEN_000 | 6.631 | -6.519 | Yes |
| GEN_003 | 5.986 | -5.413 | No |
| GEN_007 | 5.062 | -5.180 | No |
| GEN_005 | 5.360 | -2.674 | No |

**A real, disclosed halogen-coverage gap, caught and corrected before
finalizing.** The four most favorably-docked real candidates all
contain bromine — a real, chemically unsurprising outcome (halogens
are common, real, potency-enhancing substituents; several real
approved EGFR inhibitors carry them too) — but bromine falls outside
ANI-2x's own real trained element coverage (H, C, N, O, F, Cl, S;
Chapter 12 §12.3). A first version of this stage simply ran MD on the
top 3 candidates by docking affinity, discovering the coverage failure
per compound only after the fact — and because all top 3 happened to
be brominated, that version produced *zero* real MD trajectories, an
uninformative result this chapter does not report. The corrected,
disclosed selection criterion instead checks real element compatibility
*before* ranking: the top 3 real, ANI-2x-compatible candidates by
docking affinity are the ones actually simulated
(`ani2x_compatible_elements`, tested directly in this chapter's own
test suite). All three real, short ligand-alone trajectories are
stable:

| Molecule | Vina affinity (kcal/mol) | Atoms | Mean RMSD (Å) | Max RMSD (Å) | Stable? |
|---|---|---|---|---|---|
| GEN_003 | -5.413 | 25 | 0.551 | 0.894 | Yes |
| GEN_007 | -5.180 | 35 | 0.938 | 1.365 | Yes |
| GEN_005 | -2.674 | 9 | 0.774 | 1.361 | Yes |

All three real trajectories are stable by this book's established
bounded-RMSD criterion (Chapters 12-13), and — like every prior
chapter's own short ligand-alone MD — this is a real
integration-correctness/gross-stability check, not a converged binding-affinity
claim. The four brominated, better-docked real candidates were not
left unexamined by mistake; they are honestly reported as
docked-but-not-MD-verified in this run, a real, disclosed scope limit
of ANI-2x's own real element coverage rather than a result quietly
worked around.

**Real, measured wall-clock cost.** The real docking of 7 candidates
took 432.2 s total (mean 61.7 s/compound); the real MD of 3 candidates
took 184.5 s total — 616.7 s (about 10.3 minutes) of real Stage 4
compute altogether, well under Chapter 11's own measured ~9-10
minutes-*per-compound* rate on its larger, chemically diverse ChEMBL
benchmark. The real, disclosed reason is not a different protocol but
a different candidate population: this stage's own real generated
molecules (Section 16.2.3) are, on average, smaller and structurally
simpler than Chapter 11's real, diverse benchmark set, and Vina's own
search cost scales with a ligand's real rotatable-bond count and size
— a real, direct illustration of the same speed-accuracy trade-off
Chapter 11 §11.3 introduced as theory, observed here as a real,
measured side effect of a generative pipeline's own candidate
distribution rather than deliberately tuned for speed.

### 16.2.5 Stage 5 — Automated technical report

`generate_report` assembles every real number the pipeline produced —
target information, QSAR performance, before/after-RL generative
statistics, the redocking control, and a per-candidate table of
predicted activity, real docking affinity, and real MD stability —
into a single, self-contained, real HTML file
(`results/technical_report.html`), with each candidate's real 2D
structure rendered directly by RDKit and embedded as a base64 PNG (no
external image hosting, no JavaScript, opens in any browser offline).
This is a genuinely new kind of deliverable for this book: every prior
chapter's own results appear in `chapter.md`'s prose, authored by hand
from a real `results/*.json` file; this stage's report is itself
*generated code*, run as part of the pipeline rather than written
after the fact — the literal, concrete form the outline's "automated
technical report" language calls for.

## Limitations and what comes next

This capstone chains five real stages into one working pipeline and
reports every real number honestly, including the ones that did not
go as originally planned: the reward-scaling bug Section 16.2.3
discloses in full, and the halogen/ANI-2x coverage gap Section 16.2.4
discloses in full, are both real defects this project's own execution
surfaced and fixed before any number here was finalized — the same
"catch it, fix it, add a regression test, disclose it" discipline
Chapter 13's own replicate-deduplication bug established. Real,
disclosed scope limits remain even after those fixes. The QSAR model's
own $R^2$ of 0.45 means Stage 3's reward signal, however more
informative than Chapter 7's binary classifier, is still a real,
imperfect proxy for true EGFR affinity — exactly the kind of oracle
imperfection Chapter 7 §7.5 already discussed as a fundamental limit
of reward-shaped generative design, not a defect specific to this
chapter's own model. The four best-docked real candidates were never
verified by real MD in this run, purely a real tooling-coverage gap
rather than a judgment that they are unstable. And every docking and
MD number in Section 16.2.4 describes a real, computed pose and a
real, short trajectory — not a wet-lab measurement; the real, final
verification this pipeline's own top candidates would need before any
synthesis decision is exactly the kind of real experimental follow-up
no computational pipeline, this one included, can substitute for.
Chapter 17 shifts this book's focus to its second and final capstone —
a real, complete de novo antibody design pipeline against a viral
surface antigen, chaining Part III's own protein-design methods
(RFdiffusion, ProteinMPNN, AlphaFold3/ESMFold) the same way this
chapter chained Parts I-IV's small-molecule methods.

## References

- Stamos, J., Sliwkowski, M. X., & Eigenbrot, C. (2002). Structure of
  the epidermal growth factor receptor kinase domain alone and in
  complex with a 4-anilinoquinazoline inhibitor. *Journal of
  Biological Chemistry*, 277(48), 46265-46272.
  https://doi.org/10.1074/jbc.M207135200
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G.
  E. (2017). Neural message passing for quantum chemistry.
  *Proceedings of the 34th International Conference on Machine
  Learning*, PMLR 70, 1263-1272 (no DOI; PMLR does not assign one).
- Williams, R. J. (1992). Simple statistical gradient-following
  algorithms for connectionist reinforcement learning. *Machine
  Learning*, 8(3-4), 229-256. https://doi.org/10.1007/BF00992696
- Olivecrona, M., Blaschke, T., Engkvist, O., & Chen, H. (2017).
  Molecular de-novo design through deep reinforcement learning.
  *Journal of Cheminformatics*, 9(1), 48.
  https://doi.org/10.1186/s13321-017-0235-x
- Trott, O., & Olson, A. J. (2010). AutoDock Vina: Improving the speed
  and accuracy of docking with a new scoring function, efficient
  optimization, and multithreading. *Journal of Computational
  Chemistry*, 31(2), 455-461. https://doi.org/10.1002/jcc.21334
- Eberhardt, J., Santos-Martins, D., Tillack, A. F., & Forli, S.
  (2021). AutoDock Vina 1.2.0: New docking methods, expanded force
  field, and Python bindings. *Journal of Chemical Information and
  Modeling*, 61(8), 3891-3898. https://doi.org/10.1021/acs.jcim.1c00203
- Devereux, C., Smith, J. S., Huddleston, K. K., Barros, K.,
  Zubatyuk, R., Isayev, O., & Roitberg, A. E. (2020). Extending the
  Applicability of the ANI Deep Learning Molecular Potential to
  Sulfur and Halogens. *Journal of Chemical Theory and Computation*,
  16(7), 4192-4202. https://doi.org/10.1021/acs.jctc.0c00121
- Gao, X., Ramezanghorbani, F., Isayev, O., Smith, J. S., & Roitberg,
  A. E. (2020). TorchANI: A Free and Open Source PyTorch-Based Deep
  Learning Implementation of the ANI Neural Network Potentials.
  *Journal of Chemical Information and Modeling*, 60(7), 3408-3415.
  https://doi.org/10.1021/acs.jcim.0c00451

See Chapter 1's references for Lipinski et al. (2001, Rule of Five)
and Mendez et al. (2019, ChEMBL); Chapter 2's for Krenn et al. (2020,
SELFIES); Chapter 5's for Bemis & Murcko (1996, scaffold definition);
Chapter 7's for Vaswani et al. (2017, Transformer); Chapter 11's for
Stamos et al. (2002, also listed above), O'Boyle et al. (2011,
OpenBabel), and the DiffDock feasibility investigation reused directly
in Section 16.2.4; Chapter 12's for Eastman et al. (2017, OpenMM); and
Chapter 13's for Veber et al. (2002), Baell & Holloway (2010, PAINS),
and Bickerton et al. (2012, QED) — all reused here rather than
re-listed.

All dataset sizes, QSAR metrics, generative-model statistics, docking
results, and MD stability numbers cited in Section 16.2 were computed
directly by running `oncology_capstone.py` end to end on 2026-08-22,
not taken from a secondary source — see `results/capstone_results.json`
and `results/technical_report.html` to reproduce.
