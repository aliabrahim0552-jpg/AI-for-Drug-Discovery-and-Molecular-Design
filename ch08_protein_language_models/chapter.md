# Chapter 8: Protein Language Models (pLMs) & Evolutionary Dynamics

Chapters 1-7 worked almost entirely with small molecules — SMILES
strings, molecular graphs, generated ligands. Part III turns to the
other half of this book's title: macromolecules. A protein sequence is,
formally, a string over a 20-letter alphabet, and Chapter 3 (Section
3.1) already introduced the classical tools for reading meaning out of
that string — substitution matrices, pairwise and multiple sequence
alignment. This chapter asks a different question: instead of
hand-designing what a sequence's letters mean, can a model *learn* it,
purely from being shown enormous numbers of real protein sequences and
asked to predict masked-out pieces of them? The answer — yes, well
enough to be scientifically useful with zero task-specific training —
is what makes **protein language models** (pLMs) the foundation for
every remaining chapter in Part III: Chapter 9's structure predictors
and Chapter 10's protein designers both build directly on the
representations this chapter introduces.

## 8.1 The Evolutionary Scale Hypothesis

Anfinsen's (1973) thermodynamic hypothesis — that a protein's amino
acid sequence alone determines its native 3D structure, with no
additional biological machinery required — is the foundational premise
of computational structural biology: everything Chapters 3, 9, and 10
do rests on it being true. It says the information is *there*, encoded
in the sequence. It says nothing about how to decode it.

Evolution has been running that decoding experiment for billions of
years. Every sequence in a modern protein family is a descendant of
some ancestral sequence, filtered through natural selection: mutations
that broke folding or function were eliminated, mutations that were
neutral or beneficial persisted. A **multiple sequence alignment**
(MSA) of a protein family — Chapter 3, Section 3.1's tool, revisited
here as promised — makes this filtering visible as statistical
structure: positions critical to folding or function stay highly
conserved across millions of years of divergence, while tolerant
positions drift freely; pairs of positions that are physically in
contact tend to **co-evolve**, because a destabilizing substitution at
one is often compensated by a correlated substitution at the other.
Classical unsupervised methods exploit exactly this signal from a
*single* family's MSA: EVcouplings-style direct coupling analysis
infers residue-residue contacts from co-evolving pairs, and
Riesselman et al.'s (2018) DeepSequence trains a variational
autoencoder on one family's alignment to model that family's implicit
fitness landscape, reporting zero-shot mutation-effect predictions
competitive with the individual assays it evaluates against — without
ever seeing experimental labels.

The **evolutionary scale hypothesis** — the premise this chapter's
namesake models are built on — is a scale-up of that same idea, with
one structural change: instead of an MSA of one family (typically
thousands of sequences, aligned to each other's exact positions),
train a single self-supervised model on hundreds of millions of
*unaligned* sequences spanning the entire observed protein universe at
once. Devlin et al. (2019) introduced the training objective this
depends on for language generally — **masked language modeling**
(MLM): corrupt a fraction of input tokens by replacing them with a
special mask token, and train the model to reconstruct them from
context,

$$
\mathcal{L}_{\text{MLM}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \sum_{i \in M} \log p_\theta\left(x_i \mid x_{\setminus M}\right)
$$

where $M$ is the randomly chosen set of masked positions in sequence
$x$ and $x_{\setminus M}$ denotes the full sequence with those
positions replaced by the mask token. Applied to protein sequences
with the 20-letter amino acid alphabet in place of a natural-language
vocabulary, Rives et al. (2021) showed this objective, trained at
sufficient scale (250 million UniParc sequences; up to 650 million
parameters), causes structural and functional information to emerge as
a *side effect* of the reconstruction task — with no structural label,
no alignment, and no family-specific input ever supplied — measurable
directly in the trained model's internal representations (Section 8.3
makes this concrete: masking a real position and reading off the
model's implied preferences over the 20 amino acids at that position
is precisely the reconstruction task the model was trained on, applied
at inference time as a scoring tool). This is the sense in which
"evolutionary scale" replaces "one family's MSA" as the source of
selective-pressure information: not because cross-family evolutionary
signal stops mattering, but because a big enough corpus of raw
sequences turns out to contain it implicitly, recoverable without ever
constructing an alignment.

## 8.2 pLM Families

**ESM-1b** (Rives et al., 2021) is a 650M-parameter Transformer
*encoder* (bidirectional self-attention, unlike Chapter 7's causal
decoder — every position attends to every other position, masked ones
included, which is exactly what reconstructing a masked position from
its full surrounding context requires) trained with the MLM objective
above on UniRef50. **ESM-2** (Lin et al., 2023) is the direct successor
used throughout this chapter's hands-on project: architecturally the
same encoder design, with three concrete changes — rotary positional
embeddings in place of ESM-1b's learned absolute ones, a training-data
switch to UniRef50 with UniRef90 sequence sampling for diversity, and,
most consequentially, a scale sweep spanning eight model sizes from 8M
to 15B parameters, trained on an identical recipe so that model size is
isolated as the one varying factor. Lin et al. (2023) report that
structure-prediction accuracy (measured via ESMFold, the structure
head trained on top of ESM-2's representations — Chapter 9's subject)
improves smoothly and predictably with scale across this entire sweep,
with no sign of saturating even at 15B parameters; Section 8.4 checks
whether the same scale-dependence shows up in this chapter's much
smaller zero-shot mutation-scoring task, across the 8M/35M/150M range
that fits comfortably in a free Colab CPU runtime.

**ProtTrans** (Elnaggar et al., 2022) takes a parallel path to the same
goal: rather than one architecture, it trains and benchmarks several
established NLP language model architectures — BERT, ALBERT, Electra,
T5, among others — on protein sequences (UniRef and BFD, up to 393
billion residues), each adapted only by swapping the vocabulary from
natural-language tokens to amino acids. ProtBERT, its BERT-architecture
member, is the most widely used single output of that effort, sharing
ESM-1b's encoder-plus-MLM design but trained with a different data
mixture and codebase. The three families make the same underlying bet
— self-supervised reconstruction at scale extracts biologically
meaningful structure — with different choices of architecture,
training corpus, and (for ESM-2 specifically) an explicit, deliberately
isolated study of scale itself; Section 8.4 uses ESM-2 because its
published scale sweep is exactly what a scale-comparison hands-on
project needs.

## 8.3 Biological Embeddings

A trained pLM offers two distinct things a downstream task can consume,
and this chapter's hands-on project uses both.

**Masked-marginal scoring.** Because the MLM objective already trains
the model to output $p_\theta(x_i \mid x_{\setminus \{i\}})$ — a full
probability distribution over the 20 amino acids at any position, given
everything else — that same forward pass, run once at inference time
with no gradient updates at all, can be turned directly into a
zero-shot mutation-effect predictor. Meier et al. (2021) formalize this
as the **masked-marginal** method: to score a substitution at position
$i$ from wild-type residue $x_i^{\text{wt}}$ to mutant residue
$x_i^{\text{mut}}$, mask position $i$ in the wild-type sequence, run one
forward pass, and take the log-odds ratio between the model's two
resulting log-probabilities,

$$
\Delta_i(x^{\text{mut}}) = \log p_\theta\left(x_i^{\text{mut}} \mid x_{\setminus \{i\}}\right) - \log p_\theta\left(x_i^{\text{wt}} \mid x_{\setminus \{i\}}\right)
$$

A positive $\Delta_i$ means the model finds the mutant residue *more*
plausible than the wild-type one at that position, given everything
else in the sequence — evolutionary/structural plausibility, not a
direct physical measurement, but one Meier et al. (2021) show
correlates with real experimental effects (stability, binding,
catalytic activity — the "function" named in this chapter's title)
across dozens of published deep mutational scanning (DMS) datasets,
with no per-dataset fine-tuning. Because every mutant at a given
position shares the exact same masked forward pass, scoring every
possible substitution at a position costs one forward pass, not one per
substitution — Section 8.4's implementation makes this explicit.

**Embeddings as features.** The same forward pass also exposes each
layer's per-residue hidden-state vectors — the model's internal,
continuous representation of "what this residue means in this
context," pooled (commonly by averaging across the sequence length,
Section 8.4's choice) into one fixed-length vector per protein when a
single summary representation is wanted. Unlike masked-marginal
scoring, which requires no downstream training at all, embeddings are
typically consumed as *features*: frozen or fine-tuned inputs to a
separately trained predictor (a small regression head, most commonly)
for a task the base pLM was never directly trained on. Section 8.4's
project uses raw embedding distance itself, without a trained head, as
a second, independent zero-shot signal — does moving further from the
wild-type embedding, in the representation space the model learned,
track with a *larger* functional effect, regardless of the effect's
direction?

Both angles generalize past the single case Section 8.4 runs end to
end. Meier et al.'s (2021) own benchmark spans binding, stability, and
fitness assays across dozens of proteins; Tsuboyama et al. (2023)
applied essentially the same masked-marginal principle at much larger
scale specifically for **thermal stability** ($\Delta\Delta G$ of
folding, not binding), mega-scale-experimentally measuring over
775,000 folding stability values for small domains and their point
mutants — the natural larger-scale extension of the "stability" half
of this chapter's title, cited here as a pointer for exactly that
reason rather than re-run in Section 8.4, whose scope stays fixed on
one complete, tractable mutant population end to end.

Finally, embeddings are not confined to sequence-level tasks. Chapter 9
picks the same ESM-2 representations back up for a different purpose
entirely: ESMFold (Lin et al., 2023, introduced in Section 8.2) attaches
a structure-prediction head directly on top of them, producing 3D
coordinates from a single sequence with no MSA search step at all — the
structural payoff of Section 8.1's "no alignment required" premise,
in contrast to AlphaFold2 (Jumper et al., 2021), which achieves its own
state-of-the-art accuracy by explicitly building and processing an MSA
inside the model (its Evoformer block), the architectural road this
chapter's family of models chose not to take.

## 8.4 Hands-on Project: Zero-Shot Single-Point Mutation Effect Prediction with ESM-2

The project code lives in this chapter's folder
(`ch08_protein_language_models/`) and puts Section 8.3's two scoring
methods to a direct, real test: given only a pretrained ESM-2 checkpoint
and a wild-type sequence — **no GB1-specific training of any kind** —
how well do its zero-shot scores track real, experimentally measured
mutation effects?

### Data: the complete GB1 single-mutant population

GB1, the B1 immunoglobulin-binding domain of Streptococcal protein G,
is a 56-residue domain whose binding to the Fc region of IgG is one of
the most extensively mutationally characterized protein interactions in
the literature. Olson et al. (2014) developed a yeast-display,
deep-sequencing assay to measure IgG-Fc binding fitness for variants at
four specific positions (39, 40, 41, 54); Wu et al. (2016) applied it
to the full combinatorial space of all 20 amino acids at all four
positions simultaneously — $20^4 = 160{,}000$ possible variants —
producing, among the double/triple/quadruple mutants their paper's own
epistasis analysis focuses on, the complete population of every
possible **single**-point mutant at these four positions: $4 \times 19
= 76$ variants, plus the wild type itself. That completeness matters
for this project specifically: `esm_variant_effect.py`'s hands-on
result below is not computed from a sample of single mutants, it is
computed from literally all of them, so there is no sampling variance
between "the 76 mutants used here" and "the single-mutant population."

[`esm_variant_effect.py`](esm_variant_effect.py) downloads this dataset
from the FLIP benchmark's public redistribution (Dallago et al., 2021)
of Wu et al.'s (2016) original data, parses each variant's 4-letter
code against the wild-type code (`VDGV`) to recover the mutated
position, wild-type residue, and mutant residue, and extracts the
56-residue GB1 domain sequence itself (the FLIP CSV's `sequence` column
appends an unrelated ~200-residue yeast-display fusion tag after
position 56, stripped here since it plays no role in GB1's own fold or
binding site).

```python
def _row_to_mutant(row: dict) -> SingleMutant | None:
    variant_code = row["Variants"]
    if variant_code == GB1_WT_VARIANT:
        return None
    diffs = [i for i in range(4) if variant_code[i] != GB1_WT_VARIANT[i]]
    idx = diffs[0]
    position = GB1_MUTATED_POSITIONS[idx]
    ...
```

### Scoring: masked marginals and embedding distance

`ESM2VariantScorer` wraps a Hugging Face `transformers` ESM-2 checkpoint
and implements both of Section 8.3's methods. Masked-marginal scoring
exploits the position-sharing efficiency named there directly: GB1's 76
single mutants touch only 4 distinct positions, so `score_single_mutants`
masks and scores each of those 4 positions exactly once — not once per
mutant — reusing each position's resulting 20-amino-acid log-probability
distribution for every mutant that shares it.

```python
def score_single_mutants(self, wt_sequence, mutants):
    positions = sorted({m.position for m in mutants})
    log_probs_by_position = {
        pos: self.masked_marginal_log_probs(wt_sequence, pos) for pos in positions
    }
    return np.array([
        log_probs_by_position[m.position][m.mut_aa] - log_probs_by_position[m.position][m.wt_aa]
        for m in mutants
    ])
```

`embed_sequences` runs one forward pass per full mutant sequence (76 of
them, plus wild type — mean-pooling excludes the `<cls>`/`<eos>` special
tokens explicitly rather than averaging over them) and
`evaluate_embedding_perturbation` compares each mutant's cosine distance
from the wild-type embedding against its experimental fitness deviation
from 1.0 (wild-type-normalized fitness).

### Running it and reading the results

```bash
cd ch08_protein_language_models
pip install -r requirements.txt
python esm_variant_effect.py --use-cached-raw
```

Run against three ESM-2 checkpoints spanning roughly a 20x parameter
range (8M / 35M / 150M), scoring all 76 single mutants with both
methods, Spearman rank correlation against real experimental fitness
came out as follows:

| Model | Params | Masked-marginal ρ (n=76) | p-value | Embedding-distance ρ | p-value |
|---|---|---|---|---|---|
| ESM-2 8M | 7.5M | 0.095 | 0.415 | 0.111 | 0.339 |
| ESM-2 35M | 33.5M | 0.184 | 0.112 | 0.087 | 0.453 |
| ESM-2 150M | 148.1M | 0.208 | 0.072 | **0.314** | **0.0057** |

Two things are worth reading precisely here, not just the top line.
First, **masked-marginal correlation increases monotonically with model
scale** (0.095 -> 0.184 -> 0.208) — directionally consistent with Lin et
al.'s (2023) scale-dependent accuracy trend (Section 8.2) — but at
$n=76$, none of the three individually clears the conventional $p <
0.05$ significance threshold; the 150M model comes closest ($p=0.072$).
This is reported as what it is: a suggestive trend, not a statistically
confirmed one, at this sample size. Second, the **embedding-distance
method tells a sharper story at the largest scale tested**: the 150M
model's correlation between representational shift and functional
deviation ($\rho=0.314$, $p=0.0057$) is the one result in this table
that clears significance outright, while the two smaller models do not
approach it. Restricting either analysis to the 27-mutant high-read-count
subset (`keep=True` in the source data, a sequencing-depth quality flag,
not a fitness-reliability one) produces correlations
scattered close to zero at every scale — an expected consequence of
cutting the sample size by nearly two-thirds, not a contradiction of
the full-population result above; it is not treated as a separate
finding here.

As a single concrete example: mutant `IDGV` (position 39, wild-type
valine to isoleucine) has real experimental fitness 1.446 — nearly 45%
*better* binding than wild type — and the 150M model's masked-marginal
score for V->I at position 39 is positive, correctly ranking it above
wild type; not every one of the 76 mutants is ranked this cleanly,
which is exactly what a $\rho \approx 0.2$ correlation, rather than
$\rho \approx 1.0$, means in practice.

Why is the correlation modest rather than strong? GB1's assay measures
one specific, engineered property — IgG-Fc binding affinity — at four
co-varied positions chosen because mutating them tunes that specific
interface, not because they are representative of general sequence
conservation. A masked-marginal score reflects general evolutionary/
structural plausibility learned from the training corpus at large; it
is not specifically trained to predict *this* binding interaction, and
Meier et al. (2021) report exactly this kind of heterogeneity across
their own multi-assay benchmark — some DMS datasets correlate strongly
with zero-shot pLM scores, others, including affinity-tuning assays at
a small number of interface positions, considerably less so. The honest
lesson Section 8.4 is built to teach is this one: zero-shot pLM scoring
is a genuinely useful, free, no-training-required signal, but not a
universal oracle — validating it against real data for the specific
property at hand, as done here, is the point, not an afterthought.

### Reproducibility

Dependencies are version-floored (`torch>=2.2`, `transformers>=4.40`,
`scipy>=1.10` in [`requirements.txt`](requirements.txt), validated
against torch 2.13.0 and transformers 5.15.1 on Python 3.12).
`data/gb1_single_mutants_sample.csv` bundles the real, complete
wild-type + 76-single-mutant population (extracted from the live FLIP
download on 2026-08-20), so the pipeline runs offline and
deterministically with `--use-cached-raw` — verified directly:
`--use-cached-raw` and a fresh live download of the full 149,361-row
dataset produce bit-identical Spearman correlations, since both extract
the same 77 rows. All three ESM-2 checkpoints total a few hundred MB on
first download from the Hugging Face Hub; with them cached, the full
run (3 models x 2 methods x 76 mutants) takes about a minute on a
single CPU core, no GPU required. The 16-test suite in
[`tests/test_esm_variant_effect.py`](tests/test_esm_variant_effect.py)
checks data parsing against the real bundled fixture (including one
spot-checked published fitness value), the download function against a
mocked in-memory zip, and both scoring methods' mechanics — including
the position-sharing efficiency claim above, checked directly by
counting calls — against a tiny, untrained ESM architecture built from
the real ESM-2 tokenizer vocabulary, so the suite never downloads a
pretrained checkpoint and completes in under a minute, fully offline.
`pip install -r requirements-dev.txt && pytest` reproduces all 16
results.

### Limitations and what comes next

This project deliberately stays inside the single-family, single-assay
case: one 56-residue domain, one binding readout, exhaustively
enumerated at only 4 positions. Extending it to genome-scale variant
effect prediction (Section 8.3's forward reference to Tsuboyama et al.,
2023) or to positions outside this specific interface would need a
different DMS dataset entirely, not a code change — the pipeline's
data-loading layer is specific to this CSV's schema, again a deliberate
scope decision named here rather than left implicit. The
masked-marginal method also stays firmly zero-shot throughout: nothing in this
chapter fits a regression head on top of ESM-2 embeddings against
GB1 labels, which — per Section 8.3 — is the more common way embeddings
get used in practice and would very likely outperform the zero-shot
scores reported above, at the cost of needing labeled training data for
the specific property being predicted, which is exactly the dependency
zero-shot scoring is valuable for not requiring. Finally, everything in
this chapter operates on sequence alone; Chapter 9 picks these same
ESM-2 representations back up to predict the 3D structures those
sequences fold into.

### A note on Google Colab

Colab's default runtime preinstalls `torch` but not `transformers` or
`scipy`; run `!pip install transformers scipy` in the first cell. No
GPU is required for the model sizes used here (8M-150M parameters);
larger ESM-2 checkpoints (650M and up) would benefit from one.

## References

- Anfinsen, C. B. (1973). Principles that govern the folding of protein
  chains. *Science*, 181(4096), 223-230.
  https://doi.org/10.1126/science.181.4096.223
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT:
  Pre-training of deep bidirectional transformers for language
  understanding. *Proceedings of the 2019 Conference of the North
  American Chapter of the Association for Computational Linguistics:
  Human Language Technologies*, Volume 1, 4171-4186.
  https://doi.org/10.18653/v1/N19-1423
- Riesselman, A. J., Ingraham, J. B., & Marks, D. S. (2018). Deep
  generative models of genetic variation capture the effects of
  mutations. *Nature Methods*, 15(10), 816-822.
  https://doi.org/10.1038/s41592-018-0138-4
- Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo,
  D., Ott, M., Zitnick, C. L., Ma, J., & Fergus, R. (2021). Biological
  structure and function emerge from scaling unsupervised learning to
  250 million protein sequences. *Proceedings of the National Academy
  of Sciences*, 118(15), e2016239118.
  https://doi.org/10.1073/pnas.2016239118
- Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N.,
  Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A.,
  Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A. (2023).
  Evolutionary-scale prediction of atomic-level protein structure with
  a language model. *Science*, 379(6637), 1123-1130.
  https://doi.org/10.1126/science.ade2574
- Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Wang, Y.,
  Jones, L., Gibbs, T., Feher, T., Angerer, C., Steinegger, M., Bhowmik,
  D., & Rost, B. (2022). ProtTrans: Toward understanding the language of
  life through self-supervised learning. *IEEE Transactions on Pattern
  Analysis and Machine Intelligence*, 44(10), 7112-7127.
  https://doi.org/10.1109/tpami.2021.3095381
- Meier, J., Rao, R., Verkuil, R., Liu, J., Sercu, T., & Rives, A.
  (2021). Language models enable zero-shot prediction of the effects of
  mutations on protein function. *Advances in Neural Information
  Processing Systems*, 34 (NeurIPS 2021). Preprint:
  https://doi.org/10.1101/2021.07.09.450648 (no separate NeurIPS
  proceedings DOI was found for this paper as of this writing, verified
  directly against Crossref rather than assumed; the bioRxiv preprint
  DOI is cited instead, its status disclosed here rather than implied
  otherwise).
- Tsuboyama, K., Dauparas, J., Chen, J., Laine, E., Mohseni Behbahani,
  Y., Weinstein, J. J., Mangan, N. M., Ovchinnikov, S., & Rocklin, G. J.
  (2023). Mega-scale experimental analysis of protein folding stability
  in biology and design. *Nature*, 620(7973), 434-444.
  https://doi.org/10.1038/s41586-023-06328-6
- Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M.,
  Ronneberger, O., Tunyasuvunakool, K., Bates, R., Zidek, A., Potapenko,
  A., Bridgland, A., Meyer, C., Kohl, S. A. A., Ballard, A. J., Cowie,
  A., Romera-Paredes, B., Nikolov, S., Jain, R., Adler, J., ... Hassabis,
  D. (2021). Highly accurate protein structure prediction with
  AlphaFold. *Nature*, 596(7873), 583-589.
  https://doi.org/10.1038/s41586-021-03819-2 — introduced here only as
  a forward reference and architectural contrast; covered in full in
  Chapter 9.
- Olson, C. A., Wu, N. C., & Sun, R. (2014). A comprehensive biophysical
  description of pairwise epistasis throughout an entire protein
  domain. *Current Biology*, 24(22), 2643-2651.
  https://doi.org/10.1016/j.cub.2014.09.072
- Wu, N. C., Dai, L., Olson, C. A., Lloyd-Smith, J. O., & Sun, R.
  (2016). Adaptation in protein fitness landscapes is facilitated by
  indirect paths. *eLife*, 5, e16965.
  https://doi.org/10.7554/eLife.16965
- Dallago, C., Mou, J., Johnston, K. E., Wittmann, B. J., Bhatnagar,
  N., Yang, J., Amini, A., Kim, P. M., & Yang, K. K. (2021). FLIP:
  Benchmark tasks in fitness landscape inference for proteins.
  Preprint: https://doi.org/10.1101/2021.11.09.467890 (no separate
  peer-reviewed-venue DOI was found for this paper as of this writing;
  its bioRxiv preprint DOI is cited, status disclosed as above).

See Chapter 7's references for Vaswani et al. (2017, the Transformer
architecture underlying every pLM in this chapter). RDKit, DeepChem,
and the ChEMBL/PDB APIs used in Chapters 1-7 play no role in this
chapter; ESM-2's model weights and tokenizer are distributed through
the Hugging Face Hub (`facebook/esm2_t6_8M_UR50D`,
`facebook/esm2_t12_35M_UR50D`, `facebook/esm2_t30_150M_UR50D`), and the
GB1 dataset is redistributed by the FLIP benchmark under an AFL-3
license (original data CC BY 4.0 via eLife), both confirmed directly
from their respective repositories rather than assumed.

All correlations, p-values, and the worked `IDGV` example cited in
Section 8.4 were computed directly by running `esm_variant_effect.py`
against the live-downloaded GB1 dataset on 2026-08-20, not taken from a
secondary source, and independently reproduced against the bundled
offline fixture — see `data/gb1_single_mutants_sample.csv` and
`results/esm2_gb1_results.json` to reproduce.
