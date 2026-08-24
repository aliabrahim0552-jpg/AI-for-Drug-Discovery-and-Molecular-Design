"""
Chapter 8 hands-on project: zero-shot single-point mutation effect
prediction with ESM-2 protein language models.

Extract: the complete GB1 (immunoglobulin-binding domain B1 of
  Streptococcal protein G) four-site combinatorial mutagenesis dataset
  of Wu et al. (2016, eLife), curated and redistributed by the FLIP
  benchmark (Dallago et al., 2021). Positions 39, 40, 41 and 54 were
  saturation-mutagenized (all 20 amino acids) and assayed for IgG-Fc
  binding fitness by yeast display + deep sequencing (Olson et al.,
  2014). The Hamming-distance-1 subset (76 rows) is the *complete*
  population of single-point mutants at these four sites -- not a
  sample -- so results below are exact, not estimates from a subsample.
Predict: ESM-2 (Lin et al., 2023) protein language models, used with
  zero supervision (no training on GB1 data at all), score each
  mutation two ways:
    1. Masked-marginal log-odds (Meier et al., 2021): mask the mutated
       position in the wild-type sequence, read the model's log-
       probability of the mutant vs. wild-type residue at that
       position from a single forward pass.
    2. Embedding perturbation: mean-pooled per-residue embeddings
       (Rives et al., 2021) for the full mutant sequence, compared to
       the wild-type embedding by cosine distance.
Evaluate: Spearman rank correlation between each zero-shot score and
  the real experimental fitness measurement, for models of three
  different ESM-2 scales (8M / 35M / 150M parameters), replicating in
  miniature the scale-dependent zero-shot accuracy trend reported in
  Lin et al. (2023) and Meier et al. (2021).

See README.md for usage and chapter.md Section 8.4 for context.
"""
import argparse
import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
from scipy.stats import spearmanr
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

GB1_DATA_URL = "https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1/four_mutations_full_data.csv.zip"
GB1_ZIP_MEMBER = "four_mutations_full_data.csv"
GB1_WT_VARIANT = "VDGV"  # wild-type residues at positions 39, 40, 41, 54
GB1_MUTATED_POSITIONS = (39, 40, 41, 54)  # 1-indexed residue positions, Wu et al. (2016)
GB1_DOMAIN_LENGTH = 56  # canonical GB1 domain; the FLIP "sequence" column appends an unrelated yeast-display fusion tag after this
DEFAULT_TIMEOUT = 30

ESM2_MODEL_NAMES = (
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
)
DEFAULT_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class SingleMutant:
    variant_code: str  # 4-letter FLIP code, e.g. "IDGV"
    position: int  # 1-indexed residue position in the GB1 domain
    wt_aa: str
    mut_aa: str
    sequence: str  # full GB1 domain sequence (length GB1_DOMAIN_LENGTH) carrying the mutation
    fitness: float  # experimental binding fitness, WT normalized to 1.0
    keep: bool  # FLIP's high-confidence-read-count flag


# --------------------------------------------------------------------------
# Extract: GB1 deep mutational scanning data
# --------------------------------------------------------------------------


def download_gb1_dataset(dest_path: Path, timeout: int = DEFAULT_TIMEOUT) -> Path:
    """Download the full Wu et al. (2016) GB1 landscape (149,361 variants,
    curated by FLIP) and save the extracted CSV to dest_path."""
    response = requests.get(GB1_DATA_URL, timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_bytes = zf.read(GB1_ZIP_MEMBER)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(csv_bytes)
    return dest_path


def _row_to_mutant(row: dict) -> SingleMutant | None:
    """Return None for the wild-type row itself; a SingleMutant for any
    of the 76 Hamming-distance-1 variants; raises for anything else
    (this function should only ever be called on HD in {0, 1} rows)."""
    variant_code = row["Variants"]
    if variant_code == GB1_WT_VARIANT:
        return None
    diffs = [i for i in range(4) if variant_code[i] != GB1_WT_VARIANT[i]]
    if len(diffs) != 1:
        raise ValueError(f"Expected exactly one substitution vs. WT, got {diffs} for {variant_code!r}")
    idx = diffs[0]
    position = GB1_MUTATED_POSITIONS[idx]
    wt_aa = GB1_WT_VARIANT[idx]
    mut_aa = variant_code[idx]
    sequence = row["sequence"][:GB1_DOMAIN_LENGTH]
    if sequence[position - 1] != mut_aa:
        raise ValueError(f"Sequence/variant-code mismatch at position {position} for {variant_code!r}")
    return SingleMutant(
        variant_code=variant_code,
        position=position,
        wt_aa=wt_aa,
        mut_aa=mut_aa,
        sequence=sequence,
        fitness=float(row["Fitness"]),
        keep=row["keep"] == "True",
    )


def load_single_mutants(csv_path: Path) -> tuple[str, list[SingleMutant]]:
    """Parse the GB1 CSV (full 149,361-row download or the bundled 77-row
    fixture -- both share the same schema) into the wild-type GB1 domain
    sequence and the complete list of 76 single-point mutants."""
    wt_sequence = None
    mutants = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["HD"] not in ("0", "1"):
                continue
            if row["Variants"] == GB1_WT_VARIANT:
                wt_sequence = row["sequence"][:GB1_DOMAIN_LENGTH]
                if row["HD"] != "0":
                    raise ValueError("Wild-type row must have HD == 0")
                continue
            if row["HD"] != "1":
                continue
            mutants.append(_row_to_mutant(row))
    if wt_sequence is None:
        raise ValueError(f"Wild-type row (Variants == {GB1_WT_VARIANT!r}) not found in {csv_path}")
    if len(mutants) != 76:
        raise ValueError(f"Expected the complete set of 76 single-point mutants, found {len(mutants)}")
    return wt_sequence, mutants


# --------------------------------------------------------------------------
# Predict: ESM-2 zero-shot scoring
# --------------------------------------------------------------------------


class ESM2VariantScorer:
    """Wraps a pretrained ESM-2 checkpoint for zero-shot mutation scoring.
    No weights are updated anywhere in this class -- every score is
    computed from the publicly released, GB1-naive pretrained model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        tokenizer=None,
        mlm_model=None,
        embedding_model=None,
    ):
        """tokenizer/mlm_model/embedding_model can be injected directly
        (used by the test suite to exercise this class against a tiny
        untrained model, with no network access or multi-hundred-MB
        checkpoint download required); production code should leave
        them unset and let model_name resolve the real pretrained ESM-2
        checkpoint on first use."""
        self.model_name = model_name
        self.device = device
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self._mlm_model = mlm_model.to(device).eval() if mlm_model is not None else None
        self._embedding_model = embedding_model.to(device).eval() if embedding_model is not None else None

    @property
    def mlm_model(self):
        if self._mlm_model is None:
            self._mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device).eval()
        return self._mlm_model

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        return self._embedding_model

    def masked_marginal_log_probs(self, sequence: str, position: int) -> dict[str, float]:
        """Mask `position` (1-indexed) in `sequence`, run one forward
        pass, and return the model's log P(residue) for every one of
        the 20 standard amino acids at that position (Meier et al.,
        2021's masked-marginal method)."""
        encoded = self.tokenizer(sequence, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        input_ids = input_ids.clone()
        input_ids[0, position] = self.tokenizer.mask_token_id  # index `position` == residue `position` (1-indexed) because index 0 is <cls>
        with torch.no_grad():
            logits = self.mlm_model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[0, position], dim=-1)
        return {aa: log_probs[self.tokenizer.convert_tokens_to_ids(aa)].item() for aa in AMINO_ACIDS}

    def score_single_mutants(self, wt_sequence: str, mutants: list[SingleMutant]) -> np.ndarray:
        """Masked-marginal log-odds score for each mutant: one forward
        pass per *unique mutated position* (4, for GB1), not one per
        mutant (76) -- the masked distribution at a given position is
        identical for every mutant sharing that position."""
        positions = sorted({m.position for m in mutants})
        log_probs_by_position = {pos: self.masked_marginal_log_probs(wt_sequence, pos) for pos in positions}
        scores = np.array(
            [log_probs_by_position[m.position][m.mut_aa] - log_probs_by_position[m.position][m.wt_aa] for m in mutants]
        )
        return scores

    def embed_sequences(self, sequences: list[str], batch_size: int = 16) -> np.ndarray:
        """Mean-pooled last-hidden-state embedding (excluding <cls>/<eos>)
        for each sequence, batched for speed."""
        embeddings = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                hidden = self.embedding_model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            mask[:, 0, :] = 0.0  # exclude <cls>
            lengths = encoded["attention_mask"].sum(dim=1) - 1  # exclude <cls>; -1 more below for <eos>
            for i, length in enumerate(lengths.tolist()):
                mask[i, int(length), :] = 0.0  # exclude <eos>, at the last real position
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            embeddings.append(pooled.cpu().numpy())
        return np.concatenate(embeddings, axis=0)


# --------------------------------------------------------------------------
# Evaluate
# --------------------------------------------------------------------------


def evaluate_zero_shot_masked_marginal(scorer: ESM2VariantScorer, wt_sequence: str, mutants: list[SingleMutant]) -> dict:
    """Spearman correlation between ESM-2 masked-marginal zero-shot
    scores and real experimental GB1 binding fitness, across the
    complete population of 76 single-point mutants."""
    scores = scorer.score_single_mutants(wt_sequence, mutants)
    fitness = np.array([m.fitness for m in mutants])
    rho, pvalue = spearmanr(scores, fitness)
    keep_mask = np.array([m.keep for m in mutants])
    rho_keep, pvalue_keep = spearmanr(scores[keep_mask], fitness[keep_mask])
    return {
        "model": scorer.model_name,
        "method": "masked_marginal",
        "n": len(mutants),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pvalue),
        "n_high_confidence": int(keep_mask.sum()),
        "spearman_rho_high_confidence": float(rho_keep),
        "spearman_pvalue_high_confidence": float(pvalue_keep),
    }


def evaluate_embedding_perturbation(scorer: ESM2VariantScorer, wt_sequence: str, mutants: list[SingleMutant]) -> dict:
    """Spearman correlation between how far a mutant's ESM-2 embedding
    is from the wild-type embedding (cosine distance) and how far its
    experimental fitness is from wild-type (|fitness - 1|) -- do larger
    representation shifts correspond to larger functional effects?"""
    sequences = [wt_sequence] + [m.sequence for m in mutants]
    embeddings = scorer.embed_sequences(sequences)
    wt_embedding, mutant_embeddings = embeddings[0], embeddings[1:]
    wt_norm = wt_embedding / np.linalg.norm(wt_embedding)
    mut_norms = mutant_embeddings / np.linalg.norm(mutant_embeddings, axis=1, keepdims=True)
    cosine_distance = 1.0 - mut_norms @ wt_norm
    fitness_deviation = np.abs(np.array([m.fitness for m in mutants]) - 1.0)
    rho, pvalue = spearmanr(cosine_distance, fitness_deviation)
    return {
        "model": scorer.model_name,
        "method": "embedding_cosine_distance",
        "n": len(mutants),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pvalue),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot ESM-2 single-point mutation effect prediction on real GB1 DMS data.")
    parser.add_argument("--models", nargs="+", default=list(ESM2_MODEL_NAMES))
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--use-cached-raw", action="store_true")
    parser.add_argument("--results-path", default=str(Path(__file__).parent / "results" / "esm2_gb1_results.json"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.use_cached_raw:
        csv_path = out_dir / "gb1_single_mutants_sample.csv"
        print(f"Loading cached extract -> {csv_path}")
    else:
        csv_path = out_dir / "gb1_full_dataset.csv"
        try:
            print(f"Downloading full GB1 dataset (Wu et al., 2016) from {GB1_DATA_URL} ...")
            download_gb1_dataset(csv_path)
        except requests.exceptions.RequestException as exc:
            print(f"Live download failed ({exc}); falling back to bundled fixture.")
            csv_path = out_dir / "gb1_single_mutants_sample.csv"

    wt_sequence, mutants = load_single_mutants(csv_path)
    print(f"GB1 domain (WT, {len(wt_sequence)} aa): {wt_sequence}")
    print(f"Loaded the complete population of {len(mutants)} single-point mutants at positions {GB1_MUTATED_POSITIONS}\n")

    results = []
    for model_name in args.models:
        print(f"=== {model_name} ===")
        scorer = ESM2VariantScorer(model_name=model_name)
        mm_result = evaluate_zero_shot_masked_marginal(scorer, wt_sequence, mutants)
        print(f"  masked-marginal zero-shot: rho={mm_result['spearman_rho']:.4f} (p={mm_result['spearman_pvalue']:.2e}, n={mm_result['n']})")
        print(
            f"    high-confidence subset: rho={mm_result['spearman_rho_high_confidence']:.4f} "
            f"(n={mm_result['n_high_confidence']})"
        )
        emb_result = evaluate_embedding_perturbation(scorer, wt_sequence, mutants)
        print(f"  embedding-distance:       rho={emb_result['spearman_rho']:.4f} (p={emb_result['spearman_pvalue']:.2e})\n")
        results.append(mm_result)
        results.append(emb_result)

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
