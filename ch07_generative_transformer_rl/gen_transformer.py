"""
Chapter 7 hands-on project: a generative Transformer with an RL feedback
loop for de novo EGFR (CHEMBL203) inhibitor design.

Extract: paginated retrieval of raw EGFR IC50 bioactivity records from
  the ChEMBL API (same pattern as Chapters 4-5).
Transform: standardize structures (Chapter 4, Section 4.1), deduplicate
  via the median, and label active/inactive at a fixed IC50 threshold
  (Chapter 5, Section 5.5's pattern).
Reward oracle: an XGBoost classifier on ECFP4 fingerprints (Chapter 5),
  evaluated with a Bemis-Murcko scaffold split (Chapters 5-6) so its
  reported accuracy is an honest, not-random-split-inflated estimate.
Generator: a small decoder-only Transformer (Vaswani et al., 2017;
  Bagal et al., 2022) over SELFIES tokens (Krenn et al., 2020),
  pretrained by next-token prediction on real EGFR-active compounds,
  then fine-tuned with REINFORCE (Williams, 1992; Olivecrona et al.,
  2017) against a multi-objective reward (predicted EGFR activity +
  Lipinski drug-likeness).

See README.md for usage and chapter.md Section 7.6 for context.
"""
import argparse
import json
import math
import statistics
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_TARGET_CHEMBL_ID = "CHEMBL203"  # EGFR - the running target since Chapter 1
DEFAULT_MAX_RECORDS = 3000
DEFAULT_PAGE_SIZE = 100
DEFAULT_TIMEOUT = 30
DEFAULT_THRESHOLD_NM = 1_000.0  # 1 uM active/inactive cutoff - see chapter.md Section 7.6
ACCEPTED_RELATIONS = {"=", None}
UNIT_TO_NM = {"nM": 1.0, "uM": 1_000.0, "mM": 1_000_000.0, "pM": 0.001}
MAX_SEQ_LEN = 72  # SELFIES tokens, including BOS/EOS

FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

_STANDARDIZER = None
_TAUTOMER_ENUMERATOR = None


def _standardizer() -> rdMolStandardize.LargestFragmentChooser:
    global _STANDARDIZER
    if _STANDARDIZER is None:
        _STANDARDIZER = rdMolStandardize.LargestFragmentChooser()
    return _STANDARDIZER


def _tautomer_enumerator() -> rdMolStandardize.TautomerEnumerator:
    global _TAUTOMER_ENUMERATOR
    if _TAUTOMER_ENUMERATOR is None:
        _TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
    return _TAUTOMER_ENUMERATOR


@dataclass
class CleanCompoundRecord:
    molecule_chembl_id: str
    canonical_smiles: str
    ic50_nm: float
    n_measurements: int
    is_active: bool


# --------------------------------------------------------------------------
# Extract
# --------------------------------------------------------------------------


def extract_bioactivities(
    target_chembl_id: str = DEFAULT_TARGET_CHEMBL_ID,
    max_records: int = DEFAULT_MAX_RECORDS,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict]:
    records: list[dict] = []
    url = f"{CHEMBL_API_BASE}/activity.json"
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": "IC50",
        "limit": page_size,
        "format": "json",
    }
    while url and len(records) < max_records:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        records.extend(payload.get("activities", []))
        next_path = payload.get("page_meta", {}).get("next")
        url = f"https://www.ebi.ac.uk{next_path}" if next_path else None
        params = None
    return records[:max_records]


def save_raw_json(records: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"activities": records}, indent=None), encoding="utf-8")
    return out_path


def load_raw_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["activities"]


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


def standardize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _standardizer().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = _tautomer_enumerator().Canonicalize(mol)
    return Chem.MolToSmiles(mol)


def _to_nm(value: float, units: str | None) -> float | None:
    factor = UNIT_TO_NM.get(units)
    return value * factor if factor is not None else None


def clean_bioactivity_records(
    raw_records: list[dict], threshold_nm: float = DEFAULT_THRESHOLD_NM
) -> list[CleanCompoundRecord]:
    filtered = []
    for r in raw_records:
        if r.get("standard_type") != "IC50":
            continue
        if r.get("standard_relation") not in ACCEPTED_RELATIONS:
            continue
        if not r.get("canonical_smiles") or r.get("standard_value") is None:
            continue
        value_nm = _to_nm(float(r["standard_value"]), r.get("standard_units"))
        if value_nm is None or value_nm <= 0:
            continue
        filtered.append(
            {
                "molecule_chembl_id": r["molecule_chembl_id"],
                "raw_smiles": r["canonical_smiles"],
                "value_nm": value_nm,
            }
        )

    standardized_cache: dict[str, str | None] = {}
    for row in filtered:
        raw_smiles = row["raw_smiles"]
        if raw_smiles not in standardized_cache:
            standardized_cache[raw_smiles] = standardize_smiles(raw_smiles)
        row["standardized_smiles"] = standardized_cache[raw_smiles]
    filtered = [r for r in filtered if r["standardized_smiles"] is not None]

    groups: dict[str, list[dict]] = {}
    for row in filtered:
        groups.setdefault(row["standardized_smiles"], []).append(row)

    results = []
    for smiles, rows in groups.items():
        values = [r["value_nm"] for r in rows]
        median_nm = statistics.median(values)
        results.append(
            CleanCompoundRecord(
                molecule_chembl_id=rows[0]["molecule_chembl_id"],
                canonical_smiles=smiles,
                ic50_nm=round(median_nm, 3),
                n_measurements=len(values),
                is_active=median_nm <= threshold_nm,
            )
        )
    results.sort(key=lambda r: r.ic50_nm)
    return results


# --------------------------------------------------------------------------
# Reward oracle (Chapter 5's methodology: ECFP4 -> XGBoost, evaluated with
# a Bemis-Murcko scaffold split so the reported accuracy isn't inflated)
# --------------------------------------------------------------------------


def featurize(smiles_list: list[str]) -> np.ndarray:
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = FP_GENERATOR.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=np.int8)
        for bit in fp.GetOnBits():
            arr[bit] = 1
        fps.append(arr)
    return np.vstack(fps)


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(records: list[CleanCompoundRecord], frac_train: float = 0.8, seed: int = 0):
    scaffold_to_indices: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        scaffold_to_indices.setdefault(bemis_murcko_scaffold(r.canonical_smiles), []).append(i)
    rng = np.random.RandomState(seed)
    groups = list(scaffold_to_indices.values())
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]
    groups.sort(key=len, reverse=True)
    n_train_target = int(round(frac_train * len(records)))
    train_idx, test_idx = [], []
    for group in groups:
        if len(train_idx) < n_train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)
    return sorted(train_idx), sorted(test_idx)


class RewardOracle:
    """Wraps an XGBoost classifier predicting P(EGFR-active) from ECFP4
    fingerprints, trained on real ChEMBL bioactivity data."""

    def __init__(self, seed: int = 0):
        self.model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=seed, n_jobs=-1
        )
        self._fitted = False

    def fit(self, records: list[CleanCompoundRecord]) -> None:
        X = featurize([r.canonical_smiles for r in records])
        y = np.array([int(r.is_active) for r in records])
        self.model.fit(X, y)
        self._fitted = True

    def predict_proba(self, smiles_list: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RewardOracle.fit() must be called before predict_proba()")
        X = featurize(smiles_list)
        return self.model.predict_proba(X)[:, 1]


def evaluate_oracle(records: list[CleanCompoundRecord], seed: int = 0) -> dict:
    """Train/evaluate the reward oracle under a scaffold split, reporting
    honest, not-random-split-inflated held-out performance."""
    train_idx, test_idx = scaffold_split(records, seed=seed)
    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]

    oracle = RewardOracle(seed=seed)
    oracle.fit(train_records)

    y_test = np.array([int(r.is_active) for r in test_records])
    proba = oracle.predict_proba([r.canonical_smiles for r in test_records])
    pred = (proba >= 0.5).astype(int)
    return {
        "n_train": len(train_records),
        "n_test": len(test_records),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }


# --------------------------------------------------------------------------
# SELFIES tokenization
# --------------------------------------------------------------------------

PAD, BOS, EOS = "[PAD]", "[BOS]", "[EOS]"


def smiles_to_selfies(smiles: str) -> str | None:
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    @classmethod
    def build(cls, selfies_strings: list[str]) -> "Vocabulary":
        symbols: set[str] = set()
        for s in selfies_strings:
            symbols.update(sf.split_selfies(s))
        specials = [PAD, BOS, EOS]
        tokens = specials + sorted(symbols)
        return cls({tok: i for i, tok in enumerate(tokens)})

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, selfies_string: str, max_len: int = MAX_SEQ_LEN) -> list[int] | None:
        symbols = list(sf.split_selfies(selfies_string))
        ids = [self.token_to_id[BOS]] + [self.token_to_id[s] for s in symbols] + [self.token_to_id[EOS]]
        if len(ids) > max_len:
            return None
        ids = ids + [self.token_to_id[PAD]] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        symbols = []
        for i in ids:
            tok = self.id_to_token[i]
            if tok == EOS:
                break
            if tok in (BOS, PAD):
                continue
            symbols.append(tok)
        return "".join(symbols)


# --------------------------------------------------------------------------
# Generator: a small decoder-only Transformer over SELFIES tokens
# --------------------------------------------------------------------------


class GenerativeTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(ids) + self.position_embedding(pos)
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), diagonal=1)
        x = self.blocks(x, mask=causal_mask, is_causal=True)
        x = self.ln_out(x)
        return self.head(x)

    @torch.no_grad()
    def sample(self, vocab: Vocabulary, n: int, temperature: float = 1.0, seed: int | None = None) -> list[list[int]]:
        if seed is not None:
            torch.manual_seed(seed)
        device = next(self.parameters()).device
        bos, eos, pad = vocab.token_to_id[BOS], vocab.token_to_id[EOS], vocab.token_to_id[PAD]
        ids = torch.full((n, 1), bos, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(self.max_len - 1):
            logits = self(ids)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).squeeze(-1)
            next_tok = torch.where(done, torch.full_like(next_tok, pad), next_tok)
            done = done | (next_tok == eos)
            ids = torch.cat([ids, next_tok.unsqueeze(1)], dim=1)
            if done.all():
                break
        pad_amount = self.max_len - ids.size(1)
        if pad_amount > 0:
            ids = F.pad(ids, (0, pad_amount), value=pad)
        return ids.tolist()

    def sample_with_logprobs(self, vocab: Vocabulary, n: int, temperature: float = 1.0):
        """Differentiable sampling for REINFORCE: returns token ids and the
        summed log-probability of each sampled sequence under the current
        policy."""
        device = next(self.parameters()).device
        bos, eos, pad = vocab.token_to_id[BOS], vocab.token_to_id[EOS], vocab.token_to_id[PAD]
        ids = torch.full((n, 1), bos, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        seq_logprob = torch.zeros(n, device=device)
        for _ in range(self.max_len - 1):
            logits = self(ids)[:, -1, :] / temperature
            logprobs = F.log_softmax(logits, dim=-1)
            probs = logprobs.exp()
            next_tok = torch.multinomial(probs, 1).squeeze(-1)
            token_logprob = logprobs.gather(1, next_tok.unsqueeze(1)).squeeze(1)
            seq_logprob = seq_logprob + torch.where(done, torch.zeros_like(token_logprob), token_logprob)
            next_tok = torch.where(done, torch.full_like(next_tok, pad), next_tok)
            done = done | (next_tok == eos)
            ids = torch.cat([ids, next_tok.unsqueeze(1)], dim=1)
            if done.all():
                break
        pad_amount = self.max_len - ids.size(1)
        if pad_amount > 0:
            ids = F.pad(ids, (0, pad_amount), value=pad)
        return ids, seq_logprob


# --------------------------------------------------------------------------
# Pretraining (next-token cross-entropy)
# --------------------------------------------------------------------------


def pretrain(
    model: GenerativeTransformer, vocab: Vocabulary, encoded_sequences: list[list[int]], epochs: int, lr: float = 3e-4, batch_size: int = 32, seed: int = 0,
) -> list[float]:
    torch.manual_seed(seed)
    data = torch.tensor(encoded_sequences, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pad_id = vocab.token_to_id[PAD]
    losses = []
    n = data.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = data[perm[start : start + batch_size]]
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses


# --------------------------------------------------------------------------
# Reward function and RL fine-tuning (REINFORCE)
# --------------------------------------------------------------------------


def lipinski_pass(mol: Chem.Mol) -> bool:
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return violations <= 1


@dataclass
class GenerationReport:
    smiles: str | None
    valid: bool
    p_active: float
    lipinski_ok: bool
    reward: float


def score_sequences(
    id_batches: list[list[int]], vocab: Vocabulary, oracle: RewardOracle, w_activity: float = 0.7, w_lipinski: float = 0.3
) -> list[GenerationReport]:
    decoded_selfies = [vocab.decode(ids) for ids in id_batches]
    smiles_list: list[str | None] = []
    for s in decoded_selfies:
        if not s:
            smiles_list.append(None)
            continue
        try:
            smiles_list.append(sf.decoder(s))
        except Exception:
            smiles_list.append(None)

    valid_mask = []
    canon_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            valid_mask.append(False)
            canon_smiles.append(None)
        else:
            valid_mask.append(True)
            canon_smiles.append(Chem.MolToSmiles(mol))

    valid_smiles = [s for s, ok in zip(canon_smiles, valid_mask) if ok]
    p_active_by_smiles = {}
    if valid_smiles:
        proba = oracle.predict_proba(valid_smiles)
        p_active_by_smiles = dict(zip(valid_smiles, proba))

    reports = []
    for smi, ok in zip(canon_smiles, valid_mask):
        if not ok:
            reports.append(GenerationReport(smiles=None, valid=False, p_active=0.0, lipinski_ok=False, reward=0.0))
            continue
        mol = Chem.MolFromSmiles(smi)
        p_active = float(p_active_by_smiles[smi])
        lip_ok = lipinski_pass(mol)
        reward = w_activity * p_active + w_lipinski * float(lip_ok)
        reports.append(GenerationReport(smiles=smi, valid=True, p_active=p_active, lipinski_ok=lip_ok, reward=reward))
    return reports


def reinforce_finetune(
    model: GenerativeTransformer,
    vocab: Vocabulary,
    oracle: RewardOracle,
    iterations: int,
    batch_size: int = 32,
    lr: float = 1e-4,
    temperature: float = 1.0,
    seed: int = 0,
) -> list[dict]:
    """REINFORCE (Williams, 1992) fine-tuning: maximize
    E[reward * log p(sequence)] via policy-gradient ascent, with a
    running-mean baseline to reduce variance. Deliberately omits the
    KL-to-prior penalty production RL-for-generation pipelines
    typically add (e.g. Olivecrona et al., 2017) - see chapter.md
    Section 7.6 for why."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    baseline = 0.0
    history = []
    for it in range(iterations):
        model.train()
        ids, seq_logprob = model.sample_with_logprobs(vocab, n=batch_size, temperature=temperature)
        reports = score_sequences(ids.tolist(), vocab, oracle)
        rewards = torch.tensor([r.reward for r in reports], dtype=torch.float32)

        baseline = 0.9 * baseline + 0.1 * rewards.mean().item()
        advantage = rewards - baseline
        loss = -(advantage * seq_logprob).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(
            {
                "iteration": it,
                "mean_reward": round(rewards.mean().item(), 4),
                "valid_frac": round(sum(r.valid for r in reports) / len(reports), 4),
                "mean_p_active": round(float(np.mean([r.p_active for r in reports if r.valid])) if any(r.valid for r in reports) else 0.0, 4),
            }
        )
    return history


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain + RL fine-tune a generative Transformer for EGFR inhibitor design.")
    parser.add_argument("--target", default=DEFAULT_TARGET_CHEMBL_ID)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--threshold-nm", type=float, default=DEFAULT_THRESHOLD_NM)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--rl-iterations", type=int, default=25)
    parser.add_argument("--n-sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--use-cached-raw", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.use_cached_raw:
        fixture_path = out_dir / "raw_egfr_bioactivities_sample.json"
        print(f"Loading cached raw extract -> {fixture_path}")
        raw_records = load_raw_json(fixture_path)[: args.max_records]
    else:
        try:
            print(f"Extracting IC50 bioactivities for {args.target} from ChEMBL...")
            raw_records = extract_bioactivities(args.target, max_records=args.max_records)
            save_raw_json(raw_records, out_dir / f"raw_{args.target}_bioactivities.json")
        except requests.exceptions.RequestException as exc:
            print(f"Live ChEMBL extraction failed ({exc}); falling back to bundled fixture.")
            raw_records = load_raw_json(out_dir / "raw_egfr_bioactivities_sample.json")[: args.max_records]

    records = clean_bioactivity_records(raw_records, threshold_nm=args.threshold_nm)
    n_active = sum(1 for r in records if r.is_active)
    print(f"Cleaned to {len(records)} unique compounds ({n_active} active / {len(records) - n_active} inactive)\n")

    print("Evaluating reward oracle (scaffold split)...")
    oracle_metrics = evaluate_oracle(records, seed=args.seed)
    print(f"  {oracle_metrics}\n")

    oracle = RewardOracle(seed=args.seed)
    oracle.fit(records)

    active_records = [r for r in records if r.is_active]
    selfies_strings = []
    for r in active_records:
        s = smiles_to_selfies(r.canonical_smiles)
        if s is not None:
            selfies_strings.append(s)

    vocab = Vocabulary.build(selfies_strings)
    encoded = [vocab.encode(s) for s in selfies_strings]
    encoded = [e for e in encoded if e is not None]
    print(f"Pretraining corpus: {len(encoded)} active EGFR compounds, vocab size {len(vocab)}\n")

    model = GenerativeTransformer(vocab_size=len(vocab))
    print(f"Pretraining for {args.pretrain_epochs} epochs...")
    losses = pretrain(model, vocab, encoded, epochs=args.pretrain_epochs, seed=args.seed)
    print(f"  final cross-entropy loss: {losses[-1]:.4f}\n")

    training_smiles = {Chem.MolToSmiles(Chem.MolFromSmiles(r.canonical_smiles)) for r in active_records}

    print("Sampling from pretrained-only policy...")
    pre_ids = model.sample(vocab, n=args.n_sample, seed=args.seed)
    pre_reports = score_sequences(pre_ids, vocab, oracle)
    _print_generation_summary("Pretrained (before RL)", pre_reports, training_smiles)

    print(f"\nRL fine-tuning ({args.rl_iterations} iterations)...")
    history = reinforce_finetune(model, vocab, oracle, iterations=args.rl_iterations, seed=args.seed)
    print(f"  final iteration: {history[-1]}\n")

    print("Sampling from RL-fine-tuned policy...")
    post_ids = model.sample(vocab, n=args.n_sample, seed=args.seed)
    post_reports = score_sequences(post_ids, vocab, oracle)
    _print_generation_summary("RL fine-tuned (after RL)", post_reports, training_smiles)


def _print_generation_summary(label: str, reports: list[GenerationReport], training_smiles: set[str]) -> None:
    """training_smiles must be canonical RDKit SMILES (Chem.MolToSmiles
    output), matching the canonicalization score_sequences already
    applies to every valid generated molecule, so novelty is a like-for-
    like set membership check rather than a string-format artifact."""
    valid = [r for r in reports if r.valid]
    n = len(reports)
    valid_frac = len(valid) / n
    unique_frac = len({r.smiles for r in valid}) / len(valid) if valid else 0.0
    novel_frac = sum(r.smiles not in training_smiles for r in valid) / len(valid) if valid else 0.0
    mean_p_active = float(np.mean([r.p_active for r in valid])) if valid else 0.0
    lipinski_frac = sum(r.lipinski_ok for r in valid) / len(valid) if valid else 0.0
    mean_reward = float(np.mean([r.reward for r in reports]))
    print(f"[{label}] n={n}")
    print(f"  valid_frac={valid_frac:.4f}  unique_frac_of_valid={unique_frac:.4f}  novel_frac_of_valid={novel_frac:.4f}")
    print(f"  mean_p_active={mean_p_active:.4f}  lipinski_frac={lipinski_frac:.4f}  mean_reward={mean_reward:.4f}")


if __name__ == "__main__":
    main()
