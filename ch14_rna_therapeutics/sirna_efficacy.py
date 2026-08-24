"""
Chapter 14 hands-on project: a real siRNA knockdown-efficacy predictor
built from real, published RNAi silencing data and real RNA
thermodynamics -- no target-agnostic proxy, no synthetic labels.

Extract: real siRNA/target pairs and their real, measured silencing
  efficiency, drawn from Huesken, D., Giuffliger, J., Wenner, C. et al.
  (2005), "Design of a genome-wide siRNA library using an artificial
  neural network," Nat Biotechnol 23, 995-1001
  (https://doi.org/10.1038/nbt1118) -- 2,182-2,431 chemically
  synthesized 19-mer siRNAs profiled against a dual-luciferase reporter
  in H1299 cells, the field's original large-scale siRNA efficacy
  training set and still the most-cited benchmark for this task two
  decades later. The paper's own original hosting (Novartis'
  BIOPREDsi web server) is no longer online; this script fetches the
  real sequences and real measured efficiency values live from
  github.com/dimostzim/siRBench (Huesken subset: 2,133 train + 228
  test sequences, zero overlap by sequence), a small, actively
  maintained, unlicensed community redistribution that exists
  specifically because the original hosting disappeared -- exactly the
  kind of secondary-source dependency this chapter discloses openly
  rather than hides (see chapter.md Section 14.3). Only the four raw,
  factual fields (siRNA sequence, target mRNA context, measured
  efficiency, cell line) are taken from that source; every feature
  used to predict efficiency below is computed by this script itself,
  from those raw sequences, via the real ViennaRNA thermodynamic
  engine -- none of siRBench's own pre-engineered feature columns are
  reused.

Predict: knockdown efficiency (a real, continuous [0, 1] measured
  silencing score -- 1 corresponds to maximal, ~complete knockdown) for
  a held-out siRNA/target pair, from real features computed directly
  from RNA secondary-structure and duplex thermodynamics (ViennaRNA:
  Lorenz et al., 2011) plus simple, established sequence-composition
  heuristics (Reynolds et al., 2004; Ui-Tei et al., 2004; Khvorova et
  al., 2003) -- never from any activity label leaking into a feature.

Evaluate: real Spearman/Pearson correlation, R-squared, and RMSE
  against real measured efficiency on siRBench's own pre-defined,
  sequence-disjoint Huesken test split (228 sequences never seen during
  training), for a rule-based composite score alone, a Random Forest,
  an XGBoost model, and a small PyTorch MLP -- the same real,
  feature-set-vs-learned-model comparison this book has run in every
  prior QSAR/property-prediction chapter (Chapters 5, 6, 11).

See README.md for usage and chapter.md Section 14.3 for full context.
"""
import argparse
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import RNA
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# github.com/dimostzim/siRBench, commit-independent `main` raw URLs.
# Real, live-fetchable redistribution of the real Huesken et al. (2005)
# siRNA efficacy dataset (see module docstring for full provenance).
SIRBENCH_RAW_URL = "https://raw.githubusercontent.com/dimostzim/siRBench/main/data/siRBench_{split}.csv"
HUESKEN_SOURCE_TAG = "Huesken"

SIRNA_LEN = 19
FLANK_LEN = 19
EXTENDED_LEN = SIRNA_LEN + 2 * FLANK_LEN  # 57 nt: 19 nt upstream + 19 nt target + 19 nt downstream
TARGET_WINDOW = slice(FLANK_LEN, FLANK_LEN + SIRNA_LEN)  # [19:38], 0-indexed

SEED_START, SEED_END = 1, 8  # 0-indexed [1:8] = positions 2-8 (1-indexed), the established RNAi "seed" region
END_FRAGMENT_LEN = 6  # terminal-duplex-stability fragment length for the 5'/3' asymmetry feature

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "gc_content", "seed_gc_content", "internal_4mer_repeats",
    "pos1_is_AU", "pos19_is_GC",
    "duplex_energy_total", "duplex_energy_5p_end", "duplex_energy_3p_end", "duplex_end_asymmetry",
    "guide_self_mfe", "guide_ensemble_diversity",
    "target_unconstrained_mfe", "target_opening_energy",
    "design_heuristic_score",
]


# --------------------------------------------------------------------------
# Real data acquisition: the Huesken et al. (2005) siRNA efficacy set
# --------------------------------------------------------------------------


def fetch_sirbench_csv(split: str, timeout: int = 60) -> str:
    """Fetch one real siRBench CSV split ('train' or 'test') live from
    its GitHub raw URL."""
    response = requests.get(SIRBENCH_RAW_URL.format(split=split), timeout=timeout)
    response.raise_for_status()
    return response.text


def _dna_to_rna(seq: str) -> str:
    return seq.strip().upper().replace("T", "U")


def curate_huesken_subset(csv_text: str) -> list[dict]:
    """Parse one real siRBench CSV split into one record per real,
    Huesken-sourced siRNA/target pair, keeping only the four raw,
    factual fields this chapter's own feature engineering needs (no
    pre-engineered feature columns from the source are retained).
    Records whose 57-nt extended target context does not actually
    contain the expected 19-nt target window at its real, documented
    offset (a basic real-data sanity check, not assumed) are dropped
    and counted, not silently kept."""
    reader = csv.DictReader(io.StringIO(csv_text))
    records = []
    n_dropped_malformed = 0
    for row in reader:
        if row.get("source") != HUESKEN_SOURCE_TAG:
            continue
        sirna = _dna_to_rna(row["siRNA"])
        target = _dna_to_rna(row["mRNA"])
        extended = _dna_to_rna(row["extended_mRNA"])
        try:
            efficiency = float(row["efficiency"])
        except (TypeError, ValueError):
            n_dropped_malformed += 1
            continue

        valid = (
            len(sirna) == SIRNA_LEN
            and len(extended) == EXTENDED_LEN
            and extended[TARGET_WINDOW] == target
            and set(sirna) <= set("ACGU")
            and 0.0 <= efficiency <= 1.0
        )
        if not valid:
            n_dropped_malformed += 1
            continue

        records.append(
            {
                "sirna": sirna,
                "target_site": target,
                "extended_target": extended,
                "efficiency": efficiency,
                "cell_line": row.get("cell_line", ""),
            }
        )
    if n_dropped_malformed:
        print(f"  ({n_dropped_malformed} malformed/inconsistent rows dropped during curation)")
    return records


def load_or_build_dataset(split: str, refresh: bool = False) -> list[dict]:
    """Load the cached, real curated Huesken subset for one split if
    present (bundled in data/ for offline reproducibility); otherwise
    fetch live from siRBench and cache the result."""
    cache_path = DATA_DIR / f"sirna_huesken_{split}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    csv_text = fetch_sirbench_csv(split)
    records = curate_huesken_subset(csv_text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


# --------------------------------------------------------------------------
# Real RNA thermodynamic features (ViennaRNA: Lorenz et al., 2011)
# --------------------------------------------------------------------------


def guide_self_structure(sirna: str) -> dict:
    """Real MFE secondary structure and ensemble diversity of the
    isolated 19-nt guide strand -- a stable self-structure (e.g. a
    hairpin) competes with RISC loading, an established negative
    predictor of siRNA activity."""
    fc = RNA.fold_compound(sirna)
    _structure, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()
    return {"guide_self_mfe": round(mfe, 3), "guide_ensemble_diversity": round(fc.mean_bp_distance(), 3)}


def duplex_features(sirna: str, target_site: str) -> dict:
    """Real siRNA-target duplex hybridization energy (full duplex, via
    ViennaRNA's `RNA.duplexfold`, the Markham & Zuker duplex-folding
    algorithm) and the real terminal-stability asymmetry between the
    guide strand's 5' and 3' ends -- the biophysical basis of the
    established thermodynamic asymmetry rule for RISC strand selection
    (Khvorova et al., 2003; Schwarz et al., 2003): a guide strand whose
    5' end is *less* stably paired than its 3' end is preferentially
    loaded into RISC as the active antisense strand."""
    full = RNA.duplexfold(sirna, target_site)
    five_prime = RNA.duplexfold(sirna[:END_FRAGMENT_LEN], target_site[-END_FRAGMENT_LEN:])
    three_prime = RNA.duplexfold(sirna[-END_FRAGMENT_LEN:], target_site[:END_FRAGMENT_LEN])
    return {
        "duplex_energy_total": round(full.energy, 3),
        "duplex_energy_5p_end": round(five_prime.energy, 3),
        "duplex_energy_3p_end": round(three_prime.energy, 3),
        "duplex_end_asymmetry": round(five_prime.energy - three_prime.energy, 3),
    }


def target_accessibility_features(extended_target: str) -> dict:
    """Real target-site accessibility, computed the same way RNAup
    (Muckstein et al., 2006) does internally: the real MFE of the full
    57-nt local genomic context, and the real MFE after a real hard
    constraint forces the central 19-nt target window to be unpaired
    (`fold_compound.hc_add_up`). The energetic cost of opening the site,
    `opening_energy = constrained_mfe - unconstrained_mfe`, is a real,
    non-negative accessibility proxy: a target site already unpaired
    in the transcript's local secondary structure costs nothing extra
    to open (near 0), while a site buried inside a stable stem costs
    real, positive free energy -- and is correspondingly harder for
    RISC to engage."""
    fc_free = RNA.fold_compound(extended_target)
    _s_free, mfe_free = fc_free.mfe()

    fc_open = RNA.fold_compound(extended_target)
    for i in range(TARGET_WINDOW.start + 1, TARGET_WINDOW.stop + 1):  # ViennaRNA hc_add_up is 1-indexed
        fc_open.hc_add_up(i)
    _s_open, mfe_open = fc_open.mfe()

    return {
        "target_unconstrained_mfe": round(mfe_free, 3),
        "target_opening_energy": round(mfe_open - mfe_free, 3),
    }


# --------------------------------------------------------------------------
# Sequence-composition features (established RNAi design heuristics)
# --------------------------------------------------------------------------


def gc_content(seq: str) -> float:
    return sum(1 for c in seq if c in "GC") / len(seq)


def internal_4mer_repeats(seq: str, k: int = 4) -> int:
    """Count of 4-mers that occur more than once within the guide
    strand -- a simple, real proxy for the internal-repeat/self-priming
    liability Reynolds et al. (2004) flagged as reducing potency."""
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    counts = pd.Series(kmers).value_counts()
    return int((counts[counts > 1] - 1).sum())


def design_heuristic_score(sirna: str) -> float:
    """A simple composite score (0-4) built from the qualitative
    direction of established, published RNAi design heuristics --
    moderate overall GC content (Reynolds et al., 2004), an A/U (weakly
    paired) base at the guide's 5' end and a G/C-poor 3' end favoring
    correct RISC strand selection (Ui-Tei et al., 2004), and low seed
    GC content (Reynolds et al., 2004). This is a compact heuristic
    *inspired by* those papers' qualitative design principles for use
    as this section's rule-based baseline, not a reproduction of any
    single paper's own published numeric point-scoring table."""
    score = 0.0
    score += 1.0 if 0.30 <= gc_content(sirna) <= 0.52 else 0.0
    score += 1.0 if sirna[0] in "AU" else 0.0
    score += 1.0 if sirna[-1] not in "GC" else 0.0
    score += 1.0 if gc_content(sirna[SEED_START:SEED_END]) <= 0.55 else 0.0
    return score


def build_feature_row(record: dict) -> dict:
    """Assemble one real, fully self-computed feature row for one
    siRNA/target pair."""
    sirna, target_site, extended = record["sirna"], record["target_site"], record["extended_target"]
    row = {
        "gc_content": round(gc_content(sirna), 4),
        "seed_gc_content": round(gc_content(sirna[SEED_START:SEED_END]), 4),
        "internal_4mer_repeats": internal_4mer_repeats(sirna),
        "pos1_is_AU": int(sirna[0] in "AU"),
        "pos19_is_GC": int(sirna[-1] in "GC"),
        "design_heuristic_score": design_heuristic_score(sirna),
    }
    row.update(duplex_features(sirna, target_site))
    row.update(guide_self_structure(sirna))
    row.update(target_accessibility_features(extended))
    row["efficiency"] = record["efficiency"]
    return row


def build_feature_table(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(build_feature_row(r) for r in records)


# --------------------------------------------------------------------------
# Models: rule-based baseline, Random Forest, XGBoost, and a small PyTorch MLP
# --------------------------------------------------------------------------


class SirnaMLP(nn.Module):
    """A small feed-forward regressor -- the same modeling family
    Huesken et al. (2005) themselves used, on this chapter's own,
    fully-disclosed real feature set rather than their original
    proprietary one."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, n_epochs: int = 300, patience: int = 20) -> SirnaMLP:
    torch.manual_seed(RANDOM_SEED)
    model = SirnaMLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    xt, yt = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    xv, yv = torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

    best_val_loss, best_state, epochs_without_improvement = float("inf"), None, 0
    for _epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_state, epochs_without_improvement = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    spearman = spearmanr(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)
    return {
        "spearman_rho": round(float(spearman.statistic), 4),
        "spearman_p": round(float(spearman.pvalue), 6),
        "pearson_r": round(float(pearson.statistic), 4),
        "pearson_p": round(float(pearson.pvalue), 6),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    x_cols = FEATURE_COLUMNS
    x_train_full, y_train_full = train_df[x_cols].to_numpy(dtype=np.float64), train_df["efficiency"].to_numpy(dtype=np.float64)
    x_test, y_test = test_df[x_cols].to_numpy(dtype=np.float64), test_df["efficiency"].to_numpy(dtype=np.float64)

    # A real internal validation split off the training set only (never
    # touching the test set) for the MLP's own early-stopping criterion.
    rng = np.random.default_rng(RANDOM_SEED)
    val_mask = rng.random(len(x_train_full)) < 0.15
    x_tr, y_tr = x_train_full[~val_mask], y_train_full[~val_mask]
    x_val, y_val = x_train_full[val_mask], y_train_full[val_mask]

    feature_mean, feature_std = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-8
    normalize = lambda x: (x - feature_mean) / feature_std  # noqa: E731

    results: dict = {"n_train": len(train_df), "n_val": len(x_val), "n_test": len(test_df)}

    results["rule_based_baseline"] = regression_metrics(y_test, test_df["design_heuristic_score"].to_numpy(dtype=np.float64))

    rf = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(x_train_full, y_train_full)
    results["random_forest"] = regression_metrics(y_test, rf.predict(x_test))
    results["random_forest"]["feature_importances"] = {
        col: round(float(imp), 4) for col, imp in sorted(zip(x_cols, rf.feature_importances_), key=lambda t: -t[1])
    }

    xgb = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    xgb.fit(x_train_full, y_train_full)
    results["xgboost"] = regression_metrics(y_test, xgb.predict(x_test))
    results["xgboost"]["feature_importances"] = {
        col: round(float(imp), 4) for col, imp in sorted(zip(x_cols, xgb.feature_importances_), key=lambda t: -t[1])
    }

    mlp = train_mlp(normalize(x_tr), y_tr, normalize(x_val), y_val)
    with torch.no_grad():
        mlp_pred = mlp(torch.tensor(normalize(x_test), dtype=torch.float32)).numpy()
    results["pytorch_mlp"] = regression_metrics(y_test, mlp_pred)

    return results


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-cache", action="store_true", help="Re-fetch the siRBench splits live instead of using the bundled cache")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "sirna_efficacy_results.json")
    args = parser.parse_args()

    print("Loading the real Huesken et al. (2005) siRNA efficacy dataset (via siRBench)...")
    train_records = load_or_build_dataset("train", refresh=args.refresh_cache)
    test_records = load_or_build_dataset("test", refresh=args.refresh_cache)
    print(f"  {len(train_records)} real train sequences, {len(test_records)} real test sequences")

    print("Computing real ViennaRNA thermodynamic + sequence-composition features...")
    train_df = build_feature_table(train_records)
    test_df = build_feature_table(test_records)

    print("Training and evaluating real models (rule-based baseline, Random Forest, XGBoost, PyTorch MLP)...")
    results = train_and_evaluate(train_df, test_df)
    print(json.dumps({k: v for k, v in results.items() if isinstance(v, dict) and "spearman_rho" in v or not isinstance(v, dict)}, indent=2))

    output = {
        "dataset": {
            "source": "Huesken et al., 2005, Nat Biotechnol (via github.com/dimostzim/siRBench)",
            "n_train": len(train_records),
            "n_test": len(test_records),
        },
        "feature_columns": FEATURE_COLUMNS,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
