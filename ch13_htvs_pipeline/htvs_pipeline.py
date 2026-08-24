"""
Chapter 13 hands-on project: a real, end-to-end tiered high-throughput
virtual screening (HTVS) funnel against a real, emerging viral target
-- SARS-CoV-2 main protease (Mpro / 3CLpro) -- built entirely from
techniques this book already validated individually in earlier
chapters, now chained together the way an actual screening campaign
would use them: fast, cheap filters first, progressively more
expensive and more accurate methods only on whatever survives.

Extract: a real, potency-labeled screening library and a real receptor.
  - 74 real compounds from a published SARS-CoV-2 3CLpro biochemical
    IC50 assay (PubChem AID 1805203; Han et al., 2022), fetched live
    from the PubChem PUG REST API -- real SMILES, real measured IC50
    values, and PubChem's own real Active/Inactive call at its
    documented <=10 uM threshold, used here as ground truth for
    retrospective funnel validation (never shown to Tier 1's rule-based
    filter, which is target-agnostic by construction).
  - PDB 5R82 (Douangamath et al., 2020): a real SARS-CoV-2 main
    protease crystal structure in complex with a real fragment hit
    (Z219104216, PDB ligand code RZS), from the same published
    crystallographic fragment-screening campaign against this target.

Predict: three real, increasingly expensive tiers, run in sequence,
  each operating only on the previous tier's survivors --
  1. Tier 1 (fast, rule-based ADMET/drug-likeness filtering): Lipinski
     Ro5, Veber's rules, RDKit's PAINS structural-alert catalog, and a
     QED drug-likeness floor -- microseconds per compound, no
     structural or target information used at all.
  2. Tier 2 (real AutoDock Vina docking, Chapter 11's method):
     pocket-informed docking against the real 5R82 receptor, focused
     box centered on the real RZS crystallographic centroid, preceded
     by a real redocking validation control on RZS itself.
  3. Tier 3 (real ANI-2x molecular dynamics, Chapter 12's method): a
     real, short ligand-alone ANI-2x trajectory for only the top-ranked
     Tier 2 survivors, checking for a stable (non-diverging) trajectory
     rather than an unphysical one -- the scale Chapter 12's own
     measured throughput established as tractable.

Evaluate: real, quantitative funnel statistics computed directly from
  the real outputs above, retrospectively validated against the real
  PubChem Active/Inactive labels -- Tier 1 retention by label, Tier 2
  docking-score-vs-potency Spearman correlation, and an enrichment
  factor for real actives in the final Tier 3 shortlist versus random
  selection from the starting library.

See README.md for usage and chapter.md Section 13.3 for full context.
"""
import argparse
import csv
import io
import json
import math
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolAlign
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from scipy.stats import spearmanr

from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RECEPTOR_PDB = DATA_DIR / "5R82.pdb"
LIBRARY_CACHE = DATA_DIR / "sars_cov2_3clpro_library.json"

# PubChem AID 1805203: "SARS-CoV-1 and -2 3CLpro Biochemical Assay from
# Article 10.1021/acs.jmedchem.1c00598" (Han et al., 2022) -- real,
# published IC50 dose-response data, PubChem's own Active/Inactive call
# at its documented <=10 uM threshold.
PUBCHEM_ASSAY_CSV_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
MPRO_ASSAY_AID = 1805203
MAX_LIGAND_MW = 700.0  # drug-likeness sanity filter, consistent with Chapters 2/11

# PDB 5R82's co-crystallized fragment ligand: Z219104216, PDB residue
# code RZS. Canonical SMILES independently confirmed by parsing the
# official PDB Chemical Component Dictionary's RZS_ideal.sdf with
# RDKit (see chapter.md Section 13.3) -- not taken on faith from a
# secondary source.
NATIVE_LIGAND_RESN = "RZS"
NATIVE_LIGAND_SMILES = "CCNc1ccc(C#N)cn1"

FOCUSED_BOX_SIZE = 22.5  # Angstrom cube, same pocket-informed protocol as Chapter 11
VINA_EXHAUSTIVENESS = 8  # AutoDock Vina's own documented default (Trott & Olson, 2010)
VINA_NUM_MODES = 9
VINA_SEED = 42
REDOCK_SUCCESS_RMSD_A = 2.0  # conventional CASF/Vina-literature "correct pose" threshold

# ANI-2x's real, official element coverage (Devereux et al., 2020) --
# H, C, N, O, F, Cl, S. Checked per-compound, not assumed (Chapter 12
# Section 12.3).
ANI2X_ELEMENTS = {"H", "C", "N", "O", "F", "Cl", "S"}

# Tier 1 rule-based thresholds -- established medicinal-chemistry
# heuristics, not fit to this chapter's own library.
LIPINSKI_MAX_MW = 500.0
LIPINSKI_MAX_LOGP = 5.0
LIPINSKI_MAX_HBD = 5
LIPINSKI_MAX_HBA = 10
LIPINSKI_MAX_VIOLATIONS = 1  # standard Ro5 convention: "one violation is still acceptable"
VEBER_MAX_ROTB = 10
VEBER_MAX_TPSA = 140.0
QED_MIN = 0.30  # a lenient QED floor (Bickerton et al., 2012), consistent with Tier 1 being a coarse filter

TOP_N_FOR_TIER3 = 8  # real, disclosed compute-budget choice -- see chapter.md Section 13.3
MD_STEPS = 2_000  # 2 ps at 1 fs/step -- "short" stability check, Chapter 12's ligand-alone tractable scale
MD_REPORT_INTERVAL = 50
MD_TEMPERATURE_K = 300.0
MD_FRICTION_PER_PS = 1.0
MD_TIMESTEP_FS = 1.0


# --------------------------------------------------------------------------
# Real data acquisition: PubChem SARS-CoV-2 3CLpro screening library
# --------------------------------------------------------------------------


def fetch_3clpro_assay_csv(aid: int = MPRO_ASSAY_AID, timeout: int = 60) -> str:
    """Fetch the real AID 1805203 assay data table (SMILES, CID,
    Active/Inactive outcome, IC50) live from the PubChem PUG REST API."""
    response = requests.get(PUBCHEM_ASSAY_CSV_URL.format(aid=aid), timeout=timeout)
    response.raise_for_status()
    return response.text


def curate_library(csv_text: str) -> list[dict]:
    """Parse the real PubChem assay CSV into one record per real,
    RDKit-sanitizable, single-fragment, drug-likeness-sane *compound*
    (deduplicated by PubChem CID -- AID 1805203's own real assay
    protocol runs every compound in two independent dilution-series
    replicate wells, columns 3-12 and 13-22, so the raw CSV carries two
    real SIDs, and two real, independently measured IC50 values, per
    real molecule; deduplicating to one record per CID here matches
    Chapter 11's `curate_benchmark_set` real-replicate-handling method
    exactly). Each compound's real pIC50 is the median of its real
    replicate IC50 measurements (-log10, in molar); its is_active call
    is the median IC50 compared against the assay's own documented
    <=10 uM threshold -- computed directly rather than trusting a
    single replicate row's own PubChem Active/Inactive flag, which can
    occasionally disagree between a compound's two real replicates
    right at that boundary (see chapter.md Section 13.3)."""
    rows = [r for r in csv.DictReader(io.StringIO(csv_text)) if r.get("PUBCHEM_SID", "").isdigit()]
    by_cid: dict[str, dict] = {}
    for row in rows:
        smiles = row.get("PUBCHEM_EXT_DATASOURCE_SMILES", "")
        cid = row.get("PUBCHEM_CID", "")
        ic50_um_raw = row.get("PubChem Standard Value", "")
        if not smiles or not cid or not ic50_um_raw:
            continue
        by_cid.setdefault(cid, {"smiles": smiles, "ic50_values_uM": []})
        by_cid[cid]["ic50_values_uM"].append(float(ic50_um_raw))

    curated = []
    for cid, rec in by_cid.items():
        mol = Chem.MolFromSmiles(rec["smiles"])
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            continue
        mw = Descriptors.MolWt(mol)
        if mw > MAX_LIGAND_MW:
            continue
        ic50_um = float(np.median(rec["ic50_values_uM"]))
        elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
        curated.append(
            {
                "pubchem_cid": cid,
                "smiles": Chem.MolToSmiles(mol),
                "is_active": ic50_um <= 10.0,
                "ic50_uM": round(ic50_um, 4),
                "n_replicate_measurements": len(rec["ic50_values_uM"]),
                "pIC50": round(6.0 - math.log10(ic50_um), 4),
                "molecular_weight": round(mw, 2),
                "ani2x_compatible_elements": elements.issubset(ANI2X_ELEMENTS),
            }
        )
    curated.sort(key=lambda r: int(r["pubchem_cid"]))
    return curated


def load_or_build_library(refresh: bool = False) -> list[dict]:
    """Load the cached, real curated PubChem 3CLpro library if present
    (bundled in data/ for offline reproducibility); otherwise fetch
    live from PubChem and cache the result."""
    if LIBRARY_CACHE.exists() and not refresh:
        return json.loads(LIBRARY_CACHE.read_text(encoding="utf-8"))
    csv_text = fetch_3clpro_assay_csv()
    library = curate_library(csv_text)
    LIBRARY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LIBRARY_CACHE.write_text(json.dumps(library, indent=2), encoding="utf-8")
    return library


# --------------------------------------------------------------------------
# Tier 1: fast, rule-based ADMET/drug-likeness filtering
# --------------------------------------------------------------------------

_PAINS_CATALOG = None


def _pains_catalog() -> FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog(params)
    return _PAINS_CATALOG


def compute_tier1_properties(smiles: str) -> dict | None:
    """Real, instantly-computable physicochemical/drug-likeness
    properties for one compound -- no docking, no target structure, no
    activity data. Returns None if the SMILES does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    qed = QED.qed(mol)
    pains_alert = _pains_catalog().HasMatch(mol)

    lipinski_violations = sum(
        [mw > LIPINSKI_MAX_MW, logp > LIPINSKI_MAX_LOGP, hbd > LIPINSKI_MAX_HBD, hba > LIPINSKI_MAX_HBA]
    )
    veber_pass = rotb <= VEBER_MAX_ROTB and tpsa <= VEBER_MAX_TPSA

    return {
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 3),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 2),
        "rotatable_bonds": rotb,
        "qed": round(qed, 4),
        "pains_alert": bool(pains_alert),
        "lipinski_violations": lipinski_violations,
        "veber_pass": veber_pass,
        "passes_tier1": bool(
            lipinski_violations <= LIPINSKI_MAX_VIOLATIONS and veber_pass and not pains_alert and qed >= QED_MIN
        ),
    }


def run_tier1(library: list[dict]) -> list[dict]:
    """Annotate every real library compound with its real Tier 1
    properties and pass/fail call. Returns the full annotated library
    (not just survivors) so downstream analysis can compare retained
    vs. rejected compounds by real ground-truth label."""
    annotated = []
    for compound in library:
        record = dict(compound)
        props = compute_tier1_properties(compound["smiles"])
        record["tier1"] = props
        annotated.append(record)
    return annotated


# --------------------------------------------------------------------------
# Real receptor and pocket geometry (PDB 5R82)
# --------------------------------------------------------------------------


def split_receptor_and_native_ligand(pdb_path: Path, workdir: Path) -> tuple[Path, Path]:
    """Split the real PDB 5R82 file into a protein-only file (all
    `ATOM` records) and the co-crystallized fragment ligand's real
    `HETATM` records (residue name RZS; DMS/HOH crystallization
    additives excluded)."""
    lines = pdb_path.read_text().splitlines(keepends=True)
    protein = [l for l in lines if l.startswith("ATOM")]
    ligand = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() == NATIVE_LIGAND_RESN]
    if not ligand:
        raise ValueError(f"No {NATIVE_LIGAND_RESN} HETATM records found in {pdb_path}")
    receptor_path = workdir / "receptor_raw.pdb"
    ligand_path = workdir / "native_ligand.pdb"
    receptor_path.write_text("".join(protein) + "END\n")
    ligand_path.write_text("".join(ligand) + "END\n")
    return receptor_path, ligand_path


def pdb_atom_coords(pdb_path: Path) -> np.ndarray:
    coords = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(coords)


def compute_focused_box(native_ligand_pdb: Path) -> dict:
    """The real, pocket-informed search box: a fixed-size cube centered
    on RZS's real crystallographic centroid (Chapter 11's protocol)."""
    ligand_coords = pdb_atom_coords(native_ligand_pdb)
    center = ligand_coords.mean(axis=0).tolist()
    return {"center": center, "size": [FOCUSED_BOX_SIZE] * 3}


def prepare_receptor_pdbqt(receptor_pdb: Path, out_path: Path, ph: float = 7.4) -> Path:
    """Real receptor preparation via OpenBabel: protonate at pH 7.4,
    assign Gasteiger partial charges, write a rigid AutoDock PDBQT --
    identical protocol to Chapter 11."""
    subprocess.run(
        ["obabel", str(receptor_pdb), "-O", str(out_path), "-xr", "-p", str(ph)],
        check=True, capture_output=True, text=True,
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("OpenBabel receptor preparation produced no output")
    return out_path


# --------------------------------------------------------------------------
# Real ligand preparation (RDKit 3D embedding + meeko PDBQT typing)
# --------------------------------------------------------------------------


def prepare_ligand_pdbqt(smiles: str, seed: int = VINA_SEED) -> str | None:
    """Embed a real 3D conformer (ETKDG + MMFF94) and convert it to an
    AutoDock PDBQT string via meeko -- identical method to Chapter 11."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    cid = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
    if cid < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except ValueError:
        pass
    mol_setups = MoleculePreparation().prepare(mol)
    pdbqt_string, is_ok, _err = PDBQTWriterLegacy.write_string(mol_setups[0])
    return pdbqt_string if is_ok else None


# --------------------------------------------------------------------------
# Real AutoDock Vina execution (Tier 2)
# --------------------------------------------------------------------------


def locate_vina_executable() -> str | None:
    import os

    env_path = os.environ.get("VINA_EXECUTABLE")
    if env_path and Path(env_path).exists():
        return env_path
    return shutil.which("vina")


VINA_RESULT_RE = re.compile(r"REMARK VINA RESULT:\s*(-?\d+\.\d+)")


def parse_best_affinity(pdbqt_text: str) -> float | None:
    match = VINA_RESULT_RE.search(pdbqt_text)
    return float(match.group(1)) if match else None


def run_vina_docking(
    receptor_pdbqt: Path, ligand_pdbqt_text: str, center: list[float], box_size: list[float],
    workdir: Path, tag: str, exhaustiveness: int = VINA_EXHAUSTIVENESS, num_modes: int = VINA_NUM_MODES,
    seed: int = VINA_SEED,
) -> dict:
    """Run one real AutoDock Vina docking job -- identical
    bindings-then-CLI-fallback protocol as Chapter 11 (no `vina`
    Windows wheel exists; see that chapter's feasibility note)."""
    ligand_path = workdir / f"{tag}_ligand.pdbqt"
    out_path = workdir / f"{tag}_out.pdbqt"

    if ligand_path.exists() and out_path.exists() and parse_best_affinity(out_path.read_text()) is not None:
        pose_text = out_path.read_text()
        real_wall_time = max(0.0, out_path.stat().st_mtime - ligand_path.stat().st_mtime)
        return {"affinity_kcal_mol": parse_best_affinity(pose_text), "wall_time_s": round(real_wall_time, 2), "engine": "resumed", "pose_text": pose_text}

    ligand_path.write_text(ligand_pdbqt_text)
    start = time.perf_counter()
    error = None
    try:
        from vina import Vina

        v = Vina(sf_name="vina", seed=seed, cpu=1, verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(ligand_path))
        v.compute_vina_maps(center=center, box_size=box_size)
        v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
        v.write_poses(str(out_path), n_poses=num_modes, overwrite=True)
        engine = "vina-python-bindings"
    except ImportError:
        vina_bin = locate_vina_executable()
        if vina_bin is None:
            raise RuntimeError(
                "Neither the `vina` Python package nor a standalone Vina executable "
                "(VINA_EXECUTABLE env var / PATH) is available."
            )
        engine = "vina-cli-binary"
        try:
            subprocess.run(
                [
                    vina_bin,
                    "--receptor", str(receptor_pdbqt),
                    "--ligand", str(ligand_path),
                    "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
                    "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
                    "--exhaustiveness", str(exhaustiveness),
                    "--num_modes", str(num_modes),
                    "--seed", str(seed),
                    "--cpu", "1",
                    "--out", str(out_path),
                ],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            error = (exc.stderr or str(exc))[-500:]
    wall_time_s = time.perf_counter() - start

    if error is not None:
        return {"affinity_kcal_mol": None, "wall_time_s": round(wall_time_s, 2), "engine": engine, "pose_text": "", "error": error}
    pose_text = out_path.read_text() if out_path.exists() else ""
    return {"affinity_kcal_mol": parse_best_affinity(pose_text), "wall_time_s": round(wall_time_s, 2), "engine": engine, "pose_text": pose_text}


def redocking_validation(receptor_pdbqt: Path, native_ligand_pdb: Path, box: dict, workdir: Path, n_replicates: int = 3) -> dict:
    """Real redocking self-consistency control on the real 5R82
    receptor: dock RZS back into its own real crystallographic pocket
    and compute the top pose's heavy-atom RMSD to the real crystal
    structure -- validates this chapter's own docking setup on this
    new receptor before trusting it for Tier 2, exactly as Chapter 11
    validated its EGFR setup on erlotinib before trusting it on the
    ChEMBL benchmark."""
    ligand_pdbqt = prepare_ligand_pdbqt(NATIVE_LIGAND_SMILES, seed=VINA_SEED)
    if ligand_pdbqt is None:
        raise RuntimeError("Failed to prepare the native ligand (RZS) for redocking")

    template = Chem.MolFromSmiles(NATIVE_LIGAND_SMILES)
    ref_raw = Chem.MolFromPDBFile(str(native_ligand_pdb), removeHs=True, sanitize=False)
    ref_mol = Chem.RemoveHs(AllChem.AssignBondOrdersFromTemplate(template, ref_raw))

    replicates = []
    engine = None
    for i in range(n_replicates):
        result = run_vina_docking(receptor_pdbqt, ligand_pdbqt, box["center"], box["size"], workdir, tag=f"redock_{i}", seed=VINA_SEED + i)
        engine = result["engine"]
        if result["affinity_kcal_mol"] is None:
            continue
        pdbqt_mol = PDBQTMolecule.from_file(str(workdir / f"redock_{i}_out.pdbqt"), skip_typing=True)
        pose_mol = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0])
        rmsd_a = rdMolAlign.GetBestRMS(pose_mol, ref_mol, prbId=0, refId=0)
        replicates.append({
            "affinity_kcal_mol": result["affinity_kcal_mol"],
            "wall_time_s": result["wall_time_s"],
            "top_pose_rmsd_to_crystal_A": round(rmsd_a, 3),
            "correct_pose": bool(rmsd_a < REDOCK_SUCCESS_RMSD_A),
        })

    rmsds = [r["top_pose_rmsd_to_crystal_A"] for r in replicates]
    affinities = [r["affinity_kcal_mol"] for r in replicates]
    return {
        "engine": engine,
        "n_replicates_requested": n_replicates,
        "n_replicates": len(replicates),
        "replicates": replicates,
        "affinity_kcal_mol_mean": round(float(np.mean(affinities)), 3) if affinities else None,
        "affinity_kcal_mol_std": round(float(np.std(affinities)), 3) if affinities else None,
        "rmsd_to_crystal_A_mean": round(float(np.mean(rmsds)), 3) if rmsds else None,
        "rmsd_to_crystal_A_std": round(float(np.std(rmsds)), 3) if rmsds else None,
        "n_correct_pose": sum(r["correct_pose"] for r in replicates),
    }


# --------------------------------------------------------------------------
# Tier 2 orchestration: dock every Tier 1 survivor, parallelized
# --------------------------------------------------------------------------


def _dock_one_compound(args: tuple) -> dict:
    compound, receptor_pdbqt_str, box, workdir_str = args
    workdir = Path(workdir_str)
    receptor_pdbqt = Path(receptor_pdbqt_str)
    record = dict(compound)

    ligand_pdbqt = prepare_ligand_pdbqt(compound["smiles"], seed=VINA_SEED)
    if ligand_pdbqt is None:
        record["tier2"] = {"error": "3D embedding or PDBQT preparation failed"}
        return record

    tag = compound["pubchem_cid"]
    result = run_vina_docking(receptor_pdbqt, ligand_pdbqt, box["center"], box["size"], workdir, tag=f"{tag}_focused")
    record["tier2"] = {
        "affinity_kcal_mol": result["affinity_kcal_mol"],
        "wall_time_s": result["wall_time_s"],
        "engine": result["engine"],
    }
    return record


def run_tier2(survivors: list[dict], receptor_pdbqt: Path, box: dict, workdir: Path, n_workers: int = 4) -> list[dict]:
    receptor_str, workdir_str = str(receptor_pdbqt), str(workdir)
    jobs = [(c, receptor_str, box, workdir_str) for c in survivors]
    records = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_dock_one_compound, job): job[0]["pubchem_cid"] for job in jobs}
        for future in as_completed(futures):
            records.append(future.result())
    order = {c["pubchem_cid"]: i for i, c in enumerate(survivors)}
    records.sort(key=lambda r: order[r["pubchem_cid"]])
    return records


# --------------------------------------------------------------------------
# Tier 3: real, short ANI-2x ligand-alone MD stability check
# --------------------------------------------------------------------------


def kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Real, standard Kabsch superposition (identical to Chapter 12)."""
    mobile_c = mobile - mobile.mean(axis=0)
    reference_c = reference - reference.mean(axis=0)
    h = mobile_c.T @ reference_c
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1, 1, d]) @ u.T
    return (rotation @ mobile_c.T).T + reference.mean(axis=0)


def compute_rmsd_trajectory(frames: np.ndarray) -> dict:
    """Real, Kabsch-aligned RMSD per frame (identical method to
    Chapter 12), used here as a stability check rather than a
    structural-fluctuation study: a bounded, non-diverging RMSD
    trajectory is evidence the real ANI-2x potential integrates this
    docked pose without a numerical blow-up; it is not, by itself,
    evidence of genuine binding stability (Chapter 12's own
    Limitations discussion applies identically here)."""
    reference = frames[0]
    aligned = np.array([kabsch_align(f, reference) for f in frames])
    rmsd_per_frame = np.sqrt(((aligned - reference) ** 2).sum(axis=2).mean(axis=1)) * 10.0  # nm -> A
    return {
        "rmsd_per_frame_A": [round(float(x), 4) for x in rmsd_per_frame],
        "rmsd_mean_A": round(float(rmsd_per_frame.mean()), 4),
        "rmsd_max_A": round(float(rmsd_per_frame.max()), 4),
    }


def build_ligand_topology_from_smiles(smiles: str, seed: int = VINA_SEED):
    """Real 3D topology + positions for OpenMM/TorchANI, embedded fresh
    from SMILES (ETKDG + MMFF94) -- same method Chapter 12 used for its
    own SMILES-only ligand path."""
    from openmm import unit
    from openmm.app import Element, Topology

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol)

    top = Topology()
    chain = top.addChain()
    res = top.addResidue("LIG", chain)
    atom_map = {}
    conf = mol.GetConformer()
    positions_nm = []
    for atom in mol.GetAtoms():
        omm_atom = top.addAtom(atom.GetSymbol(), Element.getBySymbol(atom.GetSymbol()), res)
        atom_map[atom.GetIdx()] = omm_atom
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions_nm.append([pos.x * 0.1, pos.y * 0.1, pos.z * 0.1])
    for bond in mol.GetBonds():
        top.addBond(atom_map[bond.GetBeginAtomIdx()], atom_map[bond.GetEndAtomIdx()])
    positions = np.array(positions_nm) * unit.nanometer
    return top, positions


def run_ani2x_md(topology, positions, n_steps: int, report_interval: int, seed: int = VINA_SEED) -> dict:
    """Real, short Langevin dynamics under the real, official ANI-2x
    potential (identical protocol to Chapter 12) -- this chapter's
    Tier 3 stability check."""
    import openmm
    from openmm import unit
    from openmmml import MLPotential

    potential = MLPotential("ani2x")
    system = potential.createSystem(topology)
    integrator = openmm.LangevinMiddleIntegrator(
        MD_TEMPERATURE_K * unit.kelvin, MD_FRICTION_PER_PS / unit.picosecond, MD_TIMESTEP_FS * unit.femtosecond
    )
    integrator.setRandomNumberSeed(seed)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=100)

    frames = []
    t0 = time.perf_counter()
    n_reports = n_steps // report_interval
    for _ in range(n_reports):
        integrator.step(report_interval)
        state = context.getState(getPositions=True)
        frames.append(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    wall_time_s = time.perf_counter() - t0

    return {
        "n_atoms": topology.getNumAtoms(),
        "n_steps": n_reports * report_interval,
        "n_frames": len(frames),
        "wall_time_s": round(wall_time_s, 2),
        "ms_per_step": round(wall_time_s / (n_reports * report_interval) * 1000, 2) if n_reports else None,
        "frames": np.array(frames),
    }


def run_tier3(candidates: list[dict], n_steps: int = MD_STEPS, report_interval: int = MD_REPORT_INTERVAL) -> list[dict]:
    """Real, short ANI-2x ligand-alone MD for each Tier 2 survivor
    selected for Tier 3, skipping any compound whose elements fall
    outside ANI-2x's real trained coverage (Chapter 12 Section 12.3) --
    a real, disclosed tooling constraint, checked per-compound rather
    than assumed."""
    records = []
    for compound in candidates:
        record = dict(compound)
        if not compound.get("ani2x_compatible_elements", True):
            record["tier3"] = {"skipped": True, "reason": "elements outside ANI-2x's trained coverage"}
            records.append(record)
            continue
        try:
            top, positions = build_ligand_topology_from_smiles(compound["smiles"], seed=VINA_SEED)
            result = run_ani2x_md(top, positions, n_steps=n_steps, report_interval=report_interval)
            frames = result.pop("frames")
            analysis = compute_rmsd_trajectory(frames) if len(frames) > 1 else None
            record["tier3"] = {**result, "analysis": analysis, "stable": bool(analysis and analysis["rmsd_max_A"] < 15.0)}
        except Exception as exc:  # a real, disclosed per-compound failure -- does not abort the batch
            record["tier3"] = {"error": str(exc)[-300:]}
        records.append(record)
    return records


# --------------------------------------------------------------------------
# Real, quantitative funnel analysis (retrospective validation)
# --------------------------------------------------------------------------


def analyze_funnel(library: list[dict], tier1_records: list[dict], tier2_records: list[dict], tier3_records: list[dict]) -> dict:
    n_total = len(library)
    n_actives_total = sum(1 for c in library if c["is_active"])

    tier1_pass = [r for r in tier1_records if r["tier1"] and r["tier1"]["passes_tier1"]]
    tier1_pass_active = sum(1 for r in tier1_pass if r["is_active"])

    tier2_valid = [r for r in tier2_records if r["tier2"].get("affinity_kcal_mol") is not None]
    affinities = [r["tier2"]["affinity_kcal_mol"] for r in tier2_valid]
    pic50s = [r["pIC50"] for r in tier2_valid]
    potency_corr = spearmanr(affinities, pic50s) if len(tier2_valid) >= 3 else None

    tier3_stable = [r for r in tier3_records if r.get("tier3", {}).get("stable")]
    n_actives_in_tier3 = sum(1 for r in tier3_records if r["is_active"])
    n_actives_in_tier3_stable = sum(1 for r in tier3_stable if r["is_active"])

    library_hit_rate = n_actives_total / n_total if n_total else None
    tier3_hit_rate = n_actives_in_tier3 / len(tier3_records) if tier3_records else None
    enrichment_factor = (tier3_hit_rate / library_hit_rate) if (library_hit_rate and tier3_hit_rate is not None) else None

    return {
        "starting_library": {"n_compounds": n_total, "n_active": n_actives_total, "n_inactive": n_total - n_actives_total},
        "tier1_rule_based_filter": {
            "n_survivors": len(tier1_pass),
            "n_rejected": n_total - len(tier1_pass),
            "n_active_survivors": tier1_pass_active,
            "n_active_rejected": n_actives_total - tier1_pass_active,
            "active_retention_rate": round(tier1_pass_active / n_actives_total, 4) if n_actives_total else None,
            "overall_retention_rate": round(len(tier1_pass) / n_total, 4) if n_total else None,
        },
        "tier2_vina_docking": {
            "n_docked": len(tier2_records),
            "n_docked_successfully": len(tier2_valid),
            "affinity_kcal_mol_mean": round(float(np.mean(affinities)), 3) if affinities else None,
            "affinity_kcal_mol_std": round(float(np.std(affinities)), 3) if affinities else None,
            "spearman_affinity_vs_pIC50": {
                "rho": round(float(potency_corr.statistic), 3),
                "p_value": round(float(potency_corr.pvalue), 4),
            } if potency_corr is not None else None,
        },
        "tier3_ani2x_md": {
            "n_candidates": len(tier3_records),
            "n_stable": len(tier3_stable),
            "n_active_among_candidates": n_actives_in_tier3,
            "n_active_among_stable": n_actives_in_tier3_stable,
        },
        "retrospective_enrichment": {
            "library_active_rate": round(library_hit_rate, 4) if library_hit_rate else None,
            "tier3_shortlist_active_rate": round(tier3_hit_rate, 4) if tier3_hit_rate is not None else None,
            "enrichment_factor": round(enrichment_factor, 3) if enrichment_factor is not None else None,
        },
    }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-cache", action="store_true", help="Re-fetch the PubChem library live instead of using the bundled cache")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--top-n-tier3", type=int, default=TOP_N_FOR_TIER3)
    parser.add_argument("--md-steps", type=int, default=MD_STEPS)
    parser.add_argument("--skip-tier3", action="store_true")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "htvs_results.json")
    args = parser.parse_args()

    workdir = args.workdir or (RESULTS_DIR / "_scratch")
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading the real SARS-CoV-2 3CLpro screening library (PubChem AID 1805203)...")
    library = load_or_build_library(refresh=args.refresh_cache)
    n_active = sum(1 for c in library if c["is_active"])
    print(f"  {len(library)} real compounds ({n_active} active / {len(library) - n_active} inactive at PubChem's own <=10 uM call)")

    print("Preparing the real SARS-CoV-2 Mpro receptor (PDB 5R82)...")
    receptor_raw, native_ligand = split_receptor_and_native_ligand(RECEPTOR_PDB, workdir)
    receptor_pdbqt = prepare_receptor_pdbqt(receptor_raw, workdir / "receptor.pdbqt")
    box = compute_focused_box(native_ligand)
    print(f"  Focused box center {[round(c, 2) for c in box['center']]}, size {box['size']}")

    print("Running redocking validation control (RZS -> its own real pocket)...")
    redock = redocking_validation(receptor_pdbqt, native_ligand, box, workdir)
    print(f"  Redocked affinity {redock['affinity_kcal_mol_mean']} +/- {redock['affinity_kcal_mol_std']} kcal/mol, "
          f"RMSD to crystal pose {redock['rmsd_to_crystal_A_mean']} +/- {redock['rmsd_to_crystal_A_std']} A "
          f"({redock['n_correct_pose']}/{redock['n_replicates']} replicates <2.0 A)")

    print(f"Tier 1: real rule-based ADMET/drug-likeness filtering of all {len(library)} compounds...")
    tier1_records = run_tier1(library)
    survivors = [r for r in tier1_records if r["tier1"] and r["tier1"]["passes_tier1"]]
    print(f"  {len(survivors)}/{len(library)} compounds pass Tier 1")

    print(f"Tier 2: real AutoDock Vina docking of {len(survivors)} Tier 1 survivors against real 5R82...")
    t0 = time.perf_counter()
    tier2_records = run_tier2(survivors, receptor_pdbqt, box, workdir, n_workers=args.n_workers)
    print(f"  Done in {time.perf_counter() - t0:.1f} s wall-clock")

    tier2_valid = [r for r in tier2_records if r["tier2"].get("affinity_kcal_mol") is not None]
    tier2_valid.sort(key=lambda r: r["tier2"]["affinity_kcal_mol"])  # most favorable (most negative) first
    top_candidates = tier2_valid[: args.top_n_tier3]

    tier3_records = []
    if not args.skip_tier3:
        print(f"Tier 3: real, short ANI-2x MD stability check on the top {len(top_candidates)} Tier 2 survivors...")
        t0 = time.perf_counter()
        tier3_records = run_tier3(top_candidates, n_steps=args.md_steps)
        print(f"  Done in {time.perf_counter() - t0:.1f} s wall-clock")

    analysis = analyze_funnel(library, tier1_records, tier2_records, tier3_records)
    print(json.dumps(analysis, indent=2))

    output = {
        "target": {"pdb_id": "5R82", "protein": "SARS-CoV-2 main protease (Mpro/3CLpro)", "native_ligand": "Z219104216 (RZS)"},
        "assay_source": {"pubchem_aid": MPRO_ASSAY_AID, "reference": "Han et al., 2022, J. Med. Chem."},
        "docking_box": box,
        "redocking_validation": redock,
        "vina_settings": {"exhaustiveness": VINA_EXHAUSTIVENESS, "num_modes": VINA_NUM_MODES, "seed": VINA_SEED},
        "tier1_records": tier1_records,
        "tier2_records": tier2_records,
        "tier3_records": tier3_records,
        "analysis": analysis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
