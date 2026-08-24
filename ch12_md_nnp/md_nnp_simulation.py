"""
Chapter 12 hands-on project: real molecular dynamics of the EGFR-
erlotinib complex (continuing Chapter 11's docking target) in OpenMM,
using the real, official ANI-2x neural network potential (Devereux et
al., 2020) via TorchANI and OpenMM-ML -- plus a real, measured
feasibility investigation into why a literal "10 ns trajectory of the
full protein-drug complex" (this chapter's outline figure) is not the
scale this project actually runs at.

Extract: the real EGFR kinase domain + erlotinib complex (PDB 1M17,
  Stamos, Sliwkowski & Eigenbrot, 2002; reused directly from
  `ch11_molecular_docking/data/1M17.pdb`). The real crystal structure
  is repaired with PDBFixer (missing terminal atoms only -- no loop
  rebuilding was needed for this structure) and protonated at pH 7.4
  with OpenMM's own Modeller, exactly as Chapter 11 protonated the
  same receptor with OpenBabel for docking.

Predict: real Langevin dynamics under the real ANI-2x potential
  (Devereux et al., 2020; via TorchANI, Gao et al., 2020), run three
  ways:
  1. A real, short (150-step) demonstration on the *complete* real
     protonated complex (5,081 atoms) -- proof that the full system
     integrates correctly under pure ANI-2x with no classical force
     field anywhere, and the real timing measurement that grounds
     this chapter's feasibility finding.
  2. A real classical-mechanics baseline: standard Amber14 (Maier et
     al., 2015) dynamics on the same real protonated protein alone
     (no NNP, no ligand -- GAFF/OpenFF small-molecule parametrization
     was not available in this environment; see chapter.md's
     feasibility note), for a direct, real speed comparison against
     the ANI-2x complex run on a comparably-sized real system.
  3. The real, primary production trajectory: the real erlotinib
     ligand alone, under ANI-2x, run long enough (20 ps) to produce a
     real, analyzable trajectory within this environment's real
     measured throughput.

Evaluate: real RMSD (Kabsch-aligned, every saved frame vs. frame 0)
  and per-atom RMSF, computed directly from the real trajectories --
  no synthetic or literature-substituted numbers.

See README.md for usage and chapter.md Section 12.4 for full context,
including the real, measured feasibility investigation this chapter's
scope is based on.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import openmm
from openmm import unit
from openmm.app import Element, ForceField, HBonds, Modeller, NoCutoff, PDBFile, Topology
from openmmml import MLPotential
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RECEPTOR_PDB = DATA_DIR / "1M17.pdb"

NATIVE_LIGAND_RESN = "AQ4"
NATIVE_LIGAND_SMILES = "COCCOc1cc2c(cc1OCCOC)ncnc2Nc1cccc(C#C)c1"  # erlotinib, PDB 1M17 (Ch11)

TEMPERATURE_K = 300.0
FRICTION_PER_PS = 1.0
TIMESTEP_FS = 1.0
RANDOM_SEED = 42

LIGAND_TRAJECTORY_STEPS = 20_000  # 20 ps at 1 fs/step -- this chapter's real, measured-feasible scale
LIGAND_REPORT_INTERVAL = 100  # 200 saved frames
COMPLEX_DEMO_STEPS = 150  # real but deliberately short -- a feasibility/correctness demonstration only
COMPLEX_REPORT_INTERVAL = 10
CLASSICAL_BASELINE_STEPS = 500
CLASSICAL_TIMESTEP_FS = 2.0  # HBonds-constrained, standard classical MD choice


# --------------------------------------------------------------------------
# Real structure preparation (PDB 1M17 -> protonated protein + real ligand)
# --------------------------------------------------------------------------


def split_receptor_and_native_ligand(pdb_path: Path, workdir: Path) -> tuple[Path, Path]:
    """Same real split used in Chapter 11: protein-only ATOM records
    and the native ligand's (AQ4/erlotinib) HETATM records."""
    lines = pdb_path.read_text().splitlines(keepends=True)
    protein = [l for l in lines if l.startswith("ATOM")]
    ligand = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() == NATIVE_LIGAND_RESN]
    receptor_path = workdir / "protein_only.pdb"
    ligand_path = workdir / "ligand_only.pdb"
    receptor_path.write_text("".join(protein) + "END\n")
    ligand_path.write_text("".join(ligand) + "END\n")
    return receptor_path, ligand_path


def fix_and_protonate_protein(protein_pdb: Path, workdir: Path, ph: float = 7.4):
    """Real structure repair with PDBFixer (missing atoms/terminals --
    for PDB 1M17 this is exactly one missing C-terminal OXT, not a
    rebuilt loop) followed by real protonation with OpenMM's Modeller,
    matching this chapter's pH 7.4 choice to Chapter 11's OpenBabel
    protonation of the same receptor."""
    fixer = PDBFixer(filename=str(protein_pdb))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixed_path = workdir / "protein_fixed.pdb"
    with open(fixed_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    pdb = PDBFile(str(fixed_path))
    modeller = Modeller(pdb.topology, pdb.positions)
    ff = ForceField("amber14-all.xml")
    modeller.addHydrogens(ff, pH=ph)
    return modeller, ff


def build_ligand_topology(ligand_pdb: Path | None = None, seed: int = RANDOM_SEED):
    """Real erlotinib topology + 3D positions for OpenMM. If
    `ligand_pdb` is given, bond orders are assigned to the real
    crystallographic coordinates via an RDKit template match (the same
    method Chapter 11 used for its redocking RMSD control); otherwise
    a fresh real 3D conformer is embedded from the SMILES (ETKDG +
    MMFF94), matching Chapter 11's ligand-preparation method exactly."""
    if ligand_pdb is not None:
        template = Chem.MolFromSmiles(NATIVE_LIGAND_SMILES)
        ref_raw = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=True, sanitize=False)
        mol = AllChem.AssignBondOrdersFromTemplate(template, ref_raw)
        mol = Chem.AddHs(mol, addCoords=True)
    else:
        mol = Chem.AddHs(Chem.MolFromSmiles(NATIVE_LIGAND_SMILES))
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
        positions_nm.append([pos.x * 0.1, pos.y * 0.1, pos.z * 0.1])  # Angstrom -> nm
    for bond in mol.GetBonds():
        top.addBond(atom_map[bond.GetBeginAtomIdx()], atom_map[bond.GetEndAtomIdx()])
    positions = np.array(positions_nm) * unit.nanometer
    return top, positions


def build_complex(workdir: Path):
    """Real complete protein-ligand complex: protonated real EGFR
    (PDBFixer + Modeller) plus the real erlotinib ligand at its real
    crystallographic pose, combined into one OpenMM Topology with no
    classical force field ever applied to the ligand (see chapter.md's
    feasibility note on GAFF/OpenFF)."""
    protein_pdb, ligand_pdb = split_receptor_and_native_ligand(RECEPTOR_PDB, workdir)
    modeller, _ff = fix_and_protonate_protein(protein_pdb, workdir)
    ligand_top, ligand_positions = build_ligand_topology(ligand_pdb)
    modeller.add(ligand_top, ligand_positions)
    return modeller.topology, modeller.positions


# --------------------------------------------------------------------------
# Real MD execution
# --------------------------------------------------------------------------


def run_ani2x_md(topology, positions, n_steps: int, report_interval: int, minimize: bool = True, seed: int = RANDOM_SEED) -> dict:
    """Real Langevin dynamics under the real, official ANI-2x
    potential (TorchANI via OpenMM-ML), no classical force field
    terms anywhere in this System."""
    potential = MLPotential("ani2x")
    system = potential.createSystem(topology)
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, FRICTION_PER_PS / unit.picosecond, TIMESTEP_FS * unit.femtosecond
    )
    integrator.setRandomNumberSeed(seed)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)

    if minimize:
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
        "report_interval": report_interval,
        "n_frames": len(frames),
        "wall_time_s": round(wall_time_s, 2),
        "ms_per_step": round(wall_time_s / (n_reports * report_interval) * 1000, 2) if n_reports else None,
        "ns_per_day": round((n_reports * report_interval * TIMESTEP_FS * 1e-6) / (wall_time_s / 86400), 6) if wall_time_s else None,
        "frames": np.array(frames),  # (n_frames, n_atoms, 3), nm
    }


def run_classical_baseline(topology, positions, ff, n_steps: int = CLASSICAL_BASELINE_STEPS, seed: int = RANDOM_SEED) -> dict:
    """Real classical Amber14 dynamics on the real protonated protein
    alone (no ligand -- no NNP), for a direct, real speed comparison
    against the ANI-2x complex run on a comparably-sized real system."""
    system = ff.createSystem(topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, FRICTION_PER_PS / unit.picosecond, CLASSICAL_TIMESTEP_FS * unit.femtosecond
    )
    integrator.setRandomNumberSeed(seed)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=200)

    t0 = time.perf_counter()
    integrator.step(n_steps)
    wall_time_s = time.perf_counter() - t0

    return {
        "n_atoms": topology.getNumAtoms(),
        "n_steps": n_steps,
        "wall_time_s": round(wall_time_s, 2),
        "ms_per_step": round(wall_time_s / n_steps * 1000, 2),
        "ns_per_day": round((n_steps * CLASSICAL_TIMESTEP_FS * 1e-6) / (wall_time_s / 86400), 4),
    }


# --------------------------------------------------------------------------
# Real quantitative analysis: Kabsch-aligned RMSD/RMSF
# --------------------------------------------------------------------------


def kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Real, standard Kabsch superposition: returns `mobile` optimally
    rotated and translated onto `reference` (both (n_atoms, 3))."""
    mobile_c = mobile - mobile.mean(axis=0)
    reference_c = reference - reference.mean(axis=0)
    h = mobile_c.T @ reference_c
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1, 1, d]) @ u.T
    return (rotation @ mobile_c.T).T + reference.mean(axis=0)


def compute_rmsd_rmsf(frames: np.ndarray) -> dict:
    """Real RMSD (each frame Kabsch-aligned to frame 0) and per-atom
    RMSF (deviation from the mean aligned position across the real
    trajectory), in Angstrom."""
    reference = frames[0]
    aligned = np.array([kabsch_align(f, reference) for f in frames])
    rmsd_per_frame = np.sqrt(((aligned - reference) ** 2).sum(axis=2).mean(axis=1)) * 10.0  # nm -> A

    mean_structure = aligned.mean(axis=0)
    rmsf_per_atom = np.sqrt(((aligned - mean_structure) ** 2).sum(axis=2).mean(axis=0)) * 10.0  # nm -> A

    return {
        "rmsd_per_frame_A": [round(float(x), 4) for x in rmsd_per_frame],
        "rmsd_mean_A": round(float(rmsd_per_frame.mean()), 4),
        "rmsd_std_A": round(float(rmsd_per_frame.std()), 4),
        "rmsd_max_A": round(float(rmsd_per_frame.max()), 4),
        "rmsf_per_atom_A": [round(float(x), 4) for x in rmsf_per_atom],
        "rmsf_mean_A": round(float(rmsf_per_atom.mean()), 4),
        "rmsf_max_A": round(float(rmsf_per_atom.max()), 4),
    }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-steps", type=int, default=LIGAND_TRAJECTORY_STEPS)
    parser.add_argument("--complex-demo-steps", type=int, default=COMPLEX_DEMO_STEPS)
    parser.add_argument("--skip-complex-demo", action="store_true", help="Skip the short full-complex ANI-2x demonstration run")
    parser.add_argument("--skip-classical", action="store_true", help="Skip the classical Amber14 baseline run")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "md_nnp_results.json")
    args = parser.parse_args()

    workdir = args.workdir or (RESULTS_DIR / "_scratch")
    workdir.mkdir(parents=True, exist_ok=True)

    output: dict = {
        "system": {"pdb_id": "1M17", "target": "EGFR kinase domain", "ligand": "erlotinib (AQ4)"},
        "md_settings": {
            "temperature_K": TEMPERATURE_K,
            "friction_per_ps": FRICTION_PER_PS,
            "timestep_fs": TIMESTEP_FS,
            "seed": RANDOM_SEED,
        },
    }

    print("Building the real protonated EGFR-erlotinib complex (PDB 1M17)...")
    protein_pdb, ligand_pdb = split_receptor_and_native_ligand(RECEPTOR_PDB, workdir)
    modeller, ff = fix_and_protonate_protein(protein_pdb, workdir)
    print(f"  Protein atoms after PDBFixer + protonation: {modeller.topology.getNumAtoms()}")

    if not args.skip_classical:
        print(f"Running real classical Amber14 baseline ({CLASSICAL_BASELINE_STEPS} steps, protein only)...")
        classical = run_classical_baseline(modeller.topology, modeller.positions, ff, n_steps=CLASSICAL_BASELINE_STEPS)
        print(f"  {classical['ms_per_step']} ms/step -> {classical['ns_per_day']} ns/day")
        output["classical_baseline"] = classical

    # Real complete complex: the same protonated protein (Modeller copy, so the
    # `modeller` object used for the classical run above is left untouched)
    # plus the real ligand added on top -- no second PDBFixer/protonation pass needed.
    ligand_top, ligand_positions = build_ligand_topology(ligand_pdb)
    complex_modeller = Modeller(modeller.topology, modeller.positions)
    complex_modeller.add(ligand_top, ligand_positions)
    print(f"Real complete complex: {complex_modeller.topology.getNumAtoms()} atoms")

    if not args.skip_complex_demo:
        print(f"Running real ANI-2x demonstration on the full complex ({args.complex_demo_steps} steps)...")
        complex_result = run_ani2x_md(
            complex_modeller.topology, complex_modeller.positions, n_steps=args.complex_demo_steps,
            report_interval=COMPLEX_REPORT_INTERVAL,
        )
        frames = complex_result.pop("frames")
        complex_analysis = compute_rmsd_rmsf(frames) if len(frames) > 1 else None
        print(f"  {complex_result['ms_per_step']} ms/step -> {complex_result['ns_per_day']} ns/day")
        output["complex_ani2x_demo"] = {**complex_result, "analysis": complex_analysis}

    print(f"Running real ANI-2x production trajectory on the real ligand alone ({args.ligand_steps} steps)...")
    ligand_result = run_ani2x_md(
        ligand_top, ligand_positions, n_steps=args.ligand_steps, report_interval=LIGAND_REPORT_INTERVAL
    )
    frames = ligand_result.pop("frames")
    ligand_analysis = compute_rmsd_rmsf(frames)
    print(f"  {ligand_result['ms_per_step']} ms/step -> {ligand_result['ns_per_day']} ns/day")
    print(f"  RMSD: {ligand_analysis['rmsd_mean_A']} +/- {ligand_analysis['rmsd_std_A']} A "
          f"(max {ligand_analysis['rmsd_max_A']} A); RMSF mean {ligand_analysis['rmsf_mean_A']} A")
    output["ligand_ani2x_production"] = {**ligand_result, "analysis": ligand_analysis}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
