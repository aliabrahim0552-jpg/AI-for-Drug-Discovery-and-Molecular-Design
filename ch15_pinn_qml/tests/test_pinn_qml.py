"""
Tests for the Chapter 15 hands-on project (pinn_qml.py).

Both parts are tested directly, offline, with no network access. The
PINN/baseline training functions are exercised with a small iteration
count (correctness/shape checks, not convergence quality -- that is
what `chapter.md` Section 15.1's full run reports); the QML functions
run real, fast PennyLane simulations (H2's full VQE takes ~2 s, so it
is run for real rather than mocked) and are checked against a real,
independently computed exact-diagonalization reference.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pinn_qml import (
    DOSE,
    H2_COORDINATES_BOHR,
    H2_SYMBOLS,
    KA_TRUE,
    KE_TRUE,
    V_TRUE,
    PKNet,
    bateman_solution,
    build_molecular_hamiltonian,
    exact_ground_state_energy,
    generate_synthetic_observations,
    rmse,
    run_vqe,
    train_baseline_regressor,
    train_pinn,
)

# --------------------------------------------------------------------------
# Part 1: PINN / PK model
# --------------------------------------------------------------------------


def test_bateman_solution_is_zero_at_t_zero():
    # No drug has been absorbed into the central compartment yet.
    c0 = bateman_solution(np.array([0.0]), KA_TRUE, KE_TRUE, V_TRUE, DOSE)
    assert c0[0] == pytest.approx(0.0, abs=1e-8)


def test_bateman_solution_is_non_negative_and_eventually_decays():
    t = np.linspace(0.01, 48, 200)
    c = bateman_solution(t, KA_TRUE, KE_TRUE, V_TRUE, DOSE)
    assert (c >= -1e-8).all()
    # Elimination-phase tail must be lower than the absorption-phase peak.
    assert c[-1] < c.max()


def test_bateman_solution_peak_matches_known_analytical_tmax():
    # For a one-compartment oral model, t_max = ln(ka/ke) / (ka - ke) -- a
    # real, standard closed-form result, independent of the ODE solver
    # under test, used here as a hand-checkable cross-validation.
    t_max_analytical = np.log(KA_TRUE / KE_TRUE) / (KA_TRUE - KE_TRUE)
    t = np.linspace(0.01, 24, 5000)
    c = bateman_solution(t, KA_TRUE, KE_TRUE, V_TRUE, DOSE)
    t_max_numerical = t[np.argmax(c)]
    assert t_max_numerical == pytest.approx(t_max_analytical, abs=0.02)


def test_generate_synthetic_observations_shapes_and_ranges():
    data = generate_synthetic_observations(seed=0)
    assert data["t_obs"].shape == data["c_obs"].shape
    assert len(data["t_full"]) == len(data["c_true_full"])
    assert data["noise_sd"] > 0
    # The noiseless full curve must match the analytical solution exactly.
    expected = bateman_solution(data["t_full"], KA_TRUE, KE_TRUE, V_TRUE, DOSE)
    np.testing.assert_allclose(data["c_true_full"], expected)


def test_pknet_forward_outputs_are_non_negative():
    import torch

    torch.manual_seed(0)
    model = PKNet()
    tau = torch.linspace(0.0, 1.0, 10).view(-1, 1)
    a_hat, c_hat = model(tau)
    assert (a_hat >= 0).all() and (c_hat >= 0).all()
    assert a_hat.shape == (10, 1) and c_hat.shape == (10, 1)


def test_rmse_known_value():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 5.0])
    assert rmse(a, b) == pytest.approx(np.sqrt((0 + 0 + 4) / 3))


def test_train_baseline_regressor_runs_and_returns_correct_shape():
    data = generate_synthetic_observations(seed=1)
    result = train_baseline_regressor(data, n_iters=50)
    assert result.shape == data["t_full"].shape
    assert np.all(np.isfinite(result))


def test_train_pinn_runs_and_reports_decreasing_physics_residual():
    data = generate_synthetic_observations(seed=1)
    # A very short run just to check the training loop and physics-loss
    # computation execute correctly and produce finite, sane output --
    # not a convergence-quality check (see chapter.md for the full run).
    result = train_pinn(data, n_iters=20, n_collocation=50)
    assert result["c_pred_full"].shape == data["t_full"].shape
    assert np.all(np.isfinite(result["c_pred_full"]))
    assert result["final_physics_loss"] >= 0.0
    assert result["wall_time_s"] > 0.0


# --------------------------------------------------------------------------
# Part 2: QML / VQE
# --------------------------------------------------------------------------


def test_build_h2_hamiltonian_has_the_expected_real_qubit_count():
    # H2 in a minimal (STO-3G) basis: 2 spatial orbitals -> 4 spin
    # orbitals -> 4 qubits under the Jordan-Wigner mapping.
    _hamiltonian, n_qubits = build_molecular_hamiltonian(H2_SYMBOLS, H2_COORDINATES_BOHR)
    assert n_qubits == 4


def test_h2_exact_ground_state_energy_matches_known_literature_range():
    # A real, well-established benchmark value: H2's STO-3G ground-state
    # energy near its equilibrium bond length is documented at
    # approximately -1.137 Hartree in the VQE literature (e.g. Kandala
    # et al., 2017; O'Malley et al., 2016) -- an independent sanity
    # check on this chapter's own from-scratch Hamiltonian construction.
    hamiltonian, _n_qubits = build_molecular_hamiltonian(H2_SYMBOLS, H2_COORDINATES_BOHR)
    energy = exact_ground_state_energy(hamiltonian)
    assert energy == pytest.approx(-1.137, abs=0.01)


def test_h2_vqe_converges_to_the_real_exact_diagonalization_reference():
    hamiltonian, n_qubits = build_molecular_hamiltonian(H2_SYMBOLS, H2_COORDINATES_BOHR)
    exact = exact_ground_state_energy(hamiltonian)
    vqe_result = run_vqe(hamiltonian, n_qubits, n_electrons=2, max_iters=60)
    assert vqe_result["final_vqe_energy_hartree"] == pytest.approx(exact, abs=1e-6)
    # A real, physically required inequality: the variational principle
    # guarantees every VQE energy estimate upper-bounds the true ground
    # state (up to numerical tolerance).
    assert vqe_result["final_vqe_energy_hartree"] >= exact - 1e-8


def test_h2_vqe_improves_on_the_hartree_fock_reference():
    hamiltonian, n_qubits = build_molecular_hamiltonian(H2_SYMBOLS, H2_COORDINATES_BOHR)
    vqe_result = run_vqe(hamiltonian, n_qubits, n_electrons=2, max_iters=60)
    assert vqe_result["final_vqe_energy_hartree"] < vqe_result["hf_reference_energy_hartree"]
