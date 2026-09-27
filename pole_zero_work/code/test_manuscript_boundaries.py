"""Independent numerical checks of the manuscript's asymptotic boundaries.

These examples diagonalize concrete two-mode matrices, minimize their actual
splittings, and differentiate a nonlinear drift trajectory.  They do not use
the fitted-pole implementation or substitute formulas into themselves.
Run directly with Python; numpy and scipy are existing project dependencies.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize_scalar


def report(name: str, evidence: dict) -> None:
    print(json.dumps({"check": name, **evidence}, sort_keys=True))


def test_q_dependent_coupling_moves_minimum_at_second_order() -> None:
    # q is displacement from a finite q0.  Matrix entries are even/odd in B.
    dv, da, g0, g1, g2 = 1.7, 0.6, 0.8, 0.45, 0.18
    diagonal_crossing_coefficient = -da / dv
    expected = diagonal_crossing_coefficient - 4 * g0 * g1 / dv**2
    fields = np.array([0.08, 0.04, 0.02, 0.01])
    coefficients, gap_ratios = [], []
    for field in fields:
        def splitting_squared(q: float) -> float:
            detuning = dv * q + da * field**2
            coupling = (g0 + g1 * q + g2 * q**2) * field
            matrix = np.array([[detuning / 2, coupling],
                               [coupling, -detuning / 2]])
            poles = np.linalg.eigvalsh(matrix)
            return float((poles[1] - poles[0])**2)

        optimum = minimize_scalar(
            splitting_squared, bounds=(-0.12, 0.12), method="bounded",
            options={"xatol": 1e-15},
        )
        assert optimum.success
        coefficients.append(float(optimum.x / field**2))
        gap_ratios.append(float(np.sqrt(optimum.fun) / (2 * g0 * field)))
        # Independently verify the point is a local minimum of the spectrum.
        for offset in (-field**2 / 20, field**2 / 20):
            assert splitting_squared(optimum.x + offset) > optimum.fun

    coefficient_errors = np.abs(np.array(coefficients) - expected)
    gap_errors = np.abs(np.array(gap_ratios) - 1)
    # B-halving reduces the residual in each leading coefficient by four.
    assert np.all((coefficient_errors[:-1] / coefficient_errors[1:]) > 3.8)
    assert np.all((coefficient_errors[:-1] / coefficient_errors[1:]) < 4.2)
    assert np.all((gap_errors[:-1] / gap_errors[1:]) > 3.8)
    assert coefficient_errors[-1] < 6e-5
    assert gap_errors[-1] < 3.5e-5
    # Omitting the derivative of the coupling fails by a nonvanishing amount.
    assert abs(coefficients[-1] - diagonal_crossing_coefficient) > 0.49
    report("variable_q_coupling", {
        "fields": fields.tolist(), "qmin_over_B2": coefficients,
        "predicted_coefficient": expected,
        "diagonal_crossing_coefficient": diagonal_crossing_coefficient,
        "gap_over_2g0B": gap_ratios,
    })


def test_nonlinear_drift_requires_logarithmic_chain_rule() -> None:
    # x is proportional to v_Omega, not necessarily proportional to B.
    kappa, eta = 1.7, 0.6
    steps = np.array([0.02, 0.01, 0.005, 0.0025])
    evidence = []
    for field in (0.03, 0.2, 0.7):
        x = kappa * field * (1 + eta * field)
        chain_factor = (1 + 2 * eta * field) / (1 + eta * field)
        expected = chain_factor / (1 + x*x)

        def log_trajectory(log_offset: float) -> float:
            b = field * np.exp(log_offset)
            drift = kappa * b * (1 + eta * b)
            return float(np.log(np.sqrt(1 + drift**2) / drift))

        derivatives = np.array([
            -(log_trajectory(h) - log_trajectory(-h)) / (2*h)
            for h in steps
        ])
        errors = np.abs(derivatives - expected)
        assert np.all((errors[:-1] / errors[1:]) > 3.8)
        assert np.all((errors[:-1] / errors[1:]) < 4.2)
        assert errors[-1] < 6e-7
        assert abs(derivatives[-1] - 1 / (1 + x*x)) > 0.015
        evidence.append({"B": field, "predicted_zeta": expected,
                         "finite_difference_errors": errors.tolist()})

    # A quadratic drift correction gives an O(B), not O(B^2), relative
    # correction to q_h = const / B.  Check its nonzero limiting coefficient.
    fields = np.array([0.004, 0.002, 0.001, 0.0005])
    x = kappa * fields * (1 + eta * fields)
    qh = np.sqrt(1 + x*x) / x
    relative_over_B = (qh * kappa * fields - 1) / fields
    errors = np.abs(relative_over_B + eta)
    assert np.all((errors[:-1] / errors[1:]) > 1.9)
    assert np.all((errors[:-1] / errors[1:]) < 2.1)
    assert errors[-1] < 0.001
    report("nonlinear_drift", {"log_derivatives": evidence,
        "relative_correction_over_B": relative_over_B.tolist(),
        "predicted_relative_coefficient": -eta})


def bright_residues(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residues of [(z I - H)^-1]_{00} from biorthogonal eigenvectors."""
    poles, right = np.linalg.eig(matrix)
    left = np.linalg.inv(right)
    residues = right[0, :] * left[:, 0]
    np.testing.assert_allclose(np.sum(residues), 1, atol=2e-14, rtol=0)
    return poles, residues


def test_equal_real_parts_do_not_remove_complex_detuning() -> None:
    bright, dark = 0.3 - 0.042j, 0.3 - 0.012j
    couplings = np.array([0.003, 0.0015, 0.00075, 0.000375])
    relative_errors, dark_weights = [], []
    for coupling in couplings:
        poles, residues = bright_residues(
            np.array([[bright, coupling], [coupling, dark]], dtype=complex))
        index = int(np.argmin(np.abs(poles-dark)))
        asymptote = coupling**2 / (bright-dark)**2
        relative_errors.append(float(abs(residues[index] / asymptote - 1)))
        dark_weights.append(float(abs(residues[index])))
        # Below the unequal-linewidth exceptional-point threshold there is
        # no real-frequency pole splitting despite real-part bare resonance.
        np.testing.assert_allclose(poles.real, 0.3, atol=1e-14, rtol=0)
    ratios = np.array(relative_errors[:-1]) / relative_errors[1:]
    assert np.all((ratios > 3.9) & (ratios < 4.2))
    assert relative_errors[-1] < 4.8e-4
    measured_power = np.log2(np.array(dark_weights[:-1]) / dark_weights[1:])
    assert abs(measured_power[-1] - 2) < 0.003
    report("fixed_linewidth_mismatch", {
        "couplings": couplings.tolist(), "dark_residue_moduli": dark_weights,
        "relative_asymptotic_errors": relative_errors,
        "measured_residue_powers": measured_power.tolist(),
    })


def test_matched_or_coupling_scaled_linewidths_allow_order_one_residues() -> None:
    gamma = 0.025
    couplings = [0.003, 0.0015, 0.00075, 0.000375]
    scaled_mismatch_moduli = []
    for coupling in couplings:
        pole = 0.3 - 1j*gamma
        poles, residues = bright_residues(
            np.array([[pole, coupling], [coupling, pole]], dtype=complex))
        np.testing.assert_allclose(residues, 0.5, atol=2e-13, rtol=0)
        np.testing.assert_allclose(abs(poles[1]-poles[0]), 2*coupling,
                                   atol=2e-14, rtol=0)
        # A mismatch that shrinks with g also keeps complex detuning O(g).
        # The factor 0.6 stays away from the exceptional point at 2.
        mismatch = 0.6 * coupling
        _, residues = bright_residues(np.array([
            [pole - 0.5j*mismatch, coupling],
            [coupling, pole + 0.5j*mismatch]], dtype=complex))
        scaled_mismatch_moduli.append(float(abs(residues[0])))
        assert np.all(np.abs(residues) > 0.5)
    assert np.ptp(scaled_mismatch_moduli) < 2e-13
    report("matched_linewidths", {
        "couplings": couplings, "equal_linewidth_residues": [0.5, 0.5],
        "mismatch_over_g": 0.6,
        "scaled_mismatch_residue_moduli": scaled_mismatch_moduli,
    })


TESTS = [
    test_q_dependent_coupling_moves_minimum_at_second_order,
    test_nonlinear_drift_requires_logarithmic_chain_rule,
    test_equal_real_parts_do_not_remove_complex_detuning,
    test_matched_or_coupling_scaled_linewidths_allow_order_one_residues,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(TESTS)} manuscript boundary tests passed")
