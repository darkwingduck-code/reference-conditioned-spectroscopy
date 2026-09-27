"""Targeted regression checks for principal-part versus full-response zeros."""
from __future__ import annotations

import numpy as np

from pole_zero_tomography import (
    fit_two_pole_rational,
    reconstruct_from_poles_residues,
    residues_from_bare,
    singular_response,
)


P1 = 0.18 - 0.020j
P2 = 0.42 - 0.030j
DARK = 0.30 - 0.010j
Z = np.linspace(0.05, 0.55, 401) + 0.004j


def nearest(roots: tuple[complex, ...], target: complex) -> complex:
    return min(roots, key=lambda root: abs(root - target))


def test_canonical_two_mode_equality() -> None:
    y = singular_response(Z, P1, P2, DARK, prefactor=0.8 - 0.1j)
    fit = fit_two_pole_rational(Z, y, background_order=0)
    assert fit.response_zero == fit.principal_part_zero
    assert abs(fit.principal_part_zero - DARK) < 2e-12
    assert abs(nearest(fit.full_response_zeros, DARK) - DARK) < 2e-12


def test_analytic_prefactor_counterexample_and_correction() -> None:
    a = 1.5
    y = (1.0 + a * Z) * (Z - DARK) / ((Z - P1) * (Z - P2))
    fit = fit_two_pole_rational(Z, y, background_order=0)
    assert abs(fit.principal_part_zero - DARK) > 1e-3
    assert abs(nearest(fit.full_response_zeros, DARK) - DARK) < 2e-12
    remote = nearest(fit.full_response_zeros, -1.0 / a)
    candidate = nearest(tuple(c.root for c in fit.zero_candidates), remote)
    diagnostic = next(c for c in fit.zero_candidates if c.root == candidate)
    assert not diagnostic.inside_window and not diagnostic.eligible


def test_constant_and_linear_analytic_backgrounds() -> None:
    cases = (
        (np.array([0.35 - 0.08j]), 0),
        (np.array([0.22 + 0.03j, 0.1]), 1),
    )
    denominator = np.poly([P1, P2])
    singular_numerator = 0.7 * np.array([1.0, -DARK])
    for background_coefficients, order in cases:
        bg = np.polyval(background_coefficients, Z)
        fit = fit_two_pole_rational(
            Z, bg + singular_response(Z, P1, P2, DARK, prefactor=0.7),
            background_order=order,
        )
        expected = np.roots(np.polyadd(
            np.polymul(background_coefficients, denominator),
            singular_numerator,
        ))
        assert abs(fit.principal_part_zero - DARK) < 3e-11
        assert max(min(abs(root-value) for value in fit.full_response_zeros)
                   for root in expected) < 3e-10


def test_normalization_response_scaling_and_pole_order_invariance() -> None:
    y = (1.0 + 1.2 * Z) * (Z - DARK) / ((Z - P1) * (Z - P2))
    norm = fit_two_pole_rational(Z, y, background_order=0, normalize_frequency=True)
    raw = fit_two_pole_rational(Z, 7.0j * y, background_order=0, normalize_frequency=False)
    assert abs(norm.principal_part_zero - raw.principal_part_zero) < 2e-10
    assert abs(nearest(norm.full_response_zeros, DARK) - nearest(raw.full_response_zeros, DARK)) < 2e-10
    rm, rp, poles = residues_from_bare(0.23 - .02j, DARK, 0.003, prefactor=0.8)
    forward = reconstruct_from_poles_residues(poles[0], poles[1], rm, rp)
    reverse = reconstruct_from_poles_residues(poles[1], poles[0], rp, rm)
    assert abs(forward.response_zero - reverse.response_zero) < 1e-14


def test_common_and_near_cancellation_are_excluded() -> None:
    for root in (P1, P1 + 1e-7):
        y = (Z - root) / ((Z - P1) * (Z - P2))
        fit = fit_two_pole_rational(Z, y, background_order=0)
        cancelled = [candidate for candidate in fit.zero_candidates if candidate.cancelled]
        assert cancelled
        assert all(np.isnan(candidate.slope.real) for candidate in cancelled)
        assert all(np.isinf(candidate.root_condition_number) for candidate in cancelled)
        assert not any(candidate.eligible for candidate in fit.zero_candidates)
        assert not fit.zero_assignment_reliable


def test_remote_third_mode_marks_model_boundary() -> None:
    far = 1.8 - 0.12j
    near = 0.70 - 0.04j
    def response(p3: complex) -> np.ndarray:
        return 1/(Z-P1) + 0.6/(Z-P2) + 0.15/(Z-p3)
    far_fit = fit_two_pole_rational(Z, response(far), background_order=1)
    near_fit = fit_two_pole_rational(Z, response(near), background_order=1)
    assert far_fit.relative_rms < 1e-4
    assert near_fit.relative_rms > 1e-4
    assert "large_fit_residual" in near_fit.zero_assignment_flags
    assert not near_fit.zero_assignment_reliable


def test_outside_window_root_is_never_eligible() -> None:
    y = (1.0 + 1.5 * Z) * (Z - DARK) / ((Z - P1) * (Z - P2))
    fit = fit_two_pole_rational(Z, y, background_order=0)
    outside = [candidate for candidate in fit.zero_candidates if not candidate.inside_window]
    assert outside and all(not candidate.eligible for candidate in outside)
    assert all("outside_measured_window" in candidate.reason for candidate in outside)


def test_probe_calibration_precedes_full_zero_assignment() -> None:
    chi = singular_response(Z, P1, P2, DARK, prefactor=0.8)
    coulomb = 2.4
    epsilon_inverse = 1.0 + coulomb * chi
    direct = fit_two_pole_rational(Z, epsilon_inverse, background_order=0)
    calibrated = fit_two_pole_rational(Z, (epsilon_inverse - 1.0) / coulomb, background_order=0)
    assert abs(direct.principal_part_zero - DARK) < 2e-12
    assert abs(nearest(direct.full_response_zeros, DARK) - DARK) > 1e-3
    assert abs(nearest(calibrated.full_response_zeros, DARK) - DARK) < 2e-12


def test_zero_residue_sum_retains_two_poles_but_has_no_finite_zero() -> None:
    # Opposite nonzero residues give -1 / ((z-1)(z-2)), not zero response.
    z = np.array([0.0, 0.5, 1.5, 3.0], dtype=complex)
    response = 1.0 / (z - 1.0) - 1.0 / (z - 2.0)
    numerator = response * (z - 1.0) * (z - 2.0)
    np.testing.assert_allclose(numerator, -1.0, rtol=0, atol=1e-14)
    assert np.all(response != 0)
    for pole, residue in ((1.0, 1.0), (2.0, -1.0)):
        dz = 1e-7j
        near = 1.0 / (pole + dz - 1.0) - 1.0 / (pole + dz - 2.0)
        assert abs(dz * near - residue) < 2e-7
    try:
        reconstruct_from_poles_residues(1.0, 2.0, 1.0, -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("C=0 must not be divided into a finite zero")


def test_degenerate_input_fails_explicitly() -> None:
    try:
        fit_two_pole_rational(Z, np.zeros_like(Z), background_order=1)
    except ValueError as exc:
        assert "numerator" in str(exc)
    else:
        raise AssertionError("degenerate response should fail")


TESTS = [
    test_canonical_two_mode_equality,
    test_analytic_prefactor_counterexample_and_correction,
    test_constant_and_linear_analytic_backgrounds,
    test_normalization_response_scaling_and_pole_order_invariance,
    test_common_and_near_cancellation_are_excluded,
    test_remote_third_mode_marks_model_boundary,
    test_outside_window_root_is_never_eligible,
    test_probe_calibration_precedes_full_zero_assignment,
    test_zero_residue_sum_retains_two_poles_but_has_no_finite_zero,
    test_degenerate_input_fails_explicitly,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(TESTS)} zero-assignment revision tests passed")
