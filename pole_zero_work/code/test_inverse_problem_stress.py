"""Scientific regression checks for the bounded inverse-problem protocol."""
from __future__ import annotations

import numpy as np

from inverse_problem_stress import (
    CENTER,
    calibrated_target,
    loss_only_ambiguity,
    nonidentifiable_decompositions,
    select_window_root,
    target_root_from_exact_response,
    three_pole_full_zeros,
    two_mode_response,
)


Z = np.linspace(0.04, 0.56, 321).astype(complex)


def test_exact_known_model_and_global_probe_invariance() -> None:
    target = calibrated_target(Z)
    truth = target_root_from_exact_response(Z, target)
    base = select_window_root(Z, target)
    transformed = select_window_root(Z, (2.3-1.7j)*target)
    assert base.success and transformed.success
    assert abs(base.root-truth) < 2e-11
    assert abs(transformed.root-base.root) < 2e-11


def test_intrinsic_instrument_split_is_not_identifiable_from_spectrum() -> None:
    result = nonidentifiable_decompositions(Z)
    assert result["max_observed_difference"] < 1e-14
    assert result["target_root_separation"] > 1e-6


def test_phase_offset_invariance_but_phase_slope_bias() -> None:
    target = calibrated_target(Z)
    reference = select_window_root(Z, target)
    offset = select_window_root(Z, np.exp(0.37j)*target)
    slope = select_window_root(Z, np.exp(1j*0.18*(Z.real-CENTER))*target)
    assert reference.success and offset.success and slope.success
    assert abs(offset.root-reference.root) < 2e-11
    assert abs(slope.root-reference.root) > 1e-7


def test_root_outside_protocol_window_is_reported_as_failure() -> None:
    response = (Z-(0.82-0.01j))/((Z-(0.18-0.02j))*(Z-(0.42-0.03j)))
    estimate = select_window_root(Z, response)
    assert not estimate.success
    assert estimate.root is None
    assert estimate.reason == "no_eligible_window_root"


def test_omitted_remote_pole_exposes_model_mismatch() -> None:
    exact = two_mode_response(Z, 0.022)
    mismatch = exact + 0.08/(Z-(0.68-0.05j))
    exact_fit = select_window_root(Z, exact, halfwidths=(0.24,))
    mismatch_fit = select_window_root(Z, mismatch, halfwidths=(0.24,))
    assert exact_fit.success and mismatch_fit.success
    assert exact_fit.residual < 1e-10
    assert mismatch_fit.residual > 100*exact_fit.residual


def test_remote_physical_pole_moves_exact_full_zero() -> None:
    coupling = 0.022
    weight = 0.06
    remote = 0.72 - 0.055j
    roots = three_pole_full_zeros(coupling, weight, remote)
    full_zero = min(roots, key=lambda root: abs(root-CENTER))
    exact_value = two_mode_response(np.array([full_zero]), coupling)[0] + weight/(full_zero-remote)
    assert abs(full_zero-(0.31-0.014j)) > 1e-5
    assert abs(exact_value) < 2e-12
    assert three_pole_full_zeros(coupling, 0.0) == (0.31-0.014j,)


def test_loss_only_data_leave_distinct_full_zeros() -> None:
    ambiguity = loss_only_ambiguity(Z)
    assert ambiguity["max_loss_difference"] < 1e-14
    assert ambiguity["root_separation"] > 1e-5


def test_weak_coupling_reduces_exact_dark_residue() -> None:
    from pole_zero_tomography import fit_two_pole_rational

    residues = []
    for coupling in (0.026, 0.014, 0.006):
        fit = fit_two_pole_rational(Z, two_mode_response(Z, coupling), background_order=1)
        pole = max(fit.poles, key=lambda value: value.real)
        residues.append(abs(fit.residues[fit.poles.index(pole)]))
    assert residues[0] > residues[1] > residues[2]
    assert residues[0]/residues[2] > 10


TESTS = [
    test_exact_known_model_and_global_probe_invariance,
    test_intrinsic_instrument_split_is_not_identifiable_from_spectrum,
    test_phase_offset_invariance_but_phase_slope_bias,
    test_root_outside_protocol_window_is_reported_as_failure,
    test_omitted_remote_pole_exposes_model_mismatch,
    test_remote_physical_pole_moves_exact_full_zero,
    test_loss_only_data_leave_distinct_full_zeros,
    test_weak_coupling_reduces_exact_dark_residue,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(TESTS)} inverse-problem stress tests passed")
