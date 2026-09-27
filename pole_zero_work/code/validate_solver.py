#!/usr/bin/env python3
"""Independent numerical and structural validation for the submission package."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from weyl_fl_prb_solver import (
    ModelParams,
    ProjectedModel,
    degenerate_perturbation_data,
    minimize_gap,
    nominal_crossing,
    phase_velocity_gap_coefficient,
    sector_root,
    two_mode_residue_fractions,
    damped_two_mode_poles,
)


def rel_matrix_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main() -> None:
    p = ModelParams(quadrature_order=192)
    m2 = ProjectedModel(p, lmax=2)
    qstar, sa, wstar = nominal_crossing(p)

    # 1. Closed Möbius kernel versus independent angular quadrature, including
    # the cancellation-prone 1-b*s ~ 3e-6 region.
    kernel_cases = [
        (1.03 + 0.002j, 0.02),
        (1.51 + 0.001j, -0.02),
        (4.0 + 0.01j, 0.10),
        ((1.0 - 3.0e-6) / 0.10 + 0.002j, 0.10),
    ]
    kernel_errors = [
        rel_matrix_error(m2.kernel(s, b), m2._kernel_mobius_quadrature(s, b))
        for s, b in kernel_cases
    ]
    linear = ProjectedModel(replace(p, field_scheme="linear"), lmax=2)
    h = 1e-4
    derivative_errors = []
    for s in (1.05 + 0.002j, 1.51 + 0.001j, 4.0 + 0.01j):
        linear_coeff = linear._kernel_linear(s, 1.0) - linear._kernel_linear(s, 0.0)
        mobius_deriv = (m2.kernel(s, h) - m2.kernel(s, -h)) / (2.0 * h)
        derivative_errors.append(rel_matrix_error(linear_coeff, mobius_deriv))

    # 2. b=0 full 4x4 roots versus independent sector equations.
    b0_rows: list[dict] = []
    for q in (0.055, 0.125):
        roots = m2.bracketed_real_modes(q, 0.0, n_linear=420, n_log=190)
        expected = sorted(
            [
                (q * sector_root(p.F0a, p.F1a), 0.0),
                (
                    q
                    * sector_root(
                        p.F0c + 4.0 * p.alpha / (np.pi * q * q),
                        p.F1c,
                    ),
                    1.0,
                ),
            ]
        )
        got = sorted(roots, key=lambda mode: mode.omega_bar)
        if len(got) != 2:
            raise AssertionError(f"Expected two b=0 roots at q={q}; got {len(got)}")
        for (omega_expected, character_expected), mode in zip(expected, got):
            b0_rows.append(
                {
                    "qbar": q,
                    "omega_expected": omega_expected,
                    "omega_full4x4": mode.omega_bar,
                    "absolute_frequency_error": abs(
                        mode.omega_bar - omega_expected
                    ),
                    "charge_character_expected": character_expected,
                    "charge_character_full4x4": mode.charge_character,
                    "relative_residual": mode.relative_residual,
                }
            )

    # 3. Long-wavelength plasmon asymptote.
    qsmall = 0.003
    sc = sector_root(
        p.F0c + 4.0 * p.alpha / (np.pi * qsmall * qsmall), p.F1c
    )
    omega_small = qsmall * sc
    omega_asym = np.sqrt(4.0 * p.alpha * (1.0 + p.F1c) / (3.0 * np.pi))
    plasmon_relative_error = abs(omega_small - omega_asym) / omega_asym

    # 4. Effective 2x2 theory versus the true q-minimized gap.
    perturb = degenerate_perturbation_data(p)
    gap_rows = []
    for b in (0.0025, 0.005, 0.010, 0.015, 0.020):
        result = minimize_gap(m2, b, qcenter=qstar)
        predicted = perturb["gap_linear_coefficient"] * b
        result["perturbative_gap"] = predicted
        result["relative_gap_error"] = abs(result["gap"] - predicted) / result["gap"]
        result["q_shift_over_b2"] = (result["qmin"] - qstar) / (b * b)
        gap_rows.append(result)

    # 5. Fast bracketing versus full singular-value search, field parity, and
    # relative backward error on a representative grid.
    root_rows: list[dict] = []
    parity_errors = []
    full_fast_errors = []
    for q in np.linspace(0.040, 0.145, 9):
        for b in (0.005, 0.010, 0.020):
            fast = m2.bracketed_real_modes(float(q), b, n_linear=340, n_log=160)
            full = m2.real_modes(float(q), b, n_linear=520, n_log=260)
            minus = m2.bracketed_real_modes(float(q), -b, n_linear=340, n_log=160)
            if not (len(fast) == len(full) == len(minus) == 2):
                raise AssertionError(f"Root-count mismatch at q={q}, b={b}")
            for branch in (0, 1):
                full_fast_errors.append(
                    abs(fast[branch].omega_bar - full[branch].omega_bar)
                )
                parity_errors.append(
                    abs(fast[branch].omega_bar - minus[branch].omega_bar)
                )
                root_rows.append(
                    {
                        "qbar": q,
                        "b": b,
                        "branch": branch,
                        "omega_bar": fast[branch].omega_bar,
                        "sigma_min": fast[branch].sigma_min,
                        "sigma_max": fast[branch].sigma_max,
                        "relative_residual": fast[branch].relative_residual,
                    }
                )
    pd.DataFrame(root_rows).to_csv(
        ROOT / "data" / "root_backward_errors.csv",
        index=False,
        float_format="%.16e",
    )

    # 6. Dielectric/full-determinant factorization away from poles.
    source = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=complex)
    factorization_errors = []
    for q in (0.060, qstar, 0.130):
        for b in (0.0, 0.010, 0.020):
            for omega in (0.075 + 0.001j, 0.125 + 0.001j, 0.190 + 0.001j):
                s = omega / q
                A = m2.response_block(s, b)
                Msr = np.eye(4, dtype=complex) - A @ m2.interaction_sr()
                Mfull = m2.secular(s, q, b)
                lhs = np.linalg.det(Mfull)
                rhs = np.linalg.det(Msr) * m2.dielectric(omega, q, b)
                factorization_errors.append(
                    abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
                )

    # 7. Positive-frequency sampled passivity check over a substantially wider
    # domain than the displayed loss panels. This is a numerical test, not a
    # global theorem.
    passivity_rows = []
    min_loss = np.inf
    max_loss = 0.0
    for q in (0.030, 0.060, qstar, 0.120, 0.150):
        omega = np.linspace(0.004, 0.235, 650)
        for b in (0.0, 0.005, 0.010, 0.020):
            vals = m2.loss(omega, q, b)
            local_min = float(vals.min())
            local_max = float(vals.max())
            min_loss = min(min_loss, local_min)
            max_loss = max(max_loss, local_max)
            passivity_rows.append(
                {
                    "qbar": q,
                    "b": b,
                    "omega_min": float(omega.min()),
                    "omega_max": float(omega.max()),
                    "minimum_loss": local_min,
                    "maximum_loss": local_max,
                }
            )
    pd.DataFrame(passivity_rows).to_csv(
        ROOT / "data" / "passivity_scan.csv",
        index=False,
        float_format="%.16e",
    )

    # 8. Spectroscopic residues and broadening convergence.
    qmin = gap_rows[-1]["qmin"]
    modes = m2.bracketed_real_modes(qmin, 0.020, n_linear=520, n_log=220)
    residues = [
        m2.dielectric_residue(mode.omega_bar, qmin, 0.020) for mode in modes
    ]
    eta_table = pd.read_csv(ROOT / "data" / "eta_convergence.csv")
    eta_small = eta_table[eta_table.eta_bar == eta_table.eta_bar.min()]
    eta_position_error = float(np.abs(eta_table.peak_position_error).max())
    eta_area_error = float(np.abs(eta_small.area_over_pi_residue - 1.0).max())
    eta_fwhm_error = float(
        np.abs(eta_small.fwhm / (2.0 * eta_small.eta_bar) - 1.0).max()
    )

    # 9. l=2 exact readout closure and strict linear-streaming sensitivity.
    m3 = ProjectedModel(p, lmax=3)
    closure_errors = []
    scheme_errors = []
    for q in np.linspace(0.045, 0.145, 8):
        r2 = m2.bracketed_real_modes(float(q), 0.020)
        r3 = m3.bracketed_real_modes(float(q), 0.020)
        rl = linear.bracketed_real_modes(float(q), 0.020)
        for branch in (0, 1):
            closure_errors.append(
                abs(r2[branch].omega_bar - r3[branch].omega_bar)
                / abs(r2[branch].omega_bar)
            )
            scheme_errors.append(
                abs(r2[branch].omega_bar - rl[branch].omega_bar)
                / abs(r2[branch].omega_bar)
            )

    # 10. Selected point is not isolated in parameter space.
    robustness_rows = []
    for F0_scale in (0.9, 1.0, 1.1):
        for F1_scale in (0.9, 1.0, 1.1):
            pp = replace(
                p,
                F0a=p.F0a * F0_scale,
                F1a=p.F1a * F1_scale,
                quadrature_order=96,
            )
            q, s, w = nominal_crossing(pp)
            c = degenerate_perturbation_data(pp)["gap_linear_coefficient"]
            robustness_rows.append(
                {
                    "F0a": pp.F0a,
                    "F1a": pp.F1a,
                    "qstar": q,
                    "omega_star": w,
                    "coefficient": c,
                }
            )

    # 11. Coulomb-coupling sensitivity follows the analytic square-root law.
    alpha_rows = []
    alpha_values = (0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050)
    for alpha in alpha_values:
        pp = replace(p, alpha=alpha)
        q, s, w = nominal_crossing(pp)
        alpha_rows.append(
            {
                "alpha": alpha,
                "qstar": q,
                "omega_star": w,
                "qstar_over_sqrt_alpha": q / np.sqrt(alpha),
                "omega_star_over_sqrt_alpha": w / np.sqrt(alpha),
            }
        )
    qscale = np.asarray([r["qstar_over_sqrt_alpha"] for r in alpha_rows])
    wscale = np.asarray([r["omega_star_over_sqrt_alpha"] for r in alpha_rows])
    alpha_scaling_spread = float(
        max(np.ptp(qscale) / np.mean(qscale), np.ptp(wscale) / np.mean(wscale))
    )

    # 12. High-frequency response approaches vacuum.
    high_frequency_errors = []
    for q in (0.030, qstar, 0.150):
        for b in (0.0, 0.020):
            high_frequency_errors.append(
                abs(m2.dielectric(5.0 + 1.0e-4j, q, b) - 1.0)
            )


    # 13. Coulomb and background-screening scans collapse onto a
    # Coulomb-independent phase-velocity gap coefficient at fixed short-range
    # Landau parameters within the reduced model.
    screening_df = pd.read_csv(ROOT / "data" / "screening_independence.csv")
    kappa_values = screening_df.phase_velocity_coefficient.to_numpy()
    kappa_spread = float(np.ptp(kappa_values) / np.mean(kappa_values))
    kappa_direct = phase_velocity_gap_coefficient(p)

    # 14. Full dielectric residues obey the canonical two-mode mixing law to
    # leading order and conserve the nearby bare charge-pole residue.
    residue_df = pd.read_csv(ROOT / "data" / "residue_collapse.csv")
    residue_law_error = float(
        np.abs(residue_df.normalized_prediction_error).max()
    )
    grouped_residues = residue_df.groupby(["qbar", "b"]).agg(
        residue_sum=("dielectric_residue", "sum"),
        charge_residue_b0=("charge_residue_b0", "first"),
    )
    residue_sum_error = float(
        np.abs(
            grouped_residues.residue_sum
            / grouped_residues.charge_residue_b0
            - 1.0
        ).max()
    )

    # 15. Away from resonance the activated dark-pole residue is quadratic in
    # b. Fit only the asymptotic small-b points.
    dark_df = pd.read_csv(ROOT / "data" / "dark_weight_scaling.csv")
    dark_exponents = []
    dark_fit_errors = []
    for _, group in dark_df.groupby("qbar"):
        fit_group = group[group.b <= 0.005]
        slope = float(
            np.dot(fit_group.b_squared, fit_group.dark_residue)
            / np.dot(fit_group.b_squared, fit_group.b_squared)
        )
        prediction = slope * fit_group.b_squared
        dark_fit_errors.append(
            float(
                np.max(
                    np.abs(fit_group.dark_residue - prediction)
                    / np.maximum(fit_group.dark_residue, 1e-30)
                )
            )
        )
        exponent_group = group[group.b <= 0.010]
        dark_exponents.append(
            float(
                np.polyfit(
                    np.log(exponent_group.b),
                    np.log(exponent_group.dark_residue),
                    1,
                )[0]
            )
        )

    # 16. The damped coupled-mode helper reproduces the analytic exceptional
    # threshold 2g=|gamma_c-gamma_a| at zero detuning.
    damping_threshold_errors = []
    for ratio in (0.0, 0.2, 0.5, 0.8, 0.95):
        g = 1.0
        gamma_c, gamma_a = 2.0 * g * ratio, 0.0
        poles = damped_two_mode_poles(0.0, 0.0, gamma_c, gamma_a, g)
        numerical = (poles[1].real - poles[0].real) / (2.0 * g)
        analytic = np.sqrt(1.0 - ratio * ratio)
        damping_threshold_errors.append(abs(numerical - analytic))
    coalesced = damped_two_mode_poles(0.0, 0.0, 2.0, 0.0, 1.0)
    exceptional_point_error = abs(coalesced[1].real - coalesced[0].real)

    # 17. The generated orientation table implements |cos theta| for the gap
    # and cos^2 theta for off-resonant dark weight.
    angle_df = pd.read_csv(ROOT / "data" / "angular_selection.csv")
    radians = np.deg2rad(angle_df.theta_degrees.to_numpy())
    angular_gap_error = float(
        np.max(
            np.abs(
                angle_df.gap_over_parallel_gap.to_numpy()
                - np.abs(np.cos(radians))
            )
        )
    )
    angular_weight_error = float(
        np.max(
            np.abs(
                angle_df.off_resonant_dark_residue_over_parallel_value.to_numpy()
                - np.cos(radians) ** 2
            )
        )
    )

    result = {
        "parameters": asdict(p),
        "crossing": {
            "qstar": qstar,
            "s_axial": sa,
            "omega_star": wstar,
        },
        "kernel_validation": {
            "max_closed_vs_quadrature_relative_error": max(kernel_errors),
            "errors": kernel_errors,
            "max_small_b_derivative_relative_error": max(derivative_errors),
        },
        "b0_sector_validation": b0_rows,
        "plasmon_gap_validation": {
            "qbar": qsmall,
            "numerical_omega_bar": omega_small,
            "asymptotic_omega_bar": omega_asym,
            "relative_error": plasmon_relative_error,
        },
        "degenerate_perturbation_validation": {
            "projected_coefficients": perturb,
            "rows": gap_rows,
            "max_relative_gap_error": max(
                row["relative_gap_error"] for row in gap_rows
            ),
            "q_shift_over_b2_range": [
                min(row["q_shift_over_b2"] for row in gap_rows),
                max(row["q_shift_over_b2"] for row in gap_rows),
            ],
        },
        "root_validation": {
            "max_fast_vs_full_absolute_frequency_error": max(full_fast_errors),
            "max_b_to_minus_b_absolute_frequency_error": max(parity_errors),
            "max_relative_residual": max(
                row["relative_residual"] for row in root_rows
            ),
        },
        "dielectric_factorization": {
            "max_relative_error": max(factorization_errors)
        },
        "sampled_positive_frequency_loss": {
            "minimum": min_loss,
            "maximum": max_loss,
            "nonnegative_on_sampled_grid": bool(min_loss >= -1e-10),
        },
        "spectroscopic_residue": {
            "qmin_b_0p020": qmin,
            "residues": residues,
            "all_positive": bool(min(residues) > 0.0),
        },
        "eta_convergence": {
            "max_peak_position_error": eta_position_error,
            "smallest_eta_max_area_relative_error": eta_area_error,
            "smallest_eta_max_fwhm_relative_error": eta_fwhm_error,
        },
        "closure_and_scheme": {
            "max_l2_l3_relative_frequency_error": max(closure_errors),
            "max_mobius_vs_linear_relative_frequency_error": max(scheme_errors),
        },
        "local_parameter_robustness": robustness_rows,
        "coulomb_coupling_sensitivity": {
            "rows": alpha_rows,
            "max_relative_spread_of_sqrt_scaling": alpha_scaling_spread,
        },
        "brightening_scaling": {
            "phase_velocity_coefficient": kappa_direct,
            "screening_scan_relative_spread": kappa_spread,
            "max_residue_law_absolute_error": residue_law_error,
            "max_residue_sum_relative_error": residue_sum_error,
            "dark_weight_log_log_exponents": dark_exponents,
            "max_small_b_quadratic_fit_relative_error": max(dark_fit_errors),
            "max_damped_pole_formula_error": max(damping_threshold_errors),
            "exceptional_point_real_splitting_error": exceptional_point_error,
            "angular_gap_rule_error": angular_gap_error,
            "angular_weight_rule_error": angular_weight_error,
        },
        "high_frequency_epsilon_to_one": {
            "max_absolute_error_at_omega_bar_5": max(high_frequency_errors)
        },
        "acceptance_criteria": {
            "kernel_error_lt": 2e-8,
            "kernel_derivative_error_lt": 2e-8,
            "b0_frequency_error_lt": 3e-10,
            "plasmon_relative_error_lt": 1e-3,
            "gap_relative_error_lt": 3e-4,
            "root_relative_residual_lt": p.relative_residual_tol,
            "fast_full_frequency_error_lt": 2e-10,
            "field_parity_frequency_error_lt": 2e-10,
            "factorization_error_lt": 2e-10,
            "sampled_loss_min_ge": -1e-10,
            "residue_positive": True,
            "eta_position_error_lt": 5e-6,
            "eta_area_error_lt": 1e-2,
            "eta_fwhm_error_lt": 2e-2,
            "l2_l3_relative_error_lt": 2e-10,
            "mobius_linear_relative_error_lt": 2e-4,
            "local_robustness_qstar_lt": 0.13,
            "local_robustness_omega_star_lt": 0.20,
            "local_robustness_coefficient_gt": 0.15,
            "alpha_sqrt_scaling_spread_lt": 1e-12,
            "alpha_0p01_to_0p04_inside_low_energy_window": True,
            "phase_velocity_coefficient_spread_lt": 2e-9,
            "residue_law_error_lt": 3e-2,
            "residue_sum_error_lt": 1e-3,
            "dark_weight_exponent_between": [1.9, 2.1],
            "dark_weight_small_b_fit_error_lt": 2e-2,
            "damped_pole_formula_error_lt": 1e-12,
            "angular_selection_error_lt": 1e-14,
            "high_frequency_epsilon_error_lt": 1e-3,
        },
    }

    checks = [
        result["kernel_validation"]["max_closed_vs_quadrature_relative_error"]
        < 2e-8,
        result["kernel_validation"]["max_small_b_derivative_relative_error"]
        < 2e-8,
        max(row["absolute_frequency_error"] for row in b0_rows) < 3e-10,
        plasmon_relative_error < 1e-3,
        result["degenerate_perturbation_validation"]["max_relative_gap_error"]
        < 3e-4,
        result["root_validation"]["max_relative_residual"]
        < p.relative_residual_tol,
        result["root_validation"]["max_fast_vs_full_absolute_frequency_error"]
        < 2e-10,
        result["root_validation"]["max_b_to_minus_b_absolute_frequency_error"]
        < 2e-10,
        result["dielectric_factorization"]["max_relative_error"] < 2e-10,
        min_loss >= -1e-10,
        min(residues) > 0.0,
        eta_position_error < 5e-6,
        eta_area_error < 1e-2,
        eta_fwhm_error < 2e-2,
        max(closure_errors) < 2e-10,
        max(scheme_errors) < 2e-4,
        max(row["qstar"] for row in robustness_rows) < 0.13,
        max(row["omega_star"] for row in robustness_rows) < 0.20,
        min(row["coefficient"] for row in robustness_rows) > 0.15,
        alpha_scaling_spread < 1e-12,
        all(
            row["qstar"] <= 0.15 and row["omega_star"] <= 0.22
            for row in alpha_rows
            if row["alpha"] <= 0.040
        ),
        kappa_spread < 2e-9,
        residue_law_error < 3e-2,
        residue_sum_error < 1e-3,
        all(1.9 < exponent < 2.1 for exponent in dark_exponents),
        max(dark_fit_errors) < 2e-2,
        max(damping_threshold_errors) < 1e-12,
        exceptional_point_error < 1e-12,
        angular_gap_error < 1e-14,
        angular_weight_error < 1e-14,
        max(high_frequency_errors) < 1e-3,
    ]
    result["all_checks_passed"] = bool(all(checks))

    out = ROOT / "data" / "solver_validation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
