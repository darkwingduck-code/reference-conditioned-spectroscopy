"""Matched oscillator baselines and conditional reference-model compatibility.

One process; pre-existing physics/calibration code is imported read-only.
LP feasibility concerns a rectangular necessary condition, not model truth.
"""
from __future__ import annotations

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
import csv
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LEGACY = REPO / "research/prl_physical_discrimination_20260912"
sys.path.insert(0, str(LEGACY))
from oscillator_pilot import stiffness, response, rational_fit, predict
from discrimination_core import instrument_reference, measure, infer_reference_zero

Q0 = 0.7458353261772512
OMEGA = np.linspace(0.79, 1.15, 181)
HOLD = (OMEGA[1:] + OMEGA[:-1]) / 2
FIELDS = (0.05, 0.10, 0.20)
CONTROLS = ("valid_star", "true_dark_shift", "dark_remote", "mixed_probe",
            "bright_drift", "reference_damping_mismatch")
LP_GUARD = 1e-9
LP_OPTIONS = {"primal_feasibility_tolerance": 1e-10,
              "dual_feasibility_tolerance": 1e-10,
              "ipm_optimality_tolerance": 1e-10}


def physical_setup(q, field, gamma_dark, control="valid_star"):
    kwargs = {"dark_curvature": 0.08} if control == "true_dark_shift" else {}
    if control == "dark_remote":
        kwargs["dark_remote"] = 0.2 * field
    matrix = stiffness(q, field, **kwargs)
    damping = np.diag([0.003, gamma_dark, 0.007])
    source = np.array([1.0, 0.0, 0.3 if control == "mixed_probe" else 0.0])
    if control == "bright_drift":
        matrix[0, 0] += 0.2 * field**2
    if control == "reference_damping_mismatch":
        damping[0, 0] += 0.05 * field**2
    return matrix, damping, source


def physical_response(omega, matrix, damping, source):
    z = np.asarray(omega, complex)
    inverse = z[..., None, None]**2 * np.eye(len(matrix)) - matrix
    inverse += 2j * z[..., None, None] * damping
    solved = np.linalg.solve(inverse, np.broadcast_to(source, z.shape + source.shape)[..., None])
    return np.einsum("i,...i->...", source.conj(), solved[..., 0])


def quadratic_poles(matrix, damping):
    size = len(matrix)
    state = np.block([[np.zeros_like(matrix), np.eye(size)], [-matrix, -2 * damping]])
    return 1j * np.linalg.eigvals(state)


def compressed_root(matrix, damping, source):
    basis = null_space(source.conj()[None, :])
    roots = quadratic_poles(basis.conj().T @ matrix @ basis,
                            basis.conj().T @ damping @ basis)
    return min(roots, key=lambda value: abs(value - 0.97))


def paired_fit(omega, values, eta):
    """Six-real-parameter [3/2] fit in s^2, including an affine remainder."""
    variable = (np.asarray(omega) + 1j * eta)**2
    center = float(np.mean(variable.real))
    scale = float(max(abs(variable - center)))
    powers = (variable - center) / scale
    design = np.column_stack([values * powers, values, -powers**3,
                              -powers**2, -powers, -np.ones(len(powers))])
    target = -values * powers**2
    real_design = np.vstack([design.real, design.imag])
    parameters, _, rank, _ = np.linalg.lstsq(
        real_design, np.r_[target.real, target.imag], rcond=None)
    if rank != 6:
        raise ValueError("paired fit lacks full column rank")
    denominator = np.r_[1.0, parameters[:2]]
    numerator = parameters[2:]
    _, remainder = np.polydiv(numerator, denominator)
    pp_variable = center - scale * remainder[-1] / remainder[-2]
    to_frequency = lambda value: np.sqrt(np.asarray(value, complex)) - 1j * eta
    return dict(center=center, scale=scale, num=numerator, den=denominator,
                ppzero=complex(to_frequency(pp_variable)),
                zeros=to_frequency(center + scale * np.roots(numerator)),
                poles=to_frequency(center + scale * np.roots(denominator)),
                condition=float(np.linalg.cond(real_design)))


def transformed_data(chi, epsilon, reference, reference_epsilon):
    """Direct quotient bound; does not require separately inverting reference."""
    x, y = np.asarray(chi, complex), np.asarray(reference, complex)
    ex, ey = np.asarray(epsilon, float), np.asarray(reference_epsilon, float)
    if x.shape != y.shape or ex.shape != x.shape or ey.shape != x.shape:
        raise ValueError("signal/reference arrays and bounds must have matching shapes")
    if np.any(ex < 0) or np.any(ey < 0) or not np.all(np.isfinite([x, y, ex, ey])):
        raise ValueError("finite observations and nonnegative error radii required")
    difference = x - y
    margin = abs(difference) - ex - ey
    valid = margin > 0
    transform = x[valid] * y[valid] / difference[valid]
    bound = (abs(y[valid])**2 * ex[valid] + abs(x[valid])**2 * ey[valid]
             + abs(difference[valid]) * ex[valid] * ey[valid])
    bound /= abs(difference[valid]) * margin[valid]
    bound += 1e-12 * (1 + abs(transform))
    return valid, transform, bound


def _rational_solution(matrix, target, guess):
    """Exact elimination for the at-most-four weights in a sparse LP witness."""
    rows = [[Fraction(item) for item in row] + [Fraction(value)]
            for row, value in zip(matrix, target)]
    width = len(guess)
    pivot_columns = []
    row_index = 0
    for column in range(width):
        pivot = next((j for j in range(row_index, len(rows)) if rows[j][column]), None)
        if pivot is None:
            continue
        rows[row_index], rows[pivot] = rows[pivot], rows[row_index]
        divisor = rows[row_index][column]
        rows[row_index] = [item / divisor for item in rows[row_index]]
        for j in range(len(rows)):
            if j != row_index:
                multiple = rows[j][column]
                rows[j] = [a - multiple * b for a, b in zip(rows[j], rows[row_index])]
        pivot_columns.append(column)
        row_index += 1
    if any(not any(row[:width]) and row[-1] for row in rows):
        return None
    answer = [Fraction(float(item)) for item in guess]
    for j, column in reversed(list(enumerate(pivot_columns))):
        answer[column] = rows[j][-1] - sum(rows[j][k] * answer[k]
                                             for k in range(width) if k != column)
    return answer


def exact_separation_witness(matrix, rhs, dual):
    """Farkas signs checked exactly for the stored floating LP coefficients.

    This is not a certification of all upstream floating error propagation.
    No small negative dual residual is ignored on an unbounded parameter domain.
    """
    support = np.flatnonzero(dual > 1e-12)
    if not 1 <= len(support) <= matrix.shape[1] + 1:
        return None
    selected = matrix[support]
    selected_rhs = rhs[support]
    for size in range(matrix.shape[1], -1, -1):
        for columns in itertools.combinations(range(matrix.shape[1]), size):
            equations = [np.ones(len(support))] + [selected[:, j] for j in columns]
            weights = _rational_solution(equations, [1] + [0] * size, dual[support])
            if weights is None or any(weight < 0 for weight in weights):
                continue
            products = [sum(weight * Fraction(float(row[j]))
                            for weight, row in zip(weights, selected))
                        for j in range(matrix.shape[1])]
            separation = sum(weight * Fraction(float(value))
                             for weight, value in zip(weights, selected_rhs))
            if min(products) >= 0 and separation < 0:
                return dict(weights=[str(weight) for weight in weights],
                            rows=selected.tolist(), rhs=selected_rhs.tolist(),
                            support=support.tolist(), separation=float(separation),
                            minimum_column_product=float(min(products)),
                            scope="exact_rational_signs_for_stored_float_lp")
    return None


def verify_witness(witness):
    weights = [Fraction(item) for item in witness["weights"]]
    matrix = witness["rows"]
    products = [sum(weight * Fraction(float(row[j])) for weight, row in zip(weights, matrix))
                for j in range(len(matrix[0]))]
    separation = sum(weight * Fraction(float(value))
                     for weight, value in zip(weights, witness["rhs"]))
    return min(weights) >= 0 and min(products) >= 0 and separation < 0


def reference_lp(omega, chi, epsilon, reference, reference_epsilon, fixed_gamma=None):
    valid, transform, error = transformed_data(chi, epsilon, reference, reference_epsilon)
    result = dict(status="inconclusive", reason="insufficient_safe_samples",
                  safe_samples=int(sum(valid)), unsafe_samples=int(sum(~valid)),
                  arithmetic_row_relaxation=LP_GUARD, fixed_gamma=fixed_gamma)
    if sum(valid) < 12:
        return result
    w = np.asarray(omega)[valid]
    tr, ti = transform.real, transform.imag
    zeros = np.zeros(len(w))
    matrix = np.vstack([np.column_stack([np.ones(len(w)), tr - error, zeros]),
                        np.column_stack([-np.ones(len(w)), -tr - error, zeros]),
                        np.column_stack([zeros, -ti - error, 2 * w]),
                        np.column_stack([zeros, ti - error, -2 * w])])
    rhs = np.r_[w*w, -w*w, zeros, zeros]
    if fixed_gamma is not None:
        rhs = rhs - matrix[:, 2] * fixed_gamma
        matrix = matrix[:, :2]
    row_scale = np.maximum(1.0, np.maximum(np.max(abs(matrix), axis=1), abs(rhs)))
    matrix = matrix / row_scale[:, None]
    rhs = rhs / row_scale
    relaxed = rhs + LP_GUARD
    dimension = matrix.shape[1]
    solve = lambda objective, bound=relaxed: linprog(
        objective, A_ub=matrix, b_ub=bound, bounds=(0, None),
        method="highs-ds", options=LP_OPTIONS)
    feasible = solve(np.zeros(dimension))
    result["solver_status"] = int(feasible.status)
    if feasible.status == 2:
        # A deliberately larger relaxation tests whether a rejection is numerical.
        robust_rhs = rhs + 100 * LP_GUARD
        phase_matrix = np.column_stack([matrix, -np.ones(len(rhs))])
        phase = linprog(np.r_[np.zeros(dimension), 1.0], A_ub=phase_matrix,
                        b_ub=robust_rhs, bounds=(0, None), method="highs-ds",
                        options=LP_OPTIONS)
        if phase.success:
            result["phase_one_slack"] = float(phase.fun)
            result["phase_one_primal_residual"] = float(max(phase_matrix @ phase.x - robust_rhs))
            witness = exact_separation_witness(matrix, robust_rhs, -phase.ineqlin.marginals)
            if witness is not None and verify_witness(witness):
                result.update(status="inconsistent", reason="exact_float_lp_separation",
                              witness=witness)
                return result
        result["reason"] = "infeasibility_not_independently_separated"
        return result
    if not feasible.success:
        result["reason"] = "feasibility_solver_failure"
        return result
    primal = float(max(matrix @ feasible.x - relaxed))
    result["maximum_primal_residual"] = primal
    if primal > 5e-9:
        result["reason"] = "primal_verification_failure"
        return result
    limits = []
    for j in range(dimension):
        objective = np.eye(dimension)[j]
        lower, upper = solve(objective), solve(-objective)
        if not lower.success or not upper.success:
            result["reason"] = "unbounded_or_failed_parameter_extremum"
            return result
        for endpoint in (lower, upper):
            if max(matrix @ endpoint.x - relaxed) > 5e-9:
                result["reason"] = "extremum_primal_verification_failure"
                return result
        padding = 5e-8 * (1 + max(abs(lower.fun), abs(upper.fun)))
        limits.append((max(0.0, float(lower.fun - padding)), float(-upper.fun + padding)))
    if fixed_gamma is not None:
        limits.append((fixed_gamma, fixed_gamma))
    (rlo, rhi), (tlo, thi), (glo, ghi) = limits
    if tlo <= 0:
        result["reason"] = "coupling_interval_reaches_zero"
        return result
    # Point estimate uses the same three-real-coefficient transformed regression;
    # the LP intervals are independently obtained outer parameter intervals.
    design = (np.column_stack([w*w, np.ones(len(w)), 1j*w]) if fixed_gamma is None
              else np.column_stack([w*w+2j*fixed_gamma*w, np.ones(len(w))]))
    real_design = np.vstack([design.real, design.imag])
    weights = np.r_[1 / error, 1 / error]
    weights /= max(weights)
    theta = np.linalg.lstsq(real_design * weights[:, None],
                           np.r_[transform.real, transform.imag] * weights, rcond=None)[0]
    a, b = theta[:2]
    c = theta[2] if fixed_gamma is None else 2*a*fixed_gamma
    result.update(status="compatible", reason="rectangular_outer_set_nonempty",
                  omega_lower=float(np.sqrt(rlo)), omega_upper=float(np.sqrt(rhi)),
                  gamma_lower=glo, gamma_upper=ghi, coupling_squared_lower=tlo,
                  coupling_squared_upper=thi, r_lower=rlo, r_upper=rhi)
    if a > 0 and b < 0 and c >= 0:
        gamma = c / (2*a)
        r = -b / a
        root = np.sqrt(complex(r - gamma*gamma)) - 1j*gamma
        estimated_parameters = np.array([r, 1/a, gamma])[:dimension]
        joint_residual = float(max(matrix @ estimated_parameters - relaxed))
        result.update(inferred_omega=float(np.sqrt(r)), inferred_gamma=float(gamma),
                      root_real=float(root.real), root_imag=float(root.imag),
                      estimated_parameters_in_outer_box=bool(
                          rlo <= r <= rhi and glo <= gamma <= ghi and tlo <= 1/a <= thi),
                      estimate_joint_scaled_residual=joint_residual,
                      estimated_parameters_in_outer_set=bool(joint_residual <= 5e-9))
    # Imaginary root bounds are valid in this form only if every allowed r,g
    # combination is underdamped. Otherwise retain physical-parameter bounds only.
    result["uniformly_underdamped_outer_box"] = bool(rlo > ghi*ghi)
    if result["uniformly_underdamped_outer_box"]:
        result.update(root_real_lower=float(np.sqrt(rlo - ghi*ghi)),
                      root_real_upper=float(np.sqrt(rhi - glo*glo)),
                      root_imag_lower=-ghi, root_imag_upper=-glo)
    return result


def evaluate_baselines(omega, chi, hold, hold_truth, eta, target):
    output = {}
    for name, fitter in (("positive", lambda: rational_fit(omega, chi)),
                         ("paired", lambda: paired_fit(omega, chi, eta))):
        try:
            fitted = fitter()
            root = min(fitted["zeros"], key=lambda value: abs(value - 0.97))
            held_variable = hold if name == "positive" else (hold + 1j*eta)**2
            prediction = predict(fitted, held_variable)
            output.update({f"{name}_fit_status": "ok", f"{name}_condition": fitted["condition"],
                           f"{name}_holdout_rms": float(np.linalg.norm(prediction - hold_truth)
                                                        / np.linalg.norm(hold_truth)),
                           f"{name}_pp_error": float(abs(fitted["ppzero"] - target)),
                           f"{name}_full_error": float(abs(root - target)),
                           f"{name}_root_real": float(root.real),
                           f"{name}_root_imag": float(root.imag),
                           f"{name}_root_in_window": bool(0.79 < root.real < 1.15 and root.imag <= 0)})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, IndexError) as exc:
            output[f"{name}_fit_status"] = f"failed: {exc}"
    return output


def groups():
    for noise in (0.0, 1e-4, 1e-3):
        for replica in range(1 if noise == 0 else 5):
            for q in (Q0 - 0.03, Q0, Q0 + 0.03):
                for eta in (0.003, 0.010):
                    yield "shifted", "valid_star", q, eta, noise, replica
            for gamma in (0.0015, 0.003, 0.006):
                for control in CONTROLS:
                    yield "viscous", control, Q0, gamma, noise, replica


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="run_01")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if destination.parent != ROOT or destination.exists():
        raise ValueError("Choose a new direct-child output folder; previous runs are preserved")
    destination.mkdir()
    sources = [ROOT / name for name in ("experiment.py", "test_experiment.py", "plan.md")]
    sources += [LEGACY / name for name in ("oscillator_pilot.py", "discrimination_core.py")]
    hashes = {str(path.relative_to(REPO)): sha(path) for path in sources}
    snapshot = destination / "source_snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    rows, witnesses, illustrations = [], [], {}
    start = time.time()
    largest_identity_error = 0.0
    largest_scaled_identity_error = 0.0
    smallest_stiffness = float("inf")
    largest_pole_imaginary = -float("inf")
    smallest_loss = float("inf")
    calibration_failures = 0
    for group_index, (panel, control, q, gamma, noise, replica) in enumerate(groups()):
        seed = 2026092700 + 101 * group_index
        rng = np.random.default_rng(seed)
        instrument = instrument_reference(OMEGA, rng, noise)
        assumed_gamma = gamma if panel == "shifted" else 0.003
        if panel == "shifted":
            reference_truth = response(OMEGA + 1j*gamma, q, 0)
        else:
            ref_matrix, ref_damping, source = physical_setup(q, 0, gamma, control)
            reference_truth = physical_response(OMEGA, ref_matrix, ref_damping, source)
        observed_reference, reference_bound, _ = measure(
            OMEGA, reference_truth, rng, noise, instrument)
        calibration_failures += int(np.any(abs(observed_reference-reference_truth) > reference_bound))
        for field_index, field in enumerate(FIELDS):
            if panel == "shifted":
                truth = response(OMEGA + 1j*gamma, q, field)
                held_truth = response(HOLD + 1j*gamma, q, field)
                target = 1.3*q - 1j*gamma
                omega_natural = float(np.sqrt((1.3*q)**2 + gamma*gamma))
                matrix = stiffness(q, field) + gamma*gamma*np.eye(3)
                damping = gamma*np.eye(3)
                valid_model = True
            else:
                matrix, damping, source = physical_setup(q, field, gamma, control)
                truth = physical_response(OMEGA, matrix, damping, source)
                held_truth = physical_response(HOLD, matrix, damping, source)
                target = compressed_root(matrix, damping, source)
                omega_natural = float(np.sqrt(matrix[1, 1]))
                valid_model = control in ("valid_star", "true_dark_shift")
            smallest_stiffness = min(smallest_stiffness, float(np.linalg.eigvalsh(matrix)[0]))
            largest_pole_imaginary = max(largest_pole_imaginary, float(max(quadratic_poles(matrix, damping).imag)))
            smallest_loss = min(smallest_loss, float(min(-truth.imag)))
            if valid_model:
                exact_transform = truth * reference_truth / (truth-reference_truth)
                expected = (OMEGA**2-omega_natural**2+2j*gamma*OMEGA) / (0.8*field)**2
                largest_identity_error = max(largest_identity_error, float(max(abs(exact_transform-expected))))
                largest_scaled_identity_error = max(largest_scaled_identity_error,
                    float(max(abs(exact_transform-expected)/(1+abs(expected)))))
            observed, bound, _ = measure(OMEGA, truth, np.random.default_rng(seed+field_index+1),
                                         noise, instrument)
            calibration_failures += int(np.any(abs(observed-truth) > bound))
            row = dict(case_id=len(rows), panel=panel, control=control, q=q, B=field,
                       gamma_dark=gamma, gamma_assumed=assumed_gamma, noise_radius=noise,
                       calibration_radius=noise, replica=replica, reference_seed=seed,
                       signal_seed=seed+field_index+1, model_assumptions_hold=valid_model,
                       target_root_real=float(target.real), target_root_imag=float(target.imag),
                       target_omega=omega_natural,
                       **evaluate_baselines(OMEGA, observed, HOLD, held_truth, assumed_gamma, target))
            legacy = infer_reference_zero(OMEGA, assumed_gamma, observed, bound,
                                          observed_reference, reference_bound)
            row["legacy_box_available"] = legacy["certified"]
            row["legacy_transformed_rms"] = legacy.get("transformed_relative_rms")
            if legacy.get("inferred_frequency") is not None:
                legacy_root = legacy["inferred_frequency"] - 1j*assumed_gamma
                row["legacy_reference_root_error"] = float(abs(legacy_root-target))
            for label, fixed in (("fixed", assumed_gamma), ("free", None)):
                result = reference_lp(OMEGA, observed, bound, observed_reference, reference_bound, fixed)
                witness = result.pop("witness", None)
                if witness is not None:
                    witnesses.append(dict(case_id=row["case_id"], variant=label, **witness))
                row.update({f"{label}_{key}": value for key, value in result.items()})
                if "root_real" in result:
                    estimated = result["root_real"] + 1j*result["root_imag"]
                    row[f"{label}_root_error"] = float(abs(estimated-target))
                if result["status"] == "compatible":
                    row[f"{label}_contains_omega"] = result["omega_lower"] <= omega_natural <= result["omega_upper"]
                    row[f"{label}_contains_gamma"] = result["gamma_lower"] <= gamma <= result["gamma_upper"]
                    if result["uniformly_underdamped_outer_box"]:
                        row[f"{label}_contains_root"] = bool(
                            result["root_real_lower"] <= target.real <= result["root_real_upper"]
                            and result["root_imag_lower"] <= target.imag <= result["root_imag_upper"])
            rows.append(row)
            if panel == "shifted" and q == Q0 and gamma == 0.003 and noise == 1e-4 and replica == 0:
                illustrations[f"B_{field:g}_observed"] = observed
                illustrations[f"B_{field:g}_error"] = bound
                illustrations["reference"] = observed_reference
                illustrations["reference_error"] = reference_bound
        if (group_index+1) % 40 == 0:
            print(f"retained {len(rows)} cases", flush=True)
    assert len(rows) == 792
    assert hashes == {str(path.relative_to(REPO)): sha(path) for path in sources}
    write_csv(destination / "trials.csv", rows)
    (destination / "infeasibility_witnesses.json").write_text(json.dumps(witnesses, indent=2), encoding="utf-8")
    np.savez_compressed(destination / "illustration_inputs.npz", omega=OMEGA, **illustrations)
    import pandas as pd
    data = pd.DataFrame(rows)
    summary = []
    for keys, group in data.groupby(["panel", "control", "noise_radius"]):
        item = dict(zip(("panel", "control", "noise_radius"), keys))
        item["cases"] = len(group)
        for label in ("fixed", "free"):
            for status in ("compatible", "inconsistent", "inconclusive"):
                item[f"{label}_{status}"] = int((group[f"{label}_status"] == status).sum())
        for method in ("positive_pp", "positive_full", "paired_pp", "paired_full", "free_root"):
            finite = group[f"{method}_error"].dropna()
            item[f"median_{method}_error"] = float(finite.median()) if len(finite) else None
        summary.append(item)
    write_csv(destination / "summary.csv", summary)
    compatible_valid = data[data.model_assumptions_hold & (data.free_status == "compatible")]
    checks = dict(case_count=len(rows), elapsed_seconds=time.time()-start,
                  direct_schur_identity_max_absolute_error=largest_identity_error,
                  direct_schur_identity_max_scaled_error=largest_scaled_identity_error,
                  minimum_stiffness_eigenvalue=smallest_stiffness,
                  maximum_physical_pole_imaginary=largest_pole_imaginary,
                  minimum_positive_frequency_loss=smallest_loss,
                  pointwise_calibration_bound_failures=calibration_failures,
                  valid_free_model_false_rejections=int((data.model_assumptions_hold & (data.free_status == "inconsistent")).sum()),
                  valid_free_compatible_cases=len(compatible_valid),
                  valid_free_omega_interval_misses=int((compatible_valid.free_contains_omega == False).sum()),
                  valid_free_gamma_interval_misses=int((compatible_valid.free_contains_gamma == False).sum()),
                  exact_rational_witnesses=len(witnesses),
                  all_saved_witnesses_verified=all(verify_witness(witness) for witness in witnesses),
                  source_hashes=hashes, all_source_hashes_unchanged=True,
                  status="COMPLETED", scope="conditional finite-oscillator inference; feasible rectangles do not establish model truth")
    (destination / "validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
