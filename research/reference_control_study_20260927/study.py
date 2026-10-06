"""Raw-complex joint inference with shared unknown reference parameters."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
UPSTREAM = REPO / "research/reference_validation_20260927"
sys.path.insert(0, str(UPSTREAM))
from experiment import Q0, physical_setup, physical_response

TRAIN = np.linspace(0.79, 1.15, 81)
HOLD = (TRAIN[:-1]+TRAIN[1:])/2
MODELS = {"unchanged": 7, "drift_aware": 9, "extended": 10}
NAMES = ("bright_sq", "log_gb", "remote_sq", "log_gr", "omega_d", "log_gd",
         "log_t", "drift_u", "log_gb_signal", "link_l")
LOWER = np.array([0.6, np.log(1e-4), 4, np.log(1e-4), 0.85, np.log(1e-4),
                  np.log(1e-5), -0.05, np.log(1e-4), -0.10])
UPPER = np.array([1.6, np.log(0.03), 12, np.log(0.06), 1.10, np.log(0.03),
                  np.log(0.08), 0.05, np.log(0.04), 0.10])
STARTS = np.array([[1.05, .006, 7.5, .015, .970, .006, .010, 0, .006, 0],
                   [.90, .002, 5.0, .002, .920, .002, .004, .010, .003, .030],
                   [1.20, .012, 10., .040, 1.030, .015, .040, -.010, .015, -.030]])
STARTS[:, [1, 3, 5, 6, 8]] = np.log(STARTS[:, [1, 3, 5, 6, 8]])
PROFILE_THRESHOLD = 3.841


def prediction(omega, parameters):
    """Two responses; h is known, both reference bright/remote poles are fitted."""
    p = np.asarray(parameters)
    gb, gr, gd, squared_coupling = np.exp(p[[1, 3, 5, 6]])
    drift = p[7] if len(p) >= 9 else 0.0
    signal_damping = np.exp(p[8]) if len(p) >= 9 else gb
    link = p[9] if len(p) == 10 else 0.0
    w = np.asarray(omega)
    a0 = w*w-p[0]+2j*w*gb
    a = w*w-p[0]-drift+2j*w*signal_damping
    d = w*w-p[4]**2+2j*w*gd
    remote = w*w-p[2]+2j*w*gr
    cofactor = d*remote-link*link
    determinant = a*cofactor-squared_coupling*remote-0.8**2*d
    determinant -= 2*np.sqrt(squared_coupling)*0.8*link
    return cofactor/determinant, remote/(a0*remote-0.8**2)


def residual(parameters, omega, signal, reference, sigma):
    predicted_signal, predicted_reference = prediction(omega, parameters)
    delta = np.r_[predicted_signal-signal, predicted_reference-reference]
    return np.r_[delta.real, delta.imag]/sigma


def boundary_flags(parameters):
    n = len(parameters)
    fraction = (parameters-LOWER[:n])/(UPPER[:n]-LOWER[:n])
    return [NAMES[j] for j in range(n) if fraction[j] < 1e-4 or fraction[j] > 1-1e-4]


def fit(start, omega, signal, reference, sigma, fixed=None, max_nfev=160):
    initial = np.asarray(start, float).copy()
    n = len(initial)
    free = np.ones(n, bool)
    if fixed is not None:
        index, value = fixed
        initial[index] = value
        free[index] = False

    def expand(values):
        answer = initial.copy()
        answer[free] = values
        return answer

    solved = least_squares(lambda values: residual(expand(values), omega, signal, reference, sigma),
                           initial[free], bounds=(LOWER[:n][free], UPPER[:n][free]),
                           x_scale="jac", ftol=1e-9, xtol=1e-9, gtol=1e-9,
                           max_nfev=max_nfev)
    parameters = expand(solved.x)
    return dict(parameters=parameters, chi2=float(solved.fun @ solved.fun),
                success=bool(solved.success), status=int(solved.status), nfev=int(solved.nfev),
                optimality=float(solved.optimality), boundaries=boundary_flags(parameters),
                jacobian=solved.jac, fixed=fixed)


def truth_spectra(omega, field, gamma, control):
    matrix, damping, source = physical_setup(Q0, field, gamma)
    if control != "zero_drift":
        matrix[0, 0] += 0.2*field*field
        damping[0, 0] += 0.05*field*field
    if control == "extra_link":
        matrix[1, 2] = matrix[2, 1] = 0.2*field
    reference_matrix, reference_damping, source = physical_setup(Q0, 0, gamma)
    return (physical_response(omega, matrix, damping, source),
            physical_response(omega, reference_matrix, reference_damping, source))


def case_design():
    for field, gamma, noise, control, replica in itertools.product(
            (0.05, 0.20), (0.003, 0.009), (1e-4, 1e-3),
            ("zero_drift", "allowed_drift", "extra_link"), range(2)):
        yield field, gamma, noise, control, replica


def needs_profile(field, gamma, noise, control, replica):
    return gamma == 0.003 and replica == 0 and (
        noise == 1e-3 or (field == 0.2 and control == "allowed_drift" and noise == 1e-4))


def profile_grid(parameters, index):
    offsets = ([1e-5, 1e-4, 1e-3, .005, .02, .08] if index == 4
               else [.01, .05, .20, .70, 2, 4])
    points = [parameters[index]]
    points += [parameters[index]+sign*offset for offset in offsets for sign in (-1, 1)]
    points += [LOWER[index], UPPER[index]]
    return np.unique(np.clip(points, LOWER[index], UPPER[index]))


def fit_record(solved):
    return {"chi2": solved["chi2"], "success": solved["success"],
            "status": solved["status"], "nfev": solved["nfev"],
            "optimality": solved["optimality"],
            "boundaries": ";".join(solved["boundaries"]),
            "parameters": json.dumps(solved["parameters"].tolist())}


def summarize_fit(solved, case, signal, reference, held_signal, held_reference,
                  clean_hold_signal, clean_hold_reference, sigma):
    parameters = solved["parameters"]
    values = np.linalg.svd(solved["jacobian"], compute_uv=False)
    norms = np.linalg.norm(solved["jacobian"], axis=0)
    normalized = solved["jacobian"] / np.where(norms > 0, norms, 1)
    normalized_values = np.linalg.svd(normalized, compute_uv=False)
    covariance_status = "full_rank_local_linearization"
    local_omega_se = local_gamma_se = None
    _, singular, vectors = np.linalg.svd(solved["jacobian"], full_matrices=False)
    if singular[-1] <= np.finfo(float).eps*max(solved["jacobian"].shape)*singular[0]:
        covariance_status = "rank_deficient_no_pseudoinverse_error_bar"
    else:
        variances = np.sum(vectors.T**2/singular**2, axis=1)
        local_omega_se = float(np.sqrt(variances[4]))
        local_gamma_se = float(np.exp(parameters[5])*np.sqrt(variances[5]))
    held = residual(parameters, HOLD, held_signal, held_reference, sigma)
    clean = residual(parameters, HOLD, clean_hold_signal, clean_hold_reference, sigma)
    training = residual(parameters, TRAIN, signal, reference, sigma)
    omega = float(parameters[4])
    gamma = float(np.exp(parameters[5]))
    coupling = float(np.exp(parameters[6]))
    return dict(**case, **fit_record(solved),
                training_chi2_per_dof=float(training@training/(len(training)-len(parameters))),
                holdout_chi2_per_real_sample=float(held@held/len(held)),
                holdout_clean_scaled_mse=float(clean@clean/len(clean)),
                holdout_two_sigma_reference=float(1+2*np.sqrt(2/len(held))),
                holdout_above_reference=bool(held@held/len(held) > 1+2*np.sqrt(2/len(held))),
                inferred_omega=omega, inferred_gamma=gamma, inferred_t=coupling,
                omega_error=omega-1.3*Q0, gamma_error=gamma-case["gamma_dark"],
                relative_gamma_error=(gamma-case["gamma_dark"])/case["gamma_dark"],
                relative_t_error=(coupling-(0.8*case["B"])**2)/(0.8*case["B"])**2,
                inferred_u=float(parameters[7]) if len(parameters) >= 9 else 0.0,
                inferred_v=float(np.exp(parameters[8])-np.exp(parameters[1])) if len(parameters) >= 9 else 0.0,
                inferred_link=float(parameters[9]) if len(parameters) == 10 else 0.0,
                jacobian_condition=float(values[0]/values[-1]),
                normalized_jacobian_condition=float(normalized_values[0]/normalized_values[-1]),
                singular_values=json.dumps(values.tolist()), normalized_singular_values=json.dumps(normalized_values.tolist()),
                local_error_status=covariance_status, local_omega_se=local_omega_se,
                local_gamma_se=local_gamma_se)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="run_01")
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if output.parent != ROOT or output.exists():
        raise ValueError("Use a new direct-child output directory")
    output.mkdir()
    sources = [ROOT/name for name in ("study.py", "test_study.py", "plan.md")]
    sources += [UPSTREAM/"experiment.py", REPO/"research/physical_discrimination_20260912/oscillator_pilot.py",
                REPO/"research/physical_discrimination_20260912/discrimination_core.py"]
    hashes = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot/path.name)
    _, reference_scale = truth_spectra(TRAIN, 0.05, 0.003, "zero_drift")
    scale = float(np.sqrt(np.mean(abs(reference_scale)**2)))
    started = time.monotonic()
    fits, starts, profiles, profile_starts, saved = [], [], [], [], {}
    completed_cases = 0
    for case_id, (field, gamma, noise, control, replica) in enumerate(case_design()):
        if time.monotonic()-started > 570:
            break
        rng = np.random.default_rng(2026092710+case_id*37)
        sigma = noise*scale
        clean_signal, clean_reference = truth_spectra(TRAIN, field, gamma, control)
        held_clean_signal, held_clean_reference = truth_spectra(HOLD, field, gamma, control)
        noisy = lambda clean: clean+sigma*(rng.normal(size=len(clean))+1j*rng.normal(size=len(clean)))
        signal, reference = noisy(clean_signal), noisy(clean_reference)
        held_signal, held_reference = noisy(held_clean_signal), noisy(held_clean_reference)
        saved[f"case_{case_id}_signal"] = signal
        saved[f"case_{case_id}_reference"] = reference
        saved[f"case_{case_id}_held_signal"] = held_signal
        saved[f"case_{case_id}_held_reference"] = held_reference
        case = dict(case_id=case_id, B=field, gamma_dark=gamma, noise_scale=noise,
                    control=control, replica=replica, seed=2026092710+case_id*37, sigma=sigma,
                    training_complex_samples=162, holdout_complex_samples=160)
        for model, size in MODELS.items():
            model_case = dict(**case, model=model,
                model_correct=(model == "extended" or control == "zero_drift"
                               or (model == "drift_aware" and control == "allowed_drift")))
            candidates = []
            for start_index, start in enumerate(STARTS):
                solved = fit(start[:size], TRAIN, signal, reference, sigma)
                candidates.append(solved)
                starts.append(dict(**model_case, start_index=start_index, **fit_record(solved)))
            best = min(candidates, key=lambda item: item["chi2"])
            initial_best = best["chi2"]
            model_profiles = []
            if needs_profile(field, gamma, noise, control, replica):
                for parameter_index, parameter_name in ((4, "omega"), (5, "gamma")):
                    for point in profile_grid(best["parameters"], parameter_index):
                        if time.monotonic()-started > 570:
                            break
                        point_candidates = []
                        for start_index, start in enumerate((best["parameters"], STARTS[1, :size])):
                            solved = fit(start, TRAIN, signal, reference, sigma,
                                         fixed=(parameter_index, point), max_nfev=100)
                            point_candidates.append(solved)
                            profile_starts.append(dict(**model_case, parameter=parameter_name,
                                value=float(point if parameter_index == 4 else np.exp(point)),
                                start_index=start_index, **fit_record(solved)))
                        selected = min(point_candidates, key=lambda item: item["chi2"])
                        model_profiles.append(dict(**model_case, parameter=parameter_name,
                            value=float(point if parameter_index == 4 else np.exp(point)),
                            **fit_record(selected), successful_starts=sum(item["success"] for item in point_candidates)))
                if model_profiles:
                    lowest = min(model_profiles, key=lambda row: row["chi2"])
                    if lowest["chi2"] < best["chi2"]-1e-3:
                        restart = fit(np.array(json.loads(lowest["parameters"])), TRAIN,
                                      signal, reference, sigma)
                        starts.append(dict(**model_case, start_index="profile_restart", **fit_record(restart)))
                        if restart["chi2"] < best["chi2"]:
                            best = restart
                for row in model_profiles:
                    row["delta_chi2"] = row["chi2"]-best["chi2"]
                    row["nominal_support"] = row["delta_chi2"] <= PROFILE_THRESHOLD
                    row["global_minimum_unresolved"] = row["delta_chi2"] < -1e-3
                profiles.extend(model_profiles)
            record = summarize_fit(best, model_case, signal, reference, held_signal, held_reference,
                                   held_clean_signal, held_clean_reference, sigma)
            record["initial_multistart_best_chi2"] = initial_best
            record["profile_restart_improvement"] = initial_best-best["chi2"]
            record["successful_unrestricted_starts"] = sum(item["success"] for item in candidates)
            record["profile_requested"] = needs_profile(field, gamma, noise, control, replica)
            record["profile_sampled_points"] = len(model_profiles)
            fits.append(record)
        completed_cases += 1
        print(f"cases {completed_cases}/48, fits {len(fits)}, profiles {len(profiles)}, elapsed {time.monotonic()-started:.1f}s", flush=True)
    for name, rows in (("fits", fits), ("fit_starts", starts), ("profiles", profiles), ("profile_starts", profile_starts)):
        write_csv(output/f"{name}.csv", rows)
    np.savez_compressed(output/"observations.npz", train=TRAIN, hold=HOLD, **saved)
    assert hashes == {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report = dict(status="COMPLETED" if completed_cases == 48 else "TIME_LIMIT_PARTIAL",
                  cases=completed_cases, fitted_models=len(fits), unrestricted_starts=len(starts),
                  profile_points=len(profiles), profile_start_attempts=len(profile_starts),
                  elapsed_seconds=time.monotonic()-started, reference_rms_scale=scale,
                  total_training_complex_samples_per_case=162, noise_budget_equal_between_models=True,
                  source_hashes=hashes, all_sources_unchanged=True,
                  optimizer_failures_retained=sum(not row["success"] for row in fits),
                  profile_point_failures_retained=sum(not row["success"] for row in profiles),
                  scope="finite oscillator likelihood and nominal profile support; no material or coverage certificate")
    (output/"validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
