#!/usr/bin/env python3
"""Independent checks for the PRL-level reinforcement layer."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from weyl_fl_prb_solver import ModelParams, nominal_crossing  # noqa: E402
from weyl_prl_theory import (  # noqa: E402
    anomaly_crossing_small_field,
    charge_continuity_residual,
    l0_crossing_roots,
    l0_gap_coefficient,
    landau_D,
    landau_D_prime,
    solve_l0_zero_sound,
)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[ok] {label}")


def main() -> None:
    checks: dict[str, float | bool | str] = {}

    # Analytic derivative.
    derivative_errors = []
    for s in (1.08, 1.2, 1.7, 3.0, 8.0):
        h = 2.0e-6 * max(1.0, s)
        fd = (landau_D(s + h).real - landau_D(s - h).real) / (2.0 * h)
        derivative_errors.append(abs(fd - landau_D_prime(s)))
    checks["max_Dprime_absolute_error"] = max(derivative_errors)
    require(max(derivative_errors) < 2.0e-8, "analytic D'(s)")

    # Conserving l=0 crossing and linear gap.
    sstar = solve_l0_zero_sound(1.5)
    require(abs(1.0 - 1.5 * landau_D(sstar).real) < 3.0e-13, "l=0 zero-sound root")
    kappa = l0_gap_coefficient(sstar)
    gap_errors = []
    for b in (1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3):
        lo, hi = l0_crossing_roots(sstar, b)
        gap_errors.append(abs((hi - lo) / (b * kappa) - 1.0))
    checks["max_small_b_l0_gap_relative_error"] = max(gap_errors)
    require(max(gap_errors) < 5.0e-4, "Ward-completed l=0 gap coefficient")

    # Interacting charge continuity at random nonsingular points.
    rng = np.random.default_rng(83)
    residuals = []
    for _ in range(500):
        s = 1.07 + 3.0 * rng.random()
        b = 0.04 * (rng.random() - 0.5)
        fc = -0.15 + 2.3 * rng.random()
        fa = -0.15 + 2.3 * rng.random()
        try:
            residuals.append(abs(charge_continuity_residual(s, b, fc, fa)))
        except np.linalg.LinAlgError:
            pass
    checks["max_charge_continuity_residual"] = max(residuals)
    require(max(residuals) < 3.0e-14, "interacting vector-charge continuity")

    # Field-intercept classification from generated exact data.
    traj = pd.read_csv(ROOT / "data_prl" / "fig1_field_intercept.csv")
    b = traj.abs_b.to_numpy()
    dq = np.abs(traj.qmin_brightening.to_numpy() - nominal_crossing(ModelParams())[0])
    # Fit the nonzero shifts to C b^p.
    coeff = np.polyfit(np.log(b), np.log(dq), 1)
    checks["brightening_q_shift_power"] = float(coeff[0])
    checks["mean_brightening_zeta"] = float(np.mean(traj.zeta_brightening))
    checks["mean_anomaly_zeta"] = float(np.mean(traj.zeta_anomaly))
    expected_zeta = 1.0 / (1.0 + traj.competitor_x.to_numpy() ** 2)
    checks["max_exact_crossover_zeta_error"] = float(
        np.max(np.abs(traj.zeta_anomaly.to_numpy() - expected_zeta))
    )
    checks["weakest_field_anomaly_zeta"] = float(traj.zeta_anomaly.iloc[0])
    require(1.85 < coeff[0] < 2.15, "q_min-q0 is quadratic in field")
    require(
        checks["max_exact_crossover_zeta_error"] < 2.0e-12,
        "vanishing-velocity branch follows exact crossover exponent",
    )
    require(
        checks["weakest_field_anomaly_zeta"] > 0.99,
        "vanishing-velocity branch approaches zeta=1 at weak field",
    )
    require(np.max(np.abs(traj.zeta_brightening)) < 3.0e-3, "brightening has finite field intercept")

    # Scaling collapse improves systematically toward weak field.
    collapse = pd.read_csv(ROOT / "data_prl" / "fig2_universal_collapse.csv")
    freq_err = {}
    res_err = {}
    for field, group in collapse.groupby("b"):
        freq_err[float(field)] = float(np.max(np.abs(group.normalized_frequency - group.universal_frequency)))
        res_err[float(field)] = float(np.max(np.abs(group.normalized_residue - group.universal_residue)))
    fields = sorted(freq_err)
    require(all(freq_err[fields[i]] < freq_err[fields[i + 1]] for i in range(len(fields) - 1)), "frequency collapse converges as B->0")
    require(all(res_err[fields[i]] < res_err[fields[i + 1]] for i in range(len(fields) - 1)), "residue collapse converges as B->0")
    require(freq_err[fields[0]] < 3.0e-3, "weak-field pole collapse accuracy")
    require(res_err[fields[0]] < 3.5e-3, "weak-field residue collapse accuracy")
    checks["frequency_collapse_errors"] = freq_err
    checks["residue_collapse_errors"] = res_err

    # Run the solver and gauge dependencies unless the deterministic build driver
    # has already executed them once in this process graph.  This avoids tripling
    # the expensive regression runtime while preserving standalone behavior.
    skip_dependencies = os.environ.get("PRL_V9_SKIP_DEPENDENCY_SUITES") == "1"
    if skip_dependencies:
        solver_obj = json.loads((ROOT / "data" / "solver_validation.json").read_text())
        solver_ok = bool(solver_obj.get("all_checks_passed"))
        gauge_path = ROOT / "data_prl" / "v11_gauge.log"
        if not gauge_path.exists():
            gauge_path = ROOT / "data_prl" / "v9_gauge.log"
        gauge_log = gauge_path.read_text()
        gauge_ok = "all 18 gauge-completion checks passed" in gauge_log
        proc = subprocess.CompletedProcess([], 0 if solver_ok else 1, "dependency reused", "")
        proc2 = subprocess.CompletedProcess([], 0 if gauge_ok else 1, gauge_log, "")
    else:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "code" / "validate_solver.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=480,
        )
        gauge_test = ROOT / "code" / "tests" / "run_gauge_completion_checks.py"
        if not gauge_test.exists():
            source = pathlib.Path("/mnt/data/work_gauge_v4/weyl_prl_gauge_completion_v4/code/tests/run_gauge_completion_checks.py")
            gauge_test.parent.mkdir(parents=True, exist_ok=True)
            gauge_test.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        proc2 = subprocess.run(
            [sys.executable, str(gauge_test)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=480,
        )
    checks["original_validator_returncode"] = proc.returncode
    require(proc.returncode == 0, "original reduced-model regression suite")
    checks["gauge_validator_returncode"] = proc2.returncode
    if proc2.returncode != 0:
        print(proc2.stdout)
        print(proc2.stderr, file=sys.stderr)
    require(proc2.returncode == 0, "18-test gauge-completion regression suite")

    checks["all_checks_passed"] = True
    out = ROOT / "data_prl" / "prl_reinforcement_validation.json"
    out.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    log = ROOT / "data_prl" / "prl_reinforcement_validation.log"
    log.write_text(
        "PRL reinforcement validation passed.\n\n"
        + json.dumps(checks, indent=2)
        + "\n\nOriginal validator tail:\n"
        + "\n".join(proc.stdout.splitlines()[-30:])
        + "\n\nGauge validator:\n"
        + proc2.stdout,
        encoding="utf-8",
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
