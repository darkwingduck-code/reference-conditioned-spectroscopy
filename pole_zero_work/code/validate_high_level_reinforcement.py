#!/usr/bin/env python3
"""Reviewer-level checks added on top of the PRL reinforcement validator."""
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

from weyl_fl_prb_solver import (  # noqa: E402
    ModelParams,
    ProjectedModel,
    degenerate_perturbation_data,
    minimize_gap,
    nominal_crossing,
)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[ok] {label}")


def main() -> None:
    checks: dict[str, object] = {}

    # Standalone runs execute the full base suite.  The build driver may reuse a
    # just-generated passing JSON to avoid an unnecessary second solver/gauge run.
    if os.environ.get("PRL_V9_SKIP_DEPENDENCY_SUITES") == "1":
        base_obj = json.loads((ROOT / "data_prl" / "prl_reinforcement_validation.json").read_text())
        base = subprocess.CompletedProcess([], 0 if base_obj.get("all_checks_passed") else 1, "dependency reused", "")
    else:
        base = subprocess.run(
            [sys.executable, str(ROOT / "code" / "validate_prl_reinforcement.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
    if base.returncode != 0:
        print(base.stdout)
        print(base.stderr, file=sys.stderr)
    require(base.returncode == 0, "base PRL reinforcement suite")
    checks["base_suite_returncode"] = base.returncode

    # Symbolic normal-form derivation must terminate with exact residuals.
    sym = subprocess.run(
        [sys.executable, str(ROOT / "code" / "symbolic_normal_form_check.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    require(sym.returncode == 0 and "All symbolic residuals vanish exactly" in sym.stdout,
            "symbolic field-intercept and residue derivation")
    checks["symbolic_suite_returncode"] = sym.returncode

    # Alternative parameter set: l=0 intervalley harmonic is nonnegative.
    alt = ModelParams(F0c=1.0, F0a=0.8, F1c=0.0, F1a=1.0, alpha=0.02)
    require(alt.Fx >= 0.0, "alternative point has nonnegative l=0 intervalley harmonic")
    q0, s0, w0 = nominal_crossing(alt)
    require(0.05 < q0 < 0.20 and s0 > 1.0, "alternative crossing lies in undamped long-wave window")
    pert = degenerate_perturbation_data(alt)
    require(pert["gap_linear_coefficient"] > 0.1, "alternative point has nonzero linear gap")
    model = ProjectedModel(alt, lmax=2)
    rel = []
    qshifts = []
    bvals = np.array([0.0025, 0.005, 0.010, 0.020])
    for b in bvals:
        result = minimize_gap(model, float(b), qcenter=q0, relative_half_width=0.18)
        rel.append(abs(result["gap"] / (b * pert["gap_linear_coefficient"]) - 1.0))
        qshifts.append(abs(result["qmin"] - q0))
    require(max(rel[:3]) < 1.0e-3, "alternative exact gap agrees with perturbation theory")
    checks["alternative_qstar"] = q0
    checks["alternative_sstar"] = s0
    checks["alternative_omega_star"] = w0
    checks["alternative_gap_coefficient"] = pert["gap_linear_coefficient"]
    checks["alternative_max_gap_relative_error_b_le_0p01"] = max(rel[:3])

    # Exact inverse-dielectric two-pole weight approaches the sum rule with B^2.
    sw = pd.read_csv(ROOT / "data_prl" / "spectral_weight_sum_rule_summary.csv")
    power = float(np.polyfit(np.log(sw.b), np.log(sw.max_abs_sum_rule_error), 1)[0])
    require(1.8 < power < 2.2, "exact residue-sum correction is quadratic in field")
    require(float(sw.iloc[0].max_abs_sum_rule_error) < 1.1e-5,
            "weakest-field residue sum is numerically saturated")
    checks["residue_sum_error_power"] = power
    checks["residue_sum_smallest_field_error"] = float(sw.iloc[0].max_abs_sum_rule_error)

    # The phase map must contain a sizeable controlled region with F0x>=0.
    phase = pd.read_csv(ROOT / "data_prl" / "figS1_interaction_crossing_map.csv")
    n_nonneg = int(np.count_nonzero((phase.F0x >= -1e-12) & phase.inside_qbar_0p2))
    require(n_nonneg > 500, "finite crossing region persists for F0x>=0")
    checks["controlled_phase_map_points_F0x_nonnegative"] = n_nonneg

    # Physical-scale table arithmetic and semiclassical n_F relation.
    phys = pd.read_csv(ROOT / "data_prl" / "physical_scale_table.csv")
    require(np.allclose(phys.nF_estimate_at_b_0p01, 25.0), "n_F=1/(4b) scale check")
    require(np.all(np.diff(phys.B_T_for_b_0p01) > 0), "field conversion scales monotonically with mu^2")
    checks["physical_scale_rows"] = len(phys)

    checks["all_high_level_checks_passed"] = True
    out = ROOT / "data_prl" / "high_level_validation.json"
    out.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    log = ROOT / "data_prl" / "high_level_validation.log"
    log.write_text(
        "High-level reinforcement validation passed.\n\n"
        + json.dumps(checks, indent=2)
        + "\n\nBase suite tail:\n"
        + "\n".join(base.stdout.splitlines()[-35:])
        + "\n\nSymbolic check:\n"
        + sym.stdout,
        encoding="utf-8",
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
