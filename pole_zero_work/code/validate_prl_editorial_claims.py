#!/usr/bin/env python3
"""Independent checks for the PRL-level field-intercept claims.

This suite does not test editorial judgment.  It verifies the mathematical
identities, data-level trajectory statements, and citation hygiene on which
the editorial positioning relies.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data_prl"
OUT_JSON = DATA / "editorial_claim_validation.json"
OUT_LOG = DATA / "editorial_claim_validation.log"

checks: list[tuple[str, bool, str]] = []


def record(name: str, condition: bool, detail: str) -> None:
    checks.append((name, bool(condition), detail))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


rng = np.random.default_rng(20260623)

# 1. Exact Schur-complement determinant identity for complex response matrices.
max_schur_rel = 0.0
for _ in range(200):
    n_r, n_p = 3, 2
    m_rr = rng.normal(size=(n_r, n_r)) + 1j * rng.normal(size=(n_r, n_r))
    m_rr += 5.0 * np.eye(n_r)
    m_pp = rng.normal(size=(n_p, n_p)) + 1j * rng.normal(size=(n_p, n_p))
    v_rp = 0.2 * (rng.normal(size=(n_r, n_p)) + 1j * rng.normal(size=(n_r, n_p)))
    v_pr = 0.2 * (rng.normal(size=(n_p, n_r)) + 1j * rng.normal(size=(n_p, n_r)))
    full = np.block([[m_rr, v_rp], [v_pr, m_pp]])
    schur = m_pp - v_pr @ np.linalg.solve(m_rr, v_rp)
    lhs = np.linalg.det(full)
    rhs = np.linalg.det(m_rr) * np.linalg.det(schur)
    rel = abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))
    max_schur_rel = max(max_schur_rel, float(rel))
record("complex Schur determinant identity", max_schur_rel < 2e-12, f"max rel={max_schur_rel:.3e}")

# 2. Covariance inheritance by the Schur complement.
s_r = np.diag([1.0, -1.0, 1.0])
s_p = np.diag([1.0, -1.0])
s_full = np.block([[s_r, np.zeros((3, 2))], [np.zeros((2, 3)), s_p]])

def commuting_part(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + s_full @ a @ s_full)


def anticommuting_part(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a - s_full @ a @ s_full)

base = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
lin = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
quad = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
m0 = commuting_part(base) + 6.0 * np.eye(5)
m1 = anticommuting_part(lin)
m2 = commuting_part(quad)
max_cov = 0.0
for b in np.linspace(-0.15, 0.15, 21):
    def schur_at(x: float) -> np.ndarray:
        m = m0 + x * m1 + x * x * m2
        rr, rp = m[:3, :3], m[:3, 3:]
        pr, pp = m[3:, :3], m[3:, 3:]
        return pp - pr @ np.linalg.solve(rr, rp)
    dp = schur_at(float(b))
    dm = schur_at(float(-b))
    max_cov = max(max_cov, float(np.max(np.abs(s_p @ dp @ s_p - dm))))
record("Schur complement inherits node-exchange covariance", max_cov < 3e-12, f"max abs={max_cov:.3e}")

# 3. Pole zero set is invariant under analytic nonsingular left/right basis maps.
max_basis_rel = 0.0
for _ in range(100):
    d = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    u = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    v = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    u += 3.0 * np.eye(2)
    v += 3.0 * np.eye(2)
    lhs = np.linalg.det(u @ d @ v)
    rhs = np.linalg.det(u) * np.linalg.det(d) * np.linalg.det(v)
    rel = abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))
    max_basis_rel = max(max_basis_rel, float(rel))
record("basis-change determinant invariance", max_basis_rel < 2e-12, f"max rel={max_basis_rel:.3e}")

# 4. Figure-1 trajectory and finite-window statements.
traj = pd.read_csv(DATA / "fig1_field_intercept.csv")
qmax = float(traj["qmax_demo"].iloc[0])
bright_inside = bool(np.all(traj["qmin_brightening"] < qmax))
anom_inside = traj["anomaly_crossing_inside_qmax_demo"].astype(bool).to_numpy()
record("brightened crossing stays in illustrative q window", bright_inside, f"max q={traj.qmin_brightening.max():.6f}, qmax={qmax}")
record("field-created trajectory exits and re-enters finite q window", bool(np.any(anom_inside) and np.any(~anom_inside)), f"inside count={anom_inside.sum()}/{len(anom_inside)}")

# The plotted anomaly asymptotic is normalized by q_h(b_ref=0.01)=q0.
q0 = float(traj["qmin_brightening"].iloc[0] / traj["qmin_over_q0_brightening"].iloc[0])
b_window = 0.010 * q0 / qmax
low_outside = bool(np.all(traj.loc[traj.abs_b < b_window, "q_h_anomaly_normalized"] > qmax))
high_inside = bool(np.all(traj.loc[traj.abs_b > b_window, "q_h_anomaly_normalized"] < qmax))
record("analytic observation-window threshold", low_outside and high_inside, f"b_win={b_window:.6g}")

# 5. High-level numerical reinforcement outputs.
high = json.loads((DATA / "high_level_validation.json").read_text())
record("high-level suite passed", bool(high["all_high_level_checks_passed"]), "validator flag")
record("nonnegative-F0x controlled region exists", int(high["controlled_phase_map_points_F0x_nonnegative"]) > 1000, str(high["controlled_phase_map_points_F0x_nonnegative"]))
record("alternative perturbative gap check", float(high["alternative_max_gap_relative_error_b_le_0p01"]) < 1e-3, f"err={high['alternative_max_gap_relative_error_b_le_0p01']:.3e}")
record("residue-sum correction is quadratic", abs(float(high["residue_sum_error_power"]) - 2.0) < 0.08, f"power={high['residue_sum_error_power']:.6f}")

# 6. Citation hygiene: no invented journal or DOI for the public preprint.
bib = (ROOT / "manuscript_prl" / "references_prl.bib").read_text()
match = re.search(r"@misc\{Subrahmanyam2026,(.*?)\n\}", bib, re.S)
record("Subrahmanyam public record is present", match is not None, "bibliography entry")
entry = match.group(1) if match else ""
record("Subrahmanyam entry uses arXiv 2605.27031", "2605.27031" in entry, entry[:120])
record("no unverified DOI in Subrahmanyam entry", "doi" not in entry.lower(), "no doi field")
record("no unverified journal in Subrahmanyam entry", "journal" not in entry.lower(), "no journal field")

# 7. Manuscript carries the central classification and scope boundaries.
main = (ROOT / "manuscript_prl" / "main_prl_candidate.tex").read_text()
supp = (ROOT / "supplement_prl" / "supplement_prl_candidate.tex").read_text()
record("main defines field-intercept index", "\\zeta" in main and "q_h" in main, "main normal form")
record("main states finite observation-window distinction", "finite wave-vector window" in main, "main comparison paragraph")
record("supplement contains full-Maxwell Schur proof", "\\mathcal D_P=\\mathcal M_{PP}" in supp and "Schur complement" in supp, "supplement section")
record("supplement states additional-pole limitation", ("third pole" in supp or "three-mode" in supp), "scope boundary")
record("material-specific limitation is explicit", "material-specific landau parameters and linewidths" in main.lower(), "main scope")

payload = {
    "all_checks_passed": all(ok for _, ok, _ in checks),
    "number_of_checks": len(checks),
    "max_schur_relative_error": max_schur_rel,
    "max_covariance_absolute_error": max_cov,
    "max_basis_relative_error": max_basis_rel,
    "illustrative_qmax": qmax,
    "analytic_b_window": b_window,
    "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks],
}
OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
OUT_LOG.write_text("\n".join(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}" for name, ok, detail in checks) + "\n")
print(OUT_LOG.read_text(), end="")
print(json.dumps({k: v for k, v in payload.items() if k != "checks"}, indent=2))
sys.exit(0 if payload["all_checks_passed"] else 1)
