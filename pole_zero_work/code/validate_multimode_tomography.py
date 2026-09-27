#!/usr/bin/env python3
"""Adversarial validation of multimode pole-zero tomography and identifiability."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pole_zero_tomography import (
    bright_moments_from_poles_residues,
    classify_genealogy,
    constant_matrix_tomography,
    fit_two_pole_rational,
    multiprobe_jacobi_identity,
    poles_from_bare,
    probe_adapted_unitary,
    reconstruct_from_poles_residues,
    residues_from_bare,
    residues_of_constant_matrix,
    response_from_effective_matrix,
    scalar_cofactor_identity,
    singular_response,
)

OUT = ROOT / "data_prl" / "multimode_validation.json"
LOG = ROOT / "data_prl" / "multimode_validation.log"
checks: list[dict] = []


def check(name: str, condition: bool, detail: str) -> None:
    item = {"name": name, "passed": bool(condition), "detail": detail}
    checks.append(item)
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def relerr(a: complex, b: complex) -> float:
    return float(abs(a - b) / max(abs(a), abs(b), 1e-30))


rng = np.random.default_rng(20260627)

# 1. The finite-zero formula must reject C=R_++R_-=0.
try:
    reconstruct_from_poles_residues(0.2 - 0.01j, 0.3 - 0.02j, 1.0, -1.0)
except ValueError as exc:
    check("zero at infinity / vanishing prefactor is rejected", "summed residue" in str(exc), str(exc))
else:
    check("zero at infinity / vanishing prefactor is rejected", False, "no exception")

# 2-4. Rank-one cofactor identity for arbitrary analytic, non-Hermitian K(z).
max_scalar = 0.0
for n in range(2, 11):
    for _ in range(180):
        A0 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        A1 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        A2 = 0.1 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        z = rng.uniform(-1, 1) + 1j * rng.uniform(0.15, 1.2)
        K = A0 + z * A1 + z * z * A2 + (3.0 + 0.4j) * np.eye(n)
        v = rng.normal(size=n) + 1j * rng.normal(size=n)
        lhs, rhs = scalar_cofactor_identity(K, v)
        max_scalar = max(max_scalar, relerr(lhs, rhs))
check("rank-one cofactor identity for analytic non-Hermitian kernels", max_scalar < 8e-13, f"max rel={max_scalar:.3e}")

# General probe covariance under a random unitary basis change.
n = 7
K = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)) + 4.0 * np.eye(n)
v = rng.normal(size=n) + 1j * rng.normal(size=n)
Q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
lhs0, rhs0 = scalar_cofactor_identity(K, v)
lhs1, rhs1 = scalar_cofactor_identity(Q.conj().T @ K @ Q, Q.conj().T @ v)
err_cov = max(relerr(lhs0, lhs1), relerr(rhs0, rhs1))
check("probe-basis covariance of scalar tomography", err_cov < 3e-13, f"max rel={err_cov:.3e}")

# 5. Multi-probe Jacobi complementary-minor identity.
max_multi = 0.0
for n in range(3, 11):
    for r in range(1, min(4, n)):
        for _ in range(80):
            K = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)) + (3.5 + 0.2j) * np.eye(n)
            bright = tuple(range(r))
            lhs, rhs = multiprobe_jacobi_identity(K, bright)
            max_multi = max(max_multi, relerr(lhs, rhs))
check("multi-probe Jacobi determinant identity", max_multi < 2e-12, f"max rel={max_multi:.3e}")

# 6-8. Constant N-mode matrix: full poles, dark-block zeros, and moments.
max_zero_match = max_bright = max_coupling = 0.0
for n in range(3, 9):
    for _ in range(100):
        A = rng.normal(scale=0.12, size=(n, n))
        A = 0.5 * (A + A.T) + np.diag(np.linspace(0.1, 0.8, n))
        H = A - 1j * np.diag(rng.uniform(0.005, 0.06, n))
        v = rng.normal(size=n) + 1j * rng.normal(size=n)
        tomo = constant_matrix_tomography(H, v)
        U = probe_adapted_unitary(v)
        Hp = U.conj().T @ H @ U
        zeros = np.asarray(tomo.response_zeros)
        dark = np.linalg.eigvals(Hp[1:, 1:])
        cost = np.abs(zeros[:, None] - dark[None, :])
        rr, cc = linear_sum_assignment(cost)
        max_zero_match = max(max_zero_match, float(np.max(cost[rr, cc])))
        poles, residues = residues_of_constant_matrix(H, v)
        _, bright, coupling = bright_moments_from_poles_residues(poles, residues)
        max_bright = max(max_bright, abs(bright - Hp[0, 0]))
        max_coupling = max(max_coupling, abs(coupling - Hp[0, 1:] @ Hp[1:, 0]))
check("N-mode response zeros equal dark-compression roots", max_zero_match < 5e-12, f"max abs={max_zero_match:.3e}")
check("first spectral moment reconstructs bright diagonal", max_bright < 2e-10, f"max abs={max_bright:.3e}")
check("second connected moment reconstructs total bright-dark coupling", max_coupling < 2e-9, f"max abs={max_coupling:.3e}")

# Exact pole-zero cancellation at zero mixing and residue-level lifting.
zb = 0.19 - 0.009j
zd1 = 0.31 - 0.017j
zd2 = 0.44 - 0.026j
H0 = np.diag([zb, zd1, zd2]).astype(complex)
v0 = np.array([1.0, 0.0, 0.0], dtype=complex)
zsample = np.array([0.14 + 0.03j, 0.27 + 0.04j, 0.52 + 0.05j])
chi0 = response_from_effective_matrix(zsample, H0, v0)
check(
    "exactly decoupled dark factors cancel from scalar response",
    np.max(np.abs(chi0 - 1.0 / (zsample - zb))) < 3e-14,
    f"max abs={np.max(np.abs(chi0 - 1.0/(zsample-zb))):.3e}",
)
p0, r0 = residues_of_constant_matrix(H0, v0)
order0 = np.argsort(np.abs(p0 - zb))
bright_i = int(order0[0])
dark_i = [i for i in range(3) if i != bright_i]
check(
    "perfectly dark poles have zero auto-response residue",
    max(abs(r0[i]) for i in dark_i) < 3e-14 and abs(r0[bright_i] - 1.0) < 3e-14,
    f"residues={r0}",
)

def borrowed_weight(eps: float) -> float:
    H = H0.copy()
    H[0, 1], H[1, 0] = eps, 1.3 * eps
    H[0, 2], H[2, 0] = 0.7 * eps, 1.1 * eps
    pp, rr = residues_of_constant_matrix(H, v0)
    dark_ids = [int(np.argmin(abs(pp - zd1))), int(np.argmin(abs(pp - zd2)))]
    return float(sum(abs(rr[i]) for i in dark_ids))

w1 = borrowed_weight(2e-5)
w2 = borrowed_weight(4e-5)
check(
    "weak mixing lifts cancellation with quadratic borrowed residue",
    3.85 < w2 / w1 < 4.15,
    f"W(2eps)/W(eps)={w2/w1:.6f}",
)

# The principal-dark-block theorem is an auto-response statement; a generic
# cross response has a bordered adjugate numerator instead.
n = 5
Kx = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)) + (4.0 + 0.3j) * np.eye(n)
u = rng.normal(size=n) + 1j * rng.normal(size=n)
v = rng.normal(size=n) + 1j * rng.normal(size=n)
Ux = probe_adapted_unitary(v)
Kxp = Ux.conj().T @ Kx @ Ux
cross = complex(u.conj() @ np.linalg.solve(Kx, v))
naive_principal = complex(np.vdot(v, v) * np.linalg.det(Kxp[1:, 1:]) / np.linalg.det(Kxp))
check(
    "generic cross response is not a principal-dark cofactor",
    relerr(cross, naive_principal) > 1e-3,
    f"relative difference={relerr(cross, naive_principal):.3e}",
)

# 9. Two-mode coupling product is invariant under dark-coordinate rescaling.
zb, zd = 0.21 - 0.008j, 0.31 - 0.027j
g, h = 0.031 + 0.004j, 0.026 - 0.003j
for a in (0.2 - 0.1j, 1.7 + 0.3j, -0.8 + 1.1j):
    g_rescaled, h_rescaled = g / a, h * a
    zm, zp = poles_from_bare(zb, zd, g * h)
    rm, rp, _ = residues_from_bare(zb, zd, g_rescaled * h_rescaled)
    rec = reconstruct_from_poles_residues(zm, zp, rm, rp)
    check(f"coupling product invariant under dark rescaling a={a}", abs(rec.coupling_product - g * h) < 2e-14, f"err={abs(rec.coupling_product-g*h):.3e}")

# 10-12. Frequency normalization improves fit conditioning without moving poles/zero.
zm, zp, z0 = 1000.12 - 0.012j, 1000.31 - 0.026j, 1000.24 - 0.018j
z = np.linspace(999.95, 1000.48, 500) + 0.003j
chi = (0.03 - 0.02j) + (1.5e-4 + 2e-5j) * z + singular_response(z, zm, zp, z0, prefactor=0.8 + 0.1j)
fit_norm = fit_two_pole_rational(z, chi, background_order=1, normalize_frequency=True)
fit_raw = fit_two_pole_rational(z, chi, background_order=1, normalize_frequency=False)
check("normalized rational fit keeps exact response zero", abs(fit_norm.response_zero - z0) < 2e-9, f"err={abs(fit_norm.response_zero-z0):.3e}")
check("normalized rational fit keeps exact poles", max(abs(fit_norm.poles[i] - (zm, zp)[i]) for i in range(2)) < 2e-8, f"poles={fit_norm.poles}")
check("frequency centering reduces design conditioning", fit_norm.condition_number < fit_raw.condition_number / 1e5, f"normalized={fit_norm.condition_number:.3e}, raw={fit_raw.condition_number:.3e}")

# 13-15. Loss-only finite-window non-identifiability.
omega = np.linspace(0.15, 0.37, 800)
zm, zp, z0 = 0.215 - 0.012j, 0.305 - 0.024j, 0.267 - 0.019j
sing = singular_response(omega, zm, zp, z0, prefactor=0.75)
chi1 = sing + 5.0
chi2 = sing - 5.0
loss_diff = float(np.max(np.abs(np.imag(chi1 - chi2))))
real_diff = float(np.sqrt(np.mean(np.real(chi1 - chi2) ** 2)))
check("real analytic backgrounds leave finite-window loss unchanged", loss_diff < 1e-14, f"max Im diff={loss_diff:.3e}")
check("same loss is compatible with different phase response", real_diff > 9.9, f"RMS Re diff={real_diff:.3e}")
# A causal fit to the complex response recovers the singular zero once the model
# includes the analytic background.
fit1 = fit_two_pole_rational(omega + 0.002j, singular_response(omega + 0.002j, zm, zp, z0, prefactor=0.75) + 5.0, background_order=0)
fit2 = fit_two_pole_rational(omega + 0.002j, singular_response(omega + 0.002j, zm, zp, z0, prefactor=0.75) - 5.0, background_order=0)
check("complex causal fits remove distinct analytic backgrounds", max(abs(fit1.response_zero-z0), abs(fit2.response_zero-z0)) < 2e-12, f"errors={abs(fit1.response_zero-z0):.3e},{abs(fit2.response_zero-z0):.3e}")

# 16. Robust finite-window statement: a vanishing-velocity branch leaves every
# fixed q window even when the flat-plasmon 1/B extrapolation fails globally.
qmax = 3.0
qgrid = np.linspace(0.0, qmax, 2001)
omega_b = 1.0 + 0.12 * qgrid**2
fields = np.geomspace(1e-3, 1e-1, 12)
min_sep = []
for B in fields:
    omega_d = B * qgrid
    min_sep.append(float(np.min(np.abs(omega_b - omega_d))))
check("vanishing-velocity branch has no crossing in a fixed q window as B->0", min_sep[0] > 0.9 and min_sep[-1] > 0.7, f"min separations={min_sep[0]:.3f},{min_sep[-1]:.3f}")

# 17-19. Classifier handles several divergent exponents under its stated model.
b = np.geomspace(0.0025, 0.02, 11)
for p in (0.5, 1.0, 2.0):
    q = 0.004 + 0.0012 * b ** (-p)
    sigma = 0.003 * q
    label, _, div, dbic = classify_genealogy(b, q, sigma)
    check(f"finite-window genealogy classifier recovers p={p:g}", label == "divergent" and abs(div.parameters["p"] - p) < 0.04, f"label={label}, pfit={div.parameters.get('p')}, dBIC={dbic:.2f}")

# 20. Source artifacts for the new figure exist.
for name in (
    "figS7_multimode_poles_zeros.csv",
    "figS7_cofactor_jacobi_residuals.csv",
    "figS7_loss_only_ambiguity.csv",
    "figS7_background_dependent_total_zeros.csv",
    "multimode_tomography_summary.json",
):
    check(f"new multimode source artifact: {name}", (ROOT / "data_prl" / name).exists(), name)

payload = {
    "all_checks_passed": all(c["passed"] for c in checks),
    "number_of_checks": len(checks),
    "max_rank_one_cofactor_relative_residual": max_scalar,
    "max_multiprobe_jacobi_relative_residual": max_multi,
    "max_N_mode_dark_zero_error": max_zero_match,
    "max_bright_moment_error": max_bright,
    "max_coupling_moment_error": max_coupling,
    "normalized_fit_condition_number": fit_norm.condition_number,
    "unnormalized_fit_condition_number": fit_raw.condition_number,
    "checks": checks,
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
LOG.write_text("\n".join(f"[{'ok' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in checks) + "\n")
print(LOG.read_text(), end="")
print(json.dumps({k: v for k, v in payload.items() if k != "checks"}, indent=2))
