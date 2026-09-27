"""Direct cofactor-zero audit of the reduced Weyl model used in main Fig. 2.

Fresh roots of K_DD are compared with the archived pole/residue curve.  This
undamped, real-axis s>1 benchmark is not a general complex-plane certification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from weyl_fl_prb_solver import ModelParams, ProjectedModel, nominal_crossing

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_prl"
E = np.array([1.0, 0.0, 1.0, 0.0])
# Columns span the constant probe-orthogonal complement of E.
UD = np.column_stack(
    [np.array([1.0, 0.0, -1.0, 0.0]) / np.sqrt(2.0),
     np.array([0.0, 1.0, 0.0, 0.0]),
     np.array([0.0, 0.0, 0.0, 1.0])]
)


def kernel(model, z, q, b):
    a = model.response_block(z / q, b)
    return np.linalg.solve(a, np.eye(4)) - model.interaction_full(q)


def full_zero(model, q, b, axial_reference):
    def compressed(w):
        value = np.linalg.det(UD.T @ kernel(model, w, q, b) @ UD)
        if abs(value.imag) > 1e-9:
            raise ValueError("Real-axis cofactor audit left the undamped domain")
        return float(value.real)

    radius = 0.003 * q
    left, right = axial_reference - radius, axial_reference + radius
    if left <= q:
        raise ValueError("Root bracket intersects the particle-hole continuum")
    root = brentq(compressed, left, right, xtol=5e-16, rtol=1e-14)
    k = kernel(model, root, q, b)
    direct = E @ np.linalg.solve(k, E)
    minimum_singular_value = np.linalg.svd(k, compute_uv=False)[-1]
    if abs(direct) > 2e-8 or minimum_singular_value < 1e-10:
        raise AssertionError("Unresolved zero or possible common cancellation")
    return root, float(abs(direct)), float(minimum_singular_value)


def main():
    params = ModelParams(F0c=0.0, F0a=1.5, F1c=0.4, F1a=1.0, alpha=0.02)
    model = ProjectedModel(params, lmax=2)
    q0, sa, _ = nominal_crossing(params)
    original = DATA / "fig2_pole_zero_tomography.csv"
    frame = pd.read_csv(original)
    rows = []
    for item in frame.itertuples(index=False):
        z, residual, sigma = full_zero(
            model, item.qbar, item.b, item.zero_field_axial
        )
        rows.append({
            "qbar": item.qbar, "b": item.b,
            "principal_part_zero": item.reconstructed_zero,
            "full_screened_zero": z, "zero_field_axial": item.zero_field_axial,
            "principal_minus_full": item.reconstructed_zero - z,
            "full_minus_axial_reference": z - item.zero_field_axial,
            "direct_response_zero_residual": residual,
            "kernel_min_singular_value": sigma,
        })
    scan = pd.DataFrame(rows)
    scan.to_csv(DATA / "weyl_full_zero_revision.csv", index=False)

    field_rows = []
    for b in (0.0015, 0.003, 0.006, 0.012):
        roots = model.bracketed_real_modes(q0, b, n_linear=520, n_log=220)
        if len(roots) != 2:
            raise AssertionError("Expected two resolved real poles")
        zm, zp = [m.omega_bar for m in roots]
        rm, rp = [model.dielectric_residue(z, q0, b) for z in (zm, zp)]
        pp = (rp * zm + rm * zp) / (rm + rp)
        z, residual, sigma = full_zero(model, q0, b, q0 * sa)
        field_rows.append({
            "b": b, "qbar": q0, "principal_part_zero": pp,
            "full_screened_zero": z, "zero_field_axial": q0 * sa,
            "principal_bias_over_b2": (pp - z) / b**2,
            "full_shift_over_b2": (z - q0 * sa) / b**2,
            "direct_response_zero_residual": residual,
            "kernel_min_singular_value": sigma,
        })
    fields = pd.DataFrame(field_rows)
    fields.to_csv(DATA / "weyl_full_zero_revision_field.csv", index=False)

    # Independent matrix/normalization identities away from poles.
    identity_errors = []
    cofactor_errors = []
    for q in (0.82 * q0, q0, 1.18 * q0):
        for b in (0.003, 0.006):
            for shift in (-0.004, 0.0, 0.004):
                z = q * sa + shift + 0.002j
                k = kernel(model, z, q, b)
                direct = E @ np.linalg.solve(k, E)
                cofactor = (E @ E) * np.linalg.det(UD.T @ k @ UD) / np.linalg.det(k)
                pi = model.proper_polarization_tilde(z, q, b)
                fc = model.coulomb_FC(q)
                eps = model.dielectric(z, q, b)
                identity_errors.append(max(abs(direct - pi / eps),
                                           abs(1 / eps - 1 - fc * direct)))
                cofactor_errors.append(abs(cofactor - direct))
    if max(identity_errors + cofactor_errors) > 1e-8:
        raise AssertionError("Screened-response/cofactor normalization failed")
    # Test convergence in the sampled path only; do not assert a general theorem.
    if fields.principal_bias_over_b2.max() / fields.principal_bias_over_b2.min() > 1.01:
        raise AssertionError("Sampled principal-part bias is not consistent with B^2")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), layout="constrained")
    axes[0].plot(scan.qbar, 1e6 * scan.principal_minus_full,
                 color="#168577", label=r"$\widetilde z_0-z_{\rm full}$")
    axes[0].plot(scan.qbar, 1e6 * scan.full_minus_axial_reference,
                 color="#aa6325", label=r"$z_{\rm full}-z_a(0)$")
    axes[0].axhline(0, color="#adb8c1", linewidth=0.7)
    axes[0].set(xlabel=r"$\bar q$", ylabel=r"zero difference $(10^{-6})$",
                title=r"(a) Actual Fig. 2 window, $b=0.006$")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].semilogx(fields.b, fields.principal_bias_over_b2, "o-",
                     color="#168577", markerfacecolor="none",
                     label=r"$(\widetilde z_0-z_{\rm full})/b^2$")
    axes[1].semilogx(fields.b, -fields.full_shift_over_b2, "s-",
                     color="#aa6325", markerfacecolor="none",
                     label=r"$[z_a(0)-z_{\rm full}]/b^2$")
    axes[1].set(xlabel=r"$b$", ylabel=r"scaled difference",
                title=r"(b) Weak-field path at $\bar q_0$")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    base = ROOT / "fig_prl/figS11_weyl_full_zero_revision"
    fig.savefig(base.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(base.with_suffix(".png"), dpi=240)
    plt.close(fig)
    sources = [Path(__file__).resolve(), ROOT / "code/weyl_fl_prb_solver.py", original]
    report = {
        "scope": "43 archived Fig2 samples with freshly solved full screened zeros; "
                 "4 fresh field points and 18 complex matrix identities. "
                 "Undamped reduced model outside the continuum only.",
        "all_checks_passed": True,
        "q_samples": len(scan), "field_samples": len(fields),
        "complex_identity_samples": len(identity_errors),
        "principal_bias_min": float(scan.principal_minus_full.min()),
        "principal_bias_max": float(scan.principal_minus_full.max()),
        "full_reference_shift_abs_max": float(abs(scan.full_minus_axial_reference).max()),
        "max_direct_zero_residual": float(max(scan.direct_response_zero_residual.max(),
                                               fields.direct_response_zero_residual.max())),
        "max_normalization_residual": max(identity_errors),
        "max_cofactor_residual": max(cofactor_errors),
        "source_sha256": {p.relative_to(ROOT).as_posix():
                          hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    (DATA / "weyl_full_zero_revision.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
