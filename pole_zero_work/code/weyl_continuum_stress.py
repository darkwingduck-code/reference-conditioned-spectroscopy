"""Finite-window extraction near the *actual* reduced Weyl logarithmic cut.

This is a uniform-broadening comparator, not a conserving collision model.
The F1=0 crossing permits independent node root brackets, avoiding the loss
of close pole pairs in a coarse sign scan of their product. All fit selection
uses the prescribed zero-field window center, never the finite-field zero.
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

from pole_zero_tomography import fit_two_pole_rational
from weyl_fl_prb_solver import ModelParams, ProjectedModel, sector_root

ROOT = Path(__file__).resolve().parents[1]
E = np.array([1., 0., 1., 0.])


def crossing_model(f0: float):
    params = ModelParams(F0c=0., F0a=f0, F1c=0., F1a=0., alpha=.02)
    model = ProjectedModel(params, lmax=2)
    q = np.sqrt(4 * params.alpha / (np.pi * f0))
    return model, float(q), sector_root(f0, 0.)


def node_values(model, s: complex, b: float):
    a = model.response_block(complex(s), b)
    return a[0, 0], a[2, 2]


def node_response(model, s: complex, b: float):
    a, c = node_values(model, s, b)
    f0 = model.params.F0a
    return a / (1 - f0 * a) + c / (1 - f0 * c)


def node_numerator(model, s: complex, b: float):
    a, c = node_values(model, s, b)
    return a + c - 2 * model.params.F0a * a * c


def isolated_roots(model, b: float):
    """Two node poles and intervening full zero on the real physical sheet."""
    if not 0 < abs(b) < .05:
        raise ValueError("This scan requires nonzero weak field |b| < .05")
    f0 = model.params.F0a
    poles = sorted(brentq(
        lambda s: float((1 - f0 * node_values(model, s, b)[i]).real),
        1 + 1e-12, 2., xtol=5e-16, rtol=1e-14,
    ) for i in (0, 1))
    zero = brentq(lambda s: float(node_numerator(model, s, b).real),
                  *poles, xtol=5e-16, rtol=1e-14)
    if not 1 < poles[0] < zero < poles[1]:
        raise AssertionError("Off-cut pole/zero isolation failed")
    return np.asarray(poles), float(zero)


def fit_selected(x, y, center, background_order):
    """Numerical selection only; an accepted fit is not a certificate."""
    out = {"fit_succeeded": False, "selected": False,
           "numerical_screen_passed": False, "fit_rms": np.nan,
           "selected_real": np.nan, "selected_imag": np.nan,
           "design_condition": np.nan, "failure": ""}
    try:
        fit = fit_two_pole_rational(x, y, background_order=background_order)
        out.update(fit_succeeded=True, fit_rms=fit.relative_rms,
                   design_condition=fit.condition_number)
        eligible = [c for c in fit.zero_candidates
                    if c.eligible and c.root.imag <= 1e-12]
        if not eligible:
            out["failure"] = "no_eligible_lower_half_plane_zero"
            return out
        selected = min(eligible, key=lambda c: abs(c.root - center))
        out.update(selected=True, selected_real=selected.root.real,
                   selected_imag=selected.root.imag,
                   numerical_screen_passed=bool(fit.zero_assignment_reliable))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        out["failure"] = type(exc).__name__ + ": " + str(exc)
    return out


def source_hashes():
    names = ["code/weyl_continuum_stress.py", "code/weyl_fl_prb_solver.py",
             "code/pole_zero_tomography.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names}


def main():
    rows, identities, root_rows = [], [], []
    b = .006
    for f0 in (.2, .4, .8, 1.5):
        model, q, center = crossing_model(f0)
        poles, full_zero = isolated_roots(model, b)
        edge = center - 1  # fixed zero-field design, not fitted truth
        n0 = abs(node_numerator(model, full_zero, b))
        root_rows.append({"F0a": f0, "qbar": q, "b": b,
                          "zero_field_s": center, "full_zero_s": full_zero,
                          "lower_pole_s": poles[0], "upper_pole_s": poles[1],
                          "zero_cut_distance_s": full_zero - 1,
                          "scaled_numerator_residual": n0 * f0})
        for z in (center + .13j * edge, center - 1.3 * edge + .4j * edge,
                  center + 2 * edge + 1j * edge):
            a = model.response_block(z, b)
            direct = E @ np.linalg.solve(np.eye(4) - a @ model.interaction_full(q), a @ E)
            reduced = node_response(model, z, b)
            pi = model.proper_polarization_tilde(q * z, q, b)
            eps = model.dielectric(q * z, q, b)
            identities.append(max(abs(direct - reduced), abs(direct - pi / eps))
                              / max(1., abs(direct)))
        for gamma_ratio in (.05, .2, 1.):
            gamma = gamma_ratio * edge  # s units: gamma_omega = q * gamma
            for window_ratio in (.2, .5, 1., 2., 5.):
                half = window_ratio * edge
                x = np.linspace(center - half, center + half, 301)
                y = np.asarray([node_response(model, s + 1j * gamma, b) for s in x])
                # Exact canonical reference with matched poles and full zero.
                control = ((x + 1j * gamma - full_zero)
                           / ((x + 1j * gamma - poles[0]) * (x + 1j * gamma - poles[1])))
                control *= np.linalg.norm(y) / np.linalg.norm(control)
                for kind, values in (("weyl", y), ("canonical_control", control)):
                    for order in (0, 1):
                        fitted = fit_selected(x, values, center, order)
                        selected = complex(fitted["selected_real"], fitted["selected_imag"])
                        error = (abs(selected - (full_zero - 1j * gamma)) / edge
                                 if fitted["selected"] else np.nan)
                        rows.append({"model": kind, "F0a": f0, "qbar": q,
                                     "b": b, "background_order": order,
                                     "gamma_over_edge": gamma_ratio,
                                     "window_over_edge": window_ratio,
                                     "gamma_omega": q * gamma,
                                     "window_center_s": center,
                                     "window_half_width_s": half,
                                     "window_reaches_cut_projection": bool(center - half <= 1),
                                     "true_zero_real_s": full_zero,
                                     "true_zero_imag_s": -gamma,
                                     "zero_error_over_edge": error,
                                     "screen_accepts_error_gt_10pct_edge": bool(
                                         fitted["numerical_screen_passed"] and error > .1),
                                     **fitted})
    frame = pd.DataFrame(rows)
    root_frame = pd.DataFrame(root_rows)
    DATA = ROOT / "data_prl"
    frame.to_csv(DATA / "weyl_continuum_stress.csv", index=False)
    root_frame.to_csv(DATA / "weyl_continuum_stress_roots.csv", index=False)
    if max(identities) > 1e-8 or root_frame.scaled_numerator_residual.max() > 1e-8:
        raise AssertionError("Direct Weyl identities or exact root isolation failed")
    control = frame[frame.model == "canonical_control"]
    if not control.selected.all() or control.zero_error_over_edge.max() > 1e-7:
        raise AssertionError("Exactly matched rational controls failed")
    physical = frame[(frame.model == "weyl") & (frame.background_order == 1)]
    summary = []
    for (kind, order, gamma), group in frame.groupby(
            ["model", "background_order", "gamma_over_edge"]):
        errors = group.zero_error_over_edge.dropna()
        summary.append({"model": kind, "background_order": int(order),
                        "gamma_over_edge": gamma, "attempts": len(group),
                        "selected": int(group.selected.sum()),
                        "numerical_screen_passed": int(group.numerical_screen_passed.sum()),
                        "screen_accepts_error_gt_10pct_edge": int(
                            group.screen_accepts_error_gt_10pct_edge.sum()),
                        "max_selected_error_over_edge": float(errors.max()) if len(errors) else None})
    report = {
        "scope": "Reduced Weyl F1c=F1a=F0c=0 crossing; F0a=.2,.4,.8,1.5; b=.006. "
                 "Retarded physical logarithm, deterministic noiseless real-frequency samples. "
                 "Uniform s+i gamma_s broadening only; gamma_omega=q gamma_s. "
                 "No collision Ward identity, second-sheet resonance, or contour certificate claimed.",
        "selection": "Closest eligible lower-half-plane numerator zero to prescribed "
                     "zero-field window center; no use of finite-field truth in fit or selection.",
        "noise": "None: this isolates local rational-model mismatch from statistical error.",
        "interpretation": "The exact poles, zero, and cut all translate by -i gamma_s. "
                          "Broadening changes extraction difficulty, not the exact zero genealogy. "
                          "A window crossing the real projection of the cut is not by itself a "
                          "mathematical failure criterion; the cut is below the real axis.",
        "all_checks_passed": True, "attempts": len(frame),
        "physical_attempts": int((frame.model == "weyl").sum()),
        "exact_control_attempts": len(control),
        "complex_matrix_identity_samples": len(identities),
        "max_relative_matrix_identity_error": max(identities),
        "max_scaled_root_residual": float(root_frame.scaled_numerator_residual.max()),
        "control_max_zero_error_over_edge": float(control.zero_error_over_edge.max()),
        "affine_physical_selected": int(physical.selected.sum()),
        "affine_physical_screen_passed": int(physical.numerical_screen_passed.sum()),
        "affine_physical_screen_accepts_error_gt_10pct_edge": int(
            physical.screen_accepts_error_gt_10pct_edge.sum()),
        "summary": summary, "source_sha256": source_hashes(),
    }
    (DATA / "weyl_continuum_stress.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_figure(frame, root_frame)
    print(json.dumps(report, indent=2, allow_nan=False))


def make_figure(frame, roots):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axs = plt.subplots(2, 2, figsize=(7.15, 5.3), layout="constrained")
    ax = axs[0, 0]
    ax.semilogy(roots.F0a, roots.zero_cut_distance_s, "o-", color="#2451a4")
    ax.set(xlabel=r"$F_0^a$", ylabel=r"$s_0-1$",
           title="(a) Exact zero distance to the cut")
    ax = axs[0, 1]
    select = frame[(frame.model == "weyl") & (frame.background_order == 1)
                   & (frame.F0a == .2)]
    for gamma, color in zip((.05, .2, 1.), ("#2451a4", "#168577", "#d62728")):
        part = select[select.gamma_over_edge == gamma]
        ax.loglog(part.window_over_edge, part.zero_error_over_edge, "o-", color=color,
                  label=fr"$\gamma_s/\delta={gamma:g}$")
    ax.axhline(.1, color=".55", linestyle=":", linewidth=.8)
    ax.set(xlabel=r"window half width / $\delta$", ylabel=r"zero error / $\delta$",
           title=r"(b) Weyl extraction, $F_0^a=0.2$")
    ax.legend(frameon=False, fontsize=7)
    ax = axs[1, 0]
    part = frame[(frame.model == "weyl") & (frame.background_order == 1) & frame.selected]
    for accepted, marker, color, label in ((False, "x", ".6", "screen rejects"),
                                         (True, "o", "#d62728", "screen accepts")):
        sub = part[part.numerical_screen_passed == accepted]
        ax.loglog(sub.fit_rms, sub.zero_error_over_edge, marker, color=color,
                  markersize=4, linestyle="none", label=label)
    ax.axhline(.1, color=".55", linestyle=":", linewidth=.8)
    ax.set(xlabel="relative complex fit RMS", ylabel=r"zero error / $\delta$",
           title="(c) Small residual is insufficient")
    ax.legend(frameon=False, fontsize=7)
    ax = axs[1, 1]
    counts = []
    for gamma in (.05, .2, 1.):
        group = frame[(frame.model == "weyl") & (frame.background_order == 1)
                      & (frame.gamma_over_edge == gamma)]
        counts.append((int(group.selected.sum()), len(group)))
    positions = np.arange(3)
    ax.bar(positions, [n for n, _ in counts], color="#2451a4", label="zero selected")
    ax.bar(positions, [total - n for n, total in counts],
           bottom=[n for n, _ in counts], color="#c8cdd3", label="no eligible zero")
    ax.set(xticks=positions, xticklabels=["0.05", "0.2", "1"],
           xlabel=r"$\gamma_s/\delta$", ylabel="attempts (20 per rate)",
           title="(d) All affine Weyl fits accounted for", ylim=(0, 25))
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    base = ROOT / "fig_prl/figS13_weyl_continuum_stress"
    fig.savefig(base.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(base.with_suffix(".png"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
