#!/usr/bin/env python3
"""Generate the PRL-reinforcement figures and source data."""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from weyl_fl_prb_solver import (  # noqa: E402
    ModelParams,
    ProjectedModel,
    degenerate_perturbation_data,
    minimize_gap,
    nominal_crossing,
    sector_root,
    two_mode_residue_fractions,
)
from weyl_prl_theory import (  # noqa: E402
    anomaly_crossing_exact_crossover,
    anomaly_crossover_exponent,
    charge_continuity_residual,
    field_intercept_exponent,
    l0_crossing_roots,
    l0_gap_coefficient,
    l0_secular_determinant,
    landau_D,
    solve_l0_zero_sound,
    universal_branch_curve,
    universal_residue_fractions,
)

FIG = ROOT / "fig_prl"
DATA = ROOT / "data_prl"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "savefig.transparent": False,
    }
)


def savefig(fig: plt.Figure, stem: str, title: str) -> None:
    metadata = {
        "Title": title,
        "Author": "Authors to be inserted",
        "Subject": "PRL candidate reinforcement figure",
        "Keywords": "Weyl metal, dark mode, field intercept, universal scaling",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def zero_field_frequencies(params: ModelParams, q: float, s_a: float) -> tuple[float, float]:
    f0eff = params.F0c + 4.0 * params.alpha / (
        np.pi * (q * q + params.qTFbar * params.qTFbar)
    )
    omega_c = q * sector_root(f0eff, params.F1c)
    omega_a = q * s_a
    return float(omega_c), float(omega_a)


def q_for_zero_field_detuning(
    params: ModelParams, qstar: float, s_a: float, target: float
) -> float:
    def f(q: float) -> float:
        wc, wa = zero_field_frequencies(params, q, s_a)
        return wc - wa - target

    lo = 0.45 * qstar
    hi = 1.75 * qstar
    return float(brentq(f, lo, hi, xtol=2e-14, rtol=2e-14))


def figure_1_field_intercept(params: ModelParams) -> dict:
    model = ProjectedModel(params, lmax=2)
    qstar, s_a, omega_star = nominal_crossing(params)

    # Panel (a): exact reduced-model modes at B=0 and finite B.
    qgrid = np.linspace(0.066, 0.123, 57)
    b_show = 0.010
    rows = []
    wc0, wa0 = [], []
    low, high = [], []
    for q in qgrid:
        c0, a0 = zero_field_frequencies(params, float(q), s_a)
        wc0.append(c0)
        wa0.append(a0)
        modes = model.bracketed_real_modes(float(q), b_show, n_linear=330, n_log=160)
        if len(modes) != 2:
            raise RuntimeError(f"Expected two exact poles at q={q}, got {len(modes)}")
        low.append(modes[0].omega_bar)
        high.append(modes[1].omega_bar)
        rows.extend(
            [
                {"qbar": q, "b": 0.0, "branch": "charge", "omega_bar": c0},
                {"qbar": q, "b": 0.0, "branch": "axial", "omega_bar": a0},
                {"qbar": q, "b": b_show, "branch": "lower", "omega_bar": modes[0].omega_bar},
                {"qbar": q, "b": b_show, "branch": "upper", "omega_bar": modes[1].omega_bar},
            ]
        )
    pd.DataFrame(rows).to_csv(DATA / "fig1_dispersion.csv", index=False)

    # Panel (b): exact finite-intercept trajectory versus the exact acoustic crossover.
    fields = np.geomspace(0.0018, 0.025, 11)
    qmins = []
    gaps = []
    for b in fields:
        result = minimize_gap(model, float(b), qcenter=qstar, relative_half_width=0.16)
        qmins.append(result["qmin"])
        gaps.append(result["gap"])
    qmins = np.asarray(qmins)
    gaps = np.asarray(gaps)
    competitor_field_ref = 0.010
    competitor_x_ref = 0.20
    q_anom = anomaly_crossing_exact_crossover(
        fields,
        field_ref=competitor_field_ref,
        q_ref=qstar,
        x_ref=competitor_x_ref,
    )
    q_anom_weak = qstar * competitor_field_ref / fields
    x_anom = competitor_x_ref * fields / competitor_field_ref
    qmax_demo = 0.20
    zeta_bright = field_intercept_exponent(fields, qmins)
    zeta_anom = anomaly_crossover_exponent(fields, competitor_field_ref, competitor_x_ref)
    trajectory = pd.DataFrame(
        {
            "abs_b": fields,
            "qmin_brightening": qmins,
            "qmin_over_q0_brightening": qmins / qstar,
            "gap": gaps,
            "q_h_anomaly_normalized": q_anom,
            "q_h_over_q0_anomaly_normalized": q_anom / qstar,
            "q_h_weak_asymptote_normalized": q_anom_weak,
            "q_h_weak_asymptote_over_q0": q_anom_weak / qstar,
            "competitor_x": x_anom,
            "competitor_x_ref": competitor_x_ref,
            "competitor_field_ref": competitor_field_ref,
            "zeta_brightening": zeta_bright,
            "zeta_anomaly": zeta_anom,
            "qmax_demo": qmax_demo,
            "qmax_over_q0_demo": qmax_demo / qstar,
            "anomaly_crossing_inside_qmax_demo": q_anom <= qmax_demo,
        }
    )
    trajectory.to_csv(DATA / "fig1_field_intercept.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    ax = axes[0]
    ax.plot(qgrid, wc0, ls="--", lw=1.0, label=r"charge, $b=0$")
    ax.plot(qgrid, wa0, ls=":", lw=1.2, label=r"dark axial, $b=0$")
    ax.plot(qgrid, low, lw=1.2, label=rf"hybrid poles, $b={b_show:.3f}$")
    ax.plot(qgrid, high, lw=1.2)
    ax.axvline(qstar, lw=0.8, ls="-.")
    ax.fill_between(qgrid, 0.0, qgrid, alpha=0.08)
    ax.plot(qgrid, qgrid, lw=0.7, ls="--")
    ax.set_xlim(qgrid.min(), qgrid.max())
    ax.set_ylim(0.092, 0.183)
    ax.set_xlabel(r"$\bar q$")
    ax.set_ylabel(r"$\bar\omega$")
    ax.set_title("pre-existing dark pole")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.04, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    ax.semilogx(
        fields,
        qmins / qstar,
        marker="o",
        ms=3.5,
        lw=1.1,
        label=r"brightening: $q_{\min}/q_0$",
    )
    ax.semilogx(
        fields,
        q_anom / qstar,
        marker="s",
        ms=3.2,
        lw=1.0,
        label=r"exact vanishing-velocity crossover",
    )
    ax.axhline(1.0, lw=0.7, ls=":")
    qmax_ratio = qmax_demo / qstar
    ax.axhline(qmax_ratio, lw=0.7, ls="--")
    ax.axhspan(qmax_ratio, 7.2, alpha=0.055)
    ax.text(
        0.48,
        0.45,
        r"outside $\bar q<0.2$ window",
        transform=ax.transAxes,
        fontsize=6.7,
    )
    ax.set_xlabel(r"$|b_\parallel|$")
    ax.set_ylabel("normalized crossing wave vector")
    ax.set_title("field-intercept diagnostic")
    ax.set_ylim(0.25, 7.2)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.04, "(b)", transform=ax.transAxes, fontweight="bold")

    fig.subplots_adjust(wspace=0.32)
    savefig(
        fig,
        "fig1_field_intercept",
        "Finite-field intercept distinguishes a hidden pole from a vanishing-velocity branch",
    )
    return {
        "qstar": qstar,
        "s_a": s_a,
        "omega_star": omega_star,
        "max_abs_zeta_brightening": float(np.max(np.abs(zeta_bright))),
        "mean_zeta_anomaly": float(np.mean(zeta_anom)),
        "qmax_demo": qmax_demo,
        "qmax_over_q0_demo": float(qmax_demo / qstar),
    }


def figure_2_universal_collapse(params: ModelParams) -> dict:
    model = ProjectedModel(params, lmax=2)
    qstar, s_a, _ = nominal_crossing(params)
    perturb = degenerate_perturbation_data(params)
    gap_coeff = perturb["gap_linear_coefficient"]  # 2g/|b|
    bvals = (0.0025, 0.005, 0.010, 0.020)
    Xtargets = np.linspace(-3.2, 3.2, 21)
    rows = []

    for b in bvals:
        for Xtarget in Xtargets:
            target_delta = Xtarget * gap_coeff * b
            q = q_for_zero_field_detuning(params, qstar, s_a, target_delta)
            wc, wa = zero_field_frequencies(params, q, s_a)
            center0 = 0.5 * (wc + wa)
            g = 0.5 * gap_coeff * b
            modes = model.bracketed_real_modes(q, b, n_linear=430, n_log=200)
            if len(modes) != 2:
                raise RuntimeError(f"Collapse root failure at b={b}, X={Xtarget}")
            zero_charge_residue = model.dielectric_residue(wc, q, 0.0)
            lower_law, upper_law = universal_residue_fractions(Xtarget)
            ylaw_lower, ylaw_upper = universal_branch_curve(Xtarget)
            for branch, mode, ylaw, rlaw in (
                ("lower", modes[0], float(ylaw_lower), float(lower_law)),
                ("upper", modes[1], float(ylaw_upper), float(upper_law)),
            ):
                residue = model.dielectric_residue(mode.omega_bar, q, b)
                rows.append(
                    {
                        "b": b,
                        "X_target": Xtarget,
                        "X_exact_zero_field_detuning": (wc - wa) / (gap_coeff * b),
                        "qbar": q,
                        "branch": branch,
                        "normalized_frequency": (mode.omega_bar - center0) / g,
                        "universal_frequency": ylaw,
                        "normalized_residue": residue / zero_charge_residue,
                        "universal_residue": rlaw,
                        "relative_residual": mode.relative_residual,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "fig2_universal_collapse.csv", index=False)
    xcurve = np.linspace(-3.35, 3.35, 600)
    ylo, yup = universal_branch_curve(xcurve)
    rlo, rup = universal_residue_fractions(xcurve)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.72))
    markers = {0.0025: "o", 0.005: "s", 0.010: "^", 0.020: "D"}
    ax = axes[0]
    ax.plot(xcurve, ylo, lw=1.35, label="universal two-mode law")
    ax.plot(xcurve, yup, lw=1.35)
    for b in bvals:
        sub = df[df.b == b]
        for branch in ("lower", "upper"):
            sb = sub[sub.branch == branch]
            ax.scatter(
                sb.X_target,
                sb.normalized_frequency,
                s=14,
                marker=markers[b],
                label=rf"exact $b={b:.4f}$" if branch == "lower" else None,
            )
    ax.set_xlabel(r"$X=\delta\omega_0/(2g)$")
    ax.set_ylabel(r"$(\omega_\pm-\bar\omega_0)/g$")
    ax.set_title("pole-geometry collapse")
    ax.text(0.03, 0.04, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    ax.plot(xcurve, rlo, lw=1.35)
    ax.plot(xcurve, rup, lw=1.35)
    for b in bvals:
        sub = df[df.b == b]
        for branch in ("lower", "upper"):
            sb = sub[sub.branch == branch]
            ax.scatter(
                sb.X_target,
                sb.normalized_residue,
                s=14,
                marker=markers[b],
                label=rf"exact $b={b:.4f}$" if branch == "lower" else None,
            )
    ax.set_xlabel(r"$X=\delta\omega_0/(2g)$")
    ax.set_ylabel(r"$\mathcal{R}_\pm/\mathcal{R}_c^{(0)}$")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("spectral-weight collapse")
    ax.text(0.03, 0.04, "(b)", transform=ax.transAxes, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=6.6,
        handlelength=1.7,
        columnspacing=1.0,
    )
    fig.subplots_adjust(wspace=0.30, bottom=0.22)
    savefig(
        fig,
        "fig2_universal_collapse",
        "Universal pole and residue scaling near a finite-field brightening resonance",
    )

    return {
        "max_frequency_abs_error_b_0p0025": float(
            np.max(
                np.abs(
                    df[df.b == 0.0025].normalized_frequency
                    - df[df.b == 0.0025].universal_frequency
                )
            )
        ),
        "max_residue_abs_error_b_0p0025": float(
            np.max(
                np.abs(
                    df[df.b == 0.0025].normalized_residue
                    - df[df.b == 0.0025].universal_residue
                )
            )
        ),
        "max_frequency_abs_error_b_0p02": float(
            np.max(
                np.abs(
                    df[df.b == 0.02].normalized_frequency
                    - df[df.b == 0.02].universal_frequency
                )
            )
        ),
        "max_residue_abs_error_b_0p02": float(
            np.max(
                np.abs(
                    df[df.b == 0.02].normalized_residue
                    - df[df.b == 0.02].universal_residue
                )
            )
        ),
    }


def find_l0_roots(q: float, b: float, F0c: float, F_a: float, alpha: float) -> tuple[float, float]:
    F_c = F0c + 4.0 * alpha / (np.pi * q * q)
    grid_near = 1.0 + np.geomspace(1.0e-8, 0.25, 1600)
    grid_far = np.linspace(1.25, 8.0, 2400)
    grid = np.unique(np.concatenate([grid_near, grid_far]))
    vals = np.asarray([l0_secular_determinant(s, b, F_c, F_a).real for s in grid])
    roots = []
    for i in range(len(grid) - 1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        if vals[i] == 0.0 or vals[i] * vals[i + 1] < 0.0:
            root = brentq(
                lambda x: float(l0_secular_determinant(x, b, F_c, F_a).real),
                grid[i],
                grid[i + 1],
                xtol=2e-13,
                rtol=2e-13,
            )
            if not roots or abs(root - roots[-1]) > 2e-7:
                roots.append(float(root))
    # The strictly O(b) kernel can generate a spurious root exponentially close
    # to the particle-hole branch point, where the weak-field expansion is not
    # uniform.  Retain the two roots adiabatically connected to the b=0 charge
    # and axial zero-sound poles.
    targets = [solve_l0_zero_sound(F_c), solve_l0_zero_sound(F_a)]
    selected: list[float] = []
    remaining = list(roots)
    for target in targets:
        if not remaining:
            break
        idx = int(np.argmin([abs(r - target) for r in remaining]))
        selected.append(remaining.pop(idx))
    selected = sorted(set(selected))
    if len(selected) != 2:
        raise RuntimeError(
            f"Expected two adiabatic conserving l=0 roots at q={q}, b={b}; "
            f"all={roots}, selected={selected}"
        )
    return selected[0], selected[1]


def figure_3_conserving_completion() -> dict:
    F_a = 1.5
    F0c = 0.5
    alpha = 0.010
    s_star = solve_l0_zero_sound(F_a)
    qstar = float(np.sqrt(4.0 * alpha / (np.pi * (F_a - F0c))))
    omega_star = qstar * s_star
    analytic_kappa = l0_gap_coefficient(s_star)

    qgrid = np.linspace(0.075, 0.154, 54)
    b_show = 0.010
    rows = []
    for q in qgrid:
        for b in (0.0, b_show):
            roots = find_l0_roots(float(q), b, F0c, F_a, alpha)
            for branch, s in zip(("lower", "upper"), roots):
                rows.append(
                    {"qbar": q, "b": b, "branch": branch, "s": s, "omega_bar": q * s}
                )
    disp = pd.DataFrame(rows)
    disp.to_csv(DATA / "fig3_conserving_dispersion.csv", index=False)

    bvals = np.geomspace(2.0e-4, 1.0e-2, 18)
    gap_rows = []
    for b in bvals:
        lo, hi = l0_crossing_roots(s_star, float(b))
        gap_rows.append(
            {
                "b": b,
                "s_lower": lo,
                "s_upper": hi,
                "gap_s": hi - lo,
                "gap_s_over_b": (hi - lo) / b,
                "analytic_gap_s_over_b": analytic_kappa,
            }
        )
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(DATA / "fig3_conserving_gap.csv", index=False)

    rng = np.random.default_rng(20260623)
    residuals = []
    for _ in range(400):
        s = 1.08 + 2.8 * rng.random()
        b = 0.04 * (rng.random() - 0.5)
        fc = -0.1 + 2.2 * rng.random()
        fa = -0.1 + 2.2 * rng.random()
        # Skip accidental poles, which magnify all floating-point errors.
        try:
            res = charge_continuity_residual(s, b, fc, fa)
        except np.linalg.LinAlgError:
            continue
        residuals.append(abs(res))
    residuals = np.asarray(residuals)
    pd.DataFrame({"charge_continuity_residual": residuals}).to_csv(
        DATA / "fig3_ward_residuals.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.72))
    ax = axes[0]
    for b, ls, label in ((0.0, "--", r"$b=0$"), (b_show, "-", rf"$b={b_show:.3f}$")):
        sub = disp[disp.b == b]
        for branch in ("lower", "upper"):
            sb = sub[sub.branch == branch]
            ax.plot(sb.qbar, sb.omega_bar, ls=ls, lw=1.15, label=label if branch == "lower" else None)
    ax.axvline(qstar, ls=":", lw=0.8)
    ax.fill_between(qgrid, 0.0, qgrid, alpha=0.08)
    ax.plot(qgrid, qgrid, ls="--", lw=0.7)
    ax.set_xlabel(r"$\bar q$")
    ax.set_ylabel(r"$\bar\omega$")
    ax.set_title("conserving $l=0$ realization")
    ax.legend(frameon=False)
    ax.text(0.03, 0.04, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    ax.semilogx(gaps.b, gaps.gap_s_over_b, marker="o", ms=3.4, lw=1.0, label="exact roots")
    ax.axhline(analytic_kappa, ls="--", lw=1.0, label=r"$2|\delta/D'|$")
    ax.set_xlabel(r"$|b_\parallel|$")
    ax.set_ylabel(r"$\Delta s/|b_\parallel|$")
    ax.set_title("nonzero gap after Ward completion")
    ax.legend(frameon=False)
    ax.text(0.03, 0.04, "(b)", transform=ax.transAxes, fontweight="bold")

    fig.subplots_adjust(wspace=0.31)
    savefig(
        fig,
        "fig3_conserving_completion",
        "Gauge-conserving longitudinal existence proof for magnetic brightening",
    )

    return {
        "F_a": F_a,
        "F0c": F0c,
        "alpha": alpha,
        "s_star": s_star,
        "qstar": qstar,
        "omega_star": omega_star,
        "analytic_gap_coefficient": analytic_kappa,
        "max_relative_gap_coefficient_error": float(
            np.max(np.abs(gaps.gap_s_over_b / analytic_kappa - 1.0))
        ),
        "max_charge_continuity_residual": float(np.max(residuals)),
        "median_charge_continuity_residual": float(np.median(residuals)),
    }


def main() -> None:
    params = ModelParams()
    result = {
        "field_intercept": figure_1_field_intercept(params),
        "universal_collapse": figure_2_universal_collapse(params),
        "conserving_completion": figure_3_conserving_completion(),
    }
    (DATA / "prl_reinforcement_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
