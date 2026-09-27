#!/usr/bin/env python3
"""Generate v6 pole-zero tomography, inference, and adversarial-test figures."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pole_zero_tomography import (
    classify_genealogy_exact,
    fit_exact_vanishing_velocity,
    fit_finite_intercept,
    fit_two_pole_rational,
    reconstruct_from_poles_residues,
    residues_from_bare,
    singular_response,
)
from weyl_fl_prb_solver import (
    ModelParams,
    ProjectedModel,
    minimize_gap,
    nominal_crossing,
    sector_root,
)

FIG = ROOT / "fig_prl"
DATA = ROOT / "data_prl"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.4,
    "axes.labelsize": 8.4,
    "axes.titlesize": 8.6,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
})


def save(fig: plt.Figure, stem: str, title: str) -> None:
    metadata = {
        "Title": title,
        "Author": "Authors to be inserted",
        "Subject": "Pole-zero tomography reinforcement",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def zero_field_branches(params: ModelParams, q: float) -> tuple[float, float]:
    sa = sector_root(params.F0a, params.F1a)
    fc = params.F0c + 4.0 * params.alpha / (
        np.pi * (q * q + params.qTFbar * params.qTFbar)
    )
    sc = sector_root(fc, params.F1c)
    return q * sc, q * sa


def main_tomography_figure() -> dict:
    params = ModelParams(F0c=0.0, F0a=1.5, F1c=0.4, F1a=1.0, alpha=0.02)
    model = ProjectedModel(params, lmax=2)
    q0, sa, w0 = nominal_crossing(params)
    b_show = 0.006
    qgrid = np.linspace(0.78 * q0, 1.22 * q0, 43)
    rows = []
    for q in qgrid:
        modes = model.bracketed_real_modes(float(q), b_show, n_linear=520, n_log=220)
        if len(modes) != 2:
            continue
        zm, zp = modes[0].omega_bar, modes[1].omega_bar
        # epsilon^-1 residues determine a principal-part zero, not the full cofactor zero.
        rm = model.dielectric_residue(zm, float(q), b_show)
        rp = model.dielectric_residue(zp, float(q), b_show)
        rec = reconstruct_from_poles_residues(zm, zp, rm, rp)
        wc0, wa0 = zero_field_branches(params, float(q))
        rows.append({
            "qbar": q,
            "b": b_show,
            "omega_minus": zm,
            "omega_plus": zp,
            "residue_minus": rm,
            "residue_plus": rp,
            "reconstructed_zero": rec.response_zero.real,
            "reconstructed_bright_pole": rec.companion_pole.real,
            "reconstructed_coupling_product": rec.coupling_product.real,
            "zero_field_charge": wc0,
            "zero_field_axial": wa0,
            "dark_zero_error": rec.response_zero.real - wa0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "fig2_pole_zero_tomography.csv", index=False)

    # Circle-law data from exact Weyl poles and residues.
    circle_rows = []
    for b in (0.0025, 0.005, 0.010):
        min_gap = minimize_gap(model, b)["gap"]
        for q in np.linspace(0.82 * q0, 1.18 * q0, 33):
            modes = model.bracketed_real_modes(float(q), b, n_linear=480, n_log=210)
            if len(modes) != 2:
                continue
            zm, zp = modes[0].omega_bar, modes[1].omega_bar
            rm = model.dielectric_residue(zm, float(q), b)
            rp = model.dielectric_residue(zp, float(q), b)
            asym = (rp - rm) / (rp + rm)
            split_ratio = min_gap / (zp - zm)
            circle_rows.append({
                "b": b,
                "qbar": q,
                "residue_asymmetry": asym,
                "minimum_gap_over_local_split": split_ratio,
                "circle_sum": asym * asym + split_ratio * split_ratio,
            })
    cdf = pd.DataFrame(circle_rows)
    cdf.to_csv(DATA / "fig2_frequency_residue_circle.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.72))
    ax = axes[0]
    ax.plot(df.qbar, df.zero_field_charge, ls="--", lw=1.0, label="uncoupled charge")
    ax.plot(df.qbar, df.zero_field_axial, ls=":", lw=1.2, label="hidden axial pole, $b=0$")
    ax.plot(df.qbar, df.omega_minus, lw=1.25, label="hybrid poles")
    ax.plot(df.qbar, df.omega_plus, lw=1.25)
    ax.scatter(df.qbar, df.reconstructed_zero, s=11, marker="x", label="zero reconstructed from poles+residues")
    ax.axvline(q0, lw=0.7, ls="-.")
    ax.set_xlabel(r"$\bar q$")
    ax.set_ylabel(r"$\bar\omega$")
    ax.set_title("hidden-pole reconstruction")
    ax.legend(frameon=False, loc="upper left", fontsize=6.2)
    ax.text(0.03, 0.04, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    x = np.linspace(-1.0, 1.0, 600)
    ax.plot(x, np.sqrt(np.maximum(0.0, 1.0 - x * x)), lw=1.3, label=r"$\mathcal{A}_R^2+\mathcal{R}_\omega^2=1$")
    markers = {0.0025: "o", 0.005: "s", 0.010: "^"}
    for b, sub in cdf.groupby("b"):
        ax.scatter(sub.residue_asymmetry, sub.minimum_gap_over_local_split,
                   s=13, marker=markers[float(b)], label=rf"exact Weyl model, $b={b:g}$")
    ax.set_xlabel(r"$\mathcal{A}_R=(R_+-R_-)/(R_++R_-)$")
    ax.set_ylabel(r"$\mathcal{R}_\omega=\Delta\omega_{\min}/(\omega_+-\omega_-)$")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-0.03, 1.06)
    ax.set_title("frequency--residue circle law")
    ax.legend(frameon=False, loc="lower center", fontsize=6.1)
    ax.text(0.03, 0.04, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.subplots_adjust(wspace=0.30)
    save(fig, "fig2_pole_zero_tomography", "Pole-zero tomography and frequency-residue circle law")

    return {
        "q0": q0,
        "b_show": b_show,
        "max_abs_hidden_zero_error": float(np.max(np.abs(df.dark_zero_error))),
        "max_abs_circle_residual_b_0p0025": float(np.max(np.abs(cdf[cdf.b == 0.0025].circle_sum - 1.0))),
        "max_abs_circle_residual_b_0p01": float(np.max(np.abs(cdf[cdf.b == 0.010].circle_sum - 1.0))),
    }


def complex_tomography_robustness() -> dict:
    rng = np.random.default_rng(20260623)
    z_b = 0.198 - 0.006j
    z_d = 0.306 - 0.021j
    g2 = (0.033 + 0.006j) ** 2
    pref = 0.92 + 0.08j
    rm, rp, (zm, zp) = residues_from_bare(z_b, z_d, g2, prefactor=pref)
    omega = np.linspace(0.10, 0.40, 360) + 0.004j
    bg = (0.025 - 0.015j) + (0.018 + 0.009j) * omega
    exact = bg + singular_response(omega, zm, zp, z_d, prefactor=pref)

    snrs = np.array([30, 50, 80, 120, 200, 400, 800], float)
    rows = []
    for snr in snrs:
        scale = np.sqrt(np.mean(np.abs(exact) ** 2)) / snr
        for rep in range(80):
            noise = scale / np.sqrt(2.0) * (rng.normal(size=omega.size) + 1j * rng.normal(size=omega.size))
            fit = fit_two_pole_rational(omega, exact + noise, background_order=1)
            fit_poles = sorted(fit.poles, key=lambda z: z.real)
            true_poles = sorted((zm, zp), key=lambda z: z.real)
            rows.append({
                "snr": snr,
                "rep": rep,
                "zero_error": abs(fit.response_zero - z_d),
                "pole_error_max": max(abs(fit_poles[i] - true_poles[i]) for i in (0, 1)),
                "relative_fit_rms": fit.relative_rms,
                "design_condition_number": fit.condition_number,
            })
    ndf = pd.DataFrame(rows)
    ndf.to_csv(DATA / "figS4_complex_tomography_noise.csv", index=False)

    # Third-pole contamination: a two-pole fit should self-diagnose through its residual.
    distances = np.geomspace(0.06, 1.0, 18)
    strengths = (0.02, 0.05, 0.10, 0.20)
    crow = []
    for strength in strengths:
        for dist in distances:
            third = strength / (omega - (0.25 + dist - 0.07j))
            fit = fit_two_pole_rational(omega, exact + third, background_order=1)
            crow.append({
                "third_pole_distance": dist,
                "third_pole_strength": strength,
                "relative_two_pole_fit_rms": fit.relative_rms,
                "zero_error": abs(fit.response_zero - z_d),
                "design_condition_number": fit.condition_number,
            })
    tdf = pd.DataFrame(crow)
    tdf.to_csv(DATA / "figS4_three_mode_diagnostic.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.72))
    ax = axes[0]
    med = ndf.groupby("snr").median(numeric_only=True)
    p90 = ndf.groupby("snr").quantile(0.9, numeric_only=True)
    ax.loglog(med.index, med.zero_error, marker="o", label="median response-zero error")
    ax.loglog(p90.index, p90.zero_error, marker="s", ls="--", label="90th percentile")
    ax.loglog(med.index, med.pole_error_max, marker="^", label="median pole error")
    ax.set_xlabel("complex-response SNR")
    ax.set_ylabel("absolute frequency error")
    ax.set_title("complex response + analytic background")
    ax.legend(frameon=False)
    ax.text(0.03, 0.04, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    for strength, sub in tdf.groupby("third_pole_strength"):
        ax.loglog(sub.third_pole_distance, sub.relative_two_pole_fit_rms, marker="o", label=rf"third-pole weight {strength:g}")
    ax.axhline(1e-3, ls="--", lw=0.8, label="illustrative adequacy threshold")
    ax.set_xlabel("distance to omitted pole")
    ax.set_ylabel("relative two-pole fit residual")
    ax.set_title("automatic model-order diagnostic")
    ax.legend(frameon=False, fontsize=6.0)
    ax.text(0.03, 0.04, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.subplots_adjust(wspace=0.30)
    save(fig, "figS4_complex_tomography", "Complex pole-zero tomography and model-order diagnostics")

    return {
        "exact_complex_zero": [z_d.real, z_d.imag],
        "median_zero_error_snr_100_approx": float(med.iloc[np.argmin(np.abs(med.index.to_numpy() - 120))].zero_error),
        "two_pole_fit_residual_nearest_third_mode": float(tdf.loc[tdf.third_pole_distance.idxmin(), "relative_two_pole_fit_rms"]),
    }


def genealogy_inference_figure() -> dict:
    """Identical-pipeline comparison with the exact anomaly crossover."""
    rng = np.random.default_rng(20260624)
    spans = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    rel_noise = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.10])
    bmax = 0.020
    bref = 0.010
    q0 = 0.093
    bc_true = 0.010
    qinf_true = q0 / np.sqrt(1.0 + (bc_true / bref) ** 2)
    npts = 9
    trials = 160
    rows = []

    for span in spans:
        b = np.geomspace(bmax / span, bmax, npts)
        q_finite = q0 * (1.0 + 0.30 * b / bref + 0.05 * (b / bref) ** 2)
        q_vanish = qinf_true * np.sqrt(1.0 + (bc_true / b) ** 2)
        for noise in rel_noise:
            for truth, qtrue in (("finite", q_finite), ("vanishing", q_vanish)):
                correct = 0
                signed_dbic = []
                for _ in range(trials):
                    sigma = noise * qtrue
                    obs = qtrue + rng.normal(scale=sigma)
                    label, finite_fit, vanish_fit, dbic = classify_genealogy_exact(b, obs, sigma)
                    correct += int(label == truth)
                    signed_dbic.append(dbic if truth == "finite" else -dbic)
                rows.append({
                    "field_span": span,
                    "relative_q_noise": noise,
                    "truth": truth,
                    "accuracy": correct / trials,
                    "median_correct_model_delta_BIC": float(np.median(signed_dbic)),
                    "trials": trials,
                })
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "figS5_genealogy_model_selection.csv", index=False)

    # Representative data at the protocol point quoted in the text.
    span = 8.0
    noise = 0.01
    b = np.geomspace(bmax / span, bmax, npts)
    qfinite_true = q0 * (1.0 + 0.30 * b / bref + 0.05 * (b / bref) ** 2)
    qvanish_true = qinf_true * np.sqrt(1.0 + (bc_true / b) ** 2)
    rep_rows = []
    representative = []
    for truth, qtrue, seed in (("finite", qfinite_true, 701), ("vanishing", qvanish_true, 702)):
        local_rng = np.random.default_rng(seed)
        sigma = noise * qtrue
        obs = qtrue + local_rng.normal(scale=sigma)
        label, finite_fit, vanish_fit, dbic = classify_genealogy_exact(b, obs, sigma)
        finite_pred = (
            finite_fit.parameters["q0"]
            + finite_fit.parameters["c1"] * b
            + finite_fit.parameters["c2"] * b * b
        )
        vanish_pred = vanish_fit.parameters["q_inf"] * np.sqrt(
            1.0 + (vanish_fit.parameters["B_c"] / b) ** 2
        )
        for i in range(len(b)):
            rep_rows.append({
                "truth": truth,
                "abs_b": b[i],
                "q_true": qtrue[i],
                "q_observed": obs[i],
                "sigma": sigma[i],
                "finite_fit": finite_pred[i],
                "exact_vanishing_fit": vanish_pred[i],
                "selected_model": label,
                "delta_BIC_vanishing_minus_finite": dbic,
            })
        representative.append((truth, obs, sigma, finite_pred, vanish_pred, label, dbic))
    rdf = pd.DataFrame(rep_rows)
    rdf.to_csv(DATA / "figS5_genealogy_representative_fits.csv", index=False)

    # Premium supplement styling without relying on a global style sheet.
    INK = "#18232E"
    TEAL = "#2A7F78"
    VERMILION = "#C65A46"
    NAVY = "#244A64"
    OCHRE = "#B7862D"
    PALE = "#F3F5F7"
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.72))

    # Panel (a): worst-case accuracy across the two classes.
    pivot = {}
    for truth in ("finite", "vanishing"):
        sub = df[df.truth == truth]
        pivot[truth] = sub.pivot(index="relative_q_noise", columns="field_span", values="accuracy")
    worst = np.minimum(pivot["finite"].to_numpy(), pivot["vanishing"].to_numpy())
    ax = axes[0]
    im = ax.imshow(
        worst,
        origin="lower",
        aspect="auto",
        vmin=0.5,
        vmax=1.0,
        cmap="cividis",
        extent=[-0.5, len(spans) - 0.5, -0.5, len(rel_noise) - 0.5],
    )
    ax.set_xticks(np.arange(len(spans)), labels=[f"{x:g}" for x in spans])
    ax.set_yticks(np.arange(len(rel_noise)), labels=[f"{100*x:g}%" for x in rel_noise])
    ax.set_xlabel(r"field dynamic range $B_{\max}/B_{\min}$")
    ax.set_ylabel("relative wave-vector uncertainty")
    ax.set_title("worst-case classification probability", loc="left")
    ax.text(0.03, 0.05, "(a)", transform=ax.transAxes, fontweight="bold", color="white")
    # Print values directly to avoid a colorbar competing with the adjacent panel.
    for iy in range(worst.shape[0]):
        for ix in range(worst.shape[1]):
            value = worst[iy, ix]
            color = "#18232E" if value > 0.82 else "white"
            ax.text(
                ix,
                iy,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color=color,
            )

    # Panel (b): both truth classes through the same two-model pipeline.
    ax = axes[1]
    styles = {
        "finite": (TEAL, "o"),
        "vanishing": (VERMILION, "s"),
    }
    for truth, obs, sigma, finite_pred, vanish_pred, label, dbic in representative:
        color, marker = styles[truth]
        norm = q0
        ax.errorbar(
            b / bref,
            obs / norm,
            yerr=sigma / norm,
            fmt=marker,
            ms=4.2,
            mfc="white",
            mec=color,
            mew=1.0,
            ecolor=color,
            elinewidth=0.7,
            capsize=1.6,
            label=f"{truth} truth",
            zorder=4,
        )
        if truth == "finite":
            ax.plot(b / bref, finite_pred / norm, color=color, lw=1.6, label="selected finite fit")
            ax.plot(b / bref, vanish_pred / norm, color=OCHRE, lw=0.9, ls="--", alpha=0.75, label="rejected exact-crossover fit")
        else:
            ax.plot(b / bref, vanish_pred / norm, color=color, lw=1.6, label="selected exact-crossover fit")
            ax.plot(b / bref, finite_pred / norm, color=NAVY, lw=0.9, ls=":", alpha=0.75, label="rejected finite fit")
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1.0, 2.0], labels=["0.25", "0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.23, 2.15)
    ax.set_xlabel(r"$|B|/B_{\rm ref}$")
    ax.set_ylabel(r"$q/q_0$")
    ax.set_title("identical inference pipeline", loc="left")
    ax.grid(axis="y", color="#DDE2E6", lw=0.45)
    ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper right", columnspacing=0.8)
    ax.text(0.03, 0.05, "(b)", transform=ax.transAxes, fontweight="bold", color=INK)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")
    fig.subplots_adjust(wspace=0.34)
    save(fig, "figS5_genealogy_inference", "Exact-crossover finite-window genealogy inference")

    out = {
        "minimum_accuracy": float(df.accuracy.min()),
        "maximum_accuracy": float(df.accuracy.max()),
        "accuracy_span8_noise1pct_finite": float(df[(df.field_span == 8.0) & (df.relative_q_noise == 0.01) & (df.truth == "finite")].accuracy.iloc[0]),
        "accuracy_span8_noise1pct_vanishing": float(df[(df.field_span == 8.0) & (df.relative_q_noise == 0.01) & (df.truth == "vanishing")].accuracy.iloc[0]),
        "exact_competitor_Bc": bc_true,
        "exact_competitor_qinf": qinf_true,
    }
    (DATA / "figS5_genealogy_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def symmetry_adversarial_figure() -> dict:
    b = np.geomspace(2e-4, 3e-2, 90)
    # Three analytic normal forms: symmetric linear mixing, broken-symmetry
    # diagonal shift, and selection-rule-suppressed quadratic mixing.
    q0 = 0.1
    q_even = q0 + 0.8 * b**2
    q_odd = q0 + 0.45 * b + 0.8 * b**2
    gap_linear = 0.7 * b
    gap_quadratic = 8.0 * b**2
    df = pd.DataFrame({
        "b": b,
        "qmin_node_exchange": q_even,
        "qmin_symmetry_broken": q_odd,
        "gap_linear_mixing": gap_linear,
        "gap_quadratic_mixing": gap_quadratic,
    })
    df.to_csv(DATA / "figS6_symmetry_selection_rules.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.72))
    ax = axes[0]
    ax.loglog(b, np.abs(q_even - q0), label=r"node exchange: $|q_{\min}-q_0|\propto B^2$")
    ax.loglog(b, np.abs(q_odd - q0), label=r"broken covariance: $|q_{\min}-q_0|\propto B$")
    ax.set_xlabel(r"$|B|$ (arb. units)")
    ax.set_ylabel(r"$|q_{\min}-q_0|$")
    ax.set_title("intercept survives diagonal odd terms")
    ax.legend(frameon=False)
    ax.text(0.03, 0.04, "(a)", transform=ax.transAxes, fontweight="bold")
    ax = axes[1]
    ax.loglog(b, gap_linear, label=r"first allowed mixing $\propto B$")
    ax.loglog(b, gap_quadratic, label=r"selection rule: mixing $\propto B^2$")
    ax.set_xlabel(r"$|B|$ (arb. units)")
    ax.set_ylabel(r"$\Delta\omega_{\min}$")
    ax.set_title("gap exponent measures mixing order")
    ax.legend(frameon=False)
    ax.text(0.03, 0.04, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.subplots_adjust(wspace=0.30)
    save(fig, "figS6_symmetry_selection_rules", "Adversarial symmetry and selection-rule tests")
    return {"q0": q0}


def main() -> None:
    summary = {
        "main_tomography": main_tomography_figure(),
        "complex_robustness": complex_tomography_robustness(),
        "genealogy_inference": genealogy_inference_figure(),
        "symmetry_adversarial": symmetry_adversarial_figure(),
    }
    (DATA / "pole_zero_reinforcement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
