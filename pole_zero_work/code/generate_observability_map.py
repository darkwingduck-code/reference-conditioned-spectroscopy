#!/usr/bin/env python3
"""Generate an uncertainty map for complex pole-zero tomography."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from pole_zero_tomography import fit_two_pole_rational, residues_from_bare, singular_response

FIG = ROOT / "fig_prl"
DATA = ROOT / "data_prl"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.2,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
})


def save(fig: plt.Figure, stem: str, title: str) -> None:
    metadata = {"Title": title, "Author": "Authors to be inserted", "CreationDate": None, "ModDate": None}
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(20260628)
    omega0 = 0.25
    gamma_b = 0.010
    gamma_d = 0.016
    gamma_mean = 0.5 * (gamma_b + gamma_d)
    ratios = np.geomspace(0.2, 6.0, 9)  # minimum splitting / summed half-width scale
    snrs = np.geomspace(20, 800, 9)
    omega = np.linspace(omega0 - 0.085, omega0 + 0.085, 360) + 0.0j
    rows = []
    trials = 45

    for ratio in ratios:
        # 2g / (gamma_b+gamma_d) = ratio.
        g = 0.5 * ratio * (gamma_b + gamma_d)
        zb = omega0 - 1j * gamma_b
        zd = omega0 - 1j * gamma_d
        rm, rp, (zm, zp) = residues_from_bare(zb, zd, g * g, prefactor=1.0)
        exact = (0.04 - 0.02j) + (0.05 + 0.015j) * (omega - omega0) + singular_response(omega, zm, zp, zd)
        rms = float(np.sqrt(np.mean(np.abs(exact) ** 2)))
        for snr in snrs:
            sigma = rms / snr
            errors = []
            pole_errors = []
            conditions = []
            failures = 0
            for _ in range(trials):
                noise = sigma / np.sqrt(2.0) * (rng.normal(size=omega.size) + 1j * rng.normal(size=omega.size))
                try:
                    fit = fit_two_pole_rational(omega, exact + noise, background_order=1)
                    errors.append(abs(fit.response_zero - zd) / gamma_mean)
                    fp = sorted(fit.poles, key=lambda z: z.real)
                    tp = sorted((zm, zp), key=lambda z: z.real)
                    pole_errors.append(max(abs(fp[j] - tp[j]) for j in (0, 1)) / gamma_mean)
                    conditions.append(fit.condition_number)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    failures += 1
            arr = np.asarray(errors, float)
            parr = np.asarray(pole_errors, float)
            rows.append({
                "gap_to_sum_halfwidth_ratio": ratio,
                "complex_response_snr": snr,
                "trials": trials,
                "fit_failures": failures,
                "median_zero_error_over_gamma": float(np.median(arr)) if arr.size else np.inf,
                "p90_zero_error_over_gamma": float(np.quantile(arr, 0.9)) if arr.size else np.inf,
                "median_pole_error_over_gamma": float(np.median(parr)) if parr.size else np.inf,
                "probability_zero_error_below_0p2gamma": float(np.mean(arr < 0.2)) if arr.size else 0.0,
                "median_design_condition_number": float(np.median(conditions)) if conditions else np.inf,
            })

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "figS8_tomography_observability_map.csv", index=False)
    err = df.pivot(index="complex_response_snr", columns="gap_to_sum_halfwidth_ratio", values="median_zero_error_over_gamma")
    prob = df.pivot(index="complex_response_snr", columns="gap_to_sum_halfwidth_ratio", values="probability_zero_error_below_0p2gamma")

    x = np.log10(ratios)
    y = np.log10(snrs)
    extent = [x[0], x[-1], y[0], y[-1]]
    fig = plt.figure(figsize=(7.15, 2.78), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=(1.0, 0.045, 1.0, 0.045), wspace=0.08)
    ax0 = fig.add_subplot(gs[0, 0])
    cax0 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[0, 2])
    cax1 = fig.add_subplot(gs[0, 3])

    im0 = ax0.imshow(np.log10(err.to_numpy()), origin="lower", aspect="auto", extent=extent)
    ax0.set_xticks(x, labels=[f"{v:.2g}" for v in ratios])
    ax0.set_yticks(y, labels=[f"{v:.0f}" for v in snrs])
    ax0.set_xlabel(r"$\Delta\omega_{\min}/(\gamma_b+\gamma_d)$")
    ax0.set_ylabel("complex-response SNR")
    ax0.set_title(r"median $|\widehat z_0-z_d|/\bar\gamma$")
    cb0 = fig.colorbar(im0, cax=cax0)
    cb0.set_label(r"$\log_{10}$ error", labelpad=5)
    ax0.text(0.03, 0.05, "(a)", transform=ax0.transAxes, fontweight="bold")

    im1 = ax1.imshow(prob.to_numpy(), origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0)
    ax1.set_xticks(x, labels=[f"{v:.2g}" for v in ratios])
    ax1.set_yticks(y, labels=[f"{v:.0f}" for v in snrs])
    ax1.set_xlabel(r"$\Delta\omega_{\min}/(\gamma_b+\gamma_d)$")
    ax1.set_title(r"$P(|\widehat z_0-z_d|<0.2\bar\gamma)$")
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label("success probability", labelpad=5)
    ax1.text(0.03, 0.05, "(b)", transform=ax1.transAxes, fontweight="bold")
    save(fig, "figS8_tomography_observability_map", "Noise and linewidth observability map for pole-zero tomography")

    summary = {
        "trials_per_grid_point": trials,
        "best_median_zero_error_over_gamma": float(df.median_zero_error_over_gamma.min()),
        "worst_median_zero_error_over_gamma": float(df.median_zero_error_over_gamma.max()),
        "success_at_ratio_1_snr_100_nearest_grid": float(df.iloc[((np.log(df.gap_to_sum_halfwidth_ratio)-np.log(1.0))**2 + (np.log(df.complex_response_snr)-np.log(100.0))**2).argmin()].probability_zero_error_below_0p2gamma),
        "note": "Protocol benchmark for a specified rational model, not a universal experimental threshold.",
    }
    (DATA / "tomography_observability_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
