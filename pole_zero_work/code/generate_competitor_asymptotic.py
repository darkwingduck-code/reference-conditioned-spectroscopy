#!/usr/bin/env python3
"""Exact crossover implied by Eq. (9) of arXiv:2605.27031v1.

The script isolates the dimensionless content of the reported long-wavelength
velocity.  It does not extrapolate the acoustic approximation beyond its own
regime of validity; the large-x branch is shown only to expose the crossover
of the local trajectory exponent within that formula.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
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
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
})


def anomaly_normalized_crossing(x: np.ndarray | float) -> np.ndarray:
    """q_h/q_* = sqrt(1+x^2)/x for x>0."""
    arr = np.asarray(x, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("x must be strictly positive")
    return np.sqrt(1.0 + arr * arr) / arr


def anomaly_genealogy_exponent(x: np.ndarray | float) -> np.ndarray:
    """-d ln q_h / d ln x = 1/(1+x^2)."""
    arr = np.asarray(x, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("x must be strictly positive")
    return 1.0 / (1.0 + arr * arr)


def finite_intercept_reference(x: np.ndarray | float, a2: float = 0.08) -> np.ndarray:
    """Bounded even-in-field example with q/q0 -> 1 as x -> 0."""
    arr = np.asarray(x, dtype=float)
    return 1.0 + a2 * arr * arr / (1.0 + arr * arr)


def finite_intercept_exponent(x: np.ndarray | float, a2: float = 0.08) -> np.ndarray:
    """Local exponent for finite_intercept_reference."""
    arr = np.asarray(x, dtype=float)
    q = finite_intercept_reference(arr, a2=a2)
    # d q / d ln x = 2 a2 x^2/(1+x^2)^2
    return -(2.0 * a2 * arr * arr / (1.0 + arr * arr) ** 2) / q


def main() -> None:
    # x = |v_Omega| q_TF /(sqrt(epsilon_inf) omega_p) is proportional to |B|.
    x = np.geomspace(1.0e-3, 10.0, 600)
    q_anom = anomaly_normalized_crossing(x)
    q_weak = 1.0 / x
    zeta_anom = anomaly_genealogy_exponent(x)

    q_finite = finite_intercept_reference(x)
    zeta_finite = finite_intercept_exponent(x)

    df = pd.DataFrame({
        "x_vOmega_qTF_over_sqrt_eps_omegaP": x,
        "normalized_anomaly_crossing_q": q_anom,
        "weak_field_1_over_x": q_weak,
        "anomaly_effective_zeta": zeta_anom,
        "representative_finite_intercept_q_over_q0": q_finite,
        "representative_finite_intercept_zeta": zeta_finite,
    })
    df.to_csv(DATA / "figS9_competitor_exact_trajectory.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), layout="constrained")
    ax = axes[0]
    ax.loglog(x, q_anom, lw=1.5, label="Eq. (9) crossing")
    ax.loglog(x, q_weak, ls="--", lw=1.0, label=r"weak-field $1/x$")
    ax.loglog(x, q_finite, ls=":", lw=1.4, label="finite-intercept example")
    ax.set_xlabel(r"$x=|v_\Omega|q_{\rm TF}/(\sqrt{\epsilon_\infty}\omega_p)\propto |B|$")
    ax.set_ylabel("normalized crossing wave vector")
    ax.set_title("exact crossover within the acoustic formula")
    ax.legend(frameon=False, loc="best")
    ax.text(0.04, 0.06, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    ax.semilogx(x, zeta_anom, lw=1.5, label="vanishing-velocity branch")
    ax.semilogx(x, zeta_finite, ls=":", lw=1.4, label="finite-intercept example")
    ax.axhline(1.0, ls="--", lw=0.8)
    ax.axhline(0.0, ls="--", lw=0.8)
    ax.set_xlabel(r"$x\propto |B|$")
    ax.set_ylabel(r"$\zeta=-d\ln q_\times/d\ln|B|$")
    ax.set_title("local genealogy exponent")
    ax.set_ylim(-0.12, 1.08)
    ax.legend(frameon=False, loc="best")
    ax.text(0.04, 0.06, "(b)", transform=ax.transAxes, fontweight="bold")

    meta = {
        "Title": "Exact anomaly-mode crossing crossover and genealogy exponent",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(FIG / "figS9_competitor_exact_trajectory.pdf", bbox_inches="tight", metadata=meta)
    fig.savefig(FIG / "figS9_competitor_exact_trajectory.png", bbox_inches="tight", dpi=260)
    plt.close(fig)

    summary = {
        "definition_x": "abs(v_Omega)*q_TF/(sqrt(epsilon_inf)*omega_p)",
        "definition_q_scale": "q_TF/(sqrt(epsilon_inf-1)*abs(cos(theta)))",
        "normalized_exact_crossing": "sqrt(1+x^2)/x",
        "exact_genealogy_exponent": "1/(1+x^2)",
        "low_field_zeta_at_x_1e-3": float(zeta_anom[0]),
        "zeta_at_x_1": 0.5,
        "high_x_zeta_at_x_10": float(zeta_anom[-1]),
        "interpretation": (
            "The 1/|B| law is the x->0 asymptote. Within the reported acoustic "
            "velocity formula the local exponent crosses from one to zero as x grows; "
            "therefore genealogy must be inferred from the controlled weak-field trend, "
            "not from a single finite-field point."
        ),
    }
    (DATA / "competitor_exact_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
