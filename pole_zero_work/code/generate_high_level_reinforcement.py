#!/usr/bin/env python3
"""Additional robustness data for the PRL candidate.

This script addresses two reviewer-level questions not closed by the original
illustrative parameter set:

1. Does the zero-field axial/plasmon crossing require a negative l=0
   intervalley harmonic?  No.  An l=1 axial enhancement can produce a crossing
   even when F0x=(F0c-F0a)/2 >= 0.
2. Does the exact inverse-dielectric spectral weight obey the two-mode sum rule
   as B -> 0?  Yes, with deviations of order B^2 in the reduced model.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from weyl_fl_prb_solver import (  # noqa: E402
    ModelParams,
    ProjectedModel,
    degenerate_perturbation_data,
    landau_D_real,
    minimize_gap,
    nominal_crossing,
    sector_root,
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
    }
)


def savefig(fig: plt.Figure, stem: str, title: str) -> None:
    metadata = {
        "Title": title,
        "Author": "Authors to be inserted",
        "Subject": "Supplemental robustness figure",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def zero_field_frequency(params: ModelParams, qbar: float, s_a: float) -> tuple[float, float]:
    f0c_eff = params.F0c + 4.0 * params.alpha / (
        np.pi * (qbar * qbar + params.qTFbar * params.qTFbar)
    )
    s_c = sector_root(f0c_eff, params.F1c)
    return qbar * s_c, qbar * s_a


def interaction_robustness_figure() -> dict:
    # Fix the charge channel.  The vertical line F0a=F0c separates negative
    # and nonnegative l=0 intervalley harmonic F0x=(F0c-F0a)/2.
    F0c, F1c, alpha = 1.0, 0.0, 0.02
    F0a_grid = np.linspace(0.10, 1.45, 82)
    F1a_grid = np.linspace(0.10, 2.60, 88)
    q_map = np.full((len(F1a_grid), len(F0a_grid)), np.nan)
    s_map = np.full_like(q_map, np.nan)
    records: list[dict] = []

    for iy, F1a in enumerate(F1a_grid):
        for ix, F0a in enumerate(F0a_grid):
            try:
                s_a = sector_root(float(F0a), float(F1a))
            except RuntimeError:
                continue
            D = landau_D_real(s_a)
            f0c_star = 1.0 / D  # because F1c=0
            needed = f0c_star - F0c
            if needed <= 0.0:
                continue
            qbar = np.sqrt(4.0 * alpha / (np.pi * needed))
            # Keep the controlled long-wavelength region in the displayed map.
            if qbar > 0.30:
                continue
            q_map[iy, ix] = qbar
            s_map[iy, ix] = s_a
            records.append(
                {
                    "F0c": F0c,
                    "F1c": F1c,
                    "F0a": F0a,
                    "F1a": F1a,
                    "F0x": 0.5 * (F0c - F0a),
                    "F1x": 0.5 * (F1c - F1a),
                    "s_axial": s_a,
                    "qbar_crossing": qbar,
                    "inside_qbar_0p2": bool(qbar <= 0.2),
                }
            )

    pd.DataFrame(records).to_csv(DATA / "figS1_interaction_crossing_map.csv", index=False)

    # A concrete alternative point with nonnegative l=0 intervalley harmonic.
    alt = ModelParams(
        F0c=1.0,
        F0a=0.8,
        F1c=0.0,
        F1a=1.0,
        alpha=0.02,
        qTFbar=0.0,
        quadrature_order=128,
    )
    q0, s_a, omega0 = nominal_crossing(alt)
    perturb = degenerate_perturbation_data(alt)
    model = ProjectedModel(alt, lmax=2)
    b_show = 0.010
    qgrid = np.linspace(0.072, 0.130, 52)
    disp_rows: list[dict] = []
    wc0, wa0, lower, upper = [], [], [], []
    for q in qgrid:
        wc, wa = zero_field_frequency(alt, float(q), s_a)
        modes = model.bracketed_real_modes(float(q), b_show, n_linear=360, n_log=170)
        if len(modes) != 2:
            raise RuntimeError(f"Alternative-point root failure at q={q}")
        wc0.append(wc)
        wa0.append(wa)
        lower.append(modes[0].omega_bar)
        upper.append(modes[1].omega_bar)
        disp_rows.extend(
            [
                {"qbar": q, "b": 0.0, "branch": "charge", "omega_bar": wc},
                {"qbar": q, "b": 0.0, "branch": "axial", "omega_bar": wa},
                {"qbar": q, "b": b_show, "branch": "lower", "omega_bar": modes[0].omega_bar},
                {"qbar": q, "b": b_show, "branch": "upper", "omega_bar": modes[1].omega_bar},
            ]
        )
    pd.DataFrame(disp_rows).to_csv(DATA / "figS1_alternative_dispersion.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    ax = axes[0]
    masked = np.ma.masked_invalid(q_map)
    mesh = ax.pcolormesh(F0a_grid, F1a_grid, masked, shading="auto")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(r"$\bar q_0$")
    ax.axvline(F0c, ls="--", lw=0.9, label=r"$F_{0x}=0$")
    ax.scatter([alt.F0a], [alt.F1a], marker="*", s=72, edgecolor="k", zorder=5, label="alternative point")
    ax.set_xlabel(r"$F_{0a}$")
    ax.set_ylabel(r"$F_{1a}$")
    ax.set_title(r"crossing region at $F_{0c}=1,\ F_{1c}=0$")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.03, 0.04, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    ax.plot(qgrid, wc0, ls="--", lw=1.0, label=r"charge, $b=0$")
    ax.plot(qgrid, wa0, ls=":", lw=1.2, label=r"axial, $b=0$")
    ax.plot(qgrid, lower, lw=1.15, label=rf"hybrid, $b={b_show:.3f}$")
    ax.plot(qgrid, upper, lw=1.15)
    ax.axvline(q0, ls="-.", lw=0.8)
    ax.fill_between(qgrid, 0.0, qgrid, alpha=0.08)
    ax.plot(qgrid, qgrid, ls="--", lw=0.7)
    ax.set_xlabel(r"$\bar q$")
    ax.set_ylabel(r"$\bar\omega$")
    ax.set_title(r"$F_{0x}=+0.1$: exact avoided crossing")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.04, "(b)", transform=ax.transAxes, fontweight="bold")

    fig.subplots_adjust(wspace=0.34)
    savefig(
        fig,
        "figS1_interaction_robustness",
        "Axial-plasmon crossing without a negative l=0 intervalley harmonic",
    )

    # Exact minimum-gap checks at the alternative point.
    gap_rows = []
    for b in (0.0025, 0.005, 0.010, 0.020):
        result = minimize_gap(model, b, qcenter=q0, relative_half_width=0.18)
        gap_rows.append(
            {
                "b": b,
                "qmin": result["qmin"],
                "gap": result["gap"],
                "gap_over_b": result["gap"] / b,
                "analytic_gap_over_b": perturb["gap_linear_coefficient"],
                "relative_error": abs(result["gap"] / (b * perturb["gap_linear_coefficient"]) - 1.0),
            }
        )
    pd.DataFrame(gap_rows).to_csv(DATA / "alternative_parameter_gap_check.csv", index=False)

    summary = {
        "alternative_parameters": {
            "F0c": alt.F0c,
            "F0a": alt.F0a,
            "F1c": alt.F1c,
            "F1a": alt.F1a,
            "F0x": alt.Fx,
            "F1x": alt.F1x,
            "alpha": alt.alpha,
        },
        "qstar": q0,
        "s_axial": s_a,
        "omega_star": omega0,
        "gap_linear_coefficient": perturb["gap_linear_coefficient"],
        "phase_velocity_gap_coefficient": perturb["gap_linear_coefficient"] / q0,
        "valid_map_points": int(np.count_nonzero(np.isfinite(q_map))),
        "valid_points_with_F0x_nonnegative": int(
            sum(1 for r in records if r["F0x"] >= -1e-12)
        ),
    }
    return summary


def spectral_weight_sum_rule() -> dict:
    df = pd.read_csv(DATA / "fig2_universal_collapse.csv")
    piv = df.pivot_table(
        index=["b", "X_target"], columns="branch", values="normalized_residue"
    ).reset_index()
    piv["sum_residues"] = piv["lower"] + piv["upper"]
    piv["sum_rule_error"] = piv["sum_residues"] - 1.0
    piv.to_csv(DATA / "spectral_weight_sum_rule.csv", index=False)

    rows = []
    for b, group in piv.groupby("b"):
        rows.append(
            {
                "b": b,
                "max_abs_sum_rule_error": float(np.max(np.abs(group.sum_rule_error))),
                "mean_sum_rule_error": float(np.mean(group.sum_rule_error)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("b")
    summary.to_csv(DATA / "spectral_weight_sum_rule_summary.csv", index=False)
    slope = float(
        np.polyfit(
            np.log(summary.b.to_numpy()),
            np.log(summary.max_abs_sum_rule_error.to_numpy()),
            1,
        )[0]
    )
    return {
        "error_power_vs_b": slope,
        "max_error_at_smallest_b": float(summary.iloc[0].max_abs_sum_rule_error),
        "max_error_at_largest_b": float(summary.iloc[-1].max_abs_sum_rule_error),
    }


def physical_scale_table() -> dict:
    # SI conversion: b = e B/(2 hbar k_F^2), k_F=mu/(hbar v_F).
    # Use the main illustrative coefficient C_omega=Delta(omega/mu)/|b|.
    base = ModelParams()
    Cw = degenerate_perturbation_data(base)["gap_linear_coefficient"]
    hbar_eVs = 6.582119569e-16
    e_C = 1.602176634e-19
    hbar_Js = 1.054571817e-34
    eV_J = e_C
    hv_eVA = 1.0
    v_m_s = (hv_eVA * 1.0e-10) / hbar_eVs
    rows = []
    for mu_meV in (30.0, 50.0, 80.0):
        mu_eV = mu_meV * 1.0e-3
        kF_m = (mu_eV / hv_eVA) * 1.0e10
        B_for_b001 = 2.0 * 0.01 * hbar_Js * kF_m * kF_m / e_C
        gap_meV = mu_meV * Cw * 0.01
        gap_slope_meV_T = gap_meV / B_for_b001
        rows.append(
            {
                "mu_meV": mu_meV,
                "hbar_vF_eV_A": hv_eVA,
                "vF_m_per_s": v_m_s,
                "kF_Ainv": mu_eV / hv_eVA,
                "B_T_for_b_0p01": B_for_b001,
                "gap_meV_at_b_0p01": gap_meV,
                "gap_slope_meV_per_T": gap_slope_meV_T,
                "nF_estimate_at_b_0p01": 25.0,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "physical_scale_table.csv", index=False)
    return {"gap_coefficient": Cw, "rows": rows}


def main() -> None:
    result = {
        "interaction_robustness": interaction_robustness_figure(),
        "spectral_weight_sum_rule": spectral_weight_sum_rule(),
        "physical_scales": physical_scale_table(),
    }
    (DATA / "high_level_reinforcement_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
