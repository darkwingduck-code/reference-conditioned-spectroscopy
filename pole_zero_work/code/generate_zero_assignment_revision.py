"""Generate the synthetic zero-assignment correction evidence and Fig. S10."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pole_zero_tomography import fit_two_pole_rational


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_prl"
FIG = ROOT / "fig_prl"
P1 = 0.18 - 0.020j
P2 = 0.42 - 0.030j
DARK = 0.30 - 0.010j
A = 1.5
SEED = 20260908


def response(z: np.ndarray) -> np.ndarray:
    return (1.0 + A*z) * (z-DARK) / ((z-P1)*(z-P2))


def nearest(roots: tuple[complex, ...], target: complex) -> complex:
    return min(roots, key=lambda root: abs(root-target))


def pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    z = np.linspace(0.05, 0.55, 401) + 0.004j
    exact = response(z)
    fit = fit_two_pole_rational(z, exact, background_order=0)
    principal = fit.principal_part_zero
    corrected = nearest(fit.full_response_zeros, DARK)
    remote = nearest(fit.full_response_zeros, -1.0/A)

    background = 0.12 + 0.06*z
    bias_background = 0.0015 - 0.0020*z
    noise_scale = 2.0e-4 * float(np.sqrt(np.mean(np.abs(exact)**2)))
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    exact_roots: list[complex] = []
    biased_roots: list[complex] = []
    failures = 0
    for sample in range(240):
        noise = noise_scale/np.sqrt(2.0) * (
            rng.standard_normal(z.size) + 1j*rng.standard_normal(z.size)
        )
        observed = exact + background + noise
        try:
            fit_exact_cal = fit_two_pole_rational(
                z, observed-background, background_order=1,
            )
            fit_biased_cal = fit_two_pole_rational(
                z, observed-(background+bias_background), background_order=1,
            )
            root_exact = nearest(fit_exact_cal.full_response_zeros, DARK)
            root_biased = nearest(fit_biased_cal.full_response_zeros, DARK)
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
        exact_roots.append(root_exact)
        biased_roots.append(root_biased)
        rows.append({
            "sample": sample,
            "exact_cal_real": root_exact.real,
            "exact_cal_imag": root_exact.imag,
            "biased_cal_real": root_biased.real,
            "biased_cal_imag": root_biased.imag,
            "noise_only_abs_error": abs(root_exact-DARK),
            "calibration_shift_abs": abs(root_biased-root_exact),
        })
    exact_arr = np.asarray(exact_roots)
    biased_arr = np.asarray(biased_roots)
    exact_mean = complex(np.mean(exact_arr))
    biased_mean = complex(np.mean(biased_arr))
    uncertainty = float(np.sqrt(np.mean(np.abs(exact_arr-exact_mean)**2)))
    calibration_bias = complex(biased_mean-exact_mean)

    csv_path = DATA / "zero_assignment_revision_samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    style = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.3,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.72), constrained_layout=True)
        ax = axes[0]
        x = z.real
        ax.plot(x, np.abs(exact), color="#243B6B", lw=1.5, label=r"$|\chi_{\rm cal}|$")
        ax.axvline(DARK.real, color="#1B9E77", lw=1.4, label="full numerator zero")
        ax.axvline(principal.real, color="#D95F02", lw=1.3, ls="--", label="principal-part zero")
        ax.set(xlabel=r"$\mathrm{Re}\,z$", ylabel=r"$|\chi_{\rm cal}(z)|$",
               title="Analytic-prefactor counterexample")
        ax.legend(frameon=False, loc="upper right")
        ax.text(0.02, 0.05, r"$\chi_{\rm cal}=(1+az)(z-z_d)/D$",
                transform=ax.transAxes, color="0.25")

        ax = axes[1]
        ax.scatter(exact_arr.real, exact_arr.imag, s=8, alpha=0.28,
                   color="#377EB8", linewidths=0, label="noise, exact calibration")
        ax.scatter(biased_arr.real, biased_arr.imag, s=8, alpha=0.24,
                   color="#E41A1C", linewidths=0, label="noise + calibration bias")
        ax.plot(DARK.real, DARK.imag, marker="*", ms=9, color="black", label=r"true $z_d$")
        ax.set(xlabel=r"$\mathrm{Re}\,z_0$", ylabel=r"$\mathrm{Im}\,z_0$",
               title="Synthetic calibration stress test")
        ax.ticklabel_format(useOffset=False)
        ax.legend(frameon=False, loc="best")
        ax.text(0.02, 0.04,
                f"noise spread = {uncertainty:.2e}\ncalibration shift = {abs(calibration_bias):.2e}",
                transform=ax.transAxes, color="0.25")
        for label, ax in zip(("a", "b"), axes):
            ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold")
        fig.savefig(FIG / "figS10_zero_assignment_revision.pdf", bbox_inches="tight")
        fig.savefig(FIG / "figS10_zero_assignment_revision.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "scope": "synthetic_only",
        "seed": SEED,
        "model": "chi_cal=(1+a*z)*(z-z_dark)/((z-p1)*(z-p2))",
        "parameters": {"p1": pair(P1), "p2": pair(P2), "z_dark": pair(DARK), "a": A},
        "exact_fit": {
            "relative_rms": fit.relative_rms,
            "condition_number": fit.condition_number,
            "principal_part_zero": pair(principal),
            "full_zero_near_z_dark": pair(corrected),
            "remote_prefactor_zero": pair(remote),
            "principal_bias_abs": abs(principal-DARK),
            "corrected_error_abs": abs(corrected-DARK),
            "candidate_diagnostics": [
                {
                    "root": pair(candidate.root),
                    "inside_window": candidate.inside_window,
                    "cancelled": candidate.cancelled,
                    "slope_scaled": candidate.slope_scaled,
                    "root_condition_number": candidate.root_condition_number,
                    "eligible": candidate.eligible,
                    "reason": candidate.reason,
                }
                for candidate in fit.zero_candidates
            ],
        },
        "calibrated_background_noise_stress": {
            "attempted_samples": 240,
            "successful_samples": len(rows),
            "failed_samples": failures,
            "relative_complex_noise_rms": 2.0e-4,
            "noise_uncertainty_complex_rms": uncertainty,
            "exact_calibration_mean": pair(exact_mean),
            "calibration_bias_complex": pair(calibration_bias),
            "calibration_bias_abs": abs(calibration_bias),
            "calibration_error": "background estimate offset by 0.0015-0.0020*z",
        },
        "interpretation": {
            "response_zero_alias": "principal_part_zero",
            "full_response_zeros": "zeros of the scalar supplied to the fit",
            "calibration_rule": "calibrate the measured probe before fitting when a different scalar response is the target",
            "eligibility_limit": "a calibrated-window numerical candidate, not proof of causality or a physical compressed-sector mode",
        },
        "source_sha256": {
            "code/pole_zero_tomography.py": sha256(ROOT / "code/pole_zero_tomography.py"),
            "code/test_zero_assignment_revision.py": sha256(ROOT / "code/test_zero_assignment_revision.py"),
            "code/generate_zero_assignment_revision.py": sha256(Path(__file__)),
        },
    }
    (DATA / "zero_assignment_revision.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary["exact_fit"], indent=2))
    print(json.dumps(summary["calibrated_background_noise_stress"], indent=2))


if __name__ == "__main__":
    main()
