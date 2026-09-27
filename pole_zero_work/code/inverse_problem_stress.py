"""Bounded synthetic stress tests for calibrated complex pole-zero inference.

The estimator in this module never receives a true pole or zero.  It uses a
fixed central analysis window and nearest-root continuation across predetermined
nested windows.  Truth enters only after estimation to score error and coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
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
SEED = 20260908
CENTER = 0.30
WINDOWS = (0.24, 0.18, 0.13)
P_BRIGHT = 0.19 - 0.018j
P_DARK = 0.31 - 0.014j


@dataclass(frozen=True)
class WindowEstimate:
    root: complex | None
    success: bool
    reason: str
    residual: float
    condition_number: float
    window_spread: float
    roots_by_window: tuple[complex, ...]


def two_mode_response(z: np.ndarray, coupling: float, *, prefactor: complex = 1.0) -> np.ndarray:
    """Canonical bright response of a damped two-mode block."""
    return prefactor * (z-P_DARK) / ((z-P_BRIGHT)*(z-P_DARK)-coupling**2)


def calibrated_target(z: np.ndarray, coupling: float = 0.022) -> np.ndarray:
    """Full target, including its intrinsic analytic response remainder."""
    singular = (1.0 + 0.9*z) * two_mode_response(z, coupling)
    intrinsic_remainder = (0.014 - 0.006j) + (0.010 + 0.004j)*(z-CENTER)
    return singular + intrinsic_remainder


def three_pole_full_zeros(
    coupling: float,
    omitted_weight: float,
    remote_pole: complex = 0.72 - 0.055j,
) -> tuple[complex, ...]:
    """Exact full-numerator zeros after adding a physical remote pole.

    This score-only construction is not used by the estimator.  At zero
    remote weight the third denominator factor is absent, so no cancelled
    remote root is returned.
    """
    if omitted_weight == 0.0:
        return (P_DARK,)
    denominator_two = np.polysub(
        np.polymul([1.0, -P_BRIGHT], [1.0, -P_DARK]),
        [coupling**2],
    )
    numerator = np.polyadd(
        np.polymul([1.0, -P_DARK], [1.0, -remote_pole]),
        omitted_weight*denominator_two,
    )
    return tuple(complex(root) for root in np.roots(numerator))


def _eligible_roots(fit) -> tuple[complex, ...]:
    return tuple(candidate.root for candidate in fit.zero_candidates if candidate.eligible)


def select_window_root(
    z: np.ndarray,
    response: np.ndarray,
    *,
    center: float = CENTER,
    halfwidths: tuple[float, ...] = WINDOWS,
) -> WindowEstimate:
    """Select a root using only fixed windows and nearest-root continuation.

    The first root is the eligible full-numerator root nearest the fixed window
    center.  Each narrower window continues from the preceding selected root.
    No true root, pole, or synthetic label is accepted by this estimator.
    """
    z = np.asarray(z, complex)
    y = np.asarray(response, complex)
    selected: list[complex] = []
    residuals: list[float] = []
    conditions: list[float] = []
    previous: complex | None = None
    for halfwidth in halfwidths:
        mask = np.abs(z.real-center) <= halfwidth
        if np.count_nonzero(mask) < 20:
            return WindowEstimate(None, False, "insufficient_window_samples", np.inf, np.inf, np.inf, tuple(selected))
        try:
            fit = fit_two_pole_rational(z[mask], y[mask], background_order=1)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return WindowEstimate(None, False, "fit_failure", np.inf, np.inf, np.inf, tuple(selected))
        residuals.append(fit.relative_rms)
        conditions.append(fit.condition_number)
        roots = _eligible_roots(fit)
        if not roots:
            return WindowEstimate(None, False, "no_eligible_window_root", max(residuals), max(conditions), np.inf, tuple(selected))
        anchor = complex(center) if previous is None else previous
        chosen = min(roots, key=lambda root: abs(root-anchor))
        if previous is not None and abs(chosen-previous) > 0.08:
            return WindowEstimate(None, False, "continuation_jump", max(residuals), max(conditions), np.inf, tuple(selected))
        selected.append(chosen)
        previous = chosen
    spread = float(max(abs(root-selected[-1]) for root in selected))
    success = bool(max(residuals) <= 0.02 and max(conditions) <= 1.0e12)
    reason = "ok" if success else "fit_quality_failure"
    return WindowEstimate(selected[-1] if success else None, success, reason,
                          max(residuals), max(conditions), spread, tuple(selected))


def target_root_from_exact_response(z: np.ndarray, response: np.ndarray) -> complex:
    """Score-only exact target root; never called by the estimator."""
    fit = fit_two_pole_rational(z, response, background_order=1)
    roots = _eligible_roots(fit)
    return min(roots, key=lambda root: abs(root-CENTER))


def loss_only_ambiguity(z: np.ndarray) -> dict[str, object]:
    """Construct two targets with identical loss on a real-frequency window."""
    base = two_mode_response(z, 0.022)
    target_a = base + 0.10
    target_b = base - 0.13 + 0.22*(z-CENTER)
    fit_a = fit_two_pole_rational(z, target_a, background_order=1)
    fit_b = fit_two_pole_rational(z, target_b, background_order=1)
    root_a = min(fit_a.full_response_zeros, key=lambda root: abs(root-CENTER))
    root_b = min(fit_b.full_response_zeros, key=lambda root: abs(root-CENTER))
    return {
        "target_a": target_a,
        "target_b": target_b,
        "root_a": root_a,
        "root_b": root_b,
        "max_loss_difference": float(np.max(np.abs(target_a.imag-target_b.imag))),
        "root_separation": float(abs(root_a-root_b)),
    }


def nonidentifiable_decompositions(z: np.ndarray) -> dict[str, object]:
    """Two exact target/nuisance decompositions of the same complex spectrum."""
    target_a = calibrated_target(z)
    nuisance_a = (0.003 + 0.002j) + (-0.004 + 0.001j)*(z-CENTER)
    transfer = (0.009 - 0.003j) + (0.006 + 0.002j)*(z-CENTER)
    target_b = target_a + transfer
    nuisance_b = nuisance_a - transfer
    observed_a = target_a + nuisance_a
    observed_b = target_b + nuisance_b
    root_a = target_root_from_exact_response(z, target_a)
    root_b = target_root_from_exact_response(z, target_b)
    return {
        "max_observed_difference": float(np.max(np.abs(observed_a-observed_b))),
        "target_root_a": root_a,
        "target_root_b": root_b,
        "target_root_separation": float(abs(root_a-root_b)),
    }


def _complex_noise(rng: np.random.Generator, size: int, rms: float) -> np.ndarray:
    return rms/np.sqrt(2.0) * (rng.standard_normal(size) + 1j*rng.standard_normal(size))


def run_study(seed: int = SEED) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(seed)
    z = np.linspace(0.04, 0.56, 321).astype(complex)
    target = calibrated_target(z)
    true_root = target_root_from_exact_response(z, target)
    target_rms = float(np.sqrt(np.mean(np.abs(target)**2)))
    noise_rms = 2.5e-4*target_rms
    rows: list[dict[str, object]] = []

    def record(kind: str, level: float, trial: int, estimate: WindowEstimate,
               truth: complex, **extra: object) -> None:
        confident = bool(estimate.success and estimate.residual < 1.5e-3 and estimate.window_spread < 0.012)
        error = float(abs(estimate.root-truth)) if estimate.root is not None else np.nan
        envelope_available = bool(estimate.success and len(estimate.roots_by_window) >= 2)
        envelope_covered = bool(
            envelope_available and np.isfinite(error) and error <= estimate.window_spread
        )
        row: dict[str, object] = {
            "kind": kind, "level": level, "trial": trial,
            "success": int(estimate.success), "reason": estimate.reason,
            "root_real": estimate.root.real if estimate.root is not None else np.nan,
            "root_imag": estimate.root.imag if estimate.root is not None else np.nan,
            "error_abs": error, "residual": estimate.residual,
            "condition_number": estimate.condition_number,
            "window_spread": estimate.window_spread,
            "confident": int(confident),
            "confidence_with_envelope": int(confident and envelope_available),
            "false_confident": int(confident and envelope_available and not envelope_covered),
            "window_envelope_covered": int(envelope_covered),
            "window_envelope_available": int(envelope_available),
        }
        row.update(extra)
        rows.append(row)

    # Residual affine instrumental background after calibration.
    background_levels = (0.0, 2.0e-4, 8.0e-4, 2.0e-3, 1.0e-2, 5.0e-2)
    for level in background_levels:
        for trial in range(60):
            b0 = level*target_rms*_complex_noise(rng, 1, 1.0)[0]
            b1 = level*target_rms/0.25*_complex_noise(rng, 1, 1.0)[0]
            observed = target + b0 + b1*(z-CENTER) + _complex_noise(rng, z.size, noise_rms)
            record("instrument_background", level, trial,
                   select_window_root(z, observed), true_root)

    # Residual phase calibration. Constant phase is included but should not
    # move exact zeros; a slope is the relevant frequency-dependent nuisance.
    phase_levels = (0.0, 0.004, 0.012, 0.035, 0.15, 0.5)
    for level in phase_levels:
        for trial in range(60):
            phi0 = rng.normal(scale=0.02)
            phi1 = rng.normal(scale=level)/0.25
            phased = np.exp(1j*(phi0+phi1*(z.real-CENTER))) * target
            observed = phased + _complex_noise(rng, z.size, noise_rms)
            record("phase_slope", level, trial, select_window_root(z, observed),
                   true_root, phase_offset=phi0, phase_slope=phi1)

    # Fixed two-pole estimator under an omitted passive remote pole.  Windows
    # are predetermined and each row uses the same root-selection protocol.
    for omitted_weight in (0.0, 0.06):
        exact_zeros = three_pole_full_zeros(0.022, omitted_weight)
        exact_full_root = min(exact_zeros, key=lambda root: abs(root-CENTER))
        for halfwidth in (0.24, 0.18, 0.13):
            for trial in range(40):
                response = two_mode_response(z, 0.022)
                if omitted_weight:
                    response = response + omitted_weight/(z-(0.72-0.055j))
                scale = float(np.sqrt(np.mean(np.abs(response)**2)))
                observed = response + _complex_noise(rng, z.size, 2.5e-4*scale)
                truth = exact_full_root
                estimate = select_window_root(z, observed, halfwidths=(halfwidth,))
                record("model_order", omitted_weight, trial, estimate, truth,
                       halfwidth=halfwidth,
                       exact_full_zero_real=truth.real,
                       exact_full_zero_imag=truth.imag,
                       bare_reference_displacement=abs(truth-P_DARK))

    # Weak-field visibility.  Pole continuation uses decreasing field and the
    # high-field higher-real-part pole as a predetermined initialization.
    fields = np.array([0.036, 0.024, 0.016, 0.010, 0.0065, 0.004])
    weak_attempts: dict[float, int] = {float(field): 0 for field in fields}
    weak_visible: dict[float, int] = {float(field): 0 for field in fields}
    weak_zero_success: dict[float, int] = {float(field): 0 for field in fields}
    for trial in range(70):
        previous_pole: complex | None = None
        for field in fields:
            coupling = 0.72*field
            clean = two_mode_response(z, coupling)
            sigma = 4.0e-4*float(np.sqrt(np.mean(np.abs(clean)**2)))
            observed = clean + _complex_noise(rng, z.size, sigma)
            weak_attempts[float(field)] += 1
            try:
                fit = fit_two_pole_rational(z, observed, background_order=1)
                poles = fit.poles
                chosen_pole = max(poles, key=lambda pole: pole.real) if previous_pole is None else min(poles, key=lambda pole: abs(pole-previous_pole))
                previous_pole = chosen_pole
                residue = fit.residues[poles.index(chosen_pole)]
                distance = max(float(np.min(np.abs(z-chosen_pole))), 1e-12)
                fitted_component_snr = float(abs(residue)/(distance*sigma))
                visible = bool(fitted_component_snr >= 5.0 and 0.25 <= chosen_pole.real <= 0.38)
                estimate = select_window_root(z, observed)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                chosen_pole, residue, fitted_component_snr, visible = np.nan+1j*np.nan, np.nan+1j*np.nan, 0.0, False
                estimate = WindowEstimate(None, False, "fit_failure", np.inf, np.inf, np.inf, ())
            weak_visible[float(field)] += int(visible)
            weak_zero_success[float(field)] += int(estimate.success)
            record("weak_field", float(field), trial, estimate, P_DARK,
                   coupling=coupling, dark_pole_real=np.real(chosen_pole),
                   dark_pole_imag=np.imag(chosen_pole), residue_abs=abs(residue),
                   component_snr=fitted_component_snr, visible=int(visible))

    ambiguity = loss_only_ambiguity(z)
    decomposition = nonidentifiable_decompositions(z)
    global_phase_control = select_window_root(z, np.exp(0.37j)*target)
    summary: dict[str, object] = {
        "scope": "seeded_synthetic_inverse_problem_only",
        "seed": seed,
        "estimator_protocol": {
            "center": CENTER, "nested_halfwidths": list(WINDOWS),
            "initial_rule": "eligible full-numerator root nearest fixed window center",
            "continuation_rule": "nearest root to preceding wider-window estimate",
            "truth_used_by_estimator": False,
            "success_screen": "max residual <=0.02, max condition <=1e12, continuation jump <=0.08",
            "heuristic_confidence": "success, residual <1.5e-3, and window spread <0.012",
            "false_confident_definition": "heuristically confident but score-only truth lies outside the multiwindow stability envelope",
            "false_confidence_denominator": "heuristically confident estimates having a multiwindow envelope",
            "window_envelope": "maximum root displacement across nested windows from final root; diagnostic, not a confidence interval",
        },
        "stress_design": {
            "frequency_samples": 321,
            "frequency_range": [0.04, 0.56],
            "relative_complex_noise_rms": 2.5e-4,
            "calibration_distribution": (
                "Independent Gaussian draws. Background levels are complex RMS "
                "scales, phase levels are real standard deviations across 0.25 "
                "frequency units; neither is a worst-case bound."),
            "instrument_background_levels_fraction_of_target_rms": list(background_levels),
            "phase_slope_levels_radians_across_half_window": list(phase_levels),
            "phase_offset_standard_deviation_radians": 0.02,
            "trials_per_background_or_phase_level": 60,
            "trials_per_model_window": 40,
            "trials_per_weak_field": 70,
        },
        "global_phase_exact_control": {
            "phase_radians": 0.37,
            "success": global_phase_control.success,
            "fit_relative_residual": global_phase_control.residual,
            "root_displacement": abs(global_phase_control.root-true_root) if global_phase_control.root is not None else None,
            "interpretation": "constant nonzero complex scaling preserves full and principal zeros",
        },
        "target": {
            "intrinsic_analytic_remainder_included": True,
            "true_full_target_root": [true_root.real, true_root.imag],
            "instrument_affine_term": "external nuisance requiring calibration",
            "identifiability_limit": "complex spectra cannot freely separate intrinsic and instrumental analytic terms",
        },
        "nonidentifiable_decomposition": {
            key: ([value.real, value.imag] if isinstance(value, complex) else value)
            for key, value in decomposition.items()
        },
        "loss_only_ambiguity": {
            "max_loss_difference": ambiguity["max_loss_difference"],
            "root_separation": ambiguity["root_separation"],
            "root_a": [ambiguity["root_a"].real, ambiguity["root_a"].imag],
            "root_b": [ambiguity["root_b"].real, ambiguity["root_b"].imag],
        },
        "weak_field": {
            "attempts": weak_attempts,
            "visible_fraction": {str(field): weak_visible[field]/weak_attempts[field] for field in weak_attempts},
            "zero_success_fraction": {str(field): weak_zero_success[field]/weak_attempts[field] for field in weak_attempts},
        },
    }
    return rows, summary


def _group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["kind"], row["level"], row.get("halfwidth", ""))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    for (kind, level, halfwidth), group in sorted(groups.items(), key=lambda item: str(item[0])):
        errors = np.array([float(row["error_abs"]) for row in group if np.isfinite(float(row["error_abs"]))])
        residuals = np.array([float(row["residual"]) for row in group if np.isfinite(float(row["residual"]))])
        spreads = np.array([
            float(row["window_spread"]) for row in group
            if int(row["window_envelope_available"]) and np.isfinite(float(row["window_spread"]))
        ])
        successes = sum(int(row["success"]) for row in group)
        confident = sum(int(row["confident"]) for row in group)
        confidence_with_envelope = sum(int(row["confidence_with_envelope"]) for row in group)
        false_confident = sum(int(row["false_confident"]) for row in group)
        covered = sum(int(row["window_envelope_covered"]) for row in group)
        coverage_attempts = sum(int(row["window_envelope_available"]) for row in group)
        bare_shifts = np.array([
            float(row["bare_reference_displacement"]) for row in group
            if row.get("bare_reference_displacement", "") != ""
        ])
        out.append({
            "kind": kind, "level": level, "halfwidth": halfwidth,
            "attempts": len(group), "successes": successes,
            "failure_fraction": 1-successes/len(group),
            "median_error": float(np.median(errors)) if errors.size else np.nan,
            "p90_error": float(np.quantile(errors, 0.9)) if errors.size else np.nan,
            "median_residual": float(np.median(residuals)) if residuals.size else np.nan,
            "median_window_spread": float(np.median(spreads)) if spreads.size else None,
            "bare_reference_displacement": float(np.median(bare_shifts)) if bare_shifts.size else None,
            "confident": confident,
            "confidence_with_envelope": confidence_with_envelope,
            "false_confident": false_confident,
            "false_confidence_fraction": (
                false_confident/confidence_with_envelope if confidence_with_envelope else None
            ),
            "window_envelope_attempts": coverage_attempts,
            "window_envelope_coverage": (
                covered/coverage_attempts if coverage_attempts else None
            ),
        })
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    groups = _group_summary(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), constrained_layout=True)
    colors = {"instrument_background": "#377EB8", "phase_slope": "#E41A1C"}
    ax = axes[0, 0]
    for kind in ("instrument_background", "phase_slope"):
        selected = [row for row in groups if row["kind"] == kind]
        max_level = max(float(row["level"]) for row in selected)
        xvalues = [float(row["level"])/max_level for row in selected]
        label = ("baseline RMS scale (up to 0.05)" if kind == "instrument_background"
                 else "phase SD scale (up to 0.5 rad)")
        ax.plot(xvalues,
                [float(row["p90_error"]) for row in selected], marker="o",
                color=colors[kind], label=label)
    ax.set(xlabel="calibration level / series maximum", ylabel="90th-percentile |zero error|",
           title="Calibration remainder")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for weight, color, label in ((0.0, "#1B9E77", "two-pole in model"),
                                 (0.06, "#984EA3", "remote pole omitted")):
        selected = sorted((row for row in groups if row["kind"] == "model_order" and float(row["level"]) == weight),
                          key=lambda row: float(row["halfwidth"]))
        ax.plot([float(row["halfwidth"]) for row in selected],
                [float(row["median_residual"]) for row in selected], marker="s",
                color=color, label=label)
    ax.set(xlabel="fit half-window", ylabel="median relative residual",
           title="Window and model order")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    weak = summary["weak_field"]
    fields = sorted(float(value) for value in weak["visible_fraction"])
    ax.plot(fields, [weak["visible_fraction"][str(value)] for value in fields],
            marker="o", color="#D95F02", label="dark-pole visible")
    ax.plot(fields, [weak["zero_success_fraction"][str(value)] for value in fields],
            marker="^", color="#243B6B", label="zero fit returned")
    ax.set(xlabel=r"field proxy $B$ ($g=0.72B$)", ylabel="fraction", ylim=(-0.03, 1.03),
           title="Weak-field observability")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    z = np.linspace(0.04, 0.56, 321).astype(complex)
    ambiguity = loss_only_ambiguity(z)
    ax.plot(z.real, -np.imag(ambiguity["target_a"]), color="#4DAF4A", lw=1.6,
            label="target A loss")
    ax.plot(z.real, -np.imag(ambiguity["target_b"]), color="#FF7F00", lw=1.1, ls="--",
            label="target B loss")
    ax.axvline(ambiguity["root_a"].real, color="#4DAF4A", lw=1.0)
    ax.axvline(ambiguity["root_b"].real, color="#FF7F00", lw=1.0, ls="--")
    ax.set(xlabel=r"real frequency $z$", ylabel=r"$-\mathrm{Im}\,\chi$",
           title="Loss-only nonidentifiability")
    ax.legend(frameon=False)
    ax.text(0.03, 0.08, f"same loss; zero separation = {ambiguity['root_separation']:.2e}",
            transform=ax.transAxes, color="0.25")
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold")
    fig.savefig(FIG / "figS12_inverse_problem_stress.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(FIG / "figS12_inverse_problem_stress.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    rows, summary = run_study()
    groups = _group_summary(rows)
    _write_csv(DATA / "inverse_problem_stress_trials.csv", rows)
    _write_csv(DATA / "inverse_problem_stress_summary.csv", groups)
    make_figure(rows, summary)
    summary["counts"] = {
        "trial_rows": len(rows), "summary_rows": len(groups),
        "all_attempts_reported": True,
        "successful_estimates": sum(int(row["success"]) for row in rows),
        "failed_estimates": sum(1-int(row["success"]) for row in rows),
        "confident_estimates": sum(int(row["confident"]) for row in rows),
        "confidence_with_envelope_estimates": sum(int(row["confidence_with_envelope"]) for row in rows),
        "false_confident_estimates": sum(int(row["false_confident"]) for row in rows),
        "window_envelope_coverage_fraction": (
            sum(int(row["window_envelope_covered"]) for row in rows)
            / max(sum(int(row["window_envelope_available"]) for row in rows), 1)
        ),
        "window_envelope_attempts": sum(int(row["window_envelope_available"]) for row in rows),
    }
    summary["group_summary"] = groups
    summary["source_sha256"] = {
        "code/inverse_problem_stress.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "code/test_inverse_problem_stress.py": hashlib.sha256((ROOT / "code/test_inverse_problem_stress.py").read_bytes()).hexdigest(),
        "code/pole_zero_tomography.py": hashlib.sha256((ROOT / "code/pole_zero_tomography.py").read_bytes()).hexdigest(),
    }
    (DATA / "inverse_problem_stress.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.2,
                                "axes.titlesize": 9.0, "axes.labelsize": 8.3,
                                "legend.fontsize": 7.2, "xtick.labelsize": 7.4,
                                "ytick.labelsize": 7.4})
    main()
