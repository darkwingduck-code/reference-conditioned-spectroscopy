"""Render one scientific figure and audit the frozen production outputs."""
import argparse
import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment import ROOT, REPO, Q0, verify_witness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    production = ROOT / "run_02"
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    data = pd.read_csv(production / "trials.csv")
    original = pd.read_csv(ROOT / "run_01/trials.csv")
    shared = [column for column in original.columns
              if not column.endswith("estimated_parameters_in_outer_set")]
    pd.testing.assert_frame_equal(data[shared], original[shared], check_exact=True)
    witnesses = json.loads((production / "infeasibility_witnesses.json").read_text())
    assert all(verify_witness(witness) for witness in witnesses)
    valid = data[data.model_assumptions_hold]
    assert (valid.free_status == "compatible").all()
    assert valid.free_contains_omega.all() and valid.free_contains_gamma.all()
    representative = data[(data.panel == "shifted") & (abs(data.q-Q0) < 1e-12)
                          & (data.gamma_dark == 0.003) & (data.noise_radius == 1e-4)]
    methods = ("positive_pp_error", "positive_full_error", "paired_pp_error",
               "paired_full_error", "free_root_error")
    medians = representative.groupby("B")[list(methods)].median()/0.003
    medians.to_csv(out / "matched_baseline_medians.csv")
    damping = data[(data.panel == "viscous") & (data.control == "valid_star")
                   & (data.B == 0.2) & (data.noise_radius == 1e-4)]
    damping_rows = []
    for gamma, group in damping.groupby("gamma_dark"):
        damping_rows.append(dict(gamma=gamma, median=float(group.free_inferred_gamma.median()),
                                 lower=float(group.free_gamma_lower.min()),
                                 upper=float(group.free_gamma_upper.max())))
    pd.DataFrame(damping_rows).to_csv(out / "damping_interval_envelopes.csv", index=False)
    invalid = data[~data.model_assumptions_hold]
    counts = invalid.groupby(["noise_radius", "free_status"]).size().unstack(fill_value=0)
    counts.to_csv(out / "invalid_control_status_counts.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "legend.fontsize": 6.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), layout="constrained")
    colors = ("#A23B32", "#BC8D3A", "#297F55", "#2266A5", "#252525")
    labels = ("positive pp", "positive full", "paired pp", "paired full", "reference")
    for column, label, color in zip(methods, labels, colors):
        axes[0].plot(medians.index, medians[column], "o-", label=label,
                     color=color, linewidth=1.15, markersize=3)
    axes[0].set(xlabel="Field control B", ylabel=r"Complex-zero error / $\eta$",
                yscale="log", title="(a) Same signal samples")
    axes[0].set_xticks([0.05, 0.10, 0.20])
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=5, frameon=False, handlelength=1.5)
    axes[0].grid(axis="y", alpha=0.16)
    gamma = np.array([row["gamma"] for row in damping_rows])*1e3
    estimate = np.array([row["median"] for row in damping_rows])*1e3
    lo = np.array([row["lower"] for row in damping_rows])*1e3
    hi = np.array([row["upper"] for row in damping_rows])*1e3
    axes[1].plot([1, 6.5], [1, 6.5], color="0.7", linewidth=0.9, label="true damping")
    axes[1].axhline(3, color="#A23B32", linestyle="--", linewidth=1,
                    label="fixed assumption")
    axes[1].errorbar(gamma, estimate, yerr=np.vstack([estimate-lo, hi-estimate]),
                     fmt="o", color="#2266A5", markersize=4, capsize=3,
                     label="free damping")
    axes[1].set(xlabel=r"True $\gamma_d\;(10^{-3})$",
                ylabel=r"Inferred $\gamma_d\;(10^{-3})$", title="(b) Independent damping",
                xlim=(1, 6.5), ylim=(1, 6.5))
    axes[1].set_xticks([1.5, 3, 6])
    axes[1].legend(loc="upper left", frameon=False, handlelength=1.3)
    positions = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for status, color in (("inconsistent", "#2266A5"),
                           ("compatible", "#DBB56A"),
                           ("inconclusive", "#898989")):
        values = 100*counts.get(status, pd.Series(0, index=counts.index)).to_numpy()/counts.sum(axis=1).to_numpy()
        axes[2].bar(positions, values, bottom=bottom, color=color, width=0.65, label=status)
        bottom += values
    for position, total in zip(positions, counts.sum(axis=1)):
        axes[2].text(position, 102, f"n={total}", ha="center", fontsize=7)
    axes[2].set(xlabel="Signal / calibration radius", ylabel="Violated controls (%)",
                title="(c) Model compatibility", ylim=(0, 130))
    axes[2].set_xticks(positions, ["0", r"$10^{-4}$", r"$10^{-3}$"])
    axes[2].set_yticks([0, 50, 100])
    axes[2].legend(loc="upper center", frameon=False, ncol=1,
                   bbox_to_anchor=(0.5, 1.02), labelspacing=0.15, handlelength=1.2)
    fig.savefig(out / "reference_validation_20260927.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
    fig.savefig(out / "reference_validation_20260927.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    if args.figure_only:
        print("Regenerated figure and plotted tables from unchanged run_02; numerical experiments and tests were not rerun.")
        return

    checks = {}
    for name, relative, pattern in (
            ("new_tests", "research/reference_validation_20260927", "test_experiment.py"),
            ("legacy_tests", "research/physical_discrimination_20260912", "test_discrimination.py")):
        process = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", relative,
                                  "-p", pattern, "-v"], cwd=REPO, capture_output=True, text=True)
        (out / f"{name}.log").write_text(process.stdout+process.stderr, encoding="utf-8")
        checks[name] = process.returncode
    assert all(code == 0 for code in checks.values()), checks
    maxima = {"max_valid_gamma_point_error": float(abs(valid.free_inferred_gamma-valid.gamma_dark).max()),
              "max_valid_omega_point_error": float(abs(valid.free_inferred_omega-valid.target_omega).max()),
              "max_valid_gamma_interval_width": float((valid.free_gamma_upper-valid.free_gamma_lower).max()),
              "max_valid_omega_interval_width": float((valid.free_omega_upper-valid.free_omega_lower).max())}
    report = dict(status="PASS", case_count=len(data), valid_cases=len(valid), invalid_cases=len(invalid),
                  valid_false_rejections=0, valid_numerical_interval_misses=0,
                  invalid_status_counts={str(k): int(v) for k, v in invalid.free_status.value_counts().items()},
                  exact_rational_witnesses_verified=len(witnesses),
                  run_01_run_02_common_scientific_columns_identical=True,
                  excluded_comparison_columns=["fixed_estimated_parameters_in_outer_set",
                                               "free_estimated_parameters_in_outer_set"],
                  repeat_change="Corrected joint-vs-marginal membership labels, saved source snapshots, added an overdamped check; all scientific estimates, intervals and status decisions unchanged",
                  tests=checks, **maxima,
                  figure="reference_validation_20260927.pdf",
                  precision_scope="numerical marginal intervals, exact stored-float separation witnesses; no general model certificate")
    (out / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
