"""Summarize the frozen run without refitting or discarding numerical failures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODELS = ("unchanged", "drift_aware", "extended")
CONTROLS = ("zero_drift", "allowed_drift", "extra_link")
LABELS = ("Unchanged reference", "Bright drift", "Bright drift + extra link")
COLORS = ("#bb4b32", "#276b99", "#679343")
THRESHOLD = 3.841
OMEGA_TRUE = 1.3*0.7458353261772512


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def number(row, key):
    return float(row[key]) if row[key] else np.nan


def summarize(rows):
    answer = dict(cases=len(rows), failures=sum(r["success"] != "True" for r in rows),
                  boundary_fits=sum(bool(r["boundaries"]) for r in rows),
                  remote_boundary_fits=sum("remote_sq" in r["boundaries"] or "log_gr" in r["boundaries"] for r in rows),
                  holdout_above_descriptive_line=sum(r["holdout_above_reference"] == "True" for r in rows))
    metrics = {
        "abs_omega_error_over_gamma": [abs(number(r, "omega_error"))/number(r, "gamma_dark") for r in rows],
        "abs_relative_gamma_error": [abs(number(r, "relative_gamma_error")) for r in rows],
        "abs_relative_t_error": [abs(number(r, "relative_t_error")) for r in rows],
        "holdout_chi2": [number(r, "holdout_chi2_per_real_sample") for r in rows],
        "clean_holdout_scaled_mse": [number(r, "holdout_clean_scaled_mse") for r in rows],
        "normalized_jacobian_condition": [number(r, "normalized_jacobian_condition") for r in rows],
        "omega_local_error_ratio": [abs(number(r, "omega_error"))/number(r, "local_omega_se") for r in rows],
        "gamma_local_error_ratio": [abs(number(r, "gamma_error"))/number(r, "local_gamma_se") for r in rows],
    }
    for name, values in metrics.items():
        answer[name+"_median"] = float(np.nanmedian(values))
        answer[name+"_max"] = float(np.nanmax(values))
    answer["omega_error_above_two_local_se"] = sum(x > 2 for x in metrics["omega_local_error_ratio"])
    answer["gamma_error_above_two_local_se"] = sum(x > 2 for x in metrics["gamma_local_error_ratio"])
    return answer


def profile_summary(rows):
    rows = sorted(rows, key=lambda r: float(r["value"]))
    support = np.array([float(r["delta_chi2"]) <= THRESHOLD for r in rows])
    values = np.array([float(r["value"]) for r in rows])
    accepted = np.flatnonzero(support)
    result = {k: rows[0][k] for k in ("case_id", "B", "gamma_dark", "noise_scale", "control", "replica", "model", "parameter")}
    result.update(points=len(rows), failed_points=sum(r["success"] != "True" for r in rows),
                  accepted_points=len(accepted), accepted_failed_points=sum(r["success"] != "True" for r, yes in zip(rows, support) if yes),
                  unresolved_global_minimum=any(r["global_minimum_unresolved"] == "True" for r in rows),
                  minimum_delta=min(float(r["delta_chi2"]) for r in rows))
    if len(accepted):
        left, right = accepted[0], accepted[-1]
        low_bound, high_bound = ((.85, 1.10) if rows[0]["parameter"] == "omega" else (1e-4, .03))
        result.update(accepted_sample_min=values[left], accepted_sample_max=values[right],
                      lower_rejected_neighbor=values[left-1] if left else "",
                      upper_rejected_neighbor=values[right+1] if right+1 < len(rows) else "",
                      lower_endpoint_accepted=bool(np.isclose(values[left], low_bound, atol=1e-12, rtol=0)),
                      upper_endpoint_accepted=bool(np.isclose(values[right], high_bound, atol=1e-12, rtol=0)),
                      sampled_support_components=int(np.sum(support & np.r_[True, ~support[:-1]])),
                      support_scope="sampled nominal support; crossing brackets are not certified intervals")
    return result


def make_figure(fits, profiles, output):
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.6))
    for ax, key, ylabel, title in zip(axes[0],
            ("omega_error", "relative_gamma_error", "holdout_chi2_per_real_sample"),
            (r"$|\widehat\Omega_d-\Omega_d|/\gamma_d$", r"$|\widehat\gamma_d/\gamma_d-1|$", r"Held-out $\chi^2/N_{\rm real}$"),
            ("(a) Frequency error", "(b) Damping error", "(c) Independent prediction")):
        for mi, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
            medians = []
            for ci, control in enumerate(CONTROLS):
                chosen = [r for r in fits if r["model"] == model and r["control"] == control]
                vals = [number(r, key) for r in chosen]
                if key != "holdout_chi2_per_real_sample":
                    vals = [abs(v)/(number(r, "gamma_dark") if key == "omega_error" else 1) for v, r in zip(vals, chosen)]
                ax.scatter(ci+(mi-1)*.18+np.linspace(-.045, .045, len(vals)), vals,
                           s=10, alpha=.33, color=color)
                failed = np.array([r["success"] != "True" for r in chosen])
                if np.any(failed):
                    positions = ci+(mi-1)*.18+np.linspace(-.045, .045, len(vals))
                    ax.scatter(positions[failed], np.asarray(vals)[failed], marker="x", s=19, lw=.7, color=color)
                medians.append(np.median(vals))
            ax.plot(np.arange(3)+(mi-1)*.18, medians, "o", ms=5, color=color, label=label)
        ax.set_xticks(range(3), ("No drift", "Bright drift", "+ Extra link"))
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
    axes[0, 2].axhline(1+2*np.sqrt(2/320), color="0.45", ls=":", lw=1)
    # Fixed design representatives: gamma=.003, noise=1e-3, replica=0, allowed drift.
    for ax, field, parameter, title in zip(axes[1], (.05, .20, .05), ("omega", "omega", "gamma"),
            ("(d) Weak link: frequency profile", "(e) Strong link: frequency profile", "(f) Weak link: damping profile")):
        for model, label, color in zip(MODELS, LABELS, COLORS):
            chosen = sorted([r for r in profiles if r["model"] == model and r["control"] == "allowed_drift"
                and float(r["B"]) == field and float(r["noise_scale"]) == 1e-3 and r["parameter"] == parameter],
                key=lambda r: float(r["value"]))
            x = [(float(r["value"])-OMEGA_TRUE)/.003 if parameter == "omega" else float(r["value"])/.003 for r in chosen]
            y = [float(r["delta_chi2"]) for r in chosen]
            ax.plot(x, y, ".-", ms=4, lw=.9, color=color, label=label)
            bad = [j for j, r in enumerate(chosen) if r["success"] != "True"]
            if bad:
                ax.scatter(np.array(x)[bad], np.array(y)[bad], marker="x", s=25, color=color)
        ax.axhline(THRESHOLD, color="0.3", ls=":", lw=1)
        ax.axvline(0 if parameter == "omega" else 1, color="0.5", ls="--", lw=.8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(-.25, 2e6)
        ax.set_xscale("symlog", linthresh=.005) if parameter == "omega" else ax.set_xscale("log")
        if parameter == "omega":
            ax.set_xticks([-10, -.1, 0, .1, 10], [r"$-10$", r"$-0.1$", "0", r"$0.1$", r"$10$"])
        ax.set_xlabel(r"$(\Omega_d^{\rm fixed}-\Omega_d^{\rm true})/\gamma_d^{\rm true}$" if parameter == "omega" else r"$\gamma_d^{\rm fixed}/\gamma_d^{\rm true}$")
        ax.set_ylabel(r"Profile $\Delta\chi^2$")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .012), frameon=False)
    fig.subplots_adjust(left=.075, right=.985, top=.95, bottom=.13, hspace=.44, wspace=.37)
    fig.savefig(output/"reference_control.pdf")
    fig.savefig(output/"reference_control.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run_01")
    args = parser.parse_args()
    run = (ROOT/args.run).resolve()
    if run.parent != ROOT:
        raise ValueError("Expected a run within this study")
    output = run/"analysis"
    output.mkdir(exist_ok=True)
    fits, profiles = read_csv(run/"fits.csv"), read_csv(run/"profiles.csv")
    grouped = []
    for control in CONTROLS:
        for model in MODELS:
            selected = [r for r in fits if r["control"] == control and r["model"] == model]
            if selected:
                grouped.append(dict(control=control, model=model, **summarize(selected)))
    write_csv(output/"summary_by_control.csv", grouped)
    detailed = []
    for control in CONTROLS:
        for model in MODELS:
            for field in (.05, .2):
                for noise in (1e-4, 1e-3):
                    selected = [r for r in fits if r["control"] == control and r["model"] == model and float(r["B"]) == field and float(r["noise_scale"]) == noise]
                    if selected:
                        detailed.append(dict(control=control, model=model, B=field, noise=noise, **summarize(selected)))
    write_csv(output/"summary_by_design.csv", detailed)
    keys = sorted({(r["case_id"], r["model"], r["parameter"]) for r in profiles}, key=lambda k: (int(k[0]), k[1], k[2]))
    support = [profile_summary([r for r in profiles if (r["case_id"], r["model"], r["parameter"]) == key]) for key in keys]
    write_csv(output/"profile_support.csv", support)
    make_figure(fits, profiles, output)
    inputs = [run/name for name in ("fits.csv", "profiles.csv", "validation.json")]
    report = dict(fits=len(fits), profile_curves=len(support), requested_profile_curves=42,
                  profile_curves_with_failed_optimizations=sum(s["failed_points"] > 0 for s in support),
                  profile_curves_with_unresolved_minimum=sum(s["unresolved_global_minimum"] for s in support),
                  profile_curves_with_accepted_bound=sum(s.get("lower_endpoint_accepted", False) or s.get("upper_endpoint_accepted", False) for s in support),
                  numerical_failure_rows_retained=True,
                  input_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/"analysis_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
