"""Equal-budget comparison, preserving both successes and failed fits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import read_csv, write_csv

ROOT = Path(__file__).resolve().parent


def metrics(rows):
    result = dict(cases=len(rows), failures=sum(r["success"] != "True" for r in rows),
                  boundaries=sum(bool(r["boundaries"]) for r in rows),
                  heldout_above_descriptive_line=sum(r["holdout_above_reference"] == "True" for r in rows))
    for name, values in {
        "abs_omega_error_over_gamma": [abs(float(r["omega_error"]))/float(r["gamma_dark"]) for r in rows],
        "abs_relative_gamma_error": [abs(float(r["relative_gamma_error"])) for r in rows],
        "abs_relative_t_error": [abs(float(r["relative_t_error"])) for r in rows],
        "holdout_chi2": [float(r["holdout_chi2_per_real_sample"]) for r in rows],
        "normalized_jacobian_condition": [float(r["normalized_jacobian_condition"]) for r in rows],
    }.items():
        result[name+"_median"] = float(np.median(values))
        result[name+"_max"] = float(max(values))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run_03_wide")
    args = parser.parse_args()
    run_path = (ROOT/args.run).resolve()
    if run_path.parent != ROOT:
        raise ValueError("Expected a direct-child run directory")
    output = run_path/"analysis"
    output.mkdir(exist_ok=True)
    rows = []
    for run, design in (("run_01", "low_window"), (args.run, "remote_inclusive")):
        rows += [dict(r, design=design) for r in read_csv(ROOT/run/"fits.csv")
                 if r["control"] == "extra_link" and r["model"] != "unchanged"]
    groups = [(design, model) for model in ("drift_aware", "extended") for design in ("low_window", "remote_inclusive")]
    summary = [dict(design=d, model=m, **metrics([r for r in rows if r["design"]==d and r["model"]==m])) for d, m in groups]
    write_csv(output/"comparison.csv", summary)
    detailed = []
    for design, model in groups:
        for field in (.05, .2):
            for noise in (1e-4, 1e-3):
                chosen = [r for r in rows if r["design"]==design and r["model"]==model and float(r["B"])==field and float(r["noise_scale"])==noise]
                detailed.append(dict(design=design, model=model, B=field, noise=noise, **metrics(chosen)))
    write_csv(output/"comparison_by_design.csv", detailed)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
    names = ("Star\nlow window", "Star\n+ remote", "Extra-link\nlow window", "Extra-link\n+ remote")
    colors = ("#8eb9d4", "#276b99", "#b5c995", "#679343")
    specifications = (
        ("omega_error", r"$|\widehat\Omega_d-\Omega_d|/\gamma_d$", "(a) Bare-frequency error"),
        ("relative_t_error", r"$|\widehat{\kappa^2}/\kappa^2-1|$", "(b) Squared-coupling error"),
        ("normalized_jacobian_condition", "Column-normalized Jacobian condition", "(c) Local conditioning"),
        ("holdout_chi2_per_real_sample", r"Held-out $\chi^2/N_{\rm real}$", "(d) Independent prediction"))
    for ax, (key, ylabel, title) in zip(axes.flat, specifications):
        for i, (design, model) in enumerate(groups):
            selected = [r for r in rows if r["design"]==design and r["model"]==model]
            y = np.array([abs(float(r[key]))/(float(r["gamma_dark"]) if key=="omega_error" else 1) for r in selected])
            x = i+np.linspace(-.12, .12, len(y))
            ax.scatter(x, y, s=18, alpha=.55, color=colors[i])
            failed = np.array([r["success"] != "True" for r in selected])
            ax.scatter(x[failed], y[failed], marker="x", s=35, lw=1, color="#603b32")
            ax.plot([i-.17, i+.17], [np.median(y)]*2, color=colors[i], lw=3)
        ax.set_xticks(range(4), names)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
    axes[1, 1].axhline(1+2*np.sqrt(2/316), ls=":", color=".4", lw=1)
    fig.text(.5, .015, "16 extra-link cases per group; bars: medians; crosses: failed convergence. Same 162 complex training measurements.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.10, right=.98, top=.95, bottom=.12, hspace=.45, wspace=.31)
    fig.savefig(output/"remote_window_comparison.pdf")
    fig.savefig(output/"remote_window_comparison.png", dpi=160)
    plt.close(fig)
    source_paths = [ROOT/"run_01/fits.csv", run_path/"fits.csv", Path(__file__)]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (output/"analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
