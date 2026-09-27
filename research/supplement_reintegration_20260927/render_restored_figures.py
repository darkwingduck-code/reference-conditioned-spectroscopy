"""Restyle retained PRL supplementary plots from frozen inputs; run no fits."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, MaxNLocator
import numpy as np
import pandas as pd
import fitz

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "pole_zero_work/code"
DATA = ROOT / "pole_zero_work/data_prl"
sys.path[:0] = [str(CODE), str(ROOT / "research/figure_style_epjp_20260927")]
from figure_style import snapshot_figure, assert_science_unchanged, style_figure


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(name):
    spec = importlib.util.spec_from_file_location("restored_" + name, CODE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    with patch.object(Path, "mkdir", return_value=None):
        spec.loader.exec_module(mod)
    return mod


def capture(fn, *args):
    figures = []
    def save(fig, *unused, **options):
        if fig not in figures:
            figures.append(fig)
    with patch.object(Figure, "savefig", save), patch.object(plt, "close", lambda *a: None):
        fn(*args)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # The complete retained data/code inventory also supports the supplementary
    # derivations; hashing does not execute any of those numerical drivers.
    protected = {str(p.relative_to(ROOT)): sha(p) for base in (DATA, CODE)
                 for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    premium = module("generate_prl_supplement_figures_premium")
    figures = {}
    for number in (1, 4, 5, 6, 8, 9):
        captured = []
        with patch.object(premium, "savefig", lambda fig, path, title: captured.append((path.stem, fig))):
            getattr(premium, "figS" + str(number))()
        figures.update(captured)
    captured = []
    with patch.object(premium, "savefig", lambda fig, path, title: captured.append((path.stem, fig))):
        premium.figS2S3_fullbz()
    figures.update(captured)

    inverse = module("inverse_problem_stress")
    figures["figS12_inverse_problem_stress"] = capture(inverse.make_figure,
        pd.read_csv(DATA / "inverse_problem_stress_trials.csv").to_dict("records"),
        json.loads((DATA / "inverse_problem_stress.json").read_text()))[0]
    continuum = module("weyl_continuum_stress")
    figures["figS13_weyl_continuum_stress"] = capture(continuum.make_figure,
        pd.read_csv(DATA / "weyl_continuum_stress.csv"),
        pd.read_csv(DATA / "weyl_continuum_stress_roots.csv"))[0]

    # Execute only the original plotting block, replacing its fitted inputs by
    # the stored CSV/JSON. No synthetic samples or parameter fits are rerun.
    zero = module("generate_zero_assignment_revision")
    table = pd.read_csv(DATA / "zero_assignment_revision_samples.csv")
    summary = json.loads((DATA / "zero_assignment_revision.json").read_text())
    z = np.linspace(.05, .55, 401) + .004j
    exact_arr = table.exact_cal_real.to_numpy() + 1j * table.exact_cal_imag.to_numpy()
    biased_arr = table.biased_cal_real.to_numpy() + 1j * table.biased_cal_imag.to_numpy()
    env = dict(vars(zero), z=z, exact=zero.response(z), exact_arr=exact_arr, biased_arr=biased_arr,
               principal=complex(*summary["exact_fit"]["principal_part_zero"]),
               uncertainty=summary["calibrated_background_noise_stress"]["noise_uncertainty_complex_rms"],
               calibration_bias=complex(*summary["calibrated_background_noise_stress"]["calibration_bias_complex"]))
    tree = ast.parse((CODE / "generate_zero_assignment_revision.py").read_text())
    body = next(n.body for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    block = next(n for n in body if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr).startswith("plt.rc_context"))
    captured = capture(lambda: exec(compile(ast.Module(body=block.body, type_ignores=[]),
                                          "frozen_zero_assignment_plot", "exec"), env))
    figures["figS10_zero_assignment_revision"] = captured[0]
    records = {}
    for name, fig in figures.items():
        # Retain the prior visibility improvements for overlapping bare branches.
        if name == "figS1_interaction_robustness":
            ax = fig.axes[1]
            for i, line in enumerate(ax.lines[:4]):
                if i >= 2:
                    line.set_linewidth(1.35); line.set_zorder(2)
                else:
                    line.set_linewidth(1.05 if i == 0 else 1.0); line.set_zorder(8+i)
                    line.set_linestyle((0, (5, 2.4)) if i == 0 else (0, (1.2, 1.8)))
                    line.set_marker("o" if i == 0 else "s"); line.set_markersize(3.4)
                    line.set_markerfacecolor("none"); line.set_markeredgewidth(.75)
                    stride = max(6, len(line.get_xdata()) // 8)
                    line.set_markevery((0 if i == 0 else stride//2, stride))
            ax.legend(loc="lower right")
        before = snapshot_figure(fig)
        panels = fig.axes[:4] if name == "figS3_ward_landau_completion" else [a for a in fig.axes if a.get_label() != "<colorbar>"]
        # The old generators also used bare a/b letters; remove only those.
        for ax in panels:
            for text in list(ax.texts):
                if text.get_text() in list("abcd") and text.get_position()[1] > 1:
                    text.remove()
        descriptions = style_figure(fig, panel_axes=panels)
        fig.set_layout_engine(None)
        fig.set_size_inches(7.1, 6.2 if len(panels) == 4 else 3.7)
        fig.subplots_adjust(left=.10, right=.92, bottom=.17, top=.90,
                            wspace=.65, hspace=.62)
        if name == "figS5_genealogy_inference":
            ax = fig.axes[1]
            ticks = [.003, .006, .01, .02]
            labels = dict(zip(ticks, [r"$3\times10^{-3}$", r"$6\times10^{-3}$", r"$10^{-2}$", r"$2\times10^{-2}$"]))
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: labels.get(x, "")))
            ax.xaxis.set_minor_formatter(NullFormatter())
        if name == "figS8_tomography_observability_map":
            fig.subplots_adjust(wspace=.9)
        if name == "figS10_zero_assignment_revision":
            ax = fig.axes[1]
            for axis, center in ((ax.xaxis, .3), (ax.yaxis, -.01)):
                axis.set_major_formatter(FuncFormatter(lambda x, p, c=center: f"{(x-c)/1e-5:.1f}"))
                axis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel(r"$(\mathrm{Re}\,z_0-0.3)/10^{-5}$")
            ax.set_ylabel(r"$(\mathrm{Im}\,z_0+0.01)/10^{-5}$")
        if name == "figS3_ward_landau_completion":
            fig.axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
            fig.axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x / 1e-17:g}"))
            fig.axes[0].set_ylabel(r"relative Ward residual / $10^{-17}$")
            fig.axes[0].legend(loc="lower right", ncol=1, fontsize=9, frameon=False)
        if name == "figS13_weyl_continuum_stress":
            fig.axes[1].text(.04, .96, r"$F_0^a=0.2$", transform=fig.axes[1].transAxes,
                             va="top", fontsize=9)
        digest = assert_science_unchanged(before, fig)
        target = out / (name + ".pdf")
        fig.savefig(target, bbox_inches="tight", metadata={"CreationDate": None})
        fig.savefig(target.with_suffix(".png"), bbox_inches="tight", dpi=180)
        with fitz.open(target) as doc:
            spans = [s for b in doc[0].get_text("dict")["blocks"] if "lines" in b
                     for line in b["lines"] for s in line["spans"]]
            labels = [s for s in spans if s["text"] in ["("+x+")" for x in "abcd"]]
            assert len(labels) == len(panels) and all("Bold" in s["font"] for s in labels), name
            outside = [s["text"] for s in spans if not doc[0].rect.contains(fitz.Rect(s["bbox"]))]
            assert not outside, (name, outside)
        records[name] = {"panels": len(panels), "science_sha256": digest,
                         "pdf_sha256": sha(target), "descriptions_for_caption": descriptions}
        plt.close(fig)
    assert all(sha(ROOT/n) == digest for n, digest in protected.items())
    (out / "manifest.json").write_text(json.dumps({"status": "NUMERICALLY_VERIFIED_VISUAL_REVIEW_PENDING",
        "source_sha256": protected, "figures": records, "source_files_unchanged": True,
        "no_parameter_fits_run": True}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": len(records), "output": str(out)}))


if __name__ == "__main__":
    main()
