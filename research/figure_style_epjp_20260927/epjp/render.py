"""Restyle eight frozen EPJP figures with original plotting code, without fits."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from figure_style import assert_science_unchanged, snapshot_figure, style_figure


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def body(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return next(n.body for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def plot_only(path, env, first, last):
    nodes = body(path)
    start = next(i for i, n in enumerate(nodes) if first(ast.unparse(n)))
    stop = next(i for i, n in enumerate(nodes[start:], start) if last(ast.unparse(n)))
    nodes = [n for n in nodes[start:stop] if ".to_csv(" not in ast.unparse(n)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env["fig"]


def inventory():
    root = REPO / "paper/jpc_strengthened"
    result = []
    for document in ("manuscript", "supplement"):
        seen = set()
        def visit(path):
            if path in seen:
                return
            seen.add(path)
            content = path.read_text(encoding="utf-8-sig")
            for match in re.finditer(r"\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", content):
                kind, name = match.groups()
                if kind == "input":
                    visit(root / (name if name.endswith(".tex") else name + ".tex"))
                else:
                    file = root / name
                    if not file.exists():
                        file = root / "figures" / name
                    result.append({"document": document, "source_tex": str(path.relative_to(REPO)),
                                   "include": name, "asset": str(file.relative_to(REPO)),
                                   "sha256": sha(file)})
        visit(root / f"{document}.tex")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "render_01")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory; earlier assets are retained")
    out.mkdir(parents=True)
    code = REPO / "pole_zero_work/code"
    data = REPO / "pole_zero_work/data_prl"
    reference = REPO / "research/reference_validation_20260927"
    control = REPO / "research/reference_control_study_20260927"
    structured = REPO / "research/structured_dark_sector_20260927"
    figures = {}
    records = inventory()
    sources = [Path(__file__), HERE.parent / "figure_style.py"]
    sources += [code / name for name in ("generate_prl_main_figures_premium.py",
                "generate_prl_supplement_figures_premium.py", "generate_weyl_full_zero_revision.py")]
    csv_names = ("fig1_dispersion.csv", "fig1_field_intercept.csv", "fig2_pole_zero_tomography.csv",
                 "fig2_frequency_residue_circle.csv", "figS7_multimode_poles_zeros.csv",
                 "figS7_loss_only_ambiguity.csv", "figS7_cofactor_jacobi_residuals.csv",
                 "figS7_background_dependent_total_zeros.csv", "weyl_full_zero_revision.csv",
                 "weyl_full_zero_revision_field.csv")
    sources += [data / n for n in csv_names]
    sources += [reference / p for p in ("render_results.py", "experiment.py", "run_02/trials.csv")]
    sources += [control / p for p in ("analyze.py", "analyze_wide.py", "run_01/fits.csv",
                                    "run_01/profiles.csv", "run_03_wide/fits.csv")]
    sources += [structured / p for p in ("render_extended.py", "run_study.py", "run_02/trials.csv",
                                       "run_02/summary.json")]
    sources += [REPO / r["asset"] for r in records]
    protected = {str(p.relative_to(REPO)): sha(p) for p in sources}

    # Reuse the four original single-panel render functions in a common canvas.
    legacy = module(code / "generate_prl_main_figures_premium.py", "epjp_legacy_main")
    legacy.configure_matplotlib()
    for stem, specifications in (("jpc_field_intercept", ((legacy.render_fig1a, csv_names[0]),
                                                          (legacy.render_fig1b, csv_names[1]))),
                                 ("jpc_pole_zero", ((legacy.render_fig2a, csv_names[2]),
                                                    (legacy.render_fig2b, csv_names[3])))):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))
        axis_iterator = iter(axes)
        legacy.new_panel = lambda **unused: (fig, next(axis_iterator))
        legacy.save_single_panel = lambda *unused: None
        for fn, csv_name in specifications:
            fn(pd.read_csv(data / csv_name), out / stem)
        if stem == "jpc_field_intercept":
            for text in axes[1].get_legend().get_texts():
                if text.get_text() == "exact vanishing-velocity crossover":
                    text.set_text("linear-drift comparator crossover")
                if "weak-field" in text.get_text() and "asymptote" in text.get_text():
                    text.set_text(r"$1/|b_\parallel|$ scaling guide")
        figures[stem] = fig

    # Suppress only import-time output-directory creation in the old supplement.
    path = code / "generate_prl_supplement_figures_premium.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "mkdir")]
    env = {"__file__": str(path), "__name__": "epjp_legacy_supplement"}
    exec(compile(tree, str(path), "exec"), env)
    captured = []
    env["savefig"] = lambda fig, *unused: captured.append(fig)
    env["figS7"]()
    figures["jpc_cofactor_multimode"] = captured.pop()

    env = {"plt": plt, "scan": pd.read_csv(data / "weyl_full_zero_revision.csv"),
           "fields": pd.read_csv(data / "weyl_full_zero_revision_field.csv")}
    figures["jpc_full_zero_audit"] = plot_only(code / "generate_weyl_full_zero_revision.py", env,
        lambda s: s.startswith("plt.rcParams.update"), lambda s: s.startswith("base ="))
    audit = figures["jpc_full_zero_audit"]
    audit.axes[0].set_title("(a) Reduced-model momentum window")
    audit.axes[0].text(.98, .95, r"$b=0.006$", ha="right", va="top", transform=audit.axes[0].transAxes)
    audit.axes[1].text(.98, .94, r"$\bar q=\bar q_0$", ha="right", va="top", transform=audit.axes[1].transAxes)
    locations = [.0015, .003, .006, .012]
    mapping = dict(zip(locations, [r"$1.5\times10^{-3}$", r"$3\times10^{-3}$", r"$6\times10^{-3}$", r"$1.2\times10^{-2}$"]))
    audit.axes[1].xaxis.set_major_locator(FixedLocator(locations))
    audit.axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: mapping.get(x, "")))
    audit.axes[1].xaxis.set_minor_formatter(NullFormatter())

    sys.path.insert(0, str(reference))
    experiment = module(reference / "experiment.py", "experiment")
    table = pd.read_csv(reference / "run_02/trials.csv")
    env = {"plt": plt, "np": np, "pd": pd, "data": table, "Q0": experiment.Q0}
    figures["reference_validation_20260927"] = plot_only(reference / "render_results.py", env,
        lambda s: s.startswith("valid ="), lambda s: s.startswith("fig.savefig"))

    analyze = module(control / "analyze.py", "epjp_control_analyze")
    fits = analyze.read_csv(control / "run_01/fits.csv")
    profiles = analyze.read_csv(control / "run_01/profiles.csv")
    original_save, original_close = Figure.savefig, plt.close
    Figure.savefig = lambda fig, *a, **k: captured.append(fig) if not captured else None
    plt.close = lambda *a, **k: None
    try:
        analyze.make_figure(fits, profiles, out)
        figures["reference_control"] = captured.pop()
    finally:
        Figure.savefig, plt.close = original_save, original_close
    rows = []
    for run, design in (("run_01", "low_window"), ("run_03_wide", "remote_inclusive")):
        rows += [dict(row, design=design) for row in analyze.read_csv(control / run / "fits.csv")
                 if row["control"] == "extra_link" and row["model"] != "unchanged"]
    groups = [(d, m) for m in ("drift_aware", "extended") for d in ("low_window", "remote_inclusive")]
    env = {"plt": plt, "np": np, "rows": rows, "groups": groups}
    figures["remote_window_comparison"] = plot_only(control / "analyze_wide.py", env,
        lambda s: s.startswith("plt.rcParams.update"), lambda s: s.startswith("fig.savefig"))

    study = module(structured / "run_study.py", "epjp_structured_study")
    with (structured / "run_02/trials.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    env = {"plt": plt, "np": np, "Patch": Patch, "rows": rows,
           "report": json.loads((structured / "run_02/summary.json").read_text()),
           "MODELS": study.MODELS, "OMEGA": study.OMEGA, "DARK": study.DARK, "effective": study.effective}
    figures["structured_sector"] = plot_only(structured / "render_extended.py", env,
        lambda s: s.startswith("plt.rcParams.update"), lambda s: s.startswith("output.mkdir"))

    manifest = {"inventory": records, "source_hashes": protected, "figures": {},
                "no_fits_or_experiments_run": True, "no_pdf_or_raster_post_editing": True,
                "style": {"family": "DejaVu Sans", "panel_pt": 11, "panel_weight": "bold",
                          "axis_label_pt": 10, "tick_pt": 9, "legend_pt": 9,
                          "canvas_width_inches": 7.1}}
    for name, fig in figures.items():
        before = snapshot_figure(fig)
        descriptors = style_figure(fig)
        fig.set_layout_engine(None)
        if name == "reference_validation_20260927":
            fig.set_size_inches(7.1, 4.0)
            handles, labels = fig.axes[0].get_legend_handles_labels()
            for legend in list(fig.legends):
                legend.remove()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.33, .01),
                       ncol=2, frameon=False, fontsize=9)
            status_handles, status_labels = fig.axes[2].get_legend_handles_labels()
            fig.axes[2].get_legend().remove()
            fig.legend(status_handles, status_labels, loc="lower center", bbox_to_anchor=(.855, .01),
                       ncol=1, frameon=False, fontsize=9)
            damping_handles, _ = fig.axes[1].get_legend_handles_labels()
            fig.axes[1].legend(damping_handles,
                              [r"true $\gamma_d$", r"fixed $\gamma_d$", r"inferred $\gamma_d$"],
                              loc="lower right", frameon=False, fontsize=9, handlelength=1.2)
            fig.axes[2].set_xlabel("Signal / calibration\nradius")
            fig.subplots_adjust(left=.085, right=.98, top=.91, bottom=.38, wspace=.62)
        elif name == "reference_control":
            fig.set_size_inches(7.1, 6.5)
            for ax in fig.axes[:3]:
                ax.set_xticks(range(3), ["No\ndrift", "Bright\ndrift", "+ Extra\nlink"])
            for ax in fig.axes[3:5]:
                ax.tick_params(axis="x", labelrotation=30)
            fig.subplots_adjust(left=.10, right=.98, top=.94, bottom=.17, hspace=.60, wspace=.57)
        elif name == "remote_window_comparison":
            fig.set_size_inches(7.1, 6.2)
            for text in list(fig.texts):
                text.remove()  # Sentence already in the manuscript caption.
            for ax in fig.axes:
                ax.set_xticks(range(4), ["Star\nlow\nwindow", "Star\nlow +\nremote",
                                        "Extra-link\nlow\nwindow", "Extra-link\nlow +\nremote"])
            fig.axes[2].set_ylabel("Column-normalized\nJacobian condition")
            fig.subplots_adjust(left=.12, right=.98, top=.94, bottom=.14, hspace=.56, wspace=.43)
        elif name in ("jpc_cofactor_multimode", "structured_sector"):
            fig.set_size_inches(7.1, 5.8)
            if name == "jpc_cofactor_multimode":
                ax = fig.axes[3]
                ax.legend(loc="center left", bbox_to_anchor=(.01, .57), frameon=False, fontsize=9)
                for text in ax.texts:
                    if hasattr(text, "xy") and text.xy[0] > .4:
                        text.set_position((-5, 4))
                        text.set_horizontalalignment("right")
            else:
                fig.axes[3].legend(loc="upper left", frameon=False, fontsize=9)
            fig.subplots_adjust(left=.12, right=.98, top=.94, bottom=.13, wspace=.47, hspace=.42)
        else:
            fig.set_size_inches(7.1, 3.35)
            if name == "jpc_field_intercept":
                for text in fig.axes[0].texts:
                    if text.get_text() == "intraband continuum":
                        text.set_text("intraband\ncontinuum")
                        text.set_position((.985, .045))
            elif name == "jpc_pole_zero":
                fig.axes[1].set_anchor("N")
                for text in fig.axes[0].texts:
                    if "\\max" in text.get_text():
                        text.set_position((.02, -.30))
                        text.set_horizontalalignment("left")
            elif name == "jpc_full_zero_audit":
                for text in fig.axes[0].texts:
                    if text.get_text() == r"$b=0.006$":
                        text.set_position((.98, .84))
            fig.subplots_adjust(left=.10, right=.98, top=.87, bottom=.22, wspace=.38)
        fingerprint = assert_science_unchanged(before, fig)
        for ax in fig.axes:
            panel = [text for text in ax.texts if text.get_gid() == "publication-panel-label"]
            assert len(panel) == 1 and panel[0].get_fontsize() == 11
            assert panel[0].get_fontweight() == "bold"
            assert not any(ax.get_title(loc=loc) for loc in ("left", "center", "right"))
            assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 10
        pdf = out / f"{name}.pdf"
        fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=.08)
        assert_science_unchanged(before, fig)
        with fitz.open(pdf) as doc:
            doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(out / f"{name}.png")
            size = [doc[0].rect.width, doc[0].rect.height]
        manifest["figures"][name] = {"pdf_sha256": sha(pdf), "scientific_artist_sha256": fingerprint,
            "scientific_artists_unchanged": True, "axis_count": len(fig.axes), "pdf_size_pt": size,
            "panel_descriptors_for_caption": descriptors, "original_artist_snapshot": before}
        plt.close(fig)
    manifest["sources_unchanged"] = protected == {str(p.relative_to(REPO)): sha(p) for p in sources}
    assert manifest["sources_unchanged"]
    assert {Path(item["asset"]).stem for item in records} == set(figures)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    mapping = [{**item, "replacement": str((out / Path(item["asset"]).name).relative_to(REPO)),
                "replacement_sha256": manifest["figures"][Path(item["asset"]).stem]["pdf_sha256"]}
               for item in records]
    (out / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "figure_count": len(figures), "sources_unchanged": True,
                      "science_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
