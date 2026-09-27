"""Render four legacy figures from frozen CSVs; scoped labels only, no solvers."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
CSV_NAMES = (
    "fig1_dispersion.csv", "fig1_field_intercept.csv", "fig2_pole_zero_tomography.csv",
    "fig2_frequency_residue_circle.csv", "figS7_multimode_poles_zeros.csv",
    "figS7_loss_only_ambiguity.csv", "figS7_cofactor_jacobi_residuals.csv",
    "figS7_background_dependent_total_zeros.csv", "weyl_full_zero_revision.csv",
    "weyl_full_zero_revision_field.csv")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_content(fig):
    content = []
    for ax in fig.axes:
        content.append((ax.get_xlim(), ax.get_ylim(), ax.get_xscale(), ax.get_yscale()))
        for line in ax.lines:
            content.append((np.asarray(line.get_xdata()).tolist(), np.asarray(line.get_ydata()).tolist()))
        for coll in ax.collections:
            content.append(np.asarray(coll.get_offsets()).tolist())
            if coll.get_array() is not None:
                content.append(np.asarray(coll.get_array()).tolist())
    return json.dumps(content, sort_keys=True)


def save_pdf_png(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    with fitz.open(path.with_suffix(".pdf")) as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(path.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, default=ROOT/"selected")
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Choose a new output directory; preserved figures are not overwritten")
    output.mkdir(parents=True)
    code, data = repo/"pole_zero_work/code", repo/"pole_zero_work/data_prl"
    source_names = ("generate_prl_main_figures_premium.py", "generate_prl_supplement_figures_premium.py", "generate_weyl_full_zero_revision.py")
    sources = [code/name for name in source_names]+[data/name for name in CSV_NAMES]+[Path(__file__)]
    hashes = {str(p.relative_to(repo)): sha(p) for p in sources}
    modifications = []

    spec = importlib.util.spec_from_file_location("legacy_main_style", code/source_names[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_matplotlib()
    frames = module.load_inputs(repo/"pole_zero_work")
    panels = output/"panels"
    panels.mkdir()
    captured = []
    module.save_single_panel = lambda fig, path: captured.append(fig)
    for fn, frame, stem in ((module.render_fig1a, "disp", "field_a"),
                             (module.render_fig1b, "traj", "field_b"),
                             (module.render_fig2a, "tomo", "pole_zero_a"),
                             (module.render_fig2b, "circle", "pole_zero_b")):
        fn(frames[frame], panels/stem)
        fig = captured.pop()
        before = numeric_content(fig)
        if stem == "field_b":
            for label in fig.axes[0].get_legend().get_texts():
                if label.get_text() == "exact vanishing-velocity crossover":
                    label.set_text("linear-drift comparator crossover")
                if "weak-field" in label.get_text() and "asymptote" in label.get_text():
                    label.set_text(r"$1/|b_\parallel|$ scaling guide")
            modifications.append("Field panel b legend scopes exact crossover to the specified linear-drift comparator.")
            modifications.append("The separately normalized dotted 1/b curve is labeled a scaling guide, not an exact asymptotic coefficient.")
        assert before == numeric_content(fig)
        fig.savefig(panels/(stem+".pdf"), metadata={"CreationDate": None})
        plt.close(fig)
    for stem, children in (("jpc_field_intercept", ("field_a", "field_b")),
                           ("jpc_pole_zero", ("pole_zero_a", "pole_zero_b"))):
        module.combine_panel_pdfs([panels/(c+".pdf") for c in children], output/(stem+".pdf"))
        with fitz.open(output/(stem+".pdf")) as doc:
            doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(output/(stem+".png"))

    # Retain the original renderer, suppressing its import-time mkdir expressions.
    path = code/source_names[1]
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "mkdir")]
    env = {"__file__": str(path), "__name__": "legacy_supplement_style"}
    exec(compile(tree, str(path), "exec"), env)
    env["savefig"] = lambda fig, *unused: captured.append(fig)
    env["figS7"]()
    fig = captured.pop()
    before = numeric_content(fig)
    env["label_panel"](fig.axes[0], "a", "Pole-zero structure in a\nmultimode response")
    assert before == numeric_content(fig)
    modifications.append("Cofactor panel a title wraps onto two lines; all numerical artists unchanged.")
    save_pdf_png(fig, output/"jpc_cofactor_multimode")

    # Execute only the plotting statements of the full-zero audit, never its solver.
    path = code/source_names[2]
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main").body
    first = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and ast.unparse(n.value.func) == "plt.rcParams.update")
    last = next(i for i, n in enumerate(body[first:], first) if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "base")
    env = {"plt": plt, "scan": pd.read_csv(data/"weyl_full_zero_revision.csv"),
           "fields": pd.read_csv(data/"weyl_full_zero_revision_field.csv")}
    exec(compile(ast.Module(body=body[first:last], type_ignores=[]), str(path), "exec"), env)
    fig = env["fig"]
    before = numeric_content(fig)
    fig.axes[0].set_title("(a) Reduced-model momentum window\n"+r"$b=0.006$")
    locations = [.0015, .003, .006, .012]
    tick_labels = [r"$1.5\times10^{-3}$", r"$3\times10^{-3}$", r"$6\times10^{-3}$", r"$1.2\times10^{-2}$"]
    mapping = dict(zip(locations, tick_labels))
    fig.axes[1].xaxis.set_major_locator(FixedLocator(locations))
    fig.axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: mapping.get(x, "")))
    fig.axes[1].xaxis.set_minor_formatter(NullFormatter())
    assert before == numeric_content(fig)
    modifications.append("Full-zero audit removes the obsolete Fig.2 pointer; field tick labels use the archived editorial spacing.")
    save_pdf_png(fig, output/"jpc_full_zero_audit")
    assert hashes == {str(p.relative_to(repo)): sha(p) for p in sources}
    report = dict(source_hashes=hashes, sources_unchanged=True,
                  numerical_artists_unchanged_by_label_edits=True,
                  no_scientific_solver_executed=True, modifications=modifications,
                  output_hashes={p.name: sha(p) for p in output.glob("*.pdf")})
    (output/"render_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
