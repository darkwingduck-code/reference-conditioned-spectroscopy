#!/usr/bin/env python3
"""Generate publication-grade replacements for the two PRL main figures.

The script deliberately renders *one axes per Matplotlib figure* and then
combines the panel PDFs with PyMuPDF.  This keeps every panel independently
editable while preserving vector output in the final two-panel PDF.

Expected input files under ROOT/data_prl/
-----------------------------------------
fig1_dispersion.csv
fig1_field_intercept.csv
fig2_pole_zero_tomography.csv
fig2_frequency_residue_circle.csv

Example
-------
python generate_prl_main_figures_premium.py \
    --root /path/to/weyl_prl_v8_full \
    --output /path/to/weyl_prl_v8_full/fig_prl \
    --overwrite

Outputs
-------
fig1_field_intercept.pdf / .png
fig2_pole_zero_tomography.pdf / .png
plus separately editable panel PDFs/PNGs/SVGs in OUTPUT/premium_panels/.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import fitz  # PyMuPDF
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator, NullFormatter
import numpy as np
import pandas as pd
from PIL import Image


# -----------------------------------------------------------------------------
# A restrained, colorblind-aware palette.  The visual hierarchy is intentional:
# bare/reference objects are quieter; measured/hybrid objects carry the color.
# -----------------------------------------------------------------------------
INK = "#18232E"
NAVY = "#244A64"
TEAL = "#2A7F78"
VERMILION = "#C65A46"
OCHRE = "#B7862D"
AXIAL_RED = "#D62728"
PLUM = "#70566E"
HYBRID = "#5C405D"
SLATE = "#687682"
MID_GREY = "#9AA4AD"
GRID = "#DDE2E6"
PALE = "#F3F5F7"
PALE_BLUE = "#EEF3F6"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class PanelSize:
    width_in: float = 3.46
    height_in: float = 2.54
    dpi: int = 600


PANEL = PanelSize()


def configure_matplotlib() -> None:
    """Set a compact PRL-compatible style without requiring LaTeX."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 7.6,
            "axes.labelsize": 8.1,
            "axes.titlesize": 8.6,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.72,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.unicode_minus": False,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.major.width": 0.70,
            "ytick.major.width": 0.70,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "legend.fontsize": 6.6,
            "legend.handlelength": 2.25,
            "legend.handletextpad": 0.65,
            "legend.labelspacing": 0.34,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
        }
    )


def require_columns(df: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def new_panel(*, left: float = 0.19, right: float = 0.975,
              bottom: float = 0.19, top: float = 0.88) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(PANEL.width_in, PANEL.height_in))
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    return fig, ax


def polish_axes(ax: plt.Axes, *, ygrid: bool = False) -> None:
    """Remove default-looking clutter while keeping print legibility."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.72)
    ax.spines["bottom"].set_linewidth(0.72)
    ax.tick_params(which="both", top=False, right=False)
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.48, alpha=0.82)


def panel_title(ax: plt.Axes, letter: str, text: str) -> None:
    # One compact line; avoids oversized centered titles.
    ax.set_title(rf"$\bf{{({letter})}}$  {text}", loc="left", pad=7.0, color=INK)


def save_single_panel(fig: plt.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": outbase.stem,
        "Subject": "Publication-quality PRL figure panel",
        "Creator": "Matplotlib + PyMuPDF",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(outbase.with_suffix(".pdf"), dpi=PANEL.dpi, metadata=metadata)
    fig.savefig(outbase.with_suffix(".svg"), dpi=PANEL.dpi)
    fig.savefig(outbase.with_suffix(".png"), dpi=PANEL.dpi)
    plt.close(fig)


def combine_panel_pdfs(panel_pdfs: Sequence[Path], output_pdf: Path,
                       gap_pt: float = 10.0, margin_pt: float = 0.0) -> None:
    """Place panel PDFs side by side while preserving vector graphics."""
    opened = [fitz.open(path) for path in panel_pdfs]
    try:
        rects = [doc[0].rect for doc in opened]
        page_h = max(r.height for r in rects) + 2 * margin_pt
        page_w = sum(r.width for r in rects) + gap_pt * (len(rects) - 1) + 2 * margin_pt
        target = fitz.open()
        page = target.new_page(width=page_w, height=page_h)
        x = margin_pt
        for doc, rect in zip(opened, rects):
            y = margin_pt + (page_h - 2 * margin_pt - rect.height) / 2
            dest = fitz.Rect(x, y, x + rect.width, y + rect.height)
            page.show_pdf_page(dest, doc, 0, keep_proportion=True)
            x += rect.width + gap_pt
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        target.save(output_pdf, garbage=4, deflate=True)
        target.close()
    finally:
        for doc in opened:
            doc.close()


def combine_panel_pngs(panel_pngs: Sequence[Path], output_png: Path,
                       gap_px: int = 42) -> None:
    """Create a high-resolution raster preview with a clean white gutter."""
    images = [Image.open(path).convert("RGB") for path in panel_pngs]
    height = max(im.height for im in images)
    width = sum(im.width for im in images) + gap_px * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), WHITE)
    x = 0
    for im in images:
        y = (height - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + gap_px
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, dpi=(PANEL.dpi, PANEL.dpi), optimize=True)


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    data = root / "data_prl"
    paths = {
        "disp": data / "fig1_dispersion.csv",
        "traj": data / "fig1_field_intercept.csv",
        "tomo": data / "fig2_pole_zero_tomography.csv",
        "circle": data / "fig2_frequency_residue_circle.csv",
    }
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    require_columns(frames["disp"], ["qbar", "b", "branch", "omega_bar"], paths["disp"])
    require_columns(
        frames["traj"],
        ["abs_b", "qmin_over_q0_brightening", "q_h_over_q0_anomaly_normalized", "q_h_weak_asymptote_over_q0", "competitor_x", "qmax_over_q0_demo"],
        paths["traj"],
    )
    require_columns(
        frames["tomo"],
        ["qbar", "omega_minus", "omega_plus", "reconstructed_zero", "zero_field_charge", "zero_field_axial"],
        paths["tomo"],
    )
    require_columns(
        frames["circle"],
        ["b", "residue_asymmetry", "minimum_gap_over_local_split"],
        paths["circle"],
    )
    return frames


def branch(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df[df["branch"] == name].sort_values("qbar")
    if out.empty:
        raise ValueError(f"No branch named '{name}' in fig1_dispersion.csv")
    return out


def _lighten(color: str, amount: float = 0.18):
    """Lighten a color without alpha blending, for print-stable references."""
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


def _draw_reference_branch(
    ax: plt.Axes,
    x,
    y,
    *,
    color: str,
    marker: str,
    label: str,
    ls,
    markevery: int = 9,
    start: int = 0,
    foreground: bool = False,
    linewidth: float = 1.10,
):
    """Draw a reference guide with sparse open markers.

    Foreground guides retain a thin white edge so coincident curves remain
    readable; the wider hybrid branch is still visible on either side.
    """
    xarr = np.asarray(x)
    yarr = np.asarray(y)
    line_color = color if foreground else _lighten(color, 0.16)
    line, = ax.plot(
        xarr, yarr, color=line_color, lw=linewidth, ls=ls,
        label=label, zorder=7 if foreground else 2,
    )
    if foreground:
        line.set_dash_capstyle("round")
        line.set_path_effects([
            pe.Stroke(linewidth=linewidth + 0.50, foreground=WHITE),
            pe.Normal(),
        ])
    idx = np.arange(start, len(xarr), markevery)
    ax.scatter(
        xarr[idx], yarr[idx], s=21, marker=marker,
        facecolors="none", edgecolors=color, linewidths=0.90,
        zorder=8,
    )
    return line


def _draw_hybrid_branch(ax: plt.Axes, x, y, *, label: str | None = None):
    """Draw the hybrid spectrum as the dominant continuous object."""
    line, = ax.plot(
        x, y, color=HYBRID, lw=2.40, label=label, zorder=6,
    )
    # A narrow white under-stroke separates the solid branch from coincident
    # dashed guides while leaving open reference rings visible above it.
    line.set_path_effects([
        pe.Stroke(linewidth=3.15, foreground=WHITE),
        pe.Normal(),
    ])
    return line


def render_fig1a(disp: pd.DataFrame, outbase: Path) -> None:
    charge = branch(disp, "charge")
    axial = branch(disp, "axial")
    lower = branch(disp, "lower")
    upper = branch(disp, "upper")
    q = charge["qbar"].to_numpy()
    q0_idx = np.argmin(np.abs(charge["omega_bar"].to_numpy() - axial["omega_bar"].to_numpy()))
    q0 = float(q[q0_idx])
    b_show = float(lower["b"].iloc[0])

    fig, ax = new_panel(left=0.19, right=0.985, bottom=0.19, top=0.87)
    polish_axes(ax)

    ymin, ymax = 0.092, 0.188
    continuum_top = np.clip(q, ymin, ymax)
    ax.fill_between(q, ymin, continuum_top, color=PALE, zorder=0)
    ax.plot(q, q, color=MID_GREY, lw=0.72, ls=(0, (3.0, 2.4)), zorder=1)
    ax.text(0.985, 0.055, "intraband continuum", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.0, color=SLATE)

    # Quiet bare branches first; sparse open rings remain visible above the
    # final hybrid curve without covering its centerline.
    _draw_reference_branch(
        ax, charge.qbar, charge.omega_bar, color=NAVY, marker="o",
        label=r"charge, $b=0$", ls=(0, (5.0, 2.4)), markevery=10, start=1,
    )
    _draw_reference_branch(
        ax, axial.qbar, axial.omega_bar, color=OCHRE, marker="s",
        label=r"dark axial, $b=0$", ls=(0, (1.2, 1.8)), markevery=10, start=5,
    )

    _draw_hybrid_branch(
        ax, lower.qbar, lower.omega_bar,
        label=rf"hybrid poles, $b={b_show:.3f}$",
    )
    _draw_hybrid_branch(ax, upper.qbar, upper.omega_bar)

    ax.axvline(q0, color=MID_GREY, lw=0.85, ls=(0, (3.3, 2.3, 0.8, 2.3)), zorder=1)
    ax.annotate(r"$\bar q_0$", xy=(q0, ymin), xytext=(4, 6), textcoords="offset points",
                ha="left", va="bottom", fontsize=6.5, color=SLATE)

    ax.set_xlim(float(q.min()), float(q.max()))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\bar q$")
    ax.set_ylabel(r"$\bar\omega$")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    panel_title(ax, "a", "Pre-existing dark pole")

    legend_handles = [
        Line2D([], [], color=HYBRID, lw=2.40,
               label=rf"hybrid poles, $b={b_show:.3f}$"),
        Line2D([], [], color=_lighten(NAVY, 0.16), lw=1.10,
               ls=(0, (5.0, 2.4)), marker="o", markersize=3.7,
               markerfacecolor=WHITE, markeredgecolor=NAVY, markeredgewidth=0.90,
               label=r"charge, $b=0$"),
        Line2D([], [], color=_lighten(OCHRE, 0.16), lw=1.10,
               ls=(0, (1.2, 1.8)), marker="s", markersize=3.6,
               markerfacecolor=WHITE, markeredgecolor=OCHRE, markeredgewidth=0.90,
               label=r"dark axial, $b=0$"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False,
              borderaxespad=0.15)

    save_single_panel(fig, outbase)

def render_fig1b(traj: pd.DataFrame, outbase: Path) -> None:
    traj = traj.sort_values("abs_b")
    x = traj["abs_b"].to_numpy()
    bright = traj["qmin_over_q0_brightening"].to_numpy()
    vanish = traj["q_h_over_q0_anomaly_normalized"].to_numpy()
    weak = traj["q_h_weak_asymptote_over_q0"].to_numpy()
    qmax_ratio = float(traj["qmax_over_q0_demo"].iloc[0])

    fig, ax = new_panel(left=0.20, right=0.985, bottom=0.19, top=0.87)
    polish_axes(ax, ygrid=True)

    ax.axhspan(qmax_ratio, 7.2, color=PALE_BLUE, zorder=0)
    ax.axhline(qmax_ratio, color=SLATE, lw=0.9, ls=(0, (4.0, 2.4)), zorder=1)
    ax.text(0.975, 0.51, r"outside $\bar q<0.2$ window", transform=ax.transAxes,
            ha="right", va="center", fontsize=6.4, color=SLATE,
            bbox={"boxstyle": "round,pad=0.18", "fc": WHITE, "ec": "none", "alpha": 0.92})

    ax.semilogx(x, bright, color=TEAL, lw=1.75, marker="o", ms=4.2,
                mfc=WHITE, mec=TEAL, mew=1.15,
                label=r"brightening: $q_{\min}/q_0$", zorder=4)
    ax.semilogx(x, vanish, color=VERMILION, lw=1.85, marker="s", ms=4.0,
                mfc=WHITE, mec=VERMILION, mew=1.10,
                label=r"exact vanishing-velocity crossover", zorder=4)
    ax.semilogx(x, weak, color=OCHRE, lw=1.05, ls=(0, (1.2, 2.0)),
                alpha=0.95, label=r"weak-field $1/|b_\parallel|$ asymptote", zorder=5)
    ax.axhline(1.0, color=MID_GREY, lw=0.8, ls=(0, (1.1, 2.0)), zorder=1)

    ax.set_xlabel(r"$|b_\parallel|$")
    ax.set_ylabel(r"crossing wave vector / $q_0$")
    ax.set_ylim(0.25, 7.2)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    panel_title(ax, "b", "Field-intercept diagnostic")
    ax.legend(loc="upper right", frameon=False, borderaxespad=0.1, labelspacing=0.28)

    save_single_panel(fig, outbase)


def render_fig2a(tomo: pd.DataFrame, outbase: Path) -> None:
    df = tomo.sort_values("qbar")
    q = df["qbar"].to_numpy()
    delta = np.abs(df["zero_field_charge"].to_numpy() - df["zero_field_axial"].to_numpy())
    q0 = float(q[np.argmin(delta)])

    fig, ax = new_panel(left=0.19, right=0.985, bottom=0.19, top=0.87)
    polish_axes(ax)

    # Thin foreground guides survive exact overlap with the hybrid spectrum.
    # Stagger square/circle samples relative to the principal-part diamonds.
    guide_width = 0.70
    axial_style = (0, (1.2, 2.2))
    _draw_reference_branch(
        ax, q, df.zero_field_charge, color=NAVY, marker="o",
        label="uncoupled charge", ls=(0, (5.0, 2.4)), markevery=10, start=1,
        foreground=True, linewidth=guide_width,
    )
    _draw_reference_branch(
        ax, q, df.zero_field_axial, color=AXIAL_RED, marker="s",
        label=r"hidden axial pole, $b=0$", ls=axial_style,
        markevery=10, start=2, foreground=True, linewidth=guide_width,
    )

    _draw_hybrid_branch(ax, q, df.omega_minus, label="hybrid poles")
    _draw_hybrid_branch(ax, q, df.omega_plus)

    # Open diamonds identify the reconstructed zero while allowing both the
    # solid hybrid line and the dotted zero-field guide to remain visible.
    mark = np.arange(0, len(df), 5)
    ax.scatter(
        q[mark], df.reconstructed_zero.to_numpy()[mark],
        s=21, marker="D", facecolors="none", edgecolors=TEAL,
        linewidths=1.05, label="principal-part zero", zorder=9,
    )

    max_error = float(np.max(np.abs(df.reconstructed_zero - df.zero_field_axial)))
    exponent = int(np.floor(np.log10(max_error)))
    mantissa = max_error / (10.0 ** exponent)
    ax.text(
        0.975, 0.065,
        rf"$\max|\widetilde z_0-\bar\omega_a^0|={mantissa:.1f}\times10^{{{exponent}}}$",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=5.8, color=SLATE,
        bbox={"boxstyle": "round,pad=0.16", "fc": WHITE,
              "ec": "none", "alpha": 0.92},
        zorder=10,
    )

    ax.axvline(q0, color=MID_GREY, lw=0.85, ls=(0, (3.3, 2.3, 0.8, 2.3)), zorder=1)
    ax.text(
        q0 + 0.0008, 0.055, r"$\bar q_0$",
        transform=ax.get_xaxis_transform(), ha="left", va="bottom",
        fontsize=6.5, color=SLATE,
    )

    ax.set_xlabel(r"$\bar q$")
    ax.set_ylabel(r"$\bar\omega$")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    panel_title(ax, "a", "Principal-part zero estimate")
    legend_handles = [
        Line2D([], [], color=HYBRID, lw=2.40, label="hybrid poles"),
        Line2D([], [], color=NAVY, lw=guide_width,
               ls=(0, (5.0, 2.4)), marker="o", markersize=3.7,
               markerfacecolor=WHITE, markeredgecolor=NAVY, markeredgewidth=0.90,
               label="uncoupled charge"),
        Line2D([], [], color=AXIAL_RED, lw=guide_width,
               ls=axial_style, dash_capstyle="round", marker="s", markersize=3.6,
               markerfacecolor=WHITE, markeredgecolor=AXIAL_RED, markeredgewidth=0.90,
               label=r"hidden axial pole, $b=0$"),
        Line2D([], [], color=TEAL, marker="D", markersize=4.8,
               markerfacecolor=WHITE, markeredgewidth=1.00, lw=0,
               label="principal-part zero"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False,
              borderaxespad=0.1)

    save_single_panel(fig, outbase)

def render_fig2b(circle: pd.DataFrame, outbase: Path) -> None:
    fig, ax = new_panel(left=0.18, right=0.985, bottom=0.17, top=0.87)
    polish_axes(ax)

    x = np.linspace(-1.0, 1.0, 800)
    y = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    ax.fill_between(x, 0.0, y, color=PALE, zorder=0)
    ax.plot(x, y, color=INK, lw=1.65, label=r"$\mathcal{A}_W^2+\mathcal{R}_\omega^2=1$", zorder=2)

    palette = {0.0025: NAVY, 0.005: TEAL, 0.010: VERMILION}
    markers = {0.0025: "o", 0.005: "s", 0.010: "^"}
    sizes = {0.0025: 34, 0.005: 26, 0.010: 20}
    labels = {0.0025: r"$b=0.0025$", 0.005: r"$b=0.005$", 0.010: r"$b=0.010$"}
    # Nested, mostly open markers remain distinguishable even when the three
    # weak-field datasets collapse onto the same theoretical arc.
    for order, b in enumerate((0.0025, 0.005, 0.010)):
        sub = circle[np.isclose(circle["b"], b)].sort_values("residue_asymmetry").iloc[::2]
        filled = b == 0.010
        ax.scatter(
            sub.residue_asymmetry,
            sub.minimum_gap_over_local_split,
            s=sizes[b],
            marker=markers[b],
            facecolor=palette[b] if filled else "none",
            edgecolor=palette[b],
            linewidth=1.0,
            alpha=0.96,
            label=labels[b],
            zorder=4 + order,
        )

    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-0.025, 1.045)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel(r"$\mathcal{A}_W=(W_+-W_-)/(W_++W_-)$")
    ax.set_ylabel(r"$\mathcal{R}_\omega=\Delta\omega_{\min}/(\omega_+-\omega_-)$")
    panel_title(ax, "b", "Pole-weight circle law")

    # The empty interior of the semicircle is the cleanest legend location.
    ax.legend(loc="lower center", bbox_to_anchor=(0.50, 0.08), frameon=False,
              ncol=2, columnspacing=1.15, handletextpad=0.45)

    save_single_panel(fig, outbase)


def render_all(root: Path, output: Path, overwrite: bool) -> list[Path]:
    frames = load_inputs(root)
    panels = output / "premium_panels"
    panels.mkdir(parents=True, exist_ok=True)

    final_paths = [
        output / "fig1_field_intercept.pdf",
        output / "fig1_field_intercept.png",
        output / "fig2_pole_zero_tomography.pdf",
        output / "fig2_pole_zero_tomography.png",
    ]
    if not overwrite:
        existing = [p for p in final_paths if p.exists()]
        if existing:
            joined = "\n".join(str(p) for p in existing)
            raise FileExistsError(
                "Output files already exist. Re-run with --overwrite or choose another --output:\n" + joined
            )

    p1a = panels / "fig1a_preexisting_dark_pole"
    p1b = panels / "fig1b_field_intercept"
    p2a = panels / "fig2a_hidden_pole_reconstruction"
    p2b = panels / "fig2b_frequency_residue_circle"

    render_fig1a(frames["disp"], p1a)
    render_fig1b(frames["traj"], p1b)
    render_fig2a(frames["tomo"], p2a)
    render_fig2b(frames["circle"], p2b)

    combine_panel_pdfs(
        [p1a.with_suffix(".pdf"), p1b.with_suffix(".pdf")],
        output / "fig1_field_intercept.pdf",
    )
    combine_panel_pngs(
        [p1a.with_suffix(".png"), p1b.with_suffix(".png")],
        output / "fig1_field_intercept.png",
    )
    combine_panel_pdfs(
        [p2a.with_suffix(".pdf"), p2b.with_suffix(".pdf")],
        output / "fig2_pole_zero_tomography.pdf",
    )
    combine_panel_pngs(
        [p2a.with_suffix(".png"), p2b.with_suffix(".png")],
        output / "fig2_pole_zero_tomography.png",
    )

    return final_paths + sorted(panels.glob("*"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Root of the Weyl PRL package (contains data_prl/).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory. Default: ROOT/fig_prl")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing main-figure files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = (args.output or (root / "fig_prl")).expanduser().resolve()
    configure_matplotlib()
    created = render_all(root, output, overwrite=args.overwrite)
    print("Created publication-grade figure files:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
