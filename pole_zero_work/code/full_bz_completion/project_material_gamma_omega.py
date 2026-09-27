#!/usr/bin/env python3
"""Project a material dynamic forward vertex Gamma^omega onto Landau modes.

Input is an NPZ in one common orbital/Wannier gauge. Required arrays:
  states[P,A]              complex normalized band spinors
  area[P]                  positive Fermi-surface patch areas
  node[P]                  integer valley/node labels
  gamma[P,P,A,A,A,A]       antisymmetrized dynamic-limit 1PI vertex
and either
  velocity[P], residue[P]
or
  v0[P,3], grad_sigma[P,3], dsigma_domega[P].

The script never converts a static cRPA matrix into a Landau interaction.  The
caller must supply Gamma^omega obtained from a conserving many-body solver.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from landau_projection import (
    FermiSurface,
    mode_diagnostics,
    project_gamma,
    residue_from_sigma,
    velocity_from_sigma,
)


def load_input(path: Path) -> tuple[FermiSurface, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
        required = {"states", "area", "node", "gamma"}
        missing = required - names
        if missing:
            raise ValueError(f"missing required arrays: {sorted(missing)}")
        states = np.asarray(z["states"], complex)
        area = np.asarray(z["area"], float)
        node = np.asarray(z["node"], int)
        gamma = np.asarray(z["gamma"], complex)
        if {"velocity", "residue"}.issubset(names):
            velocity = np.asarray(z["velocity"], float)
            residue = np.asarray(z["residue"], float)
        elif {"v0", "grad_sigma", "dsigma_domega"}.issubset(names):
            residue = residue_from_sigma(z["dsigma_domega"])
            vstar = velocity_from_sigma(z["v0"], z["grad_sigma"], residue)
            velocity = np.linalg.norm(vstar, axis=1)
        else:
            raise ValueError(
                "provide either velocity+residue or v0+grad_sigma+dsigma_domega"
            )
    return FermiSurface(states, area, velocity, residue, node), gamma


def parse_signs(text: str, nodes: np.ndarray) -> dict[int, float]:
    if text:
        raw = json.loads(text)
        out = {int(k): float(v) for k, v in raw.items()}
    else:
        labels = sorted(set(int(x) for x in nodes))
        if labels == [-1, 1]:
            out = {-1: -1.0, 1: 1.0}
        elif len(labels) == 2:
            out = {labels[0]: -1.0, labels[1]: 1.0}
        else:
            raise ValueError("--axial-signs JSON is required for more than two node labels")
    if any(int(x) not in out for x in nodes):
        raise ValueError("axial-sign map does not cover all node labels")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_npz", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--axial-signs",
        default="",
        help='JSON map from node label to axial sign, e.g. \'{"-1":-1,"1":1}\'',
    )
    ap.add_argument("--no-hermitize", action="store_true")
    args = ap.parse_args()

    fs, gamma = load_input(args.input_npz)
    signs = parse_signs(args.axial_signs, fs.node)
    f = project_gamma(fs, gamma, hermitize=not args.no_hermitize)
    anti = float(np.max(np.abs(f - f.conj().T)))
    diag = mode_diagnostics(fs, f, signs)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "mode": np.arange(fs.n),
            "eigenvalue": diag["eigenvalues"],
            "charge_overlap": diag["charge_overlap"],
            "axial_overlap": diag["axial_overlap"],
        }
    ).to_csv(out / "landau_modes.csv", index=False)
    np.savez_compressed(
        out / "projected_landau_operator.npz",
        f=f,
        eigenvalues=diag["eigenvalues"],
        eigenvectors=diag["eigenvectors"],
        two_mode=diag["two_mode"],
    )
    summary = {
        "input": str(args.input_npz),
        "patches": fs.n,
        "orbitals": fs.norb,
        "node_labels": sorted(set(int(x) for x in fs.node)),
        "projected_hermiticity_residual": anti,
        "relative_two_mode_leakage": diag["relative_leakage"],
        "two_mode_matrix_re": diag["two_mode"].real.tolist(),
        "two_mode_matrix_im": diag["two_mode"].imag.tolist(),
        "leading_eigenvalues": diag["eigenvalues"][: min(12, fs.n)].tolist(),
        "leading_charge_overlap": diag["charge_overlap"][: min(12, fs.n)].tolist(),
        "leading_axial_overlap": diag["axial_overlap"][: min(12, fs.n)].tolist(),
    }
    (out / "landau_projection_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
