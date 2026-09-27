"""Isolated fresh reproduction of all archived full-BZ production tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / "data_prl" / "full_bz_completion"
sys.path.insert(0, str(HERE))
from ensemble import hall_compensated  # noqa: E402
from lattice_kubo import (  # noqa: E402
    SlabParams, common_mu_pair, density, density_kubo, diagonalize_slab,
)

FIELDS = np.array([0.0008, 0.0012, 0.0016, 0.0020])
MU0, TEMP = 0.7, 0.05


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intercept(values: list[float]) -> float:
    return float(np.polyfit(FIELDS**2, np.asarray(values), 2)[-1])


def convergence_row(p: SlabParams, raw_rows: list[dict]) -> dict[str, float | int]:
    q, omega = 2*np.pi/p.Nkz, 2j*(2*np.pi/p.Nkz)
    zero = diagonalize_slab(p, 0.0)
    n0 = density(zero, MU0, TEMP)
    c0 = density_kubo(zero, MU0, TEMP, 1, omega).total
    nq: list[float] = []
    mq: list[float] = []
    kmuq: list[float] = []
    knq: list[float] = []
    for field in FIELDS:
        minus = diagonalize_slab(p, -field)
        plus = diagonalize_slab(p, field)
        mu = common_mu_pair(minus, plus, n0, TEMP)
        nq.append((0.5*(density(minus, MU0, TEMP)+density(plus, MU0, TEMP))-n0)/field**2)
        mq.append((mu-MU0)/field**2)
        cm = density_kubo(minus, MU0, TEMP, 1, omega).total
        cp = density_kubo(plus, MU0, TEMP, 1, omega).total
        nm = density_kubo(minus, mu, TEMP, 1, omega).total
        np_ = density_kubo(plus, mu, TEMP, 1, omega).total
        kmuq.append(((0.5*(cm+cp)-c0)/field**2).real)
        knq.append(((0.5*(nm+np_)-c0)/field**2).real)
        raw_rows.append({"Lx": p.Lx, "Nky": p.Nky, "Nkz": p.Nkz,
                         "B": float(field), "n2_quotient": nq[-1],
                         "mu2_quotient": mq[-1], "K2_mu_quotient_re": kmuq[-1],
                         "K2_n_quotient_re": knq[-1]})
    n2, mu2, k2mu, k2n = map(intercept, (nq, mq, kmuq, knq))
    step = 2e-5
    n0p = (density(zero, MU0+step, TEMP)-density(zero, MU0-step, TEMP))/(2*step)
    k0p = (density_kubo(zero, MU0+step, TEMP, 1, omega).total
           - density_kubo(zero, MU0-step, TEMP, 1, omega).total)/(2*step)
    mu_chain, k_chain = hall_compensated(n0p=n0p, n2=n2, K0p=k0p, K2=k2mu)
    return {"Lx": p.Lx, "Nky": p.Nky, "Nkz": p.Nkz, "n2": n2,
            "mu2": mu2, "K2_fixed_mu_real": k2mu, "K2_fixed_n_real": k2n,
            "K2_chain_real": complex(k_chain).real,
            "mu_chain_relative_error": abs(mu2/mu_chain-1),
            "K_chain_relative_error": abs((k2n-complex(k_chain).real)/k2n)}


def numeric_comparison(actual: float, expected: float, field: str, filename: str) -> dict:
    """Predeclared mixed tolerances reflect subtraction conditioning and null residuals.

    Values of Ward/gauge residuals are machine-roundoff diagnostics, so relative
    equality of those residuals is not a reproducibility criterion. All actual
    differences remain reported. These bounds do not certify continuum limits.
    """
    if np.isnan(expected) and np.isnan(actual):
        return {"pass": True, "absolute_error": 0.0, "relative_error": None,
                "allowed_error": 0.0, "criterion": "matching_missing_value"}
    if not np.isfinite(actual) or not np.isfinite(expected):
        return {"pass": False, "absolute_error": None, "relative_error": None,
                "allowed_error": 0.0, "criterion": "unexpected_nonfinite"}
    rtol, atol = 2e-6, 2e-8
    criterion = "B2_subtraction_coefficients"
    if "convergence" in filename:
        rtol, atol, criterion = 2e-5, 2e-8, "reconstructed_B0_coefficients"
    if "torus" in filename or "landau" in filename:
        rtol, atol, criterion = 1e-8, 1e-12, "direct_spectrum_or_response"
    if field in {"vertex", "max_vertex"}:
        rtol, atol, criterion = 0.0, 1e-11, "Ward_vertex_absolute"
    elif field in {"left", "right", "max_left", "max_right"}:
        rtol, atol, criterion = 0.0, 1e-12, "Ward_response_absolute"
    elif "residual" in field:
        rtol, atol, criterion = 0.0, 1e-10, "gauge_covariance_absolute"
    elif "relerr" in field or "relative_error" in field:
        rtol, atol, criterion = 0.0, 1e-6, "chain_rule_relative_residual_absolute"
    elif field in {"Lx", "Ly", "L", "Nky", "Nkz", "nphi", "mode", "patches"}:
        rtol, atol, criterion = 0.0, 0.0, "exact_discrete_parameter"
    error = abs(actual-expected)
    allowed = atol+rtol*abs(expected)
    return {"pass": bool(error <= allowed), "absolute_error": error,
            "relative_error": error/max(abs(expected), 1e-300),
            "allowed_error": allowed, "criterion": criterion}


def compare_csv(generated: Path, reference: Path, cells: list[dict]) -> dict[str, object]:
    got, ref = pd.read_csv(generated), pd.read_csv(reference)
    if got.shape != ref.shape or list(got) != list(ref):
        return {"status": "FAIL", "rows": len(got), "max_abs": None,
                "max_rel": None, "detail": "shape_or_columns"}
    numeric = [name for name in ref if pd.api.types.is_numeric_dtype(ref[name])]
    text = [name for name in ref if name not in numeric]
    text_ok = all(got[name].astype(str).equals(ref[name].astype(str)) for name in text)
    ga, ra = got[numeric].to_numpy(float), ref[numeric].to_numpy(float)
    delta = np.abs(ga-ra)
    max_abs = float(np.nanmax(delta)) if delta.size else 0.0
    max_rel = float(np.nanmax(delta/np.maximum(np.abs(ra), 1e-12))) if delta.size else 0.0
    comparisons = []
    basis_dependent_rows: set[int] = set()
    subspace_comparisons = []
    if reference.name == "exact_landau_spectrum.csv":
        # Within a degenerate eigenspace only the sum of squared overlaps is
        # basis invariant. Keep the raw per-eigenvector discrepancies below.
        scale = max(1.0, float(np.max(np.abs(ref.eigenvalue))))
        degeneracy_tolerance = 1e-12*scale
        starts = [0]+[i for i in range(1, len(ref))
                       if abs(ref.eigenvalue.iloc[i]-ref.eigenvalue.iloc[i-1]) > degeneracy_tolerance]
        starts.append(len(ref))
        for first, last in zip(starts[:-1], starts[1:]):
            if last-first <= 1:
                continue
            indices = list(range(first, last))
            spectrum_ok = bool(np.max(np.abs(got.eigenvalue.iloc[indices]
                                             - ref.eigenvalue.iloc[indices])) <= degeneracy_tolerance)
            basis_dependent_rows.update(indices)
            for name in ("charge_overlap", "axial_overlap"):
                actual = float(got[name].iloc[indices].sum())
                expected = float(ref[name].iloc[indices].sum())
                check = numeric_comparison(actual, expected, name, reference.name)
                subspace_comparisons.append(spectrum_ok and check["pass"])
                cells.append({"file": reference.name, "row": f"subspace:{first}:{last}",
                              "field": name, "actual": actual, "reference": expected,
                              **check, "criterion": "degenerate_subspace_projector_weight",
                              "required_for_scientific_match": True})
    raw_failures = 0
    for row in range(len(ref)):
        for name in numeric:
            check = numeric_comparison(float(got.iloc[row][name]),
                                       float(ref.iloc[row][name]), name, reference.name)
            raw_failures += not check["pass"]
            basis_dependent = row in basis_dependent_rows and name in {"charge_overlap", "axial_overlap"}
            if not basis_dependent:
                comparisons.append(check["pass"])
            cells.append({"file": reference.name, "row": row, "field": name,
                          "actual": float(got.iloc[row][name]),
                          "reference": float(ref.iloc[row][name]), **check,
                          "required_for_scientific_match": not basis_dependent})
    status = "PASS" if text_ok and all(comparisons) and all(subspace_comparisons) else "FAIL"
    return {"status": status, "rows": len(got), "max_abs": max_abs,
            "max_rel": max_rel, "detail": "all_cells_compared",
            "failed_numeric_cells": comparisons.count(False)+subspace_comparisons.count(False),
            "raw_numeric_cells_outside_tolerance": raw_failures,
            "basis_dependent_overlap_cells": 2*len(basis_dependent_rows)}


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, child in value.items():
            out.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        out = {}
        for index, child in enumerate(value):
            out.update(flatten(child, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def compare_json(generated: Path, reference: Path, cells: list[dict]) -> dict[str, object]:
    got = flatten(json.loads(generated.read_text()))
    ref = flatten(json.loads(reference.read_text()))
    if set(got) != set(ref):
        return {"status": "FAIL", "rows": len(got), "max_abs": None,
                "max_rel": None, "detail": "keys"}
    max_abs = max_rel = 0.0
    scalar_ok = True
    failures = 0
    for key, expected in ref.items():
        actual = got[key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            delta = abs(float(actual)-float(expected))
            max_abs = max(max_abs, delta)
            max_rel = max(max_rel, delta/max(abs(float(expected)), 1e-12))
            check = numeric_comparison(float(actual), float(expected), key, reference.name)
            failures += not check["pass"]
            cells.append({"file": reference.name, "row": 0, "field": key,
                          "actual": actual, "reference": expected, **check})
        else:
            scalar_ok &= actual == expected
    status = "PASS" if scalar_ok and not failures else "FAIL"
    return {"status": status, "rows": len(got), "max_abs": max_abs,
            "max_rel": max_rel, "detail": "all_leaves_compared",
            "failed_numeric_cells": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")
    package = output_root / "code" / "full_bz_completion"
    package.parent.mkdir(parents=True)
    shutil.copytree(HERE, package, ignore=shutil.ignore_patterns("__pycache__"))
    data = output_root / "data_prl" / "full_bz_completion"
    commands: list[list[str]] = []
    start = time.time()
    source_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                     for path in sorted(HERE.glob("*.py"))}
    reference_hashes = {path.name: sha(path) for path in sorted(REFERENCE.iterdir())
                        if path.suffix in {".csv", ".json"}}
    environment = os.environ.copy()
    environment.update({"TMP": str(output_root/"tmp"), "TEMP": str(output_root/"tmp"),
                        "MPLCONFIGDIR": str(output_root/"mplconfig"),
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1"})
    Path(environment["TMP"]).mkdir(parents=True)
    with (output_root/"reproduction.log").open("w", encoding="utf-8") as log:
        for script in ("generate_slab_data.py", "generate_torus_data.py", "generate_landau_data.py"):
            command = [sys.executable, str(package/script)]
            commands.append(command)
            print(f"START {script}", flush=True)
            result = subprocess.run(command, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(f"===== {script} exit={result.returncode} =====\n{result.stdout}\n")
            log.flush()
            if result.returncode:
                raise SystemExit(f"{script} failed; see reproduction.log")
            print(f"DONE {script}", flush=True)
        cache: dict[tuple[int, int], dict[str, float | int]] = {}
        raw_rows: list[dict] = []
        ky_rows = []
        for nky in (12, 16, 24, 32, 48, 64, 96):
            print(f"START convergence Lx=22 Nky={nky}", flush=True)
            row = convergence_row(SlabParams(Lx=22, Nky=nky, Nkz=36), raw_rows)
            pd.DataFrame(raw_rows).to_csv(output_root/"convergence_raw.csv", index=False)
            cache[(22, nky)] = row
            ky_rows.append({key: row[key] for key in (
                "Nky", "n2", "mu2", "K2_fixed_mu_real", "K2_fixed_n_real",
                "K2_chain_real", "mu_chain_relative_error", "K_chain_relative_error")})
            print(f"DONE convergence Lx=22 Nky={nky}", flush=True)
        pd.DataFrame(ky_rows).to_csv(data/"slab_ky_convergence.csv", index=False)
        size_rows = []
        for lx in (16, 22, 28):
            row = cache.get((lx, 96))
            if row is None:
                print(f"START convergence Lx={lx} Nky=96", flush=True)
                row = convergence_row(SlabParams(Lx=lx, Nky=96, Nkz=36), raw_rows)
                pd.DataFrame(raw_rows).to_csv(output_root/"convergence_raw.csv", index=False)
                print(f"DONE convergence Lx={lx} Nky=96", flush=True)
            size_rows.append({key: row[key] for key in (
                "Lx", "Nky", "Nkz", "n2", "mu2", "K2_fixed_mu_real",
                "K2_fixed_n_real", "K2_chain_real")})
        pd.DataFrame(size_rows).to_csv(data/"slab_size_convergence.csv", index=False)
        log.write("Convergence tables reconstructed from four-field intercept fits.\n")
    comparisons = []
    cells: list[dict] = []
    for reference in sorted(REFERENCE.iterdir()):
        if reference.suffix not in {".csv", ".json"}:
            continue
        generated = data/reference.name
        if not generated.exists():
            result = {"status": "FAIL", "rows": 0, "max_abs": None,
                      "max_rel": None, "detail": "missing"}
        elif reference.suffix == ".csv":
            result = compare_csv(generated, reference, cells)
        else:
            result = compare_json(generated, reference, cells)
        comparisons.append({"file": reference.name, **result})
    pd.DataFrame(cells).to_csv(output_root/"comparison_cells.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_root/"comparison.csv", index=False)
    metadata = {
        "status": "PASS" if all(row["status"] == "PASS" for row in comparisons) else "FAIL",
        "elapsed_seconds": time.time()-start, "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__,
        "reference": str(REFERENCE), "isolated_output_root": str(output_root),
        "unchanged_generator_commands": commands,
        "convergence_reconstruction": {
            "archived_aggregator_present": False, "fields": FIELDS.tolist(),
            "fit": "quadratic in B^2; intercept retained",
            "note": "A documented reconstruction, not recovery of the missing historical aggregator.",
            "ky_grids": [12, 16, 24, 32, 48, 64, 96],
            "size_grids": [[16, 96, 36], [22, 96, 36], [28, 96, 36]]},
        "source_sha256": source_hashes,
        "reference_sha256": reference_hashes,
        "sources_unchanged": all(sha(ROOT/path) == digest for path, digest in source_hashes.items()),
        "references_unchanged": all(sha(REFERENCE/path) == digest for path, digest in reference_hashes.items()),
        "comparison_criteria": "Per-cell abs+rel bounds are recorded in comparison_cells.csv and numeric_comparison; roundoff residuals use existing physics-validator absolute thresholds.",
        "degenerate_eigenspaces": "All eigenvalues and raw overlaps are compared and recorded. Only summed projector weights are required within clusters of eigenvalues separated by <=1e-12*max(1,spectral_radius). Individual eigenvectors within a degenerate subspace have no unique physical labels.",
        "generated_sha256": {str(path.relative_to(output_root)).replace("\\", "/"): sha(path)
                             for path in sorted(data.iterdir()) if path.is_file()},
        "comparisons": comparisons}
    if not metadata["sources_unchanged"] or not metadata["references_unchanged"]:
        metadata["status"] = "FAIL"
    (output_root/"reproduction_metadata.json").write_text(
        json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": metadata["status"],
                      "elapsed_seconds": metadata["elapsed_seconds"],
                      "files_compared": len(comparisons)}, indent=2))
    if metadata["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
