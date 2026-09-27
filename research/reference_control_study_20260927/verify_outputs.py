"""Read-only artifact checks; writes only its verification report."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import study
import wide_study
from analyze import read_csv

ROOT = Path(__file__).resolve().parent


def main():
    low = np.load(ROOT/"run_01/observations.npz")
    wide = np.load(ROOT/"run_03_wide/observations.npz")
    archived_wide = np.load(ROOT/"run_02_wide/observations.npz")
    assert len(low["train"]) == len(wide["train"]) == 81
    assert len(wide["hold"]) == 79
    assert np.all((wide["hold"] < 1.15) | (wide["hold"] > 2.4))
    checked_noise = checked_residual = 0
    max_chi2_relative_error = max_common_noise_error = 0.
    current_source_differences = []
    for run, data in (("run_01", low), ("run_02_wide", archived_wide), ("run_03_wide", wide)):
        report = json.loads((ROOT/run/"validation.json").read_text())
        for rel, value in report["source_hashes"].items():
            current_hash = hashlib.sha256((study.REPO/rel).read_bytes()).hexdigest()
            assert hashlib.sha256((ROOT/run/"source_snapshot"/Path(rel).name).read_bytes()).hexdigest() == value, rel
            if current_hash != value:
                assert run == "run_02_wide" and Path(rel).name == "wide_study.py", rel
                current_source_differences.append(dict(run=run, source=rel, reason="output CLI only; scientific tail and outputs checked identical"))
        for row in read_csv(ROOT/run/"fits.csv"):
            prefix = f"case_{row['case_id']}_"
            p = np.array(json.loads(row["parameters"]))
            residual = study.residual(p, data["train"], data[prefix+"signal"], data[prefix+"reference"], float(row["sigma"]))
            recomputed = float(residual@residual)
            error = abs(recomputed-float(row["chi2"]))/max(1, recomputed)
            max_chi2_relative_error = max(error, max_chi2_relative_error)
            assert error < 1e-10
            held = study.residual(p, data["hold"], data[prefix+"held_signal"], data[prefix+"held_reference"], float(row["sigma"]))
            np.testing.assert_allclose(held@held/len(held), float(row["holdout_chi2_per_real_sample"]), rtol=1e-11)
            checked_residual += 1
    for case_id, (field, gamma, _, control, _) in enumerate(study.case_design()):
        if control != "extra_link":
            continue
        low_clean = study.truth_spectra(low["train"], field, gamma, control)
        wide_clean = study.truth_spectra(wide["train"], field, gamma, control)
        for name, l_clean, w_clean in zip(("signal", "reference"), low_clean, wide_clean):
            a = low[f"case_{case_id}_{name}"]-l_clean
            b = wide[f"case_{case_id}_{name}"]-w_clean
            max_common_noise_error = max(max_common_noise_error, float(np.max(abs(a-b))))
            np.testing.assert_allclose(a, b, rtol=0, atol=5e-13)
            checked_noise += 1
    # The archival run is untouched; the new entrypoint changes only output selection.
    for name in ("fits.csv", "fit_starts.csv"):
        assert (ROOT/"run_02_wide"/name).read_bytes() == (ROOT/"run_03_wide"/name).read_bytes(), name
    assert set(wide.files) == set(archived_wide.files)
    for name in wide.files:
        np.testing.assert_array_equal(wide[name], archived_wide[name])
    old_ast = ast.parse((ROOT/"run_02_wide/source_snapshot/wide_study.py").read_text())
    new_ast = ast.parse((ROOT/"wide_study.py").read_text())
    old_funcs = {n.name: n for n in old_ast.body if isinstance(n, ast.FunctionDef)}
    new_funcs = {n.name: n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
    assert ast.dump(old_funcs["summarize"]) == ast.dump(new_funcs["summarize"])
    def scientific_tail(function):
        index = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "original")
        return [ast.dump(n) for n in function.body[index:]]
    assert scientific_tail(old_funcs["main"]) == scientific_tail(new_funcs["main"])
    # Evaluate the closed expression at remote frequencies against the dense generator.
    matrix, _, _ = study.physical_setup(study.Q0, .2, .009)
    p = np.array([matrix[0, 0], np.log(.003), 2.6**2, np.log(.007), 1.3*study.Q0,
                  np.log(.009), np.log(.16**2), .008, np.log(.005), .04])
    for closed, dense in zip(study.prediction(wide_study.TRAIN, p), study.truth_spectra(wide_study.TRAIN, .2, .009, "extra_link")):
        np.testing.assert_allclose(closed, dense, rtol=2e-12, atol=2e-12)
    result = dict(status="PASS", recomputed_fit_and_holdout_objectives=checked_residual,
                  common_training_noise_vectors_checked=checked_noise,
                  max_common_training_noise_difference=max_common_noise_error,
                  max_training_chi2_relative_reconstruction_error=max_chi2_relative_error,
                  wide_design_dense_inverse_check=True, all_snapshot_source_hashes_match=True,
                  current_source_differences=current_source_differences,
                  current_sources_match_run_01_and_run_03=True,
                  wide_rerun_scientific_tables_byte_identical=True,
                  wide_rerun_observation_arrays_identical=True,
                  wide_driver_scientific_function_and_main_tail_ast_identical=True)
    (ROOT/"verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
