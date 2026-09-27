"""Same-budget remote-resonance acquisition; preserve original study and run."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

import study as base

ROOT = Path(__file__).resolve().parent
TRAIN = np.r_[np.linspace(.79, 1.15, 61), np.linspace(2.40, 2.80, 20)]
HOLD = np.r_[(TRAIN[:60]+TRAIN[1:61])/2, (TRAIN[61:-1]+TRAIN[62:])/2]


def summarize(solved, case, held_signal, held_reference, clean_signal, clean_reference, sigma):
    p = solved["parameters"]
    jac = solved["jacobian"]
    singular = np.linalg.svd(jac, compute_uv=False)
    norms = np.linalg.norm(jac, axis=0)
    normalized = np.linalg.svd(jac/np.where(norms > 0, norms, 1), compute_uv=False)
    held = base.residual(p, HOLD, held_signal, held_reference, sigma)
    clean = base.residual(p, HOLD, clean_signal, clean_reference, sigma)
    gamma, coupling = np.exp(p[[5, 6]])
    return dict(**case, **base.fit_record(solved),
        training_chi2_per_dof=solved["chi2"]/(4*len(TRAIN)-len(p)),
        holdout_chi2_per_real_sample=float(held@held/len(held)),
        holdout_clean_scaled_mse=float(clean@clean/len(clean)),
        holdout_two_sigma_reference=float(1+2*np.sqrt(2/len(held))),
        holdout_above_reference=bool(held@held/len(held) > 1+2*np.sqrt(2/len(held))),
        inferred_omega=float(p[4]), inferred_gamma=float(gamma), inferred_t=float(coupling),
        omega_error=float(p[4]-1.3*base.Q0), gamma_error=float(gamma-case["gamma_dark"]),
        relative_gamma_error=float(gamma/case["gamma_dark"]-1),
        relative_t_error=float(coupling/(.8*case["B"])**2-1),
        inferred_u=float(p[7]), inferred_v=float(np.exp(p[8])-np.exp(p[1])),
        inferred_link=float(p[9]) if len(p)==10 else 0.,
        jacobian_condition=float(singular[0]/singular[-1]),
        normalized_jacobian_condition=float(normalized[0]/normalized[-1]),
        singular_values=json.dumps(singular.tolist()), normalized_singular_values=json.dumps(normalized.tolist()))


def main():
    output = ROOT/"run_02_wide"
    if output.exists():
        raise ValueError("Refusing to overwrite preserved wide run")
    output.mkdir()
    original = json.loads((ROOT/"run_01/validation.json").read_text())
    sources = [base.REPO/path for path in original["source_hashes"]]
    sources += [ROOT/"wide_study.py", ROOT/"wide_plan.md", ROOT/"run_01/validation.json"]
    hashes = {str(p.relative_to(base.REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for rel, expected in original["source_hashes"].items():
        assert hashes[rel] == expected, (rel, "original source changed")
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for source in sources:
        shutil.copyfile(source, snapshot/source.name)
    scale = original["reference_rms_scale"]
    started = time.monotonic()
    fits, starts, saved = [], [], {}
    for case_id, (field, gamma, noise, control, replica) in enumerate(base.case_design()):
        if control != "extra_link":
            continue
        rng = np.random.default_rng(2026092710+37*case_id)
        sigma = scale*noise
        signal_clean, ref_clean = base.truth_spectra(TRAIN, field, gamma, control)
        held_signal_clean, held_ref_clean = base.truth_spectra(HOLD, field, gamma, control)
        noisy = lambda values: values+sigma*(rng.normal(size=len(values))+1j*rng.normal(size=len(values)))
        signal, reference = noisy(signal_clean), noisy(ref_clean)
        held_signal, held_reference = noisy(held_signal_clean), noisy(held_ref_clean)
        for key, values in (("signal", signal), ("reference", reference), ("held_signal", held_signal), ("held_reference", held_reference)):
            saved[f"case_{case_id}_{key}"] = values
        case = dict(case_id=case_id, B=field, gamma_dark=gamma, noise_scale=noise,
                    control=control, replica=replica, seed=2026092710+37*case_id, sigma=sigma,
                    training_complex_samples=162, holdout_complex_samples=158, design="remote_inclusive")
        for model, size in (("drift_aware", 9), ("extended", 10)):
            candidates = []
            for index, initial in enumerate(base.STARTS):
                solved = base.fit(initial[:size], TRAIN, signal, reference, sigma)
                candidates.append(solved)
                starts.append(dict(**case, model=model, start_index=index, **base.fit_record(solved)))
            best = min(candidates, key=lambda x: x["chi2"])
            record = summarize(best, dict(**case, model=model, model_correct=(model=="extended")),
                               held_signal, held_reference, held_signal_clean, held_ref_clean, sigma)
            record["successful_unrestricted_starts"] = sum(c["success"] for c in candidates)
            fits.append(record)
        print(f"wide cases {len(fits)//2}/16, elapsed {time.monotonic()-started:.2f}s", flush=True)
    base.write_csv(output/"fits.csv", fits)
    base.write_csv(output/"fit_starts.csv", starts)
    np.savez_compressed(output/"observations.npz", train=TRAIN, hold=HOLD, **saved)
    assert hashes == {str(p.relative_to(base.REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = dict(status="COMPLETED", cases=16, fits=len(fits), start_attempts=len(starts),
                  elapsed_seconds=time.monotonic()-started, reference_rms_scale=scale,
                  real_component_sigmas=[scale*1e-4, scale*1e-3],
                  source_hashes=hashes, all_sources_unchanged=True,
                  training_complex_samples=162, heldout_complex_samples=158,
                  failures_retained=sum(not r["success"] for r in fits),
                  boundary_fits_retained=sum(bool(r["boundaries"]) for r in fits),
                  acquisition_assumption="same number of equally costly complex points and frequency-independent absolute noise",
                  scope="finite oscillator inverse problem; no material validation or identification certificate")
    (output/"validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
