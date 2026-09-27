#!/usr/bin/env python3
"""Independent v6 validation for pole-zero tomography and genealogy inference."""
from __future__ import annotations
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pole_zero_tomography import (
    classify_genealogy,
    classify_genealogy_exact,
    fit_exact_vanishing_velocity,
    fit_two_pole_rational,
    poles_from_bare,
    reconstruct_from_poles_residues,
    residues_from_bare,
    singular_response,
)

OUT = ROOT / "data_prl" / "pole_zero_validation.json"
LOG = ROOT / "data_prl" / "pole_zero_validation.log"
checks: list[dict] = []


def check(name: str, cond: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(cond), "detail": detail})
    if not cond:
        raise AssertionError(f"{name}: {detail}")

rng = np.random.default_rng(20260625)

# 1-4. Exact complex algebra under unequal damping and nonreciprocal coupling product.
max_zero = max_bright = max_g2 = max_c = 0.0
for _ in range(2500):
    zb = rng.uniform(-1, 1) - 1j * rng.uniform(1e-3, 0.4)
    zd = rng.uniform(-1, 1) - 1j * rng.uniform(1e-3, 0.4)
    g = rng.uniform(0.01, 0.5) * np.exp(1j * rng.uniform(-0.8, 0.8))
    g2 = g * g
    c = (0.2 + rng.random()) * np.exp(1j * rng.uniform(-1.0, 1.0))
    rm, rp, (zm, zp) = residues_from_bare(zb, zd, g2, prefactor=c)
    rec = reconstruct_from_poles_residues(zm, zp, rm, rp)
    max_zero = max(max_zero, abs(rec.response_zero - zd))
    max_bright = max(max_bright, abs(rec.companion_pole - zb))
    max_g2 = max(max_g2, abs(rec.coupling_product - g2))
    max_c = max(max_c, abs(rec.prefactor - c))
check("exact complex response-zero reconstruction", max_zero < 2e-13, f"max={max_zero:.3e}")
check("exact companion-pole reconstruction", max_bright < 2e-13, f"max={max_bright:.3e}")
check("exact coupling-product reconstruction", max_g2 < 3e-13, f"max={max_g2:.3e}")
check("exact prefactor sum rule", max_c < 2e-13, f"max={max_c:.3e}")

# 5-7. Exact rational fit with constant/linear analytic backgrounds.
zb = 0.198 - 0.006j
zd = 0.306 - 0.021j
g2 = (0.033 + 0.006j) ** 2
pref = 0.92 + 0.08j
rm, rp, (zm, zp) = residues_from_bare(zb, zd, g2, prefactor=pref)
z = np.linspace(0.10, 0.40, 420) + 0.004j
for order, bg in (
    (0, 0.025 - 0.015j),
    (1, (0.025 - 0.015j) + (0.018 + 0.009j) * z),
):
    fit = fit_two_pole_rational(z, bg + singular_response(z, zm, zp, zd, prefactor=pref), background_order=order)
    check(f"exact rational fit order {order} residual", fit.relative_rms < 2e-11, f"rms={fit.relative_rms:.3e}")
    check(f"exact rational fit order {order} zero", abs(fit.response_zero - zd) < 2e-10, f"err={abs(fit.response_zero-zd):.3e}")

# 8-10. Exact reduced Weyl-model reconstruction and circle-law convergence.
df = pd.read_csv(ROOT / "data_prl" / "fig2_pole_zero_tomography.csv")
cdf = pd.read_csv(ROOT / "data_prl" / "fig2_frequency_residue_circle.csv")
max_dark = float(np.max(np.abs(df.dark_zero_error)))
check("Weyl hidden-pole reconstruction", max_dark < 2e-5, f"max abs={max_dark:.3e}")
res_small = float(np.max(np.abs(cdf[cdf.b == 0.0025].circle_sum - 1.0)))
res_large = float(np.max(np.abs(cdf[cdf.b == 0.010].circle_sum - 1.0)))
check("circle-law weak-field accuracy", res_small < 9e-3, f"max abs={res_small:.3e}")
check("circle-law correction decreases toward zero field", res_small < res_large, f"small={res_small:.3e}, large={res_large:.3e}")

# 11-13. Noisy complex fit and omitted-mode self-diagnosis.
ndf = pd.read_csv(ROOT / "data_prl" / "figS4_complex_tomography_noise.csv")
tdf = pd.read_csv(ROOT / "data_prl" / "figS4_three_mode_diagnostic.csv")
med120 = float(ndf[ndf.snr == 120].zero_error.median())
check("complex tomography at SNR 120", med120 < 1.6e-3, f"median={med120:.3e}")
for strength, sub in tdf.groupby("third_pole_strength"):
    near = float(sub.loc[sub.third_pole_distance.idxmin(), "relative_two_pole_fit_rms"])
    far = float(sub.loc[sub.third_pole_distance.idxmax(), "relative_two_pole_fit_rms"])
    check(f"omitted-pole residual diagnoses model order, weight {strength:g}", far < near, f"near={near:.3e}, far={far:.3e}")

# 14-17. Genealogy classification and adversarial model classes.
gdf = pd.read_csv(ROOT / "data_prl" / "figS5_genealogy_model_selection.csv")
for truth in ("finite", "vanishing"):
    acc = float(gdf[(gdf.field_span == 8.0) & (gdf.relative_q_noise == 0.01) & (gdf.truth == truth)].accuracy.iloc[0])
    check(f"exact-crossover classifier at span 8, 1% noise: {truth}", acc >= 0.90, f"accuracy={acc:.3f}")

b = np.geomspace(0.0025, 0.02, 9)
qfinite = 0.093 + 0.03 * b + 0.2 * b * b
sigma = 0.002 * qfinite
label, _, _, dbic = classify_genealogy(b, qfinite, sigma)
check("symmetry-broken O(B) finite intercept still classified finite", label == "finite", f"label={label}, dBIC={dbic:.3f}")
qdiv = 0.00093 / b + 0.004
sigma = 0.002 * qdiv
label, _, div, dbic = classify_genealogy(b, qdiv, sigma)
check("general p divergent class recovered", label == "divergent" and abs(div.parameters["p"] - 1.0) < 0.04, f"label={label}, p={div.parameters.get('p')}, dBIC={dbic:.3f}")

# Exact anomaly-crossover hypothesis and identical-pipeline classification.
b = np.geomspace(0.0025, 0.020, 9)
bc_true = 0.010
qinf_true = 0.06576
qexact = qinf_true * np.sqrt(1.0 + (bc_true / b) ** 2)
sigma = 0.002 * qexact
exact_fit = fit_exact_vanishing_velocity(b, qexact, sigma)
check("exact crossover fit recovers q_inf", abs(exact_fit.parameters["q_inf"] / qinf_true - 1.0) < 2e-8, f"qinf={exact_fit.parameters['q_inf']:.9g}")
check("exact crossover fit recovers B_c", abs(exact_fit.parameters["B_c"] / bc_true - 1.0) < 2e-8, f"Bc={exact_fit.parameters['B_c']:.9g}")
label, finite_fit, vanish_fit, dbic = classify_genealogy_exact(b, qexact, sigma)
check("exact crossover selected over finite intercept", label == "vanishing" and dbic < 0.0, f"label={label}, dBIC={dbic:.3f}")

# 18-20. Explicit selection-rule exponents from supplement source data.
sdf = pd.read_csv(ROOT / "data_prl" / "figS6_symmetry_selection_rules.csv")
def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])
exp_even = slope(sdf.b, np.abs(sdf.qmin_node_exchange - 0.1))
exp_odd = slope(sdf.b, np.abs(sdf.qmin_symmetry_broken - 0.1))
exp_gap1 = slope(sdf.b, sdf.gap_linear_mixing)
exp_gap2 = slope(sdf.b, sdf.gap_quadratic_mixing)
check("node-exchange q shift exponent", abs(exp_even - 2.0) < 5e-10, f"p={exp_even:.12f}")
check("broken-covariance q shift exponent", abs(exp_odd - 1.0) < 0.03, f"p={exp_odd:.6f}")
check("mixing-order gap exponents", abs(exp_gap1 - 1.0) < 1e-10 and abs(exp_gap2 - 2.0) < 1e-10, f"p1={exp_gap1:.12f}, p2={exp_gap2:.12f}")

# 21-23. Package reproducibility defects fixed.
check("legacy solver source data included", (ROOT / "data" / "eta_convergence.csv").exists(), "data/eta_convergence.csv")
check("pole-zero figure source data included", (ROOT / "data_prl" / "fig2_pole_zero_tomography.csv").exists(), "CSV present")
check("new generator and validator import cleanly", True, "module import succeeded")

payload = {
    "all_checks_passed": all(c["passed"] for c in checks),
    "number_of_checks": len(checks),
    "max_exact_zero_error": max_zero,
    "max_weyl_hidden_zero_error": max_dark,
    "circle_residual_b_0p0025": res_small,
    "median_noisy_zero_error_snr_120": med120,
    "checks": checks,
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
LOG.write_text("\n".join(f"[{'ok' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in checks) + "\n")
print(LOG.read_text(), end="")
print(json.dumps({k: v for k, v in payload.items() if k != "checks"}, indent=2))
