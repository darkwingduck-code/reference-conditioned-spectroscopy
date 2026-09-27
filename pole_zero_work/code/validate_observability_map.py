#!/usr/bin/env python3
"""Validate the pole-zero tomography observability benchmark and artifacts."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_prl"
FIG = ROOT / "fig_prl"
CSV = DATA / "figS8_tomography_observability_map.csv"
SUMMARY = DATA / "tomography_observability_summary.json"
OUT = DATA / "observability_validation.json"
LOG = DATA / "observability_validation.log"

checks: list[dict] = []


def check(name: str, condition: bool, detail: str) -> None:
    item = {"name": name, "passed": bool(condition), "detail": detail}
    checks.append(item)
    if not condition:
        raise AssertionError(f"{name}: {detail}")


df = pd.read_csv(CSV)
required = {
    "gap_to_sum_halfwidth_ratio",
    "complex_response_snr",
    "trials",
    "fit_failures",
    "median_zero_error_over_gamma",
    "p90_zero_error_over_gamma",
    "median_pole_error_over_gamma",
    "probability_zero_error_below_0p2gamma",
    "median_design_condition_number",
}
check("observability table has complete schema", required.issubset(df.columns), f"columns={list(df.columns)}")
check("observability grid contains 9x9 points", len(df) == 81 and df.gap_to_sum_halfwidth_ratio.nunique() == 9 and df.complex_response_snr.nunique() == 9, f"shape={df.shape}")
check("all Monte Carlo fits completed", int(df.fit_failures.max()) == 0, f"max failures={int(df.fit_failures.max())}")
check("all reported errors and condition numbers are finite", np.isfinite(df[list(required - {"trials", "fit_failures"})].to_numpy(float)).all(), "finite numeric table")
check("probabilities lie in [0,1]", bool(((df.probability_zero_error_below_0p2gamma >= 0) & (df.probability_zero_error_below_0p2gamma <= 1)).all()), f"range={df.probability_zero_error_below_0p2gamma.min():.3g},{df.probability_zero_error_below_0p2gamma.max():.3g}")

# Endpoint trends: increasing SNR helps at fixed small and moderate splitting.
for ratio in sorted(df.gap_to_sum_halfwidth_ratio.unique())[:5]:
    sub = df[df.gap_to_sum_halfwidth_ratio == ratio].sort_values("complex_response_snr")
    check(
        f"higher SNR improves zero inference at gap ratio {ratio:.3g}",
        sub.iloc[-1].median_zero_error_over_gamma < sub.iloc[0].median_zero_error_over_gamma,
        f"low/high={sub.iloc[0].median_zero_error_over_gamma:.4g}/{sub.iloc[-1].median_zero_error_over_gamma:.4g}",
    )

# At high SNR, a resolved interior splitting beats the unresolved edge.
snr_max = float(df.complex_response_snr.max())
sub = df[df.complex_response_snr == snr_max]
low_gap = sub.loc[sub.gap_to_sum_halfwidth_ratio.idxmin()]
interior = sub.loc[sub.median_zero_error_over_gamma.idxmin()]
check(
    "resolved interior splitting outperforms unresolved poles at high SNR",
    interior.median_zero_error_over_gamma < 0.01 * low_gap.median_zero_error_over_gamma,
    f"best/low-gap={interior.median_zero_error_over_gamma:.3e}/{low_gap.median_zero_error_over_gamma:.3e}",
)
check(
    "well-resolved high-SNR points achieve high success probability",
    float(sub[sub.gap_to_sum_halfwidth_ratio.between(1.1, 3.9)].probability_zero_error_below_0p2gamma.min()) >= 0.95,
    f"minimum success={sub[sub.gap_to_sum_halfwidth_ratio.between(1.1, 3.9)].probability_zero_error_below_0p2gamma.min():.3f}",
)
check(
    "unresolved low-SNR point fails the stringent zero criterion",
    float(df.sort_values(["gap_to_sum_halfwidth_ratio", "complex_response_snr"]).iloc[0].probability_zero_error_below_0p2gamma) <= 0.1,
    f"success={df.sort_values(['gap_to_sum_halfwidth_ratio','complex_response_snr']).iloc[0].probability_zero_error_below_0p2gamma:.3f}",
)

summary = json.loads(SUMMARY.read_text())
check("summary trial count matches table", int(summary["trials_per_grid_point"]) == int(df.trials.iloc[0]), f"summary/table={summary['trials_per_grid_point']}/{df.trials.iloc[0]}")
check("summary best error matches table", abs(float(summary["best_median_zero_error_over_gamma"]) - float(df.median_zero_error_over_gamma.min())) < 1e-14, f"delta={abs(float(summary['best_median_zero_error_over_gamma'])-float(df.median_zero_error_over_gamma.min())):.3e}")
for name in ("figS8_tomography_observability_map.pdf", "figS8_tomography_observability_map.png"):
    check(f"observability figure artifact exists: {name}", (FIG / name).exists() and (FIG / name).stat().st_size > 1000, name)

payload = {
    "all_checks_passed": all(c["passed"] for c in checks),
    "number_of_checks": len(checks),
    "best_median_zero_error_over_gamma": float(df.median_zero_error_over_gamma.min()),
    "worst_median_zero_error_over_gamma": float(df.median_zero_error_over_gamma.max()),
    "checks": checks,
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
LOG.write_text("\n".join(f"[{'ok' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in checks) + "\n")
print(LOG.read_text(), end="")
print(json.dumps({k: v for k, v in payload.items() if k != "checks"}, indent=2))
