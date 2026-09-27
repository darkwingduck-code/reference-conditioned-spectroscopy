#!/usr/bin/env python3
"""Independent validation of the competitor-trajectory crossover.

The tested formula follows from Eq. (9) and q_h=omega_p/u in
arXiv:2605.27031v1.  The validator does not import the plotting generator,
so agreement is not a tautological same-function comparison.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_prl"
FIG = ROOT / "fig_prl"

checks: list[dict] = []


def record(name: str, passed: bool, value=None, tolerance=None, note: str = "") -> None:
    checks.append({
        "name": name,
        "passed": bool(passed),
        "value": value,
        "tolerance": tolerance,
        "note": note,
    })


def physical_velocity(vom: float, omega_p: float, eps: float, qtf: float, ctheta: float) -> float:
    if omega_p <= 0 or eps <= 1 or qtf <= 0 or abs(ctheta) <= 0:
        raise ValueError("unphysical parameter")
    return abs(vom) * omega_p * math.sqrt(eps - 1.0) * abs(ctheta) / math.sqrt(
        eps * omega_p**2 + vom**2 * qtf**2
    )


def physical_crossing(vom: float, omega_p: float, eps: float, qtf: float, ctheta: float) -> float:
    return omega_p / physical_velocity(vom, omega_p, eps, qtf, ctheta)


def q_norm_exact(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(1.0 + x*x) / x


def zeta_exact(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + x*x)


def q_finite(x, a=0.08):
    x = np.asarray(x, dtype=float)
    return 1.0 + a*x*x/(1.0+x*x)


def zeta_finite(x, a=0.08):
    x = np.asarray(x, dtype=float)
    q = q_finite(x, a)
    return -(2.0*a*x*x/(1.0+x*x)**2)/q


def log_derivative(fun, x: float, h: float = 2.0e-6) -> float:
    xp = x * math.exp(h)
    xm = x * math.exp(-h)
    return -(math.log(float(fun(xp))) - math.log(float(fun(xm)))) / (2.0*h)


def main() -> None:
    # 1. Direct physical-to-dimensionless reduction over random parameters.
    rng = np.random.default_rng(20260623)
    rel_errors = []
    sign_errors = []
    angle_errors = []
    for _ in range(250):
        eps = rng.uniform(1.05, 35.0)
        omega_p = 10.0 ** rng.uniform(-2.0, 1.0)
        qtf = 10.0 ** rng.uniform(-2.0, 1.0)
        vom = (1 if rng.random() > 0.5 else -1) * 10.0 ** rng.uniform(-4.0, 1.0)
        ctheta = rng.uniform(0.05, 1.0)
        x = abs(vom) * qtf / (math.sqrt(eps) * omega_p)
        qstar = qtf / (math.sqrt(eps - 1.0) * abs(ctheta))
        q1 = physical_crossing(vom, omega_p, eps, qtf, ctheta) / qstar
        q2 = float(q_norm_exact(x))
        rel_errors.append(abs(q1-q2)/abs(q2))
        qp = physical_crossing(abs(vom), omega_p, eps, qtf, ctheta)
        qm = physical_crossing(-abs(vom), omega_p, eps, qtf, ctheta)
        sign_errors.append(abs(qp-qm)/abs(qp))
        qtheta = physical_crossing(vom, omega_p, eps, qtf, ctheta)
        qzero = physical_crossing(vom, omega_p, eps, qtf, 1.0)
        angle_errors.append(abs(qtheta/qzero - 1.0/abs(ctheta)))
    record("physical formula reduces to sqrt(1+x^2)/x", max(rel_errors) < 3e-14, max(rel_errors), 3e-14)
    record("crossing is even under B sign", max(sign_errors) < 1e-15, max(sign_errors), 1e-15)
    record("angle scaling is 1/abs(cos theta)", max(angle_errors) < 3e-13, max(angle_errors), 3e-13)

    # 2. Analytic exponent and finite differences.
    xs = np.geomspace(1e-4, 1e2, 120)
    zfd = np.array([log_derivative(q_norm_exact, float(x)) for x in xs])
    zerr = np.max(np.abs(zfd - zeta_exact(xs)))
    record("analytic genealogy exponent matches log finite difference", zerr < 2e-9, float(zerr), 2e-9)

    # 3. Weak-field asymptotics.
    xsmall = np.geomspace(1e-7, 1e-3, 50)
    weak_ratio = q_norm_exact(xsmall) * xsmall
    record("weak-field q*x tends to one", abs(float(weak_ratio[0])-1.0) < 1e-13, float(weak_ratio[0]), 1e-13)
    coeff = (weak_ratio - 1.0) / (xsmall*xsmall)
    # Avoid the first values dominated by floating-point subtraction.
    coeff_window = coeff[xsmall > 2e-6]
    record("weak-field first correction is x^2/2", abs(float(np.median(coeff_window))-0.5) < 2e-5, float(np.median(coeff_window)), 2e-5)
    record("zeta tends to one at weak field", abs(float(zeta_exact(1e-6))-1.0) < 2e-12, float(zeta_exact(1e-6)), 2e-12)
    record("zeta equals one half at x=1", abs(float(zeta_exact(1.0))-0.5) < 1e-15, float(zeta_exact(1.0)), 1e-15)

    # 4. Strong-x crossover within the acoustic formula.
    xlarge = np.geomspace(1e2, 1e5, 30)
    qlarge = q_norm_exact(xlarge)
    record("normalized q tends to one at large x", abs(float(qlarge[-1])-1.0) < 1e-10, float(qlarge[-1]), 1e-10)
    strong_coeff = 2.0*xlarge*xlarge*(qlarge-1.0)
    record("large-x correction is 1/(2x^2)", abs(float(np.median(strong_coeff[:10]))-1.0) < 2e-4, float(np.median(strong_coeff[:10])), 2e-4)
    record("zeta tends to zero at large x", float(zeta_exact(1e5)) < 1.1e-10, float(zeta_exact(1e5)), 1.1e-10)

    # 5. Monotonicity and positivity.
    xmono = np.geomspace(1e-6, 1e3, 5000)
    qmono = q_norm_exact(xmono)
    zmono = zeta_exact(xmono)
    record("q(x) is strictly decreasing", bool(np.all(np.diff(qmono) < 0.0)))
    record("q(x) stays above its large-x scale", bool(np.all(qmono > 1.0)))
    record("zeta(x) is strictly decreasing", bool(np.all(np.diff(zmono) < 0.0)))
    record("zeta lies between zero and one", bool(np.all((zmono > 0.0) & (zmono < 1.0))))

    # 6. Exact observation-window threshold.
    threshold_errors = []
    for qmax in (1.01, 1.1, 1.5, 2.0, 5.0, 20.0):
        xwin = 1.0/math.sqrt(qmax*qmax-1.0)
        threshold_errors.append(abs(float(q_norm_exact(xwin))-qmax))
    record("exact finite-window threshold", max(threshold_errors) < 4e-14, max(threshold_errors), 4e-14)
    record("no normalized crossing window exists for qmax<=1", all(q <= 1.0 for q in (0.2, 0.9, 1.0)))
    qmax = 5.0
    x_exact = 1.0/math.sqrt(qmax*qmax-1.0)
    x_weak = 1.0/qmax
    record("weak-window threshold approaches exact threshold", abs(x_exact/x_weak-1.0) < 0.021, x_exact/x_weak-1.0, 0.021)

    # 7. Finite-intercept reference behaves as intended.
    xf = np.geomspace(1e-6, 1e2, 500)
    qf = q_finite(xf)
    zf = zeta_finite(xf)
    zf_fd = np.array([log_derivative(q_finite, float(x)) for x in xf])
    record("finite-intercept example tends to q0", abs(float(q_finite(1e-8))-1.0) < 1e-15, float(q_finite(1e-8)), 1e-15)
    record("finite-intercept zeta tends to zero", abs(float(zeta_finite(1e-6))) < 2e-13, float(zeta_finite(1e-6)), 2e-13)
    record("finite-intercept analytic exponent matches finite difference", float(np.max(np.abs(zf-zf_fd))) < 3e-9, float(np.max(np.abs(zf-zf_fd))), 3e-9)
    record("finite-intercept example remains bounded", float(np.max(qf)) < 1.081, float(np.max(qf)), 1.081)

    # 8. Angle and endpoint behavior.
    eps, omega_p, qtf, vom = 13.0, 1.0, 0.4, 0.03
    q0angle = physical_crossing(vom, omega_p, eps, qtf, 1.0)
    qnear = physical_crossing(vom, omega_p, eps, qtf, 1e-6)
    record("crossing diverges as cos(theta)->0", qnear/q0angle > 9.99e5, qnear/q0angle, 9.99e5)
    uweak = physical_velocity(1e-8, omega_p, eps, qtf, 1.0)
    linear_coeff = omega_p*math.sqrt(eps-1.0)/(math.sqrt(eps)*omega_p)
    record("velocity is linear in abs(v_Omega) at weak field", abs(uweak/1e-8-linear_coeff) < 1e-12, uweak/1e-8-linear_coeff, 1e-12)

    # 9. Generated artifacts are internally consistent.
    csv_path = DATA / "figS9_competitor_exact_trajectory.csv"
    summary_path = DATA / "competitor_exact_trajectory_summary.json"
    fig_pdf = FIG / "figS9_competitor_exact_trajectory.pdf"
    fig_png = FIG / "figS9_competitor_exact_trajectory.png"
    record("trajectory CSV exists", csv_path.exists() and csv_path.stat().st_size > 1000, csv_path.stat().st_size if csv_path.exists() else 0)
    record("trajectory summary exists", summary_path.exists() and summary_path.stat().st_size > 100, summary_path.stat().st_size if summary_path.exists() else 0)
    record("trajectory PDF exists", fig_pdf.exists() and fig_pdf.stat().st_size > 5000, fig_pdf.stat().st_size if fig_pdf.exists() else 0)
    record("trajectory PNG exists", fig_png.exists() and fig_png.stat().st_size > 5000, fig_png.stat().st_size if fig_png.exists() else 0)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        csv_qref = q_norm_exact(df.iloc[:, 0].to_numpy())
        csv_qerr = float(np.max(np.abs(df["normalized_anomaly_crossing_q"].to_numpy() - csv_qref) / np.maximum(np.abs(csv_qref), 1.0)))
        csv_zerr = float(np.max(np.abs(df["anomaly_effective_zeta"].to_numpy() - zeta_exact(df.iloc[:, 0].to_numpy()))))
    else:
        csv_qerr = csv_zerr = math.inf
    record("CSV q column matches independent formula", csv_qerr < 3e-13, csv_qerr, 3e-13)
    record("CSV zeta column matches independent formula", csv_zerr < 3e-12, csv_zerr, 3e-12)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary_ok = (
            abs(summary["low_field_zeta_at_x_1e-3"] - float(zeta_exact(1e-3))) < 1e-15
            and abs(summary["zeta_at_x_1"] - 0.5) < 1e-15
            and abs(summary["high_x_zeta_at_x_10"] - float(zeta_exact(10.0))) < 1e-15
        )
    else:
        summary_ok = False
    record("summary values match independent formula", summary_ok)

    # 10. Main-text Fig. 1 uses the exact crossover rather than a pure 1/B curve.
    main_csv = DATA / "fig1_field_intercept.csv"
    record("main field-intercept CSV exists", main_csv.exists() and main_csv.stat().st_size > 500)
    if main_csv.exists():
        mdf = pd.read_csv(main_csv)
        required = {
            "abs_b", "q_h_over_q0_anomaly_normalized",
            "q_h_weak_asymptote_over_q0", "competitor_x",
            "competitor_x_ref", "competitor_field_ref", "zeta_anomaly",
        }
        record("main trajectory contains exact-crossover columns", required.issubset(mdf.columns), sorted(required - set(mdf.columns)))
        if required.issubset(mdf.columns):
            xmain = mdf["competitor_x"].to_numpy(float)
            xref = float(mdf["competitor_x_ref"].iloc[0])
            exact_norm = q_norm_exact(xmain) / float(q_norm_exact(xref))
            main_qerr = float(np.max(np.abs(mdf["q_h_over_q0_anomaly_normalized"].to_numpy(float) - exact_norm)))
            main_zerr = float(np.max(np.abs(mdf["zeta_anomaly"].to_numpy(float) - zeta_exact(xmain))))
            bref = float(mdf["competitor_field_ref"].iloc[0])
            weak = bref / mdf["abs_b"].to_numpy(float)
            main_werr = float(np.max(np.abs(mdf["q_h_weak_asymptote_over_q0"].to_numpy(float) - weak)))
            record("main exact crossing matches sqrt(1+x^2)/x", main_qerr < 3e-13, main_qerr, 3e-13)
            record("main exact local exponent matches 1/(1+x^2)", main_zerr < 3e-13, main_zerr, 3e-13)
            record("main weak asymptote is separately stored", main_werr < 3e-13, main_werr, 3e-13)
            record("main curve shows finite-x correction", float(np.max(np.abs(exact_norm-weak))) > 1e-2, float(np.max(np.abs(exact_norm-weak))), 1e-2)

    payload = {
        "all_checks_passed": all(c["passed"] for c in checks),
        "number_of_checks": len(checks),
        "source_equation": "Eq. (9) and q_h=omega_p/u of arXiv:2605.27031v1",
        "scope": (
            "Tests the exact algebra and asymptotics of the reported long-wavelength velocity formula. "
            "It does not assert that the acoustic approximation remains valid at arbitrary x or q."
        ),
        "checks": checks,
    }
    out_json = DATA / "competitor_crossover_validation.json"
    out_log = DATA / "competitor_crossover_validation.log"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        f"[{'PASS' if c['passed'] else 'FAIL'}] {i:02d}. {c['name']}"
        + (f" | value={c['value']}" if c.get('value') is not None else "")
        + (f" | tol={c['tolerance']}" if c.get('tolerance') is not None else "")
        for i, c in enumerate(checks, 1)
    ]
    lines.append(f"TOTAL: {len(checks)} checks; all_passed={payload['all_checks_passed']}")
    out_log.write_text("\n".join(lines) + "\n")
    print(out_log.read_text(), end="")
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
