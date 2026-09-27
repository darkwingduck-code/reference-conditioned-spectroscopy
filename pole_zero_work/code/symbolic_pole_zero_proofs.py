#!/usr/bin/env python3
"""Exact symbolic checks for the pole-zero theorem and its two-mode corollaries."""
from __future__ import annotations

from pathlib import Path
import json
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_prl"
OUT_TXT = DATA / "symbolic_pole_zero_proofs.txt"
OUT_JSON = DATA / "symbolic_pole_zero_proofs.json"

checks: list[dict] = []


def record(name: str, expr: sp.Expr) -> None:
    simplified = sp.factor(sp.cancel(sp.simplify(expr)))
    passed = simplified == 0
    checks.append({"name": name, "passed": bool(passed), "residual": str(simplified)})
    if not passed:
        raise AssertionError(f"{name}: {simplified}")

# Generic 3x3 rank-one cofactor identity in a probe-adapted basis.
a, b, c, d, e, f, g, h, i = sp.symbols("a b c d e f g h i", nonzero=True)
K = sp.Matrix([[a, b, c], [d, e, f], [g, h, i]])
minor_dark = sp.Matrix([[e, f], [h, i]]).det()
record("generic 3x3 scalar cofactor identity", K.inv()[0, 0] - minor_dark / K.det())

# A generic cross response is a bordered cofactor, not the same principal minor.
u1, u2, u3, v1, v2, v3 = sp.symbols("u1 u2 u3 v1 v2 v3")
u = sp.Matrix([[u1, u2, u3]])
v = sp.Matrix([v1, v2, v3])
record("cross response equals bordered adjugate numerator", (u * K.inv() * v)[0] - (u * K.adjugate() * v)[0] / K.det())

# Canonical non-Hermitian two-mode normal form.
z, zb, zd, gh = sp.symbols("z z_b z_d G2")
K2 = sp.Matrix([[z - zb, -sp.symbols("g")], [-sp.symbols("h"), z - zd]])
gsym, hsym = sp.symbols("g h")
K2 = sp.Matrix([[z - zb, -gsym], [-hsym, z - zd]])
chi = sp.factor(K2.inv()[0, 0])
record("two-mode response has dark numerator", chi - (z - zd) / ((z - zb) * (z - zd) - gsym * hsym))

# Symbolic poles represented only by their Vieta relations.
zm, zp = sp.symbols("z_minus z_plus")
# z0=zd, sum=zb+zd, product=zb*zd-g*h.
record("companion bright pole reconstruction", ((zm + zp - zd) - zb).subs(zp, zb + zd - zm))
# Apply the Vieta sum before simplifying the coupling product.
coupling_expr = sp.expand((zp - zd) * (zd - zm) - gsym * hsym)
# Eliminate the elementary symmetric polynomials of the two roots.
coupling_expr = coupling_expr.subs(zp * zm, zb * zd - gsym * hsym)
coupling_expr = sp.factor(coupling_expr).subs(zp + zm, zb + zd)
record("nonreciprocal coupling-product reconstruction", coupling_expr)

# Residue reconstruction of the finite zero.
C = sp.symbols("C", nonzero=True)
Rp = C * (zp - zd) / (zp - zm)
Rm = C * (zd - zm) / (zp - zm)
record("summed residue reconstructs prefactor", Rp + Rm - C)
record("two residues reconstruct response zero", (Rp * zm + Rm * zp) / (Rp + Rm) - zd)

# Exact cancellation at zero mixing.
zd2 = sp.symbols("z_d2")
det_full = (z - zb) * (z - zd) * (z - zd2)
cofactor_num = (z - zd) * (z - zd2)
record("decoupled dark factors cancel", cofactor_num / det_full - 1 / (z - zb))

# Lossless circle law.
Delta, G, Omega = sp.symbols("Delta G Omega", positive=True, real=True)
A = Delta / (2 * Omega)
R = G / Omega
circle = sp.together(A**2 + R**2 - 1)
circle = circle.subs(Omega**2, Delta**2 / 4 + G**2)
record("frequency-residue circle law", circle)

# Revision: the full cofactor and two-pole principal-part zeros are distinct.
aa, beta, dark, mix = sp.symbols("a beta dark mix")
den = (z - beta) * (z - dark) - mix**2
full = (1 + aa * z) * (z - dark) / den
pp_zero = dark - aa * mix**2 / (1 + aa * beta)
principal = (1 + aa * beta) * (z - pp_zero) / den
record("analytic counterexample exact background division", full - aa - principal)
record("principal numerator vanishes at residue zero", sp.cancel(principal * den).subs(z, pp_zero))
record("restored full numerator vanishes at compressed root", sp.cancel((aa + principal) * den).subs(z, dark))
record("principal/full coincidence requires zero background", sp.cancel(full.subs(z, pp_zero)) - aa)
K_counter = sp.diag(1 / (1 + aa * z), 1) * sp.Matrix([[z - beta, -mix], [-mix, z - dark]])
record("analytic inverse kernel gives full cofactor response", K_counter.inv()[0, 0] - full)

payload = {
    "all_checks_passed": all(c["passed"] for c in checks),
    "number_of_checks": len(checks),
    "checks": checks,
}
OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
OUT_TXT.write_text("\n".join(f"[{'ok' if c['passed'] else 'FAIL'}] {c['name']}: {c['residual']}" for c in checks) + "\n")
print(OUT_TXT.read_text(), end="")
print(json.dumps({"all_checks_passed": payload["all_checks_passed"], "number_of_checks": payload["number_of_checks"]}, indent=2))
