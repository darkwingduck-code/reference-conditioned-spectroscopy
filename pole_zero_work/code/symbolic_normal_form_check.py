#!/usr/bin/env python3
"""NEW reconstruction of the archived symbolic normal-form verification.

The original symbolic_normal_form_check.py source was not recovered. This
script independently verifies only the exact statements preserved in
data_prl/symbolic_normal_form_check.txt. It must not be represented as the
original source.
"""
from __future__ import annotations

import sympy as sp


SENTINEL = "All symbolic residuals vanish exactly."


def require_zero(expression: sp.Expr, label: str) -> None:
    residual = sp.simplify(expression)
    if residual != 0:
        raise AssertionError(f"{label}: residual={residual}")


def main() -> None:
    B, a_c, a_n, dq, g = sp.symbols("B a_c a_n dq g", real=True)
    v_c, v_n = sp.symbols("v_c v_n", real=True, nonzero=True)

    diagonal_difference = (v_c - v_n) * dq + (a_c - a_n) * B**2
    half_splitting_squared = diagonal_difference**2 / 4 + B**2 * g**2
    expanded = sp.expand(half_splitting_squared)

    dq_min = sp.solve(sp.diff(half_splitting_squared, dq), dq)[0]
    expected_dq_min = -(a_c - a_n) * B**2 / (v_c - v_n)
    dq_residual = sp.simplify(dq_min - expected_dq_min)
    require_zero(dq_residual, "stationary crossing shift")
    require_zero(sp.diff(half_splitting_squared, dq).subs(dq, dq_min),
                 "stationarity at crossing shift")
    require_zero(dq_min.subs(B, -B) - dq_min,
                 "crossing shift is even in field")

    minimum_half_splitting_squared = sp.simplify(
        half_splitting_squared.subs(dq, dq_min)
    )
    expected_minimum = B**2 * g**2
    minimum_residual = sp.simplify(
        minimum_half_splitting_squared - expected_minimum
    )
    require_zero(minimum_residual, "minimum squared half splitting")

    X = sp.symbols("X", positive=True, real=True)
    canonical = sp.Matrix([[X, 1], [1, -X]])
    expected_eigenvalues = {
        -sp.sqrt(X**2 + 1): 1,
        sp.sqrt(X**2 + 1): 1,
    }
    if canonical.eigenvals() != expected_eigenvalues:
        raise AssertionError(
            f"canonical eigenvalues: {canonical.eigenvals()}"
        )

    r_lower = sp.Rational(1, 2) - X / (2 * sp.sqrt(X**2 + 1))
    r_upper = sp.Rational(1, 2) + X / (2 * sp.sqrt(X**2 + 1))
    residue_sum_residual = sp.simplify(r_lower + r_upper - 1)
    require_zero(residue_sum_residual, "charge-residue sum")
    dark_limit = sp.limit(X**2 * r_lower, X, sp.oo)
    if dark_limit != sp.Rational(1, 4):
        raise AssertionError(f"large-positive-X dark limit: {dark_limit}")

    print("SYMBOLIC NORMAL-FORM CHECK")
    print("==========================")
    print()
    print("Squared half splitting:")
    sp.pprint(expanded)
    print()
    print("Stationary crossing shift:")
    print(f"dq_min = {dq_min}")
    print("Expected = -(a_c-a_n) B^2/(v_c-v_n)")
    print(f"Residual = {dq_residual}")
    print()
    print("Minimum squared half splitting:")
    print(minimum_half_splitting_squared)
    print("Expected = g^2 B^2")
    print(f"Residual = {minimum_residual}")
    print()
    print("Therefore Delta omega_min = 2 |g B| and q_min-q0 is even and O(B^2).")
    print()
    print("Canonical eigenvalues of [[X,1],[1,-X]]:")
    print(canonical.eigenvals())
    print()
    print("Charge-residue fractions:")
    print(f"r_lower = {r_lower}")
    print(f"r_upper = {r_upper}")
    print(f"sum residual = {residue_sum_residual}")
    print("large-positive-X dark fraction leading coefficient:")
    print(f"lim X^2 r_lower = {dark_limit}")
    print()
    print(SENTINEL)


if __name__ == "__main__":
    main()
