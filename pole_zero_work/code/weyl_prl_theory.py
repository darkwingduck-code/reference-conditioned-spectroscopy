"""Universal and conserving results for magnetic brightening of a dark Weyl mode.

This module has two logically separate layers.

1. A model-independent local two-mode normal form around a finite-q crossing.
2. A fixed conserving l=0 Weyl density model used as an existence proof that
   a nonzero O(B_parallel) charge-axial mixing survives Ward completion.

Fourier convention: exp(-i omega t + i q z).  Dimensionless phase velocity
s = omega/(v_F q), and b is the node-odd weak-field parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq


def landau_D(s: complex | float) -> complex:
    """D(s)=s/2 log[(s+1)/(s-1)]-1 on the supplied analytic branch."""
    z = complex(s)
    if z.imag == 0.0 and z.real <= 1.0:
        raise ValueError("The undamped implementation requires real s>1.")
    if abs(z) > 20.0:
        inv2 = 1.0 / (z * z)
        return inv2 * (
            1.0 / 3.0
            + inv2 * (1.0 / 5.0 + inv2 * (1.0 / 7.0 + inv2 / 9.0))
        )
    return 0.5 * z * np.log((z + 1.0) / (z - 1.0)) - 1.0


def landau_D_prime(s: float) -> float:
    """Derivative of D(s) for real s>1."""
    if s <= 1.0:
        raise ValueError("s must exceed one")
    return float(0.5 * np.log((s + 1.0) / (s - 1.0)) - s / (s * s - 1.0))


def delta_linear(s: complex | float) -> complex:
    """Universal first-order charge-axial density-mixing kernel."""
    z = complex(s)
    return z / (z * z - 1.0)


def solve_l0_zero_sound(F: float) -> float:
    """Solve 1-F D(s)=0 for F>0 and s>1."""
    if F <= 0.0:
        raise ValueError("The undamped l=0 root used here requires F>0.")
    f = lambda x: float((1.0 - F * landau_D(x)).real)
    lo = 1.0 + 1.0e-12
    hi = 2.0
    while f(lo) * f(hi) > 0.0 and hi < 1.0e7:
        hi *= 2.0
    if f(lo) * f(hi) > 0.0:
        raise RuntimeError("Could not bracket l=0 zero sound")
    return float(brentq(f, lo, hi, xtol=2.0e-14, rtol=2.0e-14))


def conserving_pi_ca(s: complex | float, b: float) -> np.ndarray:
    """Bare charge/axial density kernel through first order in b."""
    d = landau_D(s)
    off = b * delta_linear(s)
    return np.array([[d, off], [off, d]], dtype=complex)


def consistent_current_ca(s: complex | float, b: float) -> np.ndarray:
    """Longitudinal consistent-current kernel.

    It satisfies s Pi-J=b e_a e_c^T: vector charge is conserved while the
    axial row retains the anomaly source.
    """
    pi = conserving_pi_ca(s, b)
    anomaly = b * np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    return complex(s) * pi - anomaly


def dressed_density_current(
    s: complex | float, b: float, F_c: complex, F_a: complex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar vertex, density, and consistent current for a charge source."""
    pi = conserving_pi_ca(s, b)
    F = np.diag([F_c, F_a]).astype(complex)
    e_c = np.array([1.0, 0.0], dtype=complex)
    gamma = np.linalg.solve(np.eye(2, dtype=complex) - F @ pi, e_c)
    density = pi @ gamma
    current = consistent_current_ca(s, b) @ gamma
    return gamma, density, current


def charge_continuity_residual(
    s: complex | float, b: float, F_c: complex, F_a: complex
) -> complex:
    """Return s rho_c-j_c; identically zero within roundoff."""
    _, rho, current = dressed_density_current(s, b, F_c, F_a)
    return complex(s) * rho[0] - current[0]


def l0_secular_determinant(
    s: complex | float, b: float, F_c: complex, F_a: complex
) -> complex:
    pi = conserving_pi_ca(s, b)
    F = np.diag([F_c, F_a]).astype(complex)
    return np.linalg.det(np.eye(2, dtype=complex) - pi @ F)


def l0_crossing_roots(s_star: float, b: float) -> tuple[float, float]:
    """Exact roots of the O(b) conserving kernel at an l=0 degeneracy.

    At the crossing F_c=F_a=F*=1/D(s_star), so parity diagonalizes the
    density kernel and the two equations are 1-F*[D(s)+/-b delta(s)]=0.
    """
    F = 1.0 / float(landau_D(s_star).real)
    roots: list[float] = []
    width = max(0.03, 12.0 * abs(b))
    lo_global = max(1.0 + 1.0e-10, s_star - width)
    hi_global = s_star + width
    for sign in (-1.0, +1.0):
        fun = lambda x: float(
            1.0 - F * (landau_D(x).real + sign * b * delta_linear(x).real)
        )
        grid = np.linspace(lo_global, hi_global, 2500)
        vals = np.array([fun(x) for x in grid])
        candidates = []
        for i in range(len(grid) - 1):
            if vals[i] == 0.0 or vals[i] * vals[i + 1] < 0.0:
                candidates.append((grid[i], grid[i + 1]))
        if not candidates:
            raise RuntimeError(f"No l=0 crossing root for sign={sign}, b={b}")
        # Select the crossing root closest to s_star.
        a, c = min(candidates, key=lambda ac: abs(0.5 * sum(ac) - s_star))
        roots.append(float(brentq(fun, a, c, xtol=2e-14, rtol=2e-14)))
    roots.sort()
    return roots[0], roots[1]


def l0_gap_coefficient(s_star: float) -> float:
    """Analytic Delta s/|b| at an l=0 conserving crossing."""
    return float(2.0 * abs(delta_linear(s_star).real / landau_D_prime(s_star)))


@dataclass(frozen=True)
class TwoModeNormalForm:
    """Local reciprocal two-mode normal form near a finite-q crossing."""

    q0: float
    omega0: float
    v_c: float
    v_n: float
    g_per_field: float

    @property
    def delta_v(self) -> float:
        return self.v_c - self.v_n

    def uncoupled(self, q: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        qarr = np.asarray(q, dtype=float)
        dq = qarr - self.q0
        return self.omega0 + self.v_c * dq, self.omega0 + self.v_n * dq

    def coupled(self, q: np.ndarray | float, field: float) -> tuple[np.ndarray, np.ndarray]:
        wc, wn = self.uncoupled(q)
        center = 0.5 * (wc + wn)
        half = np.sqrt(0.25 * (wc - wn) ** 2 + (self.g_per_field * field) ** 2)
        return center - half, center + half

    def scaling_variable(self, q: np.ndarray | float, field: float) -> np.ndarray:
        wc, wn = self.uncoupled(q)
        denom = 2.0 * abs(self.g_per_field * field)
        if denom == 0.0:
            raise ValueError("field must be nonzero")
        return (wc - wn) / denom


def universal_branch_curve(X: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    Xarr = np.asarray(X, dtype=float)
    y = np.sqrt(1.0 + Xarr * Xarr)
    return -y, +y


def universal_residue_fractions(X: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Canonical charge residue fractions for lower and upper poles."""
    Xarr = np.asarray(X, dtype=float)
    den = np.sqrt(1.0 + Xarr * Xarr)
    upper = 0.5 * (1.0 + Xarr / den)
    return 1.0 - upper, upper


def field_intercept_exponent(fields: np.ndarray, qcross: np.ndarray) -> np.ndarray:
    """Return zeta=-d ln q_cross/d ln|field| using centered gradients."""
    f = np.asarray(fields, dtype=float)
    q = np.asarray(qcross, dtype=float)
    if np.any(f <= 0.0) or np.any(q <= 0.0):
        raise ValueError("fields and qcross must be positive")
    return -np.gradient(np.log(q), np.log(f))


def anomaly_crossing_small_field(
    field: np.ndarray | float, field_ref: float, q_ref: float
) -> np.ndarray:
    """Small-field q_h proportional to 1/|B_parallel|, normalized at a reference."""
    f = np.asarray(field, dtype=float)
    if np.any(f <= 0.0) or field_ref <= 0.0 or q_ref <= 0.0:
        raise ValueError("positive field and reference values required")
    return q_ref * field_ref / f


def anomaly_crossing_exact_crossover(
    field: np.ndarray | float,
    field_ref: float,
    q_ref: float,
    x_ref: float = 0.1,
) -> np.ndarray:
    """Exact normalized crossing implied by the reported acoustic velocity.

    The long-wavelength formula of arXiv:2605.27031 gives

        q_h / q_* = sqrt(1+x^2) / x,
        x proportional to |B|.

    ``x_ref`` fixes the material-dependent conversion between the plotted
    dimensionless field and x at ``field_ref``.  ``q_ref`` normalizes the
    curve at that same reference field.  The weak-field limit reproduces
    q_h proportional to 1/|B|, while the finite-x correction is retained.
    """
    f = np.asarray(field, dtype=float)
    if np.any(f <= 0.0) or field_ref <= 0.0 or q_ref <= 0.0 or x_ref <= 0.0:
        raise ValueError("positive fields, references, and x_ref are required")
    x = x_ref * f / field_ref
    shape = np.sqrt(1.0 + x * x) / x
    shape_ref = np.sqrt(1.0 + x_ref * x_ref) / x_ref
    return q_ref * shape / shape_ref


def anomaly_crossover_exponent(
    field: np.ndarray | float,
    field_ref: float,
    x_ref: float = 0.1,
) -> np.ndarray:
    """Local exponent zeta=1/(1+x^2) for the exact acoustic crossover."""
    f = np.asarray(field, dtype=float)
    if np.any(f <= 0.0) or field_ref <= 0.0 or x_ref <= 0.0:
        raise ValueError("positive fields, reference field, and x_ref are required")
    x = x_ref * f / field_ref
    return 1.0 / (1.0 + x * x)
