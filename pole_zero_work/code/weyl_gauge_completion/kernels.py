"""Longitudinal weak-field kernels through O(b^2).

Conventions
-----------
* hbar = c = v_F = k_F = 1 inside the dimensionless one-particle formulas.
* b = Q B / (2 c k_F^2) after units are restored.
* chi = +1 or -1 labels the two Weyl nodes.
* s = omega / (v_F q).
* The retarded branch is selected for real s on the particle-hole cut by
  taking s -> s + i0^+.  Complex inputs retain their supplied sheet.
* For undamped tests we use real s > 1.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class FermiSurfaceData:
    p_f: np.ndarray | float
    phase_space: np.ndarray | float
    radial_velocity: np.ndarray | float
    streaming_velocity: np.ndarray | float
    density_weight: np.ndarray | float


def _retarded_argument(s: complex | float) -> complex:
    """Choose the retarded boundary value for a real argument on (-1,1)."""
    z = complex(s)
    if z.imag == 0.0 and -1.0 < z.real < 1.0:
        # A positive imaginary part in s produces Im D(s)<0 for 0<s<1.
        z += 1.0e-15j
    if z == 1.0 or z == -1.0:
        raise ValueError("s=+/-1 is a particle-hole branch point")
    return z


def landau_D(s: complex | float) -> complex:
    """Return the retarded D(s)=s/2 log[(s+1)/(s-1)]-1.

    For a real argument inside the particle-hole continuum, the function
    returns the boundary value from the upper half s-plane.  To select a
    different analytic sheet, pass an explicitly complex argument.
    """
    z = _retarded_argument(s)
    if abs(z) > 10.0:
        inv2 = 1.0 / (z * z)
        term = inv2
        out = 0.0j
        # D(s)=sum_{n>=1} 1/[(2n+1)s^(2n)].
        for n in range(1, 18):
            out += term / (2 * n + 1)
            term *= inv2
        return out
    return 0.5 * z * cmath.log((z + 1.0) / (z - 1.0)) - 1.0


def delta_linear(s: complex | float) -> complex:
    """Universal O(b) charge-axial mixing kernel."""
    z = _retarded_argument(s)
    return z / (z * z - 1.0)


def kappa_microscopic(s: complex | float) -> complex:
    r"""Direct O(b^2) Fermi-surface kernel for the fixed Weyl model.

    This includes, at the same order, the second-order quasiparticle energy,
    the field-corrected Berry curvature, the phase-space Jacobian, the shifted
    equilibrium Fermi surface, and the corrected streaming velocity.
    """
    z = _retarded_argument(s)
    if abs(z) > 10.0:
        z2 = z * z
        return (
            3.0 / (20.0 * z2)
            + 93.0 / (140.0 * z2**2)
            + 115.0 / (84.0 * z2**3)
            + 287.0 / (132.0 * z2**4)
            + 1737.0 / (572.0 * z2**5)
            + 1023.0 / (260.0 * z2**6)
        )
    d = landau_D(z)
    num = (
        6.0 * d * (8.0 * z**6 - 17.0 * z**4 + 10.0 * z**2 - 1.0)
        - 16.0 * z**4
        + 25.0 * z**2
        - 5.0
    )
    return num / (4.0 * (z * z - 1.0) ** 2)


def fermi_surface_expansion(
    x: np.ndarray | float, chi: int, b: float
) -> FermiSurfaceData:
    """Return the analytic Fermi-surface data through O(b^2)."""
    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1")
    x = np.asarray(x)
    p_f = 1.0 + chi * b * x - 0.25 * b * b * (2.0 + 3.0 * x * x)
    phase_space = 1.0 + chi * b * x - b * b
    radial_velocity = (
        1.0 + chi * b * x - 0.25 * b * b * (6.0 + 5.0 * x * x)
    )
    streaming_velocity = (
        x + chi * b * x * x + b * b * (x - 3.75 * x**3)
    )
    density_weight = 1.0 + 2.0 * chi * b * x + b * b * (
        0.75 * x * x - 0.5
    )
    return FermiSurfaceData(
        p_f=p_f,
        phase_space=phase_space,
        radial_velocity=radial_velocity,
        streaming_velocity=streaming_velocity,
        density_weight=density_weight,
    )


def _fermi_momentum_exact(x: np.ndarray, chi: int, b: float) -> np.ndarray:
    """Solve the second-order truncated dispersion for the root near p=1."""
    p = 1.0 + chi * b * x - 0.25 * b * b * (2.0 + 3.0 * x * x)
    for _ in range(8):
        f = p - chi * b * x / p + 0.25 * b * b * (2.0 - x * x) / p**3 - 1.0
        fp = 1.0 + chi * b * x / p**2 - 0.75 * b * b * (2.0 - x * x) / p**4
        step = f / fp
        p -= step
        if np.max(np.abs(step)) < 2e-15:
            break
    if np.any(p <= 0):
        raise RuntimeError("unphysical Fermi-momentum root")
    return p


def _exact_truncated_fs_data(x: np.ndarray, chi: int, b: float) -> FermiSurfaceData:
    """Evaluate the second-order truncated band data without re-expanding in b."""
    p = _fermi_momentum_exact(x, chi, b)
    y = np.sqrt(np.maximum(0.0, 1.0 - x * x))

    # epsilon = p - chi*b*x/p + b^2(2-x^2)/(4p^3)
    ep = 1.0 + chi * b * x / p**2 - 0.75 * b * b * (2.0 - x * x) / p**4
    ex = -chi * b / p - 0.5 * b * b * x / p**3
    v_z = ep * x + (1.0 - x * x) * ex / p
    v_theta = -y * ex / p

    omega_r = chi / (2.0 * p**2) + b * x / (2.0 * p**4)
    omega_theta = b * y / (2.0 * p**4)
    v_dot_omega = ep * omega_r + v_theta * omega_theta

    phase = 1.0 + chi * b * x / p**2 + b * b * (2.0 * x * x - 1.0) / p**4
    numerator = v_z + 2.0 * b * v_dot_omega
    u = numerator / phase
    weight = p * p * phase / ep
    return FermiSurfaceData(p, phase, ep, u, weight)


def bare_pi_analytic(s: complex | float, b: float, chi: int) -> complex:
    """Bare node density response through O(b^2)."""
    return landau_D(s) + chi * b * delta_linear(s) + b * b * kappa_microscopic(s)


def bare_pi_numeric(
    s: complex | float, b: float, chi: int, n_quad: int = 800
) -> complex:
    """Numerically integrate the second-order truncated one-particle theory.

    Inside the particle-hole continuum, supply an explicitly complex ``s``
    with positive imaginary part; resolving an infinitesimal boundary value
    by fixed-order Gauss quadrature is intentionally not attempted.
    """
    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1")
    x, w = np.polynomial.legendre.leggauss(n_quad)
    fs = _exact_truncated_fs_data(x, chi, b)
    z = complex(s)
    integrand = fs.density_weight * fs.streaming_velocity / (
        z - fs.streaming_velocity
    )
    return 0.5 * np.dot(w, integrand)


def bare_current_numeric(
    s: complex | float, b: float, chi: int, n_quad: int = 800
) -> complex:
    """Transport-current response in units of nu_0 v_F."""
    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1")
    x, w = np.polynomial.legendre.leggauss(n_quad)
    fs = _exact_truncated_fs_data(x, chi, b)
    z = complex(s)
    integrand = fs.density_weight * fs.streaming_velocity**2 / (
        z - fs.streaming_velocity
    )
    return 0.5 * np.dot(w, integrand)


def pi_ca(s: complex | float, b: float) -> np.ndarray:
    """Bare density response in the orthonormal charge/axial basis."""
    p = landau_D(s) + b * b * kappa_microscopic(s)
    off = b * delta_linear(s)
    return np.array([[p, off], [off, p]], dtype=complex)


def current_transport_ca(s: complex | float, b: float) -> np.ndarray:
    """Covariant transport-current kernel before the Bardeen contact term."""
    pmat = pi_ca(s, b)
    return complex(s) * pmat - b * np.array([[0.0, 1.0], [1.0, 0.0]])


def current_consistent_ca(s: complex | float, b: float) -> np.ndarray:
    """Consistent current kernel with the vector-current contact term.

    The added c<-a entry is the Bardeen-Zumino/contact contribution.  It
    converts the covariant node currents into a vector-charge-conserving
    current while leaving the axial anomaly in the a<-c entry.
    """
    out = current_transport_ca(s, b)
    out += b * np.array([[0.0, 1.0], [0.0, 0.0]])
    return out


def dressed_scalar_vertex(
    s: complex | float, b: float, f_c: complex, f_a: complex
) -> np.ndarray:
    """Total scalar source generated by a unit external charge source."""
    pmat = pi_ca(s, b)
    fmat = np.diag([f_c, f_a]).astype(complex)
    e_c = np.array([1.0, 0.0], dtype=complex)
    return np.linalg.solve(np.eye(2, dtype=complex) - fmat @ pmat, e_c)


def dressed_density_current(
    s: complex | float, b: float, f_c: complex, f_a: complex
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total scalar vertex, density, and consistent current."""
    gamma = dressed_scalar_vertex(s, b, f_c, f_a)
    density = pi_ca(s, b) @ gamma
    current = current_consistent_ca(s, b) @ gamma
    return gamma, density, current


def charge_response(s: complex | float, b: float, f_c: complex, f_a: complex) -> complex:
    """Proper charge-density response to a unit external charge source."""
    _, density, _ = dressed_density_current(s, b, f_c, f_a)
    return density[0]


def physical_charge_susceptibility(
    s: complex | float,
    b: float,
    f_c: complex,
    f_a: complex,
    nu0: float,
) -> complex:
    """Physical two-node particle-number susceptibility to a common energy.

    ``charge_response`` is dimensionless in the orthonormal charge basis.
    A common node source and the total density each contribute ``sqrt(2)``,
    giving the physical factor ``2*nu0``.
    """
    if nu0 <= 0.0:
        raise ValueError("nu0 must be positive")
    return 2.0 * nu0 * charge_response(s, b, f_c, f_a)


def c2_charge(s: complex | float, f_c: complex, f_a: complex) -> complex:
    """Complete O(b^2) coefficient for the fixed large-N density model."""
    d = landau_D(s)
    delta = delta_linear(s)
    a_c = 1.0 - f_c * d
    a_a = 1.0 - f_a * d
    return kappa_microscopic(s) / (a_c * a_c) + f_a * delta * delta / (
        a_a * a_c * a_c
    )


def longitudinal_response_tensor(
    omega: complex,
    q: float,
    susceptibility: complex,
    charge: float = 1.0,
    c: float = 1.0,
) -> np.ndarray:
    r"""Gauge-transverse longitudinal electromagnetic response tensor.

    The ordering is ``(0,L)`` and the Fourier convention is
    ``exp(-i omega t+i q z)``.  The scalar susceptibility is the particle-
    number response to the gauge-invariant potential energy.  The returned
    tensor satisfies both left and right Ward identities.
    """
    if q == 0.0:
        raise ValueError("q must be nonzero")
    ratio = complex(omega) / (c * q)
    return charge * charge * susceptibility * np.array(
        [[1.0, -ratio], [c * ratio, -c * ratio * ratio]], dtype=complex
    )
