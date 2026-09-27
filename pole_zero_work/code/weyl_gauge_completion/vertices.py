"""Band geometry, multipole vertices, and Ward-identity utilities.

The formulas implement the conduction-band semiclassical expansion of an
isotropic Weyl cone through second order in electromagnetic fields.  The
signed quasiparticle charge is ``charge``; set it to ``-e`` for electrons.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _unit(vec: np.ndarray) -> tuple[np.ndarray, float]:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("momentum must be nonzero")
    return vec / norm, norm


def berry_curvature(
    p_vec: np.ndarray,
    chi: int,
    charge: float,
    b_vec: np.ndarray,
    e_vec: np.ndarray | None = None,
    v_f: float = 1.0,
    c: float = 1.0,
) -> np.ndarray:
    r"""Field-corrected Berry curvature through first order in ``E`` and ``B``.

    The correction is the one required by a second-order semiclassical
    treatment because the Berry curvature always appears multiplied by an
    electromagnetic field in the equations of motion.
    """
    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1")
    phat, p = _unit(p_vec)
    b_vec = np.asarray(b_vec, dtype=float)
    if e_vec is None:
        e_vec = np.zeros(3)
    e_vec = np.asarray(e_vec, dtype=float)
    omega0 = chi * phat / (2.0 * p * p)
    omega1 = charge * (
        (2.0 / c) * phat * np.dot(phat, b_vec)
        - b_vec / c
        + (2.0 / v_f) * np.cross(e_vec, phat)
    ) / (4.0 * p**4)
    return omega0 + omega1


def phase_space_measure(
    p_vec: np.ndarray,
    chi: int,
    charge: float,
    b_vec: np.ndarray,
    e_vec: np.ndarray | None = None,
    v_f: float = 1.0,
    c: float = 1.0,
) -> float:
    r"""Return ``D=1+(Q/c) B.Omega`` at the retained order."""
    b_vec = np.asarray(b_vec, dtype=float)
    omega = berry_curvature(p_vec, chi, charge, b_vec, e_vec, v_f, c)
    return float(1.0 + (charge / c) * np.dot(b_vec, omega))


def quasiparticle_energy(
    p_vec: np.ndarray,
    chi: int,
    charge: float,
    b_vec: np.ndarray,
    e_vec: np.ndarray | None = None,
    v_f: float = 1.0,
    c: float = 1.0,
) -> float:
    r"""Second-order conduction-band Weyl quasiparticle energy.

    The optional ``E`` term is bilinear in the probe electric field and the
    static background magnetic field.  It determines the field-induced
    electric dipole and the reciprocal electric correction to the magnetic
    moment.
    """
    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1")
    phat, p = _unit(p_vec)
    b_vec = np.asarray(b_vec, dtype=float)
    bdot = float(np.dot(b_vec, phat))
    energy = (
        v_f * p
        - chi * charge * v_f * bdot / (2.0 * c * p)
        + charge**2
        * v_f
        * (2.0 * np.dot(b_vec, b_vec) - bdot * bdot)
        / (16.0 * c * c * p**3)
    )
    if e_vec is not None:
        e_vec = np.asarray(e_vec, dtype=float)
        energy -= charge**2 * np.dot(e_vec, np.cross(phat, b_vec)) / (
            4.0 * c * p**3
        )
    return float(energy)


def magnetic_moment(
    p_vec: np.ndarray,
    chi: int,
    charge: float,
    b_vec: np.ndarray,
    e_vec: np.ndarray | None = None,
    v_f: float = 1.0,
    c: float = 1.0,
) -> np.ndarray:
    r"""Return ``m=-partial epsilon/partial B`` at fixed ``E``.

    The final term is the mixed ``E B`` contribution.  It vanishes in the
    static equilibrium used for the longitudinal kernel but is retained for
    the complete first-gradient electromagnetic vertex.
    """
    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1")
    phat, p = _unit(p_vec)
    b_vec = np.asarray(b_vec, dtype=float)
    if e_vec is None:
        e_vec = np.zeros(3)
    e_vec = np.asarray(e_vec, dtype=float)
    leading = chi * charge * v_f * phat / (2.0 * c * p)
    b_correction = -charge**2 * v_f * (
        2.0 * b_vec - np.dot(b_vec, phat) * phat
    ) / (8.0 * c * c * p**3)
    e_correction = charge**2 * np.cross(e_vec, phat) / (4.0 * c * p**3)
    return leading + b_correction + e_correction


def electric_dipole(
    p_vec: np.ndarray,
    charge: float,
    b_vec: np.ndarray,
    c: float = 1.0,
) -> np.ndarray:
    r"""Return ``d=-partial epsilon/partial E`` for the B-induced dipole."""
    phat, p = _unit(p_vec)
    b_vec = np.asarray(b_vec, dtype=float)
    return charge**2 * np.cross(phat, b_vec) / (4.0 * c * p**3)


def multipole_vertices(
    omega: complex,
    q_vec: np.ndarray,
    phase_space: complex,
    rdot: np.ndarray,
    dipole: np.ndarray,
    moment: np.ndarray,
    charge: float = 1.0,
    c: float = 1.0,
) -> Tuple[complex, np.ndarray]:
    r"""Return density and current vertices for ``exp(-i omega t+i q.x)``.

    ``rho = Q D - i q.d`` and
    ``j = Q D rdot - i omega d + i c q x m``.
    """
    q_vec = np.asarray(q_vec, dtype=float)
    rdot = np.asarray(rdot, dtype=complex)
    dipole = np.asarray(dipole, dtype=complex)
    moment = np.asarray(moment, dtype=complex)
    rho = charge * phase_space - 1j * np.dot(q_vec, dipole)
    current = (
        charge * phase_space * rdot
        - 1j * omega * dipole
        + 1j * c * np.cross(q_vec, moment)
    )
    return rho, current


def multipole_wti_residual(
    omega: complex,
    q_vec: np.ndarray,
    phase_space: complex,
    rdot: np.ndarray,
    dipole: np.ndarray,
    moment: np.ndarray,
    charge: float = 1.0,
    c: float = 1.0,
) -> complex:
    """Return the wave-packet continuity residual (identically zero)."""
    rho, current = multipole_vertices(
        omega, q_vec, phase_space, rdot, dipole, moment, charge, c
    )
    lhs = -1j * omega * rho + 1j * np.dot(q_vec, current)
    rhs = -1j * charge * phase_space * (
        omega - np.dot(q_vec, np.asarray(rdot, dtype=complex))
    )
    return lhs - rhs


def gradient_landau_vertex(
    p_vec: np.ndarray,
    k_vec: np.ndarray,
    charge: float,
    b_vec: np.ndarray,
    coupling: float,
    c: float = 1.0,
) -> np.ndarray:
    r"""Leading gradient Landau vector ``V=g(ell_p-ell_k)``.

    Here ``ell=d/Q`` is the electric-dipole length.  The vector is odd under
    exchange of its two quasiparticles and its contraction with ``q`` vanishes
    pointwise when ``q`` is parallel to ``B``.
    """
    if charge == 0.0:
        raise ValueError("charge must be nonzero")
    ell_p = electric_dipole(p_vec, charge, b_vec, c) / charge
    ell_k = electric_dipole(k_vec, charge, b_vec, c) / charge
    return coupling * (ell_p - ell_k)


def full_landau_kernel(
    p_vec: np.ndarray,
    k_vec: np.ndarray,
    chi_p: int,
    chi_k: int,
    charge: float,
    b_vec: np.ndarray,
    q_vec: np.ndarray,
    coupling: float,
    c: float = 1.0,
) -> complex:
    r"""Field/gradient Hartree Landau kernel through first order in ``q``.

    The kernel is defined with the ordinary momentum measure.  Its weighted
    reciprocity relation is

    ``D_p f_pk(q) = D_k f_kp(-q)``

    for a symmetric coupling matrix.
    """
    if charge == 0.0:
        raise ValueError("charge must be nonzero")
    q_vec = np.asarray(q_vec, dtype=float)
    d_p = phase_space_measure(p_vec, chi_p, charge, b_vec, c=c)
    d_k = phase_space_measure(k_vec, chi_k, charge, b_vec, c=c)
    ell_p = electric_dipole(p_vec, charge, b_vec, c) / charge
    ell_k = electric_dipole(k_vec, charge, b_vec, c) / charge
    gradient = np.dot(q_vec, (d_k / d_p) * ell_p - ell_k)
    return complex(coupling * (d_k + 1j * gradient))


def contact_number_current(
    axial_energy: complex,
    charge: float,
    b_vec: np.ndarray,
    c: float = 1.0,
) -> np.ndarray:
    r"""Low-energy consistent contact/backflow number current.

    ``j_ct = Q U_a B/(2 pi^2 c)``.  Multiplication by ``Q`` gives electric
    current.  ``axial_energy=(U_+-U_-)/2`` is an energy, not a voltage.
    """
    b_vec = np.asarray(b_vec, dtype=float)
    return charge * axial_energy * b_vec.astype(complex) / (2.0 * np.pi**2 * c)
