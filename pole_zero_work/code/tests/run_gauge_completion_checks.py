#!/usr/bin/env python3
"""Executable checks for the gauge-invariant Weyl-FL completion."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import sympy as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weyl_gauge_completion import (  # noqa: E402
    bare_current_numeric,
    bare_pi_analytic,
    bare_pi_numeric,
    berry_curvature,
    c2_charge,
    charge_response,
    contact_number_current,
    current_consistent_ca,
    current_transport_ca,
    delta_linear,
    dressed_density_current,
    electric_dipole,
    fermi_surface_expansion,
    full_landau_kernel,
    gradient_landau_vertex,
    kappa_microscopic,
    landau_D,
    longitudinal_response_tensor,
    magnetic_moment,
    multipole_wti_residual,
    phase_space_measure,
    physical_charge_susceptibility,
    pi_ca,
    quasiparticle_energy,
)


def assert_close(a, b, atol=1e-10, rtol=1e-9, message=""):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{message}: {a!r} != {b!r}")


def ok(label: str) -> None:
    print(f"[ok] {label}")



def lattice_regularization_check() -> None:
    """Verify that the explicit m=2t, t_z=t model has only two Weyl nodes."""
    t = 1.0
    tz = 1.0
    m = 2.0
    nodes = []
    for kx in (0.0, math.pi):
        for ky in (0.0, math.pi):
            rhs = (m - t * (math.cos(kx) + math.cos(ky))) / tz
            if abs(rhs) <= 1.0:
                kz0 = math.acos(max(-1.0, min(1.0, rhs)))
                candidates = (kz0,) if abs(kz0) < 1e-14 else (kz0, -kz0)
                for kz in candidates:
                    jac = (t * math.cos(kx)) * (t * math.cos(ky)) * (tz * math.sin(kz))
                    nodes.append((kx, ky, kz, math.copysign(1.0, jac)))
    assert len(nodes) == 2
    kz_values = sorted(node[2] for node in nodes)
    assert_close(kz_values, [-math.pi / 2.0, math.pi / 2.0], atol=2e-15)
    chiralities = [node[3] for node in sorted(nodes, key=lambda z: z[2])]
    assert chiralities == [-1.0, 1.0]
    ok("two-node lattice regularization and isotropic chiralities")


def retarded_branch_check() -> None:
    """Check the retarded cut and a damped direct-quadrature continuation."""
    s = 0.4
    d_plus = landau_D(s)
    d_minus = landau_D(-s)
    assert_close(d_plus.imag, -math.pi * s / 2.0, atol=2e-14, rtol=2e-14)
    assert_close(d_minus, np.conjugate(d_plus), atol=2e-14, rtol=2e-14)

    z = 0.45 + 0.07j
    b = 1.0e-3
    for chi in (-1, 1):
        numerical = bare_pi_numeric(z, b, chi, n_quad=180)
        analytic = bare_pi_analytic(z, b, chi)
        assert_close(numerical, analytic, atol=8e-9, rtol=3e-8)
    ok("retarded branch and complex-frequency analytic continuation")

def symbolic_band_expansion_check() -> None:
    b, x, chi = sp.symbols("b x chi", real=True)
    p = 1 + chi * b * x - b**2 * (2 + 3 * x**2) / 4
    eps = p - chi * b * x / p + b**2 * (2 - x**2) / (4 * p**3)
    series = sp.series(eps, b, 0, 3).removeO().expand().subs(chi**2, 1)
    assert sp.simplify(series - 1) == 0

    fs = fermi_surface_expansion(np.array([-0.7, 0.0, 0.8]), 1, 0.02)
    assert np.all(fs.p_f > 0)
    ok("second-order equilibrium Fermi surface")


def magnetic_and_electric_dipole_check() -> None:
    p = np.array([0.7, -0.4, 1.2])
    bvec = np.array([0.03, 0.02, 0.11])
    evec = np.array([-0.04, 0.06, 0.01])
    q = -1.3
    chi = 1
    h = 2e-7

    m = magnetic_moment(p, chi, q, bvec, evec)
    m_fd = np.zeros(3)
    for i in range(3):
        db = np.zeros(3)
        db[i] = h
        ep = quasiparticle_energy(p, chi, q, bvec + db, evec)
        em = quasiparticle_energy(p, chi, q, bvec - db, evec)
        m_fd[i] = -(ep - em) / (2 * h)
    assert_close(m, m_fd, atol=3e-9, rtol=3e-8, message="magnetic moment")

    d = electric_dipole(p, q, bvec)
    d_fd = np.zeros(3)
    for i in range(3):
        de = np.zeros(3)
        de[i] = h
        ep = quasiparticle_energy(p, chi, q, bvec, evec + de)
        em = quasiparticle_energy(p, chi, q, bvec, evec - de)
        d_fd[i] = -(ep - em) / (2 * h)
    assert_close(d, d_fd, atol=3e-9, rtol=3e-8, message="electric dipole")
    ok("magnetic and electric dipoles from the same band energy")


def mixed_dipole_reciprocity_check() -> None:
    p = np.array([0.51, 0.77, -0.33])
    b = np.array([0.08, -0.04, 0.12])
    e = np.array([-0.03, 0.05, 0.07])
    q = -0.9
    h = 3e-7
    dm_de = np.zeros((3, 3))
    dd_db = np.zeros((3, 3))
    for j in range(3):
        de = np.zeros(3)
        de[j] = h
        dm_de[:, j] = (
            magnetic_moment(p, -1, q, b, e + de)
            - magnetic_moment(p, -1, q, b, e - de)
        ) / (2 * h)
    for i in range(3):
        db = np.zeros(3)
        db[i] = h
        dd_db[:, i] = (
            electric_dipole(p, q, b + db) - electric_dipole(p, q, b - db)
        ) / (2 * h)
    # dm_i/dE_j = dd_j/dB_i
    assert_close(dm_de, dd_db.T, atol=3e-10, rtol=3e-9)
    ok("mixed electric/magnetic dipole reciprocity")


def berry_flux_and_measure_check() -> None:
    # The complete O(F) correction has zero angular flux through a sphere.
    p = 1.7
    charge = -0.8
    bvec = np.array([0.03, -0.02, 0.09])
    evec = np.array([-0.07, 0.04, 0.02])
    x, w = np.polynomial.legendre.leggauss(220)
    phi = np.linspace(0.0, 2.0 * np.pi, 300, endpoint=False)
    flux_correction = 0.0
    for xx, ww in zip(x, w):
        st = math.sqrt(max(0.0, 1.0 - xx * xx))
        for pp in phi:
            phat = np.array([st * math.cos(pp), st * math.sin(pp), xx])
            omega = berry_curvature(p * phat, 1, charge, bvec, evec)
            omega0 = phat / (2.0 * p * p)
            flux_correction += ww * (2.0 * np.pi / len(phi)) * p * p * np.dot(
                omega - omega0, phat
            )
    assert abs(flux_correction) < 3e-12

    pvec = np.array([0.8, -0.6, 1.1])
    omega = berry_curvature(pvec, -1, charge, bvec, evec)
    expected = 1.0 + charge * np.dot(bvec, omega)
    assert_close(
        phase_space_measure(pvec, -1, charge, bvec, evec), expected, atol=2e-15
    )
    ok("field-corrected Berry flux and phase-space measure")


def bare_kernel_check() -> None:
    for s in (1.2, 1.7, 3.0, 12.0):
        for chi in (-1, 1):
            b = 1.0e-3
            numerical = bare_pi_numeric(s, b, chi, n_quad=180)
            analytic = bare_pi_analytic(s, b, chi)
            assert_close(
                numerical,
                analytic,
                atol=8e-9,
                rtol=3e-8,
                message=f"bare Pi s={s}, chi={chi}",
            )
    ok("microscopic D + chi b delta + b^2 kappa kernel")


def finite_difference_coefficients_check() -> None:
    for s in (1.25, 1.8, 2.7):
        h = 8.0e-4
        pp = bare_pi_numeric(s, h, 1, n_quad=220)
        pm = bare_pi_numeric(s, -h, 1, n_quad=220)
        p0 = bare_pi_numeric(s, 0.0, 1, n_quad=220)
        delta_fd = (pp - pm) / (2.0 * h)
        kappa_fd = (pp + pm - 2.0 * p0) / (2.0 * h * h)
        assert_close(delta_fd, delta_linear(s), atol=2e-6, rtol=2e-6)
        assert_close(kappa_fd, kappa_microscopic(s), atol=2e-5, rtol=2e-5)
    ok("finite-difference extraction of delta(s) and kappa_micro(s)")


def current_anomaly_moment_check() -> None:
    for s in (1.3, 1.9, 4.0):
        for chi in (-1, 1):
            b = 0.006
            pi_num = bare_pi_numeric(s, b, chi, n_quad=200)
            j_num = bare_current_numeric(s, b, chi, n_quad=200)
            assert_close(s * pi_num - j_num, chi * b, atol=4e-8, rtol=4e-8)
    ok("node current identity and covariant anomaly coefficient")


def multipole_continuity_check() -> None:
    rng = np.random.default_rng(9127)
    for _ in range(50):
        omega = rng.normal() + 0.2j
        qvec = rng.normal(size=3)
        rdot = rng.normal(size=3)
        dipole = rng.normal(size=3) * 0.03
        moment = rng.normal(size=3) * 0.04
        phase = 1.0 + 0.1 * rng.normal()
        residual = multipole_wti_residual(
            omega, qvec, phase, rdot, dipole, moment, charge=-1.2, c=2.3
        )
        assert abs(residual) < 3e-14
    ok("wave-packet transport/polarization/magnetization continuity identity")


def gradient_interaction_check() -> None:
    p = np.array([0.8, -0.2, 1.1])
    k = np.array([-0.4, 0.9, 0.7])
    bvec = np.array([0.0, 0.0, 0.13])
    qvec = np.array([0.0, 0.0, 0.27])
    charge = -1.0
    coupling = 1.7
    vpk = gradient_landau_vertex(p, k, charge, bvec, coupling)
    vkp = gradient_landau_vertex(k, p, charge, bvec, coupling)
    assert_close(vpk, -vkp, atol=2e-15, rtol=2e-15)
    assert abs(np.dot(qvec, vpk)) < 2e-15

    fpk = full_landau_kernel(p, k, 1, -1, charge, bvec, qvec, coupling)
    fkp = full_landau_kernel(k, p, -1, 1, charge, bvec, -qvec, coupling)
    dp = phase_space_measure(p, 1, charge, bvec)
    dk = phase_space_measure(k, -1, charge, bvec)
    assert_close(dp * fpk, dk * fkp, atol=3e-15, rtol=3e-15)
    ok("gradient Landau antisymmetry, reciprocity, and q-parallel-B decoupling")


def matrix_continuity_and_anomaly_check() -> None:
    for s, b in ((1.35, 0.02), (1.9, -0.03), (2.8, 0.015)):
        pmat = pi_ca(s, b)
        jtr = current_transport_ca(s, b)
        jcon = current_consistent_ca(s, b)
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        expected_cov = b * sigma_x
        assert_close(s * pmat - jtr, expected_cov, atol=2e-13, rtol=2e-13)
        expected_cons = np.array([[0.0, 0.0], [b, 0.0]])
        assert_close(s * pmat - jcon, expected_cons, atol=2e-13, rtol=2e-13)
    ok("consistent contact converts covariant node current to conserved charge current")


def dressed_continuity_check() -> None:
    rng = np.random.default_rng(41)
    for _ in range(40):
        s = 1.15 + 2.2 * rng.random()
        b = 0.04 * (rng.random() - 0.5)
        f_c = -0.2 + 2.5 * rng.random()
        f_a = -0.2 + 2.5 * rng.random()
        _, density, current = dressed_density_current(s, b, f_c, f_a)
        assert abs(s * density[0] - current[0]) < 5e-12
    ok("interacting charge continuity including backflow/contact current")


def c2_check() -> None:
    for s, fc, fa in ((1.25, 0.8, 1.4), (1.7, 1.2, 2.0), (2.4, 0.4, 1.1)):
        h = 2.0e-4
        rp = charge_response(s, h, fc, fa)
        rm = charge_response(s, -h, fc, fa)
        r0 = charge_response(s, 0.0, fc, fa)
        c2_fd = (rp + rm - 2.0 * r0) / (2.0 * h * h)
        assert_close(c2_fd, c2_charge(s, fc, fa), atol=3e-6, rtol=3e-6)
    ok("complete microscopic c2 including direct and interaction pieces")


def longitudinal_tensor_wti_check() -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        omega = 0.2 + 2.0 * rng.random()
        q = 0.1 + 1.2 * rng.random()
        c = 0.7 + 2.0 * rng.random()
        chi = rng.normal() + 0.2j * rng.normal()
        tensor = longitudinal_response_tensor(omega, q, chi, charge=-1.4, c=c)
        left = np.array([omega, -q], dtype=complex) @ tensor
        right = tensor @ np.array([omega / c, q], dtype=complex)
        assert np.linalg.norm(left) < 3e-13
        assert np.linalg.norm(right) < 3e-13
    ok("left and right longitudinal Ward-Takahashi transversality")


def large_s_check() -> None:
    for s in (20.0, 100.0, 1.0e4):
        d = landau_D(s)
        k = kappa_microscopic(s)
        assert np.isfinite(d.real) and np.isfinite(k.real)
        assert_close(d * s * s, 1.0 / 3.0, atol=2e-3, rtol=2e-3)
        assert_close(k * s * s, 3.0 / 20.0, atol=4e-3, rtol=4e-3)
    ok("large-s cancellation-free asymptotics")


def contact_coefficient_check() -> None:
    # In physical units, 2 nu0 vF b = Q B/(2 pi^2 c).
    q_charge = -1.6
    b_field = 0.07
    c = 2.4
    kf = 1.3
    vf = 0.8
    ua = 0.23 - 0.04j
    nu0 = kf * kf / (2.0 * math.pi**2 * vf)
    bdim = q_charge * b_field / (2.0 * c * kf * kf)
    lhs = 2.0 * nu0 * vf * bdim * ua
    rhs = q_charge * b_field * ua / (2.0 * math.pi**2 * c)
    assert_close(lhs, rhs, atol=2e-16, rtol=2e-16)
    vec = contact_number_current(ua, q_charge, np.array([0.0, 0.0, b_field]), c)
    assert_close(vec[2], rhs, atol=2e-16, rtol=2e-16)
    ok("consistent contact/backflow current normalization")




def physical_susceptibility_normalization_check() -> None:
    s = 1.73
    b = 0.017
    fc = 0.9
    fa = 1.35
    nu0 = 0.41
    dimensionless = charge_response(s, b, fc, fa)
    physical = physical_charge_susceptibility(s, b, fc, fa, nu0)
    assert_close(physical, 2.0 * nu0 * dimensionless, atol=2e-15, rtol=2e-15)

    # A common node source U has orthonormal component sqrt(2) U, while
    # n_total=sqrt(2) n_c, giving the same factor two independently.
    source_nodes = np.array([1.0, 1.0], dtype=complex)
    transform = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)
    source_ca = transform @ source_nodes
    assert_close(source_ca, np.array([math.sqrt(2.0), 0.0]), atol=2e-15)
    total_projector_ca = source_nodes @ transform.T
    assert_close(total_projector_ca, np.array([math.sqrt(2.0), 0.0]), atol=2e-15)
    ok("orthonormal-basis to physical two-node susceptibility normalization")


def main() -> None:
    lattice_regularization_check()
    retarded_branch_check()
    symbolic_band_expansion_check()
    magnetic_and_electric_dipole_check()
    mixed_dipole_reciprocity_check()
    berry_flux_and_measure_check()
    bare_kernel_check()
    finite_difference_coefficients_check()
    current_anomaly_moment_check()
    multipole_continuity_check()
    gradient_interaction_check()
    matrix_continuity_and_anomaly_check()
    dressed_continuity_check()
    c2_check()
    longitudinal_tensor_wti_check()
    large_s_check()
    contact_coefficient_check()
    physical_susceptibility_normalization_check()
    print("[ok] all 18 gauge-completion checks passed")


if __name__ == "__main__":
    main()
