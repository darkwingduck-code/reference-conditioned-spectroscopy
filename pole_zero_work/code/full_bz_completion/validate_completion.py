#!/usr/bin/env python3
"""Regression tests for fixed-density, full-band Ward, and Gamma^omega projection.

This suite deliberately combines algebraic tests with fresh small-lattice tests.
The production-size data are read from data_prl/full_bz_completion; small
independent calculations guard against stale summaries or hard-coded values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from ensemble import fixed_density_mu, fixed_density_response, hall_compensated
from lattice_kubo import (
    SlabParams,
    diagonalize_slab,
    density,
    density_kubo,
    ward_tensor,
    rpa_dress,
)
from landau_projection import (
    FermiSurface,
    apply_phases,
    landau_operator,
    mode_diagnostics,
    project_gamma,
    project_subspaces,
    rotate_density,
    rotate_projected,
    rotate_subspaces,
    subspace_energy,
)

DATA = ROOT / "data_prl" / "full_bz_completion"
LOG = DATA / "full_bz_completion_validation.log"


class Tester:
    def __init__(self) -> None:
        self.n = 0
        self.lines: list[str] = []

    def check(self, condition: bool, name: str, detail: str = "") -> None:
        self.n += 1
        if not bool(condition):
            raise AssertionError(f"FAIL {self.n:02d}: {name}: {detail}")
        self.lines.append(f"PASS {self.n:02d}: {name}" + (f" [{detail}]" if detail else ""))

    def close(self, value, target, tol: float, name: str) -> None:
        err = float(np.max(np.abs(np.asarray(value) - np.asarray(target))))
        self.check(err <= tol, name, f"max_abs_err={err:.3e}, tol={tol:.3e}")


def rand_unitary(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    phase = np.where(np.abs(d) > 0, d / np.abs(d), 1.0)
    return q @ np.diag(phase.conj())


def test_ensemble(t: Tester) -> None:
    rng = np.random.default_rng(260623)
    # Exact polynomial identity through B^2 for scalar and matrix responses.
    n0p, n0pp, n1, n1p, n2 = 1.7, -0.4, 0.23, -0.11, 0.37
    m = fixed_density_mu(n0p=n0p, n0pp=n0pp, n1=n1, n1p=n1p, n2=n2)
    t.close(m.mu1, -n1 / n0p, 1e-15, "general fixed-density mu1")
    target_mu2 = -(n2 + m.mu1 * n1p + 0.5 * m.mu1**2 * n0pp) / n0p
    t.close(m.mu2, target_mu2, 1e-15, "general fixed-density mu2")

    K0 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    K0p = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    K0pp = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    K1 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    K1p = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    K2 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    _, K1n, K2n = fixed_density_response(
        K0=K0,
        K0p=K0p,
        K0pp=K0pp,
        K1=K1,
        K1p=K1p,
        K2=K2,
        mu1=m.mu1,
        mu2=m.mu2,
    )
    t.close(K1n, K1 + m.mu1 * K0p, 1e-14, "matrix fixed-density K1")
    t.close(
        K2n,
        K2 + m.mu1 * K1p + m.mu2 * K0p + 0.5 * m.mu1**2 * K0pp,
        1e-14,
        "matrix fixed-density K2",
    )
    mu2e, K2e = hall_compensated(n0p=1.9, n2=-0.17, K0p=K0p, K2=K2)
    t.close(mu2e, 0.17 / 1.9, 1e-15, "Hall-compensated mu2")
    t.close(K2e, K2 + mu2e * K0p, 1e-14, "Hall-compensated K2")

    s = json.loads((DATA / "slab_chain_rule_summary.json").read_text())
    t.check(s["mu2_relerr"] < 1e-6, "production mu chain rule", f"relerr={s['mu2_relerr']:.3e}")
    t.check(s["K2_relerr"] < 1e-6, "production K chain rule", f"relerr={s['K2_relerr']:.3e}")
    t.check(np.sign(s["K2mu_re"]) != np.sign(s["K2n_re"]), "ensemble sign reversal is data-driven")

    fits = pd.read_csv(DATA / "slab_full_band_fits.csv")
    for ens in ("fixed_mu", "fixed_n"):
        d = fits[fits.ensemble == ens].set_index("component")
        summed = d.loc["conduction", ["c2_re", "c2_im"]].to_numpy(float)
        summed += d.loc["interband", ["c2_re", "c2_im"]].to_numpy(float)
        summed += d.loc["other", ["c2_re", "c2_im"]].to_numpy(float)
        total = d.loc["total", ["c2_re", "c2_im"]].to_numpy(float)
        t.close(summed, total, 1e-9, f"full-band component sum ({ens})")


def test_fresh_lattice_ward(t: Tester) -> None:
    # Fresh independent small-lattice calculation, not a readback of stored data.
    p = SlabParams(Lx=8, Nky=10, Nkz=12, theta_y=0.219, theta_z=0.347)
    spec = diagonalize_slab(p, B=0.0031)
    mu, temp, qshift = 0.63, 0.08, 1
    q = 2 * np.pi * qshift / p.Nkz
    omega = 1.7 * q + 0.031j
    wr = ward_tensor(spec, mu, temp, qshift, omega)
    t.check(wr.vertex < 5e-12, "fresh lattice one-body WTI", f"res={wr.vertex:.3e}")
    t.check(wr.left < 5e-12, "fresh bare left Ward identity", f"res={wr.left:.3e}")
    t.check(wr.right < 5e-12, "fresh bare right Ward identity", f"res={wr.right:.3e}")
    Q = 2 * np.sin(q / 2)
    qv = np.array([omega, -Q])
    for V in (0.2, 0.9, -0.3):
        K = rpa_dress(wr.K, V)
        t.check(np.max(np.abs(qv @ K)) < 5e-12, f"fresh RPA left Ward V={V:g}")
        t.check(np.max(np.abs(K @ qv)) < 5e-12, f"fresh RPA right Ward V={V:g}")

    # Full-band response is finite and includes both intra- and interband pieces.
    c = density_kubo(spec, mu, temp, qshift, 1.4j * q)
    t.check(np.isfinite([c.total.real, c.total.imag]).all(), "fresh full-band Kubo finite")
    t.close(c.conduction + c.interband + c.other, c.total, 5e-13, "fresh Kubo decomposition")
    t.check(0 < density(spec, mu, temp) < 2, "fresh lattice density physical")

    slab = json.loads((DATA / "slab_chain_rule_summary.json").read_text())
    torus = json.loads((DATA / "magnetic_torus_summary.json").read_text())
    t.check(slab["max_vertex"] < 1e-11, "production slab vertex WTI")
    t.check(max(slab["max_left"], slab["max_right"]) < 1e-12, "production slab response WTI")
    t.check(torus["max_vertex"] < 1e-11, "production torus vertex WTI")
    t.check(max(torus["max_left"], torus["max_right"]) < 1e-12, "production torus response WTI")


def test_landau_projection(t: Tester) -> None:
    rng = np.random.default_rng(260624)
    P, A = 7, 3
    states = rng.normal(size=(P, A)) + 1j * rng.normal(size=(P, A))
    states /= np.linalg.norm(states, axis=1)[:, None]
    fs = FermiSurface(
        states=states,
        area=np.linspace(0.4, 1.1, P),
        velocity=np.linspace(0.7, 1.4, P),
        residue=np.linspace(0.55, 0.9, P),
        node=np.array([1, 1, 1, -1, -1, -1, -1]),
    )
    G = rng.normal(size=(P, P, A, A, A, A)) + 1j * rng.normal(size=(P, P, A, A, A, A))
    G = 0.5 * (G + G.transpose(1, 0, 3, 2, 5, 4).conj())
    f = project_gamma(fs, G)
    L = landau_operator(fs, f)
    t.close(L, L.conj().T, 2e-12, "Landau operator Hermiticity")
    phase = rng.uniform(-np.pi, np.pi, P)
    f2 = project_gamma(apply_phases(fs, phase), G)
    t.close(f2, f, 2e-11, "U(1) band-gauge invariance")
    e1 = np.linalg.eigvalsh(L)
    e2 = np.linalg.eigvalsh(landau_operator(apply_phases(fs, phase), f2))
    t.close(e2, e1, 2e-11, "U(1) Landau spectrum invariance")
    d = mode_diagnostics(fs, f, {1: 1.0, -1: -1.0})
    t.check(0 <= d["relative_leakage"] <= 1, "two-mode leakage bounded")

    # Fresh non-Abelian U(2) covariance test.
    P2, AO, R = 3, 4, 2
    U = np.empty((P2, AO, R), complex)
    Z = np.empty((P2, R, R), complex)
    W = np.empty((P2, R, R), complex)
    N = np.empty((P2, R, R), complex)
    for i in range(P2):
        U[i], _ = np.linalg.qr(rng.normal(size=(AO, R)) + 1j * rng.normal(size=(AO, R)))
        zrot = rand_unitary(rng, R)
        Z[i] = zrot @ np.diag([0.52, 0.83]) @ zrot.conj().T
        W[i] = rand_unitary(rng, R)
        a = rng.normal(size=(R, R)) + 1j * rng.normal(size=(R, R))
        N[i] = 0.5 * (a + a.conj().T)
    GG = rng.normal(size=(P2, P2, AO, AO, AO, AO)) + 1j * rng.normal(size=(P2, P2, AO, AO, AO, AO))
    GG = 0.5 * (GG + GG.transpose(1, 0, 3, 2, 5, 4).conj())
    ff = project_subspaces(U, Z, GG)
    U2, Z2 = rotate_subspaces(U, Z, W)
    ff2 = project_subspaces(U2, Z2, GG)
    pred = rotate_projected(ff, W)
    t.close(ff2, pred, 2e-11, "fresh U(2) projected-tensor covariance")
    N2 = rotate_density(N, W)
    t.close(subspace_energy(ff, N), subspace_energy(ff2, N2), 2e-11, "fresh U(2) Landau-energy invariance")

    s = json.loads((DATA / "exact_landau_projection_summary.json").read_text())
    t.check(s["U1_projection_residual"] < 1e-10, "production U(1) projection covariance")
    t.check(s["U1_eigenvalue_residual"] < 1e-12, "production U(1) spectrum covariance")
    t.check(s["nonabelian_covariance_residual"] < 1e-11, "production U(2) tensor covariance")
    t.check(s["nonabelian_energy_residual"] < 1e-11, "production U(2) energy covariance")
    t.check(0 < s["relative_two_mode_leakage"] < 0.2, "production scalar-reduction leakage quantified", f"eps={s['relative_two_mode_leakage']:.4f}")
    t.check(s["leading_charge_overlap"][0] > 0.95, "leading charge eigenmode identified")
    t.check(s["leading_axial_overlap"][1] > 0.95, "leading axial eigenmode identified")


def test_convergence_metadata(t: Tester) -> None:
    conv = pd.read_csv(DATA / "slab_ky_convergence.csv").sort_values("Nky")
    t.check({64, 96}.issubset(set(conv.Nky)), "dense transverse grids present")
    r64 = float(conv.loc[conv.Nky == 64, "K2_fixed_n_real"].iloc[0])
    r96 = float(conv.loc[conv.Nky == 96, "K2_fixed_n_real"].iloc[0])
    t.check(abs(r64 - r96) / abs(r96) < 0.01, "Nky=64 to 96 fixed-n convergence", f"rel={abs(r64-r96)/abs(r96):.3e}")
    t.check(float(conv.loc[conv.Nky == 12, "K2_fixed_n_real"].iloc[0]) != r96, "coarse-grid artifact exposed")
    size = pd.read_csv(DATA / "slab_size_convergence.csv")
    t.check(size.Lx.nunique() >= 3, "three slab widths recorded")
    t.check(np.isfinite(size.select_dtypes(include=[float, int]).to_numpy()).all(), "convergence tables finite")


def main() -> None:
    t = Tester()
    test_ensemble(t)
    test_fresh_lattice_ward(t)
    test_landau_projection(t)
    test_convergence_metadata(t)
    t.lines.append(f"ALL {t.n} TESTS PASSED")
    LOG.write_text("\n".join(t.lines) + "\n")
    print("\n".join(t.lines))


if __name__ == "__main__":
    main()
