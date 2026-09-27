#!/usr/bin/env python3
"""Validated solver for the reduced two-node Weyl Fermi-liquid model.

The implementation deliberately separates three questions:

1. Undamped collective poles are accepted only on the real axis with
   s = omega/(v_F q) > 1.
2. Spectroscopy is evaluated on the retarded real-frequency axis.
3. The finite-field kernel is a kinematic Berry-streaming model, not a
   complete conserving O(B) chiral kinetic theory.

The density of states is normalized per Weyl node,
nu_0 = mu^2/(2 pi^2 v_F^3).  Consequently
F_C = nu_0 V_C = 2 alpha/[pi(qbar^2+qTFbar^2)], and the charge sector
receives 2 F_C.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence
import json

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.special import roots_legendre
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class ModelParams:
    F0c: float = 0.0
    F0a: float = 1.5
    F1c: float = 0.4
    F1a: float = 1.0
    alpha: float = 0.02
    qTFbar: float = 0.0
    eta_bar: float = 1.0e-4
    sigma_tol: float = 3.0e-9
    relative_residual_tol: float = 3.0e-9
    root_merge_tol: float = 2.0e-6
    quadrature_order: int = 128
    field_scheme: str = "mobius"

    @property
    def Fd(self) -> float:
        return 0.5 * (self.F0c + self.F0a)

    @property
    def Fx(self) -> float:
        return 0.5 * (self.F0c - self.F0a)

    @property
    def F1d(self) -> float:
        return 0.5 * (self.F1c + self.F1a)

    @property
    def F1x(self) -> float:
        return 0.5 * (self.F1c - self.F1a)


@dataclass
class Mode:
    qbar: float
    b: float
    s: float
    omega_bar: float
    sigma_min: float
    sigma_max: float
    relative_residual: float
    det_abs: float
    charge_character: float
    nullity: int
    vector_real: list[float]

    @property
    def charge_weight(self) -> float:
        """Backward-compatible alias; this is not a spectroscopic residue."""
        return self.charge_character

    def to_dict(self) -> dict:
        out = asdict(self)
        out["charge_weight"] = self.charge_character
        return out


class ProjectedModel:
    """Angularly projected source/readout formulation."""

    def __init__(self, params: ModelParams, lmax: int = 2):
        if lmax < 1:
            raise ValueError("lmax must be at least 1")
        if params.field_scheme not in {"linear", "mobius"}:
            raise ValueError("field_scheme must be 'linear' or 'mobius'")
        self.params = params
        self.lmax = int(lmax)
        self.x, w = roots_legendre(params.quadrature_order)
        self.w = w / 2.0
        self.Y = np.vstack([self._basis(l, self.x) for l in range(self.lmax + 1)])

    @staticmethod
    def _basis(l: int, x: np.ndarray) -> np.ndarray:
        coeff = np.zeros(l + 1)
        coeff[l] = 1.0
        return np.sqrt(2 * l + 1.0) * legval(x, coeff)

    @staticmethod
    def landau_D(z: complex) -> complex:
        """Retarded Landau function on the physical sheet."""
        z = complex(z)
        if abs(z) > 18.0:
            inv2 = 1.0 / (z * z)
            return inv2 * (
                1 / 3
                + inv2 * (1 / 5 + inv2 * (1 / 7 + inv2 * (1 / 9 + inv2 / 11)))
            )
        return 0.5 * z * np.log((z + 1.0) / (z - 1.0)) - 1.0

    def _project_factor(self, factor: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                [
                    np.sum(self.w * self.Y[l] * factor * self.Y[j])
                    for j in range(2)
                ]
                for l in range(self.lmax + 1)
            ],
            dtype=complex,
        )

    def _kernel_linear(self, s: complex, bchi: float) -> np.ndarray:
        factor = self.x / (s - self.x)
        factor += bchi * s * (1.0 - self.x * self.x) / (s - self.x) ** 2
        return self._project_factor(factor)

    def _kernel_mobius_quadrature(self, s: complex, bchi: float) -> np.ndarray:
        r = (self.x + bchi) / (1.0 + bchi * self.x)
        return self._project_factor(r / (s - r))

    def _kernel_mobius(self, s: complex, bchi: float) -> np.ndarray:
        if abs(bchi) >= 1.0:
            raise ValueError("The semiclassical projection requires |b|<1.")
        den = 1.0 - bchi * s

        # The closed form contains cancellations when 1-b*s is small.
        # A deliberately conservative fallback eliminates the percent-level
        # loss of precision found in the earlier implementation.
        if abs(den) < 5.0e-4 * max(1.0, abs(bchi * s)):
            return self._kernel_mobius_quadrature(s, bchi)

        z = (s - bchi) / den
        if abs(z) < 5.0e-4:
            return self._kernel_mobius_quadrature(s, bchi)

        D = self.landau_D(z)
        K00 = (D + (bchi / z) * (D + 1.0)) / den
        K01 = np.sqrt(3.0) * (z + bchi) * D / den
        K11 = (3.0 * z * (z + bchi) * D - 1.0) / den
        rows = [[K00, K01], [K01, K11]]
        if self.lmax >= 2:
            K20 = (np.sqrt(5.0) / 2.0) * (
                (3 * z * z + 3 * bchi * z - 1.0) * D
                - 1.0
                - (bchi / z) * (D + 1.0)
            ) / den
            K21 = (np.sqrt(15.0) / 2.0) * (z + bchi) * (
                (3 * z * z - 1.0) * D - 1.0
            ) / den
            rows.append([K20, K21])
        if self.lmax > 2:
            return self._kernel_mobius_quadrature(s, bchi)
        return np.asarray(rows, dtype=complex)

    def kernel(self, s: complex, bchi: float) -> np.ndarray:
        if abs(bchi) >= 1.0:
            raise ValueError("The semiclassical projection requires |b|<1.")
        if self.params.field_scheme == "linear":
            return self._kernel_linear(s, bchi)
        return self._kernel_mobius(s, bchi)

    def readout(self, bchi: float) -> np.ndarray:
        """Rows are m0=<G u> and m1=<Y1 G u>, G=1+bchi*x."""
        R = np.zeros((2, self.lmax + 1), dtype=complex)
        R[0, 0] = 1.0
        R[0, 1] = bchi / np.sqrt(3.0)
        R[1, 0] = bchi / np.sqrt(3.0)
        R[1, 1] = 1.0
        if self.lmax >= 2:
            R[1, 2] = 2.0 * bchi / np.sqrt(15.0)
        return R

    def node_response(self, s: complex, bchi: float) -> np.ndarray:
        return self.readout(bchi) @ self.kernel(s, bchi)

    @staticmethod
    def _interaction_from_channels(
        F0c: float, F0a: float, F1c: float, F1a: float
    ) -> np.ndarray:
        Fd, Fx = 0.5 * (F0c + F0a), 0.5 * (F0c - F0a)
        F1d, F1x = 0.5 * (F1c + F1a), 0.5 * (F1c - F1a)
        return np.asarray(
            [
                [Fd, 0.0, Fx, 0.0],
                [0.0, F1d, 0.0, F1x],
                [Fx, 0.0, Fd, 0.0],
                [0.0, F1x, 0.0, F1d],
            ],
            dtype=complex,
        )

    def coulomb_FC(self, qbar: float) -> float:
        den = qbar * qbar + self.params.qTFbar * self.params.qTFbar
        if den <= 0.0:
            raise ValueError("qbar^2+qTFbar^2 must be positive")
        return 2.0 * self.params.alpha / (np.pi * den)

    def interaction_sr(self) -> np.ndarray:
        p = self.params
        return self._interaction_from_channels(p.F0c, p.F0a, p.F1c, p.F1a)

    def interaction_full(self, qbar: float) -> np.ndarray:
        T = self.interaction_sr().copy()
        FC = self.coulomb_FC(qbar)
        for row in (0, 2):
            for col in (0, 2):
                T[row, col] += FC
        return T

    def response_block(self, s: complex, b: float) -> np.ndarray:
        A = np.zeros((4, 4), dtype=complex)
        A[:2, :2] = self.node_response(s, +b)
        A[2:, 2:] = self.node_response(s, -b)
        return A

    def secular(
        self, s: complex, qbar: float, b: float, *, short_range: bool = False
    ) -> np.ndarray:
        A = self.response_block(s, b)
        T = self.interaction_sr() if short_range else self.interaction_full(qbar)
        return np.eye(4, dtype=complex) - A @ T

    @staticmethod
    def singular_values(M: np.ndarray) -> np.ndarray:
        return np.linalg.svd(M, compute_uv=False)

    @staticmethod
    def null_vector(M: np.ndarray) -> np.ndarray:
        _, _, vh = np.linalg.svd(M)
        v = vh[-1, :].conj()
        n = np.linalg.norm(v)
        return v / n if n else v

    @staticmethod
    def charge_character(v: np.ndarray) -> float:
        """Internal eigenvector diagnostic, not an oscillator strength."""
        c = (v[0] + v[2]) / np.sqrt(2.0)
        a = (v[0] - v[2]) / np.sqrt(2.0)
        den = abs(c) ** 2 + abs(a) ** 2
        return float(abs(c) ** 2 / den) if den > 0 else float("nan")

    def _mode_at(self, s: float, qbar: float, b: float) -> Mode:
        M = self.secular(s, qbar, b)
        sv = self.singular_values(M)
        v = self.null_vector(M)
        sigma_max = float(sv[0])
        sigma_min = float(sv[-1])
        relative = sigma_min / sigma_max if sigma_max > 0 else float("inf")
        nullity = int(
            np.count_nonzero(sv < max(30.0 * self.params.sigma_tol, 1e-7))
        )
        phase = np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
        vr = np.real_if_close(v * phase, tol=1e4).real
        return Mode(
            qbar=float(qbar),
            b=float(b),
            s=float(s),
            omega_bar=float(qbar * s),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            relative_residual=float(relative),
            det_abs=float(abs(np.linalg.det(M))),
            charge_character=self.charge_character(v),
            nullity=nullity,
            vector_real=[float(z) for z in vr],
        )

    def _accept(self, mode: Mode) -> bool:
        return (
            mode.sigma_min <= self.params.sigma_tol
            and mode.relative_residual <= self.params.relative_residual_tol
        )

    def real_modes(
        self,
        qbar: float,
        b: float,
        *,
        smax: float | None = None,
        n_linear: int = 700,
        n_log: int = 420,
    ) -> list[Mode]:
        """Find simple and tangent real poles with s>1."""
        if qbar <= 0:
            raise ValueError("qbar must be positive")
        if abs(b) >= 1:
            raise ValueError("|b| must be below one")
        if smax is None:
            gap_est = np.sqrt(
                max(
                    1e-12,
                    4
                    * self.params.alpha
                    * (1 + self.params.F1c)
                    / (3 * np.pi),
                )
            )
            smax = max(5.0, 2.5 * gap_est / qbar)

        s_lo = 1.0 + 2.0e-6
        split = min(3.0, smax)
        pieces = [np.linspace(s_lo, split, n_linear)]
        if smax > split:
            pieces.append(np.geomspace(split, smax, n_log))
        grid = np.unique(np.concatenate(pieces))

        detv = np.empty(grid.size)
        sigv = np.empty(grid.size)
        for i, s in enumerate(grid):
            M = self.secular(float(s), qbar, b)
            detv[i] = float(np.real(np.linalg.det(M)))
            sigv[i] = float(self.singular_values(M)[-1])

        candidates: list[float] = []
        for i in range(grid.size - 1):
            a, c = detv[i], detv[i + 1]
            if not (np.isfinite(a) and np.isfinite(c)):
                continue
            if a == 0.0:
                candidates.append(float(grid[i]))
            elif a * c < 0.0:
                try:
                    candidates.append(
                        float(
                            brentq(
                                lambda z: float(
                                    np.real(np.linalg.det(self.secular(z, qbar, b)))
                                ),
                                float(grid[i]),
                                float(grid[i + 1]),
                                xtol=2e-13,
                                rtol=2e-13,
                            )
                        )
                    )
                except (ValueError, FloatingPointError):
                    pass

        tiny = np.finfo(float).tiny
        for i in range(1, grid.size - 1):
            if sigv[i] <= sigv[i - 1] and sigv[i] <= sigv[i + 1]:
                try:
                    opt = minimize_scalar(
                        lambda z: np.log(
                            max(
                                self.singular_values(self.secular(z, qbar, b))[-1],
                                tiny,
                            )
                        ),
                        bounds=(float(grid[i - 1]), float(grid[i + 1])),
                        method="bounded",
                        options={"xatol": 5e-13, "maxiter": 200},
                    )
                    if opt.success:
                        candidates.append(float(opt.x))
                except (ValueError, FloatingPointError):
                    pass

        raw = [
            self._mode_at(s, qbar, b)
            for s in candidates
            if s > 1.0 + 5e-7 and np.isfinite(s)
        ]
        raw.sort(key=lambda m: m.s)
        clusters: list[list[Mode]] = []
        cluster_tol = max(self.params.root_merge_tol, 8.0e-5)
        for mode in raw:
            if not clusters or abs(mode.s - clusters[-1][-1].s) > cluster_tol:
                clusters.append([mode])
            else:
                clusters[-1].append(mode)
        modes = [min(group, key=lambda m: m.relative_residual) for group in clusters]
        modes = [m for m in modes if self._accept(m)]
        modes.sort(key=lambda m: m.omega_bar)
        return modes

    def bracketed_real_modes(
        self,
        qbar: float,
        b: float,
        *,
        smax: float | None = None,
        n_linear: int = 260,
        n_log: int = 140,
    ) -> list[Mode]:
        """Fast simple-root search, with singular-value acceptance."""
        if qbar <= 0:
            raise ValueError("qbar must be positive")
        if abs(b) >= 1:
            raise ValueError("|b| must be below one")
        if smax is None:
            gap_est = np.sqrt(
                max(
                    1e-12,
                    4
                    * self.params.alpha
                    * (1 + self.params.F1c)
                    / (3 * np.pi),
                )
            )
            smax = max(5.0, 2.5 * gap_est / qbar)
        s_lo = 1.0 + 2.0e-6
        split = min(3.0, smax)
        pieces = [np.linspace(s_lo, split, n_linear)]
        if smax > split:
            pieces.append(np.geomspace(split, smax, n_log))
        grid = np.unique(np.concatenate(pieces))

        detv = np.asarray(
            [
                float(np.real(np.linalg.det(self.secular(float(s), qbar, b))))
                for s in grid
            ]
        )
        roots: list[float] = []
        for i in range(grid.size - 1):
            a, c = detv[i], detv[i + 1]
            if not (np.isfinite(a) and np.isfinite(c)) or a * c >= 0.0:
                continue
            try:
                roots.append(
                    float(
                        brentq(
                            lambda z: float(
                                np.real(np.linalg.det(self.secular(z, qbar, b)))
                            ),
                            float(grid[i]),
                            float(grid[i + 1]),
                            xtol=2e-13,
                            rtol=2e-13,
                        )
                    )
                )
            except (ValueError, FloatingPointError):
                pass

        roots.sort()
        merged: list[float] = []
        for root in roots:
            if not merged or abs(root - merged[-1]) > self.params.root_merge_tol:
                merged.append(root)
        modes = [self._mode_at(root, qbar, b) for root in merged]
        modes = [m for m in modes if self._accept(m)]
        modes.sort(key=lambda m: m.omega_bar)
        return modes

    def proper_polarization_tilde(
        self, omega_bar: complex, qbar: float, b: float
    ) -> complex:
        s = omega_bar / qbar
        A = self.response_block(s, b)
        source = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=complex)
        detect = source.copy()
        m = np.linalg.solve(
            np.eye(4, dtype=complex) - A @ self.interaction_sr(), A @ source
        )
        return complex(detect @ m)

    def dielectric(self, omega_bar: complex, qbar: float, b: float) -> complex:
        return 1.0 - self.coulomb_FC(qbar) * self.proper_polarization_tilde(
            omega_bar, qbar, b
        )

    def loss(
        self,
        omega_grid: Sequence[float],
        qbar: float,
        b: float,
        eta_bar: float | None = None,
    ) -> np.ndarray:
        eta = self.params.eta_bar if eta_bar is None else float(eta_bar)
        vals = []
        for w in omega_grid:
            eps = self.dielectric(complex(w, eta), qbar, b)
            vals.append(float(-np.imag(1.0 / eps)))
        return np.asarray(vals)

    def dielectric_residue(
        self, omega_bar: float, qbar: float, b: float, *, step: float = 2.0e-7
    ) -> float:
        """Residue of epsilon^{-1} at a simple real pole."""
        h = max(step, 2.0e-7 * max(1.0, abs(omega_bar)))
        deriv = (
            self.dielectric(omega_bar + h, qbar, b)
            - self.dielectric(omega_bar - h, qbar, b)
        ) / (2.0 * h)
        residue = 1.0 / deriv
        if abs(residue.imag) > 2e-7 * max(1.0, abs(residue.real)):
            raise RuntimeError("Unexpected complex real-axis dielectric residue")
        return float(residue.real)


def landau_D_real(s: float) -> float:
    return float(0.5 * s * np.log((s + 1.0) / (s - 1.0)) - 1.0)


def sector_determinant(s: float, F0: float, F1: float) -> float:
    D = landau_D_real(s)
    return float(1.0 + F1 - D * (F0 * (1.0 + F1) + 3.0 * F1 * s * s))


def sector_root(F0: float, F1: float, smax: float = 1.0e4) -> float:
    """Unique undamped sector root when the endpoint signs permit one."""
    lo = 1.0 + 1.0e-10
    flo = sector_determinant(lo, F0, F1)
    hi = 2.0
    fhi = sector_determinant(hi, F0, F1)
    while flo * fhi > 0.0 and hi < smax:
        hi *= 2.0
        fhi = sector_determinant(hi, F0, F1)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0.0:
        raise RuntimeError(f"No sector root found for F0={F0}, F1={F1}")
    return float(
        brentq(
            lambda z: sector_determinant(z, F0, F1),
            lo,
            hi,
            xtol=1e-13,
            rtol=2e-14,
        )
    )


def nominal_crossing(params: ModelParams) -> tuple[float, float, float]:
    """Exact b=0 crossing, including optional background qTFbar."""
    sa = sector_root(params.F0a, params.F1a)
    D = landau_D_real(sa)
    F0c_star = (
        (1.0 + params.F1c) / D - 3.0 * params.F1c * sa * sa
    ) / (1.0 + params.F1c)
    needed = F0c_star - params.F0c
    if needed <= 0.0:
        raise ValueError("No positive Coulomb crossing for this parameter set")
    q2 = 4.0 * params.alpha / (np.pi * needed) - params.qTFbar**2
    if q2 <= 0.0:
        raise ValueError("Screening removes the positive-q crossing")
    qstar = np.sqrt(q2)
    return float(qstar), float(sa), float(qstar * sa)


def _parity_embeddings() -> tuple[np.ndarray, np.ndarray]:
    Pc = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=complex) / np.sqrt(2)
    Pa = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=complex) / np.sqrt(2)
    return Pc, Pa


def _left_right_null(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U, _, Vh = np.linalg.svd(A)
    left = U[:, -1]
    right = Vh[-1, :].conj()
    overlap = np.vdot(left, right)
    if abs(overlap) < 1e-12:
        raise RuntimeError("Left/right null vectors have vanishing overlap")
    left = left / np.conj(overlap)
    return left, right


def degenerate_perturbation_data(
    params: ModelParams, *, derivative_step: float = 1.0e-6
) -> dict:
    """Project dM/domega and dM/db onto the crossing null space."""
    model = ProjectedModel(params, lmax=2)
    qstar, sstar, wstar = nominal_crossing(params)
    Pc, Pa = _parity_embeddings()
    M0 = model.secular(sstar, qstar, 0.0)

    lc2, rc2 = _left_right_null(Pc.conj().T @ M0 @ Pc)
    la2, ra2 = _left_right_null(Pa.conj().T @ M0 @ Pa)
    lc, rc = Pc @ lc2, Pc @ rc2
    la, ra = Pa @ la2, Pa @ ra2

    h = derivative_step
    dMds = (
        model.secular(sstar + h, qstar, 0.0)
        - model.secular(sstar - h, qstar, 0.0)
    ) / (2.0 * h)
    dMdw = dMds / qstar
    dMdb = (
        model.secular(sstar, qstar, +h)
        - model.secular(sstar, qstar, -h)
    ) / (2.0 * h)

    Zc = np.vdot(lc, dMdw @ rc)
    Za = np.vdot(la, dMdw @ ra)
    Vca = np.vdot(lc, dMdb @ ra)
    Vac = np.vdot(la, dMdb @ rc)
    ratio = Vca * Vac / (Zc * Za)
    coefficient = 2.0 * np.sqrt(ratio)

    vals = [Zc, Za, Vca, Vac, coefficient]
    if max(abs(z.imag) for z in vals) > 2e-8:
        raise RuntimeError("Perturbative crossing data unexpectedly complex")
    if coefficient.real <= 0:
        raise RuntimeError("Nonpositive avoided-crossing coefficient")

    return {
        "qstar": qstar,
        "sstar": sstar,
        "omega_star": wstar,
        "Zc": float(Zc.real),
        "Za": float(Za.real),
        "Vca": float(Vca.real),
        "Vac": float(Vac.real),
        "gap_linear_coefficient": float(coefficient.real),
    }


def phase_velocity_gap_coefficient(params: ModelParams) -> float:
    """Leading avoided-crossing gap in phase velocity, Delta s/|b_parallel|.

    At fixed short-range Landau parameters the zero-field crossing fixes the
    on-shell effective charge interaction. Coulomb strength and background
    screening then move qstar but do not change this reduced-model coefficient.
    """
    data = degenerate_perturbation_data(params)
    return float(data["gap_linear_coefficient"] / data["qstar"])


def two_mode_residue_fractions(
    delta: float | np.ndarray,
    two_g: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical charge-residue fractions of lower and upper hybrid poles."""
    delta_arr = np.asarray(delta, dtype=float)
    two_g_arr = np.asarray(two_g, dtype=float)
    den = np.sqrt(delta_arr * delta_arr + two_g_arr * two_g_arr)
    if np.any(den <= 0.0):
        raise ValueError("delta and two_g cannot both vanish")
    upper = 0.5 * (1.0 + delta_arr / den)
    lower = 1.0 - upper
    return lower, upper


def damped_two_mode_poles(
    omega_c: float,
    omega_a: float,
    gamma_c: float,
    gamma_a: float,
    coupling: float,
) -> tuple[complex, complex]:
    """Poles of a phenomenologically damped reciprocal two-mode model.

    Positive gamma values are bare half widths. This helper is not a
    microscopic damping calculation; it is used only for the resolvability
    criterion of the effective coupled-mode theory.
    """
    if gamma_c < 0.0 or gamma_a < 0.0 or coupling < 0.0:
        raise ValueError("damping rates and coupling must be nonnegative")
    center = 0.5 * (omega_c + omega_a) - 0.5j * (gamma_c + gamma_a)
    detuning = (omega_c - omega_a) - 1j * (gamma_c - gamma_a)
    root = np.sqrt(complex(coupling * coupling + 0.25 * detuning * detuning))
    poles = (center - root, center + root)
    return tuple(sorted(poles, key=lambda z: z.real))


def minimize_gap(
    model: ProjectedModel,
    b: float,
    *,
    qcenter: float | None = None,
    relative_half_width: float = 0.18,
) -> dict:
    """Minimize the finite-b separation of the two accepted real poles."""
    if b == 0:
        qstar, _, wstar = nominal_crossing(model.params)
        return {
            "b": 0.0,
            "qmin": qstar,
            "gap": 0.0,
            "omega_lower": wstar,
            "omega_upper": wstar,
        }
    if qcenter is None:
        qcenter = nominal_crossing(model.params)[0]
    lo = max(1e-4, qcenter * (1.0 - relative_half_width))
    hi = qcenter * (1.0 + relative_half_width)

    def objective(q: float) -> float:
        modes = model.bracketed_real_modes(q, b, n_linear=340, n_log=160)
        if len(modes) < 2:
            return 1.0e3
        return modes[1].omega_bar - modes[0].omega_bar

    opt = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 2.0e-13, "maxiter": 160},
    )
    if not opt.success or not np.isfinite(opt.fun) or opt.fun >= 1.0:
        raise RuntimeError(f"Gap minimization failed for b={b}")
    modes = model.bracketed_real_modes(
        float(opt.x), b, n_linear=520, n_log=220
    )
    if len(modes) < 2:
        raise RuntimeError(f"Two poles not found at optimized q for b={b}")
    return {
        "b": float(b),
        "qmin": float(opt.x),
        "gap": float(modes[1].omega_bar - modes[0].omega_bar),
        "omega_lower": float(modes[0].omega_bar),
        "omega_upper": float(modes[1].omega_bar),
        "relative_residual_max": float(
            max(m.relative_residual for m in modes[:2])
        ),
    }


def save_metadata(
    path: str | Path, params: ModelParams, extra: dict | None = None
) -> None:
    payload = {"params": asdict(params), "extra": extra or {}}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = ModelParams()
    model = ProjectedModel(p, lmax=2)
    qstar, sa, wstar = nominal_crossing(p)
    result = {
        "qstar": qstar,
        "s_axial": sa,
        "omega_crossing": wstar,
        "perturbation": degenerate_perturbation_data(p),
        "minimum_gap_b_0p02": minimize_gap(model, 0.02),
    }
    print(json.dumps(result, indent=2))
