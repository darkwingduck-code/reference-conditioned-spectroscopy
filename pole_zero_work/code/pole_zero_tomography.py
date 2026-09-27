"""Pole-zero tomography and finite-field genealogy classifiers.

The central object is a background-subtracted scalar retarded response with
exactly two nearby simple poles,

    chi_sing(z) = C (z-z0) / [(z-z_plus)(z-z_minus)].

The principal-part numerator zero z0 and related algebraic combinations are
reconstructible from the poles and residues.  Calling those combinations a
bare companion pole or a coupling product requires the exact, strictly proper
canonical two-mode scalar response written above.  The meromorphic identities
themselves remain valid for complex poles and unequal damping.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import null_space


@dataclass(frozen=True)
class PoleZeroReconstruction:
    z_minus: complex
    z_plus: complex
    residue_minus: complex
    residue_plus: complex
    prefactor: complex
    response_zero: complex
    companion_pole: complex
    coupling_product: complex


def _sort_poles(z1: complex, z2: complex) -> tuple[complex, complex]:
    """Sort primarily by real part and secondarily by imaginary part."""
    return tuple(sorted((complex(z1), complex(z2)), key=lambda z: (z.real, z.imag)))


def reconstruct_from_poles_residues(
    z1: complex,
    z2: complex,
    residue1: complex,
    residue2: complex,
    *,
    atol: float = 1.0e-14,
) -> PoleZeroReconstruction:
    """Reconstruct the local two-mode data from two poles and their residues.

    Inputs may be in either pole order.  Residues are associated with the
    corresponding input pole.  For

        chi_sing(z)=C(z-z0)/[(z-z_+)(z-z_-)],

    one has C=R_++R_- and

        z0=(R_+ z_- + R_- z_+)/(R_++R_-).

    The algebraic combinations returned in ``companion_pole`` and
    ``coupling_product`` are

        z_b=z_++z_--z0,  G^2=(z_+-z0)(z0-z_-).

    They have the bare-pole and off-diagonal-coupling interpretations only
    when the supplied response is exactly the strictly proper canonical
    two-mode scalar response, without an analytic prefactor or background.
    """
    z1, z2 = complex(z1), complex(z2)
    r1, r2 = complex(residue1), complex(residue2)
    if (z1.real, z1.imag) <= (z2.real, z2.imag):
        zm, zp, rm, rp = z1, z2, r1, r2
    else:
        zm, zp, rm, rp = z2, z1, r2, r1
    c = rm + rp
    if abs(c) <= atol:
        raise ValueError("The summed residue/prefactor is too small for tomography")
    z0 = (rp * zm + rm * zp) / c
    zb = zp + zm - z0
    g2 = (zp - z0) * (z0 - zm)
    return PoleZeroReconstruction(zm, zp, rm, rp, c, z0, zb, g2)


def poles_from_bare(z_bright: complex, z_dark: complex, coupling_product: complex) -> tuple[complex, complex]:
    """Poles of (z-z_b)(z-z_d)-G^2=0."""
    zb, zd, g2 = complex(z_bright), complex(z_dark), complex(coupling_product)
    center = 0.5 * (zb + zd)
    root = np.sqrt(0.25 * (zb - zd) ** 2 + g2)
    return _sort_poles(center - root, center + root)


def residues_from_bare(
    z_bright: complex,
    z_dark: complex,
    coupling_product: complex,
    *,
    prefactor: complex = 1.0,
) -> tuple[complex, complex, tuple[complex, complex]]:
    """Residues of the bright-channel response for a linear two-mode block."""
    zm, zp = poles_from_bare(z_bright, z_dark, coupling_product)
    c = complex(prefactor)
    rm = c * (z_dark - zm) / (zp - zm)
    rp = c * (zp - z_dark) / (zp - zm)
    return rm, rp, (zm, zp)


def singular_response(
    z: np.ndarray | complex,
    z_minus: complex,
    z_plus: complex,
    response_zero: complex,
    *,
    prefactor: complex = 1.0,
) -> np.ndarray:
    zarr = np.asarray(z, dtype=complex)
    return complex(prefactor) * (zarr - response_zero) / (
        (zarr - z_minus) * (zarr - z_plus)
    )


@dataclass(frozen=True)
class ZeroAssignmentCandidate:
    """Diagnostics for one zero of the calibrated full-response numerator."""

    root: complex
    inside_window: bool
    pole_distance_scaled: float
    cancelled: bool
    slope: complex
    slope_scaled: float
    root_condition_number: float
    well_conditioned: bool
    eligible: bool
    reason: str


@dataclass(frozen=True)
class RationalTwoPoleFit:
    """Fit result with separate principal-part and full-numerator zeros.

    ``zero_assignment_reliable`` is only a numerical screen combining the
    fit residual, design conditioning, measured window, cancellation, and
    local slope checks.  It is neither a certified error bound nor a physical
    mode assignment, and the caller must calibrate the target scalar response
    before fitting.
    """
    poles: tuple[complex, complex]
    response_zero: complex
    residues: tuple[complex, complex]
    background: tuple[complex, ...]
    denominator: tuple[complex, ...]
    numerator: tuple[complex, ...]
    relative_rms: float
    condition_number: float
    frequency_center: complex
    frequency_scale: float
    full_response_zeros: tuple[complex, ...]
    zero_candidates: tuple[ZeroAssignmentCandidate, ...]
    zero_assignment_reliable: bool
    zero_assignment_flags: tuple[str, ...]

    @property
    def principal_part_zero(self) -> complex:
        """Background-subtracted zero; response_zero is its legacy alias."""
        return self.response_zero


def _full_numerator_zero_diagnostics(
    numerator: np.ndarray,
    denominator: np.ndarray,
    poles_t: np.ndarray,
    *,
    real_window_t: tuple[float, float],
    coefficient_rtol: float = 1.0e-11,
    cancellation_tol: float = 2.0e-5,
    slope_tol: float = 1.0e-9,
) -> tuple[np.ndarray, tuple[ZeroAssignmentCandidate, ...], tuple[str, ...]]:
    """Extract full-response numerator roots and conservative diagnostics."""
    num = np.asarray(numerator, dtype=complex)
    scale = float(np.max(np.abs(num))) if num.size else 0.0
    flags: list[str] = []
    if not np.isfinite(scale) or scale <= 0.0:
        return np.array([], complex), (), ("vanishing_full_numerator",)
    nz = np.flatnonzero(np.abs(num) > coefficient_rtol * scale)
    if nz.size == 0:
        return np.array([], complex), (), ("vanishing_full_numerator",)
    trimmed = num[nz[0]:]
    if nz[0] > 0:
        flags.append("leading_numerator_coefficients_trimmed")
    if trimmed.size <= 1:
        return np.array([], complex), (), tuple(flags + ["constant_full_numerator"])
    roots = np.asarray(sorted(np.roots(trimmed), key=lambda value: (value.real, value.imag)), dtype=complex)
    dtrim = np.polyder(trimmed)
    lo, hi = real_window_t
    candidates: list[ZeroAssignmentCandidate] = []
    for root in roots:
        pole_distance = float(np.min(np.abs(root - poles_t)))
        cancelled = pole_distance <= cancellation_tol
        if cancelled:
            slope = complex(np.nan, np.nan)
            slope_scaled = 0.0
            root_condition = float(np.inf)
            well_conditioned = False
        else:
            slope = complex(np.polyval(dtrim, root) / np.polyval(denominator, root))
            slope_scaled = float(abs(slope) / scale)
            root_condition = float(np.inf if slope_scaled == 0.0 else 1.0 / slope_scaled)
            well_conditioned = bool(np.isfinite(root_condition) and slope_scaled > slope_tol)
        inside = bool(lo <= root.real <= hi)
        reasons: list[str] = []
        if not inside:
            reasons.append("outside_measured_window")
        if cancelled:
            reasons.append("pole_zero_cancellation")
        if not well_conditioned:
            reasons.append("ill_conditioned_root")
        eligible = not reasons
        candidates.append(ZeroAssignmentCandidate(
            root=complex(root),
            inside_window=inside,
            pole_distance_scaled=pole_distance,
            cancelled=cancelled,
            slope=slope,
            slope_scaled=slope_scaled,
            root_condition_number=root_condition,
            well_conditioned=well_conditioned,
            eligible=eligible,
            reason="eligible" if eligible else ";".join(reasons),
        ))
    if not any(candidate.eligible for candidate in candidates):
        flags.append("no_eligible_full_response_zero")
    return roots, tuple(candidates), tuple(flags)


def fit_two_pole_rational(
    z: np.ndarray,
    chi: np.ndarray,
    *,
    background_order: int = 1,
    normalize_frequency: bool = True,
) -> RationalTwoPoleFit:
    """Unconstrained complex least-squares fit of a local response with two poles.

    The fit does not enforce causality, passivity, pole half-plane location, or
    a microscopic interpretation of any fitted zero.

    For ``background_order=0`` the model is [2/2], while for the default
    ``background_order=1`` it is [3/2].  Polynomial division separates the
    analytic background from a linear singular numerator.  Its zero is the
    principal-part identity returned under both ``principal_part_zero`` and
    the backwards-compatible ``response_zero`` name.

    The fit is performed in the centered and scaled coordinate
    ``t=(z-z_c)/s_z`` by default.  This leaves poles, zeros, and fitted
    residues invariant after transforming back to ``z`` but substantially
    reduces avoidable Vandermonde conditioning.  The returned polynomial
    coefficients are in the normalized ``t`` coordinate; ``frequency_center``
    and ``frequency_scale`` record that map.
    """
    z = np.asarray(z, dtype=complex).ravel()
    y = np.asarray(chi, dtype=complex).ravel()
    if z.shape != y.shape or z.size < 10:
        raise ValueError("z and chi must have equal shape and at least 10 samples")
    if background_order not in (0, 1):
        raise ValueError("background_order must be 0 or 1")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
        raise ValueError("z and chi must be finite")

    if normalize_frequency:
        zc = complex(np.mean(z))
        sz = float(np.max(np.abs(z - zc)))
        if not np.isfinite(sz) or sz <= 0.0:
            raise ValueError("frequency samples must span a nonzero window")
    else:
        zc = 0.0 + 0.0j
        sz = 1.0
    t = (z - zc) / sz

    if background_order == 0:
        # y(t^2+d1 t+d0)=n2 t^2+n1 t+n0.
        A = np.column_stack([y * t, y, -t * t, -t, -np.ones_like(t)])
        rhs = -y * t * t
        x, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        d1, d0, n2, n1, n0 = x
        num = np.array([n2, n1, n0], dtype=complex)
    else:
        # y(t^2+d1 t+d0)=n3 t^3+n2 t^2+n1 t+n0.
        A = np.column_stack([y * t, y, -t**3, -t**2, -t, -np.ones_like(t)])
        rhs = -y * t * t
        x, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        d1, d0, n3, n2, n1, n0 = x
        num = np.array([n3, n2, n1, n0], dtype=complex)

    den = np.array([1.0 + 0j, d1, d0], dtype=complex)
    t_roots = np.roots(den)
    z_roots = zc + sz * t_roots
    zm, zp = _sort_poles(z_roots[0], z_roots[1])
    quotient, remainder = np.polydiv(num, den)
    remainder = np.trim_zeros(np.asarray(remainder, complex), trim="f")
    if remainder.size == 0:
        raise ValueError("Vanishing singular numerator")
    if remainder.size == 1:
        raise ValueError("Singular numerator is constant; response zero is at infinity")
    if remainder.size > 2:
        remainder = remainder[-2:]
    r1, r0 = remainder
    if abs(r1) < 1e-15:
        raise ValueError("Singular numerator is nearly constant; response zero is ill-conditioned")
    t0 = -r0 / r1
    z0 = zc + sz * t0

    # Residues in the physical z coordinate.  Since z-z_p=sz(t-t_p),
    # Res_z = sz * Res_t.
    def residue_at(zp_: complex) -> complex:
        tp = (zp_ - zc) / sz
        return sz * (r1 * tp + r0) / (2.0 * tp + d1)

    rm = residue_at(zm)
    rp = residue_at(zp)
    pred = np.polyval(num, t) / np.polyval(den, t)
    scale = max(float(np.sqrt(np.mean(np.abs(y) ** 2))), 1e-30)
    rel = float(np.sqrt(np.mean(np.abs(pred - y) ** 2)) / scale)
    cond = float(np.linalg.cond(A))
    full_roots_t, candidates_t, assignment_flags = _full_numerator_zero_diagnostics(
        num, den, t_roots,
        real_window_t=(float(np.min(t.real)), float(np.max(t.real))),
    )
    full_roots_z = tuple(complex(zc + sz * root) for root in full_roots_t)
    candidates_z = tuple(
        ZeroAssignmentCandidate(
            root=complex(zc + sz * candidate.root),
            inside_window=candidate.inside_window,
            pole_distance_scaled=candidate.pole_distance_scaled,
            cancelled=candidate.cancelled,
            slope=complex(candidate.slope / sz),
            slope_scaled=candidate.slope_scaled,
            root_condition_number=candidate.root_condition_number,
            well_conditioned=candidate.well_conditioned,
            eligible=candidate.eligible,
            reason=candidate.reason,
        )
        for candidate in candidates_t
    )
    flags = list(assignment_flags)
    if rel > 1.0e-4:
        flags.append("large_fit_residual")
    if not np.isfinite(cond) or cond > 1.0e12:
        flags.append("ill_conditioned_linear_fit")
    failure_flags = {"no_eligible_full_response_zero", "large_fit_residual", "ill_conditioned_linear_fit"}
    reliable = bool(
        not failure_flags.intersection(flags)
        and any(candidate.eligible for candidate in candidates_z)
    )
    return RationalTwoPoleFit(
        poles=(zm, zp),
        response_zero=complex(z0),
        residues=(complex(rm), complex(rp)),
        background=tuple(complex(v) for v in quotient),
        denominator=tuple(complex(v) for v in den),
        numerator=tuple(complex(v) for v in num),
        relative_rms=rel,
        condition_number=cond,
        frequency_center=zc,
        frequency_scale=sz,
        full_response_zeros=full_roots_z,
        zero_candidates=candidates_z,
        zero_assignment_reliable=reliable,
        zero_assignment_flags=tuple(flags),
    )


def circle_law_residual(
    z_minus: float,
    z_plus: float,
    residue_minus: float,
    residue_plus: float,
    minimum_splitting: float,
) -> float:
    """Hermitian two-pole circle-law residual.

    Returns A_R^2+(Delta_min/Delta)^2-1.  This is only a real, lossless
    corollary; the complex pole-zero reconstruction is the more general law.
    """
    c = residue_minus + residue_plus
    if c == 0 or z_plus == z_minus:
        raise ValueError("Degenerate data")
    asym = (residue_plus - residue_minus) / c
    ratio = minimum_splitting / (z_plus - z_minus)
    return float(asym * asym + ratio * ratio - 1.0)


@dataclass(frozen=True)
class GenealogyFit:
    model: str
    parameters: dict[str, float]
    rss: float
    bic: float
    success: bool


def fit_finite_intercept(fields: np.ndarray, q: np.ndarray, sigma: np.ndarray | float) -> GenealogyFit:
    b = np.asarray(fields, float)
    y = np.asarray(q, float)
    s = np.broadcast_to(np.asarray(sigma, float), y.shape)
    X = np.column_stack([np.ones_like(b), b, b * b])
    Aw = X / s[:, None]
    yw = y / s
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    pred = X @ beta
    rss = float(np.sum(((y - pred) / s) ** 2))
    n, k = len(y), 3
    bic = rss + k * np.log(n)
    return GenealogyFit("finite", {"q0": beta[0], "c1": beta[1], "c2": beta[2]}, rss, float(bic), True)


def fit_vanishing_velocity(fields: np.ndarray, q: np.ndarray, sigma: np.ndarray | float) -> GenealogyFit:
    """Fit q=C |B|^{-p}+q_offset by a fast profiled exponent grid.

    For each p, C and q_offset are obtained analytically by weighted linear
    regression.  A dense exponent grid avoids nonlinear-optimizer failures in
    Monte Carlo identifiability tests while remaining more accurate than the
    data precision used there.
    """
    b = np.asarray(fields, float)
    y = np.asarray(q, float)
    sig = np.broadcast_to(np.asarray(sigma, float), y.shape)
    if np.any(b <= 0) or np.any(y <= 0) or np.any(sig <= 0):
        return GenealogyFit("divergent", {}, float("inf"), float("inf"), False)
    pgrid = np.linspace(0.05, 3.0, 181)
    x = b[None, :] ** (-pgrid[:, None])
    w = 1.0 / (sig * sig)
    sw = float(np.sum(w))
    sy = float(np.sum(w * y))
    sx = np.sum(x * w[None, :], axis=1)
    sxx = np.sum(x * x * w[None, :], axis=1)
    sxy = np.sum(x * (w * y)[None, :], axis=1)
    det = sxx * sw - sx * sx
    valid = np.abs(det) > 1e-30
    C = np.full_like(pgrid, np.nan)
    qoff = np.full_like(pgrid, np.nan)
    C[valid] = (sxy[valid] * sw - sy * sx[valid]) / det[valid]
    qoff[valid] = (sxx[valid] * sy - sx[valid] * sxy[valid]) / det[valid]
    pred = C[:, None] * x + qoff[:, None]
    rss = np.sum(((pred - y[None, :]) / sig[None, :]) ** 2, axis=1)
    rss[(~valid) | (C <= 0)] = np.inf
    idx = int(np.argmin(rss))
    if not np.isfinite(rss[idx]):
        return GenealogyFit("divergent", {}, float("inf"), float("inf"), False)
    n, k = len(y), 3
    bic = float(rss[idx] + k * np.log(n))
    pars = {"C": float(C[idx]), "p": float(pgrid[idx]), "q_offset": float(qoff[idx])}
    return GenealogyFit("divergent", pars, float(rss[idx]), bic, True)


def classify_genealogy(fields: np.ndarray, q: np.ndarray, sigma: np.ndarray | float) -> tuple[str, GenealogyFit, GenealogyFit, float]:
    finite = fit_finite_intercept(fields, q, sigma)
    divergent = fit_vanishing_velocity(fields, q, sigma)
    delta_bic = divergent.bic - finite.bic
    label = "finite" if delta_bic > 0 else "divergent"
    return label, finite, divergent, float(delta_bic)

@dataclass(frozen=True)
class DarkSubspaceTomography:
    """Exact resolvent data for a rank-one probe and a finite mode matrix."""

    poles: tuple[complex, ...]
    response_zeros: tuple[complex, ...]
    bright_diagonal: complex
    total_coupling_moment: complex


def probe_adapted_unitary(probe: np.ndarray) -> np.ndarray:
    """Return a unitary whose first column is the normalized probe vector."""
    v = np.asarray(probe, dtype=complex).ravel()
    if v.size < 2 or not np.all(np.isfinite(v)):
        raise ValueError("probe must be a finite vector of length at least two")
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        raise ValueError("probe must be nonzero")
    q0 = v / norm
    dark = null_space(q0.conj()[None, :])
    U = np.column_stack([q0, dark])
    err = np.linalg.norm(U.conj().T @ U - np.eye(v.size))
    if err > 1e-10:
        raise RuntimeError(f"failed to construct probe-adapted unitary: {err:g}")
    return U


def scalar_cofactor_identity(
    inverse_response: np.ndarray,
    probe: np.ndarray,
) -> tuple[complex, complex]:
    """Evaluate both sides of the rank-one cofactor/Jacobi identity.

    For a scalar probe ``v`` and an arbitrary nonsingular complex inverse
    response ``K``, choose a unitary basis with first vector parallel to ``v``.
    Then

        v^† K^{-1} v = ||v||^2 det(K_DD)/det(K),

    where ``K_DD`` is the complementary dark principal block.  The identity is
    algebraic and does not require Hermiticity, normality, or a frequency-
    independent effective Hamiltonian.
    """
    K = np.asarray(inverse_response, dtype=complex)
    v = np.asarray(probe, dtype=complex).ravel()
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] != v.size:
        raise ValueError("K must be square and match the probe dimension")
    U = probe_adapted_unitary(v)
    Kp = U.conj().T @ K @ U
    lhs = complex(v.conj() @ np.linalg.solve(K, v))
    rhs = complex(np.vdot(v, v) * np.linalg.det(Kp[1:, 1:]) / np.linalg.det(Kp))
    return lhs, rhs


def multiprobe_jacobi_identity(
    inverse_response: np.ndarray,
    bright_indices: tuple[int, ...] | list[int],
) -> tuple[complex, complex]:
    """Evaluate Jacobi's complementary-minor identity for several probes.

    ``lhs`` is the determinant of the bright principal block of ``K^{-1}``;
    ``rhs`` is ``det(K_DD)/det(K)`` for the complementary dark block.
    """
    K = np.asarray(inverse_response, dtype=complex)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be square")
    n = K.shape[0]
    bright = tuple(sorted(set(int(i) for i in bright_indices)))
    if not bright or any(i < 0 or i >= n for i in bright):
        raise ValueError("invalid bright indices")
    dark = tuple(i for i in range(n) if i not in bright)
    inv_b = np.linalg.inv(K)[np.ix_(bright, bright)]
    lhs = complex(np.linalg.det(inv_b))
    rhs = complex((np.linalg.det(K[np.ix_(dark, dark)]) if dark else 1.0) / np.linalg.det(K))
    return lhs, rhs


def constant_matrix_tomography(
    effective_matrix: np.ndarray,
    probe: np.ndarray,
) -> DarkSubspaceTomography:
    """Poles, response zeros, and the first two bright spectral moments.

    For ``K(z)=z I-H`` the poles are the eigenvalues of ``H`` and the scalar
    response zeros are the eigenvalues of the dark compression of ``H`` in the
    probe-adapted basis.  The first two normalized large-frequency moments give
    the bright diagonal element and the total bright-dark coupling product
    ``H_BD H_DB`` (a scalar for a rank-one bright sector).
    """
    H = np.asarray(effective_matrix, dtype=complex)
    v = np.asarray(probe, dtype=complex).ravel()
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] != v.size:
        raise ValueError("H must be square and match the probe dimension")
    U = probe_adapted_unitary(v)
    Hp = U.conj().T @ H @ U
    poles = tuple(complex(z) for z in np.linalg.eigvals(H))
    zeros = tuple(complex(z) for z in np.linalg.eigvals(Hp[1:, 1:]))
    bright = complex(Hp[0, 0])
    coupling = complex(Hp[0, 1:] @ Hp[1:, 0])
    return DarkSubspaceTomography(poles, zeros, bright, coupling)


def response_from_effective_matrix(
    z: np.ndarray | complex,
    effective_matrix: np.ndarray,
    probe: np.ndarray,
) -> np.ndarray:
    """Scalar resolvent response ``v^†(zI-H)^{-1}v``."""
    H = np.asarray(effective_matrix, dtype=complex)
    v = np.asarray(probe, dtype=complex).ravel()
    zarr = np.asarray(z, dtype=complex)
    out = np.empty(zarr.shape, dtype=complex)
    eye = np.eye(H.shape[0], dtype=complex)
    for idx in np.ndindex(zarr.shape):
        out[idx] = v.conj() @ np.linalg.solve(zarr[idx] * eye - H, v)
    return out


def residues_of_constant_matrix(
    effective_matrix: np.ndarray,
    probe: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return simple poles and scalar-response residues for ``zI-H``.

    The implementation uses the exact cofactor numerator and derivative of the
    characteristic polynomial.  It assumes distinct poles.
    """
    H = np.asarray(effective_matrix, dtype=complex)
    v = np.asarray(probe, dtype=complex).ravel()
    U = probe_adapted_unitary(v)
    Hp = U.conj().T @ H @ U
    poles = np.asarray(np.linalg.eigvals(H), dtype=complex)
    if np.min(np.abs(poles[:, None] - poles[None, :] + np.eye(len(poles)))) < 1e-10:
        raise ValueError("poles must be distinct")
    den = np.poly(poles)
    dden = np.polyder(den)
    numerator = np.vdot(v, v) * np.poly(np.linalg.eigvals(Hp[1:, 1:]))
    residues = np.polyval(numerator, poles) / np.polyval(dden, poles)
    return poles, residues


def bright_moments_from_poles_residues(
    poles: np.ndarray,
    residues: np.ndarray,
) -> tuple[complex, complex, complex]:
    """Return prefactor, bright diagonal moment, and total coupling moment."""
    z = np.asarray(poles, dtype=complex).ravel()
    r = np.asarray(residues, dtype=complex).ravel()
    if z.shape != r.shape or z.size < 2:
        raise ValueError("poles and residues must have equal nontrivial shape")
    c = np.sum(r)
    if abs(c) <= 1e-14:
        raise ValueError("summed residue is too small")
    m1 = np.sum(r * z) / c
    m2 = np.sum(r * z * z) / c
    return complex(c), complex(m1), complex(m2 - m1 * m1)



def fit_exact_vanishing_velocity(
    fields: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray | float,
) -> GenealogyFit:
    """Fit the exact p=1 acoustic crossover q=q_inf*sqrt(1+(B_c/B)^2).

    This is the trajectory implied by q_h/q_*=sqrt(1+x^2)/x with x=B/B_c.
    For a fixed B_c the amplitude q_inf is obtained analytically by weighted
    least squares; a bounded scalar minimization profiles over log(B_c).
    """
    b = np.asarray(fields, float)
    y = np.asarray(q, float)
    sig = np.broadcast_to(np.asarray(sigma, float), y.shape)
    if np.any(b <= 0) or np.any(y <= 0) or np.any(sig <= 0):
        return GenealogyFit("vanishing_exact", {}, float("inf"), float("inf"), False)
    w = 1.0 / (sig * sig)

    def profile(log_bc: float) -> tuple[float, float]:
        bc = float(np.exp(log_bc))
        shape = np.sqrt(1.0 + (bc / b) ** 2)
        denom = float(np.sum(w * shape * shape))
        if denom <= 0.0:
            return float("inf"), float("nan")
        qinf = float(np.sum(w * shape * y) / denom)
        if not np.isfinite(qinf) or qinf <= 0.0:
            return float("inf"), qinf
        rss = float(np.sum(((y - qinf * shape) / sig) ** 2))
        return rss, qinf

    lower = np.log(float(np.min(b)) / 100.0)
    upper = np.log(float(np.max(b)) * 100.0)
    result = minimize_scalar(lambda x: profile(float(x))[0], bounds=(lower, upper), method="bounded",
                             options={"xatol": 1.0e-12, "maxiter": 600})
    rss, qinf = profile(float(result.x))
    if not result.success or not np.isfinite(rss):
        return GenealogyFit("vanishing_exact", {}, float("inf"), float("inf"), False)
    bc = float(np.exp(result.x))
    n, k = len(y), 2
    bic = float(rss + k * np.log(n))
    return GenealogyFit(
        "vanishing_exact",
        {"q_inf": qinf, "B_c": bc},
        rss,
        bic,
        True,
    )


def classify_genealogy_exact(
    fields: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray | float,
) -> tuple[str, GenealogyFit, GenealogyFit, float]:
    """Compare a finite-intercept quadratic with the exact anomaly crossover."""
    finite = fit_finite_intercept(fields, q, sigma)
    vanishing = fit_exact_vanishing_velocity(fields, q, sigma)
    delta_bic = vanishing.bic - finite.bic
    label = "finite" if delta_bic > 0 else "vanishing"
    return label, finite, vanishing, float(delta_bic)
