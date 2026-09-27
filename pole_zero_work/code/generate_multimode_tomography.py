#!/usr/bin/env python3
"""Generate multimode cofactor-tomography and identifiability figures."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pole_zero_tomography import (
    constant_matrix_tomography,
    multiprobe_jacobi_identity,
    probe_adapted_unitary,
    response_from_effective_matrix,
    scalar_cofactor_identity,
)

FIG = ROOT / "fig_prl"
DATA = ROOT / "data_prl"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.2,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
})


def save(fig: plt.Figure, stem: str, title: str) -> None:
    metadata = {
        "Title": title,
        "Author": "Authors to be inserted",
        "Subject": "Multimode pole-zero tomography and identifiability",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def stable_multimode_matrix(seed: int = 20260626, n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bare = np.linspace(0.16, 0.39, n)
    A = np.diag(bare)
    coupling = rng.normal(scale=0.018, size=(n, n))
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    damping = np.linspace(0.006, 0.026, n)
    # Complex symmetric but non-normal once unequal damping is added.
    H = A + coupling - 1j * np.diag(damping)
    probe = rng.normal(size=n) + 0.35j * rng.normal(size=n)
    return H.astype(complex), probe.astype(complex)


def total_response_zeros(
    poles: tuple[complex, complex],
    singular_zero: complex,
    prefactor: complex,
    background: tuple[float, float],
) -> np.ndarray:
    """Zeros of (a1 z+a0)+C(z-z0)/[(z-z-)(z-z+)]."""
    zm, zp = poles
    a1, a0 = background
    den = np.poly1d([1.0, -(zm + zp), zm * zp])
    bg = np.poly1d([a1, a0])
    sing_num = np.poly1d([prefactor, -prefactor * singular_zero])
    num = np.polymul(bg, den) + sing_num
    return np.roots(num)


def main() -> None:
    rng = np.random.default_rng(20260626)
    H, probe = stable_multimode_matrix()
    tomo = constant_matrix_tomography(H, probe)
    U = probe_adapted_unitary(probe)
    Hp = U.conj().T @ H @ U

    pole_rows = []
    for z in tomo.poles:
        pole_rows.append({"kind": "hybrid_pole", "real": z.real, "imag": z.imag})
    for z in tomo.response_zeros:
        pole_rows.append({"kind": "response_zero", "real": z.real, "imag": z.imag})
    for z in np.linalg.eigvals(Hp[1:, 1:]):
        pole_rows.append({"kind": "dark_block_eigenvalue", "real": z.real, "imag": z.imag})
    pd.DataFrame(pole_rows).to_csv(DATA / "figS7_multimode_poles_zeros.csv", index=False)

    # Random analytic inverse responses: scalar and multiprobe Jacobi residuals.
    residual_rows = []
    for n in range(2, 11):
        scalar_max = 0.0
        multi_max = 0.0
        for rep in range(160):
            A0 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            A1 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            A2 = 0.15 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
            z = rng.uniform(-0.8, 0.8) + 1j * rng.uniform(0.2, 1.0)
            K = A0 + z * A1 + z * z * A2 + (2.5 + 0.3j) * np.eye(n)
            v = rng.normal(size=n) + 1j * rng.normal(size=n)
            lhs, rhs = scalar_cofactor_identity(K, v)
            scale = max(abs(lhs), abs(rhs), 1e-30)
            scalar_max = max(scalar_max, abs(lhs - rhs) / scale)
            if n >= 3:
                bright = tuple(range(min(2, n - 1)))
                ml, mr = multiprobe_jacobi_identity(K, bright)
                mscale = max(abs(ml), abs(mr), 1e-30)
                multi_max = max(multi_max, abs(ml - mr) / mscale)
        residual_rows.append({
            "matrix_size": n,
            "max_scalar_cofactor_relative_residual": scalar_max,
            "max_two_probe_jacobi_relative_residual": multi_max if n >= 3 else np.nan,
        })
    rdf = pd.DataFrame(residual_rows)
    rdf.to_csv(DATA / "figS7_cofactor_jacobi_residuals.csv", index=False)

    # Loss-only non-identifiability: two real analytic backgrounds produce the
    # same loss on the real axis but different full complex response and zeros.
    zm = 0.215 - 0.012j
    zp = 0.305 - 0.024j
    z0 = 0.267 - 0.019j
    pref = 0.75 + 0.0j
    omega = np.linspace(0.15, 0.37, 700)
    den = (omega - zm) * (omega - zp)
    chi_s = pref * (omega - z0) / den
    bg1 = (0.0, 5.0)  # a1 z + a0; real on the real axis
    bg2 = (0.0, -5.0)
    chi1 = chi_s + bg1[0] * omega + bg1[1]
    chi2 = chi_s + bg2[0] * omega + bg2[1]
    zeros1 = total_response_zeros((zm, zp), z0, pref, bg1)
    zeros2 = total_response_zeros((zm, zp), z0, pref, bg2)
    ambiguity = pd.DataFrame({
        "omega": omega,
        "minus_im_chi_background_1": -np.imag(chi1),
        "minus_im_chi_background_2": -np.imag(chi2),
        "re_chi_background_1": np.real(chi1),
        "re_chi_background_2": np.real(chi2),
    })
    ambiguity.to_csv(DATA / "figS7_loss_only_ambiguity.csv", index=False)
    zrows = [{"kind": "singular_zero", "background": "subtracted", "real": z0.real, "imag": z0.imag}]
    for label, roots in (("background_1", zeros1), ("background_2", zeros2)):
        for root in roots:
            zrows.append({"kind": "total_response_zero", "background": label, "real": root.real, "imag": root.imag})
    pd.DataFrame(zrows).to_csv(DATA / "figS7_background_dependent_total_zeros.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15))
    ax = axes[0, 0]
    poles = np.asarray(tomo.poles)
    zeros = np.asarray(tomo.response_zeros)
    dark = np.asarray(np.linalg.eigvals(Hp[1:, 1:]))
    ax.scatter(poles.real, -poles.imag, marker="o", s=34, label="hybrid poles")
    ax.scatter(zeros.real, -zeros.imag, marker="x", s=42, label="response zeros")
    ax.scatter(dark.real, -dark.imag, marker="+", s=66, label="dark-block roots")
    ax.set_xlabel(r"$\mathrm{Re}\,z$")
    ax.set_ylabel(r"$-\mathrm{Im}\,z$")
    ax.set_title("six-mode non-Hermitian resolvent")
    ax.legend(frameon=False)
    ax.text(0.03, 0.05, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[0, 1]
    ax.semilogy(rdf.matrix_size, rdf.max_scalar_cofactor_relative_residual, marker="o", label="rank-one probe")
    ax.semilogy(rdf.matrix_size, rdf.max_two_probe_jacobi_relative_residual, marker="s", label="two-probe determinant")
    ax.set_xlabel("matrix dimension")
    ax.set_ylabel("maximum relative residual")
    ax.set_title("cofactor/Jacobi identity")
    ax.legend(frameon=False)
    ax.text(0.03, 0.05, "(b)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 0]
    ax.plot(omega, -np.imag(chi1), label="loss, background 1")
    ax.plot(omega, -np.imag(chi2), ls="--", label="loss, background 2")
    ax.set_xlabel(r"real frequency $\omega$")
    ax.set_ylabel(r"$-\mathrm{Im}\,\chi$")
    ax.set_title("loss-only data are identical")
    ax.legend(frameon=False)
    ax.text(0.03, 0.05, "(c)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 1]
    ax.plot(omega, np.real(chi1), label=r"$\mathrm{Re}\,\chi$, background 1")
    ax.plot(omega, np.real(chi2), ls="--", label=r"$\mathrm{Re}\,\chi$, background 2")
    ax.set_xlabel(r"real frequency $\omega$")
    ax.set_ylabel(r"$\mathrm{Re}\,\chi$")
    ax.set_title("phase/background information differs")
    ax.legend(frameon=False)
    ax.text(0.03, 0.05, "(d)", transform=ax.transAxes, fontweight="bold")

    fig.subplots_adjust(wspace=0.31, hspace=0.36)
    save(fig, "figS7_multimode_and_identifiability", "Multimode cofactor tomography and loss-only non-identifiability")

    cost = np.abs(zeros[:, None] - dark[None, :])
    rr, cc = linear_sum_assignment(cost)
    max_dark_zero_match = float(np.max(cost[rr, cc]))
    summary = {
        "multimode_dimension": int(H.shape[0]),
        "max_dark_zero_match": max_dark_zero_match,
        "max_scalar_cofactor_relative_residual": float(rdf.max_scalar_cofactor_relative_residual.max()),
        "max_multiprobe_jacobi_relative_residual": float(rdf.max_two_probe_jacobi_relative_residual.max()),
        "maximum_loss_difference_between_real_backgrounds": float(np.max(np.abs(np.imag(chi1 - chi2)))),
        "real_response_difference_rms": float(np.sqrt(np.mean((np.real(chi1 - chi2)) ** 2))),
        "singular_response_zero": [float(z0.real), float(z0.imag)],
        "total_response_zeros_background_1": [[float(z.real), float(z.imag)] for z in zeros1],
        "total_response_zeros_background_2": [[float(z.real), float(z.imag)] for z in zeros2],
    }
    (DATA / "multimode_tomography_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
