# Structured hidden-sector result

Final science: `run_02/`. Final four-panel figure: `figures_v2/`. Run 01 and
its source snapshot are retained; every original trial field agrees exactly
in run 02. The follow-up was declared after run 01 exposed a model-error
failure. It adds a separately supplied physical bound, not new measurements.

## What the reference measures

For a bright coordinate coupled by kappa to one dark coordinate, which in turn
couples to damped hidden bath coordinates,

```text
Q(w) = w^2 - Omega_d^2 + 2 i gamma_d w - Sigma(w)
Sigma(w) = sum_j h_j^2 / (w^2 - Omega_j^2 + 2 i gamma_j w)
chi^(-1) = chi0^(-1) - kappa^2/Q
chi*chi0/(chi-chi0) = Q/kappa^2
```

The construction is a finite passive realization of frequency-dependent
self-energy. It is not a microscopic Weyl collision theory, a new principle
of bath spectroscopy, or recovery of an arbitrary bath spectral density.
The compressed roots differ from the bare dark-coordinate natural frequency.
A bath-only orthogonal rotation converts the two-link star into a chain with
identical scalar signal and reference. It demonstrates internal-coordinate
equivalence without claiming that an independently calibrated microscopic
basis is arbitrary.

## Results and a conditional repair

All 15 model/field combinations pass positive-stiffness/loss, stable-pole,
direct-versus-Schur and star/chain checks. Root backward errors and nonzero
full-kernel singular values are recorded. Weak bath roots make direct inversion
at a rounded eigenvalue ill-conditioned; the nonzero residuals are preserved.

The reused isolated-mode LP sees 165 cases: 33 isolated controls, 66 near-bath
cases and 66 far-bath cases. All controls are compatible. All near-bath cases
and 34 far-bath cases are inconsistent, with 100 saved exact rational witnesses
for the stored floating LPs. The remaining 32 far-bath cases are compatible,
yet **all 32 natural-frequency intervals exclude the bare frequency**.

If independent information bounds all bath frequencies below by 1.6 and the
squared link norm above by .0208, then on the real measurement window

```text
|Sigma(w)| <= H(w) = .0208 / (1.6^2 - w^2)
|r + Re(T_hat)*t - w^2| <= E*t + H
|2*g*w - Im(T_hat)*t| <= E*t + H
```

Here r=Omega_d^2, t=kappa^2, g=gamma_d, and E is the measured-transform bound.
These remain linear necessary constraints. With that extra prior, all 66
applicable far-bath cases have usable intervals containing the bare frequency.
For the first weak far-bath replica at B=.1 and epsilon=.001, the interval
widens from [.9681649855,.9681746993] to [.9615192889,.9747743775], covering
the true .9695859240. There is no analogous guarantee for the near-bath models,
which violate the supplied gap. This is not a statistical coverage theorem or
an outward-rounded interval certificate, and compatibility does not prove a
physical bath realization.

An independent reviewer verified the signs and bound, recomputed Schur error
5.34e-14, and checked all 11,753 safe samples in the 66 far-bath cases:
the truth satisfies every inequality and max |Sigma|/H is .86865. Shared
run01/run02 CSV fields agree exactly. All endpoints and outcomes are retained.

## Reproduction

Use the existing Python/NumPy/SciPy/Matplotlib environment, from repository root:

```powershell
python -B research/structured_dark_sector_20260927/run_study.py --self-check
python -B research/structured_dark_sector_20260927/run_study.py --output reproduction/structured
python -B research/structured_dark_sector_20260927/render_extended.py --run reproduction/structured --output reproduction/structured_figures
```

New output directories are required. `plan.md` records the fixed design and
the later model-remainder question. Source snapshots, all 165 trial rows,
representative spectra and separation witnesses are retained. The final figure
shows fixed representative intervals at B=.1, epsilon=.001, replica zero,
for both far-bath strengths; it does not select best-performing replicas.
