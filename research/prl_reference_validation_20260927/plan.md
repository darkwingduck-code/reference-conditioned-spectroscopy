# Frozen design: reference compatibility and matched spectral baselines

Frozen before numerical execution on 27 September 2026. Existing scientific
sources and archived results are read-only. No outcome is a stopping criterion.

## Questions

1. On exactly the same noisy calibrated spectra, how do positive-frequency
   principal-part and full-numerator zeros compare with fitted, real-coefficient
   rational functions of `s^2`, which retain the negative-frequency partners?
2. Can independently bounded signal/reference errors reject a reference model
   that an unconditional parameter-box calculation would accept?
3. Does a real quadratic reference transform recover an independently damped
   dark oscillator without supplying its damping as an estimator input?

These questions concern finite passive oscillator benchmarks. They do not
establish a material prediction, a general analytic-continuation certificate,
or a reversal of the finite-intercept class.

## Fixed inputs and cases

- Frequency window `[0.79, 1.15]`, 181 training points; 180 interlaced clean
  points used only after fitting. Root choice uses the fixed center `0.97`.
- Remote stiffness frequency `2.6`, bright-remote link `0.8`, magnetic
  stiffness link `kappa=0.8 B`, `B in {0.05, 0.10, 0.20}`.
- Noise/calibration radii are jointly `{0, 1e-4, 1e-3}`, using the existing
  measured-RMS convention and bounded complex uniform disks. One exact case
  and five replicas at each nonzero radius. Reference and detector errors
  are shared within each field sweep; signal errors are independent.
- Panel A: the existing shifted-frequency generator, `s=omega+i eta`,
  `q in {q0-0.03,q0,q0+0.03}`, `q0=0.7458353261772512`,
  `eta in {0.003,0.010}`. 198 cases.
- Panel B: physical viscous generator
  `L(omega)=omega^2 I-K+2 i omega Gamma`, `q=q0`,
  `Gamma_b=0.003`, `Gamma_r=0.007`,
  `Gamma_d in {0.0015,0.003,0.006}`. The same six predeclared controls at
  every field/damping/noise setting give 594 cases:
  1. `valid_star`;
  2. `true_dark_shift`: `d -> d+0.08 B^2` (reference remains valid);
  3. `dark_remote`: dark-remote stiffness `0.2 B`;
  4. `mixed_probe`: source/readout `(1,0,0.3)` for signal and reference;
  5. `bright_drift`: signal bright stiffness increased by `0.2 B^2`;
  6. `reference_damping_mismatch`: signal bright damping increased by
     `0.05 B^2`, reference bright damping unchanged.

Total: 792 retained cases. No case is removed for a failed fit, missing
interval, an incorrect zero, or a model violation hidden by measurement error.

## Estimators and information fairness

- Reuse the existing complex `[3/2]` fit in `omega`; report its pp and full
  numerator roots separately (12 real coefficients after monic normalization).
- Fit real `[3/2]` polynomials in centered `s^2`, including an affine remainder
  in `s^2` (six real coefficients). Report both its paired pp and full roots.
  It is given the same known damping as the original reference estimator in
  Panel A. In Panel B this is explicitly a fixed-common-damping comparator;
  its equal-damping assumption is not valid for every physical generator.
- Retain the existing reference estimate and its conditional box.
- For the new reference calculation, form directly from calibrated observations
  `T=X Y/(X-Y)` and a pointwise deterministic error radius. Use the closed
  nonnegative physical parameter domain, not arbitrary strict-positivity cuts.
- Panel B uses `T=(omega^2-r+2 i gamma omega)/t`, where `r=Omega_d^2`,
  `t=kappa^2`. Gamma_dark is not supplied. Panel A compares both known-damping
  and free-damping reference variants, converting back to the same complex
  cofactor root before scoring. Undamped frequency and damped pole position
  are distinct observables.

## LP status and numerical limits

Pointwise real/imaginary rectangular error constraints are necessary conditions
for a complex disk bound, hence a feasible LP does not prove the model. Report
`compatible`, `inconsistent`, or `inconclusive`, never a structural certificate.
Scale LP rows, record the declared arithmetic relaxation, verify returned
primal residuals, and independently inspect a phase-I dual separating witness
before reporting numerical inconsistency. Borderline or solver-failure cases
are inconclusive. Infeasibility must survive a larger arithmetic relaxation.
No claim of outward-rounded interval arithmetic is made.

## Checks and retained products

Before production: exact two-pair fit, direct 3x3 inverse versus Schur identity,
independent dark damping, passivity/stability, bounded-error propagation,
valid/noisy LP inclusion, a plainly incompatible quadratic, and tolerance
boundary controls. Run one process with BLAS/OpenMP thread counts set to one.
Save source/design hashes, all cases, grouped summaries, independent checks,
one scientific PDF/PNG figure, and English derivation/results. Existing core
tests are regression checks; fresh tests do not substitute for scientific review.
