# Frozen plan: jointly fitting a drifting reference block

Frozen before numerical runs, 27 September 2026. Ownership is limited to this
new directory. The existing oscillator/validation modules are read-only.

## Question and model

Can dark natural frequency `Omega_d`, viscous damping `gamma_d`, and squared
stiffness coupling `t=kappa^2` still be inferred when the signal bright block
differs from the reference by an unknown stiffness drift `u` and damping drift
`v`? This is a finite passive oscillator inverse problem, not a Weyl-material
validation. No arbitrary dark bath or topology-identifiability extension is
included in this study.

The generator is `L(omega)=omega^2 I-K+2i omega Gamma`. It uses the previous
three-coordinate star model at `q0=0.7458353261772512`, known structural link
`h=0.8`, remote frequency `2.6`, and baseline bright/remote damping
`0.003,0.007`. The bright reference and its noise are fitted jointly rather
than inserted as true parameters. In every estimator the shared baseline bright
stiffness/damping and remote stiffness/damping are four unknowns. Only `h=0.8`
and the oscillator functional form are supplied as structural information.

Three fits use the identical signal/reference observations:

1. `unchanged`: seven unknowns (four shared reference quantities plus the three
   dark quantities), imposing `u=v=0`.
2. `drift_aware`: nine unknowns, additionally fitting signal bright stiffness
   drift and an independent positive signal bright damping (`v` is their
   difference).
3. `extended`: the same nine unknowns plus a real dark--remote stiffness link.
   This is the more flexible same-information baseline, not an extra probe.

The zero-field reference has no bright--dark or dark--remote link. Reference
parameters are shared with the signal except for the two explicitly allowed
bright drifts. All models see the same known absolute measurement noise.

## Fixed design, noise, and held-out measurements

- Window `[0.79,1.15]`; 81 complex training points in each spectrum.
- The 80 interlaced frequencies in each spectrum are independently noisy held-out
  measurements and are never used in fitting or choosing starts. Clean errors
  are also scored afterward and explicitly labeled oracle diagnostics.
- Fields `{0.05,0.20}`; dark damping `{0.003,0.009}`; Gaussian noise scales
  `{1e-4,1e-3}`; two fixed replicas; controls `{zero_drift,allowed_drift,extra_link}`.
  Total: 48 cases and three fitted models per case.
- Noise is independent Gaussian in real and imaginary components. Its standard
  deviation equals the noise scale times the fixed clean baseline-reference RMS
  over the frozen design. This one absolute scale is shared by both spectra and
  all controls. All compared methods therefore use the same 162 complex training
  measurements with the same total acquisition noise; no method receives extra
  reference information or a better noise budget.
- `zero_drift`: `u=v=0`; `allowed_drift`: `u=0.2B^2`, `v=0.05B^2`;
  `extra_link`: these same drifts plus dark--remote stiffness `ell=0.2B`.
- No failed fit, bound-hitting estimate, or inaccurate result is deleted.

## Bounds, starts, and optimization

All fits use explicit, common physical search bounds: baseline bright stiffness
`[0.6,1.6]`, remote stiffness `[4,12]`, baseline bright damping `[1e-4,0.03]`,
remote damping `[1e-4,0.06]`, dark frequency `[0.85,1.10]`, dark damping
`[1e-4,0.03]`, squared coupling `[1e-5,0.08]`, bright stiffness drift
`[-0.05,0.05]`, signal bright damping `[1e-4,0.04]`, and extra link `[-0.10,0.10]`.
These are stated priors, not learned physical guarantees. Damping and squared
coupling are optimized in logarithmic coordinates.

Three fixed starts use (bright stiffness, bright damping, remote stiffness,
remote damping, dark frequency, dark damping, squared coupling, stiffness drift,
signal damping, extra link):

- `(1.05,.006,7.5,.015,.970,.006,.010,0,.006,0)`;
- `(.90,.002,5.0,.002,.920,.002,.004,.010,.003,.030)`;
- `(1.20,.012,10.0,.040,1.030,.015,.040,-.010,.015,-.030)`.

Use bounded SciPy least squares, finite-difference Jacobians, `x_scale='jac'`,
`ftol=xtol=gtol=1e-9`, and at most 160 function evaluations per start. Retain every
start's termination status, objective, evaluation count, and boundary flags.
The lowest finite objective is reported even when convergence failed, with the
failure flag visible. A small residual is not itself identification evidence.

## Profile support and stop conditions

Predeclared profile cases: dark damping `.003`, replica zero, all three controls
and both fields at noise `1e-3`, plus the allowed-drift strong-field case at
noise `1e-4` (seven cases). Profile `Omega_d` and `gamma_d` for all three fits.
At each profile point, fix only that parameter and reoptimize all others from
the best unrestricted solution and one alternate frozen start, with at most
100 evaluations per start.

Frequency offsets around the fitted value are
`0,+/-{1e-5,1e-4,1e-3,.005,.02,.08}`, clipped to the stated physical bounds,
with both global endpoints added. Damping uses multiplicative log offsets
`0,+/-{.01,.05,.20,.70,2,4}`, similarly clipped with endpoints added.
These multiscale grids are fixed rules, not result-dependent success tuning.

The conventional one-parameter likelihood-ratio threshold `Delta chi^2=3.841`
is only a nominal support convention for this nonlinear bounded problem, not
a coverage guarantee under misspecification. Report accepted sampled points,
brackets of threshold crossings, disconnected support, endpoints, and failed
profile optimization. Never call a single accepted grid point a zero-width
confidence interval. If a profile finds a lower objective than all unrestricted
starts by more than `1e-3`, allow one unrestricted restart from that profile
point and record the improvement.

Record raw and column-normalized Jacobian singular values/condition numbers,
estimated parameter errors, noisy held-out chi-square, clean prediction error,
profile results, and all numerical failures. A two-sigma Gaussian chi-square
reference is descriptive only; it is not an automatic model acceptance rule.
One process, BLAS/OpenMP one thread. Numerical work is bounded to about ten
minutes; if reached, retain completed cases and clearly label unfinished profiles.
No adaptive increase in data quality, starts, bounds, or acquisition budget.

## Deliverables

Run source hashes and frozen design; all fit/start/profile tables; meaningful
core tests; one scientific figure; English results/limitations and a standalone
manuscript fragment. Existing code, manuscript sources, and prior run outputs
remain unchanged.
