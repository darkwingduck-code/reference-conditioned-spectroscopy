# Frozen extension: spend the same training budget near the remote resonance

Declared after completing and preserving `run_01`, before the wide-design run.
The original plan, driver, run, and source snapshot remain unchanged.

Question: does measuring the remote dynamics relieve the frequency/coupling
ambiguity of the extra-link model, under a fixed count of complex measurements
and fixed absolute noise? This is a new acquisition design comparison, not an
additional source channel, and not a proof of material realism.

- Use only the 16 `extra_link` generator cases from the original frozen product.
- Compare `drift_aware` (nine unknowns, omits the extra link) and `extended`
  (ten unknowns, includes the extra link), each jointly fitting the reference.
- Training frequencies: 61 evenly spaced points in `[0.79,1.15]` and 20 evenly
  spaced points in `[2.40,2.80]`. Total remains 81 per spectrum / 162 complex.
- Holdout frequencies are the within-segment midpoints: 60 + 19 = 79 per
  spectrum / 158 complex. Holdout is independent and never used for fitting or
  start selection; its normalized chi-square uses 316 real components.
- Keep the absolute real/imaginary Gaussian standard deviations from `run_01`:
  `noise_scale * 26.965025258932673`. Never renormalize to a new reference RMS.
  This assumes a fixed frequency-independent measurement noise and equal cost
  per complex frequency measurement; a real frequency-dependent instrument
  noise or acquisition-time model is outside the comparison.
- Use the original case seed, so the 81-point training noise vectors are shared
  between designs (common random numbers). Independent holdout noise is drawn
  after the training noise. Every model within a design receives identical data.
- Keep all original bounds, known `h=.8`, three starts, tolerances, logarithmic
  coordinates and 160 evaluations per start. No warm start or adaptive tuning.
  No profiles or profile-induced restarts are added in this bounded extension.
- Retain all 96 start attempts, 32 selected fits, failed convergence, boundaries,
  raw/column-normalized Jacobian conditions, all three dark errors, and held-out
  residuals. A failed fit is not evidence of structural nonidentifiability, and
  a small residual is not evidence of correct bare-parameter recovery.
- Preserve all spectra, hashes and the new driver/plan snapshot in
  `run_02_wide`. Compare with the corresponding 32 fits from `run_01`, explicitly
  noting that predeclared low-window profile cases permitted an extra restart.

No upstream scientific function is edited. The new driver imports and reuses
the existing generator, likelihood, bounds and fit implementation.
