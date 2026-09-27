# Structured hidden-sector follow-up

Frozen before numerical execution. Existing studies and their code are read-only.
This is a passive oscillator realization of frequency-dependent self-energy,
not a microscopic Weyl collision calculation.

Use the prior q0, bright/remote stiffness and damping, and 181-point window.
The observed bright coordinate couples to one dark coordinate by kappa=.8 B;
the latter couples to two hidden bath coordinates. Their damping is .02.
The dark damping is .003 and bare frequency is 1.3 q0. The bath frequencies
are either (1.10,1.35) or (1.60,2.00); stiffness links are scale*(.12,.08),
with scale .5 or 1. Include one uncoupled-bath control, giving five models.
Use B=.05,.10,.20. The reference retains the exact bright/remote block but
sets kappa=0. No reference drift is introduced in this separate experiment.

1. Verify the direct five-coordinate inverse against the hidden-sector Schur
   relation, positive stiffness/loss, and lower-half-plane poles. Extract the
   compressed hidden poles and their bright-response zeros. Bath modes with
   zero coupling cancel and must not be called observable zeros.
2. Rotate the two bath coordinates to turn a star into a chain while fixing
   the bright and original dark coordinates. With equal bath damping the
   resulting passive chain has exactly the same scalar response and reference.
   Verify at real and complex frequencies. This is a concrete realization of
   an established internal-coordinate equivalence, not a new no-go theorem.
3. Apply the existing single-dark-mode compatibility test to all five models.
   Error radii are 0,1e-4,1e-3 times the RMS of the same bright reference.
   Use one exact case and five fixed-seed independent bounded-disk replicas
   at each nonzero radius: 165 cases. Retain compatible violations and failed
   separation. The reference and signal are each measured once with independent
   errors. No claim about equal acquisition time versus other estimators.
4. Report frequency-dependent apparent damping and the distinction between
   bare dark frequency and compressed-sector poles. Do not infer bath topology
   or globally recover its spectral density from this finite-window test.

Run with one BLAS thread. Save inputs, all trials, exact LP witnesses, hashes,
analytic derivation, and one figure. A self-check covers Schur and star/chain
equivalence, passivity, and the zero-bath limit before the production sweep.

Pre-production numerical-check correction: a weak far-bath root gives a small
but nonzero bright response (~7e-8) when the double-precision eigenvalue is
substituted into an ill-conditioned full inverse. Retain that residual and
validate compressed roots by normalized singular-value backward error, together
with a nonzero full-kernel singular value excluding a common factor. The
analytic cofactor identity supplies the exact statement; do not silently round
these computed bright-response values to zero. No production outcome informed
this correction.

## Follow-up declared after run 01

Run 01 retained 32 far-bath cases that passed the isolated-mode test while its
bare-frequency interval excluded the actual bare coordinate frequency. The
next calculation addresses this observed model-error limitation, not an
independent discovery-rate test. For the far-bath class assume only a bath gap
Omega_b>=1.6 and sum(h_j^2)<=.12^2+.08^2, independent of field and bath damping.
This gives |Sigma(omega)| <= H(omega)=.0208/(1.6^2-omega^2) throughout the
window. Add H to each real/imaginary kernel inequality, retaining the same
data, LP nonnegative parameter domain and numerical status conventions.
Apply this bound only to the 66 cases whose generator satisfies the declared
gap/norm prior. Report all interval widths and bare-value coverage; do not
present the extra physical prior as information obtained from the scalar data.
Run 01 and its source snapshot remain retained. Run 02 repeats the original
165 cases unchanged and adds this model-remainder calculation.
