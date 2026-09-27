# Material `Gamma^omega` projection input contract

The numerical Landau interaction is **not** obtained by reading a static cRPA
matrix as `F_d` and `F_x`. The required input is one NPZ file in a common
orbital/Wannier gauge.

Required arrays:

- `states[P,A]` complex: normalized interacting band spinors on Fermi-surface patches.
- `area[P]` float: patch areas in reciprocal-volume units.
- `node[P]` integer: valley/node labels.
- `gamma[P,P,A,A,A,A]` complex: antisymmetrized 1PI dynamic forward vertex
  `Gamma^omega`, with external transfer momentum set to zero before frequency
  is taken to zero.

Quasiparticle renormalization can be supplied in either form:

- `velocity[P]` and `residue[P]`; or
- `v0[P,3]`, `grad_sigma[P,3]`, and `dsigma_domega[P]`, from which the code
  computes `Z=[1-dSigma/domega]^{-1}` and
  `v*=Z(v0+grad_k Sigma)`.

Run:

```bash
python project_material_gamma_omega.py material_gamma_omega.npz \
  --output-dir projected_modes \
  --axial-signs '{"-1":-1,"1":1}'
```

Outputs include the full Landau spectrum, charge/axial overlaps, the projected
2x2 matrix, and the leakage of the charge-axial subspace into all other Fermi-
surface harmonics. A small leakage is a numerical result, not an assumption.

The upstream many-body calculation must be conserving: the self-energy and
Bethe-Salpeter kernel should satisfy `I=delta Sigma/delta G`, and the external
scalar/current vertices and contact term must be differentiated from the same
gauge-covariant functional.
