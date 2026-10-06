# Reference validation, 27 September 2026

Read `results.md` for the derivation and outcomes, `plan.md` for the pre-run
design, and `figures/reference_validation_20260927.pdf` for the single scientific
figure. Final data: `run_02/trials.csv`, `run_02/summary.csv`, and
`run_02/validation.json`. The first run is preserved and the scientific columns
agree exactly; the explicit comparison is in `figures/verification.json`.

The new result is a conditional model-compatibility test and independent viscous
dark-damping extension. A paired full-numerator baseline already corrects the
large positive-pole bias. Additional reference measurements are not treated as
free information. Compatible does not mean that the microscopic model is true.

From the repository root, using the existing Python/NumPy/SciPy environment:

```powershell
python -B -m unittest discover -s research/reference_validation_20260927 -p test_experiment.py -v
python -B -m unittest discover -s research/physical_discrimination_20260912 -p test_discrimination.py -v
python -B research/reference_validation_20260927/experiment.py --output reproduction_01
python -B research/reference_validation_20260927/render_results.py
```

For a layout-only figure regeneration, use `render_results.py --figure-only`;
this preserves the scientific run and does not repeat its tests.

The experiment sets BLAS/OpenMP thread counts to one before importing NumPy and
runs one process. Its output must be a new direct-child directory; it refuses
to overwrite old runs. The renderer deliberately reads the retained final
`run_02`, verifies its comparison with `run_01`, runs both small test files, and
regenerates only this folder's figure/report. It does not silently substitute a
new reproduction for the frozen final data.

No dependencies were added. The only imported scientific modules outside this
folder are the existing `oscillator_pilot.py` and `discrimination_core.py` in
`research/physical_discrimination_20260912/`. They remain unchanged.

`manuscript_extension.tex` is a standalone section for the existing supplement;
its labels use the unique `ref20260927` prefix. The leader owns supplement
integration, figure copying, bibliography integration, full paper builds, and
repository delivery. No journal submission or publication is performed here.
