# Raw-complex joint fitting with reference-block drift

This bounded follow-up estimates the noisy reference parameters jointly with
the dark parameters. It compares unchanged-reference, bright-drift, and
bright-drift-plus-extra-link models on exactly the same observations. The known
structural link is `h=0.8`; no estimator receives the true reference parameters.
This is a finite oscillator study, not a microscopic Weyl-material validation.

From the repository root, with the existing Python environment:

```powershell
python -B -m unittest discover -s research/reference_control_study_20260927 -p test_study.py -v
python -B research/reference_control_study_20260927/study.py --output run_02
python -B research/reference_control_study_20260927/analyze.py --run run_02
```

The saved frozen production run is `run_01`. Use a new direct-child output name
to reproduce it; the runner refuses to overwrite a run. One process and one
BLAS/OpenMP thread are set before importing NumPy. No dependencies are installed.
The numerical budget is 570 seconds, with incomplete work explicitly retained.
Analysis and figure regeneration do not refit any data:

```powershell
python -B research/reference_control_study_20260927/analyze.py --run run_01
```

`plan.md` fixes cases, noise, priors, starts, grids, and stopping conditions before
the run. `run_01/validation.json` records the source hashes and unchanged-source
check; `run_01/source_snapshot/` preserves the generating code and frozen plan.
`observations.npz` retains all complex training and independent holdout samples.
`fits.csv`, `fit_starts.csv`, `profiles.csv`, and `profile_starts.csv` retain
failures and boundary fits. `analysis/` contains grouped summaries, sampled
profile support, the figure, and hashes linking the analysis to its inputs.

The three core tests check the closed scalar expression against dense matrix
inversion including both drifts and the extra link, exact joint recovery with
unknown reference parameters, and correct fixed-target profiling with both
spectra included in the likelihood. Profile support uses a nominal
`Delta chi^2 = 3.841` convention. It is not a confidence-coverage certificate;
failed nuisance optimization and coarse crossing brackets remain visible.

`results.md` gives the derivation, complete outcomes and limitations.
`manuscript_fragment.tex` is a standalone integration fragment with unique labels.

The additional remote-inclusive design is frozen in `wide_plan.md`, with results
in `wide_results.md`. The final reproducible run is `run_03_wide`. It repeats the
preserved `run_02_wide` after adding only an output-directory command-line option.
Both scientific CSV tables are byte-identical and all observation arrays are
identical. The respective generating sources are preserved in each run's
`source_snapshot/`. Reproduce to a new output name, then regenerate the final
archived analysis or verify both originals:

```powershell
python -B research/reference_control_study_20260927/wide_study.py --output run_04_wide
python -B research/reference_control_study_20260927/analyze_wide.py --run run_03_wide
python -B research/reference_control_study_20260927/verify_outputs.py
python -m ruff check --no-cache --select F,E9 research/reference_control_study_20260927/*.py
```

Both runs use the identical absolute noise scale and 162 complex training
measurements; the extended design moves 20 points per spectrum near the remote
resonance. No scientific source from the original run was edited.
