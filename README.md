# Reference-conditioned spectroscopy: data and code

Research artifacts for **Reference-conditioned spectroscopy with control drift
and structured hidden sectors**, Gyeongtae Im. Version 1.2.0, 27 September 2026.
Journal target: *The European Physical Journal Plus* (Springer Nature).
This is a research-data release, not a claim of journal acceptance.

The repository contains this paper's numerical data, generators, frozen designs,
tests and figure inputs. It starts a new Git history; it does not include the
other paper, private editorial notes or manuscript drafts.

## Verify before rerunning

Use Python 3.11 or newer and the versions in `requirements.txt` (the versions
used for this release). From the repository root:

```sh
python -m pip install -r requirements.txt
python verify_release.py --tests
```

The first command installs the numerical/plotting environment. The second
checks every released file's SHA256 and executes the bounded tests and passive
network identity checks. It does not rerun all nonlinear fits.

## Primary records

| Directory under `research/` | Role |
|---|---|
| `prl_reference_validation_20260927/run_02` | 792-case reference test, exact stored-LP separation witnesses and matched baselines |
| `structured_dark_sector_20260927/run_02` | 165 cases, 32 false bare-frequency assignments and 66 conditional remainder-bound checks |
| `structured_dark_sector_20260927/figures_v2` | Final structured-sector figure |
| `reference_control_study_20260927/run_01` | 48 cases / 144 joint signal-reference fits; failed profiles retained |
| `reference_control_study_20260927/run_03_wide` | Fixed-count remote-frequency extension, including the three failed fits |

Names containing `prl` are preserved provenance paths from the earlier target;
they do not identify the target of the current article. Earlier output editions
are retained as historical records, not silently substituted for final runs.

Each study's README or `plan.md` gives its noise convention, inputs, scope and
reproduction commands. Use new output directories: the drivers refuse to
overwrite completed runs. A short independent calculation is:

```sh
python -B research/structured_dark_sector_20260927/run_study.py --self-check
```

## Recreate the four reused figures

The four earlier figures (field intercept, pole-zero response, multimode
cofactor and full-zero audit) are preserved as PDFs and PNGs in
`research/figure_revision_20260927/jpc/selected_v2`. Their render manifest
records the original CSVs, code hashes and scoped label changes. Recreate all
four from the released numerical inputs, without running the solvers:

```sh
python -B research/figure_revision_20260927/jpc/render_selected.py --repo . --output reproduction/legacy_figures
```

The output directory must be new. The required inputs are under
`pole_zero_work/data_prl`, and their renderers are under `pole_zero_work/code`.
The optional original generators and their solver modules are also included.
Those older generators write to `pole_zero_work/data_prl` and
`pole_zero_work/fig_prl`; use a separate work copy for recomputation so the
released reference files retain their checksums. The figure command above
needs neither the manuscript nor private handoff notes.

The passive oscillator calculations are conditional model tests, not a
material-specific conserving Weyl self-energy calculation. Necessary linear
constraints and compatible fits do not establish a unique microscopic model.

## Figure styling in version 1.1.0

Version 1.1.0 preserves the scientific calculations and version 1.0.0 history.
It adds the final publication figure typography: standalone bold lowercase
panel labels, consistent axis/tick/legend sizes, and verified text layout.
The shared Matplotlib style checks numerical arrays, axis limits, colors,
line styles and markers before and after each display-only revision.
No model fitting, eigensolver calculation or Monte Carlo study is repeated.

The eight final styled figures and their numerical-artist manifests are under
`research/figure_style_epjp_20260927/epjp/render_03`. Original figure-only PDFs
under `paper/jpc_strengthened/figures` preserve the frozen inventory hashes;
no manuscript is included. The portable wrapper reads that inventory from the
released JSON, so rendering does not require the manuscript's TeX files.

```sh
python -B research/figure_style_epjp_20260927/epjp/render_portable.py --output reproduction/styled_figures
python -B research/figure_style_epjp_20260927/epjp/verify.py --output reproduction/styled_figures
```

Use a new output directory for each render. The release verification also
checks every final asset and the preserved original source/data hashes.

## Restored supplementary evidence in version 1.2.0

Version 1.2.0 adds the computational evidence behind the reintegrated technical
supplement. Version 1.0.0/1.1.0 histories and all earlier scientific bytes remain
unchanged. The expanded archive includes:

- The reduced Weyl, gauge-completion, pole/zero, full-response versus principal-
  part, inverse-problem and visibility generators under `pole_zero_work/code`,
  with their retained `pole_zero_work/data_prl` tables and failed controls.
- Full-band ensemble/Ward/Landau production inputs, independent repeat and
  **historical failed comparison** under `data_prl/full_band_reproduction`.
  Its Nky=16 diagnostic still fails the original 1e-6 tolerance. The later
  current-table repeat passes using a documented replacement aggregation
  recipe; the missing historical recipe was not recovered. Degenerate Landau
  eigenvector overlaps are compared by invariant projector sums; the raw
  differences and per-cell decisions remain available.
- The earlier 11280 conditional oscillator boxes, separate 594 reduced-Weyl
  transfer controls, pilot and first-production records under
  `research/prl_physical_discrimination_20260912`. They are not pooled with the
  later 792-case reference study. Historical `certified` fields mean a
  model-conditioned numerical test, not model validation or outward-rounded
  interval arithmetic.

`release_support/verify_supporting_records.py` checks these distinct input
lineages, retained failures and source hashes without repeating the studies.
The expanded `verify_release.py --tests` also runs the small pole/zero and
gauge-completion controls. SymPy/mpmath are recorded for the existing symbolic
checks; no new scientific implementation was substituted.

Original editorial/manuscript validators inside the historical code directory
are preserved for provenance and are not part of the portable bounded test
command: their absent manuscript files are intentional. The data archive
contains no manuscript TeX, author block or cover letter. Historical README
links to those materials describe the original work tree, not released files.

## Recreate the restored supplemental figures

The eleven restored figures are in `research/supplement_reintegration_20260927/figure_run_05`.
The renderer reads stored numerical tables; it does not refit the data.

```sh
python -B research/supplement_reintegration_20260927/render_restored_figures.py --output reproduction/restored_supplement
```

Use a new output directory.

## Provenance, citation and licensing

`release_manifest.json` identifies the original paths, base commit and exact
released bytes. Some historical records retain absolute paths from the
calculation machine; these are provenance, not required install locations.
Use `CITATION.cff` and the immutable commit/release version when citing this
archive. No DOI is claimed. A DOI-bearing archival copy can be added later.

Existing file-specific licenses are preserved in their original scope. This
export does not impose a new blanket license on previously unlicensed material.
The inherited license notices for the reused `pole_zero_work` materials are
`pole_zero_work/LICENSE`, `LICENSE_CODE.txt` and `LICENSE_DATA.txt` in that
directory; their scope does not extend automatically to the newer studies.
