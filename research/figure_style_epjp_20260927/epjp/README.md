# EPJ Plus figure typography revision

Final assets: **`render_03/`**, eight vector PDFs and matching PNG previews.
Intermediate layout checks remain in the source project history; this data
release includes only the final styled edition.

The frozen source inventory was derived from the main/supplement figure inputs: five main figures and three supplementary
figures, 27 panels in total. `render_03/mapping.json` maps each original
include to its replacement and records both PDF hashes. `manifest.json`
records the original renderer/data hashes and the scientific artist snapshots.

Style: standalone lowercase panel labels in bold DejaVu Sans 11 pt; axis
labels 10 pt; ticks, legends, and ordinary annotations 9 pt. Original math
symbols remain mathematical typesetting. Descriptive panel titles move into
captions; the original descriptors are retained in the figure manifest. The 7.1-inch plotting canvases are intended
for the parent's 174 mm text width. PDF cropping changes the exact width,
so `verification.json` records the actual final panel/tick sizes at 174 mm.

From repository root, using the already installed environment:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -B research/figure_style_epjp_20260927/epjp/render_portable.py --output research/figure_style_epjp_20260927/epjp/reproduced
python -B research/figure_style_epjp_20260927/epjp/verify.py --output research/figure_style_epjp_20260927/epjp/reproduced
python -m ruff check --no-cache --select F,E9 research/figure_style_epjp_20260927/figure_style.py research/figure_style_epjp_20260927/epjp
```

Choose a new output directory; the renderer refuses to overwrite a prior
result. It calls the existing original plotting functions or their AST-extracted
plotting blocks and reads frozen CSV/JSON results. The structured-kernel curves
are evaluated by their original analytic plotting function. No fit, LP,
optimization, or new experiment is executed. All modifications occur in the
Matplotlib Figure before saving; PDFs/raster images are not post-edited.

Integrity checks compare plotted line arrays, collection offsets/paths/colors,
image arrays and color scales, bar geometry, axis limits/scales, line styles,
marker shapes/sizes and colors before and after styling. The checker includes
negative controls that deliberately alter line data, a scatter color, and an
axis limit; each must be detected. Final PDF text is independently checked for
27 DejaVuSans-Bold 11 pt panel labels and for text outside page bounds.

All eight figures were visually inspected after enlarging the fonts. Text-only
layout corrections moved legends, wrapped category labels, and separated
condition annotations from plotted points. The final two changes in render_03
were the damping legend and the bare-frequency legend; both were re-inspected.

Scientific sources, frozen results, old renderers and original JPC/PRL figures
are unchanged. No manuscript source is included in this data release. The
portable wrapper reads the frozen JSON inventory instead of the private TeX
source; all plotting code and numerical inputs remain the same.
