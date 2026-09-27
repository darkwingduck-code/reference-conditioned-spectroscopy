"""Check final vector panel fonts, source hashes and integrity-check sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from figure_style import assert_science_unchanged, snapshot_figure, style_figure


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "render_03")
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    assert all(sha(REPO / path) == value for path, value in manifest["source_hashes"].items())
    results = {}
    for name, record in manifest["figures"].items():
        path = out / f"{name}.pdf"
        assert sha(path) == record["pdf_sha256"]
        doc = fitz.open(path)
        assert len(doc) == 1
        page = doc[0]
        spans = [span for block in page.get_text("dict")["blocks"]
                 for line in block.get("lines", []) for span in line["spans"]]
        labels = [s for s in spans if re.fullmatch(r"\([a-z]\)", s["text"])]
        assert [s["text"] for s in labels] == [f"({chr(97+i)})" for i in range(record["axis_count"])]
        assert all(s["font"] == "DejaVuSans-Bold" and s["size"] == 11 for s in labels)
        outside = [s["text"] for s in spans if not page.rect.contains(fitz.Rect(s["bbox"]))]
        assert not outside, (name, outside)
        factor = (174 / 25.4 * 72) / page.rect.width
        assert 8 <= 9 * factor <= 12
        results[name] = {"panels": len(labels), "bold_sans_11pt": True,
                         "out_of_page_text": [], "tick_pt_at_174mm": 9 * factor,
                         "panel_pt_at_174mm": 11 * factor}

    # Meaningful negative controls: the publication guard must detect changes.
    detected = []
    for kind in ("line_data", "scatter_color", "axis_limit"):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color="red")
        scatter = ax.scatter([.2, .7], [.3, .6], color="blue")
        before = snapshot_figure(fig)
        style_figure(fig)
        assert_science_unchanged(before, fig)
        if kind == "line_data":
            line.set_ydata([0, 2])
        elif kind == "scatter_color":
            scatter.set_facecolor("green")
        else:
            ax.set_ylim(-1, 1)
        try:
            assert_science_unchanged(before, fig)
        except AssertionError:
            detected.append(kind)
        else:
            raise AssertionError(f"Integrity guard missed {kind}")
        plt.close(fig)
    report = {"status": "PASS", "sources_unchanged": True,
              "figure_count": len(results), "total_panels": sum(r["panels"] for r in results.values()),
              "negative_controls_detected": detected, "figures": results}
    (out / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
