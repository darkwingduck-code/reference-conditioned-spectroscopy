"""Run the frozen figure renderer using its public JSON inventory, without TeX."""
import json

import render


def public_inventory():
    manifest = render.HERE / "render_03/manifest.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))["inventory"]
    for record in records:
        record["asset"] = record["asset"].replace("\\", "/")
        record["source_tex"] = record["source_tex"].replace("\\", "/")
        path = (render.REPO / record["asset"]).resolve()
        if not path.is_relative_to(render.REPO):
            raise ValueError("Unsafe inventory asset path")
        if render.sha(path) != record["sha256"]:
            raise ValueError("Frozen inventory asset changed: " + record["asset"])
    return records


if __name__ == "__main__":
    render.inventory = public_inventory
    render.main()
