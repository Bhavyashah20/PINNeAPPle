"""The geometry library for the browser: an index of every entry and small thumbnails.

    python -m pinneapple_design.geometry_library.web data/web/geometries

``index.json`` lists, per geometry, what a gallery card needs: title, category, description, dimensions, triangle
count, the light glTF file (repo path and Hub URL), the sources of the dimensions (estimates flagged), the physics
it is built for, its parameters, a link to the builder's source and the code that regenerates it.
``thumbs/<name>.jpg`` are 360x240 JPEGs of the committed renders (``docs/assets/geometry_library``).
"""
from __future__ import annotations

import inspect
import io
import json
import sys
from pathlib import Path
from typing import Any

GITHUB = "https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main"
ROOT = Path(__file__).resolve().parents[2]


def index() -> dict[str, Any]:
    """The gallery index (see the module docstring)."""
    from . import core
    from . import list as names
    from .store import hub_url, read_manifest
    man = read_manifest()["geometries"]
    items = []
    for n in names():
        spec = core.get_spec(n)
        info = spec.info()
        try:
            src = Path(inspect.getsourcefile(spec.builder)).resolve().relative_to(ROOT).as_posix()
            code = f"{GITHUB}/{src}#L{inspect.getsourcelines(spec.builder)[1]}"
        except (TypeError, ValueError, OSError):
            code = f"{GITHUB}/pinneapple_design/geometry_library"
        m = man.get(n, {})
        glb = f"{n}.light.glb"
        items.append({
            "name": n, "title": info["title"], "category": info["category"], "description": info["description"],
            "dimensions_m": m.get("dimensions_m", {}), "triangles": m.get("n_triangles", {}).get("light"),
            "glb": f"geometries/{glb}", "glb_repo": f"data/geometries/{n}/{glb}",
            "glb_bytes": m.get("files", {}).get(glb, {}).get("bytes"), "glb_hub": hub_url(n, glb),
            "frame": info["frame"], "physics": info["physics"], "tags": info["tags"], "sources": info["sources"],
            "parameters": [{k: p.get(k) for k in ("name", "default", "unit", "description") if k in p}
                           for p in info["parameters"]],
            "code": code,
            "regenerate": f'import pinneapple as pp\ng = pp.geometries.build("{n}", lod="light")\ng.export("{n}.glb")',
        })
    return {"version": 1, "geometries": items}


def export_web(directory: str | Path, *, thumb_size=(360, 240)) -> dict[str, Any]:
    """Write ``index.json`` and ``thumbs/<name>.jpg`` into ``directory``; returns the index."""
    from PIL import Image

    from .store import THUMBS
    d = Path(directory)
    (d / "thumbs").mkdir(parents=True, exist_ok=True)
    idx = index()
    (d / "index.json").write_text(json.dumps(idx, indent=1, ensure_ascii=False) + "\n")
    for g in idx["geometries"]:
        im = Image.open(THUMBS / f"{g['name']}.png").convert("RGB")
        im.thumbnail(thumb_size, Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, "JPEG", quality=82, optimize=True)
        (d / "thumbs" / f"{g['name']}.jpg").write_bytes(b.getvalue())
    return idx


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/web/geometries")
    print(f"{len(export_web(out)['geometries'])} geometries -> {out}")
