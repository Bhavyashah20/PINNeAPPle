"""Generate, verify and publish the geometry library files.

    python scripts/geometry_library.py light                 # data/geometries/<name>/: light STL + GLB, PNG thumbnail
    python scripts/geometry_library.py hub --out dist/geom   # every format of both levels of detail (for the Hub)
    python scripts/geometry_library.py manifest --out dist/geom   # SHA-256 of all files -> manifest.json
    python scripts/geometry_library.py verify --out dist/geom     # rebuilt files == manifest (CI before upload)
    python scripts/geometry_library.py sheet                 # docs/assets/geometry_library/: contact sheet + GIF
    python scripts/geometry_library.py upload --out dist/geom     # push to the Hub (needs HF_TOKEN)
    python scripts/geometry_library.py verify-hub            # anonymous download + SHA-256 of every published file

Add ``--only name1,name2`` to restrict to some entries. The light files are committed; the Hub carries every
file (light and fine; STEP for the light level only: a faceted STEP of the fine level would be ~100 MB).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinneapple_design import geometry_library as gl  # noqa: E402
from pinneapple_design.geometry_library import store  # noqa: E402

HUB_FILES = [("light", "stl"), ("light", "glb"), ("light", "obj"), ("light", "usda"), ("light", "step"),
             ("fine", "stl"), ("fine", "glb"), ("fine", "obj"), ("fine", "usda")]
DOCS_ASSETS = ROOT / "docs" / "assets" / "geometry_library"


def _names(only):
    return [n for n in gl.list() if not only or n in only.split(",")]


def cmd_light(a):
    for n in _names(a.only):
        t = time.time()
        g = gl.build(n, lod="light")
        d = store.REPO_DATA / n
        d.mkdir(parents=True, exist_ok=True)
        for lod, fmt in sorted(store.COMMITTED):
            g.export(str(d / store.file_name(n, lod, fmt)), fmt)
        DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
        g.render(str(store.thumbnail_path(n)), size=(960, 640))
        size = sum(f.stat().st_size for f in d.iterdir()) / 1e6
        print(f"{n:28s} {g.n_triangles():7d} tris  {size:5.2f} MB  {time.time() - t:5.1f} s")


def cmd_hub(a):
    out = Path(a.out)
    for n in _names(a.only):
        t = time.time()
        for lod in ("light", "fine"):
            g = gl.build(n, lod=lod)
            for l2, fmt in HUB_FILES:
                if l2 == lod:
                    g.export(str(out / n / store.file_name(n, lod, fmt)), fmt)
        print(f"{n:28s} {time.time() - t:5.1f} s")


def _record(m, n, fname, path, location):
    e = m["geometries"].setdefault(n, {"files": {}})
    e["files"][fname] = {"sha256": store.sha256(path), "bytes": os.path.getsize(path), "location": location}


def cmd_manifest(a):
    m = store.read_manifest() if a.only else {"version": 1, "hub_repo": store.HUB_REPO,
                                              "repo_type": store.HUB_REPO_TYPE, "geometries": {}}
    out = Path(a.out) if a.out else None
    for n in _names(a.only):
        spec = gl.get_spec(n)
        m["geometries"][n] = {"files": {}, "title": spec.title, "category": spec.category,
                              "params_hash": {lod: gl.core.params_hash(n, spec.defaults(), lod) for lod in gl.LODS}}
        for lod, fmt in HUB_FILES:
            fname = store.file_name(n, lod, fmt)
            local = store.REPO_DATA / n / fname
            if (lod, fmt) in store.COMMITTED and local.exists():
                _record(m, n, fname, local, "repo+hub")
            elif out is not None and (out / n / fname).exists():
                _record(m, n, fname, out / n / fname, "hub")
        g = gl.build(n, lod="light")
        m["geometries"][n]["n_triangles"] = {"light": g.n_triangles()}
        m["geometries"][n]["dimensions_m"] = {k: round(v, 4) for k, v in g.dimensions().items()}
    store.write_manifest(m)
    print(f"wrote {store.MANIFEST} ({len(m['geometries'])} geometries)")


def cmd_verify(a):
    """Every manifest file rebuilt here must match its SHA-256 (the Hub copy is what users download)."""
    m = store.read_manifest()
    out = Path(a.out)
    bad = 0
    for n in _names(a.only):
        for fname, e in m["geometries"].get(n, {}).get("files", {}).items():
            p = out / n / fname
            if not p.exists():
                p = store.REPO_DATA / n / fname
            if not p.exists():
                print(f"MISSING {n}/{fname}")
                bad += 1
            elif store.sha256(p) != e["sha256"]:
                print(f"SHA MISMATCH {n}/{fname}")
                bad += 1
    print("verify:", "OK" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def cmd_sheet(a):
    from pinneapple_design.geometry_library.render import contact_sheet, turntable
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
    geoms = [gl.build(n) for n in _names(a.only)]
    t = time.time()
    contact_sheet(geoms, str(DOCS_ASSETS / "contact_sheet.png"), cols=4, tile=(480, 330),
                  title="PINNeAPPle geometry library")
    print("contact sheet", round(time.time() - t, 1), "s")
    t = time.time()
    turntable(geoms, str(DOCS_ASSETS / "turntable.gif"), frames=a.frames, cols=4, tile=(240, 165))
    print("turntable", round(time.time() - t, 1), "s", (DOCS_ASSETS / "turntable.gif").stat().st_size // 1024, "KB")


DOC_HEAD = """# Geometry library

Realistic parametric 3-D geometries for Physics-AI, each with the code that regenerates it and ready-made files
you can just import. Every entry is in **metres** in a documented frame, made of **closed, outward-oriented
bodies** (watertight, no self-intersections, checked by `tests/test_geometry_library.py`), at two levels of detail:
`light` (committed in `data/geometries/`, at most a few tens of thousands of triangles) and `fine` (on the Hugging
Face Hub, dataset [`PINNeAPPle-Labs/geometry-library`](https://huggingface.co/datasets/PINNeAPPle-Labs/geometry-library)).
Real-world dimensions come from the published sources listed by `info()`; values chosen by engineering judgement
are flagged as estimates.

![All geometries](../assets/geometry_library/contact_sheet.png)

```python
import pinneapple as pp

pp.geometries.list()                                  # names (also pp.geometries.categories())
pp.geometries.info("wind_turbine_nrel5mw")            # dimensions, parameters with ranges, sources, physics uses
g = pp.geometries.build("wind_turbine_nrel5mw", pitch_deg=8.0, rotor_azimuth_deg=30.0)
g.check()                                             # watertight / outward per body, bounding box
g.export("turbine.stl"); g.export("turbine.glb"); g.export("turbine.step"); g.export("turbine.usda")
g.solid()                                             # one fused watertight surface for meshing (manifold3d)
g.scene()                                             # pinneapple_tools.visualization.studio Scene: Blender, web viewer
g.render("turbine.png")                               # thumbnail (software renderer, no GPU)
pp.geometries.load("airliner_a320", format="glb")     # path of the pre-generated file
pp.geometries.load("airliner_a320", format="stl", lod="fine")   # downloaded from the Hub, SHA-256 checked, cached
```

## Files and the loader

| level | triangles | where | formats |
|---|---|---|---|
| `light` | <= 20 000 (36 000 for the airliner and the two-stage launcher) | repository (`data/geometries/<name>/`) and Hub | STL, glTF (in git); OBJ, USD, STEP (Hub) |
| `fine` | up to 400 000 | Hub | STL, glTF, OBJ, USD |

`load(name, format, lod)` looks in the repository copy, then the cache (`~/.cache/pinneapple/geometries`,
`$PINNEAPPLE_GEOMETRY_CACHE`), then the Hub, verifies the SHA-256 recorded in
`pinneapple_design/geometry_library/manifest.json`, and falls back to `build()` with the default parameters when
offline (`PINNEAPPLE_OFFLINE=1`) or when the file is not published yet. The defaults reproduce the stored files
(the test compares the rebuilt light STL with the committed one). STEP is a faceted B-rep (one solid per body, planar
triangular faces) that OpenCASCADE, FreeCAD and the usual CAD tools import as solids; it is published for the light
level (the fine one would be about 100 MB, export it locally with `g.export("x.step")` if needed).

The `geometries` job of `.github/workflows/publish-zoo.yml` rebuilds every file, checks it against the manifest and
uploads it to the Hub when the repository secret `HF_TOKEN` is set (it skips the upload with a warning otherwise), then
verifies the published files by anonymous download.
Regenerate after changing an entry:

```bash
python scripts/geometry_library.py light       # light STL + glTF, thumbnails
python scripts/geometry_library.py hub         # all files of both levels in dist/geometries
python scripts/geometry_library.py manifest    # SHA-256 of everything
python scripts/geometry_library.py sheet       # contact sheet + turntable GIF
python scripts/geometry_library.py docs        # this page
```

## Reference frames

Vehicles: right-handed, x aft from the nose (the free-stream direction), y to starboard, z up, z = 0 on the ground for
ground vehicles. Other entries state their own frame in `info(name)["frame"]` (e.g. the wind turbine: x downwind, z up
from the tower base; the aorta: patient axes). Every geometry is in metres, including the heat sink and the aorta.

## Examples

- `examples/geometry_library/01_caarc_building_wind_openfoam.py`: the CAARC tall building in an atmospheric
  boundary layer with OpenFOAM through the generic STL path of `pp.cfd.ExternalFlow` (writes the case when
  OpenFOAM is not installed).
- `examples/geometry_library/02_wind_turbine_bem.py`: blade-element momentum on the NREL 5-MW blade of the library:
  power and thrust curve, compared with the published rated power.
- `examples/geometry_library/03_airliner_vlm.py`: vortex-lattice lift slope, neutral point and induced drag of the
  airliner, plus a wetted-area friction-drag estimate from the library mesh.

![Turntable](../assets/geometry_library/turntable.gif)

## Gallery
"""


def cmd_docs(a):
    lines = [DOC_HEAD]
    for cat, names in gl.categories().items():
        lines.append(f"\n### {cat.replace('_', ' ').capitalize()}\n")
        for n in names:
            i = gl.info(n)
            d = i["dimensions"]
            dims = " x ".join(f"{d[k]['value_m']:.4g}" for k in ("length_x", "width_y", "height_z"))
            lines.append(f"#### {i['title']} (`{n}`)\n")
            lines.append(f"![{n}](../assets/geometry_library/{n}.png)\n")
            lines.append(i["description"] + "\n")
            lines.append(f"- Bounding box (default build, L x W x H): {dims} m. Frame: {i['frame']}.")
            ps = ", ".join(f"`{p['name']}`" + (f" [{p['unit']}]" if p["unit"] and p["unit"] != "-" else "")
                           for p in i["parameters"])
            lines.append(f"- Parameters: {ps}.")
            lines.append(f"- Physics: {'; '.join(i['physics'])}.")
            src = "; ".join(("*(estimate)* " if s["estimate"] else "") + f"{s['what']}: {s['ref']}" for s in i["sources"])
            lines.append(f"- Sources: {src}")
            if i["legacy"]:
                lines.append(f"- Wraps `{i['legacy']}` (still importable from there).")
            if i["known_issues"]:
                lines.append("- Known: " + "; ".join(f"{k}: {v}" for k, v in i["known_issues"].items()) + ".")
            lines.append("")
    out = ROOT / "docs" / "guides" / "geometry_library.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


def cmd_upload(a):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("::warning::HF_TOKEN is not set; skipping the upload of the geometry library to the Hub")
        return 0
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    repo = os.environ.get("PINNEAPPLE_GEOMETRY_HUB", store.HUB_REPO)
    api.create_repo(repo, repo_type=store.HUB_REPO_TYPE, exist_ok=True)
    out = Path(a.out)
    for n in _names(a.only):
        for f in list((store.REPO_DATA / n).glob("*")) + [store.thumbnail_path(n)]:   # committed files, also on the Hub
            dest = out / n / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if f.exists() and not dest.exists():
                dest.write_bytes(f.read_bytes())
    (out / "manifest.json").write_bytes(store.MANIFEST.read_bytes())
    (out / "README.md").write_text(_card())
    api.upload_folder(folder_path=str(out), repo_id=repo, repo_type=store.HUB_REPO_TYPE,
                      commit_message="Update the geometry library files")
    print(f"uploaded {out} -> https://huggingface.co/datasets/{repo}")
    return 0


def cmd_verify_hub(a):
    """Download every manifest file anonymously from the Hub and compare its SHA-256 with the manifest."""
    import hashlib
    import urllib.request
    m = store.read_manifest()
    bad = n_ok = 0
    for n in _names(a.only):
        for fname, e in m["geometries"].get(n, {}).get("files", {}).items():
            url = store.hub_url(n, fname)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pinneapple"}),
                                            timeout=120) as r:
                    h = hashlib.sha256(r.read()).hexdigest()
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {url}: {exc}")
                bad += 1
                continue
            if h != e["sha256"]:
                print(f"SHA MISMATCH {url}")
                bad += 1
            else:
                n_ok += 1
    print(f"verify-hub: {n_ok} files OK, {bad} problem(s); https://huggingface.co/datasets/{store.HUB_REPO}")
    return 1 if bad else 0


def _card() -> str:
    return ("---\nlicense: apache-2.0\ntags: [geometry, cfd, physics-ai, stl, gltf, step]\n---\n"
            "# PINNeAPPle geometry library\n\nParametric 3-D geometries for Physics-AI (metres, closed bodies, "
            "documented frames). Generated by `scripts/geometry_library.py` in "
            "https://github.com/PINNeAPPle-Labs/pinneapple; `pp.geometries.load(name, format, lod)` downloads "
            "and verifies these files (SHA-256 in `manifest.json`).\n\n" + gl.summary_table() + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["light", "hub", "manifest", "verify", "sheet", "upload", "docs", "verify-hub"])
    p.add_argument("--out", default="dist/geometries")
    p.add_argument("--only", default="")
    p.add_argument("--frames", type=int, default=24)
    a = p.parse_args(argv)
    return {"light": cmd_light, "hub": cmd_hub, "manifest": cmd_manifest, "verify": cmd_verify,
            "sheet": cmd_sheet, "upload": cmd_upload, "docs": cmd_docs, "verify-hub": cmd_verify_hub}[a.command](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
