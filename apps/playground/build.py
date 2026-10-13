"""Assemble the PINNeAPPle Playground into a self-contained static site (standard library only).

    python apps/playground/build.py                    # -> apps/playground/dist/
    python apps/playground/build.py --out site/playground
    python apps/playground/build.py --serve            # build, then serve on http://localhost:8765/

The committed sources are ``apps/playground/site/`` (HTML, JS, CSS, ONNX models, JSON). The build adds the files
that already live elsewhere in the repository instead of copying them into git twice:

* three.js r169 (MIT), vendored in ``pinneapple_tools/visualization/studio/web/vendor/three``;
* the light glTF files of the geometry library, ``data/geometries/<name>/<name>.light.glb``;
* ``sw.js`` with the offline pre-cache list and a content version.

Everything is relative, so the output works under any path (a subdomain, ``/playground/`` on GitHub Pages, or a
file server).
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import shutil
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = Path(__file__).resolve().parent / "site"
THREE = ROOT / "pinneapple_tools" / "visualization" / "studio" / "web" / "vendor" / "three"
THREE_FILES = ["three.module.min.js", "OrbitControls.js", "loaders/GLTFLoader.js", "utils/BufferGeometryUtils.js",
               "environments/RoomEnvironment.js", "LICENSE"]
GEOMETRIES = ROOT / "data" / "geometries"


def build(out: Path) -> dict:
    out = out.resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SITE, out, ignore=shutil.ignore_patterns("*.map", ".DS_Store"))
    for f in THREE_FILES:
        (out / "vendor" / "three" / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(THREE / f, out / "vendor" / "three" / f)
    index = json.loads((SITE / "data" / "geometries.json").read_text())
    (out / "geometries").mkdir()
    missing = []
    for g in index["geometries"]:
        src = GEOMETRIES / g["name"] / f"{g['name']}.light.glb"
        if src.exists():
            shutil.copy2(src, out / "geometries" / src.name)
        else:
            missing.append(g["name"])                 # the page falls back to the Hub copy
    # offline pre-cache: everything except the glTF files (cached when viewed)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
                   and not p.name.endswith(".glb") and p.name != "sw.js")
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.encode())
        digest.update((out / f).read_bytes())
    sw = (out / "sw.js").read_text()
    pre = ["./"] + [f for f in files if f != "index.html"] + ["index.html"]
    sw = sw.replace("__VERSION__", digest.hexdigest()[:12]).replace("__PRECACHE__", json.dumps(pre))
    (out / "sw.js").write_text(sw)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    first = ["index.html", "css/app.css", "js/main.js", "js/models.js", "js/onnx.js", "js/ui.js", "js/trust.js",
             "js/charts.js", "js/cst.js", "js/modes/airfoil.js", "models/airfoil_cst_rans_mlp.onnx",
             "models/airfoil_cst_rans_mlp.json", "icon.svg"]
    first_bytes = sum((out / f).stat().st_size for f in first)
    return {"out": str(out), "files": len(files) + 1, "bytes": size, "first_load_bytes": first_bytes,
            "geometries_missing": missing, "version": digest.hexdigest()[:12]}


def serve(directory: Path, port: int) -> None:
    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                          ".js": "text/javascript", ".onnx": "application/octet-stream",
                          ".webmanifest": "application/manifest+json", ".glb": "model/gltf-binary"}

        def log_message(self, *a):
            pass
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(directory)))
    print(f"serving {directory} on http://localhost:{port}/  (Ctrl-C to stop)")
    httpd.serve_forever()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "dist"))
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    info = build(Path(a.out))
    print(f"built {info['files']} files, {info['bytes'] / 1e6:.1f} MB in {info['out']} "
          f"(first load of the airfoil mode: {info['first_load_bytes'] / 1e3:.0f} kB before compression)")
    if info["geometries_missing"]:
        print(f"note: {len(info['geometries_missing'])} glTF files not in the repo; the page loads them from the Hub",
              file=sys.stderr)
    if a.serve:
        serve(Path(info["out"]), a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
