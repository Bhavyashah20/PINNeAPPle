"""PINNeAPPle Playground (apps/playground): the static build is self-contained and the app holds UI code only.
The models, checks and geometry it serves are library artefacts, tested in tests/test_zoo_web.py and
tests/js/zoo_web.test.mjs; the browser test is apps/playground/tests/e2e."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "playground"


def _build_module():
    spec = importlib.util.spec_from_file_location("playground_build", APP / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_static_build_is_self_contained(tmp_path):
    info = _build_module().build(tmp_path / "site")
    out = Path(info["out"])
    for f in ("index.html", "lib/onnx.js", "lib/zoo.js", "vendor/three/three.module.min.js",
              "models/airfoil_cst_rans_mlp.onnx", "models/cst_geometry.onnx", "data/geometries.json"):
        assert (out / f).exists(), f
    assert not (out / "models" / "fixtures.json").exists()
    assert "__PRECACHE__" not in (out / "sw.js").read_text()
    assert info["first_load_bytes"] < 1_000_000          # < 3 s on a normal connection
    assert "http://" not in (out / "index.html").read_text()


def test_the_app_holds_ui_code_only():
    """No ONNX file, model data or Python in the app: those are produced by the library."""
    site = APP / "site"
    assert not list(site.rglob("*.onnx")) and not list(site.rglob("*.npz"))
    assert not list(APP.glob("tools/*.py"))
    js = "\n".join(p.read_text() for p in site.rglob("*.js"))
    for forbidden in ("Math.acos", "naca4", "Bernstein", "parseModel"):
        assert forbidden not in js, forbidden
