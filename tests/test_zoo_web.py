"""Zoo models in ONNX and for the browser: export parity with PyTorch (1e-5), the per-query trust checks and their
verdict, the CST geometry graphs, and the committed browser bundle (data/web) being current."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pinneapple as pp  # noqa: E402
from pinneapple_hub import zoo  # noqa: E402
from pinneapple_hub.zoo import checks, export  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "web" / "zoo"
MODELS = ["airfoil_cst_rans_mlp", "heatsink_forced_mlp", "rover_contact_grc1_viper"]


@pytest.fixture(scope="module", params=MODELS)
def model(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return zoo.load(request.param, offline=True)


@pytest.fixture(scope="module")
def graphs(model):
    return export.export_bytes(model), export.export_bytes(model, checks=True)


def test_export_matches_pytorch_on_the_zoo_test_sets(model, graphs):
    plain, with_checks = graphs
    assert export.onnx_parity(model, plain) <= 1e-5
    assert export.onnx_parity(model, with_checks) <= 1e-5
    assert export.checks_parity(model, with_checks)["status_flips"] == 0


def test_export_onnx_writes_and_reports(model, tmp_path):
    info = pp.zoo.export_onnx(model, tmp_path / "m.onnx")
    assert (tmp_path / "m.onnx").read_bytes()
    assert info["parity"] <= 1e-5 and info["input_columns"] == model.input_names
    out = export.run_onnx(tmp_path / "m.onnx", export.parity_inputs(model, n_random=0)[:3])
    assert set(out) == set(export.output_names(model))


def test_range_checks_agree_with_in_range(model):
    X = export.parity_inputs(model, n_random=200)
    rng = np.random.default_rng(1)
    Y = X[rng.choice(len(X), 200)] * rng.uniform(0.6, 1.4, (200, X.shape[1]))      # many out of range
    Z = np.concatenate([X[:200], Y])
    res = checks.query_checks(model, Z)
    failed = np.array([any(c["status"] == "fail" and c["rule"] == "PC601" for c in r["checks"]) for r in res])
    assert np.array_equal(failed, ~model.in_range(Z))
    assert failed.any() and (~failed).any()


def test_held_out_inputs_are_trusted(model):
    """On the held-out inputs inside the range, nearly every query is trusted (the checks are not alarmist)."""
    X = export.parity_inputs(model, n_random=0)
    X = X[model.in_range(X)]
    st = [r["verdict"]["status"] for r in checks.query_checks(model, X)]
    assert "fail" not in st
    assert st.count("pass") / len(st) > 0.95


def test_verdict_rules():
    spec = {"checks": [checks._chk("a", "PC1", "a", "<=", 1.0, "fail"), checks._chk("b", "PC2", "b", "<=", 1.0, "warn"),
                       checks._chk("d", "", "d", "<=", 1.0, "warn", kind="design")],
            "aux": [], "groups": {}, "verdicts": checks.VERDICTS}
    assert checks.evaluate(spec, [0, 0, 5], [])["verdict"]["status"] == "pass"      # design never decides
    assert checks.evaluate(spec, [0, 2, 0], [])["verdict"]["status"] == "warn"
    assert checks.evaluate(spec, [2, 2, 0], [])["verdict"]["status"] == "fail"
    assert checks.evaluate(spec, [float("nan"), 0, 0], [])["verdict"]["status"] == "fail"


def test_airfoil_query_checks_are_the_zoo_physics_checks():
    a = zoo.load("airfoil_cst_rans_mlp", offline=True)
    from pinneapple_design.aero.geometry import naca4
    r = pp.zoo.query_checks(a, [[*naca4("2412"), 4.0], [*naca4("2412"), 16.0], [0.3, 0.15, 0.15, 0.02, 0.05, 0.03, 4]])
    assert [x["verdict"]["status"] for x in r] == ["pass", "fail", "fail"]
    slope = next(c for c in r[0]["checks"] if c["id"] == "slope")
    assert slope["value"] < 0.15 and "dCl/dα" in slope["detail_text"]
    assert next(c for c in r[2]["checks"] if c["id"] == "range:T0")["status"] == "fail"


def test_geometry_graphs_match_numpy(tmp_path):
    from pinneapple_design.aero import geometry as g
    from pinneapple_design.aero import geometry_onnx as go
    meta = go.export_onnx(tmp_path)
    rng = np.random.default_rng(0)
    S = np.r_[g.BOUNDS[:, 0] + rng.random((40, 6)) * (g.BOUNDS[:, 1] - g.BOUNDS[:, 0]), [[0.12, 0.3, -0.3, 0, 0, 0]]]
    s = ort.InferenceSession(str(tmp_path / meta["files"]["geometry"]))
    out = dict(zip([o.name for o in s.get_outputs()], s.run(None, {"x": S}), strict=True))
    ref = go.reference(S)
    for k, v in ref.items():
        np.testing.assert_allclose(out[k], v, atol=1e-10, err_msg=k)
    back = ort.InferenceSession(str(tmp_path / meta["files"]["from_handles"])).run(None, {"x": ref["handles"]})[0]
    np.testing.assert_allclose(back, S, atol=1e-12)
    assert meta["presets"]["NACA 2412"]["out_of_bounds"] == []


def test_shape_from_handles_round_trip():
    from pinneapple_design.aero import geometry as g
    s = g.naca4("4412")
    hu, hl = g.handles(s)
    np.testing.assert_allclose(g.shape_from_handles(hu, hl), s, atol=1e-12)
    assert g.out_of_bounds(s) == ["M1"]


# ---------------------------------------------------------------------- the committed bundle (data/web)
def test_committed_bundle_runs_like_pytorch(model):
    meta = json.loads((BUNDLE / f"{model.name}.json").read_text())
    data = (BUNDLE / meta["onnx"]["file"]).read_bytes()
    assert meta["onnx"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert export.onnx_parity(model, data) <= 1e-5
    assert export.checks_parity(model, data)["status_flips"] == 0


def test_committed_bundle_is_current(model):
    meta = json.loads((BUNDLE / f"{model.name}.json").read_text())
    spec = json.loads(json.dumps(checks.check_spec(model)))
    assert [c["id"] for c in meta["checks"]["checks"]] == [c["id"] for c in spec["checks"]]
    for a, b in zip(meta["checks"]["checks"], spec["checks"], strict=True):
        assert a["op"] == b["op"] and a["detail"] == b["detail"]
        assert np.allclose(np.ravel(a["limit"]), np.ravel(b["limit"]), rtol=1e-5)
    card = model.card.to_dict()
    assert meta["card"]["version"] == card["version"]
    from pinneapple_hub.zoo._artifacts import manifest
    assert meta["hub"] == f"https://huggingface.co/{manifest()['models'][model.name]['hf_repo']}"   # the published repo
    for k, v in card["validation_metrics"].items():
        assert meta["card"]["validation_metrics"][k] == pytest.approx(v, rel=1e-5)


def test_committed_fixtures_are_current():
    from pinneapple_hub.zoo.web import fixtures
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = fixtures(offline=True)
    old = json.loads((BUNDLE / "fixtures.json").read_text())
    for name, f in new["models"].items():
        o = old["models"][name]
        assert o["inputs"] == f["inputs"] and o["in_range"] == f["in_range"]
        assert [e["statuses"] for e in o["evaluation"]] == [e["statuses"] for e in f["evaluation"]]
        for k, v in f["predict"].items():                  # float32 networks: torch builds differ at ~1e-7
            v = np.asarray(v, float)
            np.testing.assert_allclose(o["predict"][k], v, rtol=1e-5, atol=1e-5 * max(np.abs(v).max(), 1e-3))
    np.testing.assert_allclose(old["geometry"]["props"], new["geometry"]["props"], atol=1e-12)


def test_committed_gallery_index_is_current():
    from pinneapple_design import geometry_library as gl
    from pinneapple_design.geometry_library.web import index
    old = json.loads((ROOT / "data" / "web" / "geometries" / "index.json").read_text())
    assert json.loads(json.dumps(index(), ensure_ascii=False)) == old
    for n in gl.list():
        assert (ROOT / "data" / "web" / "geometries" / "thumbs" / f"{n}.jpg").exists()


def test_range_check_accepts_the_bounds_themselves(model):
    # regression: the checks run in float32, so a bound such as 0.08 arrives as 0.0799999982 and a slack of 1e-9
    # rejected a query exactly at the edge of the training range as "outside the validity range"
    ranges = [c for c in checks.check_spec(model)["checks"]
              if c["id"].startswith("range:") and c["id"][6:] in model.input_names]
    X = export.parity_inputs(model, n_random=0)
    X = X[model.in_range(X)][:1].repeat(2 * len(ranges), axis=0)
    for k, c in enumerate(ranges):
        j = model.input_names.index(c["id"][6:])
        X[2 * k, j], X[2 * k + 1, j] = c["bounds"]
    res = checks.query_checks(model, X)
    for k, c in enumerate(ranges):
        for r in res[2 * k: 2 * k + 2]:
            assert next(x["status"] for x in r["checks"] if x["id"] == c["id"]) == "pass", (c["id"], c["bounds"])
