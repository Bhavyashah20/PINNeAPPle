import importlib.util
import json
import math
import shutil

import pytest

from pinneapple_design.fitmesh import TEMPLATES, FitMeshSession, LLMAssist, gci, parse, plan_mesh
from pinneapple_design.fitmesh.context import wall_unit_height
from pinneapple_design.fitmesh.llm import (
    check_code,
    propose_parameters,
    rank_geometries,
    read_problem,
    review_mesh_plan,
)
from pinneapple_design.fitmesh.suggest import sample, suggest_geometries, suggest_parameters


# ---------------------------------------------------------------- reading the problem
def test_parse_portuguese_and_english():
    c = parse("Quero reduzir o arrasto de um carro a 40 m/s no ar, comprimento 4,5 m, para triagem")
    assert c.physics == ["external_flow"] and c.get("fluid") == "air"
    assert c.get("speed") == pytest.approx(40.0) and c.get("length") == pytest.approx(4.5)
    assert c.accuracy == "screening" and "drag coefficient" in c.get("quantities")
    c = parse("Steel bracket with a bolt hole, 5 kN load, stress concentration, fatigue sign-off; 3 inch wide")
    assert c.get("material") == "steel" and c.get("load") == pytest.approx(5000.0)
    assert c.accuracy == "verification" and c.get("length") == pytest.approx(0.0762)
    assert [q["key"] for q in c.questions()] == []


def test_questions_for_missing_facts():
    keys = [q["key"] for q in parse("Flow around a cylinder").questions()]
    assert {"fluid", "speed", "accuracy"} <= set(keys)


def test_first_cell_height_matches_hand_calculation():
    re = 1.0 * 40.0 / 1.5e-5 * 1.0 / 1.0                      # L = 1 m
    w = wall_unit_height(1.0, 40.0, 1.0, 1.5e-5, re)
    cf = 0.026 * re ** (-1 / 7)
    assert w["cf"] == pytest.approx(cf) and w["y1"] == pytest.approx(1.5e-5 / (40.0 * math.sqrt(cf / 2)))
    pipe = wall_unit_height(1.0, 2.0, 0.0762, 1e-6, 152400, internal=True)
    assert pipe["cf"] == pytest.approx(0.316 * 152400 ** -0.25 / 4)


# ---------------------------------------------------------------- catalogue, suggestions, CAD
needs_cq = pytest.mark.skipif(importlib.util.find_spec("cadquery") is None, reason="cadquery not installed")


@needs_cq
@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_every_template_builds_a_valid_solid_with_the_exact_volume(key):
    t = TEMPLATES[key]
    v = t.complete({})
    solid = t.build(v).val()
    assert solid.isValid() and t.check_params(v) == []
    if t.volume is not None:
        assert solid.Volume() == pytest.approx(t.volume(v), rel=1e-6)


@needs_cq
def test_geometry_and_parameter_suggestions():
    c = parse("Erosion and pressure drop in a 3 inch pneumatic conveying pipe bend, air at 20 m/s")
    rows = suggest_geometries(c)
    assert rows[0]["template"] == "pipe_bend_90" and rows[0]["references"]
    t = TEMPLATES["pipe_bend_90"]
    ps = {p["name"]: p for p in suggest_parameters(t, c)}
    assert ps["diameter"]["vary"] and ps["diameter"]["base"] == pytest.approx(0.0762)
    lo, hi = ps["diameter"]["range"]
    assert lo == pytest.approx(0.0381) and hi == pytest.approx(0.1524)        # factor 2 around the base
    vs = sample(t, {k: p["base"] for k, p in ps.items()}, {"diameter": [lo, hi]}, n=4)
    assert vs[0]["diameter"] == pytest.approx(0.0762) and len(vs) == 4
    assert all(v["inlet_length"] >= 10 * v["diameter"] - 1e-12 for v in vs)  # development length follows D


@needs_cq
def test_mesh_plan_rules():
    t = TEMPLATES["ahmed_body"]
    v = t.complete({})
    p_screen = plan_mesh(t, v, parse("quick screening of the drag of a car at 40 m/s in air"))
    p_design = plan_mesh(t, v, parse("drag of a car at 40 m/s in air, design study"))
    assert p_screen.layers["y_plus"] == 50.0 and p_design.layers["y_plus"] == 1.0
    assert p_design.layers["first_height"] < p_screen.layers["first_height"]
    assert p_design.estimated_cells > p_screen.estimated_cells and p_design.family == [1.0, 1.5]
    assert 0 < p_design.domain["blockage"] < 0.03
    naca = TEMPLATES["naca4_wing"]
    pv = plan_mesh(naca, naca.complete({}), parse("NACA airfoil lift at 30 m/s in air, verification"))
    assert pv.domain.get("two_d") and pv.family == [1.0, 1.5, 2.25]
    plate = TEMPLATES["plate_with_hole"]
    pf = plan_mesh(plate, plate.complete({}), parse("steel plate with a hole, stress concentration, design"))
    assert pf.kind == "solid" and pf.tool == "gmsh" and pf.sizes["at_hole"] < pf.sizes["global"]


def test_gci_recovers_order_and_flags_oscillation():
    f = lambda h: 1.0 + 0.01 * h ** 2                                  # noqa: E731
    r = gci([f(1), f(1.5), f(2.25)], [1, 1.5, 2.25])
    assert r["p"] == pytest.approx(2.0) and r["extrapolated"] == pytest.approx(1.0) and r["status"] == "pass"
    assert gci([1.0, 1.02, 0.99], [1, 1.5, 2.25])["status"] == "warn"


# ---------------------------------------------------------------- language-model help (fake model)
def _fake(answers):
    def call(prompt, system):
        for key, ans in answers.items():
            if key in system:
                return json.dumps(ans)
        return "{}"
    return LLMAssist(call=call)


@needs_cq
def test_llm_outputs_are_validated():
    llm = _fake({"extract engineering facts": {"physics": ["external_flow", "magic"], "speed": "12", "fluid": "lava",
                                               "accuracy": "design"},
                 "choose reference geometries": {"choices": [{"template": "sphere", "why": "ball"},
                                                             {"template": "dragon", "why": "x"}]},
                 "which geometry parameters": {"parameters": [{"name": "diameter", "vary": True,
                                                               "range": [-1, 100]}, {"name": "bogus"}]},
                 "review a CFD/FEA mesh plan": {"changes": {"y_plus": 15, "growth": 1.1}}})
    facts = read_problem(llm, "a ball in wind")
    assert facts == {"physics": ["external_flow"], "speed": 12.0, "accuracy": "design"}
    c = parse("drag of a sphere in air at 10 m/s, design")
    got = rank_geometries(llm, c)
    assert [x["template"] for x in got["choices"]] == ["sphere"] and got["dropped"] == ["dragon"]
    t = TEMPLATES["sphere"]
    rows = propose_parameters(llm, t, c, t.defaults())
    assert len(rows) == 1 and rows[0]["range"] == [1e-6, 10.0] and rows[0]["note"]
    plan = review_mesh_plan(llm, plan_mesh(t, t.complete({}), c), c)
    assert plan.layers["y_plus"] == 1.0 and plan.layers["growth"] == 1.1          # buffer-layer y+ refused
    assert any("refused" in r for r in plan.rationale)


@needs_cq
def test_model_code_is_checked_before_it_runs(tmp_path):
    assert check_code("import os\ndef build(p):\n    return os.system('x')")
    assert check_code("def build(p):\n    return open('x')")
    assert check_code("x = 1") == ["no build(p) function"]
    from pinneapple_design.fitmesh.llm import run_free_geometry
    good = "import cadquery as cq\ndef build(p):\n    return cq.Workplane('XY').box(p['a'], p['a'], p['a'])\n"
    r = run_free_geometry(good, {"a": 0.1}, str(tmp_path), 1e-4)
    assert r["ok"] and r["valid"] and r["volume"] == pytest.approx(1e-3, rel=1e-6)
    bad = run_free_geometry("import subprocess\ndef build(p):\n    pass", {}, str(tmp_path), 1e-4)
    assert not bad["ok"] and "refused" in bad["error"]


# ---------------------------------------------------------------- end to end
needs_gmsh = pytest.mark.skipif(importlib.util.find_spec("gmsh") is None or importlib.util.find_spec("cadquery") is None,
                                reason="gmsh or cadquery not installed")


@needs_gmsh
def test_session_end_to_end_with_gmsh(tmp_path):
    s = FitMeshSession.run("Steel plate with a bolt hole in tension, stress concentration, screening",
                           str(tmp_path), n=2, log=lambda m: None)
    assert s.template.key == "plate_with_hole" and len(s.variants) == 2
    assert all(v.verdict == "pass" for v in s.variants)
    assert s.meshes and all(m.cells > 0 and not m.error for m in s.meshes)
    rep = json.load(open(tmp_path / "report.json"))
    stages = {x["stage"]: x["verdict"] for x in rep["stages"]}
    assert stages["cad"] == "pass" and stages["mesh plan"] == "pass" and "meshes" in stages
    assert (tmp_path / "report.html").read_text().count("<section>") >= 5


@needs_gmsh
@pytest.mark.skipif(shutil.which("snappyHexMesh") is None and not __import__("os").path.exists(
    "/usr/share/openfoam/etc/bashrc"), reason="OpenFOAM not installed")
def test_snappy_internal_flow_has_ports_and_layers(tmp_path):
    s = FitMeshSession("Pressure drop in a 3 inch pipe bend with water at 2 m/s, quick screening", str(tmp_path),
                       log=lambda m: None)
    s.choose_geometry("pipe_bend_90")
    s.choose_parameters(vary={}, n=1)
    s.generate_cad()
    s.plan_meshes()
    r = s.generate_meshes(levels=[1.5], max_cells=4e5)[0]
    assert not r.error and r.layers["checkMesh_ok"]
    names = {p["name"] for p in r.quality["summary"]["patches"]}
    assert {"inlet", "outlet", "wall"} <= names
    assert r.layers["patches"]["wall"]["layers"] > 0


@needs_gmsh
def test_http_api_walkthrough(tmp_path):
    from fastapi.testclient import TestClient

    from pinneapple_design.fitmesh.api import create_app
    c = TestClient(create_app(str(tmp_path)))
    r = c.post("/sessions", json={"text": "Steel plate with a bolt hole in tension, stress concentration"}).json()
    sid = r["id"]
    assert r["step"] in ("problem", "geometry") and r["options"][0]["template"] == "plate_with_hole"
    c.post(f"/sessions/{sid}/answer", json={"accuracy": "screening", "length": 0.2, "load": 5000})
    c.post(f"/sessions/{sid}/geometry", json={"key": "plate_with_hole"})
    assert c.post(f"/sessions/{sid}/parameters", json={"vary": {"hole_diameter": [0.005, 0.02]}, "n": 2}).status_code == 200
    assert all(v["verdict"] == "pass" for v in c.post(f"/sessions/{sid}/cad").json()["variants"])
    assert c.post(f"/sessions/{sid}/plan").json()["plans"]
    rep = c.get(f"/sessions/{sid}/report").json()
    assert rep["overall"] in ("pass", "warn")
    assert c.get(f"/sessions/{sid}/files/report.html").status_code == 200
    assert c.get(f"/sessions/{sid}/files/../../etc/passwd").status_code == 404
    assert c.post("/sessions/nope/cad").status_code == 404
