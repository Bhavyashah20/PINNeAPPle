import json
import os
import re

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.openfoam.campaign import (
    _V,
    Campaign,
    CampaignRun,
    area_patch,
    evaluate,
    get_entry,
    parse_solver_log,
    reset_nonuniform,
    resolve,
    set_entry,
    worker_commands,
)

U_FILE = """FoamFile { version 2.0; format ascii; class volVectorField; object U; }
dimensions [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);   // at rest
boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (0 1 0);
    }
    /* walls */
    wall { type noSlip; }
    "(top|side)" { type slip; }
}
"""


def test_get_and_set_nested_entries():
    assert get_entry(U_FILE, "boundaryField/inlet/value") == "uniform (0 1 0)"
    t = set_entry(U_FILE, "boundaryField/inlet/value", "uniform (0 9.85 1.74)")
    assert get_entry(t, "boundaryField/inlet/value") == "uniform (0 9.85 1.74)"
    assert get_entry(t, "boundaryField/inlet/type") == "fixedValue"
    assert get_entry(t, "internalField") == "uniform (0 0 0)"
    assert get_entry(t, "boundaryField/(top|side)/type") == "slip"
    t = set_entry(t, "boundaryField/outlet/type", "zeroGradient")          # created
    assert get_entry(t, "boundaryField/outlet/type") == "zeroGradient"
    assert get_entry(t, "boundaryField/wall/type") == "noSlip"
    t = set_entry("application interFoam;\nendTime 30;\nwriteInterval 0.1;\n", "endTime", "0.5")
    assert re.search(r"endTime 0.5;", t) and "writeInterval 0.1;" in t
    fv = "divSchemes\n{\n    default none;\n    div(rhoPhi,U) Gauss linearUpwind grad(U);\n}\n"
    assert get_entry(fv, "divSchemes/div(rhoPhi,U)") == "Gauss linearUpwind grad(U)"
    with pytest.raises(KeyError):
        get_entry(U_FILE, "boundaryField/missing/value")


def test_expressions_are_arithmetic_only():
    names = resolve({"U": 10.0, "theta": 10.0, "d": 87}, {"Uz": "U*sin(radians(theta))", "A": "pi*(d/1000)**2/4"})
    assert names["Uz"] == pytest.approx(1.7365, abs=1e-4) and names["A"] == pytest.approx(0.005945, abs=1e-6)
    assert evaluate("[1, U/2, -1]", {"U": 4}) == [1, 2.0, -1]
    for bad in ("__import__('os')", "U.__class__", "open('x')", "(lambda: 1)()"):
        with pytest.raises((ValueError, NameError)):
            evaluate(bad, {"U": 1.0})
    assert f"uniform {_V([0, 9.85, 1.74])}" == "uniform (0 9.85 1.74)"
    assert f"{_V(0.123456):.3f}" == "0.123" and f"{_V(4)}" == "4"


def _plan(tmp_path, n=5):
    runs = [CampaignRun(f"R{i}", {"U": float(i)}, worker=str(1 + i % 2)) for i in range(n)]
    return Campaign(base=str(tmp_path), runs=runs, results=str(tmp_path / "res"), root=str(tmp_path))


def test_shards_workers_claims_and_plan_file(tmp_path):
    c = _plan(tmp_path)
    assert [r.id for r in c.pending(shard=(1, 2))] == ["R0", "R2", "R4"]
    assert [r.id for r in c.pending(shard=(2, 2))] == ["R1", "R3"]
    assert [r.id for r in c.pending(worker="2")] == ["R1", "R3"]
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "R0.json").write_text("{}")
    assert [r.id for r in c.pending()] == ["R1", "R2", "R3", "R4"]                 # done runs are skipped
    lock = c._claim("R1", force=False)
    assert lock is not None and c._claim("R1", force=False) is None             # a second worker backs off
    assert c._claim("R1", force=True) is not None
    c.to_json(tmp_path / "plan.json")
    c2 = Campaign.from_json(tmp_path / "plan.json", procs=8)
    assert c2.procs == 8 and c2.runs[3].params == {"U": 3.0} and c2.root == str(tmp_path)
    cmds = worker_commands("plan.json", 3, procs=4)
    assert len(cmds) == 3 and "--shard 3/3" in cmds[2] and "--publish-git" in cmds[0]


def test_solver_log_health():
    log = ("Courant Number mean: 0.01 max: 0.4\nTime = 0.001\n\nCourant Number mean: 0.02 max: 0.9\n"
           "Interface Courant Number mean: 0 max: 0.3\nPhase-1 volume fraction = 0.25  Min(alpha.metal) = -1e-6  "
           "Max(alpha.metal) = 1.0002\nTime = 0.002\n\nPhase-1 volume fraction = 0.2501  Min(alpha.metal) = 0  "
           "Max(alpha.metal) = 1\nEnd\n")
    h = parse_solver_log(log, every=1)
    assert h["last_time"] == 0.002 and h["steps"] == 2 and h["courant_max"] == 0.9 and h["ended"]
    assert h["bounds"]["alpha.metal"] == [-1e-6, 1.0002] and h["phase_volume"] == [(0.001, 0.25), (0.002, 0.2501)]
    assert h["fatal"] is None


def test_reset_nonuniform_after_mesh_swap(tmp_path):
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "alpha").write_text("internalField   nonuniform List<scalar> \n5\n(\n0\n0\n1\n0\n1\n)\n;\n"
                                          "boundaryField { }\n")
    (tmp_path / "0" / "k").write_text("internalField   uniform 0.01;\n")
    assert reset_nonuniform(tmp_path, 7) == ["alpha"]
    assert "internalField   uniform 0.0;" in (tmp_path / "0" / "alpha").read_text()
    assert reset_nonuniform(tmp_path, 7) == []


def _foam():
    from pinneapple_simulation.numerical_solvers.external_flow import openfoam_available

    if not openfoam_available():
        pytest.skip("OpenFOAM not available")


def _base_case(tmp_path):
    from pinneapple_design.geometry.trough import TroughSection
    from pinneapple_simulation.numerical_solvers.free_surface import FreeSurfaceFlow, Phase

    sec = TroughSection(length=4.0, inlet_x=0.8, slope_deg=0.0, cell=0.05)
    f = FreeSurfaceFlow(mesh=sec.write_mesh(str(tmp_path / "mesh")), two_d=True,
                        roles={"inlet": "inlet", "outlet": "outlet", "atmosphere": "atmosphere"},
                        inlet_velocity={"inlet": sec.to_mesh((1.0, 0.0, -3.0))}, level=sec.level(),
                        gravity=sec.gravity(9.8), liquid=Phase("metal", 7200.0, 8.472e-6), sigma=1.8,
                        end_time=1.0, sample_every=0.1)
    base = tmp_path / "base"
    f.write(str(base), procs=2)
    return base


def test_area_patch_carves_part_of_an_opening(tmp_path):
    _foam()
    from pinneapple_simulation.external_solvers.openfoam.campaign import _mesh

    base = _base_case(tmp_path)
    pts, faces, pat = _mesh(base / "constant" / "polyMesh")
    nf, sf = pat["inlet"]
    fc = np.array([pts[faces[i]].mean(0) for i in range(sf, sf + nf)])
    full = sum(np.linalg.norm(np.cross(pts[faces[i]][2] - pts[faces[i]][0], pts[faces[i]][3] - pts[faces[i]][1])) / 2
               for i in range(sf, sf + nf))
    p = pts[faces[sf]]
    nrm = np.cross(p[1] - p[0], p[2] - p[1])
    info = area_patch(base, "inlet", full / 2, fc.mean(0), nrm / np.linalg.norm(nrm), rest="floor")
    assert info["faces"] < nf and info["area_patch"] == pytest.approx(full / 2, rel=0.35)
    pat2 = _mesh(base / "constant" / "polyMesh")[2]
    assert pat2["inlet"][0] == info["faces"] and pat2["floor"][0] == pat["floor"][0] + nf - info["faces"]


def test_campaign_runs_two_cases_in_parallel(tmp_path):
    _foam()
    _base_case(tmp_path)
    plan = {"name": "trough jet", "base": "base", "procs": 2, "end_time": 0.2, "pre": ["setFields"],
            "derived": {"Ux": "U/3", "Uz": "-U"},
            "edits": [{"file": "0/U", "key": "boundaryField/inlet/value", "value": "uniform ({Ux} 0 {Uz})"}],
            "sample": {"fields": ["k", "alpha.metal"], "keep_every": 0.2,
                       "lines": [{"name": "floor", "start": [0.0, 0.025, 0.05], "end": [4.0, 0.025, 0.05], "n": 41}]},
            "results": "res", "workdir": str(tmp_path / "work"),
            "runs": [{"id": "slow", "params": {"U": 1.5}}, {"id": "fast", "params": {"U": 3.0}}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    c = Campaign.from_json(tmp_path / "plan.json")
    staged = c.stage(c.runs[1], tmp_path / "staged")
    assert get_entry((tmp_path / "staged" / "0" / "U").read_text(), "boundaryField/inlet/value") == "uniform (1.0 0 -3.0)"
    assert get_entry((tmp_path / "staged" / "system" / "controlDict").read_text(), "endTime") == "0.2"
    assert staged["decompose"] in ("scotch", "simple")
    out = c.run(log=lambda s: None)
    assert [r["id"] for r in out] == ["slow", "fast"] and all(r["rc"] == 0 for r in out)
    for r in out:
        assert r["health"]["ended"] and r["health"]["last_time"] == pytest.approx(0.2)
        assert r["lines"]["floor"]["times"] == pytest.approx([0.1, 0.2])
        assert min(r["lines"]["floor"]["mean_alpha.metal"]) > 0.99
    assert json.loads((tmp_path / "res" / "fast.json").read_text())["params"] == {"U": 3.0}
    assert not os.path.exists(tmp_path / "work" / "trough_jet" / "fast")      # cases are removed when done
    assert c.run(log=lambda s: None) == []                                     # nothing pending
