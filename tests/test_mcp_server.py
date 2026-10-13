"""The PINNeAPPle MCP server: tool functions, limits, and the server driven by an in-process MCP client."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pinneapple_mcp import tools
from pinneapple_mcp.config import Config, set_config
from pinneapple_mcp.expressions import compile_expression

PLATE = {"name": "plate", "domain": {"x": [0, 1], "y": [0, 1]}, "fields": ["T"], "pde": "laplace",
         "bcs": [{"kind": "dirichlet", "field": "T", "on": "y_max", "value": "sin(pi*x)"},
                 {"kind": "dirichlet", "field": "T", "on": "y_min", "value": 0},
                 {"kind": "dirichlet", "field": "T", "on": "x_boundary", "value": 0}],
         "exact": "sin(pi*x)*sinh(pi*y)/sinh(pi)"}


@pytest.fixture()
def cfg(tmp_path):
    c = set_config(Config(workdir=tmp_path / "work", max_seconds=240, max_jobs=2, roots=[tmp_path]))
    yield c.ensure()
    set_config(Config(workdir=tmp_path / "work2"))


def _plate_grid(n=33, scale=1.0):
    x = y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return {"x": x.tolist(), "y": y.tolist()}, scale * np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)


# ---------------------------------------------------------------------- expressions
def test_expressions_evaluate_and_refuse_code():
    f = compile_expression("sin(pi*x)*exp(-y) + 2", ["x", "y"])
    X = np.array([[0.5, 0.0], [0.25, 1.0]])
    assert np.allclose(f(X), np.sin(np.pi * X[:, 0]) * np.exp(-X[:, 1]) + 2)
    for bad in ("__import__('os').system('true')", "x.__class__", "open('f')", "z + 1", "[x]", "lambda: 1"):
        with pytest.raises((ValueError, SyntaxError)):
            compile_expression(bad, ["x", "y"])


# ---------------------------------------------------------------------- trust card
def test_trust_card_inline_good_and_bad(cfg):
    grid, T = _plate_grid()
    edges = {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": "sin(pi*x)"}
    good = tools.trust_card(grid=grid, values=T.tolist(), pde="laplace", dirichlet=edges,
                            reference="sin(pi*x)*sinh(pi*y)/sinh(pi)", train_bounds={"x": [0, 1], "y": [0, 1]})
    assert good["decision"] == "APPROVED", good["reasons"]
    assert Path(good["files"]["markdown"]).is_file()
    names = {c["name"]: c["passed"] for c in good["checks"]}
    assert names["PDE residual"] and names["conservation (net boundary flux = integral of the source)"]
    bad = tools.trust_card(grid=grid, values=(1.2 * T).tolist(), pde="laplace", dirichlet=edges)
    assert bad["decision"] == "REJECT"
    assert any("y_max" in r for r in bad["reasons"])
    noisy = T + 0.05 * np.random.default_rng(0).standard_normal(T.shape)
    assert tools.trust_card(grid=grid, values=noisy.tolist(), pde="laplace")["decision"] == "REJECT"


def test_trust_card_from_file_and_roots(cfg, tmp_path):
    grid, T = _plate_grid()
    f = tmp_path / "pred.npz"
    np.savez(f, T=T, axis_x=np.array(grid["x"]), axis_y=np.array(grid["y"]), ref_T=T)
    card = tools.trust_card(file=str(f), pde="laplace", dirichlet={"y_max": "sin(pi*x)"})
    assert card["decision"] == "APPROVED"
    assert any(c["name"].startswith("reference") and c["passed"] for c in card["checks"])
    outside = Path(__file__).resolve()                   # not under the allowed roots
    with pytest.raises(PermissionError):
        tools.trust_card(file=str(outside), field="x")


def test_catalog_tools(cfg):
    m = tools.list_methods("solver", validated_only=True)
    assert m["catalog"] and all(r["status"] == "validated" for r in m["catalog"])
    assert "pinn" in m["solve_methods"]["neural"]
    assert tools.describe_method("S2")["status"] == "validated"
    d = tools.describe_method("pinn")
    assert "epochs" in d["options"]
    with pytest.raises(KeyError):
        tools.describe_method("no_such_method")
    assert "burgers_1d" in tools.list_presets("burgers")["presets"]
    cap = tools.capabilities()
    assert cap["limits"]["allow_network"] is False and "solve" in cap["tools"]


def test_zoo_tools(cfg):
    pytest.importorskip("pinneapple_hub.zoo")
    z = tools.zoo_list()
    names = {m["name"] for m in z["models"]}
    assert "airfoil_cst_rans_mlp" in names and z["datasets"]
    naca2412 = {"T0": 0.169591, "T1": 0.144959, "T2": 0.131433, "M0": 0.024922, "M1": 0.071088, "M2": 0.05896}
    p = tools.zoo_predict("airfoil_cst_rans_mlp", {**naca2412, "alpha_deg": [4.0, 30.0]})
    assert abs(p["outputs"]["Cl"][0] - 0.6625) < 0.01                    # OpenFOAM: 0.6625
    assert p["inside_validity_range"] == [True, False] and p["n_outside"] == 1
    with pytest.raises(ValueError):
        tools.zoo_predict("airfoil_cst_rans_mlp", {"alpha_deg": [4.0]})
    v = tools.zoo_validate("airfoil_cst_rans_mlp", {**naca2412, "alpha_deg": [4.0]})
    assert v["decision"] == "APPROVED", v["checks"]
    assert tools.zoo_validate("airfoil_cst_rans_mlp", {**naca2412, "alpha_deg": [30.0]})["decision"] != "APPROVED"


# ---------------------------------------------------------------------- jobs
def test_solve_job_fem_with_checks(cfg):
    st = tools.solve(PLATE, "fem", {"n": 24}, grid_points_per_axis=25, wait_seconds=180)
    assert st["status"] == "done", st
    res = st["result"]
    assert res["reference"] == "exact" and res["metrics"]["relative_l2"]["T"] < 0.02
    assert res["decision"] == "APPROVED", res["trust_cards"]
    assert Path(res["files"]["prediction"]).is_file()
    z = np.load(res["files"]["prediction"])
    assert z["T"].shape == (25, 25) and "axis_x" in z.files
    assert tools.job_status(st["job_id"])["status"] == "done"
    assert any(j["job_id"] == st["job_id"] for j in tools.list_jobs()["jobs"])


def test_benchmark_job_and_timeout(cfg):
    st = tools.benchmark(PLATE, ["fem", "pinn"], budget={"epochs": 15, "n_collocation": 128},
                         options={"pinn": {"hidden_dim": 8, "n_layers": 2}}, grid_points_per_axis=17,
                         wait_seconds=200)
    assert st["status"] == "done", st
    rows = {r["method"]: r for r in st["result"]["rows"]}
    assert rows["fem"]["n_ok"] == 1 and rows["pinn"]["n_ok"] == 1
    assert st["result"]["best"] == "fem"
    assert Path(st["result"]["files"]["table"]).read_text().startswith("# Equal-budget benchmark")
    slow = tools.solve(PLATE, "pinn", {"epochs": 20000}, max_seconds=3, wait_seconds=60)
    assert slow["status"] == "timeout", slow


def test_bad_inputs_fail_fast(cfg):
    with pytest.raises(KeyError):
        tools.solve(PLATE, "no_such_method")
    with pytest.raises(ValueError):
        tools.solve({"domain": {"x": [0, 1]}, "pde": "laplace", "bcs": [{"on": "x_min", "value": "os.system"}]})
    with pytest.raises(ValueError):
        tools.job_status("../etc")


def test_jobs_have_no_network():
    code = ("import socket; from pinneapple_mcp.jobs import _no_network; _no_network()\n"
            "s = socket.socket()\n"
            "try:\n    s.connect(('203.0.113.1', 80))\nexcept PermissionError as e:\n    print('blocked', e)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert "blocked" in out.stdout, out.stderr


# ---------------------------------------------------------------------- the MCP server, in process
def _data(result):
    sc = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if sc is not None:
        return sc.get("result", sc) if set(sc) == {"result"} else sc
    return json.loads(result.content[0].text)


def _is_error(result):
    return bool(getattr(result, "is_error", None) or getattr(result, "isError", None))


async def _session_calls(server, calls):
    try:
        from mcp import Client
    except ImportError:                                          # MCP SDK 1.x
        Client = None
    if Client is not None:
        async with Client(server) as client:
            tools_ = await client.list_tools()
            return tools_, [await client.call_tool(n, a) for n, a in calls]
    from mcp.shared.memory import create_connected_server_and_client_session
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        tools_ = await client.list_tools()
        return tools_, [await client.call_tool(n, a) for n, a in calls]


def test_mcp_server_in_process(cfg):
    pytest.importorskip("mcp")
    import anyio

    from pinneapple_mcp.server import build_server

    server = build_server(cfg)
    grid, T = _plate_grid(17)
    calls = [("capabilities", {}),
             ("list_methods", {"category": "solver", "validated_only": True}),
             ("describe_method", {"method": "fem"}),
             ("trust_card", {"grid": grid, "values": T.tolist(), "pde": "laplace",
                             "dirichlet": {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": "sin(pi*x)"}}),
             ("solve", {"problem": PLATE, "method": "fem", "options": {"n": 16}, "grid_points_per_axis": 17,
                        "wait_seconds": 180}),
             ("describe_method", {"method": "nope"}),
             ("list_methods", {"category": "not-a-category"})]
    listed, res = anyio.run(_session_calls, server, calls)
    names = {t.name for t in listed.tools}
    assert {"capabilities", "list_methods", "describe_method", "list_presets", "solve", "trust_card", "benchmark",
            "job_status", "list_jobs", "cancel_job"} <= names
    solve_schema = next(t for t in listed.tools if t.name == "solve")
    schema = getattr(solve_schema, "input_schema", None) or getattr(solve_schema, "inputSchema", None)
    assert "problem" in schema["properties"] and "problem" in schema.get("required", [])
    assert _data(res[0])["limits"]["allow_network"] is False
    assert all(r["status"] == "validated" for r in _data(res[1])["catalog"])
    assert _data(res[2])["kind"] == "classical"
    assert _data(res[3])["decision"] == "APPROVED"
    done = _data(res[4])
    assert done["status"] == "done" and done["result"]["decision"] == "APPROVED"
    assert _is_error(res[5]) and _is_error(res[6])
