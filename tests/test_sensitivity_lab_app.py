"""Engineering Sensitivity Lab (apps/sensitivity_lab): HTTP smoke test of the API on its examples and uploads."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from pinneapple_analysis.sensitivity.testfunctions import ishigami, ishigami_indices  # noqa: E402

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "sensitivity_lab"))
sys.path.insert(0, APP)
from sensitivity_lab.api import app  # noqa: E402

client = TestClient(app)


def test_health_meta_model_info_and_cost():
    assert client.get("/health").json()["status"] == "ok"
    assert {e["name"] for e in client.get("/api/meta").json()["examples"]} == {"ishigami", "heatsink", "dataset"}
    info = client.get("/api/model/zoo/heatsink_forced_mlp").json()
    assert info["differentiable"] and len(info["space"]) + len(info["fixed"]) == 12
    cfg = client.get("/api/example/ishigami/config").json()
    cost = client.post("/api/cost", json=cfg).json()
    assert cost["parts"]["sobol"] == 2048 * 8 and cost["total"] > cost["parts"]["sobol"]
    assert client.get("/").status_code == 200


def test_ishigami_example_matches_the_exact_indices():
    j = client.get("/api/example/ishigami").json()
    assert j["exact"]["max_abs_error_S1"] < 0.02 and j["exact"]["max_abs_error_ST"] < 0.02
    assert all(j["exact"]["S1_in_ci"]) and all(j["exact"]["ST_in_ci"])
    assert j["sobol"]["ranking"]["y"] == ["x1", "x2", "x3"]
    assert j["usage"]["model_evaluations"] == j["n_evals"]["total"]


def test_heatsink_example_freezes_ambient_and_power_with_a_check():
    j = client.get("/api/example/heatsink").json()
    dec = {r["name"]: r["decision"] for r in j["freeze"]["rows"]}
    assert dec["t_ambient_c"] == dec["power_w"] == "freeze" and dec["n_fins"] == "keep"
    assert all(abs(v - 1) < 0.01 for v in j["freeze"]["check"]["variance_ratio"])
    assert "automatic" in j["derivatives"]["method"]
    assert j["sobol"]["ranking"]["pressure_drop_pa"][0] in ("n_fins", "air_velocity_m_s")


def test_dataset_example_reports_input_correlation():
    j = client.get("/api/example/dataset").json()
    assert j["n_rows"] == 3900 and abs(j["max_input_correlation"]["r"]) > 0.5
    assert set(j["given_data"]) == {"r_base_k_w", "pressure_drop_pa"}


def test_design_then_external_runs_round_trip():
    cfg = client.get("/api/example/ishigami/config").json()
    cfg["n_sobol"], cfg["second_order"] = 512, False
    lines = client.post("/api/design", json=cfg).text.splitlines()
    spec = lines[0].split(" ", 2)[2]
    X = np.array([[float(v) for v in ln.split(",")[1:]] for ln in lines[2:]])
    ids = np.random.default_rng(0).permutation(len(X))           # runs come back in any order
    csv = "run_id,y\n" + "\n".join(f"{i},{ishigami(X[i:i + 1])[0]}" for i in ids)
    r = client.post("/api/runs", data={"spec": spec}, files={"runs": ("results.csv", csv.encode())})
    assert r.status_code == 200, r.text
    from pinneapple_analysis.sensitivity import sobol_indices
    from pinneapple_analysis.sensitivity.testfunctions import ishigami_space
    direct = sobol_indices(ishigami, ishigami_space(), n=512, n_boot=0)        # same design, run here
    np.testing.assert_allclose(r.json()["sobol"]["ST"][0], direct.ST[0], rtol=1e-6)
    short = "run_id,y\n" + "\n".join(csv.splitlines()[1:-5])
    bad = client.post("/api/runs", data={"spec": spec}, files={"runs": ("results.csv", short.encode())})
    assert bad.status_code == 422 and "missing" in bad.json()["detail"]


def test_given_data_upload_and_errors():
    rng = np.random.default_rng(1)
    X = rng.uniform(-np.pi, np.pi, (3000, 3))
    csv = "x1,x2,x3,y\n" + "\n".join(",".join(f"{v:.6f}" for v in list(x) + [ishigami(x[None])[0]]) for x in X)
    r = client.post("/api/given-data", data={"config": json.dumps({"outputs": ["y"], "n_boot": 20})},
                    files={"data": ("d.csv", csv.encode())})
    assert r.status_code == 200, r.text
    np.testing.assert_allclose(r.json()["given_data"]["y"]["S1"], ishigami_indices()["S1"], atol=0.06)
    big = client.get("/api/example/heatsink/config").json()
    big["n_sobol"] = 2 ** 16
    assert client.post("/api/study", data={"config": json.dumps(big)}).status_code == 413
    assert client.post("/api/study", data={"config": json.dumps({"model": {"source": "builtin", "name": "ishigami"}})}
                       ).status_code == 422
    assert client.post("/api/study", data={"config": "{nope"}).status_code == 422
