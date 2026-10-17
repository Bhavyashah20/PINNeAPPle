"""Engineering Uncertainty Lab (apps/uncertainty_lab): HTTP smoke test of the API on its examples and uploads."""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "uncertainty_lab"))
sys.path.insert(0, APP)
from uncertainty_lab.api import app  # noqa: E402

client = TestClient(app)


def test_health_meta_and_model_info():
    assert client.get("/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert {e["name"] for e in meta["examples"]} == {"heatsink", "airfoil"}
    info = client.get("/api/model/heatsink_forced_mlp").json()
    assert info["differentiable"] and len(info["nominal"]) == 12
    assert client.get("/api/model/nope").status_code == 404
    assert client.get("/").status_code == 200


def test_heatsink_example_measures_coverage_and_checks_the_routing_with_the_engine():
    r = client.get("/api/example/heatsink")
    assert r.status_code == 200, r.text
    j = r.json()
    c = j["conformal"]
    assert c["n_cal"] == 600 and c["n_test"] == 900 and c["meets_target"]
    rows = {x["name"]: x for x in j["cases"]["rows"]}
    assert rows["8 m/s fan (beyond range)"]["status"] == "outside"
    assert rows["8 m/s fan (beyond range)"]["route"] == "high_fidelity"
    assert "straddles_limit" in rows["fan derated to 2.3 m/s"]["codes"]
    assert rows["nominal design"]["route"] == "surrogate" and rows["nominal design"]["verdict"] == "meets limits"
    assert j["cases"]["high_fidelity"]["n_run"] == len(rows)
    assert all(x["high_fidelity"] is not None for x in rows.values())
    p = j["propagation"]
    assert p["n"] == 4096 and 0 <= p["p_within_with_model_error"] <= p["p_within"] <= 1
    assert j["usage"]["high_fidelity_runs"] == len(rows) and j["usage"]["model_evaluations"] > 4096
    assert j["study_result"]["kind"] == "uncertainty_study"


def test_airfoil_example_compares_the_raw_ensemble_interval():
    j = client.get("/api/example/airfoil").json()
    assert j["conformal"]["n_cal"] == 36 and j["conformal"]["n_test"] == 60 and j["scale"] == "ensemble"
    assert j["ensemble"]["coverage"] < j["conformal"]["coverage"]
    assert "mean_abs_pct" not in j["accuracy"]          # Cl crosses zero: no percent errors
    assert j["cases"]["high_fidelity"] is None


def test_onnx_upload_with_reference_table_and_case_file():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from pinneapple_hub import zoo
    from pinneapple_hub.zoo.export import export_bytes
    graph = export_bytes(zoo.load("heatsink_forced_mlp"))
    ref = client.get("/api/template/heatsink_forced_mlp.csv").text
    header = ref.splitlines()[0].split(",")
    names = header[:12]
    cases = "name," + ",".join(names) + "\n" + "upload case," + ",".join(ref.splitlines()[1].split(",")[:12]) + "\n"
    cfg = {"model": {"source": "onnx", "input_names": names, "output_names": ["r_base_k_w", "pressure_drop_pa", "r95"]},
           "output": "r_base_k_w", "alpha": 0.1, "scale": "relative", "limits": [None, 0.5]}
    r = client.post("/api/study", data={"config": json.dumps(cfg)},
                    files={"model": ("hs.onnx", graph), "reference": ("ref.csv", ref.encode()),
                           "cases": ("cases.csv", cases.encode())})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["conformal"]["n_cal"] == 600 and j["conformal"]["n_test"] == 900       # the split column was used
    assert j["model"]["kind"] == "onnx" and j["cases"]["rows"][0]["name"] == "upload case"
    assert abs(j["conformal"]["coverage"] - 0.9) < 0.05


def test_errors_are_reported_not_raised():
    bad = client.post("/api/study", data={"config": "{not json"})
    assert bad.status_code == 422
    r = client.post("/api/study", data={"config": json.dumps({"model": {"source": "zoo", "name": "nope"}})})
    assert r.status_code == 422 and "Zoo model" in r.json()["detail"]
    r = client.post("/api/study", data={"config": json.dumps({"model": {"source": "zoo", "name": "heatsink_forced_mlp"},
                                                             "scale": "ensemble"})})
    assert r.status_code == 422 and "ensemble" in r.json()["detail"]
    r = client.post("/api/study", data={"config": json.dumps({"model": {"source": "onnx"}})})
    assert r.status_code == 422


def test_template_is_a_valid_reference_table():
    import pandas as pd
    df = pd.read_csv(io.StringIO(client.get("/api/template/airfoil_cst_rans_mlp.csv").text))
    assert set(df["split"]) == {"cal", "test"} and np.isfinite(df["Cl"]).all()
