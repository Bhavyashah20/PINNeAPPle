"""Simulation Campaign Planner app (apps/simulation_campaign): HTTP smoke test of the real job path."""
from __future__ import annotations

import importlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "apps" / "simulation_campaign"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["CAMPAIGN_DATA_DIR"] = str(tmp_path_factory.mktemp("campaign"))
    sys.path.insert(0, str(APP))
    api = importlib.import_module("simulation_campaign.api")
    api.rt.jobs.poll_interval = 0.02
    with TestClient(api.app) as c:
        yield c
    api.rt.jobs.stop()
    os.environ.pop("CAMPAIGN_DATA_DIR", None)


def _wait(c, job, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            assert j["status"] == "done", j.get("error")
            return j
        time.sleep(0.1)
    raise TimeoutError


def test_meta_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert set(m["presets"]) == {"airfoil", "cantilever"} and "infill" in m["methods"]
    assert m["runtime"]["job_kinds"] == ["plan"]
    assert "Simulation Campaign Planner" in client.get("/").text


def test_cantilever_example_writes_calculix_decks(client):
    job = _wait(client, client.post("/api/example/cantilever").json())
    r = job["result"]
    assert r["n_cases"] == 30 and r["files"]["calculix"] and r["files"]["case_dirs"] == 4
    assert all(c["status"] in ("PASS", "WARN") for c in r["checks"])
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/campaigns/{job['id']}/download/campaign.zip").content))
    names = z.namelist()
    assert "campaign/cases/case_001/job.inp" in names and "campaign/dataset.skeleton.json" in names
    skel = json.loads(client.get(f"/api/campaigns/{job['id']}/download/dataset.skeleton.json").content)
    assert skel["kind"] == "DatasetSpec" and skel["status"] == "skeleton"
    assert client.get(f"/api/campaigns/{job['id']}/download/plan.json").status_code == 404   # base-case mode only
    usage = client.get("/api/usage").json()["units"]
    assert usage["campaigns"]["used"] >= 1 and usage["cases_planned"]["used"] >= 30


def test_custom_space_with_uploaded_runs_drops_repeats(client):
    space = {"kind": "ParameterSpace", "name": "plate", "budget": 8, "parameters": [
        {"name": "t", "low": 1.0, "high": 5.0, "unit": "mm"}, {"name": "q", "low": 10.0, "high": 1000.0, "log": True}]}
    first = _wait(client, client.post("/api/campaigns", json={"space": space, "method": "lhs", "seed": 3}).json())
    csv = client.get(f"/api/campaigns/{first['id']}/download/cases.csv").content
    up = client.post("/api/uploads", files={"file": ("done.csv", csv, "text/csv")}).json()
    info = client.post("/api/existing/inspect", json={"upload": up["id"]}).json()
    assert info["rows"] == 8 and "case_id" in info["columns"]
    again = _wait(client, client.post("/api/campaigns", json={
        "space": space, "method": "lhs", "seed": 3, "existing_upload": up["id"], "existing_id_column": "case_id"}).json())
    s = again["result"]["summary"]
    assert s["n_dropped_as_repeats"] == 8 and again["result"]["n_cases"] == 0     # the same design: all repeats
    infill = _wait(client, client.post("/api/campaigns", json={
        "space": space, "method": "infill", "n": 4, "existing_upload": up["id"], "existing_id_column": "case_id"}).json())
    s = infill["result"]["summary"]
    assert infill["result"]["n_cases"] == 4
    assert s["coverage"]["combined"]["fill_distance"] < s["coverage"]["existing"]["fill_distance"]


@pytest.mark.parametrize("body,msg", [
    ({"method": "nope", "preset": "cantilever"}, "method"),
    ({"preset": "custom"}, "parameter space"),
    ({"preset": "cantilever", "openfoam": "airfoil"}, "airfoil preset"),
    ({"preset": "cantilever", "method": "infill"}, "existing runs"),
    ({"preset": "cantilever", "n": 10 ** 6}, "plans up to"),
])
def test_invalid_requests(client, body, msg):
    r = client.post("/api/campaigns", json=body)
    assert r.status_code == 422 and msg in r.text
