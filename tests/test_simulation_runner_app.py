"""apps/simulation_runner: HTTP smoke test (templates, a real run, skip-done on rerun, dataset, Slurm dry run,
public-mode restrictions)."""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("RUNNER_SELF_HOSTED", raising=False)
    monkeypatch.setenv("RUNNER_FREE_JOBS_RUN", "5")
    sys.path.insert(0, str(ROOT / "apps" / "simulation_runner"))
    sys.modules.pop("simulation_runner.api", None)
    api = importlib.import_module("simulation_runner.api")
    api.rt.jobs.poll_interval = 0.02
    yield TestClient(api.app), api
    api.rt.jobs.stop()
    sys.path.remove(str(ROOT / "apps" / "simulation_runner"))


def _wait(c, run_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/runs/{run_id}").json()
        if r["job"] and r["job"]["status"] in ("done", "failed", "cancelled"):
            return r
        time.sleep(0.1)
    raise TimeoutError(run_id)


def test_example_run_rerun_dataset(client):
    c, api = client
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/").status_code == 200
    meta = c.get("/api/meta").json()
    tpl = {t["id"]: t for t in meta["templates"]}
    assert tpl["slab_heat"]["available"] and len(tpl["slab_heat"]["example_cases"]) == 6
    assert not meta["self_hosted"] and {e["name"]: e["available"] for e in meta["executors"]}["docker"] is False

    start = c.post("/api/runs", json={"template": "slab_heat", "max_parallel": 2}).json()
    run = _wait(c, start["run_id"])
    assert run["status"] == "done" and run["counts"]["done"] == 7 and run["compute_seconds"] > 0
    step = next(s for s in run["steps"] if s["step_id"] == "c01.solve")
    assert 300 < step["outputs"]["T_mid"] < 500
    assert "done in" in c.get(f"/api/runs/{start['run_id']}/steps/c01.solve/log").text
    assert c.get(f"/api/runs/{start['run_id']}/steps/c01.solve/files/profile.csv").text.startswith("x_m,T_K")
    assert c.get(f"/api/runs/{start['run_id']}/steps/c01.solve/files/..%2F..%2Frunner.sqlite").status_code == 404

    again = _wait(c, c.post(f"/api/runs/{start['run_id']}/rerun").json()["run_id"])
    assert again["counts"]["skipped"] == 7 and again["counts"]["done"] == 0            # skip-done

    d = c.get(f"/api/runs/{start['run_id']}/dataset-spec?format=json").json()
    assert d["spec"]["kind"] == "DatasetSpec" and d["spec"]["status"] == "complete" and len(d["preview"]) == 6
    assert {"T_mid", "T_mean"} <= {col["name"] for col in d["spec"]["columns"] if col["role"] == "output"}
    pq = c.get(f"/api/runs/{start['run_id']}/dataset?format=parquet")
    assert pq.status_code == 200 and pq.content[:4] == b"PAR1"
    assert c.get("/api/usage").json()["units"]["jobs_run"]["used"] == 2


def test_uploaded_cases_slurm_dry_run_and_public_limits(client):
    c, api = client
    csv = "case_id,L,alpha,T_left,t_end\nA,0.05,1e-5,400,60\nB,0.02,4e-6,350,120\n"
    parsed = c.post("/api/cases/parse?template=slab_heat",
                    files={"file": ("cases.csv", csv.encode(), "text/csv")}).json()
    assert parsed["rows"] == 2 and parsed["id_column"] == "case_id" and parsed["problems"] == []
    bad = csv.replace("0.05", "5.0")
    assert "outside" in c.post("/api/cases/parse?template=slab_heat",
                               files={"file": ("c.csv", bad.encode(), "text/csv")}).json()["problems"][0]
    r = c.post("/api/runs", json={"template": "slab_heat", "cases": parsed["cases"], "id_column": "case_id"})
    run = _wait(c, r.json()["run_id"])
    assert {s["step_id"] for s in run["steps"]} == {"A.solve", "B.solve", "summary"}

    beam = c.post("/api/runs", json={"template": "beam_calculix", "executor": "slurm",
                                     "executor_options": {"partition": "cpu", "account": "ignored"}})
    run = _wait(c, beam.json()["run_id"])
    solve = next(s for s in run["steps"] if s["step_id"].endswith(".solve"))
    assert solve["outputs"] == {"dry_run": True}
    script = c.get(f"/api/runs/{run['id']}/steps/{solve['step_id']}/files/step.sbatch").text
    assert "#SBATCH --partition=cpu" in script and "ccx -i job" in script and "account" not in script

    if not api.TEMPLATES["beam_calculix"].availability()[0]:
        assert c.post("/api/runs", json={"template": "beam_calculix"}).status_code == 409   # needs solver
    assert c.post("/api/runs", json={"workflow": {"name": "x", "steps": []}}).status_code == 403
    assert c.post("/api/runs", json={"template": "slab_heat", "executor": "docker"}).status_code == 403
    assert c.post("/api/runs", json={"template": "slab_heat",
                                     "cases": [{"L": "0.05; rm -rf /", "alpha": 1e-5, "T_left": 400,
                                                "t_end": 60}]}).status_code == 422
    for _ in range(3):                                   # 2 runs above + 3 = the free 5
        c.post("/api/runs", json={"template": "slab_heat"})
    assert c.post("/api/runs", json={"template": "slab_heat"}).status_code == 429         # free limit (5)


def test_runs_are_private_to_their_owner(client):
    c, api = client
    run_id = c.post("/api/runs", json={"template": "slab_heat"}).json()["run_id"]
    other = {"X-Forwarded-For": "203.0.113.9"}
    assert c.get(f"/api/runs/{run_id}", headers=other).status_code == 404
    assert c.get("/api/runs", headers=other).json() == []
