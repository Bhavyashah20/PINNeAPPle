"""Physics Validation Lab app (apps/validation_lab): HTTP smoke test of every route and of the examples."""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("meshio")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLATE = ROOT / "examples" / "physics_check" / "pass" / "fields" / "heat_plate.npz"


@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, os.path.join(ROOT, "apps", "validation_lab"))
    from validation_lab.api import app
    return TestClient(app)


def test_health_meta_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert set(m["categories"]) == {"verification", "validation", "physics"}
    assert {e["id"] for e in m["examples"]} == {"pass", "fail", "plate", "zoo"}
    assert "Physics Validation Lab" in client.get("/").text
    assert client.get("/docs").status_code == 200


@pytest.mark.parametrize("eid,decision", [("pass", "pass"), ("fail", "fail"), ("plate", "pass"), ("zoo", "fail")])
def test_examples(client, eid, decision):
    r = client.get(f"/api/example/{eid}").json()
    assert r["decision"] == decision
    assert len(r["ledger"]["entries"]) == r["summary"]["checks"]
    if eid == "zoo":                                     # 20 m/s air is outside the model's 0.5-6 m/s
        bad = [o for o in r["outcomes"] if not o["passed"]]
        assert [o["rule"] for o in bad] == ["PC601"] and bad[0]["category"] == "validation"
    page = client.post("/api/report.html", json=r)
    assert page.status_code == 200 and "Physical consistency" in page.text


def test_upload_with_spec_config_and_zip(client):
    spec = {"results": [{"path": "heat_plate.npz", "physics": "heat", "units": {"T": "K"},
                         "parameters": {"k": "10 W/(m*K)", "q": "1000 W/m3"}}]}
    r = client.post("/api/validate", files=[("files", ("heat_plate.npz", PLATE.read_bytes()))],
                    data={"spec": json.dumps(spec)}).json()
    assert r["decision"] == "pass" and {"PC301", "PC103", "PC501"} <= {o["rule"] for o in r["outcomes"]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("case/heat_plate.npz", PLATE.read_bytes())
        z.writestr("case/pinneapple-check.toml", '[[result]]\npath = "heat_plate.npz"\nbounds = { T = [0, 1] }\n')
    r = client.post("/api/validate", files=[("files", ("case.zip", buf.getvalue()))]).json()
    assert r["decision"] == "fail" and r["categories"]["physics"]["errors"] == 1
    toml = client.post("/api/config", json=spec)
    assert toml.status_code == 200 and 'physics = "heat"' in toml.text


def test_uploaded_code_and_bad_input_are_refused(client):
    cfg = '[[model]]\npath = "m.py:predict"\ninputs = ["x"]\noutputs = ["y"]\n'
    r = client.post("/api/validate", files=[("files", ("m.py", b"def predict(x):\n    return x\n"))],
                    data={"config": cfg})
    assert r.status_code == 422 and "zoo models" in r.json()["detail"]
    assert client.post("/api/validate", data={"config": ""}).status_code == 422
    assert client.post("/api/validate", files=[("files", ("x.npz", PLATE.read_bytes()))]).status_code == 422
    assert client.get("/api/example/nope").status_code == 404
