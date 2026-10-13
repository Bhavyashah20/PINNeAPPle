"""Tests for pinneapple_tools.rea_bridge (REA integration).

Pure-Python parts run everywhere; the end-to-end test runs only when an REA
executable resolves (``npm i -g rea-agents`` or ``PINNEAPPLE_REA_BIN``).
"""
from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from pinneapple_tools.rea_bridge import (
    Finding, HarvestEntry, HarvestLedger, LicenseGateError, ReaClient, ReaError,
    ReaUnavailable, detect_licenses, locate, policy_for, resolve_rea_command,
    summarize_js_evidence)

MIT = "Permission is hereby granted, free of charge, to any person obtaining a copy of this software"
LGPL = "GNU LESSER GENERAL PUBLIC LICENSE\n Version 2.1, February 1999\n ... GNU General Public License"
GPL3 = ("GNU GENERAL PUBLIC LICENSE\n Version 3, 29 June 2007\n" + "x " * 600
        + " 13. Use with the GNU Affero General Public License. ... noncommercially ...")


# ---------------------------------------------------------------- licenses

@pytest.mark.parametrize("text,expected,policy", [
    (MIT, ["MIT"], "adapt"),
    (LGPL, ["LGPL-2.1"], "clean_room"),
    (GPL3, ["GPL-3.0"], "clean_room"),          # AGPL/NC mentions in the body ignored
    ("Copyright 2024 Someone. All rights reserved.", [], "reference_only"),
])
def test_detect_licenses(tmp_path, text, expected, policy):
    (tmp_path / "LICENSE").write_text(text)
    assert detect_licenses(tmp_path) == expected
    assert policy_for(expected).policy == policy


def test_most_restrictive_license_decides(tmp_path):
    (tmp_path / "LICENSE").write_text(MIT)
    (tmp_path / "COPYING").write_text(LGPL)
    pol = policy_for(detect_licenses(tmp_path))
    assert pol.policy == "clean_room" and not pol.may_copy_code


def test_package_json_license_is_read(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"name": "x", "license": "Apache-2.0"}))
    assert detect_licenses(tmp_path) == ["Apache-2.0"]


def test_noncommercial_is_reference_only():
    assert policy_for(["CC-BY-NC"]).policy == "reference_only"


# ------------------------------------------------------------------ ledger

def _finding(status="idea"):
    return Finding(title="t", summary="s", source_refs=["file.py:1-10"], status=status)


def test_ledger_refuses_adapted_code_from_copyleft():
    with pytest.raises(LicenseGateError):
        HarvestEntry(project="p", url="u", commit="c", licenses=["LGPL-2.1"],
                     policy="clean_room", findings=[_finding("adapted")])
    e = HarvestEntry(project="p", url="u", commit="c", licenses=["LGPL-2.1"], policy="clean_room")
    e.add(_finding("reimplemented"))
    with pytest.raises(LicenseGateError):
        e.add(_finding("adapted"))


def test_ledger_roundtrip_reapplies_gate(tmp_path):
    led = HarvestLedger(tmp_path / "ledger.json")
    led.put(HarvestEntry(project="a", url="u", commit="c", licenses=["MIT"], policy="adapt",
                         findings=[_finding("adapted")]))
    path = led.save()
    assert HarvestLedger(path).summary() == {"a": {"policy": "adapt", "findings": {"adapted": 1}}}
    # Hand-editing the JSON to sneak copyleft code in is caught on load.
    data = json.loads(path.read_text())
    data["entries"][0]["policy"] = "clean_room"
    path.write_text(json.dumps(data))
    with pytest.raises(LicenseGateError):
        HarvestLedger(path)


def test_finding_needs_a_source_reference():
    with pytest.raises(ValueError):
        Finding(title="t", summary="s", source_refs=[])


# ---------------------------------------------------------------- evidence

def _fake_evidence(root: Path) -> dict:
    """Minimal payload with the shape of `rea analyze-javascript-application`."""
    def fn(nid, name, module, a, b):
        return {"node_id": nid, "kind": "function", "properties": {"name": name},
                "identity": {"module_path": module,
                             "source_range": {"start": {"line": a}, "end": {"line": b}}}}
    nodes = [fn("f_step", "step", "app.js", 1, 4), fn("f_div", "divergence", "app.js", 6, 8),
             fn("f_vendor", "x", "lib.min.js", 1, 1),
             {"node_id": "cs1", "kind": "call-site", "function_node_id": "f_step",
              "identity": {"module_path": "app.js"}}]
    rels = [{"relation": "calls", "source_node_id": "cs1", "target_node_id": "f_div"}]
    return {"evidence_id": "ev_1", "subject": {"digest": {"sha256": "abc"}},
            "limitations": ["parsed as inert text"],
            "normalized_result": {"input_path": str(root), "limitations": [],
                                  "graph": {"nodes": [{"kind": "storage",
                                                       "identity": {"key": "call:localStorage.setItem"}}]},
                                  "semantic_graph": {"nodes": nodes, "relations": rels}}}


def test_summarize_and_locate(tmp_path):
    (tmp_path / "app.js").write_text(textwrap.dedent("""\
        function step(dt) {
          const d = divergence(v);
          solvePressure(d);
        }

        function divergence(v) {
          return 0.5 * (R - L + T - B);
        }
        """))
    fmap = summarize_js_evidence(_fake_evidence(tmp_path))
    assert fmap.vendor_modules == ["lib.min.js"]
    assert [f.name for f in fmap.named_functions()] == ["step", "divergence"]
    assert fmap.calls == {"f_step": ["f_div"]}
    assert fmap.storage == ["call:localStorage.setItem"]
    hits = locate(fmap, ["pressure"])
    assert [h.function.name for h in hits] == ["step"]
    assert hits[0].callees == ["app.js:6-8 divergence"]
    assert "solvePressure" in hits[0].snippet


# ------------------------------------------------------------------ client

def _stub(tmp_path: Path, body: str) -> str:
    p = tmp_path / "rea_stub.py"
    p.write_text(f"#!{sys.executable}\nimport sys, json\n{body}\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_client_parses_json_and_progress(tmp_path):
    exe = _stub(tmp_path, textwrap.dedent("""\
        sys.stderr.write(json.dumps({"rea_progress": {"phase": "x", "sequence": 1}}) + "\\n")
        print(json.dumps({"evidence_id": "ev_9", "subject": {"digest": {"sha256": "ff"}},
                          "limitations": ["a"], "normalized_result": {"limitations": ["b"]},
                          "argv": sys.argv[1:]}))"""))
    res = ReaClient([exe]).analyze_javascript_application(tmp_path)
    assert res.evidence_id == "ev_9" and res.subject_sha256 == "ff"
    assert res.limitations == ["a", "b"]
    assert res.progress == [{"phase": "x", "sequence": 1}]
    assert res.payload["argv"][:2] == ["analyze-javascript-application", str(tmp_path.resolve())]
    assert res.payload["argv"][-2:] == ["--format", "json"]


def test_client_reports_failures(tmp_path):
    bad = _stub(tmp_path, "sys.stderr.write('boom'); sys.exit(3)")
    with pytest.raises(ReaError) as ei:
        ReaClient([bad]).doctor()
    assert ei.value.returncode == 3 and "boom" in ei.value.stderr_tail


def test_resolution_honours_env_and_reports_absence(tmp_path):
    exe = _stub(tmp_path, "print('6.4.0')")
    assert resolve_rea_command({"PINNEAPPLE_REA_BIN": exe}) == [exe]
    with pytest.raises(ReaUnavailable):
        resolve_rea_command({"PINNEAPPLE_REA_BIN": str(tmp_path / "missing")})


# ------------------------------------------------------- end-to-end (real REA)

def _rea_available() -> bool:
    # Only an installed REA; never let a test trigger an `npx` download.
    import shutil
    return bool(os.environ.get("PINNEAPPLE_REA_BIN") or shutil.which("rea"))


@pytest.mark.skipif(not _rea_available(), reason="REA executable not installed")
def test_real_rea_on_a_small_javascript_app(tmp_path):
    (tmp_path / "index.html").write_text('<script src="sim.js"></script>')
    (tmp_path / "sim.js").write_text(textwrap.dedent("""\
        function project(v) { return subtractGradient(v, solvePressure(divergence(v))); }
        function divergence(v) { return v; }
        function solvePressure(d) { return d; }
        function subtractGradient(v, p) { return v; }
        project(1);
        """))
    res = ReaClient().analyze_javascript_application(tmp_path)
    fmap = summarize_js_evidence(res.payload)
    names = {f.name for f in fmap.named_functions()}
    assert {"project", "divergence", "solvePressure", "subtractGradient"} <= names
    hits = locate(fmap, ["solvePressure"])
    assert hits and hits[0].function.name in {"project", "solvePressure"}


def test_lgpl_project_shipping_gpl_text_is_lgpl(tmp_path):
    """LGPL projects ship COPYING (GPL text) + COPYING.LESSER: the LGPL governs."""
    (tmp_path / "COPYING").write_text("GNU GENERAL PUBLIC LICENSE\n Version 3, 29 June 2007")
    (tmp_path / "COPYING.LESSER").write_text("GNU LESSER GENERAL PUBLIC LICENSE\n Version 3, 29 June 2007")
    assert detect_licenses(tmp_path) == ["LGPL-3.0"]


def test_registry_exposes_license_tool_always(tmp_path, monkeypatch):
    """The license tool needs no REA; REA tools appear only when REA resolves."""
    from pinneapple_worldmodel.physics_tools import PhysicsToolRegistry
    monkeypatch.setenv("PINNEAPPLE_REA_BIN", str(tmp_path / "missing"))
    reg = PhysicsToolRegistry()
    reg._register_research_tools()
    assert [t.name for t in reg.list_by_category("research")] == ["reference_license_policy"]
    (tmp_path / "LICENSE").write_text(MIT)
    assert reg.get("reference_license_policy").call(path=str(tmp_path))["policy"] == "adapt"


def test_gpl_text_file_and_setup_py_declaration(tmp_path):
    """JAX-Fluids ships gpl-3.0.txt; Dedalus declares license="GPL3" in setup.py."""
    (tmp_path / "gpl-3.0.txt").write_text("GNU GENERAL PUBLIC LICENSE\n Version 3, 29 June 2007")
    assert policy_for(detect_licenses(tmp_path)).policy == "clean_room"
    other = tmp_path / "other"
    other.mkdir()
    (other / "setup.py").write_text('setup(name="x", license="GPL3")')
    assert detect_licenses(other) == ["GPL-3.0"]


def test_gnu_notice_form_beats_bundled_permissive_texts(tmp_path):
    """Firedrake's LICENSE opens with the LGPL notice and later quotes BSD
    texts of bundled components: the result must be LGPL (clean_room)."""
    (tmp_path / "LICENSE").write_text(
        "Firedrake:\nFiredrake is free software: you can redistribute it and/or modify it under\n"
        "the terms of the GNU Lesser General Public License as published by the\n"
        "Free Software Foundation, either version 3 of the License, or (at your\noption) any later version.\n\n"
        "Bundled component:\nRedistributions in binary form must reproduce the above copyright notice\n")
    ids = detect_licenses(tmp_path)
    assert "LGPL-3.0" in ids and policy_for(ids).policy == "clean_room"
