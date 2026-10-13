"""pinneapple check: every rule passes on examples/physics_check/pass and fails exactly where expected on
examples/physics_check/fail; CLI exit codes, selection, output formats, baseline, zero-config discovery."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pinneapple_check import RULES, ConfigError, load_config, run, select_rules
from pinneapple_check.cli import check_main, main
from pinneapple_check.evidence import guardrail_checks
from pinneapple_check.report import render

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "physics_check"
PASS, FAIL, TORCH = EX / "pass", EX / "fail", EX / "torch"

pytest.importorskip("meshio")


@pytest.fixture(scope="module")
def pass_report():
    return run([PASS])


@pytest.fixture(scope="module")
def fail_report():
    return run([FAIL])


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------------------------- fixtures
def test_pass_fixtures_have_no_violation(pass_report):
    assert not pass_report.violations, "\n".join(f"{o.rule} {o.target}: {o.message}" for o in pass_report.violations)
    assert pass_report.exit_code("warning") == 0
    assert len(pass_report.targets) == 13


def test_fail_fixtures_break_exactly_the_expected_rules(fail_report):
    expected = json.loads((FAIL / "expected.json").read_text())
    got: dict[str, set[str]] = {}
    for o in fail_report.violations:
        got.setdefault(o.target, set()).add(o.rule)
    assert {k: sorted(v) for k, v in got.items()} == expected
    assert fail_report.exit_code() == 1


def test_every_rule_has_a_passing_and_a_failing_fixture(pass_report, fail_report):
    passed = {o.rule for o in pass_report.outcomes if o.passed}
    if _torch_ok():
        passed |= {o.rule for o in run([TORCH]).outcomes if o.passed}
    else:
        passed.add("PC803")                       # its passing fixture is a torch checkpoint
    failed = {o.rule for o in fail_report.violations}
    assert set(RULES) - passed == set()
    assert set(RULES) - failed == set()


def test_annotations_point_at_the_broken_line(fail_report):
    bc = next(o for o in fail_report.violations if o.rule == "PC201" and o.target == "cavity")
    assert bc.location.file.endswith("fail/cases/cavity/5/U")
    lines = (ROOT / bc.location.file).read_text().splitlines()
    assert "uniform (2 0 0)" in lines[bc.location.line - 1]          # the `value` entry of the movingWall patch
    assert "movingWall" in lines[bc.location.line - 4]
    q = [o for o in fail_report.violations if o.rule == "PC601"]
    assert {o.location.line for o in q} == {6, 7}                   # the two rows out of range in queries.csv
    assert all(o.location.file.endswith("models/queries.csv") for o in q)


def test_openfoam_checks_use_the_solver_flux(pass_report):
    by = {(o.target, o.rule): o for o in pass_report.outcomes}
    assert by[("cavity", "PC102")].value < 1e-8                      # phi of icoFoam: continuity to round-off
    assert by[("cavity", "PC301")].value < 1e-3                      # Gauss-linear residual matches the solver
    assert "phi" in by[("channel", "PC101")].message
    assert by[("channel", "PC101")].value < 1e-6


@pytest.mark.skipif(not _torch_ok(), reason="torch not installed")
def test_torch_checkpoint_range_from_metadata():
    r = run([TORCH])
    out = [o for o in r.violations]
    assert [(o.target, o.rule) for o in out] == [("plate-pinn-out-of-range", "PC601")]
    assert any(o.rule == "PC803" and o.passed for o in r.outcomes)
    zoo = [o for o in r.outcomes if o.target == "heatsink-zoo"]
    assert any(o.rule == "PC601" and o.passed for o in zoo)          # range from the pp.zoo model card
    assert any(o.rule == "PC401" and o.passed for o in zoo)          # predictions at the queries


def test_unsafe_checkpoint_is_never_loaded(fail_report, monkeypatch):
    import pinneapple_check.models as M
    called = []
    monkeypatch.setattr(M, "_from_checkpoint", lambda *a, **k: called.append(1))
    run([FAIL / "models" / "legacy_model.pt"], config=FAIL / "pinneapple-check.toml", select=["PC8"])
    assert not called


# ---------------------------------------------------------------------------------------------- CLI
def test_exit_codes(capsys):
    assert check_main([str(PASS), "--color", "never"]) == 0
    assert check_main([str(FAIL), "--color", "never"]) == 1
    assert check_main([str(FAIL), "--exit-zero"]) == 0
    models = str(FAIL / "models")
    assert check_main([models, "--select", "PC9"]) == 0            # only a warning: fail-on error
    assert check_main([models, "--select", "PC9", "--fail-on", "warning"]) == 1
    assert check_main([str(FAIL), "--select", "PC999"]) == 2
    assert check_main([str(ROOT / "no-such-dir")]) == 2
    capsys.readouterr()


def test_select_and_ignore_prefixes():
    assert select_rules(["PC1"]) == {r for r in RULES if r.startswith("PC1")} | {"PC001"}
    assert "PC203" not in select_rules(None, ["PC2"])
    assert "PC001" in select_rules(None, ["PC0"])                 # an unreadable target is never ignored
    r = run([FAIL], select=["PC6"])
    assert {o.rule for o in r.violations} == {"PC001", "PC601", "PC602"}   # PC001: the corrupt VTU


def test_paths_restrict_declared_targets():
    r = run([FAIL / "fields" / "porosity.npz"])
    assert [t["name"] for t in r.targets] == ["fields/porosity.npz"]


def test_list_and_explain(capsys):
    assert check_main(["--list-rules"]) == 0
    out = capsys.readouterr().out
    assert all(r in out for r in RULES)
    assert check_main(["--explain", "pc101"]) == 0
    assert "net flux" in capsys.readouterr().out
    assert check_main(["--explain", "PC000"]) == 2


def test_dispatcher_and_module_entry_points(capsys):
    assert main(["check", "--explain", "PC401"]) == 0
    assert main(["nope"]) == 2
    capsys.readouterr()
    for mod in ("pinneapple_check", "pinneapple.check"):
        p = subprocess.run([sys.executable, "-m", mod, "--explain", "PC601"], capture_output=True, text=True,
                           cwd=ROOT)
        assert p.returncode == 0 and "PC601" in p.stdout, p.stderr


def test_import_does_not_pull_torch():
    code = "import sys, pinneapple_check, pinneapple_check.cli; print('torch' in sys.modules)"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert p.stdout.strip() == "False", p.stderr


# ---------------------------------------------------------------------------------------------- formats
def test_sarif(fail_report):
    doc = json.loads(render(fail_report, "sarif"))
    assert doc["version"] == "2.1.0"
    run_ = doc["runs"][0]
    rules = run_["tool"]["driver"]["rules"]
    assert [r["id"] for r in rules] == sorted(RULES)
    assert len(run_["results"]) == len(fail_report.violations)
    for res in run_["results"]:
        assert rules[res["ruleIndex"]]["id"] == res["ruleId"]
        loc = res["locations"][0]["physicalLocation"]
        assert not loc["artifactLocation"]["uri"].startswith("/")
        assert loc["region"]["startLine"] >= 1
        assert res["partialFingerprints"]["pinneappleCheck/v1"]


def test_junit_and_github_and_summary(fail_report):
    root = ET.fromstring(render(fail_report, "junit").split("\n", 1)[1])
    assert int(root.get("failures")) == len(fail_report.violations)
    gh = render(fail_report, "github").splitlines()
    assert len(gh) == len(fail_report.violations)
    assert any(line.startswith("::error file=examples/physics_check/fail/cases/cavity/5/U,line=") for line in gh)
    md = render(fail_report, "summary")
    assert md.startswith("### ❌ Physics check failed") and "| `cavity` |" in md
    js = json.loads(render(fail_report, "json"))
    assert js["summary"]["errors"] == len(fail_report.errors)
    assert js["trust_card"]["cavity"]["physics"]["status"] == "failed"


def test_cli_writes_side_outputs(tmp_path, capsys):
    s = tmp_path / "summary.md"
    rc = check_main([str(FAIL), "--sarif", str(tmp_path / "o.sarif"), "--junit", str(tmp_path / "o.xml"),
                     "--json", str(tmp_path / "o.json"), "--summary", str(s), "--format", "github"])
    assert rc == 1
    assert (tmp_path / "o.sarif").is_file() and (tmp_path / "o.xml").is_file() and (tmp_path / "o.json").is_file()
    assert "Physics check failed" in s.read_text()
    assert capsys.readouterr().out.startswith("::")


def test_veriphysics_evidence(pass_report, fail_report):
    from pinneapple_veriphysics.applicability import build_applicability_map
    ok = guardrail_checks(pass_report, "kovasznay-pinn")
    bad = guardrail_checks(fail_report, "kovasznay-pinn")
    assert {c["name"] for c in ok} >= {"pde_residual", "conservation_divergence", "conservation_mass"}
    assert all(c["passed"] for c in ok)
    assert not next(c for c in bad if c["name"] == "conservation_mass")["passed"]
    m = build_applicability_map(architecture="PINN", coords=["x", "y"], fields=["u", "v", "p"],
                                domain_bounds={"x": (-0.5, 1.0), "y": (-0.5, 1.5)}, physical_parameters={},
                                guardrail_checks=bad, confidence_components=[])
    assert "pde_residual" in json.dumps(m.to_dict())


# ---------------------------------------------------------------------------------------------- baseline, config
def test_baseline_adopts_existing_violations(tmp_path, capsys):
    work = tmp_path / "fail"
    shutil.copytree(FAIL, work, ignore=shutil.ignore_patterns("__pycache__"))
    b = work / "baseline.json"
    assert check_main([str(work), "--write-baseline", "--baseline", str(b)]) == 0
    assert check_main([str(work), "--baseline", str(b)]) == 0
    r = run([work], baseline=b)
    assert not r.violations and len(r.baselined) > 20
    # a new violation is not covered by the baseline
    (work / "fields" / "surrogate_T.npz").unlink()
    r = run([work], baseline=b)
    assert [(o.target, o.rule) for o in r.violations] == [("fields/surrogate_T.npz", "PC001")]
    capsys.readouterr()


def test_zero_config_discovery(tmp_path):
    shutil.copytree(PASS / "cases" / "cavity", tmp_path / "cavity")
    (tmp_path / ".git").mkdir()                                   # the repository root ends the config search
    r = run([tmp_path])
    assert [t["kind"] for t in r.targets] == ["fv"]
    rules = {o.rule for o in r.outcomes}
    assert {"PC101", "PC102", "PC403"} <= rules                 # incompressible flow inferred from U and p
    assert [o.rule for o in r.violations] == ["PC203"]           # the moving lid is not declared
    assert r.violations[0].location.file == "cavity/5/U"


def test_pyproject_table_and_config_errors(tmp_path):
    (tmp_path / ".git").mkdir()
    shutil.copy(PASS / "fields" / "porosity.npz", tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n\n[tool.pinneapple.check]\nignore = ["PC2"]\n\n'
        '[[tool.pinneapple.check.result]]\npath = "porosity.npz"\nbounds = { porosity = [0.2, 0.3] }\n')
    r = run([tmp_path])
    assert [o.rule for o in r.violations] == ["PC401"]
    (tmp_path / "pinneapple-check.toml").write_text("[check]\nselectt = ['PC1']\n")
    with pytest.raises(ConfigError, match="selectt"):
        load_config(start=tmp_path)
    (tmp_path / "pinneapple-check.toml").write_text("[[result]]\npath = 'a.npz'\nphysics = 'plasma'\n")
    with pytest.raises(ConfigError, match="plasma"):
        load_config(start=tmp_path)
    (tmp_path / "pinneapple-check.toml").write_text("[[result]]\npath = 'a.npz'\ntolerances = { mas = 1 }\n")
    with pytest.raises(ConfigError, match="mas"):
        load_config(start=tmp_path)


def test_action_and_workflow_are_consistent():
    import yaml
    action = yaml.safe_load((ROOT / "action.yml").read_text())
    assert action["runs"]["using"] == "composite"
    assert {"paths", "version", "sarif", "fail-on"} <= set(action["inputs"])
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "physics-check.yml").read_text())
    steps = [s for j in wf["jobs"].values() for s in j["steps"]]
    assert any(str(s.get("uses", "")).startswith("./") for s in steps)
    example = (ROOT / "examples" / "physics_check" / "workflow.yml").read_text()
    assert "PINNeAPPle-Labs/PINNeAPPle@" in example and "security-events: write" in example
    assert os.path.isfile(ROOT / "docs" / "guides" / "physics_checks_in_ci.md")
