"""apps/ and examples/ only compose library calls: no new local helper modules, no new copies of library helpers.

``scripts/app_helpers_baseline.txt`` lists what existed when the check was added. The check fails for any NEW
finding, and for a baseline line that no longer occurs (delete it), so the debt can only shrink.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_app_helpers as chk  # noqa: E402


def test_no_new_local_helpers_in_apps_or_examples():
    found = chk.scan()
    new = sorted(k for k in found if k not in chk.load_baseline())
    assert not new, ("reusable logic goes in the library; apps and examples only compose library calls:\n"
                     + "\n".join(found[k] for k in new))


def test_baseline_only_lists_findings_that_still_exist():
    stale = sorted(chk.load_baseline() - set(chk.scan()))
    assert not stale, "fixed: remove from scripts/app_helpers_baseline.txt:\n" + "\n".join(stale)


def test_detects_a_local_helper_module_and_a_duplicate(tmp_path, monkeypatch):
    ex = tmp_path / "examples" / "demo"
    ex.mkdir(parents=True)
    (ex / "helpers.py").write_text("def rel_l2(a, b):\n    return 0.0\n")
    (ex / "run.py").write_text("from helpers import rel_l2\nprint(rel_l2(1, 1))\n")
    monkeypatch.setattr(chk, "ROOT", tmp_path)
    keys = set(chk.scan())
    assert "examples/demo/run.py -> examples/demo/helpers.py" in keys
    assert "examples/demo/helpers.py::rel_l2" in keys


def test_app_server_package_and_appkit_are_glue(tmp_path, monkeypatch):
    pkg = tmp_path / "apps" / "demo" / "demo"
    pkg.mkdir(parents=True)
    (pkg / "schemas.py").write_text("X = 1\n")
    (pkg / "api.py").write_text("from .schemas import X\nfrom appkit import install\n")
    (tmp_path / "apps" / "_shared").mkdir()
    (tmp_path / "apps" / "_shared" / "appkit.py").write_text("def install(app, prefix):\n    pass\n")
    monkeypatch.setattr(chk, "ROOT", tmp_path)
    assert chk.scan() == {}
