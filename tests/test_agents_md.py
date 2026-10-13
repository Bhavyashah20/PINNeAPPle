"""Every Python block of AGENTS.md runs as written (each block on its own, in a scratch directory)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = re.findall(r"```python\n(.*?)```", (ROOT / "AGENTS.md").read_text(), flags=re.S)


def test_agents_md_has_examples():
    assert len(BLOCKS) >= 4
    for name in ("GEMINI.md", "CLAUDE.md"):
        assert "AGENTS.md" in (ROOT / name).read_text()


@pytest.mark.parametrize("i", range(len(BLOCKS)))
def test_agents_md_block_runs(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    code = compile(BLOCKS[i], f"AGENTS.md[python block {i}]", "exec")
    exec(code, {"__name__": f"agents_md_block_{i}"})
    if "to_png(" in BLOCKS[i]:
        assert any(p.suffix == ".png" for p in tmp_path.iterdir())
    os.chdir(ROOT)
