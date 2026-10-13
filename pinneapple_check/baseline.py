"""Baseline: adopt the checker on a repository that already has violations, and fail only on new ones.

``pinneapple check --write-baseline`` records the fingerprints of today's violations (rule, target, subject; not
the values, so a violation that gets worse stays baselined, one that appears elsewhere does not). Later runs
report baselined violations separately and do not fail on them. Delete entries as they are fixed."""
from __future__ import annotations

import json
from pathlib import Path

from .result import Report

__all__ = ["write_baseline", "apply_baseline", "load_baseline"]


def load_baseline(path: Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {v["fingerprint"] for v in data.get("violations", [])}


def write_baseline(report: Report, path: Path) -> int:
    rows = sorted({(o.rule, o.target, o.subject, o.fingerprint) for o in report.outcomes if not o.passed})
    data = {"tool": "pinneapple check", "version": 1,
            "violations": [{"rule": r, "target": t, "subject": s, "fingerprint": f} for r, t, s, f in rows]}
    Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return len(rows)


def apply_baseline(report: Report, path: Path) -> None:
    known = load_baseline(path)
    for o in report.outcomes:
        if not o.passed and o.fingerprint in known:
            o.baselined = True
