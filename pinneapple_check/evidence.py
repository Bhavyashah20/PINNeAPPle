"""The checks as evidence: per-target trust card (the six stages of :mod:`pinneapple_lab.curation`) and the
guardrail checks that :func:`pinneapple_veriphysics.applicability.build_applicability_map` reads."""
from __future__ import annotations

from typing import Any

from .rules import RULES

__all__ = ["trust_card", "guardrail_checks", "STAGE_KEYS"]

STAGE_KEYS = ("data_geometry", "model", "physics", "benchmark", "uncertainty", "decision")
_FALLBACK = {"data_geometry": ("Data & geometry", "Are the inputs what we think they are?"),
             "model": ("Model", "Is the model used inside the range it was built for?"),
             "physics": ("Physics", "Does the answer obey the governing equations?"),
             "benchmark": ("Benchmark", "Does it match a trusted reference?"),
             "uncertainty": ("Uncertainty", "How wrong could it be?"),
             "decision": ("Decision", "Is it good enough for the decision at hand?")}


def _stages() -> dict[str, tuple[str, str]]:
    try:
        from . import _reuse
        st = _reuse.trust_stages()
        return {k: (v[0], v[1]) if isinstance(v, (tuple, list)) else (str(v), "") for k, v in st.items()}
    except Exception:                                                 # the lab package is optional at run time
        return dict(_FALLBACK)


def trust_card(report) -> dict[str, dict[str, Any]]:
    """{target: {stage: {"title", "question", "status", "evidence": [...]}}}. Status: answered (every check of the
    stage passed), failed (an error), partial (warnings only), open (nothing checked: say so, never assume)."""
    stages = _stages()
    out: dict[str, dict[str, Any]] = {}
    names = [t["name"] for t in report.targets] + (["integrity"] if any(o.target == "integrity"
                                                                         for o in report.outcomes) else [])
    for name in names:
        card = {}
        for key in STAGE_KEYS:
            title, question = stages.get(key, _FALLBACK[key])
            mine = [o for o in report.outcomes if o.target == name and RULES[o.rule].stage == key and o.rule != "PC001"]
            if not mine:
                status = "open"
            elif any(not o.passed and not o.baselined and o.severity == "error" for o in mine):
                status = "failed"
            elif any(not o.passed and not o.baselined for o in mine):
                status = "partial"
            else:
                status = "answered"
            card[key] = {"title": title, "question": question, "status": status,
                         "evidence": [f"{o.rule} {'pass' if o.passed else 'FAIL'}: {o.message}" for o in mine]}
        out[name] = card
    return out


def guardrail_checks(report, target: str) -> list[dict[str, Any]]:
    """``[{"name", "passed", "detail"}]`` for veriphysics: one entry per evidence name, failed if any check
    feeding it failed."""
    by: dict[str, list] = {}
    for o in report.outcomes:
        ev = RULES[o.rule].evidence
        if o.target == target and ev:
            by.setdefault(ev, []).append(o)
    return [{"name": ev, "passed": all(o.passed or o.baselined for o in os_),
             "detail": "; ".join(f"{o.rule}: {o.message}" for o in os_ if not o.passed)[:500] or
             f"{len(os_)} check(s) passed"} for ev, os_ in sorted(by.items())]
