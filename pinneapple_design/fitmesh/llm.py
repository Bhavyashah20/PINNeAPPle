"""Optional language-model help for every step, with a local model if you like (Ollama via ``pinneapple_llm``).

The model proposes; PINNeAPPle checks. Each helper validates what comes back and never lets it bypass a rule:

- ``read_problem``: facts from free text (physics, fluid, speed, size, purpose, quantities, material, load);
- ``rank_geometries``: picks templates from the real catalogue only (unknown keys are dropped);
- ``propose_parameters``: vary/keep and ranges, clipped to the template's validity limits (clipping is reported);
- ``free_geometry``: when no template fits, a CadQuery script that runs in a separate, isolated Python process
  with an import allow-list and a timeout; the solid then goes through the same CAD checks, marked as
  "language-model geometry, no published reference";
- ``review_mesh_plan``: suggestions on the rule-based plan; only y+ target, growth and layer count may change and
  only inside the rule limits; every accepted or refused change is written into the plan's rationale.

Every call can be logged with ``pinneapple_llm.ConversationStore`` for audit.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .catalog import TEMPLATES, Template
from .context import FLUIDS, MATERIALS, PHYSICS_WORDS, QUANTITY_WORDS, ProblemContext
from .mesh_plan import MeshPlan

ALLOWED_IMPORTS = {"cadquery", "math", "numpy"}
LIMITS = {"y_plus": (0.5, 300.0), "growth": (1.05, 1.25), "count": (1, 40)}


@dataclass
class LLMAssist:
    """Which model to ask. ``provider``: "ollama" (local), "anthropic" or "openai"; ``call`` overrides everything
    (a function ``(prompt, system) -> str`` returning JSON), which is how tests and custom backends plug in."""

    provider: str = "ollama"
    model: str | None = None
    api_key: str | None = None
    conversation_store: Any = None
    call: Callable[[str, str], str] | None = None
    provider_kwargs: dict[str, Any] | None = None

    def ask(self, prompt: str, system: str, module: str) -> dict[str, Any]:
        if self.call is not None:
            raw = self.call(prompt, system)
        else:
            from pinneapple_llm import call_llm
            raw = call_llm(prompt, provider=self.provider, model=self.model, api_key=self.api_key, system=system,
                           json_mode=True, module=f"fitmesh.{module}", conversation_store=self.conversation_store,
                           **(self.provider_kwargs or {}))
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start:end + 1])
            raise ValueError(f"the model did not return JSON: {raw[:300]}") from None


# ---------------------------------------------------------------- 1. read the problem
_SYS_READ = ("You extract engineering facts from a problem description. Answer only JSON with any of the keys: "
             "physics (list from %s), fluid (one of %s), speed (m/s), length (m), reynolds, accuracy (screening, "
             "design or verification), quantities (list from %s), material (one of %s), load (N). Leave out what "
             "the text does not say; never guess numbers.")


def read_problem(llm: LLMAssist, text: str) -> dict[str, Any]:
    sys_ = _SYS_READ % (list(PHYSICS_WORDS), list(FLUIDS), sorted(QUANTITY_WORDS), list(MATERIALS))
    out = llm.ask(f"PROBLEM:\n{text}", sys_, "read_problem")
    clean: dict[str, Any] = {}
    if isinstance(out.get("physics"), list):
        clean["physics"] = [p for p in out["physics"] if p in PHYSICS_WORDS]
    if out.get("fluid") in FLUIDS:
        clean["fluid"] = out["fluid"]
    if out.get("material") in MATERIALS:
        clean["material"] = out["material"]
    if out.get("accuracy") in ("screening", "design", "verification"):
        clean["accuracy"] = out["accuracy"]
    if isinstance(out.get("quantities"), list):
        clean["quantities"] = [q for q in out["quantities"] if q in QUANTITY_WORDS]
    for k in ("speed", "length", "reynolds", "load"):
        try:
            if out.get(k) is not None and float(out[k]) > 0:
                clean[k] = float(out[k])
        except (TypeError, ValueError):
            pass
    return {k: v for k, v in clean.items() if v not in ([], None)}


# ---------------------------------------------------------------- 2. geometry
_SYS_GEOM = ("You choose reference geometries for an engineering study. Answer only JSON: {\"choices\": "
             "[{\"template\": key, \"why\": text}], \"none_fits\": bool, \"freeform_description\": text}. Use only "
             "the template keys given. Set none_fits true if no template represents the problem.")


def rank_geometries(llm: LLMAssist, ctx: ProblemContext) -> dict[str, Any]:
    cat = [{"key": t.key, "title": t.title, "physics": t.physics, "description": t.description} for t in
           TEMPLATES.values()]
    out = llm.ask(f"PROBLEM:\n{ctx.text}\n\nFACTS:\n{json.dumps(ctx.to_dict()['facts'])}\n\nTEMPLATES:\n"
                  f"{json.dumps(cat)}", _SYS_GEOM, "rank_geometries")
    choices = [c for c in out.get("choices", []) if isinstance(c, dict) and c.get("template") in TEMPLATES]
    dropped = [c.get("template") for c in out.get("choices", []) if isinstance(c, dict)
               and c.get("template") not in TEMPLATES]
    return {"choices": choices, "dropped": dropped, "none_fits": bool(out.get("none_fits")),
            "freeform_description": out.get("freeform_description", "")}


_SYS_PARAMS = ("You propose which geometry parameters to vary in a study and over which range. Answer only JSON: "
               "{\"parameters\": [{\"name\": n, \"vary\": bool, \"range\": [lo, hi], \"why\": text}]}. Use only the "
               "parameter names given, in the units given.")


def propose_parameters(llm: LLMAssist, tpl: Template, ctx: ProblemContext, base: dict[str, float]
                       ) -> list[dict[str, Any]]:
    spec = [{"name": p.name, "unit": p.unit, "base": base[p.name], "typical": p.typical, "limits": p.limits,
             "role": p.role} for p in tpl.params]
    out = llm.ask(f"PROBLEM:\n{ctx.text}\n\nTEMPLATE: {tpl.title}\nPARAMETERS:\n{json.dumps(spec)}", _SYS_PARAMS,
                  "propose_parameters")
    rows = []
    names = {p.name for p in tpl.params}
    for r in out.get("parameters", []):
        if not isinstance(r, dict) or r.get("name") not in names:
            continue
        p = tpl.param(r["name"])
        try:
            lo, hi = sorted(float(x) for x in r.get("range", p.typical))
        except (TypeError, ValueError):
            lo, hi = p.typical
        clo, chi = max(lo, p.limits[0]), min(hi, p.limits[1])
        note = "" if (clo, chi) == (lo, hi) else f"clipped from [{lo:g}, {hi:g}] to the validity limits"
        rows.append({"name": p.name, "vary": bool(r.get("vary")), "range": [clo, chi], "why": r.get("why", ""),
                     "note": note, "source": "llm"})
    return rows


_SYS_FREE = ("You write CadQuery (Python) for a part described in words. Answer only JSON: {\"params\": "
             "{name: value_in_metres_or_degrees}, \"code\": text, \"min_feature\": metres, \"ref_length\": metres}. "
             "The code must define build(p) returning a cadquery Workplane or Shape, use only 'import cadquery as "
             "cq', 'import math' and 'import numpy as np', and no file, network or system access.")


def free_geometry(llm: LLMAssist, description: str) -> dict[str, Any]:
    """A CadQuery script from the model, checked statically; run it with :func:`run_free_geometry`."""
    out = llm.ask(f"PART:\n{description}", _SYS_FREE, "free_geometry")
    code = out.get("code", "")
    problems = check_code(code)
    return {"params": {k: float(v) for k, v in (out.get("params") or {}).items()}, "code": code,
            "min_feature": out.get("min_feature"), "ref_length": out.get("ref_length"), "problems": problems}


def check_code(code: str) -> list[str]:
    """Static checks before anything runs: syntax, imports on the allow-list, no dunder or I/O names, build()."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [f"import {a.name}" for a in node.names if a.name.split(".")[0] not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                bad.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Name) and node.id in {"open", "exec", "eval", "compile", "__import__", "input",
                                                        "globals", "locals", "getattr", "setattr", "vars"}:
            bad.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            bad.append(f"attribute {node.attr}")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build" for n in tree.body):
        bad.append("no build(p) function")
    return bad


_RUNNER = r'''
import json, sys
import cadquery as cq
src, params, step, stl, tol = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5])
ns = {}
exec(compile(open(src).read(), "llm_geometry.py", "exec"), ns)
shape = ns["build"](params)
solid = shape.val() if hasattr(shape, "val") else shape
cq.exporters.export(solid, step)
cq.exporters.export(solid, stl, tolerance=tol, angularTolerance=0.1)
print(json.dumps({"volume": solid.Volume(), "area": solid.Area(), "valid": solid.isValid()}))
'''


def run_free_geometry(code: str, params: dict[str, float], out_dir: str, tol: float, timeout: int = 120
                      ) -> dict[str, Any]:
    """Run model-written CadQuery in a separate isolated interpreter (``python -I``), with a timeout."""
    problems = check_code(code)
    if problems:
        return {"ok": False, "error": "refused before running: " + "; ".join(problems)}
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(out_dir, "llm_geometry.py")
    with open(src, "w") as f:
        f.write(code)
    step, stl = os.path.join(out_dir, "llm_geometry.step"), os.path.join(out_dir, "llm_geometry.stl")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_RUNNER)
        runner = f.name
    try:
        r = subprocess.run([sys.executable, "-I", runner, src, json.dumps(params), step, stl, str(tol)],
                           capture_output=True, text=True, timeout=timeout, cwd=out_dir, env={"PATH": "/usr/bin:/bin"})
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout} s"}
    finally:
        os.remove(runner)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip()[-800:]}
    info = json.loads(r.stdout.strip().splitlines()[-1])
    return {"ok": True, "step": step, "stl": stl, "code_file": src, **info}


# ---------------------------------------------------------------- 3. mesh plan review
_SYS_MESH = ("You review a CFD/FEA mesh plan for an engineering purpose. Answer only JSON: {\"changes\": "
             "{\"y_plus\": number, \"growth\": number, \"count\": integer} (only the ones you would change), "
             "\"comments\": [text]}. Keep y+ in 0.5-300 avoiding 5-30, growth in 1.05-1.25.")


def review_mesh_plan(llm: LLMAssist, plan: MeshPlan, ctx: ProblemContext) -> MeshPlan:
    out = llm.ask(f"PROBLEM:\n{ctx.text}\n\nPLAN:\n{json.dumps(plan.to_dict(), default=float)}", _SYS_MESH,
                  "review_mesh_plan")
    for c in out.get("comments", [])[:5]:
        plan.rationale.append(f"Model review: {c}")
    for k, v in (out.get("changes") or {}).items():
        if k not in LIMITS or not plan.layers or "first_height" not in plan.layers:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        lo, hi = LIMITS[k]
        if not (lo <= v <= hi) or (k == "y_plus" and 5 < v < 30):
            plan.rationale.append(f"Model suggested {k} = {v:g}: refused (outside {lo:g}-{hi:g}"
                                  f"{' or in the buffer layer 5-30' if k == 'y_plus' else ''}).")
            continue
        old = plan.layers.get(k)
        if k == "y_plus" and old:
            plan.layers["first_height"] *= v / old
        plan.layers[k] = int(v) if k == "count" else v
        g, n, y1 = plan.layers["growth"], plan.layers["count"], plan.layers["first_height"]
        plan.layers["total_thickness"] = y1 * (g ** n - 1) / (g - 1)
        plan.rationale.append(f"Model changed {k} from {old} to {plan.layers[k]} (within the rule limits).")
    return plan
