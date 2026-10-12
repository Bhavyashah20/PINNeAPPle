"""The interactive study: describe the problem, answer the questions, choose, and get one final report.

    from pinneapple_design.fitmesh import FitMeshSession
    s = FitMeshSession("Erosion and pressure drop in a 3 inch pneumatic pipe bend, air at 20 m/s", "study/")
    s.step()                        # what is needed now: questions, or options to choose from
    s.answer(accuracy="design")
    s.choose_geometry("pipe_bend_90")
    s.choose_parameters(n=4)        # the suggested parameters and ranges, or vary={"bend_ratio": [1, 3]}
    s.generate_cad()                # STEP + STL per variant, with the checks
    s.plan_meshes()                 # fit-for-purpose mesh plan per variant, with the rules used
    s.generate_meshes()             # meshes + quality (and the coarser family for design / verification)
    s.report()                      # one JSON + HTML report with a verdict per stage

``FitMeshSession.run(text, answers, ...)`` does all of it in one call. ``llm=LLMAssist(provider="ollama",
model=...)`` lets a local (or hosted) model read the problem, rank geometries, propose ranges, write a CadQuery
part when no template fits, and review the mesh plan; every proposal is validated before it is used.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from .cad import CadVariant, build_variant, compare_with_references
from .catalog import TEMPLATES, Param, Template
from .context import ProblemContext, parse
from .llm import (
    LLMAssist,
    free_geometry,
    propose_parameters,
    rank_geometries,
    read_problem,
    review_mesh_plan,
    run_free_geometry,
)
from .mesh_gen import MeshResult, generate
from .mesh_plan import MeshPlan, plan_mesh
from .report import build_report, gci, write_report
from .suggest import sample, scale_to_context, suggest_geometries, suggest_parameters

STEPS = ("problem", "geometry", "parameters", "cad", "mesh_plan", "meshes", "report")


class FitMeshSession:
    def __init__(self, text: str, out_dir: str, llm: LLMAssist | None = None,
                 log: Callable[[str], None] = print):
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.llm, self.log = llm, log
        self.ctx: ProblemContext = parse(text, llm=(lambda t: read_problem(llm, t)) if llm else None)
        self.answers: dict[str, Any] = {}
        self.template: Template | None = None
        self.freeform: dict[str, Any] | None = None
        self.geometry_why = ""
        self.base: dict[str, float] = {}
        self.parameters: list[dict[str, Any]] = []
        self.variants: list[CadVariant] = []
        self.plans: dict[int, MeshPlan] = {}
        self.meshes: list[MeshResult] = []
        self.convergence: dict[str, Any] = {}
        self.llm_notes: list[str] = []

    # ------------------------------------------------------------------ where are we
    def step(self) -> dict[str, Any]:
        """The current step with its open questions or options (what an interface shows next)."""
        if self.template is None:
            q = self.ctx.questions()
            return {"step": "geometry" if not q else "problem", "questions": q,
                    "options": self.geometry_options()}
        if not self.variants and not self.parameters:
            return {"step": "parameters", "options": self.parameter_options()}
        if not self.variants:
            return {"step": "cad", "parameters": self.parameters}
        if not self.plans:
            return {"step": "mesh_plan"}
        if not self.meshes:
            return {"step": "meshes", "plans": {i: p.to_dict() for i, p in self.plans.items()}}
        return {"step": "report"}

    def answer(self, **facts) -> FitMeshSession:
        for k, v in facts.items():
            self.ctx.set(k, v, "answer")
            self.answers[k] = v
        return self

    # ------------------------------------------------------------------ geometry
    def geometry_options(self, top: int = 3) -> list[dict[str, Any]]:
        rows = suggest_geometries(self.ctx, top=top)
        if self.llm:
            try:
                got = rank_geometries(self.llm, self.ctx)
                for c in got["choices"]:
                    row = next((r for r in rows if r["template"] == c["template"]), None)
                    if row:
                        row["why"] += f"; model: {c.get('why', '')}"
                        row["score"] += 2
                    else:
                        t = TEMPLATES[c["template"]]
                        rows.append({"template": t.key, "title": t.title, "score": 2.0, "why": "model: " +
                                     c.get("why", ""), "description": t.description,
                                     "references": [{"citation": r.citation, "what": r.what} for r in t.references]})
                if got["none_fits"]:
                    rows.append({"template": "freeform", "title": "Part written by the language model",
                                 "score": 0.0, "why": got.get("freeform_description", ""), "references": []})
                rows.sort(key=lambda r: -r["score"])
            except Exception as exc:                       # the model is optional: never block on it
                self.llm_notes.append(f"geometry ranking by the model failed: {exc}")
        return rows

    def choose_geometry(self, key: str, description: str | None = None) -> FitMeshSession:
        if key == "freeform":
            if not self.llm:
                raise ValueError("a free-form part needs a language model (llm=LLMAssist(...))")
            ff = free_geometry(self.llm, description or self.ctx.text)
            if ff["problems"]:
                raise ValueError("the model's CadQuery was refused: " + "; ".join(ff["problems"]))
            self.freeform = ff
            self.template = _freeform_template(ff, self.ctx, self)
            self.geometry_why = "no catalogue geometry fits; part written by the language model"
        else:
            self.template = TEMPLATES[key]
            opt = next((o for o in suggest_geometries(self.ctx, top=len(TEMPLATES)) if o["template"] == key), None)
            self.geometry_why = opt["why"] if opt else "chosen by the user"
        self.base = (self.freeform["params"] if self.freeform else scale_to_context(self.template, self.ctx))
        return self

    # ------------------------------------------------------------------ parameters
    def parameter_options(self) -> list[dict[str, Any]]:
        rows = suggest_parameters(self.template, self.ctx, self.base)
        if self.llm and not self.freeform:
            try:
                for r in propose_parameters(self.llm, self.template, self.ctx, self.base):
                    row = next(x for x in rows if x["name"] == r["name"])
                    row.update(vary=r["vary"], range=r["range"], reason=f"{row['reason']}; model: {r['why']}"
                               + (f" ({r['note']})" if r["note"] else ""))
            except Exception as exc:
                self.llm_notes.append(f"parameter proposal by the model failed: {exc}")
        return rows

    def choose_parameters(self, vary: dict[str, list[float]] | None = None, n: int = 4, seed: int = 0
                          ) -> FitMeshSession:
        rows = self.parameter_options()
        if vary is not None:
            for r in rows:
                r["vary"] = r["name"] in vary
                if r["vary"]:
                    p = self.template.param(r["name"])
                    lo, hi = sorted(float(x) for x in vary[r["name"]])
                    r["range"] = [max(lo, p.limits[0]), min(hi, p.limits[1])]
                    r["reason"] = r["reason"] + " (range chosen by the user)"
        self.parameters = rows
        ranges = {r["name"]: r["range"] for r in rows if r["vary"]}
        self._samples = sample(self.template, self.base, ranges, n=n, seed=seed)
        return self

    # ------------------------------------------------------------------ CAD
    def generate_cad(self) -> list[CadVariant]:
        out = os.path.join(self.out_dir, "cad")
        self.variants = []
        for i, v in enumerate(self._samples):
            if self.freeform:
                self.variants.append(self._freeform_variant(v, out, i))
            else:
                self.variants.append(build_variant(self.template, v, out, i, self.ctx.accuracy))
            self.log(f"CAD variant {i}: {self.variants[-1].verdict}")
        return self.variants

    def _freeform_variant(self, v, out, i) -> CadVariant:
        tol = 0.001 * float(self.freeform.get("min_feature") or 0.01)
        r = run_free_geometry(self.freeform["code"], v, os.path.join(out, f"freeform_{i:03d}"), tol)
        var = CadVariant(index=i, params=dict(v))
        if not r["ok"]:
            var.checks.append({"check": "CAD build (isolated)", "status": "fail", "value": r["error"][:200],
                               "limit": "builds", "note": ""})
            return var
        import trimesh
        var.step, var.stl, var.volume, var.area = r["step"], r["stl"], r["volume"], r["area"]
        m = trimesh.load(var.stl, force="mesh")
        var.bbox = list(m.extents)
        var.checks += [
            {"check": "valid solid (BRep check)", "status": "pass" if r["valid"] else "fail", "value": r["valid"],
             "limit": True, "note": "language-model geometry"},
            {"check": "positive volume", "status": "pass" if r["volume"] > 0 else "fail", "value": r["volume"],
             "limit": "> 0", "note": ""},
            {"check": "STL watertight", "status": "pass" if m.is_watertight else "fail", "value": m.is_watertight,
             "limit": True, "note": ""},
            {"check": "published reference", "status": "warn", "value": "none", "limit": "-",
             "note": "no reference object to compare with: check the dimensions against the drawing"}]
        return var

    # ------------------------------------------------------------------ mesh
    def plan_meshes(self, budget_cells: float = 5e6) -> dict[int, MeshPlan]:
        self.plans = {}
        for var in self.variants:
            if var.verdict == "fail":
                continue
            p = plan_mesh(self.template, var.params, self.ctx, budget_cells)
            if self.llm:
                try:
                    p = review_mesh_plan(self.llm, p, self.ctx)
                except Exception as exc:
                    self.llm_notes.append(f"mesh review by the model failed: {exc}")
            self.plans[var.index] = p
        return self.plans

    def generate_meshes(self, variants: list[int] | None = None, levels: list[float] | None = None,
                        max_cells: float = 3e6, procs: int = 1) -> list[MeshResult]:
        out = os.path.join(self.out_dir, "mesh")
        for var in self.variants:
            if var.index not in self.plans or (variants is not None and var.index not in variants):
                continue
            plan = self.plans[var.index]
            for lev in levels or plan.family:
                r = generate(self.template, var, plan, out, level=lev, procs=procs, max_cells=max_cells,
                             log=self.log)
                self.log(f"mesh variant {var.index} x{lev:g}: {r.cells:,} cells, {r.verdict} {r.error}")
                self.meshes.append(r)
        return self.meshes

    def add_results(self, quantity: str, values: list[float], variant: int = 0) -> dict[str, Any]:
        """Quantity computed on the mesh family (finest first) -> grid-convergence index for that variant."""
        ms = sorted([m for m in self.meshes if m.variant == variant and not m.error], key=lambda m: m.level)
        if len(ms) < 3 or len(values) < 3:
            raise ValueError("the grid-convergence index needs the quantity on three meshes of one variant")
        sizes = [(_volume(self, variant) / max(m.cells, 1)) ** (1 / 3) for m in ms[:3]]
        self.convergence[f"{quantity} (variant {variant})"] = gci(values[:3], sizes)
        return self.convergence[f"{quantity} (variant {variant})"]

    # ------------------------------------------------------------------ report
    def report(self) -> dict[str, str]:
        rep = build_report(self)
        if self.llm_notes:
            rep["llm_notes"] = self.llm_notes
        return write_report(rep, os.path.join(self.out_dir, "report"))

    def save(self) -> str:
        p = os.path.join(self.out_dir, "session.json")
        with open(p, "w") as f:
            json.dump({"context": self.ctx.to_dict(), "answers": self.answers,
                       "template": self.template.key if self.template else None, "base": self.base,
                       "parameters": self.parameters}, f, indent=1, default=str)
        return p

    # ------------------------------------------------------------------ all at once
    @classmethod
    def run(cls, text: str, out_dir: str, answers: dict[str, Any] | None = None, geometry: str | None = None,
            vary: dict[str, list[float]] | None = None, n: int = 3, mesh: bool = True, max_cells: float = 3e6,
            procs: int = 1, llm: LLMAssist | None = None, log=print) -> FitMeshSession:
        s = cls(text, out_dir, llm=llm, log=log)
        s.answer(**(answers or {}))
        opts = s.geometry_options()
        if geometry is None and not opts:
            raise ValueError("no geometry matches the problem; name one with geometry=")
        s.choose_geometry(geometry or opts[0]["template"])
        s.choose_parameters(vary=vary, n=n)
        s.generate_cad()
        s.plan_meshes()
        if mesh:
            s.generate_meshes(max_cells=max_cells, procs=procs)
        s.report()
        return s


def _volume(s: FitMeshSession, variant: int) -> float:
    v = next(x for x in s.variants if x.index == variant)
    plan = s.plans.get(variant)
    if plan and plan.kind == "external_flow":       # fluid volume = domain minus body
        lx, ly, lz = v.bbox
        d = plan.domain
        return ((lx + d["upstream"] + d["downstream"]) * (ly + 2 * d["lateral"]) * (lz + 2 * d["top"])) - v.volume
    return v.volume


def _freeform_template(ff: dict[str, Any], ctx: ProblemContext, session: FitMeshSession) -> Template:
    """A catalogue-shaped wrapper around a language-model part: its build reads the STEP made in isolation."""
    params = tuple(Param(k, float(v), "m", (0.5 * float(v), 1.5 * float(v)), (1e-9, 1e9), "set by the model",
                         "medium") for k, v in ff["params"].items())
    mf = float(ff.get("min_feature") or 0.01)
    rl = float(ff.get("ref_length") or 0.1)

    def build(v):
        import cadquery as cq
        var = next((x for x in session.variants if x.params == v and x.step), None)
        if var is None:
            raise RuntimeError("build the language-model part with generate_cad() first")
        return cq.importers.importStep(var.step)

    return Template(key="freeform", title="Part written by the language model", physics=tuple(ctx.physics),
                    keywords=(), description=ctx.text, params=params, references=(), quantities=(),
                    build=build, volume=None, min_feature=lambda v: mf, ref_length=lambda v: rl)


__all__ = ["FitMeshSession", "STEPS", "compare_with_references"]
