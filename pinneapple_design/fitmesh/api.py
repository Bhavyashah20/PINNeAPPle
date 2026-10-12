"""HTTP API over :class:`FitMeshSession`: one session per study, one call per step, the files and the report.

    from pinneapple_design.fitmesh.api import create_app
    app = create_app("fitmesh_runs/")          # uvicorn module:app

    POST /sessions                 {"text": "...", "llm": {"provider": "ollama", "model": "qwen2.5:7b"}}  (llm optional)
    GET  /sessions/{id}            current step, questions and options
    POST /sessions/{id}/answer     {"accuracy": "design", "speed": 20}
    POST /sessions/{id}/geometry   {"key": "pipe_bend_90"}  or {"key": "freeform", "description": "..."}
    POST /sessions/{id}/parameters {"vary": {"bend_ratio": [1, 3]}, "n": 4}   (omit vary for the suggestion)
    POST /sessions/{id}/cad
    POST /sessions/{id}/plan       {"budget_cells": 5e6}
    POST /sessions/{id}/meshes     {"variants": [0], "max_cells": 3e6, "procs": 4}
    POST /sessions/{id}/results    {"quantity": "pressure drop", "values": [f1, f2, f3], "variant": 0}
    GET  /sessions/{id}/report     JSON report (?format=html for the page)
    GET  /sessions/{id}/files/{path}  any file of the study (STEP, STL, .msh, .inp, report)
"""
from __future__ import annotations

import os
import uuid
from typing import Any


def create_app(root: str = "fitmesh_runs"):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    from .llm import LLMAssist
    from .report import build_report, render_html
    from .session import FitMeshSession

    app = FastAPI(title="PINNeAPPle fitmesh", description="Problem description -> checked CAD -> fit-for-purpose "
                                                           "meshes -> one trust report")
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    sessions: dict[str, FitMeshSession] = {}
    required, optional = Body(...), Body(default={})

    def get(sid: str) -> FitMeshSession:
        if sid not in sessions:
            raise HTTPException(404, "unknown session")
        return sessions[sid]

    def state(sid: str) -> dict[str, Any]:
        s = get(sid)
        return {"id": sid, **_jsonable(s.step()), "geometry": s.template.key if s.template else None,
                "variants": [{"index": v.index, "verdict": v.verdict, "params": v.params} for v in s.variants],
                "meshes": [{"variant": m.variant, "level": m.level, "cells": m.cells, "verdict": m.verdict,
                            "error": m.error} for m in s.meshes], "llm_notes": s.llm_notes}

    def guard(fn):
        try:
            return fn()
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/sessions")
    def create(body: dict[str, Any] = required):
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(422, "text is required")
        llm = None
        if body.get("llm"):
            cfg = body["llm"]
            llm = LLMAssist(provider=cfg.get("provider", "ollama"), model=cfg.get("model"),
                            provider_kwargs=cfg.get("options"))
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = FitMeshSession(text, os.path.join(root, sid), llm=llm, log=lambda m: None)
        return state(sid)

    @app.get("/sessions/{sid}")
    def show(sid: str):
        return state(sid)

    @app.post("/sessions/{sid}/answer")
    def answer(sid: str, body: dict[str, Any] = required):
        get(sid).answer(**body)
        return state(sid)

    @app.post("/sessions/{sid}/geometry")
    def geometry(sid: str, body: dict[str, Any] = required):
        guard(lambda: get(sid).choose_geometry(body["key"], body.get("description")))
        return state(sid)

    @app.post("/sessions/{sid}/parameters")
    def parameters(sid: str, body: dict[str, Any] = optional):
        guard(lambda: get(sid).choose_parameters(vary=body.get("vary"), n=int(body.get("n", 4)),
                                                 seed=int(body.get("seed", 0))))
        return {**state(sid), "parameters": get(sid).parameters}

    @app.post("/sessions/{sid}/cad")
    def cad(sid: str):
        s = get(sid)
        if not hasattr(s, "_samples"):
            raise HTTPException(409, "choose the parameters first")
        s.generate_cad()
        return {**state(sid), "checks": {v.index: v.checks for v in s.variants},
                "comparison": {v.index: v.comparison for v in s.variants}}

    @app.post("/sessions/{sid}/plan")
    def plan(sid: str, body: dict[str, Any] = optional):
        s = get(sid)
        if not s.variants:
            raise HTTPException(409, "generate the CAD first")
        s.plan_meshes(float(body.get("budget_cells", 5e6)))
        return {**state(sid), "plans": {i: p.to_dict() for i, p in s.plans.items()}}

    @app.post("/sessions/{sid}/meshes")
    def meshes(sid: str, body: dict[str, Any] = optional):
        s = get(sid)
        if not s.plans:
            raise HTTPException(409, "plan the meshes first")
        s.generate_meshes(variants=body.get("variants"), levels=body.get("levels"),
                          max_cells=float(body.get("max_cells", 3e6)), procs=int(body.get("procs", 1)))
        return state(sid)

    @app.post("/sessions/{sid}/results")
    def results(sid: str, body: dict[str, Any] = required):
        return guard(lambda: get(sid).add_results(body["quantity"], list(body["values"]), int(body.get("variant", 0))))

    @app.get("/sessions/{sid}/report")
    def report(sid: str, format: str = "json"):
        s = get(sid)
        s.report()
        rep = build_report(s)
        return HTMLResponse(render_html(rep)) if format == "html" else _jsonable(rep)

    @app.get("/sessions/{sid}/files/{path:path}")
    def files(sid: str, path: str):
        base = get(sid).out_dir
        full = os.path.realpath(os.path.join(base, path))
        if not full.startswith(os.path.realpath(base) + os.sep) or not os.path.isfile(full):
            raise HTTPException(404, "no such file in this study")
        return FileResponse(full)

    return app


def _jsonable(o: Any) -> Any:
    import json
    return json.loads(json.dumps(o, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)))


__all__ = ["create_app"]
