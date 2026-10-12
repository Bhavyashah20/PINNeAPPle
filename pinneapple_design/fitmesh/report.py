"""One final trust report for the whole study, plus the grid-convergence calculation it relies on.

Each stage gets a pass / warn / fail verdict linked to its evidence: problem facts (and where each came from),
the geometry choice and its references, the parameters and ranges (and why), every CAD variant's checks and its
comparison with the published reference objects, the mesh plan with its rules and sources, every mesh's quality,
and, when results of a mesh family are given, the grid-convergence index. Written as JSON (machine-readable) and
HTML (to read and archive).
"""
from __future__ import annotations

import html
import json
import math
from collections.abc import Sequence
from typing import Any


def gci(values: Sequence[float], sizes: Sequence[float], safety: float | None = None) -> dict[str, Any]:
    """Grid-convergence index from three meshes (Roache 1994; procedure of Celik et al. 2008, J. Fluids Eng. 130).

    ``values`` and ``sizes`` are ordered fine, medium, coarse (``sizes``: representative cell size, e.g.
    (V/N)^(1/3)). Returns the observed order p, the Richardson extrapolation, the fine-grid GCI (safety factor
    1.25 for three grids), and whether the solutions are in the asymptotic range (GCI_23 / (r^p GCI_12) close to
    1). Oscillatory convergence (sign change) is flagged instead of forced through the formula.
    """
    f1, f2, f3 = (float(x) for x in values)
    h1, h2, h3 = (float(x) for x in sizes)
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    out: dict[str, Any] = {"values": [f1, f2, f3], "sizes": [h1, h2, h3], "r21": r21, "r32": r32}
    if r21 <= 1.0 or r32 <= 1.0:
        out["status"] = "fail"
        out["note"] = "sizes must grow from fine to coarse"
        return out
    if e21 == 0 or e32 == 0:
        out.update(status="pass", p=None, extrapolated=f1, gci_fine=0.0, note="no change between meshes")
        return out
    if e32 / e21 < 0:
        out.update(status="warn", note="oscillatory convergence: the GCI formula does not apply; report the range "
                                       f"{min(f1, f2, f3):.6g} to {max(f1, f2, f3):.6g}")
        return out
    s = 1.0 if e32 / e21 > 0 else -1.0
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)            # fixed-point iteration for unequal ratios
    for _ in range(50):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    fs = safety or 1.25
    ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    ea21 = abs((f1 - f2) / f1) if f1 else float("inf")
    g12 = fs * ea21 / (r21 ** p - 1)
    ea32 = abs((f2 - f3) / f2) if f2 else float("inf")
    g23 = fs * ea32 / (r32 ** p - 1)
    asym = g23 / (r21 ** p * g12) if g12 else float("nan")
    out.update(p=p, extrapolated=ext, gci_fine=g12, gci_medium=g23, asymptotic_ratio=asym, safety_factor=fs,
               status="pass" if 0.9 <= asym <= 1.1 else "warn",
               note=f"fine-mesh numerical uncertainty about {g12:.2%} (observed order {p:.2f}); "
                    + ("in the asymptotic range" if 0.9 <= asym <= 1.1 else
                       "not yet in the asymptotic range: refine further before quoting the uncertainty"))
    if r21 < 1.3 or r32 < 1.3:
        out["note"] += "; refinement ratio below 1.3 (Celik et al. recommend > 1.3)"
        out["status"] = "warn"
    return out


def _verdict(statuses: list[str]) -> str:
    return "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass" if statuses else "not run"


def build_report(session) -> dict[str, Any]:
    """Collect every stage of a :class:`FitMeshSession` into one report with a verdict per stage."""
    s = session
    stages: list[dict[str, Any]] = []
    facts = {k: {"value": v.value, "source": v.source} for k, v in s.ctx.facts.items()}
    open_q = [q["key"] for q in s.ctx.questions()]
    stages.append({"stage": "problem", "verdict": "warn" if open_q else "pass",
                   "evidence": {"text": s.ctx.text, "facts": facts, "unanswered": open_q},
                   "note": ("unanswered: " + ", ".join(open_q) + " (defaults were used)") if open_q else
                           "every fact the choices depend on is known"})
    if s.template is not None:
        t = s.template
        stages.append({"stage": "geometry", "verdict": "warn" if s.freeform else "pass",
                       "evidence": {"template": t.key if not s.freeform else "language-model geometry",
                                    "title": t.title, "why": s.geometry_why,
                                    "references": [{"citation": r.citation, "what": r.what} for r in t.references]},
                       "note": ("language-model geometry: no published reference to compare with"
                                if s.freeform else "reference geometry with published data")})
        stages.append({"stage": "parameters", "verdict": "pass" if s.parameters else "not run",
                       "evidence": {"parameters": s.parameters, "n_variants": len(s.variants)}})
    if s.variants:
        st = [v.verdict for v in s.variants]
        stages.append({"stage": "cad", "verdict": _verdict(st),
                       "evidence": [{"variant": v.index, "params": v.params, "verdict": v.verdict,
                                     "checks": v.checks, "comparison": v.comparison, "step": v.step, "stl": v.stl,
                                     "volume_m3": v.volume} for v in s.variants]})
    if s.plans:
        any_plan = next(iter(s.plans.values()))
        stages.append({"stage": "mesh plan", "verdict": "warn" if any(p.warnings for p in s.plans.values())
                       else "pass", "evidence": {i: p.to_dict() for i, p in s.plans.items()},
                       "note": f"{any_plan.tool}, {any_plan.elements}, purpose '{any_plan.purpose}'"})
    if s.meshes:
        st = [m.verdict for m in s.meshes]
        stages.append({"stage": "meshes", "verdict": _verdict([x if x in ("pass", "warn", "fail") else "warn"
                                                               for x in st]),
                       "evidence": [{"variant": m.variant, "level": m.level, "tool": m.tool, "cells": m.cells,
                                     "seconds": m.seconds, "verdict": m.verdict, "error": m.error,
                                     "files": m.files, "layers": m.layers,
                                     "quality": {"summary": m.quality.get("summary"),
                                                 "checks": m.quality.get("checks"),
                                                 "regions": m.quality.get("regions")}} for m in s.meshes]})
    if s.convergence:
        stages.append({"stage": "grid convergence", "verdict": _verdict([c["status"] for c in s.convergence.values()]),
                       "evidence": s.convergence})
    elif s.plans and any(p.purpose == "verification" for p in s.plans.values()):
        stages.append({"stage": "grid convergence", "verdict": "warn",
                       "note": "three meshes built for verification; give the quantity on each (add_results) to "
                               "compute the GCI"})
    overall = _verdict([x["verdict"] for x in stages if x["verdict"] in ("pass", "warn", "fail")])
    return {"title": f"Geometry and mesh report: {s.template.title if s.template else 'no geometry chosen'}",
            "overall": overall, "stages": stages, "answers": s.answers,
            "sources": ["Roache, Verification and Validation in Computational Science and Engineering (1998)",
                        "Celik et al., J. Fluids Eng. 130(7):078001 (2008), grid-convergence procedure",
                        "White, Fluid Mechanics (skin-friction correlation for the first-cell height)",
                        "Menter, AIAA J. 32(8) (1994), k-omega SST (y+ ~ 1 without wall functions)"]}


_CSS = """body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f7f8fa;color:#1d2330}
main{max-width:1000px;margin:0 auto;padding:24px 16px}h1{font-size:1.6rem}h2{margin-top:2rem}
.v{display:inline-block;padding:1px 8px;border-radius:9px;font-size:.8rem;font-weight:600}
.pass{background:#dff3e4;color:#155d2b}.warn{background:#fff1cc;color:#7a5200}.fail{background:#fde0e0;color:#8a1c1c}
.not{background:#e8eaef;color:#4a5160}table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.5rem 0}
td,th{border-bottom:1px solid #e1e4ea;padding:4px 6px;text-align:left;vertical-align:top}
section{background:#fff;border:1px solid #e1e4ea;border-radius:10px;padding:12px 16px;margin:12px 0}
code{font-size:.85rem}ul{margin:.3rem 0}"""


def _badge(v: str) -> str:
    cls = v if v in ("pass", "warn", "fail") else "not"
    return f'<span class="v {cls}">{html.escape(v)}</span>'


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    if isinstance(x, (list, tuple)):
        return ", ".join(_fmt(i) for i in x)
    return html.escape(str(x))


def render_html(rep: dict[str, Any]) -> str:
    e = html.escape
    parts = [f"<!doctype html><meta charset='utf-8'><title>{e(rep['title'])}</title><style>{_CSS}</style><main>",
             f"<h1>{e(rep['title'])}</h1><p>Overall verdict {_badge(rep['overall'])}</p>",
             "<table><tr><th>stage</th><th>verdict</th><th>note</th></tr>"]
    parts += [f"<tr><td>{e(s['stage'])}</td><td>{_badge(s['verdict'])}</td><td>{e(s.get('note', ''))}</td></tr>"
              for s in rep["stages"]]
    parts.append("</table>")
    for s in rep["stages"]:
        parts.append(f"<section><h2>{e(s['stage'].capitalize())} {_badge(s['verdict'])}</h2>")
        ev = s.get("evidence")
        if s["stage"] == "problem":
            parts.append(f"<p><em>{e(ev['text'])}</em></p><table><tr><th>fact</th><th>value</th><th>source</th></tr>")
            parts += [f"<tr><td>{e(k)}</td><td>{_fmt(v['value'])}</td><td>{e(v['source'])}</td></tr>"
                      for k, v in ev["facts"].items()]
            parts.append("</table>")
        elif s["stage"] == "geometry":
            parts.append(f"<p><b>{e(ev['title'])}</b> ({e(ev['template'])}). {e(ev.get('why') or '')}</p><ul>")
            parts += [f"<li>{e(r['citation'])}: {e(r['what'])}</li>" for r in ev["references"]]
            parts.append("</ul>")
        elif s["stage"] == "parameters":
            parts.append("<table><tr><th>parameter</th><th>base</th><th>range</th><th>varied</th><th>why</th></tr>")
            parts += [f"<tr><td>{e(p['name'])} [{e(p['unit'])}]</td><td>{_fmt(p['base'])}</td><td>{_fmt(p['range'])}"
                      f"</td><td>{'yes' if p['vary'] else 'no'}</td><td>{e(p.get('reason') or p.get('why', ''))}"
                      f"</td></tr>" for p in ev["parameters"]]
            parts.append("</table>")
        elif s["stage"] == "cad":
            for v in ev:
                parts.append(f"<h3>Variant {v['variant']} {_badge(v['verdict'])}</h3><p><code>{e(v['step'])}</code></p>"
                             "<table><tr><th>check</th><th>status</th><th>value</th><th>limit</th><th>note</th></tr>")
                parts += [f"<tr><td>{e(c['check'])}</td><td>{_badge(c['status'])}</td><td>{_fmt(c['value'])}</td>"
                          f"<td>{_fmt(c['limit'])}</td><td>{e(c.get('note', ''))}</td></tr>" for c in v["checks"]]
                parts.append("</table><ul>")
                parts += [f"<li>{e(c['reference'])}: <b>{e(c['status'])}</b>"
                          + (f" (largest difference {c['max_rel_difference']:.1%})" if "max_rel_difference" in c else "")
                          + "</li>" for c in v["comparison"]]
                parts.append("</ul>")
        elif s["stage"] == "mesh plan":
            for i, p in ev.items():
                parts.append(f"<h3>Variant {e(str(i))}: {e(p['tool'])}, {e(p['elements'])}</h3>"
                             f"<p>about {p['estimated_cells']:,} cells; {e(p['estimated_cost'])}</p><ul>")
                parts += [f"<li>{e(r)}</li>" for r in p["rationale"]]
                parts += [f"<li><b>warning:</b> {e(w)}</li>" for w in p["warnings"]]
                parts.append("</ul>")
        elif s["stage"] == "meshes":
            parts.append("<table><tr><th>variant</th><th>size x</th><th>tool</th><th>cells</th><th>verdict</th>"
                         "<th>checks</th></tr>")
            for m in ev:
                chk = "; ".join(f"{c.get('key', c.get('check', '?'))}: {c.get('status')}"
                                for c in (m["quality"].get("checks") or []))
                parts.append(f"<tr><td>{m['variant']}</td><td>{m['level']:g}</td><td>{e(m['tool'])}</td>"
                             f"<td>{m['cells']:,}</td><td>{_badge(m['verdict'])}</td><td>{e(m['error'] or chk)}</td></tr>")
            parts.append("</table>")
        elif s["stage"] == "grid convergence" and isinstance(ev, dict):
            parts.append("<table><tr><th>quantity</th><th>values (fine, medium, coarse)</th><th>p</th>"
                         "<th>extrapolated</th><th>GCI fine</th><th>note</th></tr>")
            parts += [f"<tr><td>{e(k)}</td><td>{_fmt(c['values'])}</td><td>{_fmt(c.get('p', ''))}</td>"
                      f"<td>{_fmt(c.get('extrapolated', ''))}</td><td>{_fmt(c.get('gci_fine', ''))}</td>"
                      f"<td>{e(c.get('note', ''))}</td></tr>" for k, c in ev.items()]
            parts.append("</table>")
        if s.get("note") and s["stage"] not in ("mesh plan",):
            parts.append(f"<p>{e(s['note'])}</p>")
        parts.append("</section>")
    parts.append("<h2>Sources of the rules</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in rep["sources"]) + "</ul>")
    parts.append("</main>")
    return "".join(parts)


def write_report(rep: dict[str, Any], stem: str) -> dict[str, str]:
    with open(stem + ".json", "w") as f:
        json.dump(rep, f, indent=1, default=str)
    with open(stem + ".html", "w") as f:
        f.write(render_html(rep))
    return {"json": stem + ".json", "html": stem + ".html"}
