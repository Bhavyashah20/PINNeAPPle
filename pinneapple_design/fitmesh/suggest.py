"""Rank geometry templates for a problem and propose the parameters to vary, their ranges and the samples."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .catalog import TEMPLATES, Template
from .context import ProblemContext, _has


def suggest_geometries(ctx: ProblemContext, top: int = 3) -> list[dict[str, Any]]:
    """Templates ranked by physics match and keyword hits, each with why it was picked and its references."""
    t = " " + ctx.text.lower() + " "
    rows = []
    for tpl in TEMPLATES.values():
        phys = [p for p in tpl.physics if p in ctx.physics]
        hits = [w for w in tpl.keywords if _has(t, (w,))]
        qty = [q for q in ctx.get("quantities", []) if q in tpl.quantities]
        score = 3.0 * len(phys) + 2.0 * len(hits) + 1.0 * len(qty)
        if score <= 0:
            continue
        why = []
        if phys:
            why.append("physics: " + ", ".join(phys))
        if hits:
            why.append("mentions: " + ", ".join(sorted(set(hits))[:5]))
        if qty:
            why.append("reference data for: " + ", ".join(qty))
        rows.append({"template": tpl.key, "title": tpl.title, "score": score, "why": "; ".join(why),
                     "description": tpl.description,
                     "references": [{"citation": r.citation, "what": r.what} for r in tpl.references]})
    rows.sort(key=lambda r: -r["score"])
    return rows[:top]


def scale_to_context(tpl: Template, ctx: ProblemContext) -> dict[str, float]:
    """Reference values, scaled so the reference length matches the size given in the problem (if any).

    Scaling is geometric (every length parameter by the same factor); angles, ratios and counts stay."""
    v = tpl.defaults()
    L = ctx.get("length")
    if L is None:
        return v
    k = float(L) / tpl.ref_length(v)
    return {p.name: (v[p.name] * k if p.unit == "m" else v[p.name]) for p in tpl.params}


def suggest_parameters(tpl: Template, ctx: ProblemContext, base: dict[str, float] | None = None
                       ) -> list[dict[str, Any]]:
    """Every parameter with a vary / keep recommendation, a proposed range and the reason.

    High-sensitivity parameters are proposed for variation over their typical range (scaled like the base); the
    others are kept at the base value unless the user selects them. Ranges never leave the validity limits."""
    base = base or scale_to_context(tpl, ctx)
    ref = tpl.defaults()
    out = []
    for p in tpl.params:
        k = base[p.name] / ref[p.name] if (p.unit == "m" and ref[p.name]) else 1.0
        lo, hi = p.typical[0] * k, p.typical[1] * k
        if p.unit == "m":                     # sizes: stay within a factor 2 of the base, so the other lengths
            lo, hi = max(lo, base[p.name] / 2), min(hi, base[p.name] * 2)      # (development runs...) still fit
        lo, hi = max(lo, p.limits[0]), min(hi, p.limits[1])
        vary = p.sensitivity == "high"
        reason = p.role + (" (high influence on " + ", ".join(tpl.quantities[:2]) + ")" if vary else "")
        out.append({"name": p.name, "unit": p.unit, "base": base[p.name], "range": [lo, hi], "limits": list(p.limits),
                    "vary": vary, "sensitivity": p.sensitivity, "reason": reason})
    return out


def sample(tpl: Template, base: dict[str, float], ranges: dict[str, Sequence[float]], n: int, seed: int = 0,
           include_base: bool = True) -> list[dict[str, float]]:
    """Latin hypercube over the selected ranges (scipy.stats.qmc); integer parameters are rounded.

    The base (reference) design comes first when ``include_base``, so every study keeps one variant that can be
    compared with its published reference."""
    from scipy.stats import qmc

    names = list(ranges)
    variants = [tpl.complete(base)] if include_base else []
    m = n - len(variants)
    if m > 0 and names:
        u = qmc.LatinHypercube(d=len(names), seed=seed).random(m)
        lo = np.array([ranges[k][0] for k in names], float)
        hi = np.array([ranges[k][1] for k in names], float)
        for row in qmc.scale(u, lo, hi) if np.all(hi > lo) else np.tile(lo, (m, 1)):
            v = dict(base)
            for k, x in zip(names, row, strict=True):
                p = tpl.param(k)
                v[k] = float(round(x)) if p.unit == "-" and float(p.value).is_integer() else float(x)
            variants.append(tpl.complete(v))
    return variants
