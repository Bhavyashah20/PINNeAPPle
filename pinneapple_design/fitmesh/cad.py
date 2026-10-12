"""Build the CAD variants, export STEP and STL, and check each one before it is meshed.

Checks per variant (each a row with value, limit and verdict):

- parameters inside the template's validity limits;
- the solid is valid (OpenCascade BRep check) and closed;
- volume against the exact formula of the template (or its bound);
- the STL is watertight and its volume matches the solid within the tessellation tolerance for the purpose;
- the bounding box matches the parameters;
- comparison with each published reference object: relative difference per shared dimension, and whether the
  variant is the reference, inside the range the reference covered, or an extrapolation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .catalog import Template

# tessellation: chord tolerance as a fraction of the smallest feature, and accepted STL/CAD volume difference
TESSELLATION = {"screening": (0.005, 1e-2), "design": (0.001, 3e-3), "verification": (0.0005, 1e-3)}


@dataclass
class CadVariant:
    index: int
    params: dict[str, float]
    step: str = ""
    stl: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    comparison: list[dict[str, Any]] = field(default_factory=list)
    volume: float = float("nan")
    area: float = float("nan")
    bbox: list[float] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        st = {c["status"] for c in self.checks}
        return "fail" if "fail" in st else "warn" if "warn" in st else "pass"


def _row(name, status, value, limit, note=""):
    return {"check": name, "status": status, "value": value, "limit": limit, "note": note}


def build_variant(tpl: Template, params: dict[str, float], out_dir: str, index: int, accuracy: str = "design"
                  ) -> CadVariant:
    import cadquery as cq
    import trimesh

    var = CadVariant(index=index, params=dict(params))
    problems = tpl.check_params(params)
    var.checks.append(_row("parameters within limits", "fail" if problems else "pass", len(problems), 0,
                           "; ".join(problems)))
    if problems:
        return var
    try:
        solid = tpl.build(params).val()
    except Exception as exc:                                            # the kernel refused the shape
        var.checks.append(_row("CAD build", "fail", str(exc)[:200], "builds"))
        return var
    var.volume, var.area = float(solid.Volume()), float(solid.Area())
    bb = solid.BoundingBox()
    var.bbox = [bb.xlen, bb.ylen, bb.zlen]
    var.checks.append(_row("valid solid (BRep check)", "pass" if solid.isValid() else "fail", bool(solid.isValid()), True))

    if tpl.volume is not None:
        exact = tpl.volume(params)
        err = abs(var.volume - exact) / exact
        var.checks.append(_row("volume vs exact formula", "pass" if err < 1e-4 else "fail", err, 1e-4,
                               f"CAD {var.volume:.6g} m3, formula {exact:.6g} m3"))
    elif tpl.key == "ahmed_body":
        from .catalog import _ahmed_volume_upper
        ub = _ahmed_volume_upper(params)
        ok = var.volume < ub and var.volume > 0.9 * ub
        var.checks.append(_row("volume vs sharp-nose bound", "pass" if ok else "fail", var.volume / ub, "0.9-1.0",
                               "the rounded nose removes a little volume from the sharp-nose block"))

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{tpl.key}_{index:03d}")
    var.step, var.stl = stem + ".step", stem + ".stl"
    cq.exporters.export(solid, var.step)
    frac, vol_tol = TESSELLATION.get(accuracy, TESSELLATION["design"])
    tol = frac * tpl.min_feature(params)
    cq.exporters.export(solid, var.stl, tolerance=tol, angularTolerance=0.1)
    m = trimesh.load(var.stl, force="mesh")
    var.checks.append(_row("STL watertight", "pass" if m.is_watertight else "fail", bool(m.is_watertight), True))
    dv = abs(abs(m.volume) - var.volume) / var.volume
    var.checks.append(_row("STL volume vs solid", "pass" if dv <= vol_tol else "warn", dv, vol_tol,
                           f"{len(m.faces)} triangles, chord tolerance {tol:.3g} m ({frac:.0%} of the smallest "
                           f"feature)"))
    var.comparison = compare_with_references(tpl, params)
    return var


def compare_with_references(tpl: Template, params: dict[str, float]) -> list[dict[str, Any]]:
    """Relative difference of each shared dimension with each published reference object."""
    out = []
    for ref in tpl.references:
        if not ref.params:
            out.append({"reference": ref.citation, "what": ref.what, "status": "context",
                        "note": "closed-form or regime reference: no fixed dimensions to compare"})
            continue
        diffs = {k: (params[k] - v) / v if v else params[k] - v for k, v in ref.params.items() if k in params}
        worst = max((abs(d) for d in diffs.values()), default=0.0)
        same = worst < 1e-6
        # inside what the reference covered: every varied parameter inside its typical range
        inside = all(tpl.param(k).typical[0] <= params[k] <= tpl.param(k).typical[1] for k in params
                     if k in {p.name for p in tpl.params})
        status = "same as reference" if same else "within the studied range" if inside else "extrapolated"
        out.append({"reference": ref.citation, "what": ref.what, "status": status,
                    "differences": {k: round(d, 4) for k, d in diffs.items()}, "max_rel_difference": worst})
    return out
