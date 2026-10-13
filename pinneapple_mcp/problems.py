"""A physics problem from JSON: a preset, or a box problem described field by field.

Preset::

    {"preset": "burgers_1d", "preset_kwargs": {"nu": 0.003}}

Box problem (any PDE kind the physics compiler knows; values and sources are expressions of the coordinates)::

    {"name": "plate", "domain": {"x": [0, 1], "y": [0, 1]}, "fields": ["T"], "pde": "laplace",
     "params": {}, "source": "0",
     "bcs": [{"kind": "dirichlet", "field": "T", "on": "y_max", "value": "sin(pi*x)"},
             {"kind": "dirichlet", "field": "T", "on": "y_min", "value": 0},
             {"kind": "dirichlet", "field": "T", "on": "x_boundary", "value": 0}],
     "exact": "sin(pi*x)*sinh(pi*y)/sinh(pi)"}

``on`` is ``"<coord>_min"``, ``"<coord>_max"``, ``"<coord>_boundary"`` (both ends) or ``"boundary"`` (every face).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .expressions import compile_expression
from .expressions import value as _value

STEADY_CHECKED = {"laplace": "laplace", "poisson": "poisson", "heat_equation": "heat"}


@dataclass
class BuiltProblem:
    problem: Any                                  # pinneapple PhysicalProblem
    ctx: dict[str, Any] = field(default_factory=dict)
    exact: Callable[[np.ndarray], np.ndarray] | None = None
    dirichlet: dict[str, Any] = field(default_factory=dict)   # face -> number or callable, for the BC checks
    source: Callable | None = None
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def check_pde(self) -> str | None:
        kind = getattr(self.problem.pde, "kind", None)
        return STEADY_CHECKED.get(kind)

    @property
    def alpha(self) -> float:
        p = dict(getattr(self.problem.pde, "params", {}) or {})
        return float(p.get("alpha", p.get("kappa", 1e-3)))


def _faces(on: str, coords: list[str]) -> list[str]:
    if on == "boundary":
        return [f"{c}_{s}" for c in coords for s in ("min", "max")]
    if on.endswith("_boundary") and on[: -len("_boundary")] in coords:
        c = on[: -len("_boundary")]
        return [f"{c}_min", f"{c}_max"]
    if on.rpartition("_")[0] in coords and on.rpartition("_")[2] in ("min", "max"):
        return [on]
    return []


def build_problem(spec: Mapping[str, Any]) -> BuiltProblem:
    """``BuiltProblem`` from the JSON description above. Raises ``ValueError`` with the reason on bad input."""
    import pinneapple as pp

    if not isinstance(spec, Mapping):
        raise ValueError("problem must be an object: {'preset': name} or a box problem description")
    if "preset" in spec:
        name = str(spec["preset"])
        if name not in pp.list_presets():
            raise ValueError(f"unknown preset '{name}'; see list_presets")
        prob = pp.PhysicalProblem.from_preset(name, **dict(spec.get("preset_kwargs") or {}))
        built = BuiltProblem(prob, spec=dict(spec))
        coords = list(prob.coords)
        built.dirichlet = _preset_dirichlet(prob)
        if spec.get("exact"):
            built.exact = compile_expression(str(spec["exact"]), coords)
        return built

    from pinneapple_physics.pde_environment.builder import ProblemBuilder

    domain = spec.get("domain")
    if not isinstance(domain, Mapping) or not domain:
        raise ValueError("a box problem needs 'domain': {coord: [lo, hi], ...}")
    coords = [str(c) for c in domain]
    bounds = {c: (float(domain[c][0]), float(domain[c][1])) for c in coords}
    for c, (lo, hi) in bounds.items():
        if not hi > lo:
            raise ValueError(f"domain of '{c}' must have lo < hi, got {domain[c]}")
    fields = [str(f) for f in (spec.get("fields") or ["u"])]
    kind = str(spec.get("pde") or "")
    if not kind:
        raise ValueError("a box problem needs 'pde' (e.g. 'laplace', 'poisson', 'heat_equation', 'burgers')")
    b = ProblemBuilder(str(spec.get("name") or "problem")).domain(**bounds).fields(*fields)
    b = b.pde(kind, **{str(k): float(v) for k, v in dict(spec.get("params") or {}).items()})
    dirichlet: dict[str, Any] = {}
    for i, bc in enumerate(spec.get("bcs") or []):
        bkind = str(bc.get("kind", "dirichlet"))
        on = str(bc.get("on", "boundary"))
        v = _value(bc.get("value", 0.0), coords)
        fld = str(bc.get("field", fields[0]))
        faces = _faces(on, coords)
        if not faces and on != "boundary" and "=" not in on:
            raise ValueError(f"bcs[{i}].on='{on}' is not a face: use <coord>_min, <coord>_max, <coord>_boundary "
                             "or boundary")
        # the builder's tag selector 'boundary' needs a mesh; spell every face out instead
        for face in faces or [on]:
            b = b.bc(bkind, field=fld, value=v, on=face, weight=float(bc.get("weight", 10.0)))
            if bkind == "dirichlet" and fld == fields[0]:
                dirichlet[face] = v
    for ic in spec.get("ics") or []:
        b = b.ic(field=str(ic.get("field", fields[0])), value=_value(ic.get("value", 0.0), coords))
    prob = pp.PhysicalProblem.from_pde_spec(b.build())
    built = BuiltProblem(prob, dirichlet=dirichlet, spec=dict(spec))
    if spec.get("source") not in (None, "", 0, "0"):
        src = compile_expression(str(spec["source"]), coords)
        built.source = src
        built.ctx["source_fn"] = lambda X, ctx=None, _f=src: _f(X)
    if spec.get("exact"):
        built.exact = compile_expression(str(spec["exact"]), coords)
    return built


def _preset_dirichlet(prob) -> dict[str, Any]:
    """Dirichlet data of a preset per face, where a condition covers a whole face (tag 'boundary': every face)."""
    out: dict[str, Any] = {}
    coords = list(prob.coords)
    if set(prob.domain_bounds) != set(coords):
        return out
    lo = np.array([prob.domain_bounds[c][0] for c in coords])
    hi = np.array([prob.domain_bounds[c][1] for c in coords])
    rng = np.random.default_rng(0)
    for c in prob.conditions:
        if getattr(c, "kind", "") != "dirichlet" or len(getattr(c, "fields", ())) != 1 \
                or c.fields[0] != prob.fields[0]:
            continue
        for a, name in enumerate(coords):
            for side, val in (("min", lo[a]), ("max", hi[a])):
                P = lo + (hi - lo) * rng.random((64, len(coords)))
                P[:, a] = val
                try:
                    if c.selector_type == "tag":
                        hit = (c.selector or {}).get("tag") == "boundary"
                    elif c.selector_type == "all":
                        hit = True
                    else:
                        hit = bool(np.asarray(c.selector(P, {}), dtype=bool).all())
                except Exception:  # noqa: BLE001 - a selector we cannot evaluate is skipped
                    hit = False
                if hit:
                    fn = c.value_fn
                    out[f"{name}_{side}"] = (lambda X, _fn=fn: np.asarray(_fn(X, {}), dtype=np.float64).reshape(
                        len(X), -1)[:, 0])
    return out
