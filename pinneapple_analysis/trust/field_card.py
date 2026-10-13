"""Trust card of a field on a grid: the physics checks a result must pass before anyone uses it.

``trust_card`` takes a field sampled on a tensor-product grid (from any source: a PINN, a solver, a file written
by another code) and checks it with NumPy only:

* sanity: finite values, values inside the physical range;
* PDE residual: ``laplace(u) = f`` (``"laplace"`` / ``"poisson"``) or ``u_t = alpha laplace(u) + f`` (``"heat"``),
  by finite differences on the interior of the grid;
* boundary conditions: Dirichlet values on named faces (``"x_min"``, ``"y_max"``, ...);
* conservation: the divergence theorem, net flux through the boundary against the integral of the source
  (steady problems);
* maximum principle (steady problems without a source): no interior value outside the boundary range;
* reference: relative L2 error against a reference field or a closed form;
* applicability (OOD): the grid, or the query points, inside the bounds the model was trained on.

The checks feed two existing views: the six questions of the lab trust card
(``pinneapple_lab.curation.trust_stages``) and the deploy decision of ``TrustReport`` (``APPROVED``, ``REVIEW``,
``REJECT``; a failing critical check rejects).

>>> import numpy as np
>>> from pinneapple_analysis.trust.field_card import trust_card
>>> x = y = np.linspace(0, 1, 41)
>>> X, Y = np.meshgrid(x, y, indexing="ij")
>>> u = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)
>>> card = trust_card({"x": x, "y": y}, u, pde="laplace",
...                   dirichlet={"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": lambda P: np.sin(np.pi * P[:, 0])})
>>> card.decision
'APPROVED'
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

__all__ = ["FieldCheck", "FieldTrustCard", "field_checks", "trust_card", "DEFAULT_TOLERANCES"]

DEFAULT_TOLERANCES = {
    "residual": 0.05,      # rms PDE residual / rms of the sum of the absolute terms of the equation
    "bc": 0.02,            # max |u - g| on a face / range of u
    "conservation": 0.05,  # |net boundary flux - integral of f| / total absolute flux
    "max_principle": 0.01,  # overshoot beyond the boundary range / range of u
    "reference": 0.05,     # relative L2 error against the reference
    "ood": 1.0,            # fraction of query points that must lie inside the training bounds
}

Value = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]  # noqa: UP007 (runtime alias, Python 3.10)


@dataclass
class FieldCheck:
    """One check: ``kind`` is ``sanity``, ``physics``, ``reference`` or ``generalization`` (the lab vocabulary);
    ``passed`` is ``None`` when the check could not run, with the reason in ``detail``."""
    name: str
    kind: str
    passed: bool | None
    value: float | None = None
    threshold: float | None = None
    critical: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "passed": self.passed, "value": self.value,
                "threshold": self.threshold, "critical": self.critical, "detail": self.detail}

    def line(self) -> str:
        st = {True: "PASS", False: "FAIL", None: "N/A "}[self.passed]
        v = "" if self.value is None else f" {self.value:.3g}"
        t = "" if self.threshold is None else f" (limit {self.threshold:g})"
        return f"[{st}] {self.name}:{v}{t}" + (f" -- {self.detail}" if self.detail else "")


@dataclass
class FieldTrustCard:
    checks: list[FieldCheck]
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    decision: str = "REVIEW"
    reasons: list[str] = field(default_factory=list)
    name: str = "field"

    @property
    def failed(self) -> list[FieldCheck]:
        return [c for c in self.checks if c.passed is False]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "decision": self.decision, "reasons": self.reasons,
                "checks": [c.to_dict() for c in self.checks], "stages": self.stages}

    def to_markdown(self) -> str:
        L = [f"# Trust card: {self.name}", "", f"**Decision:** {self.decision}", "",
             "| check | kind | status | value | limit | detail |", "|---|---|---|---|---|---|"]
        for c in self.checks:
            st = {True: "pass", False: "FAIL", None: "not run"}[c.passed]
            L.append(f"| {c.name}{' (critical)' if c.critical else ''} | {c.kind} | {st} | "
                     f"{'' if c.value is None else f'{c.value:.3g}'} | "
                     f"{'' if c.threshold is None else f'{c.threshold:g}'} | {c.detail} |")
        if self.stages:
            L += ["", "| stage | question | status | next |", "|---|---|---|---|"]
            for s in self.stages.values():
                L.append(f"| {s['title']} | {s['question']} | {s['status']} | {s.get('next', '')} |")
        if self.reasons:
            L += ["", "Reasons:"] + [f"- {r}" for r in self.reasons]
        return "\n".join(L) + "\n"

    def __str__(self) -> str:
        return "\n".join([f"Trust card '{self.name}': {self.decision}"] + ["  " + c.line() for c in self.checks])


# ---------------------------------------------------------------------- helpers
def _axes(grid: Mapping[str, Sequence[float]]) -> tuple[list[str], list[np.ndarray]]:
    names = list(grid)
    axes = [np.asarray(grid[n], dtype=np.float64).ravel() for n in names]
    for n, a in zip(names, axes, strict=True):
        if a.size < 2 or np.any(np.diff(a) <= 0):
            raise ValueError(f"grid axis '{n}' must be strictly increasing with at least 2 points")
    return names, axes


def _points(axes: Sequence[np.ndarray]) -> np.ndarray:
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def _evaluate(v: Value, pts: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if callable(v):
        out = np.asarray(v(pts), dtype=np.float64)
    else:
        out = np.asarray(v, dtype=np.float64)
        if out.ndim == 0:
            return np.full(shape, float(out))
    return out.reshape(shape)


def _face(names: Sequence[str], axes: Sequence[np.ndarray], face: str) -> tuple[int, int]:
    coord, _, side = face.rpartition("_")
    if coord not in names or side not in ("min", "max"):
        raise ValueError(f"face '{face}' must be '<coord>_min' or '<coord>_max' with coord in {list(names)}")
    a = list(names).index(coord)
    return a, 0 if side == "min" else len(axes[a]) - 1


def _trapz(f: np.ndarray, axes: Sequence[np.ndarray]) -> float:
    out = f
    for ax in reversed(range(len(axes))):
        out = np.trapezoid(out, axes[ax], axis=ax) if hasattr(np, "trapezoid") else np.trapz(out, axes[ax], axis=ax)
    return float(out)


def _d2(u: np.ndarray, x: np.ndarray, axis: int) -> np.ndarray:
    """Second derivative along ``axis`` by the three-point formula (non-uniform spacing allowed); zero on the two
    end planes, which the residual check leaves out."""
    u = np.moveaxis(u, axis, 0)
    h = np.diff(x).reshape((-1,) + (1,) * (u.ndim - 1))
    out = np.zeros_like(u)
    out[1:-1] = 2.0 * ((u[2:] - u[1:-1]) / h[1:] - (u[1:-1] - u[:-2]) / h[:-1]) / (h[1:] + h[:-1])
    return np.moveaxis(out, 0, axis)


def _interior(ndim: int, keep_ends: Sequence[int] = ()) -> tuple[slice, ...]:
    # one layer off each face (the three-point stencil); a time axis keeps its last plane
    return tuple(slice(1, None) if a in keep_ends else slice(1, -1) for a in range(ndim))


# ---------------------------------------------------------------------- the checks
def field_checks(grid: Mapping[str, Sequence[float]], values: Any, *, pde: str | None = None,
                 source: Value | None = None, alpha: float = 1.0, time_coord: str = "t",
                 dirichlet: Mapping[str, Value] | None = None,
                 reference: Value | None = None, value_range: tuple[float, float] | None = None,
                 train_bounds: Mapping[str, tuple[float, float]] | None = None,
                 query_points: np.ndarray | None = None,
                 tolerances: Mapping[str, float] | None = None) -> list[FieldCheck]:
    """The checks of one scalar field ``values`` (shape ``(len(axis_0), len(axis_1), ...)``) on ``grid``
    (``{coord: axis}`` in the order of the array dimensions). Every argument after ``values`` is optional; a check
    without its input is reported as not run, with the reason, never silently dropped."""
    tol = {**DEFAULT_TOLERANCES, **dict(tolerances or {})}
    names, axes = _axes(grid)
    shape = tuple(len(a) for a in axes)
    u = np.asarray(values, dtype=np.float64)
    if u.size != int(np.prod(shape)):
        raise ValueError(f"values has {u.size} entries, the grid {dict(zip(names, shape, strict=True))} has {int(np.prod(shape))}")
    u = u.reshape(shape)
    pts = _points(axes)
    nd = len(axes)
    out: list[FieldCheck] = []

    finite = bool(np.isfinite(u).all())
    out.append(FieldCheck("finite values", "sanity", finite, value=float(np.mean(~np.isfinite(u))), threshold=0.0,
                          critical=True, detail="" if finite else "NaN or Inf in the field"))
    if not finite:
        return out
    urange = float(np.ptp(u)) or max(float(np.abs(u).max()), 1e-12)

    if value_range is not None:
        lo, hi = value_range
        over = float(max(lo - u.min(), u.max() - hi, 0.0)) / urange
        out.append(FieldCheck("values inside the physical range", "sanity", over <= tol["max_principle"], value=over,
                              threshold=tol["max_principle"], detail=f"range [{u.min():.4g}, {u.max():.4g}] vs "
                                                                     f"[{lo:g}, {hi:g}]"))

    # ---- PDE residual
    kind = (pde or "").lower()
    steady = kind in ("laplace", "poisson")
    if kind in ("laplace", "poisson", "heat"):
        t_ax = names.index(time_coord) if kind == "heat" and time_coord in names else None
        if kind == "heat" and t_ax is None:
            out.append(FieldCheck("PDE residual", "physics", None, detail=f"no time axis '{time_coord}' in the grid"))
        elif min(shape) < 4:
            out.append(FieldCheck("PDE residual", "physics", None, detail="needs at least 4 points per axis"))
        else:
            f = np.zeros(shape) if source is None else _evaluate(source, pts, shape)
            space = [a for a in range(nd) if a != t_ax]
            d2 = [_d2(u, axes[a], a) for a in space]
            lap = np.sum(d2, axis=0)
            if t_ax is None:
                r, terms = lap - f, d2 + [f]
            else:
                ut = np.gradient(u, axes[t_ax], axis=t_ax, edge_order=2)
                r, terms = ut - alpha * lap - f, [ut] + [alpha * t for t in d2] + [f]
            sl = _interior(nd, () if t_ax is None else (t_ax,))
            # relative to the size of the terms that must balance (each second derivative, the source, u_t): a
            # harmonic field has large u_xx and u_yy that cancel, so the residual is judged against them
            scale = float(np.sqrt(np.mean(np.sum([np.abs(t[sl]) for t in terms], axis=0) ** 2)))
            scale = max(scale, 1e-12 * urange)
            rel = float(np.sqrt(np.mean(r[sl] ** 2))) / scale
            eq = "laplace(u) = f" if t_ax is None else f"u_{time_coord} = {alpha:g} laplace(u) + f"
            out.append(FieldCheck("PDE residual", "physics", rel <= tol["residual"], value=rel,
                                  threshold=tol["residual"], critical=True,
                                  detail=f"{eq}, three-point finite differences, rms over the interior / rms of the sum of |terms|"))
    else:
        out.append(FieldCheck("PDE residual", "physics", None,
                              detail="no PDE given (pde='laplace' | 'poisson' | 'heat')" if not kind
                              else f"PDE '{pde}' is not supported by the grid check"))

    # ---- boundary conditions
    if dirichlet:
        for face, g in dirichlet.items():
            a, idx = _face(names, axes, face)
            fu = np.take(u, idx, axis=a)
            fp = _points([axes[i] if i != a else axes[a][idx:idx + 1] for i in range(nd)])
            gv = _evaluate(g, fp, fu.shape)
            err = float(np.abs(fu - gv).max()) / urange
            out.append(FieldCheck(f"boundary condition on {face}", "physics", err <= tol["bc"], value=err,
                                  threshold=tol["bc"], critical=True, detail="Dirichlet, max |u - g| / range of u"))
    else:
        out.append(FieldCheck("boundary conditions", "physics", None, detail="no Dirichlet data given"))

    # ---- conservation (divergence theorem) and maximum principle, steady problems
    if steady:
        f = np.zeros(shape) if source is None else _evaluate(source, pts, shape)
        net, total = 0.0, 0.0
        for a, x in enumerate(axes):
            g = np.gradient(u, x, axis=a, edge_order=2)
            others = [axes[i] for i in range(nd) if i != a]
            for idx, sign in ((0, -1.0), (len(x) - 1, 1.0)):
                fl = sign * np.take(g, idx, axis=a)
                net += _trapz(fl, others) if others else float(fl)
                total += _trapz(np.abs(fl), others) if others else float(abs(fl))
        src = _trapz(f, axes)
        if total + abs(src) > 1e-12:
            rel = abs(net - src) / (total + abs(src))
            out.append(FieldCheck("conservation (net boundary flux = integral of the source)", "physics",
                                  rel <= tol["conservation"], value=rel, threshold=tol["conservation"],
                                  detail=f"flux {net:.4g}, source {src:.4g}"))
        if source is None or not np.any(f):
            bmask = np.zeros(shape, dtype=bool)
            for a in range(nd):
                idx = [slice(None)] * nd
                idx[a] = 0
                bmask[tuple(idx)] = True
                idx[a] = -1
                bmask[tuple(idx)] = True
            bl, bh = u[bmask].min(), u[bmask].max()
            over = float(max(bl - u[~bmask].min(), u[~bmask].max() - bh, 0.0)) / urange if (~bmask).any() else 0.0
            out.append(FieldCheck("maximum principle (no interior extremum)", "physics", over <= tol["max_principle"],
                                  value=over, threshold=tol["max_principle"],
                                  detail="harmonic field: interior values within the boundary range"))

    # ---- reference
    if reference is not None:
        ref = _evaluate(reference, pts, shape)
        rel = float(np.linalg.norm(u - ref) / max(np.linalg.norm(ref), 1e-300))
        out.append(FieldCheck("reference (relative L2)", "reference", rel <= tol["reference"], value=rel,
                              threshold=tol["reference"], critical=True, detail="against the given reference"))
    else:
        out.append(FieldCheck("reference (relative L2)", "reference", None, detail="no reference given"))

    # ---- applicability
    if train_bounds:
        q = pts if query_points is None else np.asarray(query_points, dtype=np.float64)
        inside = np.ones(len(q), dtype=bool)
        for i, n in enumerate(names):
            if n in train_bounds:
                lo, hi = train_bounds[n]
                inside &= (q[:, i] >= lo - 1e-12) & (q[:, i] <= hi + 1e-12)
        frac = float(inside.mean())
        out.append(FieldCheck("applicability: query inside the training bounds (OOD)", "generalization",
                              frac >= tol["ood"], value=frac, threshold=tol["ood"],
                              detail=f"{int((~inside).sum())} of {len(q)} points outside {dict(train_bounds)}"))
    else:
        out.append(FieldCheck("applicability: query inside the training bounds (OOD)", "generalization", None,
                              detail="no training bounds given"))
    return out


def trust_card(grid: Mapping[str, Sequence[float]], values: Any, *, name: str = "field", is_ml: bool = True,
               limitations: Sequence[str] = (), min_coverage: float = 0.5, **kwargs: Any) -> FieldTrustCard:
    """Run ``field_checks`` and turn them into a trust card: the deploy decision (``TrustReport``: a failing
    critical check rejects, any other failure or low coverage asks for review) and the six questions of the lab
    trust card. ``kwargs`` go to ``field_checks``."""
    from pinneapple_analysis.trust.trust_report import (
        CONTRADICTS,
        NOT_RUN,
        SUPPORTS,
        Check,
        TrustReport,
    )

    checks = field_checks(grid, values, **kwargs)
    status = {True: SUPPORTS, False: CONTRADICTS, None: NOT_RUN}
    rep = TrustReport([Check(c.name, status[c.passed], critical=c.critical, value=c.value, limit=c.threshold,
                             reason=c.detail if c.passed is None else "") for c in checks],
                      min_coverage=min_coverage, model=name)
    stages: dict[str, dict[str, Any]] = {}
    try:
        from pinneapple_lab.curation import trust_stages
        failed_critical = any(c.critical and c.passed is False for c in checks)
        tier = "D" if failed_critical else ("C" if any(c.passed is False for c in checks) else "B")
        lab = [{"name": c.name, "passed": bool(c.passed), "detail": c.detail[:120], "kind": c.kind}
               for c in checks if c.passed is not None]
        with tempfile.TemporaryDirectory() as d:
            stages = trust_stages(os.fspath(d), lab, {}, {}, tier, {}, {}, is_ml, list(limitations))
    except Exception:  # noqa: BLE001 - the decision does not depend on the lab view
        stages = {}
    reasons = [f"{c.name}: failed" + (" (critical)" if c.critical else "") for c in checks if c.passed is False]
    if rep.cobertura < min_coverage:
        reasons.append(f"only {rep.cobertura:.0%} of the checks could run (minimum {min_coverage:.0%})")
    return FieldTrustCard(checks, stages, rep.decisao, reasons, name)
