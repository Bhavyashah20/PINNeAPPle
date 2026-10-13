"""Scale-aware model selection: each model declares the scales it resolves, each query the scales it needs.

A model trained or run at grid spacing ``h`` with filter ``G_Delta`` represents ``G_Delta * u`` and nothing finer.
Asked for a field filtered at a smaller width it cannot answer, however small its validation error was; asked for a
larger width it can, by post-filtering its output (:func:`~.filters.refilter`), but only exactly when the requested
filter can be built from its own (a Gaussian from a narrower Gaussian, anything from a sharp cut-off below the
requested scales; a box filter has zeros a post-filter cannot undo). So:

* :class:`ScaleProfile` (per model): output filter (kind, width), grid spacing, closure model, valid Reynolds range,
  time resolution, fidelity rung (0 surrogate, 1 coarse LES with a closure, 2 resolved solver), cost;
* :class:`ScaleQuery` (per query): requested filter (kind, width) of the output, Reynolds number, time scale;
* :func:`scale_compatibility`: 0 (cannot answer: finer than its resolved scale or its time resolution) or
  ``exp(-(mismatch / tol)^2) x Re score``, with the mismatch of :func:`~.filters.realizability_mismatch`;
* :class:`ScaleAwareRouter`: the compatibility as a prior on the ensemble weights, sleeping experts that cannot
  answer, and a fidelity ladder: only the lowest rung that covers the query is awake; it escalates one rung when the
  query needs smaller scales than the rung resolves or when the prediction checks of every model on the rung fail
  (``escalate_from``). Same interface as :class:`~pinneapple_physics.physics_audit.PhysicsAwareRouter`
  (``prior``, ``awake``, ``escalate``, ``route``), so it plugs into
  ``PhysicsEnsemble(router=...)``, and it can wrap such a router (``base=``) to add the assumption audit.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .filters import Filter, realizability_mismatch

__all__ = ["RUNGS", "ScaleProfile", "ScaleQuery", "ScaleCheck", "scale_compatibility", "ScaleRoute",
           "ScaleAwareRouter"]

RUNGS = {0: "surrogate", 1: "coarse LES with closure", 2: "resolved solver"}


@dataclass(frozen=True)
class ScaleProfile:
    """Scales a model represents. ``filter``: the filter its output carries (``Filter("sharp", 0)`` for a resolved
    field); ``grid_spacing``: its output grid; ``re_range``: Reynolds numbers it is valid for (training range of a
    surrogate, resolution limit of a solver); ``time_step``: its time resolution (output interval); ``fidelity``: rung
    in :data:`RUNGS`; ``closure``: the subfilter model (None for surrogates and resolved solvers)."""
    name: str
    filter: Filter = Filter("sharp", 0.0)
    grid_spacing: float = 0.0
    closure: str | None = None
    re_range: tuple[float, float] = (0.0, math.inf)
    time_step: float = 0.0
    fidelity: int = 0
    cost: float | None = None
    notes: str = ""

    @property
    def resolved_width(self) -> float:
        """Smallest width it can deliver: its filter width, or its grid spacing (a sharp cut-off at the grid's
        Nyquist wavenumber), whichever is larger."""
        return max(self.filter.delta, self.grid_spacing)


@dataclass(frozen=True)
class ScaleQuery:
    """Scales a query needs: the filter of the requested output, the Reynolds number of the case and the time scale
    of the requested quantity (None: not constrained). ``kmax``: largest wavenumber of the output grid (where the
    filter comparison stops)."""
    filter: Filter
    re: float | None = None
    time_scale: float | None = None
    kmax: float | None = None


@dataclass
class ScaleCheck:
    name: str
    score: float
    covered: bool
    mismatch: float
    reasons: list[str] = field(default_factory=list)


def _re_score(re: float | None, rng: tuple[float, float], band: float) -> tuple[float, str | None]:
    lo, hi = rng
    if re is None or (lo <= re <= hi):
        return 1.0, None
    d = (math.log10(lo) - math.log10(re)) if re < lo else (math.log10(re) - math.log10(hi) if math.isfinite(hi) else 0.0)
    return float(math.exp(-(d / band) ** 2)), f"Re {re:.3g} outside its range {lo:.3g}..{hi:.3g}"


def scale_compatibility(profile: ScaleProfile, query: ScaleQuery, *, mismatch_tol: float = 0.25,
                        re_band: float = 0.15, slope: float = -2.0) -> ScaleCheck:
    """Can ``profile`` answer ``query``, and how well do its filtering assumptions match? See the module docs."""
    reasons: list[str] = []
    qd = query.filter.delta
    if qd < profile.resolved_width * (1 - 1e-9):
        return ScaleCheck(profile.name, 0.0, False, 1.0,
                          [f"asks for width {qd:.3g}, below its resolved scale {profile.resolved_width:.3g}"])
    if query.time_scale is not None and profile.time_step > 0 and query.time_scale < profile.time_step * (1 - 1e-9):
        return ScaleCheck(profile.name, 0.0, False, 1.0,
                          [f"asks for time scale {query.time_scale:.3g}, below its time step {profile.time_step:.3g}"])
    kmax = query.kmax if query.kmax is not None else (np.pi / qd if qd > 0 else np.pi / max(profile.grid_spacing, 1e-6))
    kmax = max(kmax, np.pi / qd if qd > 0 else 0.0)
    m = realizability_mismatch(profile.filter, query.filter, kmax, slope=slope) if profile.filter.delta > 0 else 0.0
    score = float(math.exp(-(m / mismatch_tol) ** 2))
    if m > 0.02:
        reasons.append(f"{query.filter.kind} filter of width {qd:.3g} only partly reachable from its "
                       f"{profile.filter.kind} {profile.filter.delta:.3g} output (mismatch {m:.2f})")
    rs, why = _re_score(query.re, profile.re_range, re_band)
    if why:
        reasons.append(why)
    return ScaleCheck(profile.name, score * rs, True, m, reasons)


@dataclass
class ScaleRoute:
    chosen: str | None                 # None: not covered by any model
    weights: dict[str, float]
    checks: dict[str, ScaleCheck]
    awake: dict[str, bool]
    rung: int | None
    escalated: bool
    reasons: list[str]

    def __str__(self) -> str:
        return "\n".join([f"chosen: {self.chosen or 'NOT COVERED'} (rung {self.rung})"] + self.reasons)


class ScaleAwareRouter:
    """``profiles``: one :class:`ScaleProfile` per expert (experts without one are always awake, score 1);
    ``query_fn(query) -> ScaleQuery`` (identity by default). ``strength``: exponent of the compatibility in the
    prior; ``abstain_below``: compatibility under which an expert sleeps; ``ladder``: keep only the lowest
    fidelity rung that has an expert with compatibility >= ``rung_threshold`` awake; ``cost_weight``: subtract
    ``cost_weight * cost / max cost`` from the log-prior; ``base``: another router whose prior and awake set are
    combined with these (e.g. the assumption audit)."""

    def __init__(self, profiles: Sequence[ScaleProfile] | Mapping[str, ScaleProfile],
                 query_fn: Callable[[Any], ScaleQuery] | None = None, *, strength: float = 1.0, floor: float = 1e-4,
                 abstain_below: float = 0.05, ladder: bool = True, rung_threshold: float = 0.3,
                 cost_weight: float = 0.0, mismatch_tol: float = 0.25, re_band: float = 0.15, base: Any = None):
        if isinstance(profiles, Mapping):
            profiles = list(profiles.values())
        self.profiles = {p.name: p for p in profiles}
        self.query_fn = query_fn or (lambda q: q)
        self.strength, self.floor, self.abstain_below = float(strength), float(floor), float(abstain_below)
        self.ladder, self.rung_threshold = bool(ladder), float(rung_threshold)
        self.cost_weight, self.mismatch_tol, self.re_band = float(cost_weight), float(mismatch_tol), float(re_band)
        self.base = base
        self._cache: tuple | None = None
        self.log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------------------------------ per case
    def checks(self, query: Any, names: Sequence[str]) -> dict[str, ScaleCheck]:
        sq = self.query_fn(query)
        return {n: (scale_compatibility(self.profiles[n], sq, mismatch_tol=self.mismatch_tol, re_band=self.re_band)
                    if n in self.profiles else ScaleCheck(n, 1.0, True, 0.0)) for n in names}

    def _rung(self, n: str) -> int:
        return self.profiles[n].fidelity if n in self.profiles else 0

    def _evaluate(self, query: Any, names: Sequence[str], excluded: frozenset = frozenset()) -> dict[str, Any]:
        key = (tuple(names), excluded)
        if self._cache is not None and self._cache[0] is query and self._cache[1] == key:
            return self._cache[2]
        ch = self.checks(query, names)
        awake = {n: ch[n].score >= self.abstain_below and n not in excluded for n in names}
        rung, escalated = None, False
        if self.ladder:
            rungs = sorted({self._rung(n) for n in names})
            good = [r for r in rungs if any(awake[n] and ch[n].score >= self.rung_threshold and self._rung(n) == r
                                            for n in names)]
            if good:
                rung = good[0]
                escalated = rung > rungs[0]
                awake = {n: awake[n] and self._rung(n) == rung for n in names}
        if self.base is not None:
            ba = self.base.awake(query, names) if hasattr(self.base, "awake") else {}
            awake = {n: awake[n] and bool(ba.get(n, True)) for n in names}
        costs = np.array([self.profiles[n].cost if n in self.profiles and self.profiles[n].cost is not None else np.nan
                          for n in names], dtype=float)
        cmax = np.nanmax(costs) if np.any(np.isfinite(costs)) else 1.0
        lp = {}
        bp = self.base.prior(query, names) if self.base is not None else {}
        for k, n in enumerate(names):
            v = self.strength * math.log(max(ch[n].score, self.floor)) + float(bp.get(n, 0.0))
            if self.cost_weight > 0 and np.isfinite(costs[k]):
                v -= self.cost_weight * costs[k] / max(cmax, 1e-300)
            lp[n] = v
        out = {"checks": ch, "awake": awake, "rung": rung, "escalated": escalated, "log_prior": lp,
               "covered": any(c.covered for c in ch.values())}
        self._cache = (query, key, out)
        return out

    def prior(self, query: Any, names: Sequence[str] | None = None) -> dict[str, float]:
        return dict(self._evaluate(query, list(names or self.profiles))["log_prior"])

    def awake(self, query: Any, names: Sequence[str] | None = None) -> dict[str, bool]:
        return dict(self._evaluate(query, list(names or self.profiles))["awake"])

    def escalate(self, query: Any, names: Sequence[str] | None = None) -> bool:
        return bool(self._evaluate(query, list(names or self.profiles))["escalated"])

    def escalate_from(self, query: Any, names: Sequence[str], failed: Sequence[str]) -> dict[str, bool]:
        """Awake set after the experts in ``failed`` failed their prediction checks (e.g. a closure check): the
        failed experts are excluded and the ladder moves up to the next rung that covers the query."""
        ev = self._evaluate(query, list(names), frozenset(failed))
        self.log.append({"query": id(query), "failed": list(failed), "rung": ev["rung"]})
        return dict(ev["awake"])

    def covered(self, query: Any, names: Sequence[str] | None = None) -> list[str]:
        """Experts that can answer the query at all (may be empty: the query is not covered)."""
        names = list(names or self.profiles)
        return [n for n, c in self.checks(query, names).items() if c.covered]

    def route(self, query: Any, learned_weights: Mapping[str, float] | None = None,
              names: Sequence[str] | None = None) -> ScaleRoute:
        """Choose a model for one case with the reasons; ``chosen`` is None when no model covers the query (the
        reasons then say which scales would be needed)."""
        names = list(names or (learned_weights.keys() if learned_weights else self.profiles))
        ev = self._evaluate(query, names)
        sq = self.query_fn(query)
        reasons = []
        if not ev["covered"]:
            finest = min((self.profiles[n].resolved_width for n in names if n in self.profiles), default=float("nan"))
            reasons.append(f"not covered: the query asks for width {sq.filter.delta:.3g}; the finest model resolves "
                           f"{finest:.3g}. A resolved solver with grid spacing <= {sq.filter.delta:.3g} is needed.")
            return ScaleRoute(None, {n: 0.0 for n in names}, ev["checks"], {n: False for n in names}, None, False,
                              reasons)
        lw = np.array([max(float((learned_weights or {}).get(n, 1.0)), 1e-300) for n in names])
        logw = np.log(lw) + np.array([ev["log_prior"][n] for n in names])
        logw = np.where([ev["awake"][n] for n in names], logw, -np.inf)
        if not np.isfinite(logw).any():
            logw = np.log(lw)
        w = np.exp(logw - logw.max())
        w = w / w.sum()
        weights = dict(zip(names, w.tolist(), strict=True))
        chosen = max(weights, key=weights.get)
        if ev["rung"] is not None:
            reasons.append(f"rung {ev['rung']} ({RUNGS.get(ev['rung'], '?')})"
                           + (": escalated, the lower rungs do not cover the query" if ev["escalated"] else ""))
        for n in sorted(names, key=lambda x: -weights[x]):
            c = ev["checks"][n]
            head = f"{n}: weight {weights[n]:.2f}, scale compatibility {c.score:.2f}" + \
                   ("" if ev["awake"][n] else " (asleep)")
            reasons.append(head + (": " + "; ".join(c.reasons) if c.reasons else ""))
        return ScaleRoute(chosen, weights, ev["checks"], dict(ev["awake"]), ev["rung"], ev["escalated"], reasons)
