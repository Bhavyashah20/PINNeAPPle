"""Physics-aware model routing: weight each model by how well the *current* case fits its physical foundations, not
only by the errors it made on earlier cases.

For one case the router reads its features (dimensionless numbers, from ``features_fn(query)``) and, for each
model, the audit of its assumptions and training envelope (:func:`~.assumptions.audit_assumptions`). The
compatibility ``c_i`` in [0, 1] becomes a prior on the model's weight:

    w_i  proportional to  w_learned_i * max(c_i, floor) ** strength * exp(-cost_weight * cost_i / max cost)

A model whose assumptions the case violates is *asleep* (``c_i < abstain_below``): it is not used for this case
and, inside a :class:`~pinneapple_physics.ensemble.PhysicsEnsemble`, its weight is not changed by the case either
(the sleeping-experts rule: its loss is set to the mixture's). So a laminar correlation is not punished for the
turbulent cases it was never meant for, and is ready again when the flow returns to its regime, without waiting for
its weight to recover. When no model is compatible (best ``c_i < escalate_below``) the router escalates to the
reference solver named by ``escalate_to``.

Use standalone (:meth:`PhysicsAwareRouter.route`) or inside the ensemble: ``PhysicsEnsemble(experts,
router=PhysicsAwareRouter(profiles, features_fn))``. The learned weights and the residual look-ahead of the ensemble
still apply on top, so history and the current residual correct a compatibility that is wrong.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .assumptions import AssumptionAudit, ModelProfile, audit_assumptions

__all__ = ["RouteDecision", "PhysicsAwareRouter"]


@dataclass
class RouteDecision:
    chosen: str
    weights: dict[str, float]                 # final weights (sum 1 over awake models)
    compatibility: dict[str, float]
    awake: dict[str, bool]
    escalate: bool
    audits: dict[str, AssumptionAudit]
    reasons: list[str]

    def __str__(self) -> str:
        return "\n".join([f"chosen: {self.chosen}" + (" (escalated)" if self.escalate else "")] + self.reasons)


class PhysicsAwareRouter:
    """``profiles``: one :class:`~.assumptions.ModelProfile` per model (same names as the ensemble's experts; a model
    without a profile gets compatibility 1). ``features_fn(query) -> {feature: value}`` (identity if the queries
    are already feature dictionaries)."""

    def __init__(self, profiles: Sequence[ModelProfile] | Mapping[str, ModelProfile],
                 features_fn: Callable[[Any], Mapping[str, float]] | None = None, strength: float = 1.0,
                 floor: float = 1e-4, abstain_below: float | None = 0.05, escalate_below: float | None = None,
                 escalate_to: str | None = None, cost_weight: float = 0.0):
        if isinstance(profiles, Mapping):
            profiles = list(profiles.values())
        self.profiles = {p.name: p for p in profiles}
        self.features_fn = features_fn or (lambda q: q)
        self.strength, self.floor = float(strength), float(floor)
        self.abstain_below = abstain_below
        self.escalate_below = escalate_below
        if escalate_to is not None and escalate_to not in self.profiles:
            raise ValueError(f"escalate_to={escalate_to!r} has no profile")
        self.escalate_to = escalate_to
        self.cost_weight = float(cost_weight)
        self._cache: tuple | None = None

    # ------------------------------------------------------------------------------------------ per case
    def _evaluate(self, query: Any, names: Sequence[str]) -> dict[str, Any]:
        key = tuple(names)
        if self._cache is not None and self._cache[0] is query and self._cache[1] == key:   # same case, same models
            return self._cache[2]
        feats = dict(self.features_fn(query))
        audits = {n: audit_assumptions(self.profiles[n], feats) for n in names if n in self.profiles}
        compat = {n: (audits[n].compatibility if n in audits else 1.0) for n in names}
        awake = {n: self.abstain_below is None or compat[n] >= self.abstain_below for n in names}
        cand = [n for n in names if n != self.escalate_to]
        best = max((compat[n] for n in cand), default=0.0)
        escalate = bool(self.escalate_to in names and self.escalate_below is not None and best < self.escalate_below)
        if escalate:
            awake = {n: n == self.escalate_to for n in names}
        elif not any(awake.values()):                                # nobody fits and no solver to call: keep all
            awake = {n: True for n in names}
        costs = np.array([self.profiles[n].cost if n in self.profiles and self.profiles[n].cost is not None else np.nan
                          for n in names], dtype=float)
        cmax = np.nanmax(costs) if np.any(np.isfinite(costs)) else 1.0
        lp = {}
        for k, n in enumerate(names):
            v = self.strength * math.log(max(compat[n], self.floor))
            if self.cost_weight > 0 and np.isfinite(costs[k]):
                v -= self.cost_weight * costs[k] / max(cmax, 1e-300)
            lp[n] = v
        out = {"features": feats, "audits": audits, "compat": compat, "awake": awake, "escalate": escalate,
               "log_prior": lp}
        self._cache = (query, key, out)                              # holds the query: its id is not reused
        return out

    def prior(self, query: Any, names: Sequence[str] | None = None) -> dict[str, float]:
        """Log-weight offset of each model for this case (used by ``PhysicsEnsemble(router=...)``)."""
        return dict(self._evaluate(query, list(names or self.profiles))["log_prior"])

    def awake(self, query: Any, names: Sequence[str] | None = None) -> dict[str, bool]:
        return dict(self._evaluate(query, list(names or self.profiles))["awake"])

    def escalate(self, query: Any, names: Sequence[str] | None = None) -> bool:
        return bool(self._evaluate(query, list(names or self.profiles))["escalate"])

    def route(self, query: Any, learned_weights: Mapping[str, float] | None = None,
              names: Sequence[str] | None = None) -> RouteDecision:
        """Choose a model for one case: compatibility prior times ``learned_weights`` (uniform if None), over the
        awake models, with the reasons."""
        names = list(names or (learned_weights.keys() if learned_weights else self.profiles))
        ev = self._evaluate(query, names)
        lw = np.array([max(float((learned_weights or {}).get(n, 1.0)), 1e-300) for n in names])
        logw = np.log(lw) + np.array([ev["log_prior"][n] for n in names])
        logw = np.where([ev["awake"][n] for n in names], logw, -np.inf)
        w = np.exp(logw - logw.max())
        w = w / w.sum()
        weights = dict(zip(names, w.tolist(), strict=True))
        chosen = max(weights, key=weights.get)
        reasons = []
        for n in sorted(names, key=lambda x: -weights[x]):
            a = ev["audits"].get(n)
            head = f"{n}: weight {weights[n]:.2f}, compatibility {ev['compat'][n]:.2f}" + \
                   ("" if ev["awake"][n] else " (asleep: outside its physics)")
            if a is not None:
                bad = [c for c in a.checks if c.status != "VALID"]
                detail = [f"{c.name} {c.status.lower()} ({'; '.join(c.evidence)})" for c in bad]
                if a.envelope is not None and not a.envelope["inside"]:
                    detail.append("outside its training envelope (" + ", ".join(a.envelope["outside"]) + ")")
                if detail:
                    head += ": " + "; ".join(detail)
            reasons.append(head)
        if ev["escalate"]:
            reasons.insert(0, f"no surrogate fits this case (best compatibility below {self.escalate_below}): "
                              f"escalated to {self.escalate_to}")
        return RouteDecision(chosen, weights, dict(ev["compat"]), dict(ev["awake"]), ev["escalate"], ev["audits"],
                             reasons)
