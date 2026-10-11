"""Physics-guided model improvement: when no model is good enough, which experiment or change is worth most?

:func:`recommend` turns the diagnosis of one case (an :class:`~.attribution.Attribution`, optionally the
assumption audit and the knowledge graph) into ranked actions; :func:`improvement_plan` aggregates many cases (a
validation campaign, a week of operation) so the actions that fix the most frequent and most confident causes come
first. Each action says what to do, why (the evidence), the expected gain and the cost:

=========================  ===================================================================================
cause                      actions
=========================  ===================================================================================
``regime_change``          high-fidelity runs in the new regime; train or adapt a model for it; switch to a
                           model or equation that represents the missing mechanism (from the knowledge graph)
``extrapolation``          high-fidelity runs at and around the case (design points given); active learning
``pde_violation``          more collocation points or residual weight where it fails; hard constraints
``conservation``           conservation projection (``pinneapple_physics.conservation``) or a penalty in the loss
``boundary_condition``     check the boundary data; enforce the condition exactly (ansatz or projection)
``numerical_resolution``   refine the mesh or the time step; run a grid convergence study
``measurement``            check the sensor; keep the reading out of training and of the weights
``missing_physics``        add the missing term; fit a model-discrepancy correction; calibrate parameters
=========================  ===================================================================================

The gain and cost are ordinal (1 low .. 3 high), stated in the table ``ACTIONS`` below, not computed: they are a
reviewer's ranking aid, and the priority is ``probability * confidence * gain / cost``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .assumptions import AssumptionAudit
from .attribution import Attribution
from .knowledge import PhysicsKnowledgeGraph

__all__ = ["ACTIONS", "Recommendation", "recommend", "improvement_plan", "design_points"]

# cause -> [(action id, action, gain 1-3, cost 1-3)]
ACTIONS: dict[str, list[tuple[str, str, int, int]]] = {
    "regime_change": [
        ("hifi_new_regime", "run high-fidelity simulations in the new regime", 3, 2),
        ("specialised_model", "train or fine-tune a model specialised for the new regime", 3, 2),
        ("switch_formulation", "use a model or equation that represents the missing mechanism", 3, 1),
    ],
    "extrapolation": [
        ("hifi_around_case", "run high-fidelity simulations at and around the case, then retrain", 3, 2),
        ("active_learning", "add the case's region to an active-learning loop (sample where models disagree)", 2, 1),
    ],
    "pde_violation": [
        ("collocation", "add collocation points / residual weight where the residual fails", 2, 1),
        ("loss_change", "modify the loss: residual-based adaptive weighting or a stronger physics term", 2, 1),
        ("hard_constraint", "enforce the equation's structure exactly (divergence-free or symmetric ansatz)", 2, 2),
    ],
    "conservation": [
        ("projection", "add a conservation projection after the model (pinneapple_physics.conservation)", 3, 1),
        ("conservation_loss", "add a conservation penalty to the loss", 2, 1),
    ],
    "boundary_condition": [
        ("check_bc_data", "check the boundary condition data and units against the case", 2, 1),
        ("hard_bc", "enforce the boundary condition exactly (distance-function ansatz or projection)", 3, 1),
    ],
    "numerical_resolution": [
        ("refine", "refine the mesh or the time step where the solution is unresolved", 3, 2),
        ("gci_study", "run a grid convergence study (three grids, GCI) to size the numerical error", 2, 2),
    ],
    "measurement": [
        ("check_sensor", "check the sensor and its calibration; compare with redundant readings", 2, 1),
        ("exclude_reading", "keep the reading out of training data and of the ensemble weights", 2, 1),
    ],
    "missing_physics": [
        ("add_term", "add the missing physical term or mechanism to the model's equations", 3, 2),
        ("discrepancy_model", "fit a model-discrepancy correction on reference data (hybrid model)", 2, 1),
        ("calibrate", "calibrate the uncertain physical parameters against the reference", 2, 1),
    ],
}


@dataclass
class Recommendation:
    action_id: str
    action: str
    cause: str
    priority: float
    gain: int
    cost: int
    rationale: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    cases: int = 1

    def line(self) -> str:
        n = f", {self.cases} cases" if self.cases > 1 else ""
        return f"[{self.priority:.2f}] {self.action} (cause: {self.cause}; gain {self.gain}/3, cost {self.cost}/3{n})"


def design_points(features: Mapping[str, float], spread: float = 0.15, n: int = 5, seed: int = 0,
                  keys: Sequence[str] | None = None) -> list[dict[str, float]]:
    """High-fidelity design points at and around a case: the case itself, then ``n - 1`` points spread by ``spread``
    (in decades for positive features) on a Latin hypercube."""
    keys = list(keys or features)
    rng = np.random.default_rng(seed)
    pts = [dict(features)]
    if n <= 1:
        return pts
    m = n - 1
    u = (np.argsort(rng.random((len(keys), m)), axis=1) + rng.random((len(keys), m))) / m * 2 - 1   # in [-1, 1]
    for j in range(m):
        p = dict(features)
        for i, k in enumerate(keys):
            v = float(features[k])
            p[k] = v * 10 ** (spread * u[i, j]) if v > 0 else v + spread * u[i, j] * (abs(v) or 1.0)
        pts.append(p)
    return pts


def recommend(attribution: Attribution, audit: AssumptionAudit | None = None,
              graph: PhysicsKnowledgeGraph | None = None, features: Mapping[str, float] | None = None,
              model: str | None = None, min_probability: float = 0.1) -> list[Recommendation]:
    """Ranked actions for one case. With ``graph``, ``model`` and ``features`` the "switch formulation" and "add
    term" actions name the mechanism the model lacks and the models or equations that represent it; with
    ``features`` the high-fidelity actions list design points."""
    out = []
    unrep = graph.unrepresented(model, features) if graph is not None and model and features is not None and \
        graph.has_model(model) else []
    for c in attribution.causes:
        if c.code == "unexplained" or c.probability < min_probability:
            continue
        for aid, text, gain, cost in ACTIONS.get(c.code, []):
            details: dict[str, Any] = {}
            rationale = list(c.evidence)
            if aid in ("switch_formulation", "add_term") and unrep:
                mechs = sorted({m for u in unrep for m in u["missing"]})
                details["missing_mechanisms"] = mechs
                alts = {m: [x for x in graph.models_for(m) if x != model] or graph.equations_for(m) for m in mechs}
                details["alternatives"] = alts
                rationale += [f"{m}: {', '.join(a) if a else 'no model or equation in the graph represents it'}"
                              for m, a in alts.items()]
            elif aid in ("switch_formulation", "add_term") and audit is not None and audit.missing_mechanisms:
                details["missing_mechanisms"] = list(audit.missing_mechanisms)
            if aid in ("hifi_new_regime", "hifi_around_case") and features is not None:
                keys = None
                if audit is not None:
                    bad = [f for ch in audit.checks if ch.status != "VALID" for f in ch.features if f in features]
                    if audit.envelope is not None:
                        bad += [f for f in audit.envelope.get("outside", []) if f in features]
                    keys = sorted(set(bad)) or None
                details["design_points"] = design_points(features, keys=keys)
            pr = c.probability * max(attribution.confidence, 0.1) * gain / cost
            out.append(Recommendation(aid, text, c.code, float(pr), gain, cost, rationale, details))
    out.sort(key=lambda r: -r.priority)
    return out


def improvement_plan(cases: Iterable[Attribution | tuple[Attribution, list[Recommendation]]],
                     top: int | None = None) -> list[Recommendation]:
    """Aggregate the recommendations of many cases: the same action summed over the cases that call for it (each
    case's priority), so frequent, confident causes rank first. Items are attributions (recommended with
    :func:`recommend` without context) or ``(attribution, recommendations)`` pairs."""
    acc: dict[tuple[str, str], Recommendation] = {}
    for item in cases:
        recs = item[1] if isinstance(item, tuple) else recommend(item)
        for r in recs:
            key = (r.cause, r.action_id)
            if key not in acc:
                acc[key] = Recommendation(r.action_id, r.action, r.cause, 0.0, r.gain, r.cost, [], {}, 0)
            a = acc[key]
            a.priority += r.priority
            a.cases += 1
            for e in r.rationale:
                if e not in a.rationale and len(a.rationale) < 6:
                    a.rationale.append(e)
            for k, v in r.details.items():
                if k == "design_points":
                    a.details.setdefault(k, []).extend(v[:1])           # the cases themselves
                else:
                    a.details.setdefault(k, v)
    plan = sorted(acc.values(), key=lambda r: -r.priority)
    return plan[:top] if top else plan
