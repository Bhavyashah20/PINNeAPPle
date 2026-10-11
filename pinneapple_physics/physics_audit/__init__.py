"""Physics audit: which model to use, when to trust it and when its physical formulation itself must be revisited.

"It is in the fundamental knowledge that we must look for the cause of a model's imperfections." The modules here
add, on top of the predictive models (PINN, FNO, GNN, DeepONet, CNN, solvers), the layers that judge them by their
physics:

========================  =====================================================================================
layer                     module
========================  =====================================================================================
physical evaluation       :mod:`.monitor` (residuals, conservation, bounds, boundary, invariants, extrapolation:
                          no ground truth needed), :mod:`.assumptions` (each model's assumptions and the
                          conditions they hold in), :mod:`.regime` (dimensionless numbers)
adaptive decision         :mod:`.routing` (compatibility of the current case with each model's physics, as a prior
                          on the weights of :class:`~pinneapple_physics.ensemble.PhysicsEnsemble`; sleeping
                          experts; escalation to the solver)
diagnosis and evolution   :mod:`.attribution` (probable cause of a failure, with evidence and confidence),
                          :mod:`.improvement` (next experiment or change of highest value), :mod:`.knowledge`
                          (graph of phenomena, equations, assumptions, parameters, mechanisms and models)
========================  =====================================================================================

:func:`diagnose` runs the evaluation and diagnosis layers on one case and returns a :class:`Diagnosis` with a
readable report. Plain numpy: nothing here imports torch.

Low residuals do not guarantee a correct prediction, and high residuals alone do not identify the cause: the
diagnosis combines physical evidence, observations, validity conditions and uncertainty, and says which of them
it had.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .assumptions import (
    ASSUMPTIONS,
    MARGINAL,
    UNKNOWN,
    VALID,
    VIOLATED,
    Assumption,
    AssumptionAudit,
    AssumptionCheck,
    Condition,
    Envelope,
    ModelProfile,
    advection_dominated,
    attached_flow,
    audit_assumptions,
    boussinesq,
    constant_properties,
    continuum,
    creeping_flow,
    diffusion_dominated,
    forced_convection,
    get_assumption,
    hydraulically_smooth,
    incompressible,
    laminar,
    linear_elastic,
    lumped_capacitance,
    shallow_water,
    small_strain,
    turbulent,
)
from .attribution import CAUSES, Attribution, Cause, attribute_failure
from .improvement import ACTIONS, Recommendation, design_points, improvement_plan, recommend
from .knowledge import EQUATIONS, PHENOMENA, PhysicsKnowledgeGraph, default_graph
from .monitor import (
    ALARM,
    OK,
    WARN,
    CheckResult,
    PhysicsValidityMonitor,
    ValidityCheck,
    ValidityReport,
    boundary_check,
    bounds_check,
    conservation_check,
    constitutive_check,
    envelope_check,
    invariant_check,
    residual_check,
)
from .regime import regime_features, roughness_reynolds
from .routing import PhysicsAwareRouter, RouteDecision

__all__ = [
    "VALID", "MARGINAL", "VIOLATED", "UNKNOWN", "Condition", "Assumption", "AssumptionCheck", "AssumptionAudit",
    "ASSUMPTIONS", "get_assumption", "Envelope", "ModelProfile", "audit_assumptions",
    "laminar", "turbulent", "hydraulically_smooth", "incompressible", "creeping_flow", "continuum", "boussinesq", "constant_properties",
    "small_strain", "linear_elastic", "lumped_capacitance", "forced_convection", "diffusion_dominated",
    "advection_dominated", "attached_flow", "shallow_water",
    "OK", "WARN", "ALARM", "CheckResult", "ValidityCheck", "ValidityReport", "PhysicsValidityMonitor",
    "residual_check", "conservation_check", "bounds_check", "boundary_check", "constitutive_check",
    "invariant_check", "envelope_check",
    "PhysicsAwareRouter", "RouteDecision",
    "CAUSES", "Cause", "Attribution", "attribute_failure",
    "ACTIONS", "Recommendation", "recommend", "improvement_plan", "design_points",
    "PHENOMENA", "EQUATIONS", "PhysicsKnowledgeGraph", "default_graph",
    "regime_features", "roughness_reynolds", "Diagnosis", "diagnose",
]


@dataclass
class Diagnosis:
    model: str
    features: dict[str, float]
    audit: AssumptionAudit | None
    validity: ValidityReport | None
    attribution: Attribution
    recommendations: list[Recommendation]
    unrepresented: list[dict[str, Any]] = field(default_factory=list)

    @property
    def trusted(self) -> bool:
        """No failure found, assumptions valid (or marginal at worst) and the validity checks passed."""
        ok_audit = self.audit is None or self.audit.status in (VALID, MARGINAL)
        ok_validity = self.validity is None or self.validity.status == OK
        return not self.attribution.failed and ok_audit and ok_validity

    def report(self) -> str:
        out = [f"# Physics audit: {self.model}", "",
               "Features: " + ", ".join(f"{k} = {v:.4g}" for k, v in self.features.items()), "",
               f"Verdict: {'trusted for this case' if self.trusted else 'NOT trusted for this case'}", ""]
        if self.audit is not None:
            out += ["## Assumptions", *self.audit.lines(), ""]
        if self.unrepresented:
            out += ["## Physics the model does not represent"]
            out += [f"- {u['phenomenon']} ({u['status'].lower()}): missing {', '.join(u['missing'])}; {u['evidence']}"
                    for u in self.unrepresented]
            out.append("")
        if self.validity is not None:
            out += ["## Validity checks", *self.validity.lines(), ""]
        out += ["## Failure attribution", *self.attribution.lines(), ""]
        if self.recommendations:
            out.append("## Recommended next steps")
            for k, r in enumerate(self.recommendations[:6]):
                out.append(f"{k + 1}. {r.line()}")
                for m, alts in r.details.get("alternatives", {}).items():
                    out.append(f"   - {m}: " + (", ".join(alts) if alts else "nothing in the graph represents it"))
                if "design_points" in r.details:
                    pts = r.details["design_points"]
                    keys = sorted({kk for p in pts[1:] for kk, v in p.items() if v != pts[0].get(kk)})
                    out.append(f"   - {len(pts)} design points around the case, varying {', '.join(keys) or 'nothing'}")
        return "\n".join(out)

    def __str__(self) -> str:
        return self.report()


def diagnose(profile: ModelProfile, features: Mapping[str, float], prediction: Any = None, query: Any = None,
             monitor: PhysicsValidityMonitor | None = None, error: float | None = None, error_tolerance: float = 0.05,
             graph: PhysicsKnowledgeGraph | None = None, **attribution_kwargs) -> Diagnosis:
    """Audit the assumptions of ``profile`` for the case's ``features``, run ``monitor`` on ``prediction`` (if both
    given), attribute the failure (``error``: relative error against a reference, if any; other evidence through
    ``attribution_kwargs``: ``resolution``, ``measurement_suspect``, ``spread``, ``residual_baseline``) and recommend
    the next steps. ``graph`` (default: :func:`default_graph` with ``profile``) names the missing mechanisms."""
    feats = dict(features)
    audit = audit_assumptions(profile, feats)
    validity = monitor.check(prediction, query) if monitor is not None and prediction is not None else None
    if graph is None:
        graph = default_graph([profile])
    elif not graph.has_model(profile.name):
        graph.add_model(profile)
    unrep = graph.unrepresented(profile.name, feats)
    missing = sorted({m for u in unrep for m in u["missing"]}) or list(audit.missing_mechanisms)
    att = attribute_failure(validity=validity, audit=audit, error=error, error_tolerance=error_tolerance,
                            missing_mechanisms=missing or None, **attribution_kwargs)
    recs = recommend(att, audit, graph, feats, profile.name) if att.failed or audit.status != VALID else []
    return Diagnosis(profile.name, feats, audit, validity, att, recs, unrep)
