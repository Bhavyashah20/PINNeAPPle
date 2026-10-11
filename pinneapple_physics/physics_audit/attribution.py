"""Physics failure attribution: not only *that* a prediction is wrong (or suspect), but its probable cause, with the
evidence and a confidence.

The causes, and the evidence each one rests on:

=========================  ==============================================================================
``pde_violation``          the governing-equation residual fails its tolerance or jumps above its baseline
``conservation``           a conserved total (mass, energy, momentum) is off
``boundary_condition``     the prediction does not meet the boundary condition
``regime_change``          the case violates an assumption of the model (the physics changed under it)
``extrapolation``          the assumptions hold but the case is outside the training envelope
``numerical_resolution``   grid or time-step dependence (GCI, change under refinement), non-finite or out-of-
                           bounds values from an unresolved solution
``measurement``            the reference itself is suspect (sensor check), so the "error" may not be the model's
``missing_physics``        the model satisfies its own equations, inside its assumptions and envelope, and still
                           disagrees with the reference: its equations miss a mechanism (model-form error)
=========================  ==============================================================================

Each cause gets a score in [0, 1] from its evidence; the scores, plus an "unexplained" share when nothing points
anywhere, are normalised into probabilities. The confidence of the diagnosis is the top probability times the
fraction of evidence types that were actually available: a diagnosis made without a residual, an audit or a
reference says so. This is the point the residual alone cannot make: a low residual with a high error points to
``missing_physics`` or ``measurement``; a high residual says *that* something failed, the other evidence says *why*.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .assumptions import MARGINAL, VIOLATED, AssumptionAudit
from .monitor import ValidityReport

__all__ = ["CAUSES", "Cause", "Attribution", "attribute_failure"]

CAUSES = {
    "pde_violation": "Governing equations violated",
    "conservation": "Conservation law violated",
    "boundary_condition": "Boundary conditions inconsistent",
    "regime_change": "Regime change: a model assumption no longer holds",
    "extrapolation": "Extrapolation outside the training data",
    "numerical_resolution": "Numerical failure or insufficient resolution",
    "measurement": "Suspect reference or sensor reading",
    "missing_physics": "Missing physics (model-form error)",
    "unexplained": "No cause supported by the available evidence",
}
EVIDENCE_TYPES = ("validity", "audit", "reference", "resolution", "spread")


@dataclass
class Cause:
    code: str
    title: str
    probability: float
    score: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class Attribution:
    failed: bool                       # is there anything to explain (error above tolerance, failed checks, alarm)?
    causes: list[Cause]                # sorted by probability
    confidence: float
    evidence_available: list[str]
    error: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def top(self) -> Cause:
        return self.causes[0]

    def lines(self) -> list[str]:
        head = ("failure" if self.failed else "no failure detected") + \
               (f", relative error {100 * self.error:.3g} %" if self.error is not None else "")
        out = [f"{head}; most probable cause: {self.top.title} (p = {self.top.probability:.2f}, "
               f"confidence {self.confidence:.2f})"]
        for c in self.causes:
            if c.probability < 0.05 and c is not self.top:
                continue
            out.append(f"  {c.probability:.2f}  {c.title}")
            out += [f"        - {e}" for e in c.evidence]
        missing = [e for e in EVIDENCE_TYPES if e not in self.evidence_available]
        if missing:
            out.append("  evidence not available: " + ", ".join(missing))
        out += [f"  note: {n}" for n in self.notes]
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())

    def to_dict(self) -> dict[str, Any]:
        return {"failed": self.failed, "error": self.error, "confidence": self.confidence,
                "evidence_available": self.evidence_available, "notes": self.notes,
                "causes": [c.__dict__ for c in self.causes]}


def _ratio_score(ratio: float) -> float:
    """Failed check with value / tolerance = ratio -> score: 0.5 at the tolerance, about 0.9 at 10x."""
    if not np.isfinite(ratio):
        return 1.0
    return float(1.0 - 0.5 / max(ratio, 1.0) ** 0.6) if ratio >= 1 else 0.0


def attribute_failure(*, validity: ValidityReport | None = None, audit: AssumptionAudit | None = None,
                      error: float | None = None, error_tolerance: float = 0.05,
                      residual_baseline: float | Mapping[str, float] | None = None,
                      resolution: Mapping[str, float] | None = None, measurement_suspect: bool | None = None,
                      spread: float | None = None, spread_tolerance: float = 0.1,
                      missing_mechanisms: list[str] | None = None) -> Attribution:
    """Weigh the evidence of one case.

    ``validity``: the monitor's report; ``audit``: the model's assumption audit for this case; ``error``: relative
    error against a reference when one exists (``error_tolerance`` marks it as failed); ``residual_baseline``: the
    model's in-domain residual level (number, or {check name: level}), to tell a residual jump from a residual
    that is just above an absolute tolerance; ``resolution``: grid or time-step evidence, any of ``{"gci": ...,
    "gci_tol": ...}``, ``{"refinement_change": ..., "refinement_tol": ...}`` (relative change of the prediction
    under refinement), ``{"cfl": ..., "cfl_max": ...}``; ``measurement_suspect``: the reading failed its sensor
    check; ``spread``: ensemble disagreement (relative); ``missing_mechanisms``: from the knowledge graph, named in
    the evidence of ``missing_physics``."""
    s = {k: 0.0 for k in CAUSES if k != "unexplained"}
    ev: dict[str, list[str]] = {k: [] for k in CAUSES}
    available, notes = [], []

    residual_ok = None                                          # None: no residual check at all
    if validity is not None:
        available.append("validity")
        for r in validity.results:
            if r.kind == "residual":
                ratio = r.ratio
                base = residual_baseline.get(r.name) if isinstance(residual_baseline, Mapping) else residual_baseline
                jump = r.value / base if base else None
                residual_ok = (residual_ok is not False) and r.passed and (jump is None or jump < 3.0)
                sc = _ratio_score(ratio)
                if jump is not None and jump >= 3.0:
                    sc = max(sc, _ratio_score(jump / 3.0))
                    ev["pde_violation"].append(f"{r.name} is {jump:.1f} x its in-domain level ({r.detail})")
                if not r.passed:
                    ev["pde_violation"].append(f"{r.name}: {r.detail}")
                s["pde_violation"] = max(s["pde_violation"], sc)
            elif r.kind == "conservation" and not r.passed:
                s["conservation"] = max(s["conservation"], _ratio_score(r.ratio))
                ev["conservation"].append(f"{r.name}: {r.detail}")
            elif r.kind == "boundary" and not r.passed:
                s["boundary_condition"] = max(s["boundary_condition"], _ratio_score(r.ratio))
                ev["boundary_condition"].append(f"{r.name}: {r.detail}")
            elif r.kind == "bounds" and not r.passed:
                s["numerical_resolution"] = max(s["numerical_resolution"], 0.6 if np.isfinite(r.value) else 0.9)
                ev["numerical_resolution"].append(f"{r.name}: {r.detail}")
            elif r.kind in ("constitutive", "invariant") and not r.passed:
                s["pde_violation"] = max(s["pde_violation"], 0.7 * _ratio_score(r.ratio))
                ev["pde_violation"].append(f"{r.name} ({r.kind}): {r.detail}")
            elif r.kind == "envelope" and not r.passed:
                s["extrapolation"] = max(s["extrapolation"], 0.6)
                ev["extrapolation"].append(f"{r.name}: {r.detail}")

    if audit is not None:
        available.append("audit")
        for c in audit.checks:
            if c.status == VIOLATED:
                sc = 0.95 if not np.isfinite(c.score) else 0.6 + 0.4 * (1.0 - c.score)
                s["regime_change"] = max(s["regime_change"], sc)
                ev["regime_change"].append(f"assumption '{c.name}' violated: {'; '.join(c.evidence)}"
                                           + (f" -> {c.breaks_into}" if c.breaks_into else ""))
            elif c.status == MARGINAL:
                s["regime_change"] = max(s["regime_change"], 0.35)
                ev["regime_change"].append(f"assumption '{c.name}' marginal: {'; '.join(c.evidence)}")
        env = audit.envelope
        if env is not None and not env["inside"]:
            sc = 0.5 + 0.45 * (1.0 - float(np.nan_to_num(env["score"], nan=0.0)))
            s["extrapolation"] = max(s["extrapolation"], sc)
            ev["extrapolation"].append("outside the training envelope: " + "; ".join(
                e for e in env["evidence"] if "outside" in e))
        elif env is not None and env.get("knn_distance", 0) > 3.0:
            s["extrapolation"] = max(s["extrapolation"], 0.4)
            ev["extrapolation"].append(f"inside the ranges but far from the training cases "
                                       f"({env['knn_distance']:.1f} x the typical spacing)")
        if s["regime_change"] > 0 and s["extrapolation"] > 0:
            s["extrapolation"] *= 0.7                             # the regime explains the extrapolation too
            notes.append("the case is both outside the training data and outside the model's physics; "
                         "the regime change is the deeper cause")

    if resolution:
        available.append("resolution")
        gci, gtol = resolution.get("gci"), resolution.get("gci_tol", 0.05)
        if gci is not None and gci > gtol:
            s["numerical_resolution"] = max(s["numerical_resolution"], _ratio_score(gci / gtol))
            ev["numerical_resolution"].append(f"grid convergence index {100 * gci:.3g} % > {100 * gtol:.3g} %")
        ch, ctol = resolution.get("refinement_change"), resolution.get("refinement_tol", 0.02)
        if ch is not None and ch > ctol:
            s["numerical_resolution"] = max(s["numerical_resolution"], _ratio_score(ch / ctol))
            ev["numerical_resolution"].append(f"prediction changes {100 * ch:.3g} % under refinement "
                                              f"(> {100 * ctol:.3g} %)")
        cfl, cmax = resolution.get("cfl"), resolution.get("cfl_max", 1.0)
        if cfl is not None and cfl > cmax:
            s["numerical_resolution"] = max(s["numerical_resolution"], 0.8)
            ev["numerical_resolution"].append(f"CFL {cfl:.3g} above the stability limit {cmax:.3g}")

    if measurement_suspect:
        s["measurement"] = 0.85
        ev["measurement"].append("the reference reading failed its physical consistency check")

    if spread is not None:
        available.append("spread")
        if spread > spread_tolerance:
            ev["extrapolation"].append(f"models disagree by {100 * spread:.3g} % (> {100 * spread_tolerance:.3g} %)")
            s["extrapolation"] = max(s["extrapolation"], 0.3 * min(spread / spread_tolerance, 3.0) / 3.0 + 0.2)

    if error is not None:
        available.append("reference")
        if error > error_tolerance:
            explained = max(s.values())
            in_physics = audit is not None and not audit.violated and (audit.envelope is None or audit.envelope["inside"])
            if residual_ok and in_physics and not measurement_suspect and explained < 0.5:
                sc = min(0.95, 0.5 + 0.15 * math.log10(error / error_tolerance) + 0.3)
                s["missing_physics"] = sc
                ev["missing_physics"].append(
                    f"error {100 * error:.3g} % while the model satisfies its equations, inside its assumptions and "
                    "training envelope: its equations do not describe this case")
                if missing_mechanisms:
                    ev["missing_physics"].append("mechanisms not represented: " + ", ".join(missing_mechanisms))
            elif residual_ok and not measurement_suspect and explained < 0.5:
                s["missing_physics"] = 0.4
                ev["missing_physics"].append(f"error {100 * error:.3g} % with a residual inside tolerance")
    elif residual_ok:
        notes.append("residual within tolerance, but a low residual does not prove the prediction right: "
                     "no reference was available to check")

    failed = bool((error is not None and error > error_tolerance) or (validity is not None and validity.failed)
                  or any(v >= 0.5 for v in s.values()))
    total = sum(s.values())
    unexplained = max(0.0, 1.0 - max(s.values())) if failed else 1.0 - min(total, 1.0)
    if not failed and total == 0:
        unexplained = 1.0
    raw = dict(s, unexplained=unexplained)
    if not failed:
        ev["unexplained"].append("no check failed and no reference error above tolerance")
    norm = sum(raw.values()) or 1.0
    causes = [Cause(k, CAUSES[k], v / norm, raw[k], ev[k]) for k, v in raw.items() if v > 0]
    causes.sort(key=lambda c: -c.probability)
    coverage = len(available) / len(EVIDENCE_TYPES)
    conf = float(causes[0].probability * (0.4 + 0.6 * coverage)) if causes else 0.0
    return Attribution(failed, causes, conf, available, error, notes)
