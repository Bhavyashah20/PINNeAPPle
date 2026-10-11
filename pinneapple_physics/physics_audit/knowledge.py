"""Physics knowledge graph: phenomena, equations, assumptions, parameters, mechanisms and model families, and the
relations between them.

Nodes have a kind (``phenomenon``, ``equation``, ``assumption``, ``parameter``, ``mechanism``, ``model``); edges a
relation:

=====================  ===================================================================
``assumes``            equation or model -> assumption it rests on
``valid_when``         assumption -> parameter (with the condition, in the edge data)
``breaks_into``        assumption -> phenomenon that appears when it fails
``triggered_by``       phenomenon -> parameter (with the trigger condition)
``requires``           phenomenon -> mechanism a model needs to capture it
``uses``               model -> equation
``represents``         model or equation -> mechanism
=====================  ===================================================================

:func:`default_graph` builds the common fluid, heat and solid relations from the assumption catalogue. The queries the
router, the attribution and the improvement advisor use: :meth:`~PhysicsKnowledgeGraph.phenomena_triggered` (which
phenomena a case is in or near), :meth:`~PhysicsKnowledgeGraph.unrepresented` (triggered phenomena whose mechanism a
model lacks: "Re went up, is turbulence represented?"), :meth:`~PhysicsKnowledgeGraph.models_for` (models that
represent a mechanism) and :meth:`~PhysicsKnowledgeGraph.explain`. :meth:`~PhysicsKnowledgeGraph.to_mermaid` draws it.
Plain Python, no graph library needed.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .assumptions import (
    ASSUMPTIONS,
    MARGINAL,
    VALID,
    VIOLATED,
    Assumption,
    Condition,
    ModelProfile,
    get_assumption,
)

__all__ = ["Node", "Edge", "PhysicsKnowledgeGraph", "PHENOMENA", "EQUATIONS", "default_graph"]

KINDS = ("phenomenon", "equation", "assumption", "parameter", "mechanism", "model")


@dataclass
class Node:
    id: str
    kind: str
    label: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    rel: str
    dst: str
    data: dict[str, Any] = field(default_factory=dict)


# phenomenon: (trigger condition, mechanism needed, description). Thresholds are generic order-of-magnitude values;
# a model's own assumptions carry its specific thresholds (pipe vs plate transition, for instance).
PHENOMENA: dict[str, tuple[Condition, str, str]] = {
    "turbulence": (Condition("Re", lo=2300, band=0.12), "turbulent transport (eddy viscosity)",
                   "chaotic velocity fluctuations that transport momentum and heat"),
    "laminar or transitional flow": (Condition("Re", hi=4000, band=0.12), "laminar or transitional regime",
                                     "flow below or inside the transition range"),
    "rough-wall turbulence": (Condition("eps_plus", lo=5.0, band=0.15), "wall roughness",
                              "wall roughness pokes through the viscous sublayer and adds form drag"),
    "compressibility": (Condition("Ma", lo=0.3, band=0.1), "density variation, shocks",
                        "density changes with pressure; shocks above Ma 1"),
    "rarefaction": (Condition("Kn", lo=0.01, band=0.3), "velocity slip, non-equilibrium",
                    "the gas stops behaving as a continuum near walls"),
    "inertial effects": (Condition("Re", lo=1.0, band=0.3), "convective inertia", "inertia comparable to viscosity"),
    "mixed or natural convection": (Condition("Ri", lo=0.1, band=0.3), "buoyancy-driven flow",
                                    "buoyancy drives or deflects the flow"),
    "large density variation": (Condition("beta_dT", lo=0.1, band=0.15), "variable-density buoyancy",
                                "density varies beyond the Boussinesq range"),
    "temperature-dependent properties": (Condition("dT_over_T", lo=0.1, band=0.15),
                                         "property variation with temperature",
                                         "viscosity and conductivity change across the domain"),
    "advection-dominated transport": (Condition("Pe", lo=1.0, band=0.3), "advective transport, sharp fronts",
                                      "steep fronts and thin layers carried by the flow"),
    "diffusion-dominated transport": (Condition("Pe", hi=10.0, band=0.3), "diffusive smoothing",
                                      "diffusion smooths the field faster than the flow carries it"),
    "internal temperature gradients": (Condition("Bi", lo=0.1, band=0.15), "internal conduction",
                                       "the body is not at a uniform temperature"),
    "large deformation": (Condition("strain", lo=0.01, band=0.15), "geometric nonlinearity",
                          "the geometry changes enough to change the stiffness"),
    "plasticity": (Condition("stress_ratio", lo=1.0, band=0.08), "plastic flow", "stress above yield"),
    "flow separation and stall": (Condition("alpha_deg", lo=12.0, band=1.5, log=False), "separated flow",
                                  "the boundary layer separates; lift stops growing with incidence"),
    "non-hydrostatic flow": (Condition("H_over_L", lo=0.05, band=0.15), "vertical acceleration, dispersion",
                             "vertical accelerations matter (short waves, steep topography)"),
}

# equation: (assumptions, mechanisms represented)
EQUATIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "incompressible Navier-Stokes": (("incompressible", "continuum"), ("convective inertia", "viscous diffusion")),
    "Stokes equations": (("incompressible", "continuum", "creeping_flow"), ("viscous diffusion",)),
    "Hagen-Poiseuille / laminar correlations": (("incompressible", "laminar"), ("viscous diffusion",)),
    "RANS with eddy viscosity": (("incompressible", "continuum"),
                                 ("convective inertia", "turbulent transport (eddy viscosity)")),
    "smooth-pipe friction correlations (Blasius, Prandtl-Karman)": (("fully_turbulent", "hydraulically_smooth"),
                                                                    ("turbulent transport (eddy viscosity)",)),
    "Colebrook-White correlation": (("fully_turbulent",), ("turbulent transport (eddy viscosity)", "wall roughness")),
    "compressible Navier-Stokes / Euler": (("continuum",), ("convective inertia", "density variation, shocks")),
    "Boussinesq natural convection": (("incompressible", "boussinesq"), ("buoyancy-driven flow", "convective inertia")),
    "heat equation (constant k)": (("constant_properties",), ("internal conduction",)),
    "lumped thermal model": (("lumped_capacitance",), ()),
    "advection-diffusion": ((), ("advective transport, sharp fronts", "diffusive smoothing")),
    "pure diffusion": (("diffusion_dominated",), ("diffusive smoothing",)),
    "linear elasticity": (("small_strain", "linear_elastic"), ()),
    "nonlinear elasticity / plasticity": ((), ("geometric nonlinearity", "plastic flow")),
    "thin airfoil / vortex lattice": (("incompressible", "attached_flow"), ()),
    "shallow water equations": (("shallow_water",), ()),
}


class PhysicsKnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    # ------------------------------------------------------------------------------------------ building
    def add_node(self, id: str, kind: str, label: str = "", **data) -> Node:
        if kind not in KINDS:
            raise ValueError(f"unknown node kind {kind!r}; use one of {KINDS}")
        n = self.nodes.get(id)
        if n is None:
            n = self.nodes[id] = Node(id, kind, label or id, dict(data))
        elif n.kind != kind:
            raise ValueError(f"node {id!r} already exists as a {n.kind}")
        else:
            n.data.update(data)
        return n

    def add_edge(self, src: str, rel: str, dst: str, **data) -> Edge:
        for nid in (src, dst):
            if nid not in self.nodes:
                raise KeyError(f"unknown node {nid!r}; add it first")
        e = Edge(src, rel, dst, dict(data))
        if not any(x.src == src and x.rel == rel and x.dst == dst for x in self.edges):
            self.edges.append(e)
        return e

    def add_assumption(self, a: Assumption) -> None:
        self.add_node(a.name, "assumption", a.description, assumption=a)
        for c in a.conditions:
            self.add_node(c.feature, "parameter")
            self.add_edge(a.name, "valid_when", c.feature, condition=c)
        if a.breaks_into:
            self.add_node(a.breaks_into, "phenomenon")
            self.add_edge(a.name, "breaks_into", a.breaks_into)
        if a.missing_mechanism:
            self.add_node(a.missing_mechanism, "mechanism")
            if a.breaks_into:
                self.add_edge(a.breaks_into, "requires", a.missing_mechanism)

    def add_phenomenon(self, name: str, trigger: Condition | None, mechanism: str = "", description: str = "") -> None:
        self.add_node(name, "phenomenon", description or name, trigger=trigger)
        if trigger is not None:
            self.add_node(trigger.feature, "parameter")
            self.add_edge(name, "triggered_by", trigger.feature, condition=trigger)
        if mechanism:
            self.add_node(mechanism, "mechanism")
            self.add_edge(name, "requires", mechanism)

    def add_equation(self, name: str, assumptions: Iterable[str | Assumption] = (), mechanisms: Iterable[str] = ()) -> None:
        self.add_node(name, "equation")
        for a in assumptions:
            a = get_assumption(a)
            self.add_assumption(a)
            self.add_edge(name, "assumes", a.name)
        for m in mechanisms:
            self.add_node(m, "mechanism")
            self.add_edge(name, "represents", m)

    @staticmethod
    def model_id(name: str) -> str:
        """Models live in their own namespace, so a model may share a name with an assumption or an equation."""
        return name if name.startswith("model:") else f"model:{name}"

    def has_model(self, name: str) -> bool:
        return self.model_id(name) in self.nodes

    def add_model(self, profile: ModelProfile) -> None:
        """A model, its equations (added if new), its own assumptions (with its thresholds) and the mechanisms it
        represents (given, plus those of its equations)."""
        mid = self.model_id(profile.name)
        self.add_node(mid, "model", profile.name, family=profile.family, profile=profile)
        for eq in profile.equations:
            if eq not in self.nodes:
                self.add_node(eq, "equation")
            self.add_edge(mid, "uses", eq)
        for a in profile.assumptions:
            self.add_assumption(a)
            self.add_edge(mid, "assumes", a.name)
        for m in profile.mechanisms:
            self.add_node(m, "mechanism")
            self.add_edge(mid, "represents", m)

    # ------------------------------------------------------------------------------------------ queries
    def out(self, src: str, rel: str | None = None) -> list[Edge]:
        return [e for e in self.edges if e.src == src and (rel is None or e.rel == rel)]

    def into(self, dst: str, rel: str | None = None) -> list[Edge]:
        return [e for e in self.edges if e.dst == dst and (rel is None or e.rel == rel)]

    def of_kind(self, kind: str) -> list[str]:
        return [n.id for n in self.nodes.values() if n.kind == kind]

    def mechanisms_of(self, model: str) -> set[str]:
        """Mechanisms a model represents: its own plus those of the equations it uses."""
        model = self.model_id(model)
        mech = {e.dst for e in self.out(model, "represents")}
        for e in self.out(model, "uses"):
            mech |= {x.dst for x in self.out(e.dst, "represents")}
        return mech

    def assumptions_of(self, model: str) -> list[Assumption]:
        """The model's own assumptions, then those of its equations it does not already state (a model's threshold
        wins over the generic one of the same name)."""
        model = self.model_id(model)
        own = [self.nodes[e.dst].data["assumption"] for e in self.out(model, "assumes")]
        names = {a.name for a in own}
        for e in self.out(model, "uses"):
            for x in self.out(e.dst, "assumes"):
                a = self.nodes[x.dst].data["assumption"]
                if a.name not in names:
                    own.append(a)
                    names.add(a.name)
        return own

    def phenomena_triggered(self, features: Mapping[str, float], include_marginal: bool = True) -> list[dict[str, Any]]:
        """Phenomena whose trigger condition the case meets (or nearly meets, as ``MARGINAL``)."""
        out = []
        for n in self.nodes.values():
            t = n.data.get("trigger") if n.kind == "phenomenon" else None
            if t is None or features.get(t.feature) is None or not math.isfinite(features[t.feature]):
                continue
            s = t.status(features[t.feature])
            if s == VALID or (s == MARGINAL and include_marginal):
                out.append({"phenomenon": n.id, "status": "ACTIVE" if s == VALID else "NEAR",
                            "evidence": f"{t.describe()} ({t.feature} = {features[t.feature]:.3g})",
                            "requires": [e.dst for e in self.out(n.id, "requires")]})
        return out

    def unrepresented(self, model: str, features: Mapping[str, float], generic: bool = True) -> list[dict[str, Any]]:
        """Phenomena the case is in (or near) whose mechanism ``model`` lacks: from the model's violated or marginal
        assumptions (its own thresholds), then (``generic``) from the generic triggers, for numbers the model sets no
        threshold on. The generic part needs the model to declare its mechanisms: without them nothing can be said."""
        if not self.has_model(model):
            raise KeyError(f"unknown model {model!r}; add it with add_model")
        have = self.mechanisms_of(model)
        out, seen = [], set()
        for a in self.assumptions_of(model):
            chk = a.check(features)
            if chk.status in (VIOLATED, MARGINAL) and a.breaks_into and a.missing_mechanism not in have:
                seen.add(a.breaks_into)
                out.append({"phenomenon": a.breaks_into, "status": "ACTIVE" if chk.status == VIOLATED else "NEAR",
                            "via": f"assumption {a.name}", "evidence": "; ".join(chk.evidence),
                            "missing": [a.missing_mechanism]})
        own_features = {c.feature for a in self.assumptions_of(model) for c in a.conditions}
        for p in (self.phenomena_triggered(features) if generic and have else []):
            if p["phenomenon"] in seen:
                continue
            t = self.nodes[p["phenomenon"]].data["trigger"]
            if t.feature in own_features:                     # the model states its own threshold on this number
                continue
            missing = [m for m in p["requires"] if m not in have]
            if missing:
                out.append({"phenomenon": p["phenomenon"], "status": p["status"], "via": "generic trigger",
                            "evidence": p["evidence"], "missing": missing})
        return out

    def models_for(self, mechanism: str) -> list[str]:
        return [n.label for n in self.nodes.values() if n.kind == "model" and mechanism in self.mechanisms_of(n.label)]

    def equations_for(self, mechanism: str) -> list[str]:
        return [e.src for e in self.into(mechanism, "represents") if self.nodes[e.src].kind == "equation"]

    def explain(self, model: str, features: Mapping[str, float]) -> list[str]:
        lines = []
        for u in self.unrepresented(model, features):
            word = "is in" if u["status"] == "ACTIVE" else "is near"
            fix = []
            for m in u["missing"]:
                eqs, mods = self.equations_for(m), [x for x in self.models_for(m) if x != model]
                if mods:
                    fix.append(f"models that represent {m}: {', '.join(mods)}")
                elif eqs:
                    fix.append(f"equations that represent {m}: {', '.join(eqs)}")
            lines.append(f"the case {word} the regime of {u['phenomenon']} ({u['evidence']}); {model} does not represent "
                         f"{', '.join(u['missing'])}" + (f" -> {'; '.join(fix)}" if fix else ""))
        return lines

    def to_mermaid(self, kinds: Iterable[str] = KINDS) -> str:
        keep = set(kinds)
        ids = {nid: f"n{k}" for k, nid in enumerate(self.nodes)}
        shape = {"phenomenon": ("((", "))"), "equation": ("[", "]"), "assumption": ("{{", "}}"),
                 "parameter": ("([", "])"), "mechanism": ("[/", "/]"), "model": ("[[", "]]")}
        lines = ["graph LR"]
        for n in self.nodes.values():
            if n.kind in keep:
                a, b = shape[n.kind]
                lines.append(f'  {ids[n.id]}{a}"{n.id}"{b}')
        for e in self.edges:
            if self.nodes[e.src].kind in keep and self.nodes[e.dst].kind in keep:
                lines.append(f"  {ids[e.src]} -->|{e.rel}| {ids[e.dst]}")
        return "\n".join(lines)


def default_graph(models: Iterable[ModelProfile] = ()) -> PhysicsKnowledgeGraph:
    """The catalogue of assumptions, phenomena and equations above, plus ``models``."""
    g = PhysicsKnowledgeGraph()
    for a in ASSUMPTIONS.values():
        g.add_assumption(a)
    for name, (trigger, mech, desc) in PHENOMENA.items():
        g.add_phenomenon(name, trigger, mech, desc)
    for eq, (assumptions, mechs) in EQUATIONS.items():
        g.add_equation(eq, assumptions, mechs)
    for m in models:
        g.add_model(m)
    return g
