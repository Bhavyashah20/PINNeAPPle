"""Physics assumptions of a model, the conditions under which each holds, and an auditor that checks them case by
case.

Every surrogate, PINN or solver rests on simplifications: incompressible flow, laminar regime, constant properties,
small strains. Each holds inside a range of dimensionless numbers. A model used outside that range can still return
a smooth, plausible field, and its PDE residual can even stay low (it satisfies *its* equations, which no longer
describe the case). So the first place to look for the cause of an error is the model's own physical foundations.

* :class:`Condition`: ``lo <= feature <= hi``, with a soft band around the edges (in decades for numbers that span
  orders of magnitude) so the auditor says *marginal* near a transition instead of flipping from valid to violated;
* :class:`Assumption`: a named simplification, its conditions, the phenomenon that appears when it breaks and the
  physical mechanism the model then lacks. :data:`ASSUMPTIONS` holds the common ones; the factory functions
  (:func:`laminar`, :func:`incompressible`...) build them with other thresholds;
* :class:`Envelope`: where the model was trained (box of feature ranges, optionally the training samples for a
  nearest-neighbour distance), for extrapolation;
* :class:`ModelProfile`: the model card the audit reads: equations, assumptions, mechanisms represented, envelope;
* :func:`audit_assumptions` / :class:`AssumptionAudit`: one case against one profile, with a status per assumption
  (``VALID``, ``MARGINAL``, ``VIOLATED``, ``UNKNOWN`` when the feature is missing: never a silent pass), a
  compatibility score in [0, 1] and the phenomena the case probably triggers.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["VALID", "MARGINAL", "VIOLATED", "UNKNOWN", "Condition", "Assumption", "ASSUMPTIONS", "get_assumption",
           "laminar", "turbulent", "hydraulically_smooth", "incompressible", "creeping_flow", "continuum", "boussinesq",
           "constant_properties", "small_strain", "linear_elastic", "lumped_capacitance", "forced_convection",
           "diffusion_dominated", "advection_dominated", "attached_flow", "shallow_water", "Envelope",
           "ModelProfile", "AssumptionCheck", "AssumptionAudit", "audit_assumptions"]

VALID, MARGINAL, VIOLATED, UNKNOWN = "VALID", "MARGINAL", "VIOLATED", "UNKNOWN"


@dataclass(frozen=True)
class Condition:
    """``lo <= features[feature] <= hi``. ``band``: width of the soft edge, in decades when ``log`` (default for
    positive bounds), in feature units otherwise. Inside but within ``band`` of an edge is marginal; outside by more
    than ``band`` is violated. :meth:`score` is 1 inside and decays as exp(-(d / band)^2) outside."""
    feature: str
    lo: float = -math.inf
    hi: float = math.inf
    band: float | None = None
    log: bool | None = None

    def _log(self) -> bool:
        if self.log is not None:
            return self.log
        return all(b > 0 for b in (self.lo, self.hi) if math.isfinite(b))

    def _band(self) -> float:
        if self.band is not None:
            return self.band
        if self._log():
            return 0.15                                              # a factor of about 1.4
        span = [abs(b) for b in (self.lo, self.hi) if math.isfinite(b)]
        return 0.1 * max(span) if span and max(span) > 0 else 0.1

    def distance(self, value: float) -> float:
        """Signed distance to the nearest edge: negative inside, positive outside (decades when log)."""
        v = float(value)
        if self._log():
            if v <= 0:
                return math.inf if self.lo > 0 else -math.inf
            v = math.log10(v)
            lo = math.log10(self.lo) if math.isfinite(self.lo) and self.lo > 0 else -math.inf
            hi = math.log10(self.hi) if math.isfinite(self.hi) else math.inf
        else:
            lo, hi = self.lo, self.hi
        if v < lo:
            return lo - v
        if v > hi:
            return v - hi
        return -min(v - lo, hi - v)

    def status(self, value: float) -> str:
        d, b = self.distance(value), self._band()
        if d > b:
            return VIOLATED
        if d > -b:
            return MARGINAL
        return VALID

    def score(self, value: float) -> float:
        d = self.distance(value)
        return 1.0 if d <= 0 else float(math.exp(-(d / self._band()) ** 2))

    def describe(self) -> str:
        if math.isfinite(self.lo) and math.isfinite(self.hi):
            return f"{_num(self.lo)} <= {self.feature} <= {_num(self.hi)}"
        if math.isfinite(self.hi):
            return f"{self.feature} <= {_num(self.hi)}"
        if math.isfinite(self.lo):
            return f"{self.feature} >= {_num(self.lo)}"
        return f"{self.feature} (any value)"


def _num(x: float) -> str:
    return f"{x:.3g}"


@dataclass(frozen=True)
class Assumption:
    """A simplification a model rests on. ``breaks_into``: the phenomenon that appears when it fails (turbulence,
    compressibility...); ``missing_mechanism``: what the model then lacks (turbulent transport, density
    variation...)."""
    name: str
    description: str
    conditions: tuple[Condition, ...]
    breaks_into: str = ""
    missing_mechanism: str = ""

    def check(self, features: Mapping[str, float]) -> AssumptionCheck:
        per = []
        for c in self.conditions:
            v = features.get(c.feature)
            if v is None or not np.isfinite(v):
                per.append((c, None, UNKNOWN, None, math.nan))
            else:
                per.append((c, float(v), c.status(v), c.distance(v), c.score(v)))
        statuses = [p[2] for p in per]
        if VIOLATED in statuses:
            st = VIOLATED
        elif UNKNOWN in statuses:
            st = UNKNOWN
        elif MARGINAL in statuses:
            st = MARGINAL
        else:
            st = VALID
        known = [p[4] for p in per if p[2] != UNKNOWN]
        score = float(np.prod(known)) if known else math.nan
        lines = []
        for c, v, s, _, _ in per:
            got = "not given" if v is None else f"{c.feature} = {_num(v)}"
            lines.append(f"{c.describe()} ({got}): {s}")
        return AssumptionCheck(self.name, st, score, lines, self.breaks_into, self.missing_mechanism,
                               [c.feature for c in self.conditions])


@dataclass
class AssumptionCheck:
    name: str
    status: str
    score: float                    # 1 inside the validity range, -> 0 far outside; NaN when the feature is missing
    evidence: list[str]
    breaks_into: str
    missing_mechanism: str
    features: list[str]


# ------------------------------------------------------------------------------------------------ catalogue
def laminar(re_crit: float = 2300.0, feature: str = "Re", geometry: str = "pipe") -> Assumption:
    """Laminar flow below the transition Reynolds number (2300 in a pipe; about 5e5 on a flat plate, by x)."""
    return Assumption("laminar" if geometry == "pipe" else f"laminar_{geometry}",
                      f"laminar flow ({geometry}): no turbulent fluctuations, {feature} below transition",
                      (Condition(feature, hi=re_crit, band=0.12),), "turbulence", "turbulent transport (eddy viscosity)")


def turbulent(re_min: float = 4000.0, re_max: float = math.inf, feature: str = "Re") -> Assumption:
    """Fully turbulent flow (wall functions, turbulent correlations such as Blasius or Colebrook). The phenomenon
    named is the one below ``re_min``; state an upper limit of a correlation through the training envelope or a
    separate assumption (:func:`hydraulically_smooth`)."""
    return Assumption("fully_turbulent", f"fully developed turbulence, {feature} above the transition range",
                      (Condition(feature, lo=re_min, hi=re_max, band=0.12),), "laminar or transitional flow",
                      "laminar or transitional regime")


def hydraulically_smooth(eps_plus_max: float = 5.0) -> Assumption:
    """Smooth wall: roughness buried in the viscous sublayer, roughness Reynolds number eps+ = Re (eps / D)
    sqrt(f / 8) below about 5 (see :func:`~.regime.roughness_reynolds`)."""
    return Assumption("hydraulically_smooth", "smooth wall: roughness inside the viscous sublayer",
                      (Condition("eps_plus", hi=eps_plus_max, band=0.15),), "rough-wall turbulence", "wall roughness")


def incompressible(ma_max: float = 0.3) -> Assumption:
    return Assumption("incompressible", "constant density: density changes with pressure are negligible",
                      (Condition("Ma", hi=ma_max, band=0.1),), "compressibility", "density variation, shocks")


def creeping_flow(re_max: float = 1.0) -> Assumption:
    return Assumption("creeping_flow", "Stokes flow: inertia negligible next to viscosity",
                      (Condition("Re", hi=re_max, band=0.3),), "inertial effects", "convective inertia")


def continuum(kn_max: float = 0.01) -> Assumption:
    return Assumption("continuum", "continuum fluid with no-slip walls (Navier-Stokes valid)",
                      (Condition("Kn", hi=kn_max, band=0.3),), "rarefaction", "velocity slip, non-equilibrium")


def boussinesq(max_beta_dt: float = 0.1) -> Assumption:
    return Assumption("boussinesq", "density varies only in the buoyancy term, linearly with temperature",
                      (Condition("beta_dT", hi=max_beta_dt, band=0.15),), "large density variation",
                      "variable-density buoyancy")


def constant_properties(max_dt_over_t: float = 0.1) -> Assumption:
    return Assumption("constant_properties", "viscosity, conductivity and heat capacity independent of temperature",
                      (Condition("dT_over_T", hi=max_dt_over_t, band=0.15),), "temperature-dependent properties",
                      "property variation with temperature")


def small_strain(max_strain: float = 0.01) -> Assumption:
    return Assumption("small_strain", "small deformations: linear strain measure, undeformed geometry",
                      (Condition("strain", hi=max_strain, band=0.15),), "large deformation", "geometric nonlinearity")


def linear_elastic(max_stress_ratio: float = 1.0) -> Assumption:
    return Assumption("linear_elastic", "stress below yield: linear elastic material",
                      (Condition("stress_ratio", hi=max_stress_ratio, band=0.08),), "plasticity", "plastic flow")


def lumped_capacitance(bi_max: float = 0.1) -> Assumption:
    return Assumption("lumped_capacitance", "uniform temperature inside the body",
                      (Condition("Bi", hi=bi_max, band=0.15),), "internal temperature gradients", "internal conduction")


def forced_convection(ri_max: float = 0.1) -> Assumption:
    return Assumption("forced_convection", "buoyancy negligible next to the imposed flow",
                      (Condition("Ri", hi=ri_max, band=0.3),), "mixed or natural convection", "buoyancy-driven flow")


def diffusion_dominated(pe_max: float = 1.0) -> Assumption:
    return Assumption("diffusion_dominated", "transport dominated by diffusion; advection is a small correction",
                      (Condition("Pe", hi=pe_max, band=0.3),), "advection-dominated transport",
                      "advective transport, sharp fronts")


def advection_dominated(pe_min: float = 10.0) -> Assumption:
    return Assumption("advection_dominated", "transport dominated by advection (thin layers, steep fronts)",
                      (Condition("Pe", lo=pe_min, band=0.3),), "diffusion-dominated transport", "diffusive smoothing")


def attached_flow(alpha_stall_deg: float = 12.0) -> Assumption:
    return Assumption("attached_flow", "attached flow over the wing or airfoil (no stall, linear lift)",
                      (Condition("alpha_deg", lo=-alpha_stall_deg, hi=alpha_stall_deg, band=1.5, log=False),),
                      "flow separation and stall", "separated flow")


def shallow_water(max_aspect: float = 0.05) -> Assumption:
    return Assumption("shallow_water", "depth small next to the horizontal scale: hydrostatic pressure",
                      (Condition("H_over_L", hi=max_aspect, band=0.15),), "non-hydrostatic flow",
                      "vertical acceleration, dispersion")


ASSUMPTIONS: dict[str, Assumption] = {a.name: a for a in (
    laminar(), turbulent(), hydraulically_smooth(), incompressible(), creeping_flow(), continuum(), boussinesq(), constant_properties(),
    small_strain(), linear_elastic(), lumped_capacitance(), forced_convection(), diffusion_dominated(),
    advection_dominated(), attached_flow(), shallow_water())}


def get_assumption(a: str | Assumption) -> Assumption:
    if isinstance(a, Assumption):
        return a
    if a not in ASSUMPTIONS:
        raise KeyError(f"unknown assumption {a!r}; known: {sorted(ASSUMPTIONS)} (or pass an Assumption)")
    return ASSUMPTIONS[a]


# ------------------------------------------------------------------------------------------------ training envelope
@dataclass
class Envelope:
    """Where a model was trained or validated. ``ranges``: {feature: (lo, hi)}; ``samples``: optional
    (n, d) training points over ``sample_features`` for a nearest-neighbour distance (in units of the median
    neighbour spacing of the training set, on standardised features, log10 for positive features)."""
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    samples: np.ndarray | None = None
    sample_features: tuple[str, ...] = ()
    band: float = 0.1                    # relative (log10 decades for positive ranges, else fraction of the span)

    @classmethod
    def from_samples(cls, samples: Sequence[Mapping[str, float]], features: Sequence[str] | None = None) -> Envelope:
        feats = tuple(features or sorted(set().union(*[s.keys() for s in samples])))
        X = np.array([[float(s[f]) for f in feats] for s in samples])
        ranges = {f: (float(X[:, j].min()), float(X[:, j].max())) for j, f in enumerate(feats)}
        return cls(ranges, X, feats)

    def _conditions(self) -> list[Condition]:
        out = []
        for f, (lo, hi) in self.ranges.items():
            log = lo > 0 and hi > 0
            band = self.band if log else self.band * max(hi - lo, 1e-12)
            out.append(Condition(f, lo, hi, band=band, log=log))
        return out

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            col = self.samples[:, j]
            if np.all(col > 0):
                X[:, j] = np.log10(np.maximum(X[:, j], 1e-300))
        return X

    def check(self, features: Mapping[str, float]) -> dict[str, Any]:
        """``inside`` (all ranges), ``score`` in [0, 1], per-feature lines and, with samples, ``knn_distance``."""
        lines, scores, outside = [], [], []
        for c in self._conditions():
            v = features.get(c.feature)
            if v is None:
                lines.append(f"{c.feature}: not given (trained on {c.describe()})")
                continue
            s = c.score(v)
            scores.append(s)
            if c.distance(v) > 0:
                outside.append(c.feature)
            lines.append(f"{c.feature} = {_num(v)}, trained on {_num(c.lo)} .. {_num(c.hi)}"
                         + ("" if c.distance(v) <= 0 else " (outside)"))
        out: dict[str, Any] = {"inside": not outside, "outside": outside,
                               "score": float(np.prod(scores)) if scores else math.nan, "evidence": lines}
        if self.samples is not None and len(self.samples) > 1 and all(f in features for f in self.sample_features):
            T = self._transform(self.samples)
            mu, sd = T.mean(0), T.std(0) + 1e-12
            Z = (T - mu) / sd
            q = (self._transform(np.array([[float(features[f]) for f in self.sample_features]])) - mu) / sd
            d_train = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d_train, np.inf)
            spacing = float(np.median(d_train.min(1))) or 1.0
            d = float(np.sqrt(((Z - q) ** 2).sum(-1)).min()) / spacing
            out["knn_distance"] = d
            out["evidence"].append(f"nearest training case at {d:.2f} x the typical training spacing")
            if d > 3.0:                                               # inside the box but in a hole of the data
                out["score"] = (out["score"] if np.isfinite(out["score"]) else 1.0) * float(math.exp(-((d - 3.0) / 3.0) ** 2))
        return out


# ------------------------------------------------------------------------------------------------ model profile
@dataclass
class ModelProfile:
    """What the audit knows about a model: family (``"PINN"``, ``"FNO"``, ``"solver"``...), governing equations,
    assumptions (names from :data:`ASSUMPTIONS` or :class:`Assumption` objects), mechanisms it represents
    (``"turbulent transport"``...), training envelope and relative cost."""
    name: str
    family: str = ""
    equations: tuple[str, ...] = ()
    assumptions: tuple[Assumption | str, ...] = ()
    mechanisms: tuple[str, ...] = ()
    envelope: Envelope | None = None
    cost: float | None = None
    notes: str = ""

    def __post_init__(self):
        self.assumptions = tuple(get_assumption(a) for a in self.assumptions)
        self.equations = tuple(self.equations)
        self.mechanisms = tuple(self.mechanisms)


@dataclass
class AssumptionAudit:
    model: str
    status: str                       # worst status over the assumptions and the envelope
    compatibility: float              # product of the assumption and envelope scores, in [0, 1]
    checks: list[AssumptionCheck]
    envelope: dict[str, Any] | None
    triggered: list[str]              # phenomena the case probably triggers (from violated / marginal assumptions)
    missing_mechanisms: list[str]

    @property
    def violated(self) -> list[AssumptionCheck]:
        return [c for c in self.checks if c.status == VIOLATED]

    @property
    def marginal(self) -> list[AssumptionCheck]:
        return [c for c in self.checks if c.status == MARGINAL]

    @property
    def unknown(self) -> list[AssumptionCheck]:
        return [c for c in self.checks if c.status == UNKNOWN]

    def lines(self) -> list[str]:
        out = [f"{self.model}: {self.status} (compatibility {self.compatibility:.2f})"]
        for c in self.checks:
            out.append(f"  [{c.status}] {c.name}: " + "; ".join(c.evidence))
        if self.envelope is not None:
            out.append(f"  [{'INSIDE' if self.envelope['inside'] else 'OUTSIDE'}] training envelope: "
                       + "; ".join(self.envelope["evidence"]))
        if self.triggered:
            out.append("  probable phenomena: " + ", ".join(self.triggered))
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "status": self.status, "compatibility": self.compatibility,
                "checks": [c.__dict__ for c in self.checks], "envelope": self.envelope,
                "triggered": self.triggered, "missing_mechanisms": self.missing_mechanisms}


_ORDER = {VALID: 0, MARGINAL: 1, UNKNOWN: 2, VIOLATED: 3}


def audit_assumptions(profile: ModelProfile, features: Mapping[str, float]) -> AssumptionAudit:
    """Check every assumption of ``profile`` and its training envelope for one case's ``features``."""
    checks = [a.check(features) for a in profile.assumptions]
    env = profile.envelope.check(features) if profile.envelope is not None else None
    statuses = [c.status for c in checks]
    if env is not None and not env["inside"]:
        statuses.append(VIOLATED if env["score"] < 0.5 else MARGINAL)
    status = max(statuses, key=_ORDER.get) if statuses else UNKNOWN
    scores = [c.score for c in checks if np.isfinite(c.score)]
    if env is not None and np.isfinite(env["score"]):
        scores.append(env["score"])
    compat = float(np.prod(scores)) if scores else 1.0
    bad = [c for c in checks if c.status in (VIOLATED, MARGINAL)]
    triggered = _unique(c.breaks_into for c in bad if c.breaks_into)
    missing = _unique(c.missing_mechanism for c in bad if c.missing_mechanism and c.missing_mechanism not in profile.mechanisms)
    return AssumptionAudit(profile.name, status, compat, checks, env, triggered, missing)


def _unique(xs: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
