"""Physics validity monitor: is a prediction still physically plausible, when no ground truth is available?

A monitor holds a list of checks; each takes ``(prediction, query)`` and returns a number with a tolerance:

* :func:`residual_check`: norm of the governing-equation residual (from ``pp.compile``, a finite-difference stencil,
  the residual function of a :class:`~pinneapple_physics.ensemble.PhysicsEnsemble`...);
* :func:`conservation_check`: a conserved total (mass, energy, momentum) against its expected value;
* :func:`bounds_check`: admissible values (non-negative density or concentration, temperature above 0 K,
  probabilities in [0, 1]);
* :func:`boundary_check`: the prediction at the boundary against the boundary condition;
* :func:`constitutive_check`: any relation the fields must satisfy (equation of state, a material law);
* :func:`invariant_check`: a quantity that must not change from case to case or step to step (symmetry, energy of a
  conservative system);
* :func:`envelope_check`: the case's features against the model's training envelope (extrapolation).

The monitor keeps a history per check and raises the status from ``OK`` to ``WARN`` (a check failed) and ``ALARM``
(a critical check failed, or the same check failed ``persistence`` cases in a row, or a residual drifted to
``drift_ratio`` times its own baseline). Low residuals do not prove a prediction right (a model can satisfy its own
simplified equations and miss the real phenomenon) and a high residual does not say why it failed; the monitor
reports evidence, :mod:`.attribution` weighs it.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .assumptions import Envelope

__all__ = ["OK", "WARN", "ALARM", "CheckResult", "ValidityCheck", "residual_check", "conservation_check",
           "bounds_check", "boundary_check", "constitutive_check", "invariant_check", "envelope_check",
           "ValidityReport", "PhysicsValidityMonitor"]

OK, WARN, ALARM = "OK", "WARN", "ALARM"
KINDS = ("residual", "conservation", "bounds", "boundary", "constitutive", "invariant", "envelope")


@dataclass
class CheckResult:
    name: str
    kind: str                         # one of KINDS
    value: float                      # the measured quantity (residual norm, relative imbalance, fraction outside...)
    tolerance: float
    passed: bool
    critical: bool
    detail: str = ""

    @property
    def ratio(self) -> float:
        """value / tolerance (> 1 means failed)."""
        return float(self.value / self.tolerance) if self.tolerance > 0 else (np.inf if self.value > 0 else 0.0)


@dataclass
class ValidityCheck:
    """``fn(prediction, query) -> value``; passes when ``value <= tolerance``. ``critical``: a single failure is an
    alarm (default for bounds: a negative density is never acceptable)."""
    name: str
    kind: str
    fn: Callable[[np.ndarray, Any], float]
    tolerance: float
    critical: bool = False
    describe: Callable[[float], str] | None = None

    def __call__(self, prediction: np.ndarray, query: Any) -> CheckResult:
        try:
            v = float(self.fn(prediction, query))
        except Exception as exc:                                       # noqa: BLE001 - a broken check is evidence too
            return CheckResult(self.name, self.kind, np.inf, self.tolerance, False, self.critical,
                               f"check failed to run: {type(exc).__name__}: {exc}")
        ok = bool(np.isfinite(v) and v <= self.tolerance)
        det = self.describe(v) if self.describe is not None else f"{v:.3g} (tolerance {self.tolerance:.3g})"
        return CheckResult(self.name, self.kind, v, self.tolerance, ok, self.critical, det)


def residual_check(residual_fn: Callable[[np.ndarray, Any], float], tolerance: float, name: str = "pde_residual",
                   critical: bool = False) -> ValidityCheck:
    """``residual_fn(prediction, query)``: a residual norm (RMS of the PDE residual, for instance)."""
    return ValidityCheck(name, "residual", residual_fn, tolerance, critical)


def _relative(delta: float, target: float, rel_tol: float, abs_tol: float) -> float:
    """``|delta|`` over ``max(|target|, abs_tol / rel_tol)``: the check passes when ``|delta| <= rel_tol |target|`` or
    ``|delta| <= abs_tol``, so a conserved total whose expected value is zero (net charge, net momentum of a closed
    system) can be checked by giving ``abs_tol``."""
    floor = abs_tol / rel_tol if abs_tol > 0 and rel_tol > 0 else 0.0
    return abs(delta) / max(abs(target), floor, 1e-300)


def conservation_check(total_fn: Callable[[np.ndarray, Any], float], target_fn: Callable[[Any], float],
                       rel_tol: float = 0.01, name: str = "conservation", critical: bool = False,
                       abs_tol: float = 0.0) -> ValidityCheck:
    """Relative imbalance ``|total(prediction) - target(query)| / |target(query)|``. ``total_fn`` is typically a
    quadrature (``np.sum(cell_volumes * rho)``); ``target_fn`` the initial total plus sources minus outflow.
    ``abs_tol``: an absolute imbalance that is always accepted (needed when the target is zero); the value reported
    is then ``|imbalance| / max(|target|, abs_tol / rel_tol)``."""
    def fn(p, q):
        t = float(target_fn(q))
        return _relative(float(total_fn(p, q)) - t, t, rel_tol, abs_tol)
    return ValidityCheck(name, "conservation", fn, rel_tol, critical,
                         lambda v: f"imbalance {100 * v:.3g} % (tolerance {100 * rel_tol:.3g} %)")


def bounds_check(lo: float | None = None, hi: float | None = None, name: str = "bounds",
                 select: Callable[[np.ndarray, Any], np.ndarray] | None = None, max_fraction: float = 0.0,
                 critical: bool = True) -> ValidityCheck:
    """Fraction of values outside ``[lo, hi]`` (a field selected by ``select``, all of it by default)."""
    def fn(p, q):
        x = np.asarray(select(p, q) if select is not None else p, dtype=float)
        bad = np.zeros(x.shape, dtype=bool)
        if lo is not None:
            bad |= x < lo
        if hi is not None:
            bad |= x > hi
        return float(bad.mean()) if x.size else 0.0
    rng = f"[{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]"
    return ValidityCheck(name, "bounds", fn, max_fraction, critical,
                         lambda v: f"{100 * v:.3g} % of values outside {rng}")


def boundary_check(predict_boundary: Callable[[np.ndarray, Any], np.ndarray],
                   boundary_value: Callable[[Any], np.ndarray], rel_tol: float = 0.01, name: str = "boundary_condition",
                   critical: bool = False, scale: float | None = None) -> ValidityCheck:
    """Relative RMS mismatch between the prediction on the boundary and the boundary condition: RMS(a - b) divided
    by ``scale`` if given, else by the larger of the RMS of the boundary condition and the RMS of the whole
    prediction. (Dividing by the RMS of the boundary prediction itself would make a homogeneous condition, b = 0,
    fail for any non-zero value however small.)"""
    def fn(p, q):
        a = np.asarray(predict_boundary(p, q), dtype=float)
        b = np.asarray(boundary_value(q), dtype=float)
        if scale is not None:
            ref = float(scale)
        else:
            pv = np.asarray(p, dtype=float)
            pv = pv[np.isfinite(pv)] if pv.size else pv
            ref = max(float(np.sqrt(np.mean(b ** 2))) if b.size else 0.0,
                      float(np.sqrt(np.mean(pv ** 2))) if pv.size else 0.0)
        return float(np.sqrt(np.mean((a - b) ** 2))) / max(ref, 1e-300)
    return ValidityCheck(name, "boundary", fn, rel_tol, critical,
                         lambda v: f"boundary mismatch {100 * v:.3g} % (tolerance {100 * rel_tol:.3g} %)")


def constitutive_check(fn: Callable[[np.ndarray, Any], float], tolerance: float, name: str = "constitutive",
                       critical: bool = False) -> ValidityCheck:
    """Any relation the prediction must satisfy (equation of state, material law): ``fn`` returns its violation."""
    return ValidityCheck(name, "constitutive", fn, tolerance, critical)


def invariant_check(invariant_fn: Callable[[np.ndarray, Any], float], expected: float | Callable[[Any], float],
                    rel_tol: float = 0.01, name: str = "invariant", critical: bool = False,
                    abs_tol: float = 0.0) -> ValidityCheck:
    """Relative change of an invariant (energy of a conservative system, a symmetry measure). ``abs_tol`` as in
    :func:`conservation_check`, for an invariant whose expected value is zero (an antisymmetry measure)."""
    def fn(p, q):
        e = float(expected(q) if callable(expected) else expected)
        return _relative(float(invariant_fn(p, q)) - e, e, rel_tol, abs_tol)
    return ValidityCheck(name, "invariant", fn, rel_tol, critical)


def envelope_check(envelope: Envelope, features_fn: Callable[[Any], Mapping[str, float]], min_score: float = 0.5,
                   name: str = "training_envelope") -> ValidityCheck:
    """Extrapolation: 1 - envelope score (0 inside the training ranges, -> 1 far outside)."""
    return ValidityCheck(name, "envelope", lambda p, q: 1.0 - float(np.nan_to_num(envelope.check(features_fn(q))["score"], nan=1.0)),
                         1.0 - min_score, False, lambda v: f"envelope score {1 - v:.2f} (minimum {min_score:.2f})")


@dataclass
class ValidityReport:
    status: str
    results: list[CheckResult]
    alarms: list[str] = field(default_factory=list)
    index: int = 0

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    def by_kind(self, kind: str) -> list[CheckResult]:
        return [r for r in self.results if r.kind == kind]

    def lines(self) -> list[str]:
        out = [f"case {self.index}: {self.status}"]
        out += [f"  [{'ok' if r.passed else 'FAIL'}] {r.name} ({r.kind}): {r.detail}" for r in self.results]
        out += [f"  alarm: {a}" for a in self.alarms]
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())


class PhysicsValidityMonitor:
    """Run a set of checks on every prediction and keep their history.

    ``persistence``: a check failing this many cases in a row is an alarm. ``drift_ratio``: a residual check whose
    value, smoothed (EWMA with ``drift_alpha``), reaches this multiple of its baseline (median of the first
    ``baseline_cases`` cases, or ``baselines[name]``) raises an alarm even before it crosses its tolerance."""

    def __init__(self, checks: Sequence[ValidityCheck], persistence: int = 3, drift_ratio: float = 3.0,
                 drift_alpha: float = 0.3, baseline_cases: int = 10, baselines: Mapping[str, float] | None = None):
        names = [c.name for c in checks]
        if len(set(names)) != len(names):
            raise ValueError(f"check names must be unique: {names}")
        for c in checks:
            if c.kind not in KINDS:
                raise ValueError(f"unknown check kind {c.kind!r}; use one of {KINDS}")
        self.checks = list(checks)
        self.persistence = max(1, int(persistence))
        self.drift_ratio, self.drift_alpha = float(drift_ratio), float(drift_alpha)
        self.baseline_cases = int(baseline_cases)
        self.baselines = dict(baselines or {})
        self.history: list[ValidityReport] = []
        self._streak = {c.name: 0 for c in self.checks}
        self._ewma: dict[str, float] = {}
        self._values: dict[str, list[float]] = {c.name: [] for c in self.checks}

    def check(self, prediction: Any, query: Any = None) -> ValidityReport:
        p = np.asarray(prediction, dtype=float)
        results, alarms = [], []
        if not np.all(np.isfinite(p)):
            results.append(CheckResult("finite", "bounds", float(np.mean(~np.isfinite(p))), 0.0, False, True,
                                       "prediction has NaN or infinite values"))
        for c in self.checks:
            r = c(p, query)
            results.append(r)
            self._streak[c.name] = 0 if r.passed else self._streak[c.name] + 1
            if not r.passed and r.critical:
                alarms.append(f"{r.name}: critical check failed ({r.detail})")
            elif self._streak[c.name] >= self.persistence:
                alarms.append(f"{r.name}: failed {self._streak[c.name]} cases in a row")
            if c.kind == "residual" and np.isfinite(r.value):
                vals = self._values[c.name]
                vals.append(r.value)
                e = self._ewma.get(c.name)
                self._ewma[c.name] = r.value if e is None else (1 - self.drift_alpha) * e + self.drift_alpha * r.value
                base = self.baselines.get(c.name)
                if base is None and len(vals) >= self.baseline_cases:
                    base = self.baselines[c.name] = float(np.median(vals[:self.baseline_cases]))
                if base and self._ewma[c.name] >= self.drift_ratio * base and r.passed:
                    alarms.append(f"{r.name}: residual drifting, {self._ewma[c.name] / base:.1f} x its baseline")
        if any(r.critical and not r.passed for r in results) and not alarms:
            alarms.append("prediction is not finite")
        status = ALARM if alarms else (WARN if any(not r.passed for r in results) else OK)
        rep = ValidityReport(status, results, alarms, len(self.history))
        self.history.append(rep)
        return rep

    def summary(self) -> dict[str, Any]:
        n = len(self.history)
        fails: dict[str, int] = {}
        for rep in self.history:
            for r in rep.failed:
                fails[r.name] = fails.get(r.name, 0) + 1
        return {"cases": n, "ok": sum(r.status == OK for r in self.history),
                "warn": sum(r.status == WARN for r in self.history),
                "alarm": sum(r.status == ALARM for r in self.history),
                "failures_by_check": fails,
                "first_alarm": next((r.index for r in self.history if r.status == ALARM), None)}
