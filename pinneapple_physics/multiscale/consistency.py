"""Cross-scale consistency: a fine and a coarse prediction of the same state must agree once both are brought to
the same scale.

The fine prediction is filtered to the coarse prediction's filter (with :func:`~.filters.refilter`, exact when the
fine filter is narrower and of a composable kind) and restricted to the coarse grid; then the two are compared on

* the field (relative L2),
* the energy spectrum over the shared wavenumbers (mean ``|log10 E_fine / E_coarse|``),
* integral quantities (mean, energy, enstrophy, or any ``{name: fn(field)}``), each with its own relative
  tolerance; the report names the integral that broke.

Tolerances can be scaled by the known discretisation error (``gci``: e.g. the grid-convergence index of the coarse
model, #437): a pair is only declared inconsistent beyond ``max(tol, safety * gci)``. Works for solver/solver,
surrogate/surrogate and surrogate/solver pairs.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .filters import Filter, refilter, restrict

__all__ = ["ConsistencyReport", "cross_scale_consistency", "VORTICITY_INTEGRALS"]


def _mean(w):
    return float(np.mean(w))


def _enstrophy(w):
    return 0.5 * float(np.mean(np.asarray(w) ** 2))


def _energy(w):
    from .ns2d import velocity_from_vorticity
    u, v = velocity_from_vorticity(np.asarray(w, float))
    return 0.5 * float(np.mean(u ** 2 + v ** 2))


VORTICITY_INTEGRALS: dict[str, Callable[[np.ndarray], float]] = {"circulation (mean)": _mean, "energy": _energy,
                                                                  "enstrophy": _enstrophy}


@dataclass
class ConsistencyReport:
    passed: bool
    field_error: float
    spectrum_error: float
    integrals: dict[str, dict[str, float]]
    failed: list[str]
    tolerance: float
    reasons: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """1 when consistent, decaying with the worst excess over the tolerances (for ensemble checks)."""
        ex = [self.field_error / self.tolerance] + [v["rel_diff"] / v["tol"] for v in self.integrals.values()
                                                    if v["tol"] > 0]
        worst = max(ex) if ex else 0.0
        return 1.0 if worst <= 1 else float(np.exp(-(worst - 1) ** 2))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __str__(self) -> str:
        head = "consistent" if self.passed else "INCONSISTENT"
        return f"{head}: field {self.field_error:.3g} (tol {self.tolerance:.3g}), spectrum {self.spectrum_error:.3g}" + \
            ("" if not self.reasons else "; " + "; ".join(self.reasons))


def cross_scale_consistency(fine: np.ndarray, coarse: np.ndarray, coarse_filter: Filter,
                            fine_filter: Filter | None = None, *, length: float = 2 * np.pi,
                            tol: float = 0.05, integrals: Mapping[str, Callable[[np.ndarray], float]] | None = None,
                            integral_tol: float | Mapping[str, float] = 0.02, gci: float | None = None,
                            safety: float = 1.25, spectrum: bool = True) -> ConsistencyReport:
    """Compare ``fine`` (filtered by ``fine_filter``, default unfiltered, on its grid) with ``coarse`` (filtered by ``coarse_filter``, on
    a grid of equal or fewer points) after bringing the fine one to the coarse scale and grid. Arrays are ``(n, n)``
    or with leading axes (e.g. snapshots). ``integrals`` default to :data:`VORTICITY_INTEGRALS` for 2-D fields."""
    fine, coarse = np.asarray(fine, float), np.asarray(coarse, float)
    fine_filter = fine_filter if fine_filter is not None else Filter("sharp", 0.0)
    f2c = refilter(fine, fine_filter, coarse_filter, length)
    if f2c.shape[-1] != coarse.shape[-1]:
        f2c = restrict(f2c, coarse.shape[-1])
    t = max(tol, safety * gci) if gci is not None else tol
    ferr = float(np.linalg.norm(f2c - coarse) / max(np.linalg.norm(coarse), 1e-300))
    serr = float("nan")
    if spectrum and coarse.ndim >= 2:
        from .closures import spectrum_error
        kc = min(coarse_filter.cutoff, np.pi * coarse.shape[-1] / length) * length / (2 * np.pi)
        serr = spectrum_error(f2c, coarse, kc, length)
    integrals = VORTICITY_INTEGRALS if integrals is None else integrals
    rep, failed, reasons = {}, [], []
    for name, fn in integrals.items():
        a = np.mean([fn(x) for x in f2c.reshape(-1, *f2c.shape[-2:])])
        b = np.mean([fn(x) for x in coarse.reshape(-1, *coarse.shape[-2:])])
        scale = max(abs(a), abs(b))
        if name.startswith("circulation"):                       # zero-mean quantities: relative to the field rms
            scale = float(np.sqrt(np.mean(coarse ** 2)))
        floor = 1e-10 * max(1.0, float(np.sqrt(np.mean(coarse ** 2))))  # both zero to round-off: equal
        rd = 0.0 if abs(a - b) <= floor else abs(a - b) / max(scale, 1e-300)
        it = integral_tol.get(name, 0.02) if isinstance(integral_tol, Mapping) else float(integral_tol)
        it = max(it, safety * gci) if gci is not None else it
        ok = rd <= it
        rep[name] = {"fine": float(a), "coarse": float(b), "rel_diff": float(rd), "tol": float(it), "ok": bool(ok)}
        if not ok:
            failed.append(name)
            reasons.append(f"{name} differs by {100 * rd:.2g} % (tol {100 * it:.2g} %)")
    if ferr > t:
        reasons.insert(0, f"field differs by {100 * ferr:.3g} % after filtering (tol {100 * t:.3g} %)")
    return ConsistencyReport(bool(ferr <= t and not failed), ferr, serr, rep, failed, t, reasons)
