"""Physics residuals evaluated at several filter widths.

A residual computed at the grid scale only is one number: a field can satisfy the equations on average and violate
them at small scales, or be smooth at large scales and noisy at small ones. Filtering the field with ``G_Delta`` at
several widths and evaluating the residual of the *filtered* equations at each gives a curve against ``Delta``.

The filtered equations are not the original ones. For ``N(u)`` nonlinear, ``G * N(u) != N(G * u)``: the residual of an
exact field, filtered, is the subfilter term (the closure), not a model error. And when the width varies in space
the filter does not commute with derivatives, which adds a commutation term. Both are reported separately, so

    raw residual  =  subfilter term  +  commutation term  +  model residual

and only the last one is blamed on the model. The subfilter term is exact when the fine field is available
(``subfilter="exact"``, e.g. for a reference or a resolved prediction) and modelled otherwise (``"dynamic"``: the
dynamic Smagorinsky estimate from the filtered field itself, or any closure).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .filters import Filter, apply_filter

__all__ = ["MultiscaleResidual", "multiscale_residuals", "ns2d_multiscale_residuals"]


@dataclass
class MultiscaleResidual:
    filters: list[Filter]
    raw: np.ndarray                    # rms(R) / scale per width
    subfilter: np.ndarray              # rms(subfilter term) / scale
    commutation: np.ndarray            # rms(commutation term) / scale
    model: np.ndarray                  # rms(R - subfilter - commutation) / scale: the part blamed on the model

    @property
    def widths(self) -> np.ndarray:
        return np.array([f.delta for f in self.filters])

    def as_dict(self) -> dict[str, Any]:
        return {"filters": [f.label() for f in self.filters], "raw": self.raw.tolist(),
                "subfilter": self.subfilter.tolist(), "commutation": self.commutation.tolist(),
                "model": self.model.tolist()}

    def to_markdown(self) -> str:
        rows = ["| filter | raw residual | subfilter term | commutation term | model residual |",
                "|---|---|---|---|---|"]
        for i, f in enumerate(self.filters):
            rows.append(f"| {f.label()} | {self.raw[i]:.3g} | {self.subfilter[i]:.3g} | {self.commutation[i]:.3g} "
                        f"| {self.model[i]:.3g} |")
        return "\n".join(rows)


def multiscale_residuals(state: np.ndarray, residual_fn: Callable[[np.ndarray, Filter], tuple[np.ndarray, float]],
                         filters: Sequence[Filter], *, length: float = 2 * np.pi,
                         subfilter_fn: Callable[[np.ndarray, Filter], np.ndarray] | None = None,
                         commutation_fn: Callable[[np.ndarray, Filter], np.ndarray] | None = None,
                         filter_fn: Callable[[np.ndarray, Filter], np.ndarray] | None = None) -> MultiscaleResidual:
    """Residual curve of any equation. ``residual_fn(filtered_state, filter) -> (R, scale)`` evaluates the residual
    field of the filtered equations and a normalising scale; ``subfilter_fn(state, filter)`` the subfilter term
    (from the unfiltered ``state``, or modelled); ``commutation_fn(state, filter)`` the commutation term;
    ``filter_fn`` (default: uniform filtering of the last two axes) how ``state`` is filtered."""
    filter_fn = filter_fn or (lambda s, f: apply_filter(s, f, length))
    raw, sub, com, mod = [], [], [], []
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))  # noqa: E731
    for f in filters:
        R, scale = residual_fn(filter_fn(state, f), f)
        S = subfilter_fn(state, f) if subfilter_fn is not None else 0.0
        C = commutation_fn(state, f) if commutation_fn is not None else 0.0
        raw.append(rms(R) / scale)
        sub.append(rms(S) / scale)
        com.append(rms(C) / scale)
        mod.append(rms(R - S - C) / scale)
    return MultiscaleResidual(list(filters), np.array(raw), np.array(sub), np.array(com), np.array(mod))


def ns2d_multiscale_residuals(w_prev: np.ndarray, w_next: np.ndarray, dt: float, nu: float,
                              filters: Sequence[Filter], *, length: float = 2 * np.pi,
                              forcing: np.ndarray | None = None, drag: float = 0.0,
                              subfilter: str | Callable | None = "exact") -> MultiscaleResidual:
    """Multiscale residual of the 2-D vorticity equation between two snapshots ``dt`` apart (see
    :func:`~.ns2d.vorticity_residual`). ``subfilter``: ``"exact"`` (computed from the given, unfiltered snapshots:
    for a resolved field, the model residual is then the true error of the filtered dynamics), ``"dynamic"``
    (dynamic Smagorinsky with the filter's width and kind, from the filtered snapshots only), a closure object
    (``closure(w, solver) -> tendency`` with a ``delta`` attribute), or None (raw residual only). The filters are
    uniform, so the commutation term is zero; use :func:`multiscale_residuals` with
    :func:`~.filters.commutation_error` for variable widths."""
    from .closures import DynamicSmagorinsky, true_subfilter
    from .ns2d import VorticityNS2D, vorticity_residual

    pair = np.stack([np.asarray(w_prev, float), np.asarray(w_next, float)])
    n = pair.shape[-1]
    ffilt = None if forcing is None else forcing

    def res(fp, f):
        return vorticity_residual(fp[0], fp[1], dt, nu, length, None if ffilt is None else apply_filter(ffilt, f, length),
                                  drag)

    def sub(state, f):
        if subfilter is None or f.delta == 0:
            return 0.0
        if subfilter == "exact":
            return 0.5 * (true_subfilter(state[0], f, None, length)["tendency"]
                          + true_subfilter(state[1], f, None, length)["tendency"])
        solver = VorticityNS2D(n, nu, length, dealias=False)
        cl = DynamicSmagorinsky(delta=f.delta, test_kind=f.kind) if subfilter == "dynamic" else subfilter
        old = getattr(cl, "delta", None)
        try:
            if hasattr(cl, "delta"):
                cl.delta = f.delta
            fs = apply_filter(state, f, length)
            return 0.5 * (cl(fs[0], solver) + cl(fs[1], solver))
        finally:
            if hasattr(cl, "delta"):
                cl.delta = old

    return multiscale_residuals(pair, res, filters, length=length, subfilter_fn=sub)
