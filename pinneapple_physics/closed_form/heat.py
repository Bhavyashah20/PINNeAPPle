"""Transient 1-D heat conduction in a slab, closed form.

``T_t = alpha T_xx`` on ``0 < x < L`` with fixed end temperatures ``T(0, t) = T_left``, ``T(L, t) = T_right`` and a
uniform initial temperature ``T(x, 0) = T0``. Splitting off the linear steady profile and expanding the rest in the
Dirichlet eigenfunctions ``sin(n pi x / L)`` gives (Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed., 1959,
§3.3-3.4)

    T = T_left + (T_right - T_left) x / L
        + sum_n (2 / (n pi)) [ (T0 - T_left) (1 - (-1)^n) + (T_right - T_left) (-1)^n ] sin(n pi x / L)
          exp(-alpha (n pi / L)^2 t).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["slab_transient"]


def slab_transient(x, t, L: float, alpha: float, T_left: float, T_right: float, T0: float,
                   n_terms: int | None = None) -> np.ndarray:
    """Temperature ``T(x, t)`` [same unit as the inputs] of the slab problem above; ``x`` [m] and ``t`` [s] broadcast,
    ``alpha`` thermal diffusivity [m^2/s].

    ``n_terms`` defaults to the number of modes whose decay factor at the smallest ``t > 0`` is above 1e-16 (capped
    at 20000), so the truncation error is at round-off level away from ``t = 0``. At ``t = 0`` the initial value
    ``T0`` is returned inside the slab and the wall values at its ends.
    """
    if not (L > 0 and alpha > 0):
        raise ValueError("L and alpha must be positive")
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    x, t = np.broadcast_arrays(x, t)
    if np.any(t < 0):
        raise ValueError("t must be non-negative")
    xf, tf = x.ravel(), t.ravel()
    steady = T_left + (T_right - T_left) * xf / L
    out = steady.copy()
    pos = tf > 0
    if pos.any():
        if n_terms is None:
            t_min = float(tf[pos].min())
            n_terms = int(min(20000, math.ceil(L / math.pi * math.sqrt(37.0 / (alpha * t_min))) + 1))
        n = np.arange(1, n_terms + 1, dtype=np.float64)
        sign = (-1.0) ** n
        b = 2.0 / (n * math.pi) * ((T0 - T_left) * (1.0 - sign) + (T_right - T_left) * sign)
        k = n * math.pi / L
        xp, tp = xf[pos], tf[pos]
        chunk = max(1, 2_000_000 // n_terms)
        vals = np.empty(len(xp))
        for s in range(0, len(xp), chunk):
            sl = slice(s, s + chunk)
            vals[sl] = (np.sin(np.outer(xp[sl], k)) * np.exp(-alpha * np.outer(tp[sl], k * k))) @ b
        out[pos] += vals
    zero = ~pos
    interior = (xf > 0) & (xf < L)
    out[zero & interior] = T0
    return out.reshape(x.shape)
