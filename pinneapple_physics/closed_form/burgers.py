"""Exact solution of the viscous Burgers equation with u(x, 0) = -sin(pi x), u(+-1, t) = 0 on x in [-1, 1].

Cole-Hopf transform (Basdevant et al., Computers & Fluids 14, 1986): with phi_0(y) = exp(-cos(pi y) / (2 pi nu)),

    u(x, t) = - int sin(pi (x - eta)) phi_0(x - eta) G(eta, t) d eta / int phi_0(x - eta) G(eta, t) d eta,

G the heat kernel with variance 2 nu t. Substituting eta = sqrt(4 nu t) z turns both integrals into
Gauss-Hermite quadratures. The constant exp(1 / (2 pi nu)) is factored out of phi_0 (it cancels) so the exponent
stays non-negative and bounded: no overflow for nu down to about 1e-3 with the default 120 nodes.
This is the reference used by the PINN literature for the nu = 0.01/pi benchmark (Raissi et al., 2019).

:func:`burgers_sine_inviscid_exact` is the nu -> 0 limit: the entropy (vanishing-viscosity) solution of the inviscid
equation with the same data, by characteristics (Whitham, *Linear and Nonlinear Waves*, 1974, ch. 2).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["burgers_sine_exact", "burgers_sine_inviscid_exact"]


def burgers_sine_exact(x, t, nu: float, n_nodes: int = 120) -> np.ndarray:
    """``u(x, t)`` for arrays ``x``, ``t`` of the same shape (broadcast). ``t = 0`` returns the initial condition."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    x, t = np.broadcast_arrays(x, t)
    if nu <= 0:
        raise ValueError("nu must be positive")
    z, w = np.polynomial.hermite.hermgauss(n_nodes)
    xf, tf = x.ravel(), t.ravel()
    s = np.sqrt(4.0 * nu * np.maximum(tf, 0.0))
    y = xf[:, None] - s[:, None] * z[None, :]
    # log phi_0 shifted by its maximum 1/(2 pi nu): exponent (1 - cos(pi y)) / (2 pi nu) is in [0, 1/(pi nu)];
    # subtract the per-point maximum before exponentiating so the largest term is exp(0) = 1.
    e = (1.0 - np.cos(np.pi * y)) / (2.0 * np.pi * nu)
    e -= e.max(axis=1, keepdims=True)
    phi = w[None, :] * np.exp(e)
    u = -np.sum(phi * np.sin(np.pi * y), axis=1) / np.sum(phi, axis=1)
    u = np.where(tf <= 0.0, -np.sin(np.pi * xf), u)
    return u.reshape(x.shape)


def burgers_sine_inviscid_exact(x, t) -> np.ndarray:
    """Entropy solution ``u(x, t)`` of ``u_t + u u_x = 0``, ``u(x, 0) = -sin(pi x)``, ``u(+-1, t) = 0`` on [-1, 1].

    Along characteristics u is constant: ``u(x, t) = -sin(pi x0)`` with ``x = x0 - t sin(pi x0)`` (Whitham 1974,
    §2.1). Characteristics first cross at ``t = 1 / pi`` (min of ``1 / -u0'``), at x = 0; by odd symmetry the
    entropy shock then stays at x = 0, so the foot ``x0`` is taken on the same side as ``x`` (where
    ``x0 - t sin(pi x0) - x`` is convex and changes sign exactly once) and found with Brent's method.
    ``u = 0`` at the shock and at the walls; ``t = 0`` returns the initial condition. ``x``, ``t`` broadcast.
    """
    from scipy.optimize import brentq

    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    x, t = np.broadcast_arrays(x, t)
    if np.any(t < 0):
        raise ValueError("t must be non-negative")
    out = np.empty(x.shape)
    for i, (xi, ti) in enumerate(zip(x.ravel().tolist(), t.ravel().tolist(), strict=True)):
        if ti == 0.0:
            u = -math.sin(math.pi * xi)
        elif xi == 0.0 or abs(xi) >= 1.0:  # shock location and the fixed walls
            u = 0.0
        else:
            lo, hi = (0.0, 1.0) if xi > 0 else (-1.0, 0.0)
            x0 = brentq(lambda s, xi=xi, ti=ti: s - ti * math.sin(math.pi * s) - xi, lo, hi, xtol=1e-14)
            u = -math.sin(math.pi * x0)
        out.flat[i] = u
    return out
