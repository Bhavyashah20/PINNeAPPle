"""Prism (inflation) layer sizing: a geometric stack of wall layers.

Layer ``k`` (1-based, from the wall) has height ``h_k = h1 r^(k-1)``, so a stack of ``n`` layers with expansion
ratio ``r`` is ``total = h1 (r^n - 1) / (r - 1)`` thick (``n h1`` for ``r = 1``), the finite geometric series used by
every mesher's layer controls (e.g. snappyHexMesh ``expansionRatio``/``firstLayerThickness``; Ferziger, Peric &
Street, *Computational Methods for Fluid Dynamics*, 4th ed., 2020, §3.10 on stretched grids).
"""
from __future__ import annotations

import math

__all__ = ["layer_total", "size_layers"]


def layer_total(first_height: float, expansion: float, n_layers: int) -> float:
    """Total thickness ``h1 (r^n - 1) / (r - 1)`` of ``n`` layers (``n h1`` when ``r = 1``)."""
    if abs(expansion - 1.0) < 1e-12:
        return first_height * n_layers
    return first_height * (expansion ** n_layers - 1.0) / (expansion - 1.0)


def size_layers(first_height: float, expansion: float | None = None, n_layers: int | None = None,
                total_thickness: float | None = None) -> dict[str, float]:
    """Solve the geometric layer stack for the one missing quantity among ``expansion``, ``n_layers`` and
    ``total_thickness`` (exactly one must be ``None``).

    - missing ``total_thickness``: the series sum;
    - missing ``n_layers``: ``n = log(1 + T (r - 1) / h1) / log r``, rounded up to an integer, and the total is
      recomputed for that integer count (so it is at least the requested one);
    - missing ``expansion``: the root ``r > 0`` of ``h1 (r^n - 1) / (r - 1) = T`` (Brent's method; ``r < 1`` when
      ``T < n h1``).

    Returns ``first_height``, ``expansion``, ``n_layers``, ``total_thickness`` and ``last_height`` (``h1 r^(n-1)``).
    """
    missing = [k for k, v in (("expansion", expansion), ("n_layers", n_layers), ("total_thickness", total_thickness))
               if v is None]
    if len(missing) != 1:
        raise ValueError("give exactly two of expansion, n_layers and total_thickness")
    if not first_height > 0:
        raise ValueError("first_height must be positive")
    if expansion is not None and not expansion > 0:
        raise ValueError("expansion must be positive")
    if n_layers is not None and (int(n_layers) != n_layers or n_layers < 1):
        raise ValueError("n_layers must be a positive integer")
    if total_thickness is not None and not total_thickness > 0:
        raise ValueError("total_thickness must be positive")

    if total_thickness is None:
        n = int(n_layers)
        r = float(expansion)
    elif n_layers is None:
        r = float(expansion)
        ratio = total_thickness / first_height
        if abs(r - 1.0) < 1e-12:
            n_real = ratio
        else:
            arg = 1.0 + ratio * (r - 1.0)
            if arg <= 0:  # r < 1: the stack converges to h1 / (1 - r) and never reaches the total
                raise ValueError(f"expansion {r} < 1 cannot reach total_thickness {total_thickness}")
            n_real = math.log(arg) / math.log(r)
        n = max(1, math.ceil(n_real - 1e-9))
    else:
        from scipy.optimize import brentq

        n = int(n_layers)
        ratio = total_thickness / first_height
        if ratio < 1.0 - 1e-12:
            raise ValueError("total_thickness is smaller than first_height")
        if n == 1:
            if abs(ratio - 1.0) > 1e-12:
                raise ValueError("one layer: total_thickness must equal first_height")
            r = 1.0
        elif abs(ratio - n) < 1e-12 * n:
            r = 1.0
        else:
            hi = max(ratio ** (1.0 / (n - 1)), 1.0) + 1e-9   # sum >= r^(n-1), so the root is below this
            r = brentq(lambda q: layer_total(1.0, q, n) - ratio, 1e-12, hi, xtol=1e-15, rtol=1e-15)
    total = layer_total(first_height, r, n)
    return {"first_height": float(first_height), "expansion": float(r), "n_layers": int(n),
            "total_thickness": float(total), "last_height": float(first_height * r ** (n - 1))}
