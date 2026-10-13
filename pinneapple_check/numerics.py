"""Derivatives of named scalar fields on three kinds of discretisation, behind one interface.

A *provider* holds scalar components by name (a vector ``U`` becomes ``u, v, w``) and gives ``val(name)``,
``d(name, axis)``, ``dd(name, a, b)`` and ``dt(name)``, plus ``interior`` (a mask of the points where the stencil is
fully inside the data) and ``where(i)`` (coordinates of point ``i`` for messages). The physics residuals in
:mod:`pinneapple_check.physics` are written once against this interface.

* :class:`GridD`: second-order finite differences (``numpy.gradient``) on a tensor-product grid, time optional.
* :class:`PointD`: weighted least-squares quadratic fit over the nearest neighbours of each point.
* :class:`ModelD`: central differences of a model's predictions at sampled points.

Finite-volume (OpenFOAM) residuals are conservative sums over faces and live in :mod:`pinneapple_check.fv`.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import combinations_with_replacement

import numpy as np

__all__ = ["GridD", "PointD", "ModelD", "components", "rel_rms"]

VEC_NAMES = {"U": ("u", "v", "w"), "u": ("u", "v", "w"), "velocity": ("u", "v", "w"), "vel": ("u", "v", "w"),
             "D": ("ux", "uy", "uz"), "disp": ("ux", "uy", "uz"), "displacement": ("ux", "uy", "uz")}


def components(fields: dict[str, np.ndarray], vector_ndim: int, ncomp: int = 3) -> dict[str, np.ndarray]:
    """Split vector fields into named scalars: ``U`` -> ``u, v, w``, ``D`` -> ``ux, uy, uz``, another vector ``F``
    -> ``F_x, F_y, F_z``; ``Ux``/``U_x`` style scalars -> ``u``... Scalars keep their names."""
    out: dict[str, np.ndarray] = {}
    for k, v in fields.items():
        v = np.asarray(v, float)
        if v.ndim == vector_ndim and v.shape[-1] in (2, 3):
            names = VEC_NAMES.get(k, tuple(f"{k}_{a}" for a in "xyz"))
            for i in range(min(v.shape[-1], ncomp)):
                out.setdefault(names[i], v[..., i])
        else:
            out.setdefault(k, v)
    for base, names in (("U", ("u", "v", "w")), ("D", ("ux", "uy", "uz"))):
        for sep in ("", "_"):
            for a, n in zip("xyz", names, strict=False):
                if f"{base}{sep}{a}" in fields and n not in out:
                    out[n] = np.asarray(fields[f"{base}{sep}{a}"], float)
    return out


def rel_rms(residual: np.ndarray, terms: list[np.ndarray], mask: np.ndarray | None = None,
            floor: float = 0.0) -> float:
    """RMS of the residual over RMS of the sum of the absolute terms: 0 = exact, 1 = terms do not balance.
    ``floor`` bounds the denominator from below (a characteristic size of the terms), so a field whose terms all
    vanish (a linear profile without source) is not judged on round-off."""
    r = np.asarray(residual, float)
    s = np.sum([np.abs(np.asarray(t, float)) for t in terms], axis=0)
    if mask is not None:
        r, s = r[mask], s[mask]
    else:
        r, s = r.ravel(), s.ravel()
    if r.ndim > 1:
        r, s = np.linalg.norm(r, axis=-1), np.linalg.norm(s, axis=-1)
    num = float(np.sqrt(np.mean(r ** 2))) if r.size else 0.0
    den = max(float(np.sqrt(np.mean(s ** 2))) if s.size else 0.0, floor)
    if den <= 1e-300:
        return 0.0 if num <= 1e-300 else float("inf")
    return num / den


class _Base:
    axes: list[str]          # spatial axis names in order
    has_time: bool = False
    length: float = 1.0      # size of the domain (largest span of the spatial axes)
    duration: float = 1.0    # time span (when there is a time axis)
    comps: dict[str, np.ndarray]

    def has(self, *names: str) -> bool:
        return all(n in self.comps for n in names)

    def val(self, name: str) -> np.ndarray:
        return self.comps[name]


# ---------------------------------------------------------------------------------------------- grid
class GridD(_Base):
    def __init__(self, axes: dict[str, np.ndarray], comps: dict[str, np.ndarray]):
        self.axis_names = list(axes)
        self.axes = [a for a in axes if a != "t"]
        self.has_time = "t" in axes
        self.coords = [np.asarray(axes[a], float) for a in axes]
        self.comps = {k: v for k, v in comps.items() if v.shape == tuple(len(c) for c in self.coords)}
        self._cache: dict = {}
        shape = tuple(len(c) for c in self.coords)
        m = np.ones(shape, bool)
        for i, a in enumerate(self.axis_names):                 # drop the boundary layer of every spatial axis
            if a == "t":
                continue
            sl = [slice(None)] * len(shape)
            sl[i] = [0, -1] if shape[i] > 2 else slice(None)
            m[tuple(sl)] = False
        self.interior = m
        self.length = max([float(np.ptp(axes[a])) for a in self.axes] or [1.0]) or 1.0
        self.duration = float(np.ptp(axes["t"])) if self.has_time else 1.0

    def _ax(self, a: str) -> int:
        return self.axis_names.index(a)

    def d(self, name: str, a: str) -> np.ndarray:
        key = ("d", name, a)
        if key not in self._cache:
            i = self._ax(a)
            self._cache[key] = np.gradient(self.comps[name], self.coords[i], axis=i, edge_order=2)
        return self._cache[key]

    def dd(self, name: str, a: str, b: str) -> np.ndarray:
        key = ("dd", name, *sorted((a, b)))
        if key not in self._cache:
            j = self._ax(b)
            self._cache[key] = np.gradient(self.d(name, a), self.coords[j], axis=j, edge_order=2)
        return self._cache[key]

    def dt(self, name: str) -> np.ndarray:
        return self.d(name, "t")

    def where(self, flat_index: int) -> str:
        idx = np.unravel_index(flat_index, self.interior.shape)
        return ", ".join(f"{a}={self.coords[k][i]:.4g}" for k, (a, i) in enumerate(zip(self.axis_names, idx, strict=False)))


# ---------------------------------------------------------------------------------------------- point cloud
class PointD(_Base):
    """Weighted least-squares quadratic fit around each point (``k`` neighbours). Points whose neighbourhood is
    one-sided (on the hull of the cloud) are left out of ``interior``."""

    def __init__(self, points: np.ndarray, comps: dict[str, np.ndarray], axes: list[str], k: int | None = None):
        P = np.asarray(points, float)
        self.axes = axes
        idx = [("xyz").index(a) for a in axes]
        X = P[:, idx]
        n, d = X.shape
        self.X = X
        self.comps = {kk: v for kk, v in comps.items() if v.shape[:1] == (n,) and v.ndim == 1}
        npoly = 1 + d + d * (d + 1) // 2
        k = k or min(n, 2 * npoly + 4)
        try:
            from scipy.spatial import cKDTree
            _, nb = cKDTree(X).query(X, k=k)
        except ImportError:                                           # pragma: no cover - scipy is a dependency
            D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
            nb = np.argsort(D, axis=1)[:, :k]
        self.nb = nb
        dx = X[nb] - X[:, None, :]                                    # (n, k, d)
        h = np.sqrt((dx ** 2).sum(-1)).max(1, keepdims=True) + 1e-300
        cols = [np.ones(dx.shape[:2])] + [dx[..., i] for i in range(d)]
        self.pairs = list(combinations_with_replacement(range(d), 2))
        cols += [dx[..., i] * dx[..., j] * (0.5 if i == j else 1.0) for i, j in self.pairs]
        A = np.stack(cols, axis=-1)                                   # (n, k, npoly)
        w = np.exp(-((np.sqrt((dx ** 2).sum(-1)) / h) ** 2))
        Aw = A * w[..., None]
        # pseudo-inverse per point: coefficients = pinv(Aw) @ (w * values)
        self.pinv = np.linalg.pinv(Aw)                                # (n, npoly, k)
        self.w = w
        centred = np.abs((dx / h[..., None]).mean(1)).max(1)
        self.interior = centred < 0.25
        self.length = float(np.ptp(X, axis=0).max()) or 1.0
        self._coef: dict[str, np.ndarray] = {}

    def _c(self, name: str) -> np.ndarray:
        if name not in self._coef:
            v = self.comps[name][self.nb] * self.w
            self._coef[name] = np.einsum("npk,nk->np", self.pinv, v)
        return self._coef[name]

    def d(self, name: str, a: str) -> np.ndarray:
        return self._c(name)[:, 1 + self.axes.index(a)]

    def dd(self, name: str, a: str, b: str) -> np.ndarray:
        i, j = sorted((self.axes.index(a), self.axes.index(b)))
        return self._c(name)[:, 1 + len(self.axes) + self.pairs.index((i, j))]

    def dt(self, name: str) -> np.ndarray:
        raise KeyError("a point cloud has no time axis")

    def where(self, i: int) -> str:
        return ", ".join(f"{a}={self.X[i, k]:.4g}" for k, a in enumerate(self.axes))


# ---------------------------------------------------------------------------------------------- model
class ModelD(_Base):
    """Central differences of ``f`` (``(N, n_in) -> dict of (N,)``) at the points ``X``. ``axes`` are the inputs
    differentiated (spatial names, plus ``t`` if it is an input); other inputs are held fixed."""

    def __init__(self, f: Callable[[np.ndarray], dict[str, np.ndarray]], X: np.ndarray, inputs: list[str],
                 h: dict[str, float], spans: dict[str, float] | None = None):
        self.f, self.X, self.inputs, self.h = f, np.asarray(X, float), inputs, h
        spans = spans or {}
        self.axes = [a for a in inputs if a in ("x", "y", "z")]
        self.has_time = "t" in inputs
        self.comps = f(self.X)
        self.interior = np.ones(len(self.X), bool)
        self.length = max([spans.get(a, 0.0) for a in self.axes] or [1.0]) or 1.0
        self.duration = spans.get("t", 1.0) or 1.0
        self._cache: dict = {}

    def _shift(self, steps: dict[str, float]) -> dict[str, np.ndarray]:
        key = tuple(sorted(steps.items()))
        if key not in self._cache:
            Y = self.X.copy()
            for a, s in steps.items():
                Y[:, self.inputs.index(a)] += s
            self._cache[key] = self.f(Y)
        return self._cache[key]

    def d(self, name: str, a: str) -> np.ndarray:
        h = self.h[a]
        return (self._shift({a: h})[name] - self._shift({a: -h})[name]) / (2 * h)

    def dd(self, name: str, a: str, b: str) -> np.ndarray:
        if a == b:
            h = self.h[a]
            return (self._shift({a: h})[name] - 2 * self.comps[name] + self._shift({a: -h})[name]) / h ** 2
        ha, hb = self.h[a], self.h[b]
        pp, mm = self._shift({a: ha, b: hb})[name], self._shift({a: -ha, b: -hb})[name]
        pm, mp = self._shift({a: ha, b: -hb})[name], self._shift({a: -ha, b: hb})[name]
        return (pp - pm - mp + mm) / (4 * ha * hb)

    def dt(self, name: str) -> np.ndarray:
        return self.d(name, "t")

    def where(self, i: int) -> str:
        return ", ".join(f"{a}={self.X[i, k]:.4g}" for k, a in enumerate(self.inputs))
