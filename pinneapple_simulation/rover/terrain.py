"""Lunar terrain for rover simulation: a height function with a planar slope, craters and rocks, plus soil regions.

Coordinates are metres, z up. The height is an analytic function ``h(x, y)`` (so contact geometry is exact and needs
no mesh); :meth:`LunarTerrain.grid` samples it for plots and :meth:`LunarTerrain.mesh` for the 3D viewer.

Crater shape
------------
Bowl ``h = h_rim + d (rho^2 - 1)`` inside the rim (``rho = r / (D/2)``) and an ejecta flank ``h = h_rim rho^-3``
outside it, joined by a smooth minimum so the crest is rounded over ``crest_width``.

* Depth: the default depth/diameter ratio is 0.2, the value Pike (1977) reports for fresh simple lunar craters
  (D below about 15 km): Pike, R.J. (1977), "Apparent depth/apparent diameter relation for lunar craters",
  Proc. Lunar Sci. Conf. 8th, 3427-3436. Craters a few metres across are mostly degraded and shallower; pass
  ``depth_ratio`` for those (no published small-crater ratio is built in).
* Flank: the ``rho^-3`` decay is the radial decay of ejecta thickness of McGetchin, Settle & Head (1973),
  "Radial thickness variation in impact crater ejecta: implications for lunar basin deposits",
  Earth Planet. Sci. Lett. 20, 226-236 (thickness proportional to (r/R)^-3).
* Rim height: a required input (``rim_ratio`` = h_rim / D). No published value is built in.

Rocks are smooth bumps ``H (1 - (r/a)^2)^2``. They are geometry only: a wheel on a rock still gets the regolith
law of the region (no rigid rock contact), so runs over rocks are qualitative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..particle_dynamics.terramechanics import SoilParams

PIKE_1977_DEPTH_RATIO = 0.2       # fresh simple lunar craters, D < ~15 km (Pike 1977)


@dataclass
class Crater:
    x: float
    y: float
    diameter: float
    rim_ratio: float                       # rim height / diameter (input, no default)
    depth_ratio: float = PIKE_1977_DEPTH_RATIO   # rim crest to floor / diameter
    crest_width: float = 0.05              # rounding of the crest, as a fraction of the diameter

    def height(self, x, y):
        Rc = 0.5 * self.diameter
        rho = np.hypot(np.asarray(x, float) - self.x, np.asarray(y, float) - self.y) / Rc
        h_rim = self.rim_ratio * self.diameter
        d = self.depth_ratio * self.diameter
        inner = h_rim + d * (rho ** 2 - 1.0)
        outer = h_rim * np.maximum(rho, 1e-6) ** -3.0
        k = self.crest_width * self.diameter
        hh = np.clip(0.5 + 0.5 * (outer - inner) / k, 0.0, 1.0)      # polynomial smooth minimum
        return outer * (1 - hh) + inner * hh - k * hh * (1 - hh)

    def contains(self, x, y):
        return np.hypot(np.asarray(x) - self.x, np.asarray(y) - self.y) <= 0.5 * self.diameter


@dataclass
class Rock:
    x: float
    y: float
    radius: float
    height: float

    def height_at(self, x, y):
        q = ((np.asarray(x, float) - self.x) ** 2 + (np.asarray(y, float) - self.y) ** 2) / self.radius ** 2
        return self.height * np.clip(1.0 - q, 0.0, None) ** 2


@dataclass
class SoilRegion:
    """Soil inside a circle (``kind="circle"``: centre, radius) or an axis-aligned box (``"box"``: x0, x1, y0, y1)."""
    soil: SoilParams
    kind: str = "circle"
    params: tuple = (0.0, 0.0, 1.0)
    name: str = ""

    def contains(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.kind == "circle":
            cx, cy, r = self.params
            return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
        if self.kind == "box":
            x0, x1, y0, y1 = self.params
            return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        raise ValueError(f"unknown region kind {self.kind!r}")


@dataclass
class LunarTerrain:
    """Height function and soil map.

    ``slope_deg``: planar grade rising along ``slope_dir_deg`` (0 = +x). ``ramp``: optional (x0, x1, a0_deg, a1_deg),
    a grade along x that grows linearly from a0 at x0 to a1 at x1 (constant outside), for climb-until-failure runs.
    The default soil is GRC-1 (``SoilParams()``, the values of the terramechanics use case); ``regions`` override it.
    """
    slope_deg: float = 0.0
    slope_dir_deg: float = 0.0
    ramp: tuple | None = None
    craters: list[Crater] = field(default_factory=list)
    rocks: list[Rock] = field(default_factory=list)
    soil: SoilParams = field(default_factory=SoilParams)
    regions: list[SoilRegion] = field(default_factory=list)

    # ------------------------------------------------------------------ height
    def _ramp(self, x):
        x0, x1, a0, a1 = self.ramp
        t0, t1 = math.tan(math.radians(a0)), math.tan(math.radians(a1))
        k = (t1 - t0) / (x1 - x0)                            # d(tan a)/dx, with tan a linear in x on [x0, x1]
        xc = np.clip(x, x0, x1)
        h = t0 * (xc - x0) + 0.5 * k * (xc - x0) ** 2      # integral of tan a(x)
        h = h + np.where(x > x1, t1 * (x - x1), 0.0) + np.where(x < x0, t0 * (x - x0), 0.0)
        return h

    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        a = math.radians(self.slope_dir_deg)
        h = math.tan(math.radians(self.slope_deg)) * (x * math.cos(a) + y * math.sin(a))
        if self.ramp is not None:
            h = h + self._ramp(x)
        for c in self.craters:
            h = h + c.height(x, y)
        for r in self.rocks:
            h = h + r.height_at(x, y)
        return h

    def gradient(self, x, y, eps: float = 1e-4):
        x, y = np.asarray(x, float), np.asarray(y, float)
        hx = (self.height(x + eps, y) - self.height(x - eps, y)) / (2 * eps)
        hy = (self.height(x, y + eps) - self.height(x, y - eps)) / (2 * eps)
        return hx, hy

    def normal(self, x, y):
        hx, hy = self.gradient(x, y)
        n = np.stack([-hx, -hy, np.ones_like(hx)], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def slope_along(self, x, y, heading_deg: float = 0.0):
        """Grade angle [deg] along a heading (positive uphill)."""
        hx, hy = self.gradient(x, y)
        a = math.radians(heading_deg)
        return np.degrees(np.arctan(hx * math.cos(a) + hy * math.sin(a)))

    # ------------------------------------------------------------------ soil
    def soil_index(self, x, y) -> np.ndarray:
        """Index per point: 0 = default soil, k = regions[k-1] (the last matching region wins)."""
        idx = np.zeros(np.shape(np.asarray(x, float)), dtype=int)
        for k, reg in enumerate(self.regions, start=1):
            idx = np.where(reg.contains(x, y), k, idx)
        return idx

    def soils(self) -> list[SoilParams]:
        return [self.soil] + [r.soil for r in self.regions]

    # ------------------------------------------------------------------ sampling
    def grid(self, x_range, y_range, n: tuple[int, int] = (200, 120)):
        X, Y = np.meshgrid(np.linspace(*x_range, n[0]), np.linspace(*y_range, n[1]))
        return X, Y, self.height(X, Y)

    def mesh(self, x_range, y_range, n: tuple[int, int] = (200, 120)):
        """Triangulated heightmap: vertices (V, 3), faces (F, 3)."""
        X, Y, Z = self.grid(x_range, y_range, n)
        V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        nx, ny = n
        i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
        a = (j * nx + i).ravel()
        F = np.concatenate([np.stack([a, a + 1, a + nx + 1], 1), np.stack([a, a + nx + 1, a + nx], 1)])
        return V, F


def random_rocks(rng: np.random.Generator, n: int, x_range, y_range, radius_range, height_ratio_range,
                 keep_out=None) -> list[Rock]:
    """``n`` rocks with uniform positions and radii (no published size-frequency law is used); ``keep_out(x, y)``
    returns True where no rock may be placed (e.g. the rover path)."""
    rocks: list[Rock] = []
    while len(rocks) < n:
        x, y = rng.uniform(*x_range), rng.uniform(*y_range)
        if keep_out is not None and keep_out(x, y):
            continue
        r = rng.uniform(*radius_range)
        rocks.append(Rock(x, y, r, r * rng.uniform(*height_ratio_range)))
    return rocks


__all__ = ["LunarTerrain", "Crater", "Rock", "SoilRegion", "random_rocks", "PIKE_1977_DEPTH_RATIO"]
