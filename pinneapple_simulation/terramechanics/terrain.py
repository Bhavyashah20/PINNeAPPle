"""Deformable terrain for wheel-soil simulation: plastic sinkage memory (ruts, multi-pass), grouser imprints, contact
stress maps and rigid rocks.

The undisturbed surface is ``base.height(x, y)`` (any object with that method, e.g. the rover module's
``LunarTerrain``) plus an optional small-scale relief grid. On top of it a regular grid keeps, per cell:

* ``z_max`` -- the deepest sinkage below the undisturbed surface the soil has carried (Wong's repetitive loading:
  the soil remembers it; a later wheel reloads it elastically up to that depth, see :mod:`.wong_reece`). The visible
  rut is ``z_max`` minus the elastic rebound.
* ``imprint`` -- grouser marks stamped where a grouser left the soil (visual layer: depth of the lug trench).
* ``wall`` / ``berm`` -- the rut walls slumped to the angle of repose and the soil displaced to both sides of the
  rut (visual layers: depth and height [m]). Together with ``z_max`` they give the rut its cross-section: a flat
  floor as wide as the wheel, walls at the angle of repose and rounded berms beyond them.
* ``j_mem`` -- the shear displacement the last wheel left in the soil [m] (multi-pass shear, see :mod:`.wong_reece`).
* ``sigma`` / ``tau`` -- the largest normal and shear stress the cell has seen [Pa], for contact colour maps.

Rocks (:class:`RigidRock`) are rigid half-buried ellipsoids: wheels touching them get a penalty contact, not soil
forces. Their surface is :meth:`RigidRock.height`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RigidRock:
    """Ellipsoid centred at (x, y, soil height - ``burial`` c), semi-axes (a, b, c) [m], rotated by ``yaw`` [rad]."""
    x: float
    y: float
    a: float
    b: float
    c: float
    burial: float = 0.4
    yaw: float = 0.0
    z0: float = 0.0            # soil height at the centre (set by DeformableTerrain.add_rock)
    seed: int = 0

    @property
    def zc(self) -> float:
        return self.z0 - self.burial * self.c

    def _local(self, x, y):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = np.asarray(x, float) - self.x, np.asarray(y, float) - self.y
        return cy * dx + sy * dy, -sy * dx + cy * dy

    def height(self, x, y):
        """Top surface of the rock [m], -inf outside its footprint."""
        u, w = self._local(x, y)
        q = 1.0 - (u / self.a) ** 2 - (w / self.b) ** 2
        return np.where(q > 0, self.zc + self.c * np.sqrt(np.clip(q, 0, None)), -np.inf)

    def penetration(self, P):
        """Approximate penetration depth (>0 inside) and outward normal for points (N, 3)."""
        u, w = self._local(P[:, 0], P[:, 1])
        dz = P[:, 2] - self.zc
        f = np.sqrt((u / self.a) ** 2 + (w / self.b) ** 2 + (dz / self.c) ** 2)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        gl = np.stack([u / self.a ** 2, w / self.b ** 2, dz / self.c ** 2], 1)
        g = np.stack([cy * gl[:, 0] - sy * gl[:, 1], sy * gl[:, 0] + cy * gl[:, 1], gl[:, 2]], 1)
        n = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        depth = (1.0 - f) * min(self.a, self.b, self.c)       # exact on a sphere, first order otherwise
        return depth, n

    def mesh(self, n: int = 28, roughness: float = 0.12):
        """A rough rock surface (the ellipsoid displaced by smooth random lumps of relative size ``roughness``),
        clipped to above the soil. Visual only."""
        rng = np.random.default_rng(self.seed)
        th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 2 * n, endpoint=False), np.linspace(0, np.pi, n))
        d = np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], -1)
        bump = np.zeros(th.shape)
        for _ in range(9):
            k = rng.normal(size=3)
            k /= np.linalg.norm(k)
            bump += rng.uniform(0.3, 1.0) * np.exp(-(1 - d @ k) * rng.uniform(4, 14))
        bump = 1.0 + roughness * (bump / max(bump.max(), 1e-9) - 0.4)
        L = d * np.array([self.a, self.b, self.c]) * bump[..., None]
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        V = np.stack([self.x + cy * L[..., 0] - sy * L[..., 1], self.y + sy * L[..., 0] + cy * L[..., 1],
                      self.zc + L[..., 2]], -1).reshape(-1, 3)
        nr, nc = th.shape
        i, j = np.meshgrid(np.arange(nr - 1), np.arange(nc))
        a = i * nc + j
        b = i * nc + (j + 1) % nc
        F = np.concatenate([np.stack([a, b, b + nc], -1).reshape(-1, 3), np.stack([a, b + nc, a + nc], -1)
                            .reshape(-1, 3)])
        return V, F


def fractal_relief(shape, cell: float, amplitude: float, seed: int = 0, beta: float = 2.2,
                   min_wavelength: float = 0.05) -> np.ndarray:
    """Gaussian random surface with a power-law spectrum (exponent ``beta``) of RMS ``amplitude`` [m], sampled on a
    grid of ``shape`` with spacing ``cell``; wavelengths below ``min_wavelength`` are removed."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    kx = np.fft.fftfreq(nx, cell)
    ky = np.fft.fftfreq(ny, cell)
    K = np.hypot(*np.meshgrid(kx, ky))
    amp = np.where(K > 0, np.where(K > 0, K, 1.0) ** (-beta / 2 - 0.5), 0.0) * (K < 1.0 / min_wavelength)
    Z = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=shape)) * amp))
    return amplitude * Z / max(Z.std(), 1e-12)


class DeformableTerrain:
    """Grid memory over ``x_range`` x ``y_range`` with spacing ``cell`` on top of ``base``.

    ``rebound(z_max)`` gives the elastic rebound of the soil (use ``WongReeceModel.rebound``); ``relief`` [m] is the RMS
    of a fractal small-scale relief added to the base (0 = none). ``berm`` is the share of the rut cross-section
    pushed sideways into the two berms (the rest is compaction); ``wall_angle_deg`` is the slope of the rut walls
    (the angle of repose of the regolith, about 35-40 deg for lunar simulants). Both are visual only.
    """

    def __init__(self, base, x_range, y_range, cell: float = 0.01, rebound=None, relief: float = 0.0,
                 relief_seed: int = 0, berm: float = 0.3, wall_angle_deg: float = 38.0):
        self.base = base
        self.x0, self.x1 = map(float, x_range)
        self.y0, self.y1 = map(float, y_range)
        self.cell = float(cell)
        self.nx = int(round((self.x1 - self.x0) / cell)) + 1
        self.ny = int(round((self.y1 - self.y0) / cell)) + 1
        self.z_max = np.zeros((self.ny, self.nx), np.float32)
        self.imprint = np.zeros((self.ny, self.nx), np.float32)
        self.sigma = np.zeros((self.ny, self.nx), np.float32)
        self.tau = np.zeros((self.ny, self.nx), np.float32)
        self.j_mem = np.zeros((self.ny, self.nx), np.float32)
        self.wall = np.zeros((self.ny, self.nx), np.float32)
        self.berm_h = np.zeros((self.ny, self.nx), np.float32)
        self.wall_angle_deg = float(wall_angle_deg)
        self.relief = (fractal_relief((self.ny, self.nx), cell, relief, relief_seed) if relief > 0
                       else np.zeros((self.ny, self.nx)))
        self.rebound = rebound or (lambda z: np.zeros_like(np.asarray(z, float)))
        self.berm = float(berm)
        self.rocks: list[RigidRock] = []
        X, Y = self.grid_xy()
        self.h0 = self.base_soil_height(X, Y) + self.relief      # undisturbed surface sampled once on the grid

    # ------------------------------------------------------------------ grid helpers
    def _ij(self, x, y):
        fx = (np.asarray(x, float) - self.x0) / self.cell
        fy = (np.asarray(y, float) - self.y0) / self.cell
        return fx, fy

    def _bilinear(self, G, x, y):
        fx, fy = self._ij(x, y)
        fx = np.clip(fx, 0, self.nx - 1.000001)
        fy = np.clip(fy, 0, self.ny - 1.000001)
        i, j = fx.astype(int), fy.astype(int)
        tx, ty = fx - i, fy - j
        return ((1 - tx) * (1 - ty) * G[j, i] + tx * (1 - ty) * G[j, i + 1] + (1 - tx) * ty * G[j + 1, i]
                + tx * ty * G[j + 1, i + 1])

    def grid_xy(self):
        return np.meshgrid(self.x0 + self.cell * np.arange(self.nx), self.y0 + self.cell * np.arange(self.ny))

    # ------------------------------------------------------------------ surfaces
    def height0(self, x, y):
        """Undisturbed soil surface (base + relief) [m], bilinear on the grid (the base function outside it)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = self._bilinear(self.h0, x, y)
        out = (x < self.x0) | (x > self.x1) | (y < self.y0) | (y > self.y1)
        if np.any(out):
            h = np.where(out, self.base_soil_height(x, y), h)
        return h

    def base_soil_height(self, x, y):
        f = getattr(self.base, "soil_height", None) or self.base.height
        return f(x, y)

    def normal0(self, x, y, eps: float = 0.01):
        x, y = np.asarray(x, float), np.asarray(y, float)
        hx = (self.height0(x + eps, y) - self.height0(x - eps, y)) / (2 * eps)
        hy = (self.height0(x, y + eps) - self.height0(x, y - eps)) / (2 * eps)
        n = np.stack([-hx, -hy, np.ones_like(hx)], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def rut_depth(self, x, y):
        zm = self._bilinear(self.z_max, x, y)
        return np.clip(zm - self.rebound(zm), 0.0, None)

    def _visible_offset(self, rut, wall, berm, imprint):
        """Height change of the visible surface [m]: the deeper of the rut and the slumped wall, the berms where
        there is no rut, the grouser marks."""
        return berm * (rut <= 1e-5) - np.maximum(rut, wall) - imprint

    def surface(self, x, y):
        """Visible soil surface: undisturbed surface minus the rut (floor and walls) and the grouser imprints, plus
        the berms (no rocks) [m]."""
        return self.height0(x, y) + self._visible_offset(self.rut_depth(x, y), self._bilinear(self.wall, x, y),
                                                         self._bilinear(self.berm_h, x, y),
                                                         self._bilinear(self.imprint, x, y))

    def add_rock(self, rock: RigidRock) -> RigidRock:
        rock.z0 = float(self.height0(rock.x, rock.y))
        self.rocks.append(rock)
        return rock

    # ------------------------------------------------------------------ queries for the contact
    def precompaction(self, P, fwd, half_width: float, n_across: int = 5):
        """Largest remembered sinkage over a segment across the track at points ``P`` (N, 2) [m]."""
        P = np.asarray(P, float).reshape(-1, 2)
        fwd = np.asarray(fwd, float).reshape(-1, 2)
        lat = np.stack([-fwd[:, 1], fwd[:, 0]], 1)
        s = np.linspace(-half_width, half_width, n_across)
        Q = P[:, None, :] + s[None, :, None] * lat[:, None, :]
        return self._bilinear(self.z_max, Q[..., 0], Q[..., 1]).max(1)

    def prior_shear(self, P, fwd, half_width: float, n_across: int = 5):
        """Shear displacement left in the soil at points ``P`` (N, 2), averaged across the track [m]."""
        P = np.asarray(P, float).reshape(-1, 2)
        fwd = np.asarray(fwd, float).reshape(-1, 2)
        lat = np.stack([-fwd[:, 1], fwd[:, 0]], 1)
        s = np.linspace(-half_width, half_width, n_across)
        Q = P[:, None, :] + s[None, :, None] * lat[:, None, :]
        return self._bilinear(self.j_mem, Q[..., 0], Q[..., 1]).mean(1)

    # ------------------------------------------------------------------ writing
    def _footprint(self, cx, cy, fwd, half_len, half_w):
        """Cells inside an oriented rectangle: (rows, cols, along, across) arrays."""
        r = math.hypot(half_len, half_w) + self.cell
        fx, fy = self._ij(np.array([cx - r, cx + r]), np.array([cy - r, cy + r]))
        i0, i1 = max(int(fx[0]), 0), min(int(fx[1]) + 2, self.nx)
        j0, j1 = max(int(fy[0]), 0), min(int(fy[1]) + 2, self.ny)
        if i0 >= i1 or j0 >= j1:
            return None
        X, Y = np.meshgrid(self.x0 + self.cell * np.arange(i0, i1), self.y0 + self.cell * np.arange(j0, j1))
        dx, dy = X - cx, Y - cy
        al = dx * fwd[0] + dy * fwd[1]
        ac = -dx * fwd[1] + dy * fwd[0]
        m = (np.abs(al) <= half_len) & (np.abs(ac) <= half_w)
        return (slice(j0, j1), slice(i0, i1)), m, al, ac

    def record_wheel(self, cx, cy, fwd, radius: float, width: float, sinkage: float, sigma_max: float = 0.0,
                     tau_max: float = 0.0, j_exit: float | None = None):
        """Write the plastic trace of a wheel whose lowest rim point is ``sinkage`` below the undisturbed surface at
        (cx, cy): every cell under the rim gets ``z_max = max(z_max, rim depth)``, plus side berms; the cells behind
        the bottom point keep the shear displacement ``j_exit`` the wheel leaves."""
        if sinkage <= 0:
            return
        f = np.asarray(fwd, float)[:2]
        f = f / max(np.linalg.norm(f), 1e-12)
        half_len = math.sqrt(max(2 * radius * sinkage - sinkage ** 2, 0.0)) + self.cell
        hw = 0.5 * width
        run = sinkage / math.tan(math.radians(self.wall_angle_deg))     # horizontal extent of a wall
        wb = max(0.35 * width, 2 * self.cell)                            # berm width beyond the wall top
        fp = self._footprint(cx, cy, f, half_len, hw + run + wb + self.cell)
        if fp is None:
            return
        sl, _, al, ac = fp
        depth = sinkage - (radius - np.sqrt(np.clip(radius ** 2 - al ** 2, 0.0, None)))
        edge = np.clip((hw - np.abs(ac)) / (0.5 * self.cell) + 0.5, 0.0, 1.0)   # anti-aliased track edge
        inside = (depth > 0) & (np.abs(ac) <= hw + self.cell)
        d = np.where(inside, depth * edge, 0.0).astype(np.float32)
        self.z_max[sl] = np.maximum(self.z_max[sl], d)
        fresh = inside & (al > 0) & (self.imprint[sl] > 0)        # the wheel ahead flattens older grouser marks
        self.imprint[sl] = np.where(fresh, 0.0, self.imprint[sl])
        self.berm_h[sl] = np.where(inside & (edge > 0.5), 0.0, self.berm_h[sl])   # a wheel running over a berm
        if j_exit is not None:
            behind = inside & (al <= 0) & (np.abs(ac) <= hw)
            self.j_mem[sl] = np.where(behind, np.float32(j_exit), self.j_mem[sl])
        if sigma_max:
            self.sigma[sl] = np.maximum(self.sigma[sl], np.where(d > 0, sigma_max * edge, 0.0).astype(np.float32))
        if tau_max:
            self.tau[sl] = np.maximum(self.tau[sl], np.where(d > 0, tau_max * edge, 0.0).astype(np.float32))
        # walls at the angle of repose: the depth falls linearly from the floor edge to the original surface
        dmax = np.clip(depth, 0.0, None)
        out = np.abs(ac) - hw
        w = np.where((out > 0) & (np.abs(al) <= half_len), dmax * np.clip(1.0 - out / max(run, 1e-9), 0.0, 1.0), 0.0)
        self.wall[sl] = np.maximum(self.wall[sl], w.astype(np.float32))
        if self.berm > 0:
            # rounded berm beyond the wall top, holding a share ``berm`` of the rut cross-section on each side
            s_ = (out - run) / wb
            prof = np.where((s_ > -0.15) & (s_ < 1) & (np.abs(al) <= half_len),
                            np.sin(np.pi * np.clip((s_ + 0.15) / 1.15, 0, 1)), 0.0)
            h = 0.5 * self.berm * dmax * width / (wb * 1.15 * 2 / np.pi) * prof
            self.berm_h[sl] = np.maximum(self.berm_h[sl], h.astype(np.float32))

    def stamp_grouser(self, gx, gy, fwd, width: float, depth: float, thickness: float, shape: str = "straight",
                      chevron_angle_deg: float = 0.0):
        """A grouser mark centred at (gx, gy): a trench ``depth`` deep and ``thickness`` wide across a track of
        ``width`` (a V for chevrons, its apex towards ``fwd``)."""
        if depth <= 0:
            return
        f = np.asarray(fwd, float)[:2]
        f = f / max(np.linalg.norm(f), 1e-12)
        sweep = math.tan(math.radians(chevron_angle_deg)) * 0.5 * width if shape == "chevron" else 0.0
        fp = self._footprint(gx, gy, f, abs(sweep) + thickness + 2 * self.cell, 0.5 * width)
        if fp is None:
            return
        sl, m, al, ac = fp
        centre = -abs(sweep) * np.abs(ac) / max(0.5 * width, 1e-9) + (abs(sweep) / 2 if sweep else 0.0)
        dist = np.abs(al - centre)
        w = 0.5 * thickness + 0.75 * self.cell
        prof = np.clip(1.0 - dist / w, 0.0, 1.0) ** 0.7 * m
        self.imprint[sl] = np.maximum(self.imprint[sl], (depth * prof).astype(np.float32))

    # ------------------------------------------------------------------ meshes
    def mesh(self, x_range=None, y_range=None, step: int = 1):
        """Triangulated visible surface on the grid (every ``step`` cells): vertices, faces and per-vertex fields
        ``rut_depth`` [mm], ``normal_stress`` [kPa], ``shear_stress`` [kPa]."""
        xr = x_range or (self.x0, self.x1)
        yr = y_range or (self.y0, self.y1)
        fx, fy = self._ij(np.array(xr), np.array(yr))
        i = np.arange(max(int(fx[0]), 0), min(int(math.ceil(fx[1])) + 1, self.nx), step)
        j = np.arange(max(int(fy[0]), 0), min(int(math.ceil(fy[1])) + 1, self.ny), step)
        X, Y = np.meshgrid(self.x0 + self.cell * i, self.y0 + self.cell * j)
        jj, ii = np.meshgrid(j, i, indexing="ij")
        zm = self.z_max[jj, ii].astype(float)
        rut = np.clip(zm - self.rebound(zm), 0, None)
        Z = self.h0[jj, ii] + self._visible_offset(rut, self.wall[jj, ii], self.berm_h[jj, ii], self.imprint[jj, ii])
        V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        nyy, nxx = X.shape
        a, b = np.meshgrid(np.arange(nxx - 1), np.arange(nyy - 1))
        k = (b * nxx + a).ravel()
        F = np.concatenate([np.stack([k, k + 1, k + nxx + 1], 1), np.stack([k, k + nxx + 1, k + nxx], 1)])
        fields = {"rut_depth": 1e3 * (np.maximum(rut, self.wall[jj, ii]) + np.clip(self.imprint[jj, ii], 0, None)).ravel(),
                  "normal_stress": 1e-3 * self.sigma[jj, ii].ravel().astype(float),
                  "shear_stress": 1e-3 * self.tau[jj, ii].ravel().astype(float)}
        return V, F, fields

    def copy_empty(self) -> DeformableTerrain:
        """Same base, grid, relief and rocks, with no ruts or marks (to replay recorded terrain events)."""
        import copy
        t = copy.copy(self)
        for k in ("z_max", "imprint", "sigma", "tau", "j_mem", "wall", "berm_h"):
            setattr(t, k, np.zeros_like(getattr(self, k)))
        t.rocks = list(self.rocks)
        return t

    def snapshot(self) -> dict:
        return {k: getattr(self, k).copy() for k in ("z_max", "imprint", "sigma", "tau", "j_mem", "wall", "berm_h")}

    def restore(self, snap: dict) -> None:
        for k, v in snap.items():
            getattr(self, k)[...] = v


__all__ = ["DeformableTerrain", "RigidRock", "fractal_relief"]
