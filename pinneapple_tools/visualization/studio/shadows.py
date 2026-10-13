"""Sun shadows and planetary-surface shading for heightfield terrains, independent of the renderer.

* :func:`terrain_self_shadow` -- cells of a heightfield hidden from the Sun by the terrain itself (horizon march).
* :func:`cast_shadow` -- the shadow of arbitrary triangle meshes (a vehicle) on a heightfield: every triangle is
  projected along the Sun direction onto the terrain and rasterised.
* :func:`lunar_lambert` -- the Lunar-Lambert photometric function of McEwen (1991), ``I = A [(1 - L) mu0 +
  2 L mu0 / (mu0 + mu)]`` (mu0, mu: cosines of the incidence and emission angles; L = 1 is the Lommel-Seeliger law
  of dark regolith), which keeps regolith lit at grazing Sun angles where a Lambert surface turns black.
  McEwen, A.S. (1991). Photometric functions for photoclinometry and other applications. Icarus 92, 298-311.
* :func:`shade_terrain` -- RGB per terrain vertex: photometry x (1 - shadow) + a weak fill, tone mapped.

Shading is baked into vertex colours, so any renderer (PyVista without its shadow pass, three.js, matplotlib) shows
the same hard shadows.
"""
from __future__ import annotations

import numpy as np


def _bilinear(Z, fx, fy):
    ny, nx = Z.shape
    fx = np.clip(fx, 0, nx - 1.000001)
    fy = np.clip(fy, 0, ny - 1.000001)
    i, j = fx.astype(int), fy.astype(int)
    tx, ty = fx - i, fy - j
    return ((1 - tx) * (1 - ty) * Z[j, i] + tx * (1 - ty) * Z[j, i + 1] + (1 - tx) * ty * Z[j + 1, i]
            + tx * ty * Z[j + 1, i + 1])


def terrain_self_shadow(Z, cell: float, sun_dir, max_distance: float | None = None) -> np.ndarray:
    """Boolean (ny, nx): True where the heightfield ``Z`` (rows along y, columns along x, spacing ``cell``) hides
    the Sun (``sun_dir`` points from the ground to the Sun)."""
    s = np.asarray(sun_dir, float)
    s = s / np.linalg.norm(s)
    horiz = np.hypot(s[0], s[1])
    if horiz < 1e-9:
        return np.zeros(Z.shape, bool)
    dx, dy = s[0] / horiz, s[1] / horiz
    slope = s[2] / horiz                                   # rise of the ray per metre travelled horizontally
    ny, nx = Z.shape
    span = max_distance if max_distance is not None else cell * max(nx, ny)
    jj, ii = np.mgrid[0:ny, 0:nx].astype(float)
    shadow = np.zeros(Z.shape, bool)
    zmax_rise = float(Z.max() - Z.min())
    k = 1
    while k * cell <= span and k * cell * slope <= zmax_rise + 1e-9:
        d = k * cell
        h = _bilinear(Z, ii + d * dx / cell, jj + d * dy / cell)
        inside = (ii + d * dx / cell >= 0) & (ii + d * dx / cell <= nx - 1) & (jj + d * dy / cell >= 0) & \
                 (jj + d * dy / cell <= ny - 1)
        shadow |= inside & (h > Z + d * slope)
        k += 1 if k < 40 else max(1, k // 20)
    return shadow


def points_in_terrain_shadow(P, X, Y, Z, sun_dir, step: float | None = None) -> np.ndarray:
    """Boolean per point (N, 3): the heightfield (X, Y, Z) blocks the Sun as seen from the point."""
    s = np.asarray(sun_dir, float)
    s = s / np.linalg.norm(s)
    P = np.asarray(P, float)
    x0, y0 = float(X[0, 0]), float(Y[0, 0])
    cx, cy = float(X[0, 1] - X[0, 0]), float(Y[1, 0] - Y[0, 0])
    ny, nx = Z.shape
    h = step or 2 * cx
    horiz = max(np.hypot(s[0], s[1]), 1e-9)
    n = int(min(np.hypot(nx * cx, ny * cy), (Z.max() - Z.min() + 1.0) / max(s[2], 1e-3) * horiz) / h) + 1
    out = np.zeros(len(P), bool)
    for k in range(1, n + 1):
        Q = P + (k * h / horiz) * s
        fx, fy = (Q[:, 0] - x0) / cx, (Q[:, 1] - y0) / cy
        inside = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
        if not inside.any():
            break
        out |= inside & (_bilinear(Z, fx, fy) > Q[:, 2])
    return out


def cast_shadow(meshes, X, Y, Z, sun_dir, iterations: int = 3, supersample: int = 1) -> np.ndarray:
    """Boolean (ny, nx): cells of the regular grid (X, Y, Z) in the shadow of the triangle ``meshes`` [(V, F), ...].

    With ``supersample`` > 1 the shadow is rasterised that many times finer and averaged: a float (ny, nx)
    shadow fraction, which keeps the shadow edge smooth on a coarse grid."""
    ss = max(int(supersample), 1)
    from PIL import Image, ImageDraw
    s = np.asarray(sun_dir, float)
    s = s / np.linalg.norm(s)
    x0, y0 = float(X[0, 0]), float(Y[0, 0])
    cx, cy = float(X[0, 1] - X[0, 0]), float(Y[1, 0] - Y[0, 0])
    ny, nx = Z.shape
    img = Image.new("1", (nx * ss, ny * ss), 0)
    draw = ImageDraw.Draw(img)
    for V, F in meshes:
        V = np.asarray(V, float)
        t = np.zeros(len(V))
        for _ in range(iterations):                       # move along -s until the point reaches the terrain
            P = V - t[:, None] * s
            h = _bilinear(Z, (P[:, 0] - x0) / cx, (P[:, 1] - y0) / cy)
            t = np.maximum(t + (P[:, 2] - h) / max(s[2], 1e-3), 0.0)
        P = V - t[:, None] * s
        px = (P[:, 0] - x0) / cx
        py = (P[:, 1] - y0) / cy
        T = (np.stack([px[F], py[F]], -1) + 0.5) * ss - 0.5
        keep = np.all((T[..., 0] > -nx * ss) & (T[..., 0] < 2 * nx * ss) & (T[..., 1] > -ny * ss)
                      & (T[..., 1] < 2 * ny * ss), 1)
        for tri in T[keep]:
            draw.polygon([tuple(p) for p in tri], fill=1)
    a = np.asarray(img, bool)
    return a if ss == 1 else a.reshape(ny, ss, nx, ss).mean(axis=(1, 3))


def lunar_lambert(normals, sun_dir, view_dir, L: float = 1.0) -> np.ndarray:
    """McEwen (1991) Lunar-Lambert reflectance (without the albedo) for unit ``normals`` (N, 3); ``view_dir`` (3,) or
    (N, 3) points from the surface to the camera."""
    n = np.asarray(normals, float)
    s = np.asarray(sun_dir, float)
    s = s / np.linalg.norm(s)
    v = np.asarray(view_dir, float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    mu0 = np.clip(n @ s, 0.0, None)
    mu = np.clip(np.sum(n * v, -1) if v.ndim == 2 else n @ v, 1e-3, None)
    return (1 - L) * mu0 + 2 * L * mu0 / (mu0 + mu + 1e-9)


def relief_noise(X, Y, rms: float, wavelengths=(0.6, 8.0), n: int = 40, seed: int = 7, slope: float = 1.2):
    """World-anchored smooth random relief [m] at points (X, Y): a sum of ``n`` plane waves with random directions
    and phases, wavelengths log-spaced in ``wavelengths`` and amplitudes ~ wavelength^slope, scaled to ``rms``.
    The same (x, y) always gives the same height, so a moving window of terrain does not swim."""
    rng = np.random.default_rng(seed)
    lam = np.exp(rng.uniform(np.log(wavelengths[0]), np.log(wavelengths[1]), n))
    ang = rng.uniform(0, 2 * np.pi, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    amp = lam ** slope
    amp = amp / np.sqrt(0.5 * np.sum(amp ** 2)) * rms
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    out = np.zeros(np.broadcast(X, Y).shape)
    for a, lw, t, p in zip(amp, lam, ang, ph, strict=True):
        out += a * np.sin(2 * np.pi / lw * (X * np.cos(t) + Y * np.sin(t)) + p)
    return out


def grid_normals(Z, cell: float) -> np.ndarray:
    gy, gx = np.gradient(Z, cell)
    n = np.stack([-gx, -gy, np.ones_like(Z)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def shade_terrain(X, Y, Z, sun_dir, camera_position, albedo=(0.45, 0.44, 0.42), shadow=None, exposure: float = 1.0,
                  fill: float = 0.04, L: float = 0.6, tint=None) -> np.ndarray:
    """(ny, nx, 3) colours in [0, 1]: Lunar-Lambert photometry, shadows from ``shadow`` (bool grid or 0-1 fraction), a weak
    fill light, a filmic shoulder. ``tint`` (ny, nx, 3) multiplies the albedo (e.g. a field colour map)."""
    cell = float(X[0, 1] - X[0, 0])
    n = grid_normals(Z, cell)
    P = np.stack([X, Y, Z], -1)
    v = np.asarray(camera_position, float) - P
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    s = np.asarray(sun_dir, float)
    s = s / np.linalg.norm(s)
    mu0 = np.clip(n @ s, 0.0, None)
    mu = np.clip(np.sum(n * v, -1), 1e-3, None)
    refl = (1 - L) * mu0 + 2 * L * mu0 / (mu0 + mu + 1e-9)
    if shadow is not None:
        shadow = np.asarray(shadow)
        refl = np.where(shadow, 0.0, refl) if shadow.dtype == bool else refl * (1.0 - np.clip(shadow, 0, 1))
    a = np.asarray(albedo, float)
    base = a[None, None, :] * (tint if tint is not None else 1.0)
    rgb = base * (exposure * refl[..., None] + fill * (0.5 + 0.5 * n[..., 2:3]))
    return 1.0 - np.exp(-1.6 * rgb)                      # soft shoulder instead of clipping


__all__ = ["relief_noise", "terrain_self_shadow", "points_in_terrain_shadow", "cast_shadow", "lunar_lambert", "shade_terrain", "grid_normals"]
