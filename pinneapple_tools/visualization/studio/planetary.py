"""Renders of a vehicle on a planetary surface with PyVista: heightfield terrain with baked photometry and hard Sun
shadows (:mod:`.shadows`), vehicle parts with their studio materials, rocks, optional field colour maps on the terrain
and on the parts (with a colour bar), Earth in the sky and a title.

    from pinneapple_tools.visualization.studio.planetary import SurfaceView, render_surface
    view = SurfaceView(X, Y, Z, parts=[(V, F, "white_paint"), ...], rocks=[(V, F)], preset="lunar_south_pole")
    render_surface(view, "frame.png", camera=((-4, -5, 2), (0, 0, 0.8)), title="t = 12.0 s")

``parts`` entries are ``(vertices, faces, material)`` or ``(vertices, faces, material, values)`` where ``values`` (one
per vertex) are coloured with ``cmap`` / ``clim`` instead of the material. ``terrain_field`` (ny, nx) colours the
terrain where it is finite (e.g. normal stress in the ruts), blended over the regolith by ``field_alpha``: the colour
multiplies the regolith albedo before the photometry, so the relief, the shading and the shadows stay visible
through it; ``field_glow`` adds a share of unshaded colour that keeps the map readable inside shadows (0 = a pure
semi-transparent tint); ``field_saturation`` < 1 mutes the colours so the relief dominates. ``far_field`` colours
the far grid the same way.

Horizon: with ``horizon`` > 0 an annulus of terrain runs from the edge of the ``far`` grid out to ``horizon`` metres,
its height following ``far_fn`` near the grid and dropping with the curvature of the body (``planet_radius``,
the Moon by default) further out, so the visible edge of the ground is the real horizon (about 2.6 km for a camera
2 m above the lunar surface) and no mesh border is seen. :func:`border_visibility` checks this for a camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .lighting import apply_pyvista, get_preset
from .scene import MATERIALS
from .shadows import cast_shadow, points_in_terrain_shadow, shade_terrain, terrain_self_shadow


@dataclass
class SurfaceView:
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    parts: list = field(default_factory=list)
    rocks: list = field(default_factory=list)
    preset: object = "lunar_south_pole"
    terrain_field: np.ndarray | None = None
    field_alpha: float = 0.85
    field_glow: float = 0.55            # unshaded share of the colour (relative to field_alpha)
    field_saturation: float = 1.0       # 1 = the colour map as is, 0 = grey (the map's lightness only)
    far_field: np.ndarray | None = None  # field on the ``far`` grid (same convention as terrain_field)
    cmap: str = "inferno"
    clim: tuple | None = None
    colorbar_title: str = ""
    albedo: tuple = (0.42, 0.41, 0.39)
    exposure: float = 1.0
    earth: bool = False
    far: tuple | None = None            # (X, Y, Z) coarse terrain around the detailed patch, drawn outside it
    mesh_light: float = 1.35            # intensity of the Sun on the meshes (the terrain photometry is baked)
    lunar_lambert_L: float = 0.45       # McEwen's L: 1 = Lommel-Seeliger (flat-looking), 0 = Lambert (more relief)
    horizon: float = 0.0                # outer radius [m] of the curved annulus around ``far`` (0 = none)
    far_fn: object = None               # height function (x, y) -> z near the far grid (default: its mean edge level)
    planet_radius: float = 1_737_400.0  # for the curvature drop of the annulus [m] (the Moon)
    horizon_relief: float = 0.35        # RMS [m] of the long-wavelength relief of the annulus
    horizon_blend: float = 30.0         # distance [m] over which the annulus goes from ``far_fn`` to a level surface
    terrain_overlay: bool = True        # draw ``terrain_field`` (False: geometry and photometry only)


MOON_RADIUS = 1_737_400.0


def horizon_distance(height: float, planet_radius: float = MOON_RADIUS) -> float:
    """Distance [m] to the horizon of a smooth sphere seen from ``height`` metres above it."""
    return math.sqrt(2 * planet_radius * max(height, 0.0) + height * height)


def _far_frame(view: SurfaceView):
    """Centre, half sizes and edge level of the far grid (or of the detailed patch without one)."""
    X, Y, Z = view.far if view.far is not None else (view.X, view.Y, view.Z)
    c = np.array([0.5 * (X.min() + X.max()), 0.5 * (Y.min() + Y.max())])
    edge = np.r_[Z[0], Z[-1], Z[:, 0], Z[:, -1]]
    return c, 0.5 * np.ptp(X), 0.5 * np.ptp(Y), float(np.median(edge))


def ring_height(view: SurfaceView, x, y):
    """Height of the horizon annulus [m] at (x, y): ``far_fn`` (or the grid's edge level) blended into the edge
    level over ``horizon_blend`` metres, a long-wavelength relief faded in with distance, and the curvature drop."""
    from .shadows import relief_noise
    c, hx, hy, z0 = _far_frame(view)
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = max(hx, hy)
    r = np.hypot(x - c[0], y - c[1])
    w = np.clip((r - h) / max(view.horizon_blend, 1e-6), 0.0, 1.0)
    w = w * w * (3 - 2 * w)
    near = view.far_fn(x, y) if view.far_fn is not None else np.full(r.shape, z0)
    rel = relief_noise(x, y, view.horizon_relief, wavelengths=(15.0, 400.0), n=48, seed=11, slope=1.0) if \
        view.horizon_relief else 0.0
    return (1 - w) * near + w * z0 + w * rel - r * r / (2 * view.planet_radius)


def _ring_mesh(view: SurfaceView, n_theta: int = 256, n_r: int = 90):
    """Polar annulus from the far grid's rectangle out to ``view.horizon`` (geometric radial spacing)."""
    c, hx, hy, _ = _far_frame(view)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    inset = 0.4                                             # overlap under the far grid (hidden 1 cm below it)
    rs = 1.0 / np.maximum(np.abs(ct) / max(hx - inset, 1e-3), np.abs(st) / max(hy - inset, 1e-3))
    t = np.linspace(0.0, 1.0, n_r)[:, None]
    R = rs[None, :] * (view.horizon / rs[None, :]) ** t
    Xr, Yr = c[0] + R * ct[None, :], c[1] + R * st[None, :]
    Zr = ring_height(view, Xr, Yr) - 0.01
    Xr, Yr, Zr = (np.concatenate([a, a[:, :1]], 1) for a in (Xr, Yr, Zr))   # close the seam
    return Xr, Yr, Zr


def _ring_normals(view, X, Y):
    e = 0.02 + 0.004 * np.hypot(X - X.mean(), Y - Y.mean())
    gx = (ring_height(view, X + e, Y) - ring_height(view, X - e, Y)) / (2 * e)
    gy = (ring_height(view, X, Y + e) - ring_height(view, X, Y - e)) / (2 * e)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _project(points, camera, size, view_angle):
    """Pixel coordinates (N, 2) and depth along the view axis (N,) of world points for a pinhole camera."""
    pos, foc = np.asarray(camera[0], float), np.asarray(camera[1], float)
    up = np.asarray(camera[2] if len(camera) > 2 else (0.0, 0.0, 1.0), float)
    f = foc - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = np.asarray(points, float) - pos
    z = d @ f
    t = math.tan(math.radians(view_angle) / 2)            # vertical half angle (VTK convention)
    w, h = size
    px = w / 2 + (d @ r) / np.maximum(z, 1e-9) / t * (h / 2)
    py = h / 2 - (d @ u) / np.maximum(z, 1e-9) / t * (h / 2)
    return np.stack([px, py], 1), z


def terrain_height(view: SurfaceView, x, y):
    """Height of the rendered ground at (x, y): the detailed patch, the far grid, then the horizon annulus."""
    from .shadows import _bilinear
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.full(np.broadcast(x, y).shape, -np.inf)
    layers = [(view.X, view.Y, view.Z)] + ([view.far] if view.far is not None else [])
    for X, Y, Z in layers[::-1]:
        cx, cy = float(X[0, 1] - X[0, 0]), float(Y[1, 0] - Y[0, 0])
        fx, fy = (x - X[0, 0]) / cx, (y - Y[0, 0]) / cy
        ins = (fx >= 0) & (fx <= Z.shape[1] - 1) & (fy >= 0) & (fy <= Z.shape[0] - 1)
        out = np.where(ins, _bilinear(Z, fx, fy), out)
    if view.horizon > 0:
        c, _, _, _ = _far_frame(view)
        ring = np.isinf(out) & (np.hypot(x - c[0], y - c[1]) <= view.horizon)
        if ring.any():
            out = np.where(ring, ring_height(view, x, y), out)
    return out


def border_points(view: SurfaceView, n: int = 720) -> np.ndarray:
    """Points (n, 3) on the outer border of the rendered ground (the annulus rim, else the far grid's edge)."""
    if view.horizon > 0:
        c, _, _, _ = _far_frame(view)
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x, y = c[0] + view.horizon * np.cos(th), c[1] + view.horizon * np.sin(th)
        return np.stack([x, y, ring_height(view, x, y) - 0.01], 1)
    X, Y, Z = view.far if view.far is not None else (view.X, view.Y, view.Z)
    edge = [(X[0], Y[0], Z[0]), (X[-1], Y[-1], Z[-1]), (X[:, 0], Y[:, 0], Z[:, 0]), (X[:, -1], Y[:, -1], Z[:, -1])]
    return np.concatenate([np.stack(e, 1) for e in edge])


def border_visibility(view: SurfaceView, camera, size=(1600, 900), view_angle: float = 35.0, samples: int = 400,
                      margin: float = 0.0) -> dict:
    """Is the outer border of the ground mesh visible from ``camera``? Every border point is projected through the
    camera; a point inside the frame (grown by ``margin`` pixels) counts as visible unless the ground between the
    camera and the point hides it (``samples`` steps along the sight line, geometric spacing). Returns the number of
    border points tested, inside the frame, and visible, plus the visible points' pixel coordinates."""
    P = border_points(view)
    px, depth = _project(P, camera, size, view_angle)
    w, h = size
    inside = (depth > 0) & (px[:, 0] >= -margin) & (px[:, 0] <= w + margin) & (px[:, 1] >= -margin) & \
        (px[:, 1] <= h + margin)
    cam = np.asarray(camera[0], float)
    vis = np.zeros(len(P), bool)
    for i in np.flatnonzero(inside):
        d = P[i] - cam
        L = np.linalg.norm(d)
        t = np.geomspace(0.05 / L, 1.0, samples)[:-1]
        Q = cam + t[:, None] * d
        ground = terrain_height(view, Q[:, 0], Q[:, 1])
        vis[i] = not np.any(ground > Q[:, 2] + 1e-3)
    return {"n": len(P), "in_frame": int(inside.sum()), "visible": int(vis.sum()), "pixels": px[vis]}


def _poly(V, F):
    import pyvista as pv
    F = np.asarray(F)
    return pv.PolyData(np.asarray(V, float), np.hstack([np.full((len(F), 1), 3), F]).ravel())


def _rgb_from_cmap(vals, cmap, clim):
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    lo, hi = clim
    return cm(np.clip((np.asarray(vals, float) - lo) / max(hi - lo, 1e-12), 0, 1))[..., :3]


def _field_blend(view: SurfaceView, field):
    """Albedo tint (before the photometry) and a glow function (after it) for a field map on a grid."""
    ok = np.isfinite(field)
    if not ok.any():
        return None, None
    col = _rgb_from_cmap(np.where(ok, field, view.clim[0]), view.cmap, view.clim)
    if view.field_saturation != 1.0:
        lum = (col @ np.array([0.299, 0.587, 0.114]))[..., None]
        col = np.clip(lum + view.field_saturation * (col - lum), 0.0, 1.0)
    a = view.field_alpha * ok[..., None]
    tint = (1 - a) + a * col / np.asarray(view.albedo)[None, None, :] * 0.9
    g = view.field_glow * view.field_alpha * ok[..., None]
    if not view.field_glow:
        return tint, None

    def glow(rgb):
        return rgb * (1 - g) + g * col * np.maximum(rgb.mean(-1, keepdims=True) * 1.6, 0.45)
    return tint, glow


def render_surface(view: SurfaceView, path: str | None, camera, *, size=(1600, 900), view_angle: float = 35.0,
                   title: str = "", subtitle: str = "", plotter=None, return_image: bool = False,
                   shadow_cache: dict | None = None):
    """Render ``view`` with ``camera = (position, focal_point[, up])``. Returns the image path or (with
    ``return_image``) the RGB array."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    P = get_preset(view.preset)
    sun = P.sun_direction()
    cam_pos = np.asarray(camera[0], float)
    X, Y, Z = view.X, view.Y, view.Z
    cell = float(X[0, 1] - X[0, 0])
    shadow = terrain_self_shadow(Z, cell, sun) if shadow_cache is None or "self" not in shadow_cache else \
        shadow_cache["self"]
    if shadow_cache is not None:
        shadow_cache["self"] = shadow
    casters = [(p[0], p[1]) for p in view.parts] + [(r[0], r[1]) for r in view.rocks]
    if casters:
        shadow = shadow | cast_shadow(casters, X, Y, Z, sun)
    overlay = view.terrain_overlay and view.terrain_field is not None and view.clim is not None
    tint, glow_rgb = _field_blend(view, view.terrain_field) if overlay else (None, None)
    rgb = shade_terrain(X, Y, Z, sun, cam_pos, view.albedo, shadow, exposure=view.exposure * 2.2,
                        fill=0.05 + P.ambient * 0.3, tint=tint, L=view.lunar_lambert_L)
    if glow_rgb is not None:
        rgb = glow_rgb(rgb)
    own = plotter is None
    p = plotter or pv.Plotter(off_screen=True, window_size=list(size), lighting="none")
    grid = pv.StructuredGrid(X, Y, Z)
    grid["rgb"] = (np.clip(rgb, 0, 1).transpose(1, 0, 2).reshape(-1, 3) * 255).astype(np.uint8)
    p.add_mesh(grid, scalars="rgb", rgb=True, lighting=False, show_scalar_bar=False)
    if view.far is not None:
        Xf, Yf, Zf = view.far
        cf = float(Xf[0, 1] - Xf[0, 0])
        shadow_f = terrain_self_shadow(Zf, cf, sun)
        if casters:                               # the vehicle's shadow continues past the detailed patch
            shadow_f = np.maximum(shadow_f, cast_shadow(casters, Xf, Yf, Zf, sun, supersample=4))
        far_on = view.terrain_overlay and view.far_field is not None and view.clim is not None
        tint_f, glow_f = _field_blend(view, view.far_field) if far_on else (None, None)
        rgb_f = shade_terrain(Xf, Yf, Zf, sun, cam_pos, view.albedo, shadow_f,
                              exposure=view.exposure * 2.2, fill=0.05 + P.ambient * 0.3,
                              L=view.lunar_lambert_L, tint=tint_f)
        if glow_f is not None:
            rgb_f = glow_f(rgb_f)
        g = pv.StructuredGrid(Xf, Yf, Zf - 0.004)
        g["rgb"] = (np.clip(rgb_f, 0, 1).transpose(1, 0, 2).reshape(-1, 3) * 255).astype(np.uint8)
        cc = g.cell_centers().points
        outside = ~((cc[:, 0] > X.min() + cf) & (cc[:, 0] < X.max() - cf) & (cc[:, 1] > Y.min() + cf)
                    & (cc[:, 1] < Y.max() - cf))
        p.add_mesh(g.extract_cells(np.flatnonzero(outside)), scalars="rgb", rgb=True, lighting=False,
                   show_scalar_bar=False)
    if view.horizon > 0:
        Xr, Yr, Zr = _ring_mesh(view)
        Pr = np.stack([Xr, Yr, Zr], -1)
        nr = _ring_normals(view, Xr, Yr)
        v = cam_pos - Pr
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        mu0 = np.clip(nr @ sun, 0.0, None)
        mu = np.clip(np.sum(nr * v, -1), 1e-3, None)
        L = view.lunar_lambert_L
        refl = (1 - L) * mu0 + 2 * L * mu0 / (mu0 + mu + 1e-9)
        a = np.asarray(view.albedo, float)
        rgb_r = 1.0 - np.exp(-1.6 * a * (view.exposure * 2.2 * refl[..., None] + (0.05 + P.ambient * 0.3)
                                         * (0.5 + 0.5 * nr[..., 2:3])))
        gr = pv.StructuredGrid(Xr, Yr, Zr)
        gr["rgb"] = (np.clip(rgb_r, 0, 1).transpose(1, 0, 2).reshape(-1, 3) * 255).astype(np.uint8)
        p.add_mesh(gr, scalars="rgb", rgb=True, lighting=False, show_scalar_bar=False)
    meshes = [(r[0], r[1], "rock", None) for r in view.rocks] + \
        [(it[0], it[1], it[2], it[3] if len(it) > 3 else None) for it in view.parts]
    if meshes:                                    # vertices the terrain hides from the Sun are darkened
        allV = np.concatenate([np.asarray(m_[0], float) for m_ in meshes])
        lit = ~points_in_terrain_shadow(allV + 0.01 * sun, X, Y, Z, sun)
        offs = np.cumsum([0] + [len(m_[0]) for m_ in meshes])
    for j, (V, F, mat, vals) in enumerate(meshes):
        mesh = _poly(V, F)
        m = MATERIALS.get(mat, MATERIALS["grey"])
        light = np.where(lit[offs[j]:offs[j + 1]], 1.0, 0.07)
        base = np.asarray(m["color"], float)[None, :] * light[:, None]
        if vals is not None and view.clim is not None:
            v = np.asarray(vals, float)
            ok = np.isfinite(v) & (v > 0)
            col = _rgb_from_cmap(np.where(ok, v, view.clim[0]), view.cmap, view.clim)
            base = np.where(ok[:, None], col * np.maximum(light[:, None], 0.6), base)
        mesh["rgb"] = (np.clip(base, 0, 1) * 255).astype(np.uint8)
        kw = dict(scalars="rgb", rgb=True, smooth_shading=mat == "rock", diffuse=0.95, ambient=0.08,
                  specular=0.1 + 0.5 * m.get("metallic", 0.0), specular_power=8 + 50 * (1 - m.get("roughness", .5)),
                  show_scalar_bar=False)
        if m.get("emissive"):
            kw.update(ambient=1.0, diffuse=0.2)
        p.add_mesh(mesh, **kw)
    bar = False
    if view.clim is not None and view.colorbar_title:
        import matplotlib  # noqa: F401
        dummy = pv.PolyData(np.array([[X.mean(), Y.mean(), Z.min() - 50.0]] * 2))
        dummy["v"] = np.array(view.clim, float)
        p.add_mesh(dummy, scalars="v", cmap=view.cmap, clim=view.clim, opacity=0.0, show_scalar_bar=True,
                   scalar_bar_args=dict(title=view.colorbar_title, color="#E8E8E8", vertical=True, position_x=0.90,
                                        position_y=0.18, height=0.64, width=0.035, title_font_size=16,
                                        label_font_size=14, fmt="%.0f", n_labels=5))
        bar = True
    ext = float(max(np.ptp(X), np.ptp(Y)))
    apply_pyvista(p, P.with_(shadows=False, sun_intensity=view.mesh_light), target=np.asarray(camera[1], float), extent=ext, earth=view.earth)
    p.camera.position = tuple(cam_pos)
    p.camera.focal_point = tuple(np.asarray(camera[1], float))
    p.camera.up = tuple(camera[2]) if len(camera) > 2 else (0.0, 0.0, 1.0)
    p.camera.view_angle = view_angle
    far_clip = 20 * ext
    if view.far is not None:
        far_clip = max(far_clip, 3.0 * float(max(np.ptp(view.far[0]), np.ptp(view.far[1]))))
    if view.horizon > 0:
        far_clip = max(far_clip, 1.5 * view.horizon)
    p.camera.clipping_range = (0.05, far_clip)
    if title:
        p.add_text(title, position="upper_left", font_size=13, color="#E8E8E8")
    if subtitle:
        p.add_text(subtitle, position="lower_left", font_size=10, color="#BBBBBB")
    del bar
    img = p.screenshot(path, return_img=True) if path else p.screenshot(return_img=True)
    if own:
        p.close()
    return img if return_image else path


__all__ = ["SurfaceView", "render_surface", "border_visibility", "border_points", "terrain_height", "ring_height",
           "horizon_distance", "MOON_RADIUS"]
