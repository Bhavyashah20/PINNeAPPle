"""Renders of a vehicle on a planetary surface with PyVista: heightfield terrain with baked photometry and hard Sun
shadows (:mod:`.shadows`), vehicle parts with their studio materials, rocks, optional field colour maps on the terrain
and on the parts (with a colour bar), Earth in the sky and a title.

    from pinneapple_tools.visualization.studio.planetary import SurfaceView, render_surface
    view = SurfaceView(X, Y, Z, parts=[(V, F, "white_paint"), ...], rocks=[(V, F)], preset="lunar_south_pole")
    render_surface(view, "frame.png", camera=((-4, -5, 2), (0, 0, 0.8)), title="t = 12.0 s")

``parts`` entries are ``(vertices, faces, material)`` or ``(vertices, faces, material, values)`` where ``values`` (one
per vertex) are coloured with ``cmap`` / ``clim`` instead of the material. ``terrain_field`` (ny, nx) colours the
terrain where it is finite (e.g. normal stress in the ruts), blended over the regolith by ``field_alpha``.
"""
from __future__ import annotations

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
    cmap: str = "inferno"
    clim: tuple | None = None
    colorbar_title: str = ""
    albedo: tuple = (0.42, 0.41, 0.39)
    exposure: float = 1.0
    earth: bool = False
    far: tuple | None = None            # (X, Y, Z) coarse terrain around the detailed patch, drawn outside it
    mesh_light: float = 1.35            # intensity of the Sun on the meshes (the terrain photometry is baked)
    lunar_lambert_L: float = 0.45       # McEwen's L: 1 = Lommel-Seeliger (flat-looking), 0 = Lambert (more relief)


def _poly(V, F):
    import pyvista as pv
    F = np.asarray(F)
    return pv.PolyData(np.asarray(V, float), np.hstack([np.full((len(F), 1), 3), F]).ravel())


def _rgb_from_cmap(vals, cmap, clim):
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    lo, hi = clim
    return cm(np.clip((np.asarray(vals, float) - lo) / max(hi - lo, 1e-12), 0, 1))[..., :3]


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
    tint = None
    if view.terrain_field is not None and view.clim is not None:
        f = view.terrain_field
        ok = np.isfinite(f)
        if ok.any():
            col = _rgb_from_cmap(np.where(ok, f, view.clim[0]), view.cmap, view.clim)
            a = view.field_alpha * ok[..., None]
            tint = (1 - a) + a * col / np.asarray(view.albedo)[None, None, :] * 0.9
    rgb = shade_terrain(X, Y, Z, sun, cam_pos, view.albedo, shadow, exposure=view.exposure * 2.2,
                        fill=0.05 + P.ambient * 0.3, tint=tint, L=view.lunar_lambert_L)
    if view.terrain_field is not None and view.clim is not None:
        ok = np.isfinite(view.terrain_field)
        col = _rgb_from_cmap(np.where(ok, view.terrain_field, view.clim[0]), view.cmap, view.clim)
        glow = 0.55 * view.field_alpha * ok[..., None]       # keep the colour readable inside shadows
        rgb = rgb * (1 - glow) + glow * col * np.maximum(rgb.mean(-1, keepdims=True) * 1.6, 0.45)
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
        rgb_f = shade_terrain(Xf, Yf, Zf, sun, cam_pos, view.albedo, shadow_f,
                              exposure=view.exposure * 2.2, fill=0.05 + P.ambient * 0.3,
                              L=view.lunar_lambert_L)
        g = pv.StructuredGrid(Xf, Yf, Zf - 0.004)
        g["rgb"] = (np.clip(rgb_f, 0, 1).transpose(1, 0, 2).reshape(-1, 3) * 255).astype(np.uint8)
        cc = g.cell_centers().points
        outside = ~((cc[:, 0] > X.min() + cf) & (cc[:, 0] < X.max() - cf) & (cc[:, 1] > Y.min() + cf)
                    & (cc[:, 1] < Y.max() - cf))
        p.add_mesh(g.extract_cells(np.flatnonzero(outside)), scalars="rgb", rgb=True, lighting=False,
                   show_scalar_bar=False)
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
    p.camera.clipping_range = (0.05, 20 * ext)
    if title:
        p.add_text(title, position="upper_left", font_size=13, color="#E8E8E8")
    if subtitle:
        p.add_text(subtitle, position="lower_left", font_size=10, color="#BBBBBB")
    del bar
    img = p.screenshot(path, return_img=True) if path else p.screenshot(return_img=True)
    if own:
        p.close()
    return img if return_image else path


__all__ = ["SurfaceView", "render_surface"]
