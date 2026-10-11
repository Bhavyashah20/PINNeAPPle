"""Off-screen renders of a studio :class:`~.scene.Scene` with PyVista (VTK): the Blender-free counterpart of
:func:`pinneapple_tools.visualization.studio.render`, for machines without ``bpy``.

    from pinneapple_tools.visualization.studio.pyvista_render import render_scene
    render_scene(scene, "beauty.png")                                   # materials (paint, carbon, tyre...)
    render_scene(scene, "cp.png", field="cp", cmap="jet", clim=(-2, 1))  # a surface field with a colour bar
    render_scene(scene, "lines.png", lines=True)                         # streamlines coloured by speed
    render_scene(scene, "wake.png", slice_name="wake: vorticity", view="back")

``view``: "front", "back", "side", "top", "bottom", "iso" or a direction (x, y, z) from the target to the camera.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .scene import MATERIALS, Scene

__all__ = ["render_scene", "VIEWS"]

VIEWS = {"front": (-1.0, -0.35, 0.25), "back": (1.0, 0.0, 0.15), "side": (0.0, -1.0, 0.12), "top": (0.0, 0.0, 1.0),
         "bottom": (0.0, 0.0, -1.0), "iso": (-0.9, -1.1, 0.55), "rear_iso": (1.0, -1.0, 0.6)}


def _rgb(mat: str):
    m = MATERIALS.get(mat, MATERIALS.get("grey", {"color": [0.6, 0.6, 0.6]}))
    return tuple(float(c) for c in m.get("color", [0.6, 0.6, 0.6])[:3]), float(m.get("metallic", 0.0)), \
        float(m.get("roughness", 0.5))


def _poly(V, F):
    import pyvista as pv

    return pv.PolyData(np.asarray(V, float), np.hstack([np.full((len(F), 1), 3), np.asarray(F)]).ravel())


def render_scene(scene: Scene, path: str, *, field: str | None = None, cmap: str = "jet",
                 clim: tuple[float, float] | None = None, lines: bool = False, slice_name: str | None = None,
                 view: str | Sequence[float] = "iso", size=(1600, 900), background: str = "#11151C",
                 zoom: float = 1.0, ground: bool = False, title: str = "", line_width: float = 2.0,
                 max_lines: int | None = None, hide: Sequence[str] = ()) -> str:
    import pyvista as pv

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=list(size), lighting="three lights")
    p.set_background(background, top="#2B3340")
    label = scene.labels.get(field, field) if field else None
    bar_done = False
    lo_all, hi_all = scene.bounds()
    for s in scene.surfaces:
        if s.name in hide:
            continue
        mesh = _poly(s.vertices, s.faces)
        if field and field in s.fields and not lines and slice_name is None:
            v = np.asarray(s.fields[field], float)
            mesh[field] = np.linalg.norm(v, axis=1) if v.ndim > 1 else v
            p.add_mesh(mesh, scalars=field, cmap=cmap, clim=clim or scene.field_range(field), smooth_shading=True,
                       show_scalar_bar=not bar_done, scalar_bar_args=dict(title=label, color="#E8E8E8", vertical=True,
                                                                         position_x=0.9, position_y=0.15, height=0.7))
            bar_done = True
        else:
            rgb, metal, rough = _rgb(s.material)
            dim = 0.35 if (lines or slice_name) else 1.0
            p.add_mesh(mesh, color=tuple(c * dim + (1 - dim) * 0.12 for c in rgb), smooth_shading=True,
                       specular=0.25 + 0.6 * metal, specular_power=10 + 40 * (1 - rough),
                       opacity=1.0 if not slice_name else 0.85)
    if lines and scene.lines:
        for ln in scene.lines:
            pts = list(ln.points)
            vals = list(ln.values) if ln.values is not None else [None] * len(pts)
            step = max(1, len(pts) // max_lines) if max_lines else 1
            for P, val in list(zip(pts, vals, strict=True))[::step]:
                P = np.asarray(P, float)
                if len(P) < 2:
                    continue
                poly = pv.lines_from_points(P)
                if val is not None:
                    poly["speed"] = np.asarray(val, float)
                    p.add_mesh(poly, scalars="speed", cmap=cmap, clim=clim or (0.0, 1.4), line_width=line_width,
                               show_scalar_bar=not bar_done, render_lines_as_tubes=False,
                               scalar_bar_args=dict(title=ln.label or "speed", color="#E8E8E8", vertical=True,
                                                    position_x=0.9, position_y=0.15, height=0.7))
                    bar_done = True
                else:
                    p.add_mesh(poly, color="#9CC9FF", line_width=line_width)
    if slice_name:
        for sl in scene.slices:
            if sl.name != slice_name:
                continue
            G = np.asarray(sl.grid, float)
            nv, nu = G.shape
            s_, t_ = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
            X = sl.origin[None, None, :] + s_[..., None] * sl.u[None, None, :] + t_[..., None] * sl.v[None, None, :]
            grid = pv.StructuredGrid(X[..., 0], X[..., 1], X[..., 2])
            grid["v"] = G.ravel(order="F") if False else G.T.ravel()
            fin = np.isfinite(G)
            rng = clim or (float(np.nanpercentile(G[fin], 2)), float(np.nanpercentile(G[fin], 98)))
            p.add_mesh(grid.threshold(value=(-1e30, 1e30), scalars="v"), scalars="v", cmap=cmap, clim=rng,
                       opacity=0.95, show_scalar_bar=not bar_done,
                       scalar_bar_args=dict(title=sl.label or sl.name, color="#E8E8E8", vertical=True,
                                            position_x=0.9, position_y=0.15, height=0.7))
            bar_done = True
    if ground:
        L = hi_all - lo_all
        plane = pv.Plane(center=((lo_all[0] + hi_all[0]) / 2, 0, lo_all[2] - 1e-3), direction=(0, 0, 1),
                         i_size=L[0] * 3, j_size=max(L[1], 1.0) * 4)
        p.add_mesh(plane, color="#2A2D31", specular=0.1)
    ctr = (lo_all + hi_all) / 2
    d = np.asarray(VIEWS.get(view, view) if isinstance(view, str) else view, float)
    d /= np.linalg.norm(d)
    dist = 1.9 * float(np.linalg.norm(hi_all - lo_all))
    p.camera.position = tuple(ctr + d * dist)
    p.camera.focal_point = tuple(ctr)
    p.camera.up = (0, 0, 1) if abs(d[2]) < 0.95 else (1, 0, 0)
    p.camera.zoom(zoom)
    if title:
        p.add_text(title, position="upper_left", font_size=12, color="#E8E8E8")
    p.screenshot(path)
    p.close()
    return path
