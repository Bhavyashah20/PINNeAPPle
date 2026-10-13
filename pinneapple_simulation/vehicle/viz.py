"""Pictures and animations of a :class:`~.sim.VehicleRun`: the vehicle CAD posed by the multibody state (wheels roll
with their joint angles), the deformable terrain with ruts and grouser marks replayed to each instant, contact
stresses as colour maps on the wheels and the terrain, Sun shadows and the lunar sky.

    from pinneapple_simulation.vehicle import viz
    viz.render_frame(run, terrain, k, "frame.png", field="normal_stress")
    viz.animate(run, terrain, "run.gif", mp4="run.mp4", field="normal_stress", frames=120)
    viz.beauty(spec, "viper.png")

``field``: None (materials), ``"normal_stress"`` / ``"shear_stress"`` [kPa] (wheel contact arc and the peak stress
left in the ruts) or ``"sinkage"`` [mm] (rut depth).
"""
from __future__ import annotations

import math
import shutil

import numpy as np

from ..multibody.tree import quat_to_matrix
from ..terramechanics.terrain import DeformableTerrain

FIELDS = {"normal_stress": ("Normal stress [kPa]", "inferno"), "shear_stress": ("Shear stress [kPa]", "viridis"),
          "sinkage": ("Rut depth [mm]", "cividis")}


def posed_parts(model, q, *, skip=()):
    """World (V, F, material, body, part name) of every CAD part for the generalised position ``q``."""
    kin = model.tree.kinematics(q)
    tr = model.transforms(kin)
    out = []
    for p in model.spec.cad.parts:
        if p.body in skip:
            continue
        R, t = tr[p.body]
        out.append((p.vertices @ R.T + t, p.faces, p.material, p.body, p.name))
    return out, kin


def _wheel_values(V, centre, axle, normal, tangent, theta, values, R_in):
    """Per-vertex values of a wheel: the stress at the vertex's angle on the contact arc (rim and grousers)."""
    d = V - centre
    r = d - np.outer(d @ axle, axle)
    rad = np.linalg.norm(r, axis=1)
    ang = np.arctan2(r @ tangent, -(r @ normal))           # 0 at the bottom, > 0 towards the front
    out = np.zeros(len(V))
    if theta is None or len(theta) == 0:
        return out
    o = np.argsort(theta)
    th, vv = theta[o], values[o]
    m = (rad > R_in) & (ang >= th[0]) & (ang <= th[-1])
    out[m] = np.interp(ang[m], th, vv)
    return out


def terrain_at(run, terrain: DeformableTerrain, t_end: float, cache: dict | None = None) -> DeformableTerrain:
    """A copy of ``terrain`` with the ruts and marks of ``run`` up to ``t_end`` (incremental with ``cache``)."""
    if cache is not None and cache.get("t", -1) <= t_end and "terrain" in cache:
        cache["k"] = run.replay_terrain(cache["terrain"], t_end + run.meta.get("settle_s", 0.0), cache["k"])
        cache["t"] = t_end
        return cache["terrain"]
    ter = terrain.copy_empty()
    k = run.replay_terrain(ter, t_end + run.meta.get("settle_s", 0.0))
    if cache is not None:
        cache.update(terrain=ter, k=k, t=t_end)
    return ter


def frame_view(run, terrain, k: int, *, field: str | None = None, clim=None, half_window=(3.2, 2.4),
               far_margin: float = 14.0, far_cell: float = 0.04, cache=None, earth: bool = False,
               preset="lunar_low_sun", sun_azimuth_deg: float | None = None, far_relief: float = 0.03,
               colorbar: bool = True, trail: float = 0.0, horizon: float = 4000.0, overlay: bool = True,
               field_alpha: float = 0.85, field_glow: float = 0.55, field_saturation: float = 1.0):
    """The :class:`~pinneapple_tools.visualization.studio.planetary.SurfaceView` of sample ``k``.

    The detailed terrain (ruts with their walls, berms and grouser marks as real geometry, optional stress maps)
    covers ``half_window`` around the vehicle, shifted back by ``trail`` metres along the direction of travel so
    more of the ruts behind it stay in view; a coarser far field (:func:`far_surface`, ruts included, with the
    vehicle's shadow) fills ``far_margin`` metres around it and a curved annulus continues the ground to
    ``horizon`` metres, past the lunar horizon, so no edge of the ground mesh is visible. ``field_alpha`` and
    ``field_glow`` (and ``field_saturation``) set how the ruts' colour map sits on the relief (see
    :class:`~pinneapple_tools.visualization.studio.planetary.SurfaceView`); e.g. 0.6 / 0.1 is a semi-transparent
    map through which the rut geometry, grouser marks and Sun shading stay visible. ``field`` colours the wheels and the ruts; ``overlay=False`` keeps the
    wheel colours but draws the ruts by their geometry only."""
    from pinneapple_tools.visualization.studio.planetary import SurfaceView
    ter = terrain_at(run, terrain, float(run.t[k]), cache)
    cx, cy = run.q[k, 0], run.q[k, 1]
    if trail:
        d = run.q[k, :2] - run.q[max(k - 10, 0), :2]
        if np.linalg.norm(d) > 1e-3:
            cx, cy = np.array([cx, cy]) - trail * d / np.linalg.norm(d)
    hx, hy = half_window
    xr = (max(cx - hx, ter.x0), min(cx + hx, ter.x1))
    yr = (max(cy - hy, ter.y0), min(cy + hy, ter.y1))
    V, F, flds = ter.mesh(xr, yr)
    nxx = len(np.unique(np.round(V[:, 0], 6)))
    X, Y, Z = (V[:, i].reshape(-1, nxx) for i in range(3))
    parts, kin = posed_parts(run.model, run.q[k])
    items = []
    tfield = None
    label, cmap = FIELDS.get(field, ("", "inferno"))
    if field in ("normal_stress", "shear_stress"):
        key = "sigma" if field == "normal_stress" else "tau_x"
        st = run.stresses
        names = run.corner_names
        Rw = run.model.spec.wheel.radius
        for V_, F_, mat, body, _name in parts:
            if body.startswith("wheel_"):
                i = names.index(body[len("wheel_"):])
                vals = _wheel_values(V_, run.wheel["centre"][k, i], kin.axis[run.model.wheel[i]],
                                     run.wheel["normal"][k, i], run.wheel["tangent"][k, i],
                                     st["theta"][k, i] if st else None,
                                     1e-3 * np.abs(st[key][k, i]) if st else None, 0.9 * Rw)
                items.append((V_, F_, mat, vals))
            else:
                items.append((V_, F_, mat))
        g = flds["normal_stress" if field == "normal_stress" else "shear_stress"].reshape(X.shape)
        tfield = np.where(g > 0, g, np.nan)
    else:
        items = [(V_, F_, mat) for V_, F_, mat, _, _ in parts]
        if field == "sinkage":
            g = flds["rut_depth"].reshape(X.shape)
            tfield = np.where(g > 0.3, g, np.nan)
    rocks = [r.mesh() for r in ter.rocks if abs(r.x - cx) < far_margin and abs(r.y - cy) < far_margin]
    far = None
    if far_margin > 0:
        xs = np.arange(cx - far_margin, cx + far_margin + far_cell, far_cell)
        ys = np.arange(cy - far_margin, cy + far_margin + far_cell, far_cell)
        Xf, Yf = np.meshgrid(xs, ys)
        far = (Xf, Yf, far_surface(ter, Xf, Yf, far_relief))
    far_field = None
    if far is not None and field in ("normal_stress", "shear_stress", "sinkage"):
        far_field = _far_field(ter, far[0], far[1], field)
    if sun_azimuth_deg is not None:
        from pinneapple_tools.visualization.studio.lighting import get_preset
        preset = get_preset(preset).with_(sun_azimuth_deg=sun_azimuth_deg)
    if clim is None and tfield is not None and np.isfinite(tfield).any():
        clim = (0.0, float(np.nanpercentile(tfield, 99)))
    return SurfaceView(X, Y, Z, parts=items, rocks=rocks, preset=preset, terrain_field=tfield, cmap=cmap,
                       clim=clim, colorbar_title=label if colorbar and field else "", far=far, earth=earth,
                       horizon=horizon if far is not None else 0.0, far_fn=ter.base_soil_height,
                       terrain_overlay=overlay, field_alpha=field_alpha, field_glow=field_glow,
                       field_saturation=field_saturation, far_field=far_field)


def _far_field(ter: DeformableTerrain, X, Y, field: str):
    """The terrain field of :func:`frame_view` (kPa or mm, NaN where nothing was recorded) on far-grid points."""
    inside = (X >= ter.x0) & (X <= ter.x1) & (Y >= ter.y0) & (Y <= ter.y1)
    if field == "sinkage":
        g = 1e3 * ter.rut_depth(X, Y)
        return np.where(inside & (g > 0.3), g, np.nan)
    g = 1e-3 * ter._bilinear(ter.sigma if field == "normal_stress" else ter.tau, X, Y)
    return np.where(inside & (g > 0), g, np.nan)


def far_surface(ter: DeformableTerrain, X, Y, relief: float = 0.03, feather: float = 1.0):
    """Ground height for the far field around a deformable terrain: its visible surface (ruts, berms, relief)
    inside its grid, faded over ``feather`` metres into the base surface plus a world-anchored relief of RMS
    ``relief`` outside it, so neither the grid nor its ruts end at a visible step."""
    from pinneapple_tools.visualization.studio.shadows import relief_noise
    base = ter.base_soil_height(X, Y)
    out = base + relief_noise(X, Y, relief) * _taper(X, Y, (ter.x0, ter.x1), (ter.y0, ter.y1), feather)
    inside = (X >= ter.x0) & (X <= ter.x1) & (Y >= ter.y0) & (Y <= ter.y1)
    if inside.any():
        d = np.minimum(np.minimum(X - ter.x0, ter.x1 - X), np.minimum(Y - ter.y0, ter.y1 - Y))
        w = np.clip(d / feather, 0.0, 1.0)
        w = w * w * (3 - 2 * w)
        out = np.where(inside, w * ter.surface(X, Y) + (1 - w) * out, out)
    return out


def _taper(X, Y, xr, yr, width: float = 3.0):
    """0 inside the detailed patch, rising to 1 at ``width`` metres outside it (no seam at its edge)."""
    dx = np.maximum(np.maximum(xr[0] - X, X - xr[1]), 0.0)
    dy = np.maximum(np.maximum(yr[0] - Y, Y - yr[1]), 0.0)
    d = np.hypot(dx, dy)
    s = np.clip(d / width, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def follow_camera(run, k, offset=(-4.4, -5.0, 2.1), look=(0.5, 0.0, 0.75), smooth: int = 15):
    """Camera trailing the vehicle: ``offset`` in a frame turned with the (smoothed) heading."""
    lo, hi = max(0, k - smooth), min(len(run.t), k + smooth + 1)
    c = run.q[lo:hi, :3].mean(0)
    yaw = float(np.mean(run.chassis["yaw"][lo:hi]))
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return tuple(c + Rz @ np.asarray(offset)), tuple(c + Rz @ np.asarray(look))


def render_frame(run, terrain, k, path, *, field=None, clim=None, camera=None, size=(1600, 900), title=None,
                 cache=None, earth=False, **kw):
    from pinneapple_tools.visualization.studio.planetary import render_surface
    view = frame_view(run, terrain, k, field=field, clim=clim, cache=cache, earth=earth, **kw)
    cam = camera(run, k) if callable(camera) else (camera or follow_camera(run, k))
    ttl = title if title is not None else (f"t = {run.t[k]:5.1f} s   x = {run.q[k, 0]:5.2f} m   "
                                           f"v = {run.chassis['speed'][k]:4.2f} m/s")
    return render_surface(view, path, cam, size=size, title=ttl, return_image=path is None)


def animate(run, terrain, gif: str | None, *, mp4: str | None = None, field=None, clim=None, frames: int = 120,
            fps: int = 15, size=(1280, 720), camera=None, earth=False, title_fn=None, progress=None,
            frame_hook=None, gif_size=(640, 360), gif_every: int = 2) -> dict:
    """GIF (and MP4 with ffmpeg) of the run: ``frames`` samples evenly spread over it. The MP4 keeps ``size`` and
    every frame; the GIF (much larger per frame) is downscaled to ``gif_size`` and keeps every ``gif_every``-th
    frame at the same playback speed. ``frame_hook(img, k)`` may return a modified image."""
    import imageio.v2 as imageio
    ks = np.unique(np.linspace(0, len(run.t) - 1, frames).astype(int))
    cache: dict = {}
    imgs = []
    for j, k in enumerate(ks):
        img = render_frame(run, terrain, k, None, field=field, clim=clim, camera=camera, size=size,
                           title=title_fn(run, k) if title_fn else None, cache=cache, earth=earth)
        if frame_hook is not None:
            img = frame_hook(img, k)
        imgs.append(np.asarray(img)[..., :3])
        if progress:
            progress(j + 1, len(ks))
    out = {}
    if gif:
        write_gif(gif, imgs[::gif_every], fps / gif_every, gif_size)
        out["gif"] = gif
    if mp4 and shutil.which("ffmpeg"):
        w = imageio.get_writer(mp4, fps=fps, codec="libx264", quality=8, macro_block_size=8)
        for im in imgs:
            w.append_data(im)
        w.close()
        out["mp4"] = mp4
    out["frames"] = len(imgs)
    return out


def write_gif(path, imgs, fps: float, size=None) -> str:
    """Animated GIF of RGB frames, optionally downscaled to ``size`` (w, h)."""
    import imageio.v2 as imageio
    from PIL import Image
    if size is not None:
        imgs = [np.asarray(Image.fromarray(np.asarray(im)[..., :3]).resize(size, Image.LANCZOS)) for im in imgs]
    imageio.mimsave(path, imgs, duration=1.0 / fps, loop=0)
    return path


def beauty_view(spec, terrain=None, preset="lunar_south_pole", earth=False, horizon: float = 4000.0,
                sun_azimuth_deg: float | None = None):
    """The :class:`~pinneapple_tools.visualization.studio.planetary.SurfaceView` of :func:`beauty`."""
    if sun_azimuth_deg is not None:
        from pinneapple_tools.visualization.studio.lighting import get_preset
        preset = get_preset(preset).with_(sun_azimuth_deg=sun_azimuth_deg)
    from pinneapple_simulation.rover.terrain import LunarTerrain
    from pinneapple_tools.visualization.studio.planetary import SurfaceView
    ter = terrain or DeformableTerrain(LunarTerrain(), (-5, 5), (-5, 5), cell=0.025, relief=0.01, relief_seed=4)
    X, Y = ter.grid_xy()
    Z = ter.surface(X, Y)
    Xf, Yf = np.meshgrid(np.arange(-36, 36.01, 0.1), np.arange(-36, 36.01, 0.1))
    return SurfaceView(X, Y, Z, parts=[(p.vertices, p.faces, p.material) for p in spec.cad.parts],
                       rocks=[r.mesh() for r in ter.rocks], preset=preset, earth=earth,
                       far=(Xf, Yf, far_surface(ter, Xf, Yf, 0.03)), horizon=horizon, far_fn=ter.base_soil_height)


BEAUTY_CAMERA = ((-3.2, -4.6, 1.7), (0.15, 0.0, 0.95))


def beauty(spec, path, *, terrain=None, camera=BEAUTY_CAMERA, size=(1600, 900), title="", preset="lunar_south_pole",
           earth=False, sun_azimuth_deg: float | None = None) -> str:
    """The vehicle at its reference pose on a patch of terrain (default: level regolith with a fine relief), the
    ground continued to beyond the horizon. ``sun_azimuth_deg`` turns the preset's Sun (e.g. to light the face
    the camera sees)."""
    from pinneapple_tools.visualization.studio.planetary import render_surface
    view = beauty_view(spec, terrain, preset, earth, sun_azimuth_deg=sun_azimuth_deg)
    return render_surface(view, path, camera, size=size, title=title)


# ---------------------------------------------------------------------- charts
POS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")


def _style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_run(run, path, title=""):
    """Per wheel versus distance: sinkage, slip, normal load, drawbar pull, drive torque, lateral force; plus the
    chassis pitch and roll."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = run.distance()
    series = [("sinkage", "Sinkage [mm]", 1e3), ("slip", "Slip [-]", 1.0), ("load", "Normal load [N]", 1.0),
              ("drawbar_pull", "Drawbar pull [N]", 1.0), ("drive_torque", "Drive torque [N m]", 1.0),
              ("lateral", "Lateral force [N]", 1.0)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True)
    for ax, (key, lab, sc) in zip(axes.ravel()[:6], series, strict=True):
        for i, c in enumerate(run.corner_names):
            ax.plot(d, sc * run.wheel[key][:, i], color=POS[i % 4], lw=1.3, label=c)
        ax.set_title(lab, fontsize=10, loc="left")
        _style(ax)
    ax = axes.ravel()[6]
    ax.plot(d, np.degrees(run.chassis["pitch"]), color="#2a78d6", lw=1.5, label="pitch")
    ax.plot(d, np.degrees(run.chassis["roll"]), color="#eb6834", lw=1.5, label="roll")
    ax.set_title("Chassis attitude [deg]", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    ax = axes.ravel()[7]
    for i, c in enumerate(run.corner_names):
        ax.plot(d, np.degrees(run.joints["susp"][:, i]), color=POS[i % 4], lw=1.3, label=c)
    ax.set_title("Suspension arm angle [deg]", fontsize=10, loc="left")
    _style(ax)
    axes[0, 0].legend(fontsize=8, frameon=False, ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("distance [m]", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_testbed(curves: dict, path: str, title: str = ""):
    """Single-wheel curves versus slip: ``curves[name] = WongReeceModel.steady_state(...)`` dicts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, (name, r) in enumerate(curves.items()):
        c = POS[j % 4]
        ax[0].plot(r["slip"], r["drawbar_pull"], color=c, lw=1.8, label=name)
        ax[1].plot(r["slip"], r["torque"], color=c, lw=1.8)
        ax[2].plot(r["slip"], 1e3 * r["sinkage"], color=c, lw=1.8)
    for a, t in zip(ax, ("Drawbar pull [N]", "Drive torque [N m]", "Sinkage [mm]"), strict=True):
        a.set_title(t, fontsize=10, loc="left")
        a.set_xlabel("slip [-]", fontsize=9)
        _style(a)
    ax[0].legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def chassis_rotation(run, k):
    return quat_to_matrix(run.q[k, 3:7])


__all__ = ["posed_parts", "terrain_at", "frame_view", "follow_camera", "render_frame", "animate", "write_gif", "beauty",
           "beauty_view", "BEAUTY_CAMERA", "far_surface",
           "plot_run", "plot_testbed", "FIELDS"]
